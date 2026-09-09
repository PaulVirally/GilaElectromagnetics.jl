# Build-time comparison for a self volume, before and after the far field is taken over by
# farfield.jl.  Nothing under src/ is modified: the "after" build replicates GlaVacOprMem's own
# pipeline (genEgoCrcSlf! + gthEgoCmp! + genEgoFur, called through GilaElectromagnetics.GilaVacuum)
# with the separated offsets filled by farBlock! instead of egoFunInn!, and ends in the existing
# constructor GlaVacOprMem(cmpInf, egoFur, vol, vol), so the result is a usable operator.
#   JULIA_NUM_THREADS=<n> julia --startup-file=no --project=<env> bench.jl 32/32 64/32 128/32 32/8 32/4
# One argument is N/den = an N^3 self volume of (1/den)^3 cells.  Writes tables/bench_<n>.txt.
# Stages timed on both sides: contact integrals (wekTrp), far fill, embedding (egoToeCrc!),
# gather, FFT (genEgoFur); the plan creation of the final constructor is common to both and is
# timed separately.  The raw far fill is also attributed to the quadOrd bands, and the two
# egoFur are compared entry by entry; at N <= 32 both operators are applied to the same random
# vector.  A 4^3 build of each side warms up the compiler before anything is timed.
using GilaElectromagnetics, StaticArrays, BenchmarkTools, Printf, LinearAlgebra, Random
const GV = GilaElectromagnetics.GilaVacuum
include(joinpath(@__DIR__, "farfield.jl"))
function tabDir()
    for c in (get(ENV, "FARFIELD_TABDIR", ""), joinpath(@__DIR__, "shapetab"))
        isempty(c) && continue
        isdir(c) && any(endswith(f, "_p$(TABPRC)_seg.txt") for f in readdir(c)) && return c
    end
    mkpath(joinpath(@__DIR__, "shapetab"))
end
TABDIR[] = tabDir()
KSRDIR[] = joinpath(@__DIR__, "ksrcache")

const NT = Threads.nthreads()
const FRQ = ComplexF64(1)
const OUT = joinpath(@__DIR__, "tables", "bench_$(NT).txt")
mkpath(joinpath(@__DIR__, "tables"))
tee(s) = (println(s); flush(stdout); open(io -> println(io, s), OUT, "a"))

optOf(f) = GV.CPUKerOpt{Float64}(f, 48, false, GV.CPU())
volOf(N, den) = GlaVol((N, N, N), (1//den, 1//den, 1//den), (0//1, 0//1, 0//1))
shpOf(den) = (QI(1)//den, QI(1)//den, QI(1)//den)

# the eight offsets with every index <= 2 (contact and touching shell) plus the identity term:
# Gila's own egoFunSng!, used unchanged by both builds
function nearFil!(egoToe, v, o, f)
    facLst = GV.facPar(); fac = Float64.(GV.cubFac(v.scl)); srcGrd = GV.sepGrd(v, v, 0)
    wS, wE, wV = GV.wekTrp(v.scl, o)
    for pos in CartesianIndices(ntuple(d -> 1:min(v.cel[d], 2), 3))
        GV.egoFunSng!(view(egoToe, :, :, pos), pos, wS, wE, wV, srcGrd, v, fac, fac, facLst, o)
    end
    for a in 1:3; egoToe[a, a, 1, 1, 1] -= 1 / f^2; end
    return egoToe
end

# quadrature fill of every separated offset, exactly as genEgoCrcSlf! does it
function farFilOld!(egoToe, v, o)
    facLst = GV.facPar(); fac = Float64.(GV.cubFac(v.scl)); srcGrd = GV.sepGrd(v, v, 0)
    ids = collect(CartesianIndices(size(egoToe)[3:5]))
    Threads.@threads for crt in ids
        @inbounds GV.egoFunInn!(egoToe, crt, srcGrd, v.scl, fac, fac, facLst, o)
    end
    return egoToe
end

# embedding, gather and Fourier transform: GlaVacOprMem's own stages, shared by both builds
function furStage(egoToe, v, o)
    mixInf = GV.genEveExtInf(v, v)
    totCelCrc = mixInf.trgCel .+ mixInf.srcCel
    egoCrc = Array{ComplexF64}(undef, 3, 3, totCelCrc..., 1, 1)
    crc = selectdim(selectdim(egoCrc, 7, 1), 6, 1)
    ids = collect(CartesianIndices(axes(crc)[3:5]))
    tEmb = @elapsed begin
        Threads.@threads for crt in ids
            @inbounds GV.egoToeCrc!(view(crc, :, :, crt), egoToe, crt, div.(size(crc)[3:5], 2))
        end
    end
    cmp = Array{ComplexF64}(undef, totCelCrc..., 6, 1, 1)
    tGth = @elapsed GV.gthEgoCmp!(cmp, egoCrc)
    egoCrc = nothing
    truInf = [max(cld(mixInf.trgCel[d], 2) + iseven(mixInf.trgCel[d]), 2) for d in 1:3]
    tFft = @elapsed egoFur = GV.genEgoFur(cmp, truInf, o)
    cmp = nothing
    return (egoFur, tEmb, tGth, tFft)
end

# wekTrp depends only on (cell scale, order, frequency) and Gila memoizes it, so it is timed once
# per shape and that time is carried into both builds
const WEKT = Dict{Int,Float64}()
function wekTim(v, o, den)
    haskey(WEKT, den) && (GV.wekTrp(v.scl, o); return WEKT[den])
    empty!(GV.wekMem)
    WEKT[den] = @elapsed GV.wekTrp(v.scl, o)
    return WEKT[den]
end

"Gila's build, staged: contact integrals, quadrature far fill, embedding, gather, FFT."
function bldOld(N, den, f)
    v = volOf(N, den); o = optOf(f)
    tWek = wekTim(v, o, den)
    egoToe = Array{ComplexF64}(undef, 3, 3, N, N, N)
    tFar = @elapsed farFilOld!(egoToe, v, o)
    tSng = @elapsed nearFil!(egoToe, v, o, f)
    egoFur, tEmb, tGth, tFft = furStage(egoToe, v, o)
    return (egoFur, [tWek, tFar, tSng, tEmb, tGth, tFft], egoToe)
end

"The same build with every offset of max-norm separation >= 2 filled by farBlock!."
function bldNew(N, den, f)
    v = volOf(N, den); o = optOf(f); sQ = shpOf(den)
    tWek = wekTim(v, o, den)
    tTab = @elapsed fs = farSetup(sQ, f; nBlk = N)
    egoToe = Array{ComplexF64}(undef, 3, 3, N, N, N)
    tFar = @elapsed farBlock!(egoToe, sQ, f; fs = fs)
    tSng = @elapsed nearFil!(egoToe, v, o, f)
    egoFur, tEmb, tGth, tFft = furStage(egoToe, v, o)
    return (egoFur, [tWek, tTab, tFar, tSng, tEmb, tGth, tFft], egoToe)
end

furDif(a, b) = maximum(maximum(abs, a[i] .- b[i]) for i in 1:8) /
               maximum(maximum(abs, b[i]) for i in 1:8)

# per-offset cost of the quadrature fill in each quadOrd band, and the band populations
function bandTbl(N, den, f)
    v = volOf(N, den); o = optOf(f)
    fac = Float64.(GV.cubFac(v.scl)); fp = GV.facPar(); grd = GV.sepGrd(v, v, 0)
    kScl = 2pi * abs(f) * Float64(1//den)
    cnt = Dict{Int,Int}(); rep = Dict{Int,CartesianIndex{3}}()
    for I in CartesianIndices((N, N, N))
        all(Tuple(I) .<= 2) && continue
        ord = GV.quadOrd(maximum(Tuple(I) .- 1), kScl)
        cnt[ord] = get(cnt, ord, 0) + 1
        haskey(rep, ord) || (rep[ord] = I)
    end
    egoToe = Array{ComplexF64}(undef, 3, 3, N, N, N)
    tee(@sprintf("  %-6s %-10s %-12s %-12s %s", "order", "offsets", "us/offset", "band s", "kernel evals"))
    tot = 0.0; evl = 0
    for ord in sort(collect(keys(cnt)))
        I = rep[ord]
        t = @belapsed GV.egoFunInn!($egoToe, $I, $grd, $(v.scl), $fac, $fac, $fp, $o) samples = 5 evals = 1
        tot += t * cnt[ord]; evl += cnt[ord] * 36 * ord^4
        tee(@sprintf("  %-6d %-10d %-12.1f %-12.3f %.3e", ord, cnt[ord], 1e6 * t, t * cnt[ord],
                     cnt[ord] * 36 * ord^4))
    end
    tee(@sprintf("  serial far fill %.2f s over %.3e kernel evaluations, %.1f ns/eval", tot, evl,
                 1e9 * tot / evl))
    return tot
end

function run(N, den, f)
    tee(@sprintf("== N = %d, s = 1/%d, f = %s, threads = %d", N, den, string(f), NT))
    furA, tA, toeA = bldOld(N, den, f)
    GC.gc()
    furB, tB, toeB = bldNew(N, den, f)
    GC.gc()
    tee(@sprintf("  before: contact %.2f  far(egoFunInn!) %.2f  touching %.3f  embed %.2f  gather %.2f  FFT %.2f  total %.2f s",
                 tA[1], tA[2], tA[3], tA[4], tA[5], tA[6], sum(tA)))
    tee(@sprintf("  after : contact %.2f  table %.2f  far(farBlock!) %.3f  touching %.3f  embed %.2f  gather %.2f  FFT %.2f  total %.2f s",
                 tB[1], tB[2], tB[3], tB[4], tB[5], tB[6], tB[7], sum(tB)))
    tee(@sprintf("  far fill %.1fx faster (%.2f s -> %.3f s), whole build %.1fx (%.2f s -> %.2f s)",
                 tA[2] / tB[3], tA[2], tB[3], sum(tA) / sum(tB), sum(tA), sum(tB)))
    dTo = 0.0; dOf = (0, 0, 0)
    for i3 in 1:N, i2 in 1:N, i1 in 1:N
        max(i1, i2, i3) >= 3 || continue
        d = maximum(abs, view(toeA, :, :, i1, i2, i3) .- view(toeB, :, :, i1, i2, i3)) /
            maximum(abs, view(toeA, :, :, i1, i2, i3))
        d > dTo && (dTo = d; dOf = (i1 - 1, i2 - 1, i3 - 1))
    end
    tee(@sprintf("  egoFur: max |after - before| / max |before| = %.3e", furDif(furB, furA)))
    tee(@sprintf("  egoToe: worst separated offset %.3e at %s (Gila's quadrature error, not this library's)",
                 dTo, string(dOf)))
    if N <= 32
        v = volOf(N, den); o = optOf(f)
        tRef = @elapsed mem = GlaVacOprMem(o, v)
        tee(@sprintf("  GlaVacOprMem(cmpInf, vol) itself: %.2f s (contact memoized); its egoFur vs the staged before build %.3e",
                     tRef, furDif(mem.egoFur, furA)))
        tAsm = @elapsed mA = GlaVacOprMem(o, furA, v, v)
        mB = GlaVacOprMem(o, furB, v, v)
        tee(@sprintf("  GlaVacOprMem(cmpInf, egoFur, vol, vol): %.2f s of FFT plans and phases, common to both",
                     tAsm))
        Random.seed!(20260907)
        x = rand(ComplexF64, N, N, N, 3)
        yA = GV.egoOpr!(mA, copy(x)); yB = GV.egoOpr!(mB, copy(x))
        tee(@sprintf("  operator on a random vector: max |dy| / max |y| = %.3e  (|y| = %.3e)",
                     maximum(abs, yA .- yB) / maximum(abs, yA), maximum(abs, yA)))
    end
    if NT == 1
        tee("  attribution of the quadrature far fill by quadOrd band:")
        bandTbl(N, den, f)
    end
    tee("")
    return nothing
end

tee(@sprintf("# %s, %d threads, load %s", Sys.CPU_NAME, NT, strip(read(pipeline(`uptime`), String))))
tee("# warm-up on 4^3")
run(4, 32, FRQ)
for arg in ARGS
    N, den = parse.(Int, split(arg, "/"))
    run(N, den, FRQ)
end
