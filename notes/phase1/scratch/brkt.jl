# Phase 1 bracketing: cntBlk! against the DIRECTFN + order-9 path it replaces,
# at intOrd 48 and 64.  Follows notes/moments/gilacmp.jl parts (a) and (b): the
# tensor metric is merge.md's max_ab |old - ser| / max_ab |old| per offset, the
# face-pair metric is gilacmp (b)'s order-9 rule on the non-touching shell pairs.
# Run BEFORE the integrators are deleted.  Writes notes/phase1/scratch/brkt.txt.
using GilaElectromagnetics, Printf, DoubleFloats
const GV = GilaElectromagnetics.GilaVacuum
const VOL = GilaElectromagnetics.GilaVolumes
module Mom; include("/Users/pvirally/.julia/dev/GilaElectromagnetics/notes/moments/moments.jl"); end

const SCLS = [("a", (1//32, 1//32, 1//32)), ("b", (1//8, 1//8, 1//8)),
              ("c", (1//32, 1//32, 1//512)), ("d", (1//4, 1//4, 1//4))]
const ORDS = Dict("a" => [48, 64], "b" => [48, 64], "c" => [48, 64], "d" => [48, 64])
const FRQS = [("1", 1.0 + 0.0im), ("1+0.1i", 1.0 + 0.1im)]
const OFFS = [Tuple(c) .- 1 for c in CartesianIndices((2, 2, 2))]
const SHELL = ["P1", "P2", "P1s", "P1s-b", "P2s", "P2s-b", "Xg", "Xg-b", "Xg-c", "Xg-d"]

struct Duo <: IO; a::IO; b::IO; end
Base.write(d::Duo, x::UInt8) = (write(d.a, x); write(d.b, x))
Base.flush(d::Duo) = (flush(d.a); flush(d.b))

opt(frq) = GV.CPUKerOpt(frq, Float64, false, GV.CPU())

# the DIRECTFN triple at a chosen order, then egoFunSng! exactly as
# genEgoCrcSlf! called it
function oldToe(vol, frq, ord)
    o = opt(frq)
    qud = GV.gauQud(ord)
    celInv = ^(prod(Float64.(vol.scl)), -1)
    wS = celInv .* GV.wekS(vol.scl, qud, o)
    wE = celInv .* GV.wekE(vol.scl, qud, o)
    wV = celInv .* GV.wekV(vol.scl, qud, o)
    fac = Float64.(GV.cubFac(vol.scl))
    grd = VOL.sepGrd(vol, vol, 0)
    toe = zeros(ComplexF64, 3, 3, 2, 2, 2)
    for pos in CartesianIndices((2, 2, 2))
        GV.egoFunSng!(view(toe, :, :, pos), pos, wS, wE, wV, grd, vol, fac, fac,
                      GV.facPar, o)
    end
    return toe
end

function newToe(vol, frq)
    toe = zeros(ComplexF64, 3, 3, 2, 2, 2)
    GV.cntBlk!(toe, vol, opt(frq))
    return toe
end

# merge.md's metric, plus the worst per-entry relative deviation over the
# entries that are not zero by symmetry (below 1e-3 of the block an entry is
# accurate only in absolute terms, PLAN section 3.11)
function tnsErr(old, ser)
    nrm = maximum(abs, ser)
    ent = 0.0
    for k in eachindex(ser)
        abs(ser[k]) >= 1e-3 * nrm &&
            (ent = max(ent, abs(old[k] - ser[k]) / abs(ser[k])))
    end
    return maximum(abs, old .- ser) / maximum(abs, old), ent
end

function ord9(D, F, Fp, scl, frq)
    s = Float64.(scl)
    fac = Float64.(GV.cubFac(scl))
    srf = zeros(ComplexF64, 36)
    fp = (F - 1) * 6 + Fp
    GV.egoSrfFxd!(D[1] * s[1], D[2] * s[2], D[3] * s[3], srf, fac, fac, [fp],
                  GV.facPar, Float64.(GV.srfScl(s, s)), opt(frq), GV.cntOrd)
    return srf[fp] * prod(s), fp
end

open(joinpath(@__DIR__, "brkt.txt"), "w") do fil
io = Duo(stdout, fil)
println(io, "Phase 1 bracketing, ", GV.cntOrd, " = cntOrd, threads = ", Threads.nthreads())
println(io)
println(io, "## assembled contact tensors: cntBlk! vs wekTrp + egoFunSng!")
println(io, "   rel = max_ab |old - ser| / max_ab |old| (merge.md); ent = worst per-entry")
println(io, "   |old - ser| / |ser|; cnv = rel(64) < rel(48)")
println(io)
println(io, rpad("scl", 22), rpad("f", 8), rpad("offset", 11), rpad("mMax", 6),
        rpad("rel48", 12), rpad("rel64", 12), rpad("ent48", 12), rpad("ent64", 12), "verdict")
sm = Dict("cnv" => 0, "plt" => 0, "row" => 0)
tms = Dict{String,Float64}()
for (ky, scl) ∈ SCLS
    vol = GlaVol((4, 4, 4), scl, (0//1, 0//1, 0//1))
    for (fk, f) ∈ FRQS
        mM = GV.serOrd(scl, scl, f)
        tim = @elapsed ser = newToe(vol, f)
        tms[string(scl) * " " * fk] = tim
        old = Dict(ord => oldToe(vol, f, ord) for ord ∈ ORDS[ky])
        for D ∈ OFFS
            pos = CartesianIndex((D .+ 1)...)
            sv = view(ser, :, :, pos)
            r = Dict{Int,Tuple{Float64,Float64}}()
            for ord ∈ ORDS[ky]
                r[ord] = tnsErr(view(old[ord], :, :, pos), sv)
            end
            hs = haskey(r, 64)
            vd = "--"
            if hs
                sm["row"] += 1
                vd = r[64][1] < r[48][1] ? (sm["cnv"] += 1; "cnv") :
                     (sm["plt"] += 1; "PLATEAU")
            end
            println(io, rpad(string(scl), 22), rpad(fk, 8), rpad(string(D), 11),
                    rpad(mM, 6), rpad(@sprintf("%.3e", r[48][1]), 12),
                    rpad(hs ? @sprintf("%.3e", r[64][1]) : "--", 12),
                    rpad(@sprintf("%.3e", r[48][2]), 12),
                    rpad(hs ? @sprintf("%.3e", r[64][2]) : "--", 12), vd)
        end
        flush(io)
    end
end
println(io)
println(io)
println(io, "## cold cntBlk! per (shape, frequency), seconds, ", Threads.nthreads(), " threads")
for (k, v) in sort(collect(tms)); println(io, rpad(k, 32), @sprintf("%.2f", v)); end
println(io)
println(io, "rows with both orders ", sm["row"], "  converging ", sm["cnv"],
        "  not converging ", sm["plt"])
println(io)
println(io, "## the order-9 touching shell, face pair by face pair, f = 1")
println(io, "   ord9 = egoSrfFxd!(..., cntOrd) * V_t against the exact series;")
println(io, "   the shell is every (offset, F, Fp) whose sub-code is not a contact code")
println(io)
prs = [(D, F, Fp, Mom.geomCode(D, F, Fp)) for D ∈ OFFS, F ∈ 1:6, Fp ∈ 1:6]
prs = [p for p ∈ prs if p[4] ∈ SHELL]
println(io, rpad("scl", 22), rpad("worst shell rel", 17), rpad("at", 30), "sub-code")
for (ky, scl) ∈ SCLS
    mM = GV.serOrd(scl, scl, 1.0 + 0.0im)
    sD = Double64.(scl)
    wst = 0.0; wat = ""; wcd = ""
    for (D, F, Fp, cod) ∈ prs
        g9, fp = ord9(D, F, Fp, scl, 1.0 + 0.0im)
        sv = first(GV.momSer(GV.parFac(D, F, Fp, sD)..., 1.0 + 0.0im, mM))
        rl = Float64(abs(ComplexDF64(g9) - sv) / abs(sv))
        rl > wst && (wst = rl; wat = string(D) * " fp " * string(fp); wcd = cod)
    end
    println(io, rpad(string(scl), 22), rpad(@sprintf("%.3e", wst), 17), rpad(wat, 30), wcd)
    flush(io)
end
end
