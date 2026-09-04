#=
Convergence study behind the fixed quadrature schedule in quadOrd. Usage:

    julia --project=. -t auto benchmark/quadCnv.jl [den ...]

    den : cell size denominators in wavelengths, default 4 8 16 32 64 128

For every cell size, separation and offset direction it integrates the 36 face
pairs with tensor Gauss-Legendre rules of increasing order, with a single
Genz-Malik step (what the production adaptive tolerances amount to at these
separations), and with a reference: adaptive hcubature at rtol 1e-12 for
separations of 4 cells and more, GL24 at 2 and 3 cells, where the adaptive rule
converges too slowly to trust. It then prints, per cell size, the smallest
order reaching 1e-6 to 1e-12 relative error on the assembled dyadic, the
observed convergence rate, the face-pair error against the assembled error
(srfSum! differences the face pairs, so the two are not the same), and timings.

Raw results are cached in benchmark/results/quadCnv_<den>.jls and reused, so an
interrupted run resumes. Budget a few minutes per cell size, nearly all of it
in the adaptive references at separations of 4 to 6 cells. Prints only, asserts
nothing, and CI does not run it.
=#
include("bmkEnv.jl")

using GilaElectromagnetics
using LinearAlgebra
using Printf
using Serialization
using Statistics

const GV = GilaElectromagnetics.GilaVacuum
import GilaElectromagnetics.GilaVacuum: srfKer, srfSum!, srfScl, cubFac, facPar,
    gauQud

const cnvOpt = CPUKerOpt{Float64}()
const cnvFpr = facPar()
const cnvOrd = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16]
const cnvSep = [2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32]
const cnvDir = ["ax" => (1,0,0), "fd" => (1,1,0), "bd" => (1,1,1)]
const cnvDen = isempty(ARGS) ? [4, 8, 16, 32, 64, 128] : parse.(Int, ARGS)
const cnvOut = joinpath(@__DIR__, "results")

cnvRelFro(matA, matB) = norm(matA .- matB) / norm(matB)

function cnvAsm(srfVec)
    ego = zeros(ComplexF64, 3, 3)
    srfSum!(ego, srfVec)
    return ego
end

# tensor Gauss-Legendre integration of all 36 face pairs at order ord
function cnvGl(ker, sclVec, ord)
    qud = gauQud(ord)
    pos = (qud[:, 1] .+ 1) ./ 2
    wgt = qud[:, 2] ./ 2
    srfVec = zeros(ComplexF64, 36)
    Threads.@threads for fp ∈ 1:36
        acc = zero(ComplexF64)
        for itrI ∈ 1:ord, itrJ ∈ 1:ord, itrK ∈ 1:ord, itrL ∈ 1:ord
            acc += (wgt[itrI] * wgt[itrJ] * wgt[itrK] * wgt[itrL]) *
                ker(GV.SVector(pos[itrI], pos[itrJ], pos[itrK], pos[itrL]), fp)
        end
        srfVec[fp] = acc * sclVec[fp]
    end
    return srfVec
end

# adaptive integration, or a single Genz-Malik step when maxevals is 1
function cnvAdp(ker, sclVec, rtol, atol; maxevals=typemax(Int))
    srfVec = zeros(ComplexF64, 36)
    estVec = zeros(36)
    Threads.@threads for fp ∈ 1:36
        val, est = GV.hcubature(ordVec -> ker(ordVec, fp),
            GV.SVector(0.0, 0.0, 0.0, 0.0), GV.SVector(1.0, 1.0, 1.0, 1.0);
            rtol=rtol, atol=atol, maxevals=maxevals)
        srfVec[fp] = val * sclVec[fp]
        estVec[fp] = est
    end
    return srfVec, estVec
end

# fill the cache for one cell size
function cnvGen(den)
    mkpath(cnvOut)
    out = joinpath(cnvOut, "quadCnv_$(den).jls")
    res = isfile(out) ? deserialize(out) : Dict{Any,Any}()
    scl = ntuple(_ -> 1 // den, 3)
    fac = Float64.(cubFac(scl))
    sclVec = srfScl(Float64.(scl), Float64.(scl))
    for sep ∈ cnvSep, (dirNam, dir) ∈ cnvDir
        haskey(res, (dirNam, sep)) && continue
        off = Float64.(sep .* dir .* scl)
        ker = (ordVec, fp) -> srfKer(ordVec, off[1], off[2], off[3], fp, fac,
            fac, cnvFpr, cnvOpt)
        ent = Dict{Any,Any}()
        for ord ∈ vcat(cnvOrd, sep <= 3 ? [20, 24] : Int[])
            tim = @elapsed srfVec = cnvGl(ker, sclVec, ord)
            ent[("gl", ord)] = (vals=srfVec, ego=cnvAsm(srfVec), t=tim)
        end
        tim = @elapsed (srfVec, estVec) = cnvAdp(ker, sclVec, 1e-6, 1e-9;
            maxevals=1)
        ent[("gm", 57)] = (vals=srfVec, ego=cnvAsm(srfVec), est=estVec, t=tim)
        rtol, atol = sep >= 4 ? (1e-12, 1e-14) : (1e-10, 1e-12)
        tim = @elapsed (srfVec, estVec) = cnvAdp(ker, sclVec, rtol, atol)
        ent[("ref", 0)] = (vals=srfVec, ego=cnvAsm(srfVec), est=estVec, t=tim,
            rtol=rtol)
        res[(dirNam, sep)] = ent
        serialize(out, res)
        @printf("1//%d %s sep=%2d  ref %6.2f s  gl16 %5.2f s\n", den, dirNam,
            sep, tim, ent[("gl", 16)].t); flush(stdout)
    end
    return res
end

# GL24 is the reference at small separations, where GL20 and GL24 already agree
function cnvRef(ent)
    adp = ent[("ref", 0)]
    if haskey(ent, ("gl", 24)) &&
        cnvRelFro(ent[("gl", 20)].ego, ent[("gl", 24)].ego) < adp.rtol
        return ent[("gl", 24)].ego, ent[("gl", 24)].vals, "GL24"
    end
    return adp.ego, adp.vals, @sprintf("adp%.0e", adp.rtol)
end

cnvErr(ent, ref) = Dict(ord => cnvRelFro(ent[("gl", ord)].ego, ref)
    for ord ∈ cnvOrd if haskey(ent, ("gl", ord)))

function cnvMinOrd(err, thr)
    ords = [ord for ord ∈ cnvOrd if haskey(err, ord) && err[ord] < thr]
    return isempty(ords) ? ">16" : string(minimum(ords))
end

function cnvRep(den, res)
    @printf("\n==== scl = 1//%d λ (k·scl = %.3f) ====\n", den, 2π / den)
    println(rpad("s", 4), rpad("ref", 9), rpad("n<1e-6", 8), rpad("n<1e-8", 8),
        rpad("n<1e-10", 9), rpad("n<1e-12", 9), rpad("err GM57", 10),
        rpad("err n=4", 10), rpad("err n=8", 10), rpad("slope/n", 9),
        rpad("worstFp n=4", 13), rpad("t ref s", 9), "t GL8 s")
    for sep ∈ sort(unique(key[2] for key ∈ keys(res) if key[1] == "ax"))
        ent = res[("ax", sep)]
        ref, refVec, tag = cnvRef(ent)
        err = cnvErr(ent, ref)
        pts = [(ord, log10(err[ord])) for ord ∈ cnvOrd
            if haskey(err, ord) && 1e-13 < err[ord] < 1e-2]
        slope = length(pts) >= 3 ?
            cov(first.(pts), last.(pts)) / var(first.(pts)) : NaN
        big = abs.(refVec) .> 1e-6 * maximum(abs.(refVec))
        wfp = maximum(abs.(ent[("gl", 4)].vals[big] .- refVec[big]) ./
            abs.(refVec[big]))
        println(rpad(sep, 4), rpad(tag, 9), rpad(cnvMinOrd(err, 1e-6), 8),
            rpad(cnvMinOrd(err, 1e-8), 8), rpad(cnvMinOrd(err, 1e-10), 9),
            rpad(cnvMinOrd(err, 1e-12), 9),
            rpad(@sprintf("%.1e", cnvRelFro(ent[("gm", 57)].ego, ref)), 10),
            rpad(@sprintf("%.1e", err[4]), 10),
            rpad(@sprintf("%.1e", err[8]), 10),
            rpad(@sprintf("%.2f", slope), 9), rpad(@sprintf("%.1e", wfp), 13),
            rpad(@sprintf("%.2f", ent[("ref", 0)].t), 9),
            @sprintf("%.2f", ent[("gl", 8)].t))
    end
    # how much srfSum! amplifies face-pair error, over all offset directions
    amp = Float64[]
    for (dirNam, _) ∈ cnvDir, sep ∈ cnvSep
        haskey(res, (dirNam, sep)) || continue
        ent = res[(dirNam, sep)]
        ref, refVec, _ = cnvRef(ent)
        big = abs.(refVec) .> 1e-6 * maximum(abs.(refVec))
        for ord ∈ (4, 5, 6, 7)
            wfp = maximum(abs.(ent[("gl", ord)].vals[big] .- refVec[big]) ./
                abs.(refVec[big]))
            push!(amp, cnvRelFro(ent[("gl", ord)].ego, ref) / wfp)
        end
    end
    @printf("assembled / worst face pair error: median %.1f, max %.1f\n",
        median(amp), maximum(amp))
    return nothing
end

for den ∈ cnvDen
    cnvRep(den, cnvGen(den))
end
