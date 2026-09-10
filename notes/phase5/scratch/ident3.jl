# The operator Gila actually applies, entry by entry, against the tensor and against 220-bit truth.
using GilaElectromagnetics, LinearAlgebra
const GV = GilaElectromagnetics.GilaVacuum
include(joinpath(@__DIR__, "refgen.jl"))
pr(x...) = (println(x...); flush(stdout))
mx(A) = Float64(maximum(abs, A))
dns(mem) = begin
    n = prod(mem.srcVol.cel) * 3
    m = zeros(ComplexF64, n, n)
    for i in 1:n
        v = zeros(ComplexF64, mem.srcVol.cel..., 3); v[i] = 1
        m[:, i] .= vec(egoOpr!(mem, v))
    end
    m
end
for (cel, scl, frq) in (((5,5,5), (3//64,5//64,7//64), 1.0 + 0.1im),
                        ((4,4,6), (1//32,1//32,1//512), 0.5 + 2.0im))
    o = CPUKerOpt{Float64}(); o.frqPhz = frq
    m = dns(GlaVacOprMem(o, GlaVol(cel, scl, (0//1,0//1,0//1))))
    li = LinearIndices((cel..., 3))
    sQ = ntuple(d -> Rational{BigInt}(scl[d]), 3)
    Ds = unique(Tuple(c) .- Tuple(d) for c in CartesianIndices(cel), d in CartesianIndices(cel)
                if maximum(abs, Tuple(c) .- Tuple(d)) >= 2)
    fs = GV.farSet(sQ, frq; offs = Tuple(Ds), nBlk = 2 * maximum(maximum(abs, D) for D in Ds))
    T = Dict(D => copy(GV.farTns(D, scl, frq; fs = fs)) for D in Ds)
    wT = 0.0; n = 0
    for c in CartesianIndices(cel), d in CartesianIndices(cel)
        D = Tuple(c) .- Tuple(d)
        maximum(abs, D) >= 2 || continue
        n += 1
        M = [m[li[Tuple(c)..., a], li[Tuple(d)..., b]] for a in 1:3, b in 1:3]
        wT = max(wT, mx(M .- T[D]) / mx(T[D]))
    end
    # and five of them against the independent 220-bit quadrature
    wR = 0.0; nr = 0
    for D in Ds
        (all(!=(0), D) && nr < 5) || continue
        nr += 1
        R = rgTns(ntuple(q -> Rational{BigInt}(D[q]) * sQ[q], 3), sQ, sQ, frq; ord = 24)
        c0 = ntuple(q -> max(1, 1 - D[q]), 3); c1 = ntuple(q -> c0[q] + D[q], 3)
        M = [m[li[c1..., a], li[c0..., b]] for a in 1:3, b in 1:3]
        wR = max(wR, mx(M .- R) / mx(R))
    end
    pr("  cel=", cel, " scl=", Float64.(scl), " f=", frq, ": ", n, " separated pairs, ",
       length(Ds), " distinct offsets; worst |M - farTns|/|T| = ", wT,
       "; worst |M - 220-bit|/|T| on ", nr, " = ", wR)
end
pr("ALLDONE")
