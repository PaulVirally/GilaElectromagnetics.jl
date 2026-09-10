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
slfMem(cel, scl, frq) = (o = CPUKerOpt{Float64}(); o.frqPhz = frq;
                         GlaVacOprMem(o, GlaVol(cel, scl, (0//1, 0//1, 0//1))))

pr("### 2. the operator's separated entries against the tensor, and against 220-bit truth")
for (cel, scl, frq) in (((5,5,5), (3//64,5//64,7//64), 1.0 + 0.1im),
                        ((5,5,5), (1//32,1//32,1//512), 0.5 + 2.0im))
    m = dns(slfMem(cel, scl, frq))
    li = LinearIndices((cel..., 3))
    wT = 0.0; wR = 0.0; n = 0; nr = 0
    sQ = ntuple(d -> Rational{BigInt}(scl[d]), 3)
    for c in CartesianIndices(cel), d in CartesianIndices(cel)
        D = Tuple(c) .- Tuple(d)
        maximum(abs, D) >= 2 || continue
        n += 1
        T = GV.farTns(D, scl, frq)
        M = [m[li[Tuple(c)..., a], li[Tuple(d)..., b]] for a in 1:3, b in 1:3]
        wT = max(wT, mx(M .- T) / mx(T))
        if nr < 5 && all(!=(0), D)
            nr += 1
            R = rgTns(ntuple(q -> Rational{BigInt}(D[q]) * sQ[q], 3), sQ, sQ, frq; ord = 24)
            wR = max(wR, mx(M .- R) / mx(R))
        end
    end
    pr("  cel=", cel, " scl=", Float64.(scl), " f=", frq, ": ", n,
       " separated pairs; worst |M - farTns|/|T| = ", wT,
       "; worst |M - 220-bit|/|T| on ", nr, " = ", wR)
end
pr("### 1b. reciprocity, coarse cube")
for (cel, scl, frq) in (((3,3,3), (1//8,1//8,1//8), 1.0 + 0.0im),)
    t0 = time(); m = dns(slfMem(cel, scl, frq))
    pr("  [", round(time() - t0, digits = 1), " s] cel=", cel, " scl=", Float64.(scl),
       " f=", frq, "  |M - Mᵀ|/|M| = ", mx(m .- transpose(m)) / mx(m))
end
pr("ALLDONE")
