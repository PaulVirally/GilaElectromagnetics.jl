# Identities that hold whatever the implementation: reciprocity, evenness, reflection, the
# cross-scale swap, moment scale homogeneity, and the operator's own entries against the tensor.
using GilaElectromagnetics, LinearAlgebra
const GV = GilaElectromagnetics.GilaVacuum
const GVM = GilaElectromagnetics.GilaVacuum
include(joinpath(@__DIR__, "refgen.jl"))
pr(x...) = (println(x...); flush(stdout))
mx(A) = Float64(maximum(abs, A))

dns(mem::GlaVacOprMem) = begin
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

pr("### 3. T(-R) = T(R) and T_ab(sig R) = sig_a sig_b T_ab(R), equal cells, new shapes")
for (scl, frq) in (((3//64,5//64,7//64), 1.0 + 0.1im), ((1//6,1//96,1//96), 0.5 + 2.0im),
                   ((1//2,1//2,1//2), 1.0 + 0.0im))
    sQ = ntuple(d -> Rational{BigInt}(scl[d]), 3)
    we = 0.0; wr = 0.0
    for D in ((2,0,0), (3,1,2), (2,2,2), (0,0,3), (4,-3,2))
        T = GV.farTns(D, scl, frq)
        we = max(we, mx(T .- GV.farTns(.-D, scl, frq)) / mx(T))
        for sg in ((-1,1,1), (1,-1,1), (1,1,-1), (-1,-1,1))
            S = GV.farTns(ntuple(d -> sg[d] * D[d], 3), scl, frq)
            P = [sg[a] * sg[b] * T[a,b] for a in 1:3, b in 1:3]
            wr = max(wr, mx(S .- P) / mx(T))
        end
    end
    pr("  scl=", Float64.(scl), " f=", frq, "  evenness ", we, "  reflection ", wr)
end

pr("### 4. cross-scale swap  T(R; sT, sS) = (V_s/V_t) T(-R; sS, sT)")
for (sT, sS, frq, Rs) in (((1//32,1//32,1//32), (3//32,3//32,3//32), 1.0 + 0.1im,
                           ((7//32,0//1,0//1), (5//32,5//32,3//32))),
                          ((1//8,1//8,1//8), (1//32,1//32,1//32), 0.5 + 2.0im,
                           ((5//16,0//1,0//1), (3//8,1//8,1//8))),
                          ((1//16,1//48,1//48), (1//16,1//16,1//48), 2.0 + 2.0im,
                           ((5//16,0//1,0//1),)))
    vt = prod(Float64.(sT)); vs = prod(Float64.(sS))
    for R in Rs
        A = GV.farTnsX(R, sT, sS, frq)
        B = GV.farTnsX(ntuple(d -> -R[d], 3), sS, sT, frq)
        C = GV.farTnsX(R, sS, sT, frq)
        pr("  sT=", Float64.(sT), " sS=", Float64.(sS), " R=", Float64.(R),
           "  swap ", mx(A .- (vs / vt) .* B) / mx(A),
           "  evenness of the swap ", mx(B .- C) / mx(B))
    end
end

pr("### 5. moment scale homogeneity, bitwise: parMom(lam s) === lam^(m+4) parMom(s)")
let
    bad = 0; tot = 0; wnd = 0.0
    for base in ((1.0, 1.0, 1.0), (1.0, 0.5, 0.25), (1.0, 1.0, 1 / 16), (2.0, 3.0, 5.0)),
        D in ((0,0,0), (1,0,0), (1,1,0), (1,1,1)), F in 1:6, Fp in 1:6, lam in (2.0, 0.5, 0.125)
        v = GVM.parMom(GVM.parFac(D, F, Fp, base)..., 10, Float64)
        w = GVM.parMom(GVM.parFac(D, F, Fp, lam .* base)..., 10, Float64)
        for m in -1:10
            tot += 1
            v[m + 2] === zero(Float64) && w[m + 2] === zero(Float64) && continue
            lam^(m + 4) * v[m + 2] === w[m + 2] || (bad += 1)
        end
    end
    pr("  dyadic lam, === : ", tot - bad, " / ", tot, " exact, ", bad, " not bitwise")
    for base in ((1.0, 1.0, 1.0), (1.0, 1.0, 1 / 16)), D in ((0,0,0), (1,1,1)),
        F in 1:6, Fp in 1:6, lam in (3.0, 1 / 7, 1 / 10)
        v = GVM.parMom(GVM.parFac(D, F, Fp, base)..., 10, Float64)
        w = GVM.parMom(GVM.parFac(D, F, Fp, lam .* base)..., 10, Float64)
        for m in -1:10
            iszero(w[m + 2]) && continue
            wnd = max(wnd, abs(lam^(m + 4) * v[m + 2] - w[m + 2]) / abs(w[m + 2]))
        end
    end
    pr("  non-dyadic lam, worst relative = ", wnd)
end
pr("### 1. reciprocity of the self operator: M complex-symmetric (uniform cells, so dV cancels)")
for (cel, scl, frq) in (((4,4,4), (1//32,1//32,1//32), 1.0 + 0.0im),
                        ((4,4,4), (1//32,1//32,1//32), 1.0 + 0.1im),
                        ((3,4,5), (3//64,5//64,7//64), 1.0 + 0.0im),
                        ((3,3,6), (1//8,1//24,1//48), 0.5 + 2.0im),
                        ((4,4,4), (1//6,1//96,1//96), 2.0 + 2.0im),
                        ((3,3,3), (1//8,1//8,1//8), 1.0 + 0.0im))
    t0 = time(); m = dns(slfMem(cel, scl, frq)); tb = round(time() - t0, digits = 1)
    pr("  [", tb, " s] cel=", cel, " scl=", Float64.(scl), " f=", frq,
       "  |M - Mᵀ|/|M| = ", mx(m .- transpose(m)) / mx(m))
end

pr("### 2. the operator's separated entries against the tensor, and against 220-bit truth")
let cel = (5,5,5), scl = (3//64,5//64,7//64), frq = 1.0 + 0.1im
    m = dns(slfMem(cel, scl, frq))
    li = LinearIndices((cel..., 3))
    wT = 0.0; wR = 0.0; n = 0
    sQ = ntuple(d -> Rational{BigInt}(scl[d]), 3)
    for c in CartesianIndices(cel), d in CartesianIndices(cel)
        D = Tuple(c) .- Tuple(d)
        maximum(abs, D) >= 2 || continue
        (D[1] < 0 || (D[1] == 0 && D[2] < 0) || (D[1] == 0 && D[2] == 0 && D[3] < 0)) && continue
        n += 1
        T = GV.farTns(D, scl, frq)
        M = [m[li[Tuple(c)..., a], li[Tuple(d)..., b]] for a in 1:3, b in 1:3]
        wT = max(wT, mx(M .- T) / mx(T))
        if n <= 6
            R = rgTns(ntuple(q -> Rational{BigInt}(D[q]) * sQ[q], 3), sQ, sQ, frq; ord = 24)
            wR = max(wR, mx(M .- R) / mx(R))
        end
    end
    pr("  ", n, " separated cell pairs; worst |M - farTns|/|T| = ", wT,
       "; worst |M - 220-bit|/|T| on 6 = ", wR)
end

pr("ALLDONE")
