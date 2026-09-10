# Are the f = 1 + 0.1i rows of test/ref/far.txt evaluated at the decimal 0.1 rather than at
# Float64(0.1), which is the frequency Gila is given?
include(joinpath(@__DIR__, "refgen.jl"))
include(joinpath(@__DIR__, "rdref.jl"))
pr(x...) = (println(x...); flush(stdout))
setprecision(BigFloat, 260)
mx(A) = Float64(maximum(abs, A))

# rgTns takes a ComplexF64; here the frequency is carried in full, so a small local variant
function rgAt(R, s, fB::Complex{BigFloat}; ord = 32, prc = 256)
    bQ = ntuple(d -> s[d], 3)
    setprecision(BigFloat, prc) do
        T = BigFloat; C = Complex{T}
        k = 2 * T(pi) * fB
        Rb = ntuple(d -> T(R[d]), 3); sb = ntuple(d -> T(s[d]), 3)
        gx, gw = glr(ord, T)
        cut = ntuple(d -> [-sb[d], zero(T), sb[d]], 3)
        tot = zeros(C, 3, 3); Gk = zeros(C, 3, 3)
        for j1 in 1:2, j2 in 1:2, j3 in 1:2
            jj = (j1, j2, j3)
            lo = ntuple(d -> cut[d][jj[d]], 3); hi = ntuple(d -> cut[d][jj[d] + 1], 3)
            hw = ntuple(d -> (hi[d] - lo[d]) / 2, 3); md = ntuple(d -> (hi[d] + lo[d]) / 2, 3)
            pt = ntuple(d -> [md[d] + hw[d] * gx[q] for q in 1:ord], 3)
            wt = ntuple(d -> [hw[d] * gw[q] * (sb[d] - abs(md[d] + hw[d] * gx[q])) for q in 1:ord], 3)
            for q1 in 1:ord, q2 in 1:ord, q3 in 1:ord
                rgKer!(Gk, (Rb[1] + pt[1][q1], Rb[2] + pt[2][q2], Rb[3] + pt[3][q3]), k, fB)
                ww = wt[1][q1] * wt[2][q2] * wt[3][q3]
                for a in 1:3, b in 1:3; tot[a, b] += ww * Gk[a, b]; end
            end
        end
        tot ./ prod(sb)
    end
end

F = rdFar()
for (s, f, D, G) in F
    (f == 1.0 + 0.1im && D in ((3,0,0), (6,0,0), (3,3,0), (2,0,0), (8,8,0))) || continue
    R = ntuple(d -> Rational{BigInt}(D[d]) * s[d], 3)
    fDec = Complex{BigFloat}(BigFloat(1), parse(BigFloat, "0.1"))
    f64  = Complex{BigFloat}(BigFloat(1), BigFloat(0.1))
    A = rgAt(R, s, fDec); B = rgAt(R, s, f64)
    m = mx(B)
    pr("  s=", Float64.(s), " D=", D, "  file vs decimal-0.1 ", mx(G .- A) / m,
       "   file vs Float64-0.1 ", mx(G .- B) / m)
end
pr("ALLDONE")
