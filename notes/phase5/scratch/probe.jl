include(joinpath(@__DIR__, "refgen.jl"))
include(joinpath(@__DIR__, "rdref.jl"))
const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
module RQM
include(joinpath(Main.ROOT, "notes", "farfield", "refquad.jl"))
end
pr(x...) = (println(x...); flush(stdout))
F = rdFar()
for i in (30, 170, 1)
    s, f, D, G = F[i]
    pr("\n--- i=", i, " s=", Float64.(s), " f=", f, " D=", D)
    R = ntuple(d -> QI(D[d]) * s[d], 3)
    A24 = rgTns(R, s, s, f; ord = 24)
    A32 = rgTns(R, s, s, f; ord = 32)
    A40 = rgTns(R, s, s, f; ord = 40)
    Q = RQM.refQuad(D, s, ComplexF64(f); ord = 32, lvl = 1, cache = false)
    m = maximum(abs, G)
    pr("  self 24-32 ", Float64(maximum(abs, A24 .- A32) / m),
       "  32-40 ", Float64(maximum(abs, A32 .- A40) / m),
       "  vs refQuad ", Float64(maximum(abs, A32 .- Q) / m),
       "  refQuad vs file ", Float64(maximum(abs, Q .- G) / m))
    for q in 1:9
        d = abs(A32[q] - G[q]) / m
        d > 1e-30 && pr("   entry ", q, "  mine ", Float64(real(A32[q])), ";", Float64(imag(A32[q])),
                        "  file ", Float64(real(G[q])), ";", Float64(imag(G[q])), "  reldiff ", Float64(d))
    end
end
pr("done")
