# Validate the new independent reference generator against the existing curated 220-bit
# references, and measure its own quadrature convergence.
include(joinpath(@__DIR__, "refgen.jl"))
include(joinpath(@__DIR__, "rdref.jl"))

pr(x...) = (println(x...); flush(stdout))
mx(A) = maximum(abs, A)
rel(A, B) = mx(A .- B) / max(mx(B), eps(BigFloat))

pr("threads = ", Threads.nthreads())

F = rdFar(); X = rdFarX()
pr("far.txt ", length(F), " records, farx.txt ", length(X))

# equal-cell spot checks across the shapes and frequencies the file covers
pick = [1, 30, 60, 90, 120, 150, 170]
for i in pick
    s, f, D, G = F[i]
    t = @elapsed A = rgTns(ntuple(d -> QI(D[d]) * s[d], 3), s, s, f; ord = 32)
    pr("eq  i=", i, " s=", Float64.(s), " f=", f, " D=", D,
       "  rel=", Float64(rel(A, G)), "  maxG=", Float64(mx(G)), "  ", round(t, digits = 1), "s")
end

# unequal-cell spot checks
pickx = [1, 20, 60, 100, 140, 180]
for i in pickx
    sT, sS, f, R, G = X[i]
    t = @elapsed A = rgTns(R, sT, sS, f; ord = 32)
    pr("neq i=", i, " sT=", Float64.(sT), " sS=", Float64.(sS), " f=", f, " R=", Float64.(R),
       "  rel=", Float64(rel(A, G)), "  maxG=", Float64(mx(G)), "  ", round(t, digits = 1), "s")
end

# self-convergence in ord and lvl on one equal and one unequal case
let (s, f, D, G) = F[60]
    R = ntuple(d -> QI(D[d]) * s[d], 3)
    A = rgTns(R, s, s, f; ord = 24)
    B = rgTns(R, s, s, f; ord = 32)
    C = rgTns(R, s, s, f; ord = 40)
    pr("conv eq  ord24-32 ", Float64(rel(A, B)), "  ord32-40 ", Float64(rel(B, C)))
end
let (sT, sS, f, R, G) = X[60]
    A = rgTns(R, sT, sS, f; ord = 24)
    B = rgTns(R, sT, sS, f; ord = 32)
    C = rgTns(R, sT, sS, f; ord = 40)
    pr("conv neq ord24-32 ", Float64(rel(A, B)), "  ord32-40 ", Float64(rel(B, C)))
end
pr("done")
