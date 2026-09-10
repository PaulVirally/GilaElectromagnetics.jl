# The one far.txt row where the independent quadrature disagreed with both the file and Gila at
# 6.7e-13: the slender cell at D = (1,1,2), whose difference box touches the singularity on two
# axes.  Re-run with the face-touching grading and two orders, so the reference carries its own
# uncertainty.
include(joinpath(@__DIR__, "refgen.jl"))
include(joinpath(@__DIR__, "rdref.jl"))
using GilaElectromagnetics
const GV = GilaElectromagnetics.GilaVacuum
pr(x...) = (println(x...); flush(stdout))
mx(A) = Float64(maximum(abs, A))
setprecision(BigFloat, 260)

F = rdFar()
for (s, f, D, G) in F
    s == (Rational{BigInt}(1,32), Rational{BigInt}(1,32), Rational{BigInt}(1,512)) || continue
    maximum(abs, D) <= 3 || continue
    R = ntuple(d -> Rational{BigInt}(D[d]) * s[d], 3)
    A = rgTns(R, s, s, ComplexF64(f); ord = 20)
    B = rgTns(R, s, s, ComplexF64(f); ord = 28)
    m = mx(B)
    Gg = GV.farTns(D, s, ComplexF64(f))
    pr("  D=", D, " f=", f, "  ref self ord20-28 ", mx(A .- B) / m,
       "  file-vs-ref ", mx(G .- B) / m, "  gila-vs-ref ", mx(Gg .- B) / m,
       "  gila-vs-file ", mx(Gg .- G) / m)
end
pr("ALLDONE")
