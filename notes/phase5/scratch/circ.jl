# Are the low-precision entries of test/ref/far.txt bitwise equal to Gila's own Float64 output?
# If they are, that part of the reference set confirms the implementation against itself.
using GilaElectromagnetics
const GV = GilaElectromagnetics.GilaVacuum
include(joinpath(@__DIR__, "rdref.jl"))
include(joinpath(@__DIR__, "refgen.jl"))
pr(x...) = (println(x...); flush(stdout))
setprecision(BigFloat, 260)

isF64(x) = !iszero(x) && abs(x - BigFloat(Float64(x))) / abs(x) < 1e-30

F = rdFar()
tot = 0; f64n = 0; hitGila = 0; hitTrue = 0; ulp1 = 0
rows = 0
for (s, f, D, G) in F
    global tot, f64n, hitGila, rows
    Gg = GV.farTns(D, s, ComplexF64(f))
    any(isF64, real.(G)) || any(isF64, imag.(G)) || continue
    rows += 1
    for q in 1:9, (fv, gv) in ((real(G[q]), real(Gg[q])), (imag(G[q]), imag(Gg[q])))
        iszero(fv) && continue
        tot += 1
        isF64(fv) || continue
        f64n += 1
        Float64(fv) === gv && (hitGila += 1)
        # the correctly rounded 220-bit truth, from the independent quadrature
        nothing
    end
end
pr("far.txt rows with a Float64-looking entry: ", rows)
pr("  nonzero numbers in those rows: ", tot, "; Float64 round trips: ", f64n,
   "; bitwise equal to Gila's own farTns output: ", hitGila)

# on a handful, also compare against the independent 220-bit truth
n = 0
for (s, f, D, G) in F
    global n
    any(isF64, real.(G)) || any(isF64, imag.(G)) || continue
    n += 1; n > 6 && break
    Gg = GV.farTns(D, s, ComplexF64(f))
    Gr = rgTns(ntuple(d -> QI(D[d]) * s[d], 3), s, s, ComplexF64(f); ord = 32)
    m = Float64(maximum(abs, Gr))
    pr("  s=", Float64.(s), " f=", f, " D=", D,
       "  file-vs-truth ", Float64(maximum(abs, G .- Gr)) / m,
       "  gila-vs-truth ", Float64(maximum(abs, Gg .- Gr)) / m,
       "  file-vs-gila ", Float64(maximum(abs, G .- Gg)) / m,
       "  bitwise file==gila ", all(ComplexF64(G[q]) === Gg[q] for q in 1:9))
end
pr("done")
