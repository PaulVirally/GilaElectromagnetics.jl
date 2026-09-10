# Re-derive every curated reference in test/ref/{far,farx}.txt from an independent 220-bit
# quadrature and score both the file and Gila against it.
using GilaElectromagnetics
const GV = GilaElectromagnetics.GilaVacuum
include(joinpath(@__DIR__, "rdref.jl"))
include(joinpath(@__DIR__, "refgen.jl"))
pr(x...) = (println(x...); flush(stdout))
setprecision(BigFloat, 260)
const WHICH = length(ARGS) >= 1 ? ARGS[1] : "far"

if WHICH == "far"
    F = rdFar()
    wf = 0.0; wg = 0.0; nbad = 0
    for (i, (s, f, D, G)) in enumerate(F)
        Gr = rgTns(ntuple(d -> QI(D[d]) * s[d], 3), s, s, ComplexF64(f); ord = 32)
        m = Float64(maximum(abs, Gr))
        ef = Float64(maximum(abs, G .- Gr)) / m
        Gg = GV.farTns(D, s, ComplexF64(f))
        eg = Float64(maximum(abs, Gg .- Gr)) / m
        global wf = max(wf, ef); global wg = max(wg, eg)
        if ef > 1e-25
            global nbad += 1
            pr("  LOWPREC row ", i, " s=", Float64.(s), " f=", f, " D=", D,
               " file-vs-truth ", ef, "  gila-vs-truth ", eg)
        end
        eg > 1e-13 && pr("  GILA row ", i, " s=", Float64.(s), " f=", f, " D=", D, " err ", eg)
        i % 20 == 0 && pr("  .. ", i, " worst file ", wf, " worst gila ", wg)
    end
    pr("far.txt: ", length(F), " rows; worst file-vs-truth ", wf, "; worst gila-vs-truth ", wg,
       "; rows above 1e-25 ", nbad)
else
    X = rdFarX()
    wf = 0.0; wg = 0.0; nbad = 0
    for (i, (sT, sS, f, R, G)) in enumerate(X)
        Gr = rgTns(R, sT, sS, ComplexF64(f); ord = 28)
        m = Float64(maximum(abs, Gr))
        ef = Float64(maximum(abs, G .- Gr)) / m
        Gg = GV.farTnsX(R, sT, sS, ComplexF64(f))
        eg = Float64(maximum(abs, Gg .- Gr)) / m
        global wf = max(wf, ef); global wg = max(wg, eg)
        if ef > 1e-25
            global nbad += 1
            pr("  LOWPREC row ", i, " sT=", Float64.(sT), " sS=", Float64.(sS), " f=", f,
               " R=", Float64.(R), " file-vs-truth ", ef, "  gila-vs-truth ", eg)
        end
        eg > 1e-13 && pr("  GILA row ", i, " sT=", Float64.(sT), " sS=", Float64.(sS),
                         " f=", f, " R=", Float64.(R), " err ", eg)
        i % 20 == 0 && pr("  .. ", i, " worst file ", wf, " worst gila ", wg)
    end
    pr("farx.txt: ", length(X), " rows; worst file-vs-truth ", wf, "; worst gila-vs-truth ", wg,
       "; rows above 1e-25 ", nbad)
end
pr("ALLDONE")
