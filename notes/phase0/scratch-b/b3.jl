# Phase 0-B, item 3: srfSum! assembly amplification Lambda on the eight
# near (contact) offsets per production shape. Read-only use of
# GilaElectromagnetics (GV.facPar, GV.srfSum!) plus notes/moments/moments.jl.
# Nothing under src/ is modified.
using Printf
using GilaElectromagnetics
const GV = GilaElectromagnetics.GilaVacuum
include(joinpath(@__DIR__, "..", "..", "moments", "moments.jl"))

const OUT = joinpath(@__DIR__, "b3_out.txt")
io = open(OUT, "w")
tee(x...) = (println(x...); println(io, x...))

const PRODSHAPES = [("c32", (1/32, 1/32, 1/32)),
                     ("c8",  (1/8,  1/8,  1/8)),
                     ("c4",  (1/4,  1/4,  1/4)),
                     ("sl",  (1/32, 1/32, 1/512))]
const OFFS8 = [(d1, d2, d3) for d1 in 0:1, d2 in 0:1, d3 in 0:1][:]
const FREQS3 = [("1", 1.0 + 0.0im), ("0.37", 0.37 + 0.0im), ("3", 3.0 + 0.0im),
                ("1+0.1i", 1.0 + 0.1im), ("1+1i", 1.0 + 1.0im),
                ("0.5+2i", 0.5 + 2.0im), ("2+2i", 2.0 + 2.0im)]

FP = GV.facPar()  # 2x36: row1 = target face, row2 = source face

# well-converged mMax for these small production shapes (largest edge here is
# 1/4; a priori guess with generous margin, cheap at these scales)
function serGss(pA, pB, f, tol)
    r2 = sum(max(abs(pA[d][1] - pB[d][2]), abs(pA[d][2] - pB[d][1]))^2 for d in 1:3)
    x = 2 * pi * abs(f) * sqrt(r2)
    lg = 0.0; n = 0
    while n < 300
        n += 1
        lg += log(x) - log(n)
        lg < log(tol) && break
    end
    return max(4, n)
end

function srfMatFor(D, s, f)
    m = zeros(ComplexF64, 36)
    for k in 1:36
        F, Fp = FP[1, k], FP[2, k]
        pA, pB = facePair(D, F, Fp, s)
        g = max(serGss(pA, pB, f, 1e-16), 8)
        v, rel = momentSeries(pA, pB, f, g)
        if rel > 1e-14
            v, rel = momentSeries(pA, pB, f, min(2g, 160))
        end
        m[k] = v
    end
    return m
end

tee("=== item 3: srfSum! amplification Lambda on the eight contact offsets ===")
tee(rpad("shape", 6), rpad("D", 12), rpad("freq", 8), rpad("a", 3), rpad("b", 3),
    rpad("|Gab|", 14), rpad("sum|fp|", 14), rpad("Lambda", 12))

epsB = setprecision(BigFloat, 128) do
    Float64(eps(BigFloat))
end
tee()
tee("eps at 128 bits (BigFloat, precision=128) = ", epsB)
tee()

worstPerShape = Dict{String,Float64}()
worstRow = Dict{String,Any}()
for (nm, s) in PRODSHAPES, D in OFFS8, (fnm, f) in FREQS3
    srfMat = srfMatFor(D, s, f)
    egoCrc = zeros(ComplexF64, 3, 3)
    GV.srfSum!(egoCrc, srfMat)
    # which face pairs feed each entry: reproduce srfSum!'s own index sets by
    # inspecting which srfMat perturbations move which entries (index sets
    # copied from the same source read for this report, not re-derived).
    idxSets = Dict(
        (1,1) => [15,16,21,22,29,30,35,36], (2,1) => [13,14,19,20],
        (3,1) => [25,26,31,32], (1,2) => [3,4,9,10],
        (2,2) => [1,2,7,8,29,30,35,36], (3,2) => [27,28,33,34],
        (1,3) => [5,6,11,12], (2,3) => [17,18,23,24],
        (3,3) => [1,2,7,8,15,16,21,22])
    for (ab, idx) in idxSets
        a, b = ab
        Gab = egoCrc[a, b]
        iszero(Gab) && continue
        s36 = sum(abs(srfMat[k]) for k in idx)
        lam = s36 / abs(Gab)
        tee(rpad(nm, 6), rpad(string(D), 12), rpad(fnm, 8), rpad(a, 3), rpad(b, 3),
            rpad(@sprintf("%.4e", abs(Gab)), 14), rpad(@sprintf("%.4e", s36), 14),
            rpad(@sprintf("%.4e", lam), 12))
        if lam > get(worstPerShape, nm, -Inf)
            worstPerShape[nm] = lam
            worstRow[nm] = (D, fnm, a, b, lam)
        end
    end
end

tee()
tee("worst Lambda per shape, and eps_128 * Lambda:")
for (nm, _) in PRODSHAPES
    lam = worstPerShape[nm]
    tee("  ", nm, "  Lambda=", @sprintf("%.4e", lam), "  at ", worstRow[nm],
        "  eps128*Lambda=", @sprintf("%.4e", epsB * lam))
end

close(io)
