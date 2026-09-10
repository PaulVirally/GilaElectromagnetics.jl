# Phase 0-B, items 1 and 2. Standalone: includes notes/moments/moments.jl only.
# Run: JULIA_NUM_THREADS=auto julia --project --startup-file=no notes/phase0/scratch-b/b12.jl
using Printf
include(joinpath(@__DIR__, "..", "..", "moments", "moments.jl"))

const OUT = joinpath(@__DIR__, "b12_out.txt")
io = open(OUT, "w")
tee(x...) = (println(x...); println(io, x...))

# ===== item 1: 288 face pairs per shape (8 contact offsets x 36 face pairs),
# mMax = 12 and 30, serial and threaded, for the four production shapes
# (farfield/verify.jl's SHP: c32, c8, c4, sl).
const PRODSHAPES = [("c32", (1/32, 1/32, 1/32)),
                     ("c8",  (1/8,  1/8,  1/8)),
                     ("c4",  (1/4,  1/4,  1/4)),
                     ("sl",  (1/32, 1/32, 1/512))]

# the eight contact offsets: D_i in {0,1}, matching genEgoCrcSlf!'s egoFunSng!
# domain (posItr indices 1 or 2 in each axis)
const OFFS8 = [(d1, d2, d3) for d1 in 0:1, d2 in 0:1, d3 in 0:1][:]

function facePairCost(s, mMax; threaded::Bool)
    n = 0
    if threaded
        acc = zeros(Float64, Threads.maxthreadid())
        pairs = [(D, F, Fp) for D in OFFS8 for F in 1:6 for Fp in 1:6]
        Threads.@threads for i in eachindex(pairs)
            D, F, Fp = pairs[i]
            v = faceMoments(D, F, Fp, s, mMax)
            acc[Threads.threadid()] += abs(v[1])
        end
        return sum(acc)
    else
        acc = 0.0
        for D in OFFS8, F in 1:6, Fp in 1:6
            v = faceMoments(D, F, Fp, s, mMax)
            acc += abs(v[1])
        end
        return acc
    end
end

tee("=== item 1: 288 face pairs/shape, mMax in (12, 30), nthreads = ", Threads.nthreads(), " ===")
tee(rpad("shape", 8), rpad("mMax", 6), rpad("serial s", 12), rpad("serial us/pair", 16),
    rpad("thread s", 12), rpad("thread us/pair", 16))
for (nm, s) in PRODSHAPES, mMax in (12, 30)
    # warm up (compile)
    facePairCost(s, mMax; threaded = false)
    facePairCost(s, mMax; threaded = true)
    # serial: best of 3
    tS = minimum(begin
        t0 = time_ns(); facePairCost(s, mMax; threaded = false); (time_ns() - t0) / 1e9
    end for _ in 1:3)
    tT = minimum(begin
        t0 = time_ns(); facePairCost(s, mMax; threaded = true); (time_ns() - t0) / 1e9
    end for _ in 1:3)
    tee(rpad(nm, 8), rpad(mMax, 6), rpad(@sprintf("%.5f", tS), 12),
        rpad(@sprintf("%.2f", tS * 1e6 / 288), 16),
        rpad(@sprintf("%.5f", tT), 12),
        rpad(@sprintf("%.2f", tT * 1e6 / 288), 16))
end

# ===== item 2: series accuracy vs a 256-bit sum.
# five canonical touching-shell sub-codes (S, Ef, Ec, Vc, Vp), the same ones
# moments.tex section "The weak integrals" uses, at six cell scales (the
# production shapes plus cube = lambda and lambda/2) and seven frequencies.
const SCALES2 = [("l32", (1//32, 1//32, 1//32)), ("l8", (1//8, 1//8, 1//8)),
                  ("l4", (1//4, 1//4, 1//4)), ("l2", (1//2, 1//2, 1//2)),
                  ("l1", (1//1, 1//1, 1//1)), ("sl", (1//32, 1//32, 1//512))]
const FREQS2 = [("1", 1.0 + 0.0im), ("0.37", 0.37 + 0.0im), ("3", 3.0 + 0.0im),
                ("1+0.1i", 1.0 + 0.1im), ("1+1i", 1.0 + 1.0im),
                ("0.5+2i", 0.5 + 2.0im), ("2+2i", 2.0 + 2.0im)]
const SUBC2 = ["S", "Ef", "Ec", "Vc", "Vp"]
const RELTOL = 1e-16
const MMAXCAP = 100 # a single Float64 momentSeries call at mMax=100 costs
# well under 1s (measured 0.265s at mMax=80, cubic-ish growth); mMax=140 in a
# unit-step scan measured 95s for ONE combination, so the search below only
# ever tries a handful of geometrically-spaced mMax values, never a scan.

# a priori guess (no moments evaluated): smallest n with (2 pi |f| rMax)^n / n! < tol
function serGss(pA, pB, f, tol)
    r2 = sum(max(abs(pA[d][1] - pB[d][2]), abs(pA[d][2] - pB[d][1]))^2 for d in 1:3)
    x = 2 * pi * abs(f) * sqrt(r2)
    lg = 0.0; n = 0
    while n < 400
        n += 1
        lg += log(x) - log(n)
        lg < log(tol) && break
    end
    return max(4, n)
end

# geometric search from the a priori guess: a handful of momentSeries calls,
# not a unit-step scan (which is O(mMax^2) calls of O(mMax^3) cost each and was
# measured to take 95s for one combination at mMax ~ 140).
function findMMax(pA, pB, f, tol; cap = MMAXCAP)
    g = min(serGss(pA, pB, f, tol), cap)
    for mMax in unique(min.(cap, (g, round(Int, 1.3g), round(Int, 1.7g), 2g, cap)))
        _, rel = momentSeries(pA, pB, f, mMax)
        rel < tol && return (mMax, rel, true)
    end
    _, rel = momentSeries(pA, pB, f, cap)
    return (cap, rel, false)
end

tee()
tee("=== item 2: series accuracy vs 256-bit sum, tol = ", RELTOL, " ===")
tee(rpad("scale", 6), rpad("code", 6), rpad("freq", 8), rpad("mMax", 6),
    rpad("rel(64)", 12), rpad("converged", 10), rpad("digits lost", 14), "note")

worstDigits = Dict{String,Float64}()
noConverge = Tuple{String,String,String,Float64,Int}[]
for (snm, s) in SCALES2
    a, b, c = Float64(s[1]), Float64(s[2]), Float64(s[3])
    for code in SUBC2
        pA64, pB64 = canonPanels(code, a, b, c)
        for (fnm, f) in FREQS2
            mMax, rel64, conv = findMMax(pA64, pB64, f, RELTOL)
            if !conv
                push!(noConverge, (snm, code, fnm, rel64, mMax))
                tee(rpad(snm, 6), rpad(code, 6), rpad(fnm, 8), rpad(mMax, 6),
                    rpad(@sprintf("%.2e", rel64), 12), rpad(conv, 10),
                    rpad("-", 14), "NOT CONVERGED at cap $(MMAXCAP)")
                continue
            end
            val64, _ = momentSeries(pA64, pB64, f, mMax)
            # 256-bit reference at the SAME mMax (no search): the point is
            # truncation, not cancellation, and 256 bits has ~200 bits of
            # spare headroom over Float64's cancellation floor, so the same
            # term count that satisfied Float64's own indicator is enough.
            setprecision(BigFloat, 256) do
                aB, bB, cB = BigFloat(s[1]), BigFloat(s[2]), BigFloat(s[3])
                pAB, pBB = canonPanels(code, aB, bB, cB)
                fB = Complex{BigFloat}(f)
                valB, relB = momentSeries(pAB, pBB, fB, mMax)
                dig = Float64(abs(val64 - valB) / (eps(Float64) * abs(valB)))
                dig = dig <= 0 ? 0.0 : log10(dig)
                worstDigits[snm] = max(get(worstDigits, snm, -Inf), dig)
                note = relB > 1e-30 ? "ref rel=$(relB)" : ""
                tee(rpad(snm, 6), rpad(code, 6), rpad(fnm, 8), rpad(mMax, 6),
                    rpad(@sprintf("%.2e", rel64), 12), rpad(conv, 10),
                    rpad(@sprintf("%.3f", dig), 14), note)
            end
        end
    end
end

tee()
tee("worst digits lost per scale:")
for (snm, _) in SCALES2
    tee("  ", snm, "  ", @sprintf("%.3f", worstDigits[snm]))
end
tee()
tee("non-converged (scale, code, freq) at mMax = ", MMAXCAP, " cap: ", noConverge)

close(io)
