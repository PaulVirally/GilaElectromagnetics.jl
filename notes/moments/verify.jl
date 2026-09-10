# Verification of moments.jl. Runnable in parts, since (a) and (b) are slow:
#   julia --startup-file=no verify.jl a   correctness, 15 sub-codes x 11 shapes, m = -1..12
#   julia --startup-file=no verify.jl b   all 144 face pairs vs pairMom, m = -1..8
#   julia --startup-file=no verify.jl c   explicit forms, homogeneity, symmetries, orientation, tol
#   julia --startup-file=no verify.jl d   digits lost in Float64 over the 13 x 14 aspect grid
#   julia --startup-file=no verify.jl e   Float64 timing per sub-code at mMax = 12
# Each part writes tables/<part>_*.txt as well as stdout. Only (b) calls pairMom
# and only (c) calls pairKer (notes/gen/ref/mom.jl); (a) uses refcache.txt for
# m = -1 and odd m and an exact Rational polynomial for even m, so no cached case
# is recomputed.
using Printf
include(joinpath(@__DIR__, "moments.jl"))

const DIR = joinpath(@__DIR__, "tables")
const PART = length(ARGS) == 0 ? "all" : ARGS[1]

# collect a table, then print it and write it to tables/<nam>
function tee(f, nam)
    buf = IOBuffer()
    f(buf)
    txt = String(take!(buf))
    print(stdout, txt)
    write(joinpath(DIR, nam), txt)
end

fm(x) = iszero(x) ? "0       " : @sprintf("%.2e", x)

# the 11 BRIEF shapes, built exactly as the reference cache was built
function shapes(::Type{T}) where {T}
    o = one(T)
    return [("cube", (o, o, o)),
            ("thinX", (o / 1000, o, o)), ("thinY", (o, o / 1000, o)),
            ("thinZ", (o, o, o / 1000)),
            ("fatX", (T(1000), o, o)), ("fatY", (o, T(1000), o)),
            ("fatZ", (o, o, T(1000))),
            ("small", (o / 32, o / 32, o / 32)),
            ("sliver_zzz", (o / 32, o / 32, o / 512)),
            ("sliver_zyz", (o / 32, o / 512, o / 32)),
            ("sliver_zzy", (o / 512, o / 32, o / 32))]
end

# ===== exact even-m reference, independent of the box decomposition.
# r^{2k} = sum_{i+j+l=k} k!/(i!j!l!) prod_d (x_d - y_d)^{2 e_d}, and each axis
# factor is an elementary polynomial integral, exact in Rational{BigInt}:
#   both axes degenerate    (p-r)^n
#   A degenerate            [(p-r)^{n+1} - (p-s)^{n+1}]/(n+1)
#   B degenerate            [(q-r)^{n+1} - (p-r)^{n+1}]/(n+1)
#   neither                 [H(q-r) - H(q-s) - H(p-r) + H(p-s)], H(u) = u^{n+2}/((n+1)(n+2))
function axsPow(A, B, n::Integer)
    p, q = A; r, s = B
    p == q && r == s && return (p - r)^n
    p == q && return ((p - r)^(n + 1) - (p - s)^(n + 1)) // (n + 1)
    r == s && return ((q - r)^(n + 1) - (p - r)^(n + 1)) // (n + 1)
    H(u) = u^(n + 2) // ((n + 1) * (n + 2))
    return H(q - r) - H(q - s) - H(p - r) + H(p - s)
end

function momPol(pA, pB, m::Integer)
    iseven(m) && m >= 0 || error("momPol: even m >= 0 only")
    Q = Rational{BigInt}
    qA = ntuple(d -> (Q(pA[d][1]), Q(pA[d][2])), 3)
    qB = ntuple(d -> (Q(pB[d][1]), Q(pB[d][2])), 3)
    k = m ÷ 2
    fac(n) = factorial(big(n))
    acc = zero(Q)
    for i in 0:k, j in 0:(k - i)
        l = k - i - j
        cof = fac(k) ÷ (fac(i) * fac(j) * fac(l))
        acc += cof * axsPow(qA[1], qB[1], 2i) * axsPow(qA[2], qB[2], 2j) *
               axsPow(qA[3], qB[3], 2l)
    end
    return acc
end

# ===== (a) 15 sub-codes x 11 shapes x m = -1..12, BigFloat 320
function partA()
    setprecision(BigFloat, 320)
    ref = loadRef(joinpath(@__DIR__, "refcache.txt"))
    shp = shapes(BigFloat)
    tee("a_correctness.txt") do io
        println(io, "(a) pairMoments vs reference, BigFloat 320 bits, m = -1..12")
        println(io, "    m = -1 and odd m: refcache.txt (pairMom, 320 bits, orders 44/30/44)")
        println(io, "    even m: exact Rational polynomial (momPol), independent of pairBxs")
        println(io, "    target <= 1e-30")
        println(io)
        wrst = Dict{String,Tuple{Float64,String,Int}}()
        wshp = Dict{Tuple{String,String},Float64}()
        wm = Dict{Tuple{String,Int},Float64}()
        nbad = 0
        for cod in SUBCODES, (snm, (a, b, c)) in shp
            A, B = canonPanels(cod, a, b, c)
            v = pairMoments(A, B, 12)
            for m in -1:12
                if iseven(m)
                    rf = BigFloat(momPol(A, B, m))
                else
                    ky = (cod, Float64(a), Float64(b), Float64(c), m)
                    haskey(ref, ky) || error("cache miss $ky")
                    rf = ref[ky]
                end
                e = Float64(abs(v[m + 2] - rf) / abs(rf))
                e > 1e-30 && (nbad += 1;
                    println(io, "  ABOVE BAR: ", cod, " ", snm, " m=", m, " rel=", e))
                e > get(wrst, cod, (0.0, "", 0))[1] && (wrst[cod] = (e, snm, m))
                wshp[(cod, snm)] = max(get(wshp, (cod, snm), 0.0), e)
                wm[(cod, m)] = max(get(wm, (cod, m), 0.0), e)
            end
        end
        println(io, "worst relative error per sub-code and shape")
        print(io, rpad("code", 8)); for (s, _) in shp; print(io, rpad(s, 11)); end; println(io)
        for cod in SUBCODES
            print(io, rpad(cod, 8))
            for (s, _) in shp; print(io, rpad(fm(wshp[(cod, s)]), 11)); end
            println(io)
        end
        println(io)
        println(io, "worst per sub-code and m")
        print(io, rpad("code", 8)); for m in -1:12; print(io, rpad("m=$m", 10)); end; println(io)
        for cod in SUBCODES
            print(io, rpad(cod, 8))
            for m in -1:12; print(io, rpad(fm(wm[(cod, m)]), 10)); end
            println(io)
        end
        println(io)
        println(io, "worst per sub-code")
        for cod in SUBCODES
            e, s, m = wrst[cod]
            println(io, "  ", rpad(cod, 8), fm(e), "   at ", s, " m=", m)
        end
        gl = maximum(v[1] for v in values(wrst))
        println(io, "GLOBAL worst ", gl, "   values above 1e-30: ", nbad,
                " of ", 15 * 11 * 14)
    end
end

# ===== (b) all 144 face pairs vs pairMom
function partB()
    setprecision(BigFloat, 256)
    s = (BigFloat(3) / 7, BigFloat(5) / 11, BigFloat(2) / 3)
    tee("b_faces.txt") do io
        println(io, "(b) all 144 face pairs, s = (3/7, 5/11, 2/3), BigFloat 256, m = -1..8")
        println(io, "    faceMoments vs pairMom (orders 44/30/44)")
        println(io)
        wrst = 0.0; wat = ""; nbad = 0; n = 0
        bycod = Dict{String,Float64}()
        for D in ((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)), F in 1:6, Fp in 1:6
            cod = geomCode(D, F, Fp)
            A, B = facePair(D, F, Fp, s)
            rf = pairMom(A, B, 8)
            v = pairMoments(A, B, 8)
            mx = 0.0; mm = 0
            for m in -1:8
                e = Float64(abs(v[m + 2] - rf[m + 2]) / abs(rf[m + 2]))
                n += 1
                e > 1e-30 && (nbad += 1)
                e > mx && (mx = e; mm = m)
            end
            bycod[cod] = max(get(bycod, cod, 0.0), mx)
            mx > wrst && (wrst = mx; wat = "D=$D $(FACENAMES[F])-$(FACENAMES[Fp]) $cod m=$mm")
            println(io, "  D=", D, " ", rpad(FACENAMES[F], 4), rpad(FACENAMES[Fp], 5),
                    rpad(cod, 7), " worst ", fm(mx), " at m=", mm)
            println(stderr, "  b: ", D, " ", FACENAMES[F], "-", FACENAMES[Fp], " ", fm(mx))
        end
        println(io)
        println(io, "worst per sub-code")
        for cod in SUBCODES
            haskey(bycod, cod) && println(io, "  ", rpad(cod, 8), fm(bycod[cod]))
        end
        println(io, "values ", n, "   above 1e-30: ", nbad, "   worst ", wrst, "  at ", wat)
    end
end

# ===== (c) explicit forms, homogeneity, symmetries, orientation invariance
prm3(p, X) = (X[p[1]], X[p[2]], X[p[3]])

function partC()
    setprecision(BigFloat, 320)
    shp = shapes(BigFloat)
    tee("c_crosschecks.txt") do io
        println(io, "(c) cross-checks, BigFloat 320 bits, m = -1..12")
        println(io)
        println(io, "1. pairMoments vs the plain explicit closed forms")
        exf = Dict{String,Any}(
            "S" => (m, a, b, c) -> momSlf(m, a, b),
            "Ef" => (m, a, b, c) -> momFlt(m, a, b, zero(a)),
            "Vc" => (m, a, b, c) -> momVtxCop(m, a, b),
            "P1" => (m, a, b, c) -> momPar(m, a, b, c),
            "P2" => (m, a, b, c) -> momPar(m, a, b, 2c),
            "P1s" => (m, a, b, c) -> momFlt(m, a, b, c),
            "P2s" => (m, a, b, c) -> momFlt(m, a, b, 2c))
        alt = Dict{String,Any}("Vc" => (m, a, b, c) -> vcMom(m, a, b),
                               "Vp" => (m, a, b, c) -> vpMom(m, a, b, c),
                               "Xg" => (m, a, b, c) -> xgMom(m, a, b, c))
        prp = ("Ec", "Vp", "Xg", "Xg-b", "Xg-c", "Xg-d")
        for cod in SUBCODES
            wf = 0.0; wp = 0.0; wa = 0.0; at = ""
            for (snm, (a, b, c)) in shp
                A, B = canonPanels(cod, a, b, c)
                v = pairMoments(A, B, 12)
                for m in -1:12
                    rel(x) = Float64(abs(v[m + 2] - x) / abs(v[m + 2]))
                    if haskey(exf, cod)
                        e = rel(exf[cod](m, a, b, c))
                        e > wf && (wf = e; at = "$snm m=$m")
                    end
                    cod in prp && (wp = max(wp, rel(prpMom(m, A, B))))
                    haskey(alt, cod) && (wa = max(wa, rel(alt[cod](m, a, b, c))))
                end
            end
            nms = String[]
            haskey(exf, cod) && push!(nms, "polar " * fm(wf))
            cod in prp && push!(nms, "prpMom " * fm(wp))
            haskey(alt, cod) && push!(nms, "direct " * fm(wa))
            isempty(nms) && push!(nms, "no explicit form (2D, both in-plane axes shifted)")
            println(io, "  ", rpad(cod, 8), join(nms, "   "), at == "" ? "" : "   worst at " * at)
        end
        println(io)
        println(io, "2. homogeneity I_m(lam s) = lam^(m+4) I_m(s), lam = 3 and 1/7")
        wh = 0.0; at = ""
        for cod in SUBCODES, (snm, (a, b, c)) in shp, lam in (BigFloat(3), BigFloat(1) / 7)
            v = pairMoments(canonPanels(cod, a, b, c)..., 12)
            w = pairMoments(canonPanels(cod, lam * a, lam * b, lam * c)..., 12)
            for m in -1:12
                e = Float64(abs(w[m + 2] - lam^(m + 4) * v[m + 2]) / abs(w[m + 2]))
                e > wh && (wh = e; at = "$cod $snm lam=$(Float64(lam)) m=$m")
            end
        end
        println(io, "  worst ", wh, "  at ", at)
        println(io)
        println(io, "3. symmetries: max rel difference under each length swap")
        println(io, "   (a swap is a symmetry when the entry is at the reference floor;")
        println(io, "    n/a = S, Ef, Vc have no c, so a swap involving it tests nothing)")
        print(io, rpad("code", 8), rpad("a<->b", 14), rpad("a<->c", 14), "b<->c")
        println(io)
        a0 = BigFloat(7) / 3; b0 = BigFloat(5) / 11; c0 = BigFloat(13) / 6
        for cod in SUBCODES
            v = pairMoments(canonPanels(cod, a0, b0, c0)..., 12)
            print(io, rpad(cod, 8))
            for sw in ((b0, a0, c0), (c0, b0, a0), (a0, c0, b0))
                if noC(cod) && sw != (b0, a0, c0)
                    print(io, rpad("n/a", 14)); continue
                end
                w = pairMoments(canonPanels(cod, sw...)..., 12)
                e = maximum(Float64(abs(w[k] - v[k]) / abs(v[k])) for k in eachindex(v))
                print(io, rpad(fm(e), 14))
            end
            println(io)
        end
        println(io)
        println(io, "4. orientation invariance: all 6 axis permutations of both panels")
        wo = 0.0; at = ""
        for cod in SUBCODES
            A, B = canonPanels(cod, a0, b0, c0)
            v = pairMoments(A, B, 12)
            for p in ((1,2,3),(1,3,2),(2,1,3),(2,3,1),(3,1,2),(3,2,1))
                w = pairMoments(prm3(p, A), prm3(p, B), 12)
                for k in eachindex(v)
                    e = Float64(abs(w[k] - v[k]) / abs(v[k]))
                    e > wo && (wo = e; at = "$cod perm=$p m=$(k-2)")
                end
            end
        end
        println(io, "  worst ", wo, "  at ", at)
        println(io)
        println(io, "5. momentSeries vs pairKer (direct quadrature of the kernel)")
        A, B = canonPanels("Ef", BigFloat(1) / 32, BigFloat(1) / 32, BigFloat(1) / 32)
        for f in (BigFloat(1), BigFloat(4))
            v, rel = momentSeries(A, B, f, 12)
            r = pairKer(A, B, f)
            println(io, "  Ef at 1/32 cell, f = ", Float64(f), "  rel err ",
                    fm(Float64(abs(v - r) / abs(r))), "   last-term/sum ", fm(Float64(rel)))
        end
        println(io)
        println(io, "6. tol keyword on Float64 panels whose breakpoints split by an ulp")
        println(io, "   (s = 0.1, 0.3, 0.7 is inexact, so 3s/2 - s/2 != s: the exact route")
        println(io, "    emits ulp-wide sliver boxes, tol merges the breakpoints instead)")
        s6 = (0.1, 0.3, 0.7)
        sb = ntuple(d -> BigFloat(s6[d]), 3)
        nb = 0; nt = 0; nsl = 0; we = 0.0; wt = 0.0
        for D in ((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)), F in 1:6, Fp in 1:6
            A, B = facePair(D, F, Fp, s6)
            bx = pairBxs(A, B, Float64)[1]
            nb += length(bx); nt += length(pairBxsTol(A, B)[1])
            for (lo, hi, cw, dw) in bx
                length(lo) == 3 && slvAxs(lo, hi)[1] != 0 && (nsl += 1)
            end
            rf = pairMoments(facePair(D, F, Fp, sb)..., 12, BigFloat)
            v1 = pairMoments(A, B, 12)
            v2 = pairMoments(A, B, 12; tol = 8 * eps(Float64))
            for m in -1:12
                we = max(we, Float64(abs(BigFloat(v1[m + 2]) - rf[m + 2]) / abs(rf[m + 2])))
                wt = max(wt, Float64(abs(BigFloat(v2[m + 2]) - rf[m + 2]) / abs(rf[m + 2])))
            end
        end
        println(io, "  144 face pairs: boxes ", nb, " exact / ", nt, " with tol,  ",
                nsl, " of the exact ones thin-gated")
        println(io, "  worst rel error vs the BigFloat panels, m = -1..12: exact ",
                fm(we), "   tol = 8 eps ", fm(wt))
    end
end

# ===== (d) digits lost in Float64
digLos(x64, xbg) = (r = abs(BigFloat(x64) - xbg) / (eps(Float64) * abs(xbg));
                    r <= 1 ? 0.0 : Float64(log10(r)))

const EXPS = -6:6
is2D(cod) = cod in ("S", "Ef", "Vc", "P1", "P2", "P1s", "P2s", "P1s-b", "P2s-b")
# codes whose canonical panels do not use c, so their c columns repeat
noC(cod) = cod in ("S", "Ef", "Vc")
# the grid point the label names: 10.0^-2 is 0.010000000000000002, not 1e-2
pw10(e::Int) = e >= 0 ? 10.0^e : 1 / 10.0^(-e)

function partD()
    setprecision(BigFloat, 320)
    tee("d_digits.txt") do io
        println(io, "(d) digits lost in Float64: log10(|x64 - xbig|/(eps |xbig|)), xbig the")
        println(io, "    320-bit evaluation of the same formulas at BigFloat(Float64 inputs).")
        println(io, "    Grid b = 1, a and c in 1e-6..1e6; c = 0 as well for the 2D codes.")
        println(io, "    Cells are the exact decimals the labels name (1e-2, not 10.0^-2).")
        println(io, "    S, Ef and Vc have no c in their canonical panels, so their c columns")
        println(io, "    repeat: 13 distinct geometries per grid, not 13 x 14.")
        println(io, "    Bar: <= 1.0 everywhere.")
        println(io)
        grd = Dict{String,Matrix{Float64}}()
        wm = Dict{Tuple{String,Int},Tuple{Float64,String}}()
        wc = Dict{String,Tuple{Float64,String}}()
        ovr = String[]
        for cod in SUBCODES
            cs = is2D(cod) ? vcat(collect(EXPS), 99) : collect(EXPS)
            grd[cod] = zeros(Float64, length(EXPS), length(cs))
            for (ia, ea) in enumerate(EXPS), (ic, ec) in enumerate(cs)
                a6 = pw10(ea); c6 = ec == 99 ? 0.0 : pw10(ec)
                loc = "(a=1e$ea,c=" * (ec == 99 ? "0)" : "1e$ec)")
                v6 = pairMoments(canonPanels(cod, a6, 1.0, c6)..., 12)
                vb = pairMoments(canonPanels(cod, BigFloat(a6), BigFloat(1),
                                             BigFloat(c6))..., 12, BigFloat)
                for m in -1:12
                    d = digLos(v6[m + 2], vb[m + 2])
                    d > get(wm, (cod, m), (0.0, ""))[1] && (wm[(cod, m)] = (d, loc))
                    d > get(wc, cod, (0.0, ""))[1] && (wc[cod] = (d, "$loc m=$m"))
                    grd[cod][ia, ic] = max(grd[cod][ia, ic], d)
                    d > 1.0 && push!(ovr, "  $cod m=$m $loc  $(round(d, digits = 2))")
                end
            end
        end
        println(io, "worst digits lost per sub-code and m")
        print(io, rpad("code", 8)); for m in -1:12; print(io, lpad("m=$m", 7)); end; println(io)
        for cod in SUBCODES
            print(io, rpad(cod, 8))
            for m in -1:12; print(io, lpad(round(get(wm, (cod, m), (0.0, "-"))[1], digits = 2), 7)); end
            println(io)
        end
        println(io)
        println(io, "worst per sub-code, with the cell")
        for cod in SUBCODES
            v = get(wc, cod, (0.0, "-"))
            println(io, "  ", rpad(cod, 8), round(v[1], digits = 2), "   at ", v[2])
        end
        println(io)
        println(io, "worst over m, rows a/b = 1e-6..1e6, cols c/b = 1e-6..1e6 (last col c = 0)")
        for cod in SUBCODES
            cs = is2D(cod) ? vcat(collect(EXPS), 99) : collect(EXPS)
            println(io, "-- ", cod, noC(cod) ? "   (c unused: every column repeats)" : "")
            print(io, rpad("a\\c", 7))
            for e in cs; print(io, lpad(e == 99 ? "0" : "1e$e", 6)); end
            println(io)
            for (ia, ea) in enumerate(EXPS)
                print(io, rpad("1e$ea", 7))
                for ic in eachindex(cs); print(io, lpad(round(grd[cod][ia, ic], digits = 1), 6)); end
                println(io)
            end
        end
        println(io)
        gl = maximum(x[1] for x in values(wc))
        println(io, "GLOBAL worst ", round(gl, digits = 2), " digits")
        println(io, "cells above 1.0: ", length(ovr))
        for l in ovr; println(io, l); end
    end
end

# ===== (e) Float64 timing, one face pair at mMax = 12
function partE()
    tee("e_timing.txt") do io
        println(io, "(e) Float64 cost of pairMoments(A, B, 12) for one face pair, per sub-code")
        println(io, "    shapes: cube (1,1,1) and the slender (1/32,1/32,1/512)")
        println(io)
        println(io, rpad("code", 8), rpad("cube us", 12), rpad("slender us", 12), "boxes")
        for cod in SUBCODES
            ts = Float64[]
            for (a, b, c) in ((1.0, 1.0, 1.0), (1 / 32, 1 / 32, 1 / 512))
                A, B = canonPanels(cod, a, b, c)
                pairMoments(A, B, 12)
                t = @elapsed for _ in 1:200; pairMoments(A, B, 12); end
                push!(ts, t / 200 * 1e6)
            end
            nb = length(pairBxs(canonPanels(cod, 1.0, 1.0, 1.0)..., Float64)[1])
            println(io, rpad(cod, 8), rpad(round(ts[1], digits = 1), 12),
                    rpad(round(ts[2], digits = 1), 12), nb)
        end
    end
end

# pairMom / pairKer, needed by parts (b) and (c)
PART in ("b", "c", "all") &&
    include("/Users/pvirally/.julia/dev/GilaElectromagnetics/notes/gen/ref/mom.jl")

PART in ("a", "all") && partA()
PART in ("b", "all") && partB()
PART in ("c", "all") && partC()
PART in ("d", "all") && partD()
PART in ("e", "all") && partE()
