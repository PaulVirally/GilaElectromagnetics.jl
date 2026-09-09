# Comparison of the moment series of moments.jl with GilaElectromagnetics' own
# numerics. Runnable in parts, since (a) runs Gila's contact integrals:
#   JULIA_NUM_THREADS=1 julia --startup-file=no --project=<env> gilacmp.jl a
#   ... b   ... c
#   (a) wekS/wekE/wekV at intOrd 48 and 64 vs the series, 18 entries, f = 1 and
#       1 + 0.1i, plus the confirming m = -1 check against Gila's static parts
#   (b) the order-9 fixed rule egoSrfFxd! on every non-touching face pair of the
#       four touching-cell offsets, f = 1
#   (c) the m = -1 term scales as 1/f^2 and term n as f^(n - 2)
# Each part writes tables/f_gila_<part>.txt as well as stdout.
#
# <env> is any environment in which `using GilaElectromagnetics` works:
#   julia> using Pkg
#   julia> Pkg.activate("gilaenv")
#   julia> Pkg.develop(path = "/Users/pvirally/.julia/dev/GilaElectromagnetics")
#
# Moments come from moments.jl and nothing else. notes/gen/ref/mom.jl is loaded
# by (b) only, for an independent pairKer cross-check on one pair per sub-code;
# (b) runs without it if the file is absent.
#
# Cost, single threaded, measured: (a) 37 min, 36 of them inside Gila's
# wekS/wekE/wekV; (b) 35 min, nearly all of it in the 280 BigFloat series;
# (c) 5 s. Cell scale (1//4)^3 is run at intOrd 48 only in (a) to keep (a)
# under 40 min; the other three scales carry both orders.
using Printf
using GilaElectromagnetics
const GV = GilaElectromagnetics.GilaVacuum
include(joinpath(@__DIR__, "moments.jl"))

const DIR = joinpath(@__DIR__, "tables")
const PART = length(ARGS) == 0 ? "all" : ARGS[1]
const MOMREF = "/Users/pvirally/.julia/dev/GilaElectromagnetics/notes/gen/ref/mom.jl"
const TOL = 1e-20                     # relative size of the last series term
const Q = Rational{BigInt}
const HASREF = isfile(MOMREF)
PART in ("b", "all") && HASREF && include(MOMREF)   # pairKer only

# stdout and tables/<nam> at once, streamed: part (a) runs for an hour
struct Duo <: IO
    a::IO
    b::IO
end
Base.write(d::Duo, x::UInt8) = (write(d.a, x); write(d.b, x))
Base.flush(d::Duo) = (flush(d.a); flush(d.b))
tee(f, nam) = open(io -> f(Duo(stdout, io)), joinpath(DIR, nam), "w")

rel(a, b) = Float64(abs(a - b) / abs(b))

# smallest n with (2 pi |f| rMax)^n / n! < tol, as an even mMax = n - 1
function serGss(pA, pB, f, tol)
    r2 = sum(max(abs(Float64(pA[d][1] - pB[d][2])),
                 abs(Float64(pA[d][2] - pB[d][1])))^2 for d in 1:3)
    x = 2 * pi * abs(f) * sqrt(r2)
    lg = 0.0
    n = 0
    while n < 150
        n += 1
        lg += log(x) - log(n)
        lg < log(tol) && break
    end
    return max(4, 2 * cld(n - 1, 2))
end

# smallest even mMax whose last series term is below tol relative; the series
# then has mMax + 2 terms, n = 0 .. mMax + 1
function serCnv(pA, pB, f, tol = TOL)
    mM = serGss(pA, pB, f, tol)
    v, r = momentSeries(pA, pB, f, mM)
    if r < tol
        while mM > 4
            v2, r2 = momentSeries(pA, pB, f, mM - 2)
            r2 < tol || break
            mM -= 2
            v, r = v2, r2
        end
    else
        while r >= tol && mM < 96
            mM += 2
            v, r = momentSeries(pA, pB, f, mM)
        end
    end
    r < tol || error("serCnv: no convergence")
    return (mM, v, Float64(r))
end

# term n of the series, (2 pi i f)^n I_{n-1} / (n! 4 pi f^2)
function serTrms(mom, f)
    T = eltype(mom)
    F = Complex{T}
    z = 2 * T(pi) * im * F(f)
    trm = Vector{F}(undef, length(mom))
    cof = one(F)
    for n in 0:(length(mom) - 1)
        trm[n + 1] = cof * mom[n + 1] / (4 * T(pi) * F(f)^2)
        cof *= z / (n + 1)
    end
    return trm
end

# ===== (a) the 18 weak entries
# wekGrdPts! local frame: dir picks which cell edge plays gX, gY, gZ
function wekFrm(dir::Integer, scl)
    dir == 1 && return (scl[1], scl[2], scl[3])
    dir == 2 && return (scl[3], scl[1], scl[2])
    dir == 3 && return (scl[2], scl[3], scl[1])
    error("wekFrm: bad dir")
end

# the eight panel pairs of wekSInt/wekEInt/wekVInt; panel A is grid points
# 1, 2, 5, 4 of wekGrdPts!, i.e. [0,gX] x [0,gY] x {0}
function wekPnl(nam::Symbol, gX::T, gY::T, gZ::T) where {T}
    z = zero(T)
    A = ((z, gX), (z, gY), (z, z))
    nam === :slf     && return (A, A)
    nam === :edgFltX && return (A, ((gX, 2gX), (z, gY), (z, z)))
    nam === :edgFltY && return (A, ((z, gX), (gY, 2gY), (z, z)))
    nam === :edgCrnX && return (A, ((gX, gX), (z, gY), (z, gZ)))
    nam === :edgCrnY && return (A, ((z, gX), (gY, gY), (z, gZ)))
    nam === :vtxCop  && return (A, ((gX, 2gX), (gY, 2gY), (z, z)))
    nam === :vtxPrpX && return (A, ((gX, gX), (gY, 2gY), (z, gZ)))
    nam === :vtxPrpY && return (A, ((gX, 2gX), (gY, gY), (z, gZ)))
    error("wekPnl: bad name")
end

# the same pair as a canonPanels sub-code with permuted edge lengths
function canonArg(nam::Symbol, gX::T, gY::T, gZ::T) where {T}
    nam === :slf     && return ("S", gX, gY, gZ)
    nam === :edgFltX && return ("Ef", gX, gY, gZ)
    nam === :edgFltY && return ("Ef", gY, gX, gZ)
    nam === :edgCrnX && return ("Ec", gY, gX, gZ)
    nam === :edgCrnY && return ("Ec", gX, gY, gZ)
    nam === :vtxCop  && return ("Vc", gX, gY, gZ)
    nam === :vtxPrpX && return ("Vp", gX, gY, gZ)
    nam === :vtxPrpY && return ("Vp", gY, gX, gZ)
    error("canonArg: bad name")
end

# entry -> the (panel pair, dir) it integrates; two of them mean Gila averages
# the two orientations, (xyA + xyB) / 2
const MAP = Dict(
    ("S", 1) => [(:slf, 3)], ("S", 2) => [(:slf, 2)], ("S", 3) => [(:slf, 1)],
    ("E", 1) => [(:edgFltX, 3)], ("E", 2) => [(:edgFltY, 3)],
    ("E", 3) => [(:edgFltY, 2)], ("E", 4) => [(:edgFltX, 2)],
    ("E", 5) => [(:edgFltX, 1)], ("E", 6) => [(:edgFltY, 1)],
    ("E", 7) => [(:edgCrnX, 3), (:edgCrnY, 2)],
    ("E", 8) => [(:edgCrnY, 3), (:edgCrnX, 1)],
    ("E", 9) => [(:edgCrnX, 2), (:edgCrnY, 1)],
    ("V", 1) => [(:vtxCop, 3)], ("V", 2) => [(:vtxCop, 2)], ("V", 3) => [(:vtxCop, 1)],
    ("V", 4) => [(:vtxPrpX, 3), (:vtxPrpY, 2)],
    ("V", 5) => [(:vtxPrpY, 3), (:vtxPrpX, 1)],
    ("V", 6) => [(:vtxPrpX, 2), (:vtxPrpY, 1)])
const LBL = Dict(("S", 1) => "xx", ("S", 2) => "yy", ("S", 3) => "zz",
    ("E", 1) => "xxY", ("E", 2) => "xxZ", ("E", 3) => "yyX", ("E", 4) => "yyZ",
    ("E", 5) => "zzX", ("E", 6) => "zzY", ("E", 7) => "xy", ("E", 8) => "xz",
    ("E", 9) => "yz", ("V", 1) => "xx", ("V", 2) => "yy", ("V", 3) => "zz",
    ("V", 4) => "xy", ("V", 5) => "xz", ("V", 6) => "yz")
const ENTS = vcat([("S", i) for i in 1:3], [("E", i) for i in 1:9],
                  [("V", i) for i in 1:6])

const SCLS = [("a", (1//32, 1//32, 1//32)), ("b", (1//8, 1//8, 1//8)),
              ("c", (1//32, 1//32, 1//512)), ("d", (1//4, 1//4, 1//4))]
const ORDS = Dict("a" => [48, 64], "b" => [48, 64], "c" => [48, 64], "d" => [48])
const FRQS = [("1", 1.0 + 0.0im), ("1+0.1i", 1.0 + 0.1im)]

entKey(k) = k[1] * string(k[2])
entCod(k) = canonArg(MAP[k][1][1], 1, 1, 1)[1]
oriStr(nm, d) = string(nm) * " d" * string(d)
mapStr(k) = join([oriStr(nm, d) for (nm, d) in MAP[k]], " + ")

# Gila's raw weak integrals for one (scale, intOrd, frequency), with timings
function wekRun(scl, ord, frq)
    opt = GV.CPUKerOpt(frq, ord, false, GV.CPU())
    qud = GV.gauQud(ord)
    let o = GV.CPUKerOpt(frq, 4, false, GV.CPU()), q = GV.gauQud(4)
        GV.wekS(scl, q, o); GV.wekE(scl, q, o); GV.wekV(scl, q, o)
    end
    tS = @elapsed wS = GV.wekS(scl, qud, opt)
    tE = @elapsed wE = GV.wekE(scl, qud, opt)
    tV = @elapsed wV = GV.wekV(scl, qud, opt)
    val = Dict{Tuple{String,Int},ComplexF64}()
    for (nm, v) in (("S", wS), ("E", wE), ("V", wV)), i in eachindex(v)
        val[(nm, i)] = v[i]
    end
    return val, (tS, tE, tV)
end

# entry -> (name of Gila's closed-form static part, its value = I_{-1}/(4 pi f^2))
function rSrfEnt(scl, frq)
    opt = GV.CPUKerOpt(frq, 8, false, GV.CPU())
    s1, s2, s3 = Float64.(scl)
    return Dict(
        ("S", 1) => ("rSrfSlf(s2,s3)", GV.rSrfSlf(s2, s3, opt)),
        ("S", 2) => ("rSrfSlf(s1,s3)", GV.rSrfSlf(s1, s3, opt)),
        ("S", 3) => ("rSrfSlf(s1,s2)", GV.rSrfSlf(s1, s2, opt)),
        ("E", 1) => ("rSrfEdgFlt(s3,s2)", GV.rSrfEdgFlt(s3, s2, opt)),
        ("E", 2) => ("rSrfEdgFlt(s2,s3)", GV.rSrfEdgFlt(s2, s3, opt)),
        ("E", 3) => ("rSrfEdgFlt(s3,s1)", GV.rSrfEdgFlt(s3, s1, opt)),
        ("E", 4) => ("rSrfEdgFlt(s1,s3)", GV.rSrfEdgFlt(s1, s3, opt)),
        ("E", 5) => ("rSrfEdgFlt(s2,s1)", GV.rSrfEdgFlt(s2, s1, opt)),
        ("E", 6) => ("rSrfEdgFlt(s1,s2)", GV.rSrfEdgFlt(s1, s2, opt)),
        ("E", 7) => ("rSrfEdgCrn(s3,s2,s1)", GV.rSrfEdgCrn(s3, s2, s1, opt)),
        ("E", 8) => ("rSrfEdgCrn(s2,s3,s1)", GV.rSrfEdgCrn(s2, s3, s1, opt)),
        ("E", 9) => ("rSrfEdgCrn(s1,s3,s2)", GV.rSrfEdgCrn(s1, s3, s2, opt)))
end

function partA()
    setprecision(BigFloat, 256)
    tee("f_gila_a.txt") do io
        println(io, "(a) Gila's weak face-pair integrals vs the moment series")
        println(io, "    wekS/wekE/wekV, intOrd 48 and 64, f = 1 and 1 + 0.1i")
        println(io, "    series: momentSeries, BigFloat 256 bits, mMax raised until the last term")
        println(io, "    is below ", TOL, " relative; nTrm = mMax + 2 terms, n = 0 .. mMax + 1")
        println(io)
        println(io, "## the 18 entries, their panel pair and its canonPanels sub-code")
        println(io, "   panels in the wekGrdPts! frame (gX,gY,gZ) = wekFrm(dir, scl), A = grid")
        println(io, "   points 1,2,5,4 = [0,gX] x [0,gY] x {0}; two pairs mean Gila averages them")
        println(io)
        println(io, rpad("entry", 6), rpad("lbl", 5), rpad("code", 5), rpad("pair, dir", 28),
                rpad("canonPanels", 22), "panels of the first orientation, scl = (1//8)^3")
        for k in ENTS
            nm, d = MAP[k][1]
            g = wekFrm(d, (Q(1)//8, Q(1)//8, Q(1)//8))
            A, B = wekPnl(nm, g...)
            cod, ca, cb, cc = canonArg(nm, "gX", "gY", "gZ")
            println(io, rpad(entKey(k), 6), rpad(LBL[k], 5), rpad(entCod(k), 5),
                    rpad(mapStr(k), 28), rpad(cod * "(" * ca * "," * cb * "," * cc * ")", 22),
                    "A=", A, " B=", B)
        end
        println(io)
        println(io, "## wekPnl against canonPanels, and the two averaged orientations against")
        println(io, "   each other: max relative difference of pairMoments over m = -1 .. 12")
        println(io)
        println(io, rpad("scl", 22), rpad("entry", 6), rpad("lbl", 5), rpad("code", 5),
                rpad("orientation A", 14), rpad("orientation B", 14),
                rpad("wekPnl-vs-canon", 17), "A-vs-B")
        for (ky, scl) in SCLS
            sq = Q.(scl)
            for k in ENTS
                lst = MAP[k]
                mms = Vector{Vector{BigFloat}}()
                cnv = 0.0
                for (nm, d) in lst
                    g = wekFrm(d, sq)
                    push!(mms, pairMoments(wekPnl(nm, g...)..., 12))
                    cn = pairMoments(canonPanels(canonArg(nm, g...)...)..., 12)
                    cnv = max(cnv, maximum(rel.(mms[end], cn)))
                end
                ab = length(mms) == 1 ? "--" : @sprintf("%.3e", maximum(rel.(mms[1], mms[2])))
                println(io, rpad(string(scl), 22), rpad(entKey(k), 6), rpad(LBL[k], 5),
                        rpad(entCod(k), 5), rpad(oriStr(lst[1]...), 14),
                        rpad(length(lst) == 1 ? "--" : oriStr(lst[2]...), 14),
                        rpad(@sprintf("%.3e", cnv), 17), ab)
            end
        end
        println(io)
        println(io, "## m = -1 check: pairMoments I_{-1} against Gila's static part x 4 pi f^2, f = 1")
        println(io)
        println(io, rpad("scl", 22), rpad("entry", 6), rpad("lbl", 5), rpad("code", 5),
                rpad("Gila static part", 22), "rel diff")
        for (ky, scl) in SCLS
            sq = Q.(scl)
            gil = rSrfEnt(scl, 1.0 + 0.0im)
            for k in ENTS
                haskey(gil, k) || continue
                nm, d = MAP[k][1]
                g = wekFrm(d, sq)
                im1 = pairMoments(wekPnl(nm, g...)..., -1)[1]
                nam, v = gil[k]
                println(io, rpad(string(scl), 22), rpad(entKey(k), 6), rpad(LBL[k], 5),
                        rpad(entCod(k), 5), rpad(nam, 22),
                        @sprintf("%.3e", rel(im1, BigFloat(real(v)) * 4 * BigFloat(pi))))
            end
        end
        println(io)
        println(io, "## Gila run timings, single threaded, seconds")
        println(io)
        println(io, rpad("scl", 22), rpad("intOrd", 8), rpad("f", 8), rpad("wekS", 10),
                rpad("wekE", 10), rpad("wekV", 10), "total")
        gil = Dict{Tuple{String,Int,String},Dict{Tuple{String,Int},ComplexF64}}()
        ttl = 0.0
        for (ky, scl) in SCLS, ord in ORDS[ky], (fk, f) in FRQS
            println(stderr, "gila ", ky, " ", ord, " ", fk)
            v, t = wekRun(scl, ord, f)
            gil[(ky, ord, fk)] = v
            ttl += sum(t)
            println(io, rpad(string(scl), 22), rpad(ord, 8), rpad(fk, 8),
                    rpad(@sprintf("%.2f", t[1]), 10), rpad(@sprintf("%.2f", t[2]), 10),
                    rpad(@sprintf("%.2f", t[3]), 10), @sprintf("%.2f", sum(t)))
            flush(io)
        end
        println(io, "total ", @sprintf("%.1f", ttl), " s")
        println(io)
        println(io, "## series vs Gila")
        println(io, "   dRe/dIm are signed deviations of Gila from the series in units of |series|;")
        println(io, "   cnv = |d64| < |d48| in both components, brk = d48 and d64 straddle the")
        println(io, "   series in the real part, PLATEAU = order 64 no closer than order 48;")
        println(io, "   f64ser = Float64 series (Float64 moments) against the BigFloat series")
        cnt = Dict("cnv" => 0, "brk" => 0, "plt" => 0, "row" => 0)
        for (ky, scl) in SCLS
            sq = Q.(scl)
            sf = Float64.(scl)
            for (fk, f) in FRQS
                println(stderr, "series ", ky, " ", fk)
                o1 = ORDS[ky][1]
                hs = length(ORDS[ky]) > 1
                o2 = hs ? ORDS[ky][2] : o1
                l1 = string(o1)
                l2 = string(o2)
                println(io)
                println(io, "### scl = ", scl, "  f = ", f, "  intOrd ", ORDS[ky])
                println(io, rpad("entry", 6), rpad("lbl", 5), rpad("code", 5), rpad("nTrm", 6),
                        rpad("series Re", 34), rpad("series Im", 34),
                        rpad("gila" * l1 * " Re", 25), rpad("gila" * l1 * " Im", 25),
                        rpad(hs ? "gila" * l2 * " Re" : "--", 25),
                        rpad(hs ? "gila" * l2 * " Im" : "--", 25),
                        rpad("rel" * l1, 11), rpad(hs ? "rel" * l2 : "--", 11),
                        rpad("dRe" * l1, 11), rpad(hs ? "dRe" * l2 : "--", 11),
                        rpad("dIm" * l1, 11), rpad(hs ? "dIm" * l2 : "--", 11),
                        rpad("verdict", 9), "f64ser")
                for k in ENTS
                    lst = MAP[k]
                    ser = zero(Complex{BigFloat})
                    serF = zero(ComplexF64)
                    nt = 0
                    for (nm, d) in lst
                        g = wekFrm(d, sq)
                        mM, v, _ = serCnv(wekPnl(nm, g...)..., f)
                        ser += v / length(lst)
                        nt = max(nt, mM + 2)
                        serF += momentSeries(wekPnl(nm, wekFrm(d, sf)...)..., f, mM)[1] / length(lst)
                    end
                    g1 = gil[(ky, o1, fk)][k]
                    g2 = gil[(ky, o2, fk)][k]
                    v48 = Complex{BigFloat}(g1)
                    v64 = Complex{BigFloat}(g2)
                    a = abs(ser)
                    dR48 = Float64((real(v48) - real(ser)) / a)
                    dI48 = Float64((imag(v48) - imag(ser)) / a)
                    dR64 = Float64((real(v64) - real(ser)) / a)
                    dI64 = Float64((imag(v64) - imag(ser)) / a)
                    brk = sign(dR48) != sign(dR64)
                    cnv = abs(dR64) < abs(dR48) && abs(dI64) < abs(dI48)
                    vd = !hs ? "--" : brk ? "brk" : cnv ? "cnv" : "PLATEAU"
                    if hs
                        cnt["row"] += 1
                        brk && (cnt["brk"] += 1)
                        cnv ? (cnt["cnv"] += 1) : (cnt["plt"] += 1)
                    end
                    println(io, rpad(entKey(k), 6), rpad(LBL[k], 5), rpad(entCod(k), 5),
                            rpad(nt, 6),
                            rpad(@sprintf("%+.26e", real(ser)), 34),
                            rpad(@sprintf("%+.26e", imag(ser)), 34),
                            rpad(@sprintf("%+.16e", real(g1)), 25),
                            rpad(@sprintf("%+.16e", imag(g1)), 25),
                            rpad(hs ? @sprintf("%+.16e", real(g2)) : "--", 25),
                            rpad(hs ? @sprintf("%+.16e", imag(g2)) : "--", 25),
                            rpad(@sprintf("%.3e", rel(v48, ser)), 11),
                            rpad(hs ? @sprintf("%.3e", rel(v64, ser)) : "--", 11),
                            rpad(@sprintf("%+.3e", dR48), 11),
                            rpad(hs ? @sprintf("%+.3e", dR64) : "--", 11),
                            rpad(@sprintf("%+.3e", dI48), 11),
                            rpad(hs ? @sprintf("%+.3e", dI64) : "--", 11),
                            rpad(vd, 9),
                            @sprintf("%.3e", rel(Complex{BigFloat}(serF), ser)))
                end
                flush(io)
            end
        end
        println(io)
        println(io, "## bracketing / convergence summary over the rows that carry both orders")
        println(io, "rows ", cnt["row"], "  converging ", cnt["cnv"], "  straddling ",
                cnt["brk"], "  not converging ", cnt["plt"])
    end
end

# ===== (b) the non-touching face pairs of the touching shell
const SHELL = ["P1", "P2", "P1s", "P1s-b", "P2s", "P2s-b", "Xg", "Xg-b", "Xg-c", "Xg-d"]
const OFFS = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)]

# every (offset, target face, source face) whose sub-code is not a contact code
function shellPairs()
    out = Tuple{NTuple{3,Int},Int,Int,String}[]
    for D in OFFS, F in 1:6, Fp in 1:6
        cod = geomCode(D, F, Fp)
        cod in SHELL && push!(out, (D, F, Fp, cod))
    end
    return out
end

# Gila's order-9 fixed rule; the raw face-pair integral is srfMat[fp] * V_t
function ord9(D, F, Fp, scl, frq)
    s = Float64.(scl)
    opt = GV.CPUKerOpt(frq, 48, false, GV.CPU())
    fac = Float64.(GV.cubFac(scl))
    srf = zeros(ComplexF64, 36)
    fp = (F - 1) * 6 + Fp
    GV.egoSrfFxd!(D[1] * s[1], D[2] * s[2], D[3] * s[3], srf, fac, fac, [fp],
                  GV.facPar(), Float64.(GV.srfScl(s, s)), opt, GV.cntOrd)
    return srf[fp] * prod(s), fp
end

function partB()
    setprecision(BigFloat, 256)
    prs = shellPairs()
    frq = 1.0 + 0.0im
    rep = Dict(cod => first(p for p in prs if p[4] == cod) for cod in SHELL)
    tee("f_gila_b.txt") do io
        println(io, "(b) Gila's order-9 fixed rule vs the moment series, f = 1")
        println(io, "    egoSrfFxd!(..., pairListUn, ..., cntOrd = ", GV.cntOrd, "); the raw")
        println(io, "    face-pair integral is srfMat[fp] * V_t with V_t = s1 s2 s3")
        println(io, "    series: momentSeries, BigFloat 256 bits, last term below ", TOL, " relative")
        println(io)
        println(io, "## the non-touching set, fp = 6 (targetFace - 1) + sourceFace")
        for D in OFFS
            lst = sort([(F - 1) * 6 + Fp for (Dd, F, Fp, c) in prs if Dd == D])
            println(io, rpad(string(D), 12), "n = ", rpad(length(lst), 5), "fp = ", lst)
        end
        println(io, "total ", length(prs), " pairs over the four offsets")
        println(io)
        println(io, "## sub-code census")
        for cod in SHELL
            println(io, rpad(cod, 8), count(p -> p[4] == cod, prs))
        end
        println(io)
        println(io, "## the series is truncated at ", TOL, " on its last term, so its own residual")
        println(io, "   is a few 1e-23; ser60 is the same series carried to a last term of 1e-60,")
        println(io, "   which is what the pairKer column compares against. pairKer")
        println(io, "   (notes/gen/ref/mom.jl) runs on one representative per sub-code:")
        println(io, "   ", join([string(rep[c][1]) * " " * FACENAMES[rep[c][2]] * "-" *
                                 FACENAMES[rep[c][3]] * " " * c for c in SHELL], ", "))
        HASREF || println(io, "   mom.jl not found, the pairKer column is empty")
        println(io)
        println(io, "## results")
        println(io, rpad("scl", 22), rpad("D", 12), rpad("fp", 4), rpad("F-Fp", 10),
                rpad("code", 7), rpad("nTrm", 6), rpad("ord9 Re", 25), rpad("ord9 Im", 25),
                rpad("series Re", 34), rpad("series Im", 34), rpad("ord9-vs-ser", 12),
                rpad("ser-vs-ser60", 13), "ser60-vs-pairKer")
        wst = Dict{Tuple{String,String},Float64}()
        bad = String[]
        for (ky, scl) in SCLS
            sq = Q.(scl)
            println(stderr, "partB ", ky)
            for (D, F, Fp, cod) in prs
                A, B = facePair(D, F, Fp, sq)
                mM, ser, _ = serCnv(A, B, frq)
                g9, fp = ord9(D, F, Fp, scl, frq)
                r9 = rel(Complex{BigFloat}(g9), ser)
                pk = "--"
                tr = "--"
                if HASREF && rep[cod] == (D, F, Fp, cod)
                    s60 = serCnv(A, B, frq, 1e-60)[2]
                    kv = pairKer(ntuple(i -> (BigFloat(A[i][1]), BigFloat(A[i][2])), 3),
                                 ntuple(i -> (BigFloat(B[i][1]), BigFloat(B[i][2])), 3),
                                 Complex{BigFloat}(frq))
                    tr = @sprintf("%.3e", rel(ser, s60))
                    pk = @sprintf("%.3e", rel(s60, kv))
                end
                wst[(ky, cod)] = max(get(wst, (ky, cod), 0.0), r9)
                r9 > 1e-9 && push!(bad, @sprintf("%-22s %-12s fp %2d %-10s %-7s %.3e",
                    string(scl), string(D), fp, FACENAMES[F] * "-" * FACENAMES[Fp], cod, r9))
                println(io, rpad(string(scl), 22), rpad(string(D), 12), rpad(fp, 4),
                        rpad(FACENAMES[F] * "-" * FACENAMES[Fp], 10), rpad(cod, 7),
                        rpad(mM + 2, 6),
                        rpad(@sprintf("%+.16e", real(g9)), 25),
                        rpad(@sprintf("%+.16e", imag(g9)), 25),
                        rpad(@sprintf("%+.26e", real(ser)), 34),
                        rpad(@sprintf("%+.26e", imag(ser)), 34),
                        rpad(@sprintf("%.3e", r9), 12), rpad(tr, 13), pk)
            end
            flush(io)
        end
        println(io)
        println(io, "## worst order-9 relative error per (cell scale, sub-code)")
        println(io, rpad("code", 8), join([rpad(string(s[2]), 24) for s in SCLS]))
        for cod in SHELL
            println(io, rpad(cod, 8),
                    join([rpad(@sprintf("%.3e", get(wst, (s[1], cod), 0.0)), 24) for s in SCLS]))
        end
        println(io)
        println(io, "## pairs where the order-9 rule is worse than 1e-9 relative")
        isempty(bad) && println(io, "none")
        for l in bad
            println(io, l)
        end
    end
end

# ===== (c) frequency homogeneity
function partC()
    setprecision(BigFloat, 256)
    frs = [("1", 1.0 + 0.0im), ("1+0.1i", 1.0 + 0.1im), ("0.37", 0.37 + 0.0im)]
    scl = (Q(1)//8, Q(1)//8, Q(1)//8)
    geo = [("S", :slf, 3), ("Ef", :edgFltX, 3), ("Ec", :edgCrnX, 3),
           ("Vc", :vtxCop, 3), ("Vp", :vtxPrpX, 3)]
    momOf(nm, d) = nm === :none ? pairMoments(facePair((0, 0, 0), 1, 2, scl)..., 8) :
                   pairMoments(wekPnl(nm, wekFrm(d, scl)...)..., 8)
    tee("f_gila_c.txt") do io
        println(io, "(c) frequency homogeneity of the series terms, scl = ", scl)
        println(io, "    term_n = (2 pi i f)^n I_{n-1} / (n! 4 pi f^2), so f^(2-n) term_n is")
        println(io, "    f-free; in particular f^2 term_0 = I_{-1} / (4 pi)")
        println(io, "    frequencies ", join([f[1] for f in frs], ", "))
        println(io)
        println(io, "## f^2 term_0 against I_{-1} / (4 pi), and Gila's static part against both")
        println(io)
        println(io, rpad("code", 6), rpad("pair", 12), rpad("f", 9),
                rpad("f^2 term_0 vs I_-1/4pi", 24), rpad("Gila fn", 22),
                rpad("f^2 rSrf vs I_-1/4pi", 22), "f^2 rSrf vs its f=1 value")
        for (cod, nm, d) in geo
            g = Float64.(wekFrm(d, scl))
            mom = momOf(nm, d)
            ref = mom[1] / (4 * BigFloat(pi))
            g0 = nothing
            for (fk, f) in frs
                z = Complex{BigFloat}(f)
                t0 = serTrms(mom, f)[1] * z^2
                gs = "--"
                gr = "--"
                gc = "--"
                if cod in ("S", "Ef", "Ec")
                    opt = GV.CPUKerOpt(f, 8, false, GV.CPU())
                    v, gs = cod == "S" ? (GV.rSrfSlf(g[1], g[2], opt), "rSrfSlf(gX,gY)") :
                            cod == "Ef" ? (GV.rSrfEdgFlt(g[2], g[1], opt), "rSrfEdgFlt(gY,gX)") :
                            (GV.rSrfEdgCrn(g[2], g[1], g[3], opt), "rSrfEdgCrn(gY,gX,gZ)")
                    w = Complex{BigFloat}(v) * z^2
                    gr = @sprintf("%.3e", rel(w, ref))
                    g0 === nothing && (g0 = w)
                    gc = @sprintf("%.3e", rel(w, g0))
                end
                println(io, rpad(cod, 6), rpad(string(nm), 12), rpad(fk, 9),
                        rpad(@sprintf("%.3e", rel(t0, ref)), 24), rpad(gs, 22),
                        rpad(gr, 22), gc)
            end
        end
        println(io)
        println(io, "## a non-touching pair too, D = (0,0,0) yzL-yzU, code ",
                geomCode((0, 0, 0), 1, 2), ", no Gila static part")
        mom = momOf(:none, 0)
        ref = mom[1] / (4 * BigFloat(pi))
        for (fk, f) in frs
            t0 = serTrms(mom, f)[1] * Complex{BigFloat}(f)^2
            println(io, rpad("P1", 6), rpad("yzL-yzU", 12), rpad(fk, 9),
                    @sprintf("%.3e", rel(t0, ref)))
        end
        println(io)
        println(io, "## f^(2-n) term_n against its value at f = 1, n = 0 .. 9")
        println(io)
        println(io, rpad("code", 6), rpad("pair", 12), rpad("f", 9),
                join([rpad("n=" * string(n), 11) for n in 0:9]))
        for (cod, nm, d) in vcat(geo, [("P1", :none, 0)])
            mom = momOf(nm, d)
            ref = nothing
            for (fk, f) in frs
                z = Complex{BigFloat}(f)
                tr = serTrms(mom, f)
                sc = [tr[n + 1] * z^(2 - n) for n in 0:9]
                ref === nothing && (ref = sc)
                println(io, rpad(cod, 6), rpad(nm === :none ? "yzL-yzU" : string(nm), 12),
                        rpad(fk, 9),
                        join([rpad(@sprintf("%.3e", rel(sc[n + 1], ref[n + 1])), 11) for n in 0:9]))
            end
        end
    end
end

PART in ("a", "all") && partA()
PART in ("b", "all") && partB()
PART in ("c", "all") && partC()
