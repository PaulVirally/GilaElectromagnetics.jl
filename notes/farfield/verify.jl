# Verification of farfield.jl. Runnable in parts, and every part reads the caches rather than
# recomputing a reference:
#   JULIA_NUM_THREADS=1 julia --startup-file=no --project=<env> verify.jl a
#   (a) geometry: rho whole box and octant, the fixed-size slow band, mu_n, the exact-rational
#       divergence identity (36 signed face-pair moments = the volume form, in Rational{BigInt})
#   (b) the volume identity and the normalisation: volTensor vs refTensor for the four sign
#       conventions, and farTensor against both
#   (c) Gila's far field TODAY: egoSrfFxd! + srfSum! vs the 220-bit reference on the item-3 grid
#   (d) farTensor vs every cached 220-bit reference: route, L, error, Re/Im
#   (e) term counts from the bounds: L per shape/separation/direction, nCut, routing counts for
#       32^3/64^3/128^3, and the Float64 digits lost against BigFloat at the same L
#   (f) the join: k-series band edges on the routed (iii) set, (iii) vs the volume reference, and
#       the overlap against famG's Taylor route over a 16^3 octant
#   (g) symmetries, homogeneity and number types, 50 random offsets per case
#   (h) anti-Hermitian positivity of a 6^3 and a (6,6,12) block, Gila-built vs farBlock!-built
#   (i) cost per offset against L, and the term histogram over a 32^3 octant
#   (j) unequal cells: farTensorX / farBlockX! against every cached 220-bit cross-scale
#       reference, both orientations, certificates, reflections (reads scratch/xwork/xref)
#   (k) unequal cells, adversarial geometries: odd and mixed ratios, non-integer ratios, lambda/2
#       and lambda coarse cells, complex f, mixed-sign offsets, Float32, a reused FrqSetX
#   (l) unequal cells: routing and lattice logic on the r = 2, 4, 8, 16 xyz test blocks, thread
#       determinism of farBlockX! (run with 1, 4, 12 threads; the dumps are compared)
#   (m) unequal cells: Gila's egoFunOut! today vs farTensorX vs reference on the r = 4 / 16 xyz
#       blocks, per-offset cost, and the 1.4e6-offset realistic block timing (needs Gila)
# Each part writes tables/<part>_*.txt as well as stdout. Parts (c) and (h) need Gila; the rest
# need only farfield.jl, ref.jl (which loads notes/moments/moments.jl and notes/gen/ref/mom.jl)
# and the caches refcache/reftensors.txt, ksrcache/ and the shape tables.
# Measured single-threaded run times with every cache present, on an M3 Pro carrying three other
# Julia processes (load average 8-10): a 5.8 s, b 14.4 s, c 4.5 s, d 22.4 s, e 85.7 s, f 89.3 s,
# g 37.7 s, h 553 s (492 s of it Gila's own wekTrp at intOrd 48 and its 6^3/(6,6,12) fills),
# i 23.9 s.  Nothing here recomputes a 220-bit reference; the only part that can is (c), which
# prints "no cached reference" instead (the slender (0,0,n) rows at n = 3, 6, 32, 64: pairKer
# needs more than 25 min per offset there, so refquad.jl is the tool for them).
# The 19.5 MB per-shape geometry tables are looked for in $FARFIELD_TABDIR, then shapetab/;
# farfield.jl rebuilds a missing one in 15-20 s and 0.7 GB.
using Printf, LinearAlgebra, BenchmarkTools
const PART = length(ARGS) == 0 ? "all" : ARGS[1]
include(joinpath(@__DIR__, "farfield.jl"))

const DIR = joinpath(@__DIR__, "tables")
mkpath(DIR)
function tabDir()
    for c in (get(ENV, "FARFIELD_TABDIR", ""), joinpath(@__DIR__, "shapetab"))
        isempty(c) && continue
        isdir(c) && any(endswith(f, "_p$(TABPRC)_seg.txt") for f in readdir(c)) && return c
    end
    mkpath(joinpath(@__DIR__, "shapetab"))
end
TABDIR[] = tabDir()
KSRDIR[] = joinpath(@__DIR__, "ksrcache")
setprecision(BigFloat, 256)

const CB = Complex{BigFloat}
const C32 = (QI(1)//32, QI(1)//32, QI(1)//32)
const C8 = (QI(1)//8, QI(1)//8, QI(1)//8)
const C4 = (QI(1)//4, QI(1)//4, QI(1)//4)
const SL = (QI(1)//32, QI(1)//32, QI(1)//512)
const SHP = [("c32", C32), ("c8", C8), ("c4", C4), ("sl", SL)]
const SNM = Dict(s => n for (n, s) in SHP)
const FRQ = (ComplexF64(1), ComplexF64(1, 0.1))
const SEPS = (2, 3, 4, 6, 8, 16, 32, 64)

struct Duo <: IO
    a::IO
    b::IO
end
Base.write(d::Duo, x::UInt8) = (write(d.a, x); write(d.b, x))
Base.flush(d::Duo) = (flush(d.a); flush(d.b))
tee(f, nam) = open(io -> f(Duo(stdout, io)), joinpath(DIR, nam), "w")

fm(x) = x == 0 ? "0       " : @sprintf("%.2e", x)
rp(x, n) = rpad(x, n)
dstr(D) = string("(", D[1], ",", D[2], ",", D[3], ")")

PART in ("a", "b", "c", "d", "f", "all") && include(joinpath(@__DIR__, "ref.jl"))
PART in ("f", "all") && include(joinpath(@__DIR__, "crosscheck", "famG.jl"))
if PART in ("c", "h", "all")
    @eval using GilaElectromagnetics
    @eval using StaticArrays
    @eval const GV = GilaElectromagnetics.GilaVacuum
end

# ---- the common reference cache ---------------------------------------------
prsQ(t) = (b = split(t, "//"); QI(parse(BigInt, b[1])) // parse(BigInt, b[2]))
"Every record of refcache/reftensors.txt, keyed by (kind, D, s, (Re f, Im f)); highest ord wins."
function loadRef()
    R = Dict{Tuple{String,NTuple{3,Int},NTuple{3,QI},NTuple{2,Float64}},Vector{CB}}()
    O = Dict{keytype(R),Int}()
    fn = joinpath(@__DIR__, "refcache", "reftensors.txt")
    isfile(fn) || error("no refcache/reftensors.txt; run mkrefcache.jl")
    for ln in eachline(fn)
        (isempty(strip(ln)) || startswith(ln, "#")) && continue
        p = split(strip(ln), '|')
        length(p) >= 7 || continue
        D = Tuple(parse.(Int, split(p[2][3:end], ',')))
        s = Tuple(prsQ.(split(p[3][3:end], ',')))
        fr = parse.(Float64, split(p[4][3:end], ';'))
        ord = parse(Int, p[6][3:end])
        ky = (String(p[1]), D, s, (fr[1], fr[2]))
        ord < get(O, ky, -1) && continue
        O[ky] = ord
        R[ky] = [CB(parse(BigFloat, x[1]), parse(BigFloat, x[2])) for x in split.(split(p[7]), ';')]
    end
    return R
end

"220-bit tensor for one case: the 36 face pairs if cached, else the volume rule, else any agent's."
function refOf(R, D, s, f)
    ky(k) = (k, D, s, (real(f), imag(f)))
    haskey(R, ky("pairs")) && return (srfSum(R[ky("pairs")]), "pairs")
    haskey(R, ky("vol")) && return (reshape(R[ky("vol")], 3, 3), "vol")
    haskey(R, ky("tns:refQuad")) && return (reshape(R[ky("tns:refQuad")], 3, 3), "refQuad")
    haskey(R, ky("tns:famDQ")) && return (reshape(R[ky("tns:famDQ")], 3, 3), "famDQ")
    for k in keys(R)
        k[2] == D && k[3] == s && k[4] == (real(f), imag(f)) && startswith(k[1], "tns") &&
            return (reshape(R[k], 3, 3), k[1][5:end])
    end
    return (nothing, "")
end

"(max entry error / largest entry, worst per entry, worst per Re, worst per Im), entries > 1e-8 mx."
function errs(G, Gr)
    mx = maximum(abs, Gr)
    eMx = Float64(maximum(abs, CB.(G) .- Gr) / mx)
    pEn = 0.0; eRe = 0.0; eIm = 0.0
    for i in 1:9
        abs(Gr[i]) > BigFloat(1e-8) * mx || continue
        pEn = max(pEn, Float64(abs(CB(G[i]) - Gr[i]) / abs(Gr[i])))
        abs(real(Gr[i])) > BigFloat(1e-8) * mx &&
            (eRe = max(eRe, Float64(abs(real(G[i]) - real(Gr[i])) / abs(real(Gr[i])))))
        abs(imag(Gr[i])) > BigFloat(1e-8) * mx &&
            (eIm = max(eIm, Float64(abs(imag(G[i]) - imag(Gr[i])) / abs(imag(Gr[i])))))
    end
    return (eMx, pEn, eRe, eIm)
end

# offsets of the item-3 grid: axis, face diagonal and body diagonal at eight separations
dirOff(tag, n) = tag == "axis" ? (n, 0, 0) : tag == "face" ? (n, n, 0) : tag == "body" ? (n, n, n) :
                 tag == "ax1" ? (n, 0, 0) : tag == "ax2" ? (0, n, 0) : (0, 0, n)
const DIRS = ("axis", "face", "body")
const DIRSL = ("ax1", "ax2", "ax3")
dirsOf(s) = s == SL ? DIRSL : DIRS

rhoWhl(D, s) = sqrt(sum(Float64(s[i])^2 for i in 1:3)) /
               sqrt(sum((Float64(D[i]) * Float64(s[i]))^2 for i in 1:3))
function rhoOct(D, s)
    rd = sqrt(sum(Float64(s[i])^2 for i in 1:3)) / 2
    mx = 0.0
    for sg in SGN8
        v = ntuple(d -> sg[d] * Float64(D[d]) * Float64(s[d]) + Float64(s[d]) / 2, 3)
        mx = max(mx, rd / sqrt(sum(v[i]^2 for i in 1:3)))
    end
    return mx
end

# ===== (a) geometry, exact ====================================================
# exact triangle-weight moment as two elementary halves, independent of wgtT's closed form
function muHlf(n::Int, s::QI)
    up = s * s^(n + 1) // (n + 1) - s^(n + 2) // (n + 2)
    dn = -s * (-s)^(n + 1) // (n + 1) - (-s)^(n + 2) // (n + 2)
    return up + dn
end

const Ply = Dict{NTuple{3,Int},QI}
pAdd!(P::Ply, e, c) = (P[e] = get(P, e, zero(QI)) + c; P)
function pMul(A::Ply, B::Ply)
    C = Ply()
    for (ea, ca) in A, (eb, cb) in B
        pAdd!(C, (ea[1] + eb[1], ea[2] + eb[2], ea[3] + eb[3]), ca * cb)
    end
    return C
end
function pDif(P::Ply, a::Int)
    C = Ply()
    for (e, c) in P
        e[a] == 0 && continue
        pAdd!(C, ntuple(d -> d == a ? e[d] - 1 : e[d], 3), c * e[a])
    end
    return C
end
pInt(P::Ply, s) = sum((c * muHlf(e[1], s[1]) * muHlf(e[2], s[2]) * muHlf(e[3], s[3])
                       for (e, c) in P); init = zero(QI))

# |R + delta|^{2j} as a polynomial in delta, exact
function rsqPow(R::NTuple{3,QI}, j::Int)
    B = Ply()
    for d in 1:3
        pAdd!(B, ntuple(q -> q == d ? 2 : 0, 3), one(QI))
        pAdd!(B, ntuple(q -> q == d ? 1 : 0, 3), 2 * R[d])
    end
    pAdd!(B, (0, 0, 0), sum(R[d]^2 for d in 1:3))
    P = Ply(); pAdd!(P, (0, 0, 0), one(QI))
    for _ in 1:j; P = pMul(P, B); end
    return P
end

# int_F int_F' |x-y|^{2j}: multinomial in the three axes, each axis an elementary polynomial
function axsPow(A, B, n::Integer)
    p, q = A; r, s = B
    p == q && r == s && return (p - r)^n
    p == q && return ((p - r)^(n + 1) - (p - s)^(n + 1)) // (n + 1)
    r == s && return ((q - r)^(n + 1) - (p - r)^(n + 1)) // (n + 1)
    H(u) = u^(n + 2) // ((n + 1) * (n + 2))
    return H(q - r) - H(q - s) - H(p - r) + H(p - s)
end
function momPol(pA, pB, m::Integer)
    k = div(m, 2)
    fac(n) = factorial(big(n))
    acc = zero(QI)
    for i in 0:k, j in 0:(k - i)
        l = k - i - j
        acc += QI(fac(k) ÷ (fac(i) * fac(j) * fac(l))) *
               axsPow(pA[1], pB[1], 2i) * axsPow(pA[2], pB[2], 2j) * axsPow(pA[3], pB[3], 2l)
    end
    return acc
end

function partA()
    tee("a_rho.txt") do io
        println(io, "(a) convergence ratios.  rho = sqrt(sum s_i^2)/|R| for the whole difference box;")
        println(io, "    rho_oct = max over the 8 octants of (r_d/2)/|sigma R + s/2|.  Cubic rows are")
        println(io, "    scale invariant; the slender rows are (1/32,1/32,1/512).")
        println(io)
        println(io, rp("class", 12), rp("n", 5), rp("rho", 10), rp("rho_oct", 10), "rho/rho_oct")
        cls = (("axis", n -> (n, 0, 0)), ("facediag", n -> (n, n, 0)), ("bodydiag", n -> (n, n, n)),
               ("(n,1,0)", n -> (n, 1, 0)), ("(n,2,1)", n -> (n, 2, 1)))
        for (nm, fn) in cls, n in SEPS
            D = fn(n)
            r = rhoWhl(D, C32); ro = rhoOct(D, C32)
            println(io, rp(nm, 12), rp(n, 5), rp(round(r, digits = 6), 10),
                    rp(round(ro, digits = 6), 10), round(r / ro, digits = 3))
        end
        println(io)
        for (nm, fn) in (("sl (n,0,0)", n -> (n, 0, 0)), ("sl (n,n,0)", n -> (n, n, 0)),
                         ("sl (n,n,n)", n -> (n, n, n)), ("sl (0,0,n)", n -> (0, 0, n)))
            for n in SEPS
                D = fn(n)
                r = rhoWhl(D, SL); ro = rhoOct(D, SL)
                println(io, rp(nm, 12), rp(n, 5), rp(round(r, digits = 6), 10),
                        rp(round(ro, digits = 6), 10), round(r / ro, digits = 3))
            end
        end
        println(io)
        println(io, "first n with rho < 1 / rho < 0.5, slender short axis (0,0,n): ",
                findfirst(n -> rhoWhl((0, 0, n), SL) < 1, 1:200), " / ",
                findfirst(n -> rhoWhl((0, 0, n), SL) < 0.5, 1:200))
        println(io, "same for rho_oct: ",
                findfirst(n -> rhoOct((0, 0, n), SL) < 1, 1:200), " / ",
                findfirst(n -> rhoOct((0, 0, n), SL) < 0.5, 1:200))
    end
    tee("a_band.txt") do io
        println(io, "(a) the slow band is a fixed set of offsets, independent of N: counts of egoToe")
        println(io, "    octant offsets (0 <= n_i < N, max n_i >= 2) with rho above a threshold.")
        println(io, "    The 7 touching offsets with max n_i = 1 are excluded here (geometry.md")
        println(io, "    included them and reported 37/141/436/3096 for the cube).  For the slender")
        println(io, "    cell the band is NOT of fixed size: rho > 1 along (0,0,n) up to n = 22.")
        println(io)
        println(io, rp("shape", 6), rp("N", 6), rp("offsets", 10), rp("rho>0.5", 9), rp("rho>0.3", 9),
                rp("rho>0.2", 9), "rho>0.1")
        for (tg, s) in (("cube", C32), ("sl", SL)), N in (32, 64, 128)
            c = zeros(Int, 4); tot = 0
            for i3 in 0:(N - 1), i2 in 0:(N - 1), i1 in 0:(N - 1)
                maximum((i1, i2, i3)) >= 2 || continue
                tot += 1
                r = rhoWhl((i1, i2, i3), s)
                r > 0.5 && (c[1] += 1); r > 0.3 && (c[2] += 1)
                r > 0.2 && (c[3] += 1); r > 0.1 && (c[4] += 1)
            end
            println(io, rp(tg, 6), rp(N, 6), rp(tot, 10), rp(c[1], 9), rp(c[2], 9), rp(c[3], 9), c[4])
        end
    end
    tee("a_mu.txt") do io
        println(io, "(a) mu_n(s) = int_{-s}^{s} (s-|t|) t^n dt = 2 s^{n+2}/((n+1)(n+2)) for even n, 0 odd.")
        println(io, "    wgtT of farfield.jl against the same integral done as two elementary halves,")
        println(io, "    both in Rational{BigInt}: the column is `exact` when they are the same rational.")
        println(io)
        println(io, rp("s", 10), rp("n", 4), rp("mu_n (halves)", 34), "wgtT == halves")
        for s in (QI(1)//32, QI(1)//512, QI(1)//4, QI(7)//3), n in 0:12
            h = muHlf(n, s); w = wgtT(n, s)
            n <= 4 || s == QI(1)//32 || continue
            println(io, rp(string(s), 10), rp(n, 4), rp(string(h), 34), h == w)
        end
        bad = 0
        for s in (QI(1)//32, QI(1)//512, QI(1)//4, QI(7)//3, QI(1000)//1), n in 0:24
            muHlf(n, s) == wgtT(n, s) || (bad += 1)
        end
        println(io)
        println(io, "disagreements over 5 lengths x n = 0..24: ", bad)
    end
    tee("a_div.txt") do io
        println(io, "(a) the divergence identity in exact rational arithmetic, on F = |x-y|^{2j}:")
        println(io, "      (1/V_t) srfSum!(int_F int_F' F) == (1/V_t) int_D w(d) [(da db - dab lap) F](R+d)")
        println(io, "    left: 36 face-pair moments from facePair panels by an exact multinomial")
        println(io, "    integrator; right: the difference-box form with the exact weight moments.")
        println(io, "    Both sides are Rational{BigInt}; `max |diff|` is exactly 0 when they agree.")
        println(io)
        println(io, rp("shape", 6), rp("D", 10), rp("j", 4), rp("max |diff|", 14), rp("G[1,1]", 34), "G[1,2]")
        for (tg, s) in (("c32", C32), ("sl", SL)), D in ((2, 0, 0), (3, 1, 0), (2, 2, 2)), j in 1:3
            R = ntuple(d -> QI(D[d]) * s[d], 3)
            P = rsqPow(R, j)
            vt = s[1] * s[2] * s[3]
            lap = sum(pInt(pDif(pDif(P, a), a), s) for a in 1:3)
            V = zeros(QI, 3, 3)
            for a in 1:3, b in 1:3
                V[a, b] = pInt(pDif(pDif(P, a), b), s) - (a == b ? lap : zero(QI))
            end
            V ./= vt
            sm = [momPol(facePair(D, F, Fp, s)..., 2j) / vt for F in 1:6 for Fp in 1:6]
            S = srfSum(sm)
            df = maximum(abs, S .- V)
            println(io, rp(tg, 6), rp(dstr(D), 10), rp(j, 4), rp(string(df), 14),
                    rp(string(V[1, 1]), 34), string(V[1, 2]))
        end
    end
end

# ===== (b) the volume identity and the normalisation ==========================
function partB()
    R = loadRef()
    vks = sort([k for k in keys(R) if k[1] == "vol"]; by = k -> (SNM[k[3]], k[4], maximum(abs, k[2])))
    tee("b_volume.txt") do io
        println(io, "(b) volTensor (independent BigFloat Gauss-Legendre rule on the difference box, graded")
        println(io, "    to the near corner) against refTensor (36 x pairKer at 220 bits, srfSum! signs),")
        println(io, "    for the four sign conventions of the volume form, and farTensor against both.")
        println(io, "    ePP = (+diag,+off) = max_ab |G_vol - G_ref| / max|G_ref|; ePM, eMP, eMM the")
        println(io, "    other three; entPP the worst per-entry ePP.  fMx/fEn = farTensor vs refTensor,")
        println(io, "    fVol = farTensor vs volTensor (max norm).  rt = route taken by farTensor.")
        println(io)
        println(io, rp("shape", 6), rp("f", 10), rp("D", 11), rp("ePP", 10), rp("entPP", 10), rp("ePM", 10),
                rp("eMP", 10), rp("eMM", 10), rp("rt", 4), rp("fMx", 10), rp("fEn", 10), "fVol")
        fsC = Dict{Tuple{NTuple{3,QI},NTuple{2,Float64}},Any}()
        # one set per (shape, f), so nBlk must cover the largest offset of the group, not the first
        nBk = Dict{Tuple{NTuple{3,QI},NTuple{2,Float64}},Int}()
        for k in vks; nBk[(k[3], k[4])] = max(get(nBk, (k[3], k[4]), 4), maximum(abs, k[2])); end
        wst = zeros(3)
        for k in vks
            _, D, s, fr = k
            haskey(SNM, s) || continue
            f = ComplexF64(fr[1], fr[2])
            Gv = reshape(R[k], 3, 3)
            Gr, src = refOf(R, D, s, f)
            (Gr === nothing || src != "pairs") && continue
            mx = maximum(abs, Gr)
            cnv(dg, of) = [i == j ? dg * Gv[i, j] : of * Gv[i, j] for i in 1:3, j in 1:3]
            eOf(A) = Float64(maximum(abs, A .- Gr) / mx)
            ePP = eOf(Gv); ePM = eOf(cnv(1, -1)); eMP = eOf(cnv(-1, 1)); eMM = eOf(cnv(-1, -1))
            entPP = maximum(abs(Gr[i]) > BigFloat(1e-8) * mx ?
                            Float64(abs(Gv[i] - Gr[i]) / abs(Gr[i])) : 0.0 for i in 1:9)
            fs = get!(fsC, (s, fr)) do
                farSetup(s, f; nBlk = nBk[(s, fr)])
            end
            kind, L, Lc, _ = farRoute(fs, D)
            G = farTensor(D, s, f; fs = fs)
            fMx, fEn, _, _ = errs(G, Gr)
            fVol = Float64(maximum(abs, CB.(G) .- Gv) / mx)
            wst[1] = max(wst[1], ePP); wst[2] = max(wst[2], fMx); wst[3] = max(wst[3], fEn)
            println(io, rp(SNM[s], 6), rp(string(fr[1], ",", fr[2]), 10), rp(dstr(D), 11), rp(fm(ePP), 10),
                    rp(fm(entPP), 10), rp(fm(ePM), 10), rp(fm(eMP), 10), rp(fm(eMM), 10), rp(kind, 4),
                    rp(fm(fMx), 10), rp(fm(fEn), 10), fm(fVol))
        end
        println(io)
        println(io, "worst over the table: volume form vs face pairs ", fm(wst[1]),
                ";  farTensor vs reference ", fm(wst[2]), " (max norm), ", fm(wst[3]), " (per entry)")
    end
end

# ===== (c) Gila's far field today =============================================
function partC()
    R = loadRef()
    for (tg, s) in SHP, f in FRQ
        rws = NTuple{3,Any}[]
        for dr in dirsOf(s), n in SEPS
            push!(rws, (dr, n, dirOff(dr, n)))
        end
        any(refOf(R, r[3], s, f)[1] !== nothing for r in rws) || continue
        tee("c_gila_$(tg)_$(imag(f) == 0 ? "r" : "c").txt") do io
            println(io, "(c) Gila's egoSrfFxd! + srfSum! against the 220-bit reference, shape ", tg,
                    " = ", Float64.(s), ", f = ", f)
            println(io, "    ord = quadOrd(sep, 2 pi |f| max s); pairErr = worst of the 36 face pairs;")
            println(io, "    amp = max_ab (sum of |srfMat_ref| over the pairs summed into ab)/|G_ref[ab]|;")
            println(io, "    tMx = max_ab |G_gila - G_ref| / max|G_ref|; tEn = worst per entry.")
            println(io, "    A row with no cached 36-pair reference falls back to the cached tensor")
            println(io, "    (refquad.jl's graded volume rule): pairErr/fp/amp are then blank and only")
            println(io, "    the assembled tMx/tEn columns are meaningful.  A row with neither prints")
            println(io, "    \"no cached reference\"; build it with refPairs(D, s, f) of ref.jl.")
            println(io)
            println(io, rp("dir", 6), rp("sep", 5), rp("ord", 5), rp("pairErr", 11), rp("fp", 5),
                    rp("amp", 9), rp("ab", 4), rp("tMx", 10), rp("tEn", 10), rp("ab", 4),
                    rp("max|G_ref|", 10), "src")
            fac = Float64.(GV.cubFac(s)); fp36 = GV.facPar()
            scl = Float64.(GV.srfScl(Float64.(s), Float64.(s)))
            opt = GV.CPUKerOpt{Float64}(f, 48, false, GV.CPU())
            kScl = 2pi * abs(f) * maximum(Float64.(s))
            for (dr, n, D) in rws
                ky = ("pairs", D, s, (real(f), imag(f)))
                sm = get(R, ky, nothing)
                Gt, src = sm === nothing ? refOf(R, D, s, f) : (nothing, "pairs")
                (sm === nothing && Gt === nothing) &&
                    (println(io, rp(dr, 6), rp(n, 5), "no cached reference"); continue)
                Gr = sm === nothing ? Gt : srfSum(sm); mx = maximum(abs, Gr)
                grd = ntuple(d -> Float64(D[d]) * Float64(s[d]), 3)
                srf = zeros(MVector{36,ComplexF64})
                ord = GV.quadOrd(n, kScl)
                GV.egoSrfFxd!(grd[1], grd[2], grd[3], srf, fac, fac, 1:36, fp36, scl, opt, ord)
                G = zeros(ComplexF64, 3, 3); GV.srfSum!(G, srf)
                pe = 0.0; pfp = 0; amp = 0.0; aab = 0
                if sm !== nothing
                    for i in 1:36
                        abs(sm[i]) > 0 || continue
                        e = Float64(abs(CB(srf[i]) - sm[i]) / abs(sm[i]))
                        e > pe && (pe = e; pfp = i)
                    end
                    for a in 1:3, b in 1:3
                        abs(Gr[a, b]) > BigFloat(1e-8) * mx || continue
                        v = Float64(sum(abs(sm[i]) for i in SUMIDX[a, b]) / abs(Gr[a, b]))
                        v > amp && (amp = v; aab = 10a + b)
                    end
                end
                tMx = Float64(maximum(abs, CB.(G) .- Gr) / mx)
                tEn = 0.0; eab = 0
                for a in 1:3, b in 1:3
                    abs(Gr[a, b]) > BigFloat(1e-8) * mx || continue
                    e = Float64(abs(CB(G[a, b]) - Gr[a, b]) / abs(Gr[a, b]))
                    e > tEn && (tEn = e; eab = 10a + b)
                end
                println(io, rp(dr, 6), rp(n, 5), rp(ord, 5),
                        rp(sm === nothing ? "-" : fm(pe), 11), rp(sm === nothing ? "-" : string(pfp), 5),
                        rp(sm === nothing ? "-" : string(round(amp, sigdigits = 3)), 9),
                        rp(sm === nothing ? "-" : string(aab), 4), rp(fm(tMx), 10), rp(fm(tEn), 10),
                        rp(eab, 4), rp(fm(Float64(mx)), 10), sm === nothing ? src : "pairs")
            end
        end
    end
end

# ===== (d) the new method against every cached reference ======================
function partD()
    R = loadRef()
    cs = Set((k[2], k[3], k[4]) for k in keys(R))          # the distinct (D, s, f) cases
    grd = Set{Tuple{NTuple{3,Int},NTuple{3,QI}}}()
    for (_, s) in SHP, dr in dirsOf(s), n in SEPS
        push!(grd, (dirOff(dr, n), s))
    end
    tee("d_ref.txt") do io
        println(io, "(d) farTensor against every cached 220-bit reference.  rt = route (1 whole box,")
        println(io, "    2 octants, 3 k-series), L = whole-box L or the largest sub-box L; eMx =")
        println(io, "    max_ab |G-Gr|/max|Gr|; pEn = worst per entry (entries above 1e-8 of the largest);")
        println(io, "    eRe/eIm = the same for the real and imaginary parts separately; grd marks the")
        println(io, "    item-3 grid (8 separations x 3 directions); src = which cache the reference is.")
        println(io)
        println(io, rp("shp", 5), rp("D", 14), rp("f", 12), rp("rho", 8), rp("kR", 9), rp("rt", 4),
                rp("L", 5), rp("eMx", 10), rp("pEn", 10), rp("eRe", 10), rp("eIm", 10), rp("grd", 5), "src")
        tot = Dict{Tuple{String,NTuple{2,Float64}},Vector{Float64}}()
        nrow = 0
        for (tg, s) in SHP
            frs = sort(unique([k[3] for k in cs if k[2] == s]))
            for fr in frs
                f = ComplexF64(fr[1], fr[2])
                ks = sort([k for k in cs if k[2] == s && k[3] == fr];
                          by = k -> (maximum(abs, k[1]), k[1]))
                isempty(ks) && continue
                fs = farSetup(s, f; nBlk = max(4, maximum(maximum(abs, k[1]) for k in ks)))
                for k in ks
                    D = k[1]
                    maximum(abs, D) >= 2 || continue
                    Gr, src = refOf(R, D, s, f)
                    Gr === nothing && continue
                    kind, L, Lc, _ = farRoute(fs, D)
                    G = farTensor(D, s, f; fs = fs)
                    e = errs(G, Gr)
                    rr = sqrt(sum((Float64(D[i]) * Float64(s[i]))^2 for i in 1:3))
                    v = get!(tot, (tg, fr), zeros(5))
                    v[1] = max(v[1], e[1]); v[2] = max(v[2], e[2])
                    v[3] = max(v[3], e[3]); v[4] = max(v[4], e[4]); v[5] += 1
                    nrow += 1
                    println(io, rp(tg, 5), rp(dstr(D), 14), rp(string(fr[1], ",", fr[2]), 12),
                            rp(round(rhoWhl(D, s), digits = 4), 8),
                            rp(round(2pi * abs(f) * rr, digits = 3), 9), rp(kind, 4),
                            rp(kind == 1 ? L : kind == 2 ? maximum(Lc) : -1, 5),
                            rp(fm(e[1]), 10), rp(fm(e[2]), 10), rp(fm(e[3]), 10), rp(fm(e[4]), 10),
                            rp((D, s) in grd ? "yes" : "", 5), src)
                end
            end
        end
        println(io)
        println(io, "# summary, worst over each (shape, f)")
        println(io, rp("shp", 5), rp("f", 12), rp("n", 6), rp("eMx", 10), rp("pEn", 10), rp("eRe", 10), "eIm")
        for (k, v) in sort(collect(tot))
            println(io, rp(k[1], 5), rp(string(k[2][1], ",", k[2][2]), 11), rp(Int(v[5]), 6),
                    rp(fm(v[1]), 10), rp(fm(v[2]), 10), rp(fm(v[3]), 10), fm(v[4]))
        end
        println(io)
        println(io, "rows: ", nrow, ";  worst over the whole table: eMx ",
                fm(maximum(v[1] for v in values(tot))), "  pEn ",
                fm(maximum(v[2] for v in values(tot))))
    end
end

# ===== (e) term counts from the bounds ========================================
function partE()
    tee("e_terms.txt") do io
        println(io, "(e) L(1e-13) from the proven tail bound, per shape, separation and direction.")
        println(io, "    Lwb = smallest even L <= $LMAX whose whole-box tail is below tol*est(R);")
        println(io, "    Loct = largest of the 8 sub-box L_j for the octant split (budget tol*est/8);")
        println(io, "    -1 = the bound is not met at L <= $LMAX.  cost = complex multiply-adds issued.")
        println(io)
        println(io, rp("shp", 5), rp("f", 12), rp("dir", 6), rp("sep", 5), rp("rho", 8), rp("kR", 9),
                rp("Lwb", 5), rp("cost", 8), rp("Loct", 5), rp("costOct", 9), "route")
        for (tg, s) in SHP, f in FRQ
            fs = farSetup(s, f; nBlk = 128)
            for dr in dirsOf(s), n in SEPS
                D = dirOff(dr, n)
                Rv = ntuple(d -> Float64(D[d]) * Float64(s[d]), 3)
                rr = sqrt(sum(Rv[i]^2 for i in 1:3))
                L = boundL(fs, rr)
                Lc = boundLoct(fs, Rv)
                kind, Lr, Lcr, _ = farRoute(fs, D)
                println(io, rp(tg, 5), rp(string(f), 12), rp(dr, 6), rp(n, 5),
                        rp(round(rhoWhl(D, s), digits = 4), 8), rp(round(2pi * abs(f) * rr, digits = 3), 9),
                        rp(L, 5), rp(L < 0 ? -1 : costWhl(fs.whl, L), 8), rp(maximum(Lc), 5),
                        rp(any(<(0), Lc) ? -1 : costOct(fs.oct, Lc), 9), kind)
            end
        end
    end
    tee("e_ncut.txt") do io
        println(io, "(e) n-truncation: nCut[l+1] is the smallest N whose j_l n-tail bound is below")
        println(io, "    tol*est/(NBUD (L+1)), evaluated at the smallest radius the expansion is used")
        println(io, "    at and at the far end of the block.  nNd = max nCut is what the table must")
        println(io, "    carry, nMx what it does; the table is built to max(NMAX = $NMAX, nNd), so nNd")
        println(io, "    <= nMx always holds and farSetup refuses when the bound needs N > NCAP = $NCAP.")
        println(io, "    |k| r_d is the parameter that drives N (r_d = sqrt(sum s_i^2)).")
        println(io)
        println(io, rp("shp", 5), rp("f", 12), rp("|k| r_d", 10), rp("nNd", 6), rp("nMx", 6),
                "nCut at l = 0,2,4,...")
        for (tg, s) in SHP, f in FRQ
            dim = s == SL ? (64, 64, 128) : (128, 128, 128)
            fs = farSetup(s, f; nBlk = maximum(dim))
            sf = ntuple(d -> Float64(s[d]), 3)
            krd = 2pi * abs(f) * sqrt(sum(sf[d]^2 for d in 1:3))
            println(io, rp(tg, 5), rp(string(f), 12), rp(round(krd, sigdigits = 4), 10),
                    rp(fs.nNd, 6), rp(fs.nMx, 6), join(fs.nCut[1:2:end], " "))
        end
    end
    tee("e_route.txt") do io
        println(io, "(e) routing counts over the egoToe octant (offsets with max index >= 3, i.e. max-norm")
        println(io, "    separation >= 2), and the whole-box L histogram, at the default tol = $TOL.")
        println(io)
        println(io, rp("shp", 5), rp("f", 12), rp("dim", 16), rp("(i)", 11), rp("(ii)", 7), rp("(iii)", 7),
                "setup s")
        hst = Dict{Tuple{String,String},Dict{Int,Int}}()
        for (tg, s) in SHP, f in FRQ
            dims = s == SL ? ((16, 16, 32), (32, 32, 64), (64, 64, 128)) :
                   ((32, 32, 32), (64, 64, 64), (128, 128, 128))
            for dim in dims
                t0 = time()
                fs = farSetup(s, f; nBlk = maximum(dim))
                cnt, hs, ksr, _ = farRouteStat(fs, dim)
                dim == dims[end] && (hst[(tg, string(f))] = hs)
                println(io, rp(tg, 5), rp(string(f), 12), rp(string(dim), 16), rp(cnt[1], 11),
                        rp(cnt[2], 7), rp(cnt[3], 7), round(time() - t0, digits = 1))
            end
        end
        println(io)
        println(io, "# whole-box L histograms at the largest block")
        for k in sort(collect(keys(hst)))
            h = hst[k]
            println(io, rp(k[1], 5), rp(k[2], 11), join(["$l:$(h[l])" for l in sort(collect(keys(h)))], " "))
        end
    end
    tee("e_digits.txt") do io
        println(io, "(e) Float64 digits lost by the expansion itself: the same route and the same L")
        println(io, "    evaluated in Float64 and in BigFloat(192), so truncation cancels exactly.")
        println(io, "    digits = log10( max_ab|G64 - GB| / (eps(Float64) max_ab|GB|) ).")
        println(io)
        println(io, rp("shp", 5), rp("f", 12), rp("D", 12), rp("rho", 8), rp("kR", 9), rp("rt", 4),
                rp("L", 5), rp("rel err", 10), "digits lost")
        wst = -Inf; wat = ""
        for (tg, s) in SHP, f in FRQ
            (s == C8 || (s != C32 && imag(f) != 0)) && continue
            fs = farSetup(s, f; nBlk = 128)
            fB = setprecision(BigFloat, 192) do
                farSetup(s, CB(BigFloat(real(f)), BigFloat(imag(f))); nBlk = 128)
            end
            ws = FarWs(fs.L, Float64)
            wB = setprecision(() -> FarWs(fB.L, BigFloat), BigFloat, 192)
            G = zeros(ComplexF64, 3, 3)
            GB = zeros(CB, 3, 3)
            for dr in dirsOf(s), n in SEPS
                D = dirOff(dr, n)
                kind, L, Lc, _ = farRoute(fs, D)
                kind == 3 && continue
                Rv = ntuple(d -> Float64(D[d]) * Float64(s[d]), 3)
                RB = ntuple(d -> BigFloat(QI(D[d]) * s[d]), 3)
                kind == 1 ? tnsWhl!(G, fs, ws, Rv, L) : tnsOct!(G, fs, ws, Rv, Lc)
                setprecision(BigFloat, 192) do
                    kind == 1 ? tnsWhl!(GB, fB, wB, RB, L) : tnsOct!(GB, fB, wB, RB, Lc)
                end
                mx = maximum(abs, GB)
                rel = Float64(maximum(abs, CB.(G) .- GB) / mx)
                dg = log10(max(rel, 1e-30) / eps(Float64))
                rr = sqrt(sum(Rv[i]^2 for i in 1:3))
                dg > wst && (wst = dg; wat = "$tg $f $(dstr(D))")
                println(io, rp(tg, 5), rp(string(f), 12), rp(dstr(D), 12),
                        rp(round(rhoWhl(D, s), digits = 4), 8), rp(round(2pi * abs(f) * rr, digits = 3), 9),
                        rp(kind, 4), rp(kind == 1 ? L : maximum(Lc), 5), rp(fm(rel), 10),
                        round(dg, digits = 2))
            end
        end
        println(io)
        println(io, "worst digits lost: ", round(wst, digits = 2), " at ", wat)
    end
end

# ===== (f) the join ===========================================================
function partF()
    R = loadRef()
    tee("f_bands.txt") do io
        println(io, "(f) the k-series band, recomputed on the offsets the router actually sends to (iii).")
        println(io, "    Dmax = largest |x-y| over the 36 face pairs; x = |k| Dmax; Lam = the assembly")
        println(io, "    amplification (sum_fp I_{-1})/(4 pi |f|^2 V_t)/est(R); N = smallest order with")
        println(io, "    x^{N+1}/(N+1)! min(e^x, 1/(1-x/(N+2))) <= 1e-16/Lam.  epsLam = eps(Float64)*Lam")
        println(io, "    is what a Float64 face-pair evaluation would carry, which is why (iii) is done")
        println(io, "    in BigFloat($KSRPRC).")
        println(io)
        bf(p) = map(iv -> (BigFloat(iv[1]), BigFloat(iv[2])), p)
        for (tg, s) in SHP
            dim = s == SL ? (64, 64, 128) : (128, 128, 128)
            f = ComplexF64(1)
            fs = farSetup(s, f; nBlk = maximum(dim))
            cnt, _, ksr, _ = farRouteStat(fs, dim)
            println(io, "shape ", tg, " = ", Float64.(s), ", block ", dim, ": route (iii) offsets ",
                    length(ksr), " of ", sum(cnt))
            if isempty(ksr)
                println(io, "    none: the k-series is never needed on this shape\n")
                continue
            end
            println(io, rp("  D", 12), rp("Dmax", 10), rp("x=|k|Dmax", 11), rp("Lam", 11), rp("epsLam", 10),
                    rp("N(f=1)", 8), "N(f=1+0.1i)")
            xs = Float64[]; Ns = Int[]; lms = Float64[]
            for D in ksr
                pn = [map(bf, facePair(D, F, Fp, s)) for F in 1:6 for Fp in 1:6]
                dmx = maximum(first(panSpan(p[1], p[2])) for p in pn)
                lam, N1, N2 = setprecision(BigFloat, KSRPRC) do
                    lm = sum(pairMoments(p[1], p[2], -1)[1] for p in pn)
                    vtB = prod(ntuple(d -> BigFloat(s[d]), 3))
                    rr = sqrt(sum((BigFloat(D[d]) * BigFloat(s[d]))^2 for d in 1:3))
                    v = Float64[]
                    for fq in FRQ
                        fB = CB(BigFloat(real(fq)), BigFloat(imag(fq)))
                        kk = 2 * BigFloat(pi) * fB
                        am = lm / (4 * BigFloat(pi) * abs2(fB) * vtB) / est(rr, kk, fB, vtB)
                        push!(v, Float64(am))
                        push!(v, Float64(ksrOrd(abs(kk) * dmx, 1e-16 / max(1.0, Float64(am)), 200)))
                    end
                    (v[1], Int(v[2]), Int(v[4]))
                end
                push!(xs, Float64(2pi * dmx)); push!(Ns, max(N1, N2)); push!(lms, lam)
                println(io, rp("  " * dstr(D), 12), rp(round(Float64(dmx), digits = 5), 10),
                        rp(round(Float64(2pi * dmx), digits = 5), 11), rp(round(lam, sigdigits = 4), 11),
                        rp(fm(eps(Float64) * lam), 10), rp(N1, 8), N2)
            end
            println(io, "  band: x in [", round(minimum(xs), digits = 4), ", ",
                    round(maximum(xs), digits = 4), "], N in [", minimum(Ns), ", ", maximum(Ns),
                    "], Lam up to ", round(maximum(lms), sigdigits = 4), ", eps*Lam up to ",
                    fm(eps(Float64) * maximum(lms)))
            println(io)
        end
    end
    tee("f_ksr.txt") do io
        println(io, "(f) route (iii) against the graded BigFloat Gauss-Legendre volume reference")
        println(io, "    (cache kind tns:refQuad / vol), on the slender near needle, both frequencies.")
        println(io, "    rt = the route farTensor takes at that offset; (0,0,20) is outside the (iii)")
        println(io, "    set (n3 = 2..19) and is routed to the octant split, so its k-series column is")
        println(io, "    the forced tnsKsr value and the eMx column is route (ii).")
        println(io)
        println(io, rp("D", 12), rp("f", 12), rp("rt", 4), rp("eMx", 10), rp("pEn", 10), rp("eRe", 10),
                rp("eIm", 10), rp("ksr eMx", 10), "reference")
        for D in ((0, 0, 2), (0, 0, 8), (0, 0, 20), (1, 0, 2), (1, 1, 2), (1, 1, 12), (0, 1, 30)),
            f in FRQ
            Gr, src = refOf(R, D, SL, f)
            Gr === nothing && (println(io, rp(dstr(D), 12), rp(string(f), 12), "no cached reference"); continue)
            fs = farSetup(SL, f; nBlk = 64)
            kind, _, _, _ = farRoute(fs, D)
            G = farTensor(D, SL, f; fs = fs)
            e = errs(G, Gr)
            Gk = kind == 3 ? G : first(tnsKsr(SL, D, f))
            ek = errs(Gk, Gr)
            println(io, rp(dstr(D), 12), rp(string(f), 12), rp(kind, 4), rp(fm(e[1]), 10), rp(fm(e[2]), 10),
                    rp(fm(e[3]), 10), rp(fm(e[4]), 10), rp(fm(ek[1]), 10), src)
        end
    end
    tee("f_overlap.txt") do io
        println(io, "(f) the overlap: farTensor against famG's (G,S) Taylor recurrence at famG's own")
        println(io, "    order rule q = 2 ceil((17/log10(1/rho) + 8)/2), over every offset of a 16^3")
        println(io, "    octant with max-norm >= 2.  dMx = max_ab|Gb-Gg|/max|Gg|, dEn = worst per entry.")
        println(io)
        println(io, rp("shp", 5), rp("f", 12), rp("band", 7), rp("n", 7), rp("dMx", 10), rp("dEn", 10),
                rp("dRe", 10), rp("dIm", 10), "worst offset")
        for (tg, s) in (("c32", C32), ("c4", C4)), f in FRQ
            fs = farSetup(s, f; nBlk = 16)
            s6 = ntuple(d -> Float64(s[d]), 3); rd = sqrt(sum(s6[i]^2 for i in 1:3))
            bnd = Dict{String,Vector{Float64}}(); wo = Dict{String,NTuple{3,Int}}()
            for i3 in 1:16, i2 in 1:16, i1 in 1:16
                D = (i1 - 1, i2 - 1, i3 - 1)
                maximum(D) >= 2 || continue
                Rv = ntuple(d -> Float64(D[d]) * s6[d], 3)
                rho = rd / sqrt(sum(Rv[i]^2 for i in 1:3))
                rho >= 1 && continue
                q = 2 * ceil(Int, (17 / log10(1 / rho) + 8) / 2)
                q > 96 && continue
                Gg = farTen(first(taylorSys(Rv, f, q + 2)), s6, f, q)
                Gb = farTensor(D, s, f; fs = fs)
                mx = maximum(abs, Gg)
                dMx = maximum(abs, Gb .- Gg) / mx
                dEn = 0.0; dRe = 0.0; dIm = 0.0
                for i in 1:9
                    abs(Gg[i]) > 1e-8 * mx || continue
                    dEn = max(dEn, abs(Gb[i] - Gg[i]) / abs(Gg[i]))
                    abs(real(Gg[i])) > 1e-8 * mx &&
                        (dRe = max(dRe, abs(real(Gb[i]) - real(Gg[i])) / abs(real(Gg[i]))))
                    abs(imag(Gg[i])) > 1e-8 * mx &&
                        (dIm = max(dIm, abs(imag(Gb[i]) - imag(Gg[i])) / abs(imag(Gg[i]))))
                end
                mn = maximum(D)
                bd = mn <= 2 ? "2" : mn <= 4 ? "3-4" : mn <= 8 ? "5-8" : "9-15"
                v = get!(bnd, bd, zeros(5)); v[5] += 1
                v[1] < dMx && (wo[bd] = D)
                v[1] = max(v[1], dMx); v[2] = max(v[2], dEn)
                v[3] = max(v[3], dRe); v[4] = max(v[4], dIm)
            end
            for bd in ("2", "3-4", "5-8", "9-15")
                haskey(bnd, bd) || continue
                v = bnd[bd]
                println(io, rp(tg, 5), rp(string(f), 12), rp(bd, 7), rp(Int(v[5]), 7), rp(fm(v[1]), 10),
                        rp(fm(v[2]), 10), rp(fm(v[3]), 10), rp(fm(v[4]), 10), dstr(get(wo, bd, (0, 0, 0))))
            end
        end
    end
end

# ===== (g) symmetries, homogeneity, number types ==============================
function partG()
    rel(A, B) = maximum(abs, A .- B) / maximum(abs, B)
    tee("g_sym.txt") do io
        println(io, "(g) invariances on 50 random offsets per case, drawn from [-40,40]^3 with max|n| >= 2")
        println(io, "    (fixed seed).  D -> -D, transpose and the axis reflections are exact by")
        println(io, "    construction in this expansion; the axis permutation mixes different (l,m)")
        println(io, "    entries of the table and is the honest test.  Homogeneity is")
        println(io, "    T(lam f, s/lam) = lam^{-2} T(f, s) at lam = 3.7, both sides at L = 20, nMax = 6.")
        println(io)
        println(io, rp("shp", 5), rp("f", 12), rp("D->-D", 10), rp("transpose", 11), rp("reflect", 10),
                rp("permute", 10), "homog(3.7)")
        for (tg, s) in (("c32", C32), ("c4", C4), ("sl", SL)), f in FRQ
            fs = farSetup(s, f; nBlk = 64)
            seed = 20260907
            ds = NTuple{3,Int}[]
            st = seed
            while length(ds) < 50
                st = (1103515245 * st + 12345) % 2147483648
                D = ((st >> 3) % 81 - 40, (st >> 11) % 81 - 40, (st >> 19) % 81 - 40)
                maximum(abs, D) >= 2 && farRoute(fs, D)[1] != 3 && push!(ds, D)
            end
            e = zeros(5)
            for D in ds
                G = farTensor(D, s, f; fs = fs)
                e[1] = max(e[1], rel(farTensor(.-D, s, f; fs = fs), G))
                e[2] = max(e[2], rel(transpose(G), G))
                for c in 1:3
                    sg = ntuple(d -> d == c ? -1 : 1, 3)
                    Gr = farTensor(ntuple(d -> sg[d] * D[d], 3), s, f; fs = fs)
                    e[3] = max(e[3], rel([Gr[i, j] * sg[i] * sg[j] for i in 1:3, j in 1:3], G))
                end
                if s[1] == s[2] == s[3]
                    for p in ((2, 1, 3), (1, 3, 2), (3, 2, 1), (2, 3, 1), (3, 1, 2))
                        Gp = farTensor(ntuple(d -> D[p[d]], 3), s, f; fs = fs)
                        e[4] = max(e[4], rel([Gp[findfirst(==(i), p), findfirst(==(j), p)]
                                              for i in 1:3, j in 1:3], G))
                    end
                end
            end
            lam = QI(37)//10
            s2 = ntuple(d -> s[d] / lam, 3)
            fsA = farSetup(s, f; L = 20, nMax = 6, nBlk = 64, disk = false)
            fsB = farSetup(s2, f * Float64(lam); L = 20, nMax = 6, nBlk = 64, disk = false)
            nh = 0
            for D in ds
                nh >= 10 && break
                (farRoute(fsA, D)[1] == 1 && farRoute(fsB, D)[1] == 1) || continue
                nh += 1
                e[5] = max(e[5], rel(farTensor(D, s2, f * Float64(lam); fs = fsB) .* Float64(lam)^2,
                                     farTensor(D, s, f; fs = fsA)))
            end
            println(io, rp(tg, 5), rp(string(f), 12), rp(fm(e[1]), 10), rp(fm(e[2]), 11), rp(fm(e[3]), 10),
                    rp(s[1] == s[3] ? fm(e[4]) : "n/a", 10), fm(e[5]))
        end
        println(io)
        G64 = farTensor((5, 1, 0), C32, ComplexF64(1))
        G32 = farTensor((5, 1, 0), C32, ComplexF32(1))
        GB = setprecision(() -> farTensor((5, 1, 0), C32, Complex{BigFloat}(1)), BigFloat, 160)
        println(io, "number type at D = (5,1,0), s = (1/32)^3, f = 1:")
        println(io, "  eltype in -> out: ComplexF64 -> ", eltype(G64), ", ComplexF32 -> ", eltype(G32),
                ", Complex{BigFloat} -> ", eltype(GB), " at ", precision(real(GB[1, 1])), " bits")
        println(io, "  Float32 vs Float64 ", fm(Float64(rel(ComplexF64.(G32), G64))),
                " (eps(Float32) = ", fm(Float64(eps(Float32))), ")")
        println(io, "  Float64 vs BigFloat(160) ", fm(Float64(rel(CB.(G64), CB.(GB)))))
    end
end

# ===== (h) anti-Hermitian positivity ==========================================
function gilaToe(dim, sR, f)
    vol = GlaVol(dim, sR)
    opt = GV.CPUKerOpt{Float64}(f, 48, false, GV.CPU())
    facLst = GV.facPar(); fac = Float64.(GV.cubFac(vol.scl))
    srcGrd = GV.sepGrd(vol, vol, 0)
    toe = zeros(ComplexF64, 3, 3, dim...)
    for crt in CartesianIndices(dim)
        GV.egoFunInn!(toe, crt, srcGrd, vol.scl, fac, fac, facLst, opt)
    end
    wS, wE, wV = GV.wekTrp(vol.scl, opt)
    for pos in CartesianIndices(ntuple(d -> 1:min(dim[d], 2), 3))
        GV.egoFunSng!(view(toe, :, :, pos), pos, wS, wE, wV, srcGrd, vol, fac, fac, facLst, opt)
    end
    for a in 1:3; toe[a, a, 1, 1, 1] -= 1 / f^2; end
    return toe
end
function dense(toe, dim)
    n = prod(dim)
    M = zeros(ComplexF64, 3n, 3n)
    lin = LinearIndices(dim)
    for c1 in CartesianIndices(dim), c2 in CartesianIndices(dim)
        d = (c1[1] - c2[1], c1[2] - c2[2], c1[3] - c2[3])
        sg = ntuple(m -> d[m] < 0 ? -1.0 : 1.0, 3)
        p = lin[c1]; q = lin[c2]
        for a in 1:3, b in 1:3
            M[3(p - 1) + a, 3(q - 1) + b] = toe[a, b, abs(d[1]) + 1, abs(d[2]) + 1, abs(d[3]) + 1] *
                                            sg[a] * sg[b]
        end
    end
    return M
end
function partH()
    tee("h_pos.txt") do io
        println(io, "(h) anti-Hermitian part of a small block, Gila's own build against the same block")
        println(io, "    with every offset of max-norm separation >= 2 replaced by farBlock!.  The")
        println(io, "    contact and touching-shell entries (indices all <= 2) are Gila's in both.")
        println(io, "    M is assembled by egoToeCrc!'s rule and is complex symmetric, so the")
        println(io, "    anti-Hermitian part is Im M.  At real f the operator is lossless and Im M is")
        println(io, "    positive semi-definite: the meaningful number is the most negative eigenvalue.")
        println(io)
        for (tg, dim, sR, sQ) in (("c32", (6, 6, 6), (1//32, 1//32, 1//32), C32),
                                  ("c4", (6, 6, 6), (1//4, 1//4, 1//4), C4),
                                  ("sl", (6, 6, 12), (1//32, 1//32, 1//512), SL))
            for f in FRQ
                tc = time()
                tA = gilaToe(dim, sR, f)
                tB = copy(tA)
                farBlock!(tB, sQ, f)
                dmx = 0.0; doff = (0, 0, 0)
                for i in 1:dim[1], j in 1:dim[2], k in 1:dim[3]
                    max(i, j, k) >= 3 || continue
                    d = maximum(abs, view(tA, :, :, i, j, k) .- view(tB, :, :, i, j, k)) /
                        maximum(abs, view(tA, :, :, i, j, k))
                    d > dmx && (dmx = d; doff = (i - 1, j - 1, k - 1))
                end
                println(io, tg, "  dim ", dim, "  s ", sR, "  f ", f,
                        "   max |Gila - farBlock!| / max entry = ", fm(dmx), " at ", dstr(doff))
                println(io, "  Gila build (wekTrp at intOrd 48 + egoFunInn! + egoFunSng!) ",
                        round(time() - tc, digits = 1), " s")
                for (nm, t) in (("gila", tA), ("farfield", tB))
                    M = dense(t, dim)
                    sy = maximum(abs, M .- transpose(M)) / maximum(abs, M)
                    ev = eigvals(Hermitian((M .- M') ./ (2im)))
                    println(io, "  ", rp(nm, 10), "symmetry ", fm(sy), "  lam_min ",
                            @sprintf("%.6e", ev[1]), "  lam_max ", @sprintf("%.6e", ev[end]),
                            "  eps*lam_max ", fm(eps() * ev[end]), "  negatives ",
                            count(<(0), ev), "/", length(ev))
                end
                println(io)
            end
        end
    end
end

# ===== (i) cost ===============================================================
function partI()
    tee("i_cost.txt") do io
        println(io, "(i) cost of one offset against L, single-threaded, zero allocations.  terms =")
        println(io, "    costWhl(L), the complex multiply-adds issued by route (i); the h_l and Y_lm")
        println(io, "    recursions and (for real f) Miller's downward j_l are inside the timing.")
        println(io)
        println(io, rp("shp", 5), rp("f", 12), rp("L", 5), rp("terms", 8), rp("ns/offset", 11), "ns/term")
        for (tg, s) in (("c32", C32), ("c4", C4)), f in FRQ
            fs = farSetup(s, f; nBlk = 64)
            ws = FarWs(fs.L, Float64)
            G = zeros(ComplexF64, 3, 3)
            Rv = ntuple(d -> 5.0 * Float64(s[d]), 3)
            for L in (4, 8, 12, 16, 20, 30, 40, 56)
                t = @belapsed tnsWhl!($G, $fs, $ws, $Rv, $L) samples = 400 evals = 20
                c = costWhl(fs.whl, L)
                println(io, rp(tg, 5), rp(string(f), 12), rp(L, 5), rp(c, 8),
                        rp(round(1e9 * t, digits = 1), 11), round(1e9 * t / c, digits = 2))
            end
        end
        println(io)
        println(io, "# route (ii), the eight octants, at the offsets that need it")
        println(io, rp("shp", 5), rp("f", 12), rp("D", 11), rp("max L_j", 9), rp("terms", 9),
                rp("ns/offset", 11), "ns/term")
        for (tg, s) in (("c32", C32), ("c4", C4)), f in (FRQ[1],)
            fs = farSetup(s, f; nBlk = 64)
            ws = FarWs(fs.L, Float64)
            G = zeros(ComplexF64, 3, 3)
            for D in ((2, 0, 0), (2, 1, 1))
                Rv = ntuple(d -> Float64(D[d]) * Float64(s[d]), 3)
                Lc = boundLoct(fs, Rv)
                any(<(0), Lc) && continue
                t = @belapsed tnsOct!($G, $fs, $ws, $Rv, $Lc) samples = 100 evals = 2
                c = costOct(fs.oct, Lc)
                println(io, rp(tg, 5), rp(string(f), 12), rp(dstr(D), 11), rp(maximum(Lc), 9), rp(c, 9),
                        rp(round(1e9 * t, digits = 1), 11), round(1e9 * t / c, digits = 2))
            end
        end
        println(io)
        println(io, "# terms summed per offset over a 32^3 octant (offsets with max index >= 3)")
        println(io, rp("shp", 5), rp("f", 12), rp("offsets", 10), rp("total terms", 14),
                rp("mean/offset", 12), "histogram terms:offsets")
        for (tg, s) in SHP, f in FRQ
            fs = farSetup(s, f; nBlk = 32)
            trm = 0; nOf = 0; hs = Dict{Int,Int}()
            for i3 in 1:32, i2 in 1:32, i1 in 1:32
                max(i1, i2, i3) >= 3 || continue
                kind, L, Lc, c = farRoute(fs, (i1 - 1, i2 - 1, i3 - 1); cst = true)
                nOf += 1; trm += c
                k = kind == 1 ? c : -kind
                hs[k] = get(hs, k, 0) + 1
            end
            println(io, rp(tg, 5), rp(string(f), 12), rp(nOf, 10), rp(trm, 14),
                    rp(round(trm / nOf, digits = 1), 12),
                    join([(k < 0 ? "route$(-k)" : string(k)) * ":$(hs[k])" for k in sort(collect(keys(hs)))], " "))
        end
        println(io)
        println(io, "# farBlock! over a whole block, single-threaded (the k-series offsets are read")
        println(io, "# from ksrcache/ before the timed call)")
        println(io, rp("shp", 5), rp("f", 12), rp("N", 6), rp("offsets", 10), rp("time s", 10), "ns/offset")
        for (tg, s, N) in (("c32", C32, 32), ("c32", C32, 64), ("c4", C4, 32), ("sl", SL, 32)),
            f in (FRQ[1],)
            fs = farSetup(s, f; nBlk = N)
            toe = zeros(ComplexF64, 3, 3, N, N, N)
            farBlock!(toe, s, f; fs = fs)
            t = @elapsed farBlock!(toe, s, f; fs = fs)
            t = min(t, @elapsed farBlock!(toe, s, f; fs = fs))
            nOf = count(i -> maximum(Tuple(i)) >= 3, CartesianIndices((N, N, N)))
            println(io, rp(tg, 5), rp(string(f), 12), rp(N, 6), rp(nOf, 10),
                    rp(round(t, digits = 4), 10), round(1e9 * t / nOf, digits = 1))
        end
    end
end

# ===== (j)-(m) unequal cells (cross-scale) ===================================
# The 220-bit unequal-cell references (agent xref, rule B ord 32 / rule A ordN 44; key
# vol|R=..|sT=..|sS=..|f=re;im|p=220) are read from refcache/reftensors_x.txt, then from the
# scratch cache; the first file that holds a key wins.
const XREFS = (joinpath(@__DIR__, "refcache", "reftensors_x.txt"),
               joinpath(@__DIR__, "scratch", "xwork", "xref", "reftensors_x.txt"))
"Every 'vol' record of the unequal-cell caches, keyed (R, sT, sS, (Re f, Im f))."
function loadRefX()
    R = Dict{Tuple{NTuple{3,QI},NTuple{3,QI},NTuple{3,QI},NTuple{2,Float64}},Matrix{CB}}()
    for fn in XREFS
        isfile(fn) || continue
        for ln in eachline(fn)
            (isempty(strip(ln)) || startswith(ln, "#")) && continue
            p = split(strip(ln), '|')
            (length(p) == 7 && p[1] == "vol" && p[6] == "p=220") || continue
            v = split(p[7]); length(v) == 9 || continue
            Rq = Tuple(prsQ.(split(p[2][3:end], ',')))
            sT = Tuple(prsQ.(split(p[3][4:end], ','))); sS = Tuple(prsQ.(split(p[4][4:end], ',')))
            fr = parse.(Float64, split(p[5][3:end], ';'))
            ky = (Rq, sT, sS, (fr[1], fr[2]))
            haskey(R, ky) && continue
            R[ky] = reshape([CB(parse(BigFloat, x[1]), parse(BigFloat, x[2])) for x in split.(v, ';')], 3, 3)
        end
    end
    isempty(R) && error("no unequal-cell reference cache (reftensors_x.txt)")
    R
end
const GQ = (QI(1)//32, QI(1)//32, QI(1)//32)
"Pair label sS/g x sT/g in gcd cells, e.g. 4x1x1;1x1x1."
xPair(sT, sS) = (gq = min.(sT, sS); string(join(string.(Int.(sS ./ gq)), "x"), ";", join(string.(Int.(sT ./ gq)), "x")))
"R in units of the gcd cell, 1 decimal."
xRg(R, sT, sS) = (gq = min.(sT, sS); join(string.(round.(Float64.(R ./ gq), digits = 1)), ","))
bitsEq(A, B) = all(reinterpret(UInt64, real(A[i])) == reinterpret(UInt64, real(B[i])) &&
                   reinterpret(UInt64, imag(A[i])) == reinterpret(UInt64, imag(B[i])) for i in 1:9)
absErr(G, Gr) = Float64(maximum(abs(CB(G[i]) - Gr[i]) for i in 1:9))
digs(e) = e <= 0 ? -Inf : log10(e / eps(Float64))
med(v) = (s = sort(v); n = length(s); n == 0 ? NaN : isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2)
fmt1(x) = @sprintf("%.1f", x)
"Reflection identity: G_ab(sigma R) = sigma_a sigma_b G_ab(R) over the axis flips and -R; worst relative deviation."
function flipChk(fx, R, sT, sS, f, G)
    mg = maximum(abs, G); w = 0.0
    for sg in ((-1, 1, 1), (1, -1, 1), (1, 1, -1), (-1, -1, -1))
        (sg != (-1, -1, -1) && any(sg[d] < 0 && R[d] == 0 for d in 1:3)) && continue
        Gf = farTensorX(ntuple(d -> sg[d] * R[d], 3), sT, sS, f; fs = fx)
        w = max(w, maximum(abs(Gf[a, b] - sg[a] * sg[b] * G[a, b]) for a in 1:3, b in 1:3) / mg)
    end
    w
end

function partJ()
    RX = loadRefX()
    grp = Dict{Tuple{NTuple{3,QI},NTuple{3,QI},NTuple{2,Float64}},Vector{NTuple{3,QI}}}()
    for k in keys(RX); push!(get!(grp, (k[2], k[3], k[4]), NTuple{3,QI}[]), k[1]); end
    gk = sort(collect(keys(grp)); by = k -> (k[3], Float64(prod(k[2])), Float64.(k[2]), Float64.(k[1])))
    tee("j_xref.txt") do io
        println(io, "(j) farTensorX against every cached 220-bit unequal-cell reference (", length(RX),
                " records).  pair = sS/g;sT/g in gcd cells g = min(sT, sS) per axis; R/g the target centre")
        println(io, "    (source at 0); rt = route (1 whole trapezoid box, 2 gcd box sum, 3 k-series), L = whole-box")
        println(io, "    L; eMx = max_ab |G - Gr| / max |Gr|; pEn = worst per entry (entries above 1e-8 of the largest);")
        println(io, "    dig = log10(pEn / eps); c/G = certificate / max |G| (cert = true); c/e = certificate / actual")
        println(io, "    absolute error (< 1 is a VIOLATION, column viol); srt/sid/seMx = route of the swapped orientation")
        println(io, "    G(-R; sS, sT) (coarse target), |G - (Vs/Vt) G_swap| / max |G|, and (Vs/Vt) G_swap vs Gr; flip =")
        println(io, "    worst |G_ab(sigma R) - sigma_a sigma_b G_ab(R)| / max |G| over the axis reflections and -R;")
        println(io, "    bw = farTensorX bitwise equal to farBlockX! on the group with the same certificate.")
        println(io, "    load average at start ", Sys.loadavg(), "; threads ", Threads.nthreads())
        println(io)
        println(io, rp("pair", 16), rp("f", 9), rp("R/g", 17), rp("rt", 3), rp("L", 4), rp("eMx", 10), rp("pEn", 10),
                rp("dig", 6), rp("c/G", 10), rp("c/e", 9), rp("viol", 5), rp("srt", 4), rp("sid", 10), rp("seMx", 10),
                rp("flip", 10), "bw")
        acc = Dict(k => (Float64[], Float64[], Float64[], Float64[], Float64[], Float64[], Int[]) for k in 1:3)
        gsum = []
        nrec = 0; nbw = 0; nviol = 0
        for k in gk
            sT, sS, fr = k; f = ComplexF64(fr[1], fr[2])
            Rs = sort(grp[k]; by = R -> (sqrt(sum(Float64(R[i])^2 for i in 1:3)), Float64.(R)))
            n = length(Rs)
            t0 = time(); fx = farSetupX(sT, sS, f; offs = Rs); tS = time() - t0
            Gb = zeros(ComplexF64, 3, 3, n)
            t0 = time(); rt, cs = farBlockX!(Gb, Rs, sT, sS, f; fs = fx, cert = true); tB = time() - t0
            Rm = [ntuple(d -> -R[d], 3) for R in Rs]
            t0 = time(); fx2 = farSetupX(sS, sT, f; offs = Rm); tS2 = time() - t0
            rat = Float64(prod(sS) / prod(sT))
            w = zeros(6); nv = 0; nb = 0; rts = zeros(Int, 3); ts3 = 0.0
            for (i, R) in enumerate(Rs)
                Gr = RX[(R, sT, sS, fr)]
                t0 = time(); G, c = farTensorX(R, sT, sS, f; fs = fx, cert = true); t1 = time() - t0
                kind, L = farRouteX(fx, R); rts[kind] += 1; kind == 3 && (ts3 += t1)
                e = errs(G, Gr); ae = absErr(G, Gr); mg = maximum(abs, G)
                viol = c < ae; bw = bitsEq(G, view(Gb, :, :, i)) && c == cs[i]
                G2, c2 = farTensorX(Rm[i], sS, sT, f; fs = fx2, cert = true); k2, _ = farRouteX(fx2, Rm[i])
                Gs = rat .* G2; sid = maximum(abs, G .- Gs) / mg; es = errs(Gs, Gr)
                viol2 = rat * c2 < absErr(Gs, Gr)
                fl = flipChk(fx, R, sT, sS, f, G)
                a = acc[kind]
                push!(a[1], e[1]); push!(a[2], e[2]); push!(a[3], c / mg); push!(a[4], c / ae)
                push!(a[5], sid); push!(a[6], fl); push!(a[7], viol + viol2)
                w[1] = max(w[1], e[1]); w[2] = max(w[2], e[2]); w[3] = max(w[3], sid); w[4] = max(w[4], es[1])
                w[5] = max(w[5], fl); w[6] = max(w[6], c / mg)
                nv += viol + viol2; nb += bw; nrec += 1; nbw += bw; nviol += viol + viol2
                println(io, rp(xPair(sT, sS), 16), rp(string(fr[1], fr[2] == 0 ? "" : string("+", fr[2], "i")), 9),
                        rp(xRg(R, sT, sS), 17), rp(kind, 3), rp(L, 4), rp(fm(e[1]), 10), rp(fm(e[2]), 10),
                        rp(fmt1(digs(e[2])), 6), rp(fm(c / mg), 10), rp(fmt1(c / ae), 9),
                        rp(viol ? "VIOL" : (viol2 ? "sVIOL" : ""), 5), rp(k2, 4), rp(fm(sid), 10), rp(fm(es[1]), 10),
                        rp(fm(fl), 10), bw ? "y" : "N")
            end
            push!(gsum, (xPair(sT, sS), fr, n, rts, w, nv, nb, tS, tB, tS2, ts3, fx.Lw, maximum(fx.nCut)))
            flush(io)
        end
        println(io)
        println(io, "# per (pair, f): n, routes 1/2/3, worst eMx, pEn, dig, swap identity, swap vs ref, flip,")
        println(io, "# worst c/G, violations (direct + swapped), bitwise farTensorX == farBlockX!, setup s (direct,")
        println(io, "# swapped), farBlockX! s, k-series s (farTensorX route 3, direct), table Lw, nCut max")
        println(io, rp("pair", 16), rp("f", 9), rp("n", 4), rp("r1/2/3", 10), rp("eMx", 10), rp("pEn", 10), rp("dig", 6),
                rp("sid", 10), rp("seMx", 10), rp("flip", 10), rp("c/G", 10), rp("viol", 5), rp("bw", 6),
                rp("setup", 12), rp("blk", 7), rp("ksr", 8), rp("Lw", 4), "nCut")
        for s in gsum
            println(io, rp(s[1], 16), rp(string(s[2][1], s[2][2] == 0 ? "" : string("+", s[2][2], "i")), 9), rp(s[3], 4),
                    rp(join(s[4], "/"), 10), rp(fm(s[5][1]), 10), rp(fm(s[5][2]), 10), rp(fmt1(digs(s[5][2])), 6),
                    rp(fm(s[5][3]), 10), rp(fm(s[5][4]), 10), rp(fm(s[5][5]), 10), rp(fm(s[5][6]), 10), rp(s[6], 5),
                    rp(string(s[7], "/", s[3]), 6), rp(string(fmt1(s[8]), ",", fmt1(s[10])), 12), rp(fmt1(s[9]), 7),
                    rp(fmt1(s[11]), 8), rp(s[12], 4), s[13])
        end
        println(io)
        println(io, "# per route over all records: n, eMx median / worst, pEn median / worst, dig worst,")
        println(io, "# c/G min / median / max, c/e min / median / max, swap identity worst, flip worst, violations")
        for kd in 1:3
            a = acc[kd]; isempty(a[1]) && continue
            println(io, "route ", kd, ": n ", length(a[1]), "  eMx ", fm(med(a[1])), " / ", fm(maximum(a[1])),
                    "  pEn ", fm(med(a[2])), " / ", fm(maximum(a[2])), "  dig ", fmt1(digs(maximum(a[2]))),
                    "  c/G ", fm(minimum(a[3])), " / ", fm(med(a[3])), " / ", fm(maximum(a[3])),
                    "  c/e ", fmt1(minimum(a[4])), " / ", fmt1(med(a[4])), " / ", fmt1(maximum(a[4])),
                    "  sid ", fm(maximum(a[5])), "  flip ", fm(maximum(a[6])), "  viol ", sum(a[7]))
        end
        println(io)
        println(io, "records ", nrec, "; worst eMx ", fm(maximum(maximum(acc[k][1]) for k in 1:3 if !isempty(acc[k][1]))),
                "; worst pEn ", fm(maximum(maximum(acc[k][2]) for k in 1:3 if !isempty(acc[k][2]))),
                "; certificate violations ", nviol, " (direct and swapped, of ", 2nrec, "); farTensorX == farBlockX! ",
                nbw, " of ", nrec, "; load ", Sys.loadavg())
    end
end

using Random
if PART == "m"
    @eval using GilaElectromagnetics
    @eval using StaticArrays
    @eval const GV = GilaElectromagnetics.GilaVacuum
end

# ===== (k) unequal cells, adversarial geometries ==============================
emsg(e) = (s = sprint(showerror, e); replace(s[1:min(end, 170)], '\n' => ' '))
fstr(f) = string(real(f), imag(f) == 0 ? "" : string("+", imag(f), "i"))
"One group (sT, sS, f) at the offsets Rs: farSetupX, then per offset farTensorX (cert), route, lattice m, error vs the cached reference, swap and reflection identities.  Returns (n, nref, worst eMx, worst pEn, worst sid, worst flip, violations)."
function xGroup(io, RX, tag, sT, sS, f, Rs; swap::Bool = true, flip::Bool = true)
    fr = (real(f), imag(f)); w = zeros(4); nv = 0; nref = 0
    t0 = time()
    fx = try
        farSetupX(sT, sS, f; offs = Rs)
    catch e
        println(io, rp(tag, 14), "farSetupX ERROR: ", emsg(e)); return (0, 0, w..., 0)
    end
    tS = time() - t0
    fx2 = swap ? (try farSetupX(sS, sT, f; offs = [ntuple(d -> -R[d], 3) for R in Rs]) catch e; nothing end) : nothing
    rat = Float64(prod(sS) / prod(sT))
    println(io, rp(tag, 14), "pair ", xPair(sT, sS), " f ", fstr(f), "  b/g ", join(string.(round.(Float64.(fx.tbQ ./ min.(sT, sS)), digits = 2)), ","),
            "  nT ", fx.nT, " nS ", fx.nS, "  Lw ", fx.Lw, " nCut max ", maximum(fx.nCut), "  setup ", fmt1(tS), " s")
    for R in Rs
        kl = try
            farRouteX(fx, R)
        catch e
            println(io, rp(tag, 14), rp(xRg(R, sT, sS), 17), "farRouteX ERROR: ", emsg(e)); continue
        end
        kind, L = kl
        t0 = time()
        G, c = try
            farTensorX(R, sT, sS, f; fs = fx, cert = true)
        catch e
            println(io, rp(tag, 14), rp(xRg(R, sT, sS), 17), rp(kind, 3), "farTensorX ERROR: ", emsg(e)); continue
        end
        t1 = time() - t0
        mg = maximum(abs, G); m = xLat(fx, R); ms = m === nothing ? "off" : join(m, ",")
        Gr = get(RX, (R, sT, sS, fr), nothing)
        e = (NaN, NaN, NaN, NaN); ae = NaN; viol = false
        if Gr !== nothing
            e = errs(G, Gr); ae = absErr(G, Gr); viol = c < ae; nref += 1
            w[1] = max(w[1], e[1]); w[2] = max(w[2], e[2]); nv += viol
        end
        sid = NaN; k2 = -1; es1 = NaN
        if fx2 !== nothing
            try
                G2, c2 = farTensorX(ntuple(d -> -R[d], 3), sS, sT, f; fs = fx2, cert = true)
                k2, _ = farRouteX(fx2, ntuple(d -> -R[d], 3))
                Gs = rat .* G2; sid = maximum(abs, G .- Gs) / mg; w[3] = max(w[3], sid)
                Gr === nothing || (es1 = errs(Gs, Gr)[1]; nv += rat * c2 < absErr(Gs, Gr))
            catch e2
                println(io, rp(tag, 14), rp(xRg(R, sT, sS), 17), "swap ERROR: ", emsg(e2))
            end
        end
        fl = flip ? flipChk(fx, R, sT, sS, f, G) : NaN
        flip && (w[4] = max(w[4], fl))
        nfo = kind == 3 && haskey(fx.ksrI, R) ? @sprintf("N %d Lam %.2g", fx.ksrI[R]...) : ""
        println(io, rp(tag, 14), rp(xRg(R, sT, sS), 17), rp(kind, 3), rp(L, 4), rp(ms, 12),
                rp(Gr === nothing ? "noref" : fm(e[1]), 10), rp(Gr === nothing ? "" : fm(e[2]), 10),
                rp(Gr === nothing ? "" : fmt1(digs(e[2])), 6), rp(fm(c / mg), 10),
                rp(Gr === nothing ? "" : fmt1(c / ae), 9), rp(viol ? "VIOL" : "", 5), rp(k2, 4), rp(fm(sid), 10),
                rp(fm(es1), 10), rp(fm(fl), 10), rp(@sprintf("%.2e", t1), 10), rp(@sprintf("%.6e", real(G[1, 1])), 14), nfo)
        flush(io)
    end
    (length(Rs), nref, w..., nv)
end

function partK()
    RX = loadRefX()
    g = GQ[1]; z = QI(0); f1 = ComplexF64(1)
    tot = Float64[0, 0, 0, 0]; cnt = [0, 0, 0]
    acc(r) = (cnt[1] += r[1]; cnt[2] += r[2]; tot .= max.(tot, r[3:6]); cnt[3] += r[7])
    tee("k_adv.txt") do io
        println(io, "(k) adversarial unequal-cell geometries.  Columns as in (j): R/g in gcd cells, rt/L route and")
        println(io, "    whole-box L, m = gcd-lattice coordinate (off = not on the lattice), eMx/pEn/dig vs the 220-bit")
        println(io, "    reference (noref = none cached), c/G c/e viol certificate, srt/sid/seMx swapped orientation,")
        println(io, "    flip reflection identity, t farTensorX seconds (fs held), G11 = Re G_11, N/Lam of route 3.")
        println(io, "    load average at start ", Sys.loadavg(), "; threads ", Threads.nthreads())
        println(io)
        hdr() = println(io, rp("case", 14), rp("R/g", 17), rp("rt", 3), rp("L", 4), rp("m", 12), rp("eMx", 10), rp("pEn", 10),
                        rp("dig", 6), rp("c/G", 10), rp("c/e", 9), rp("viol", 5), rp("srt", 4), rp("sid", 10), rp("seMx", 10),
                        rp("flip", 10), rp("t s", 10), rp("Re G11", 14), "ksr")
        # (1) odd ratios 3 and 6 on one and three axes: R/g is an integer (nS - nT even), so the lattice test differs
        println(io, "# (1) odd ratios 3 and 6 (target g, coarse source at 0)"); hdr()
        acc(xGroup(io, RX, "r3x", GQ, (3g, g, g), f1, [(2g + g, z, z), (2g + 2g, z, z), (3g, g, z)]))
        acc(xGroup(io, RX, "r3xyz", GQ, (3g, 3g, 3g), f1, [(3g, z, z), (4g, z, z), (3g, g, g), (3g, 3g, z)]))
        acc(xGroup(io, RX, "r6x", GQ, (6g, g, g), f1, [(QI(9)//2 * g, z, z), (QI(15)//2 * g, z, z)]))
        acc(xGroup(io, RX, "r6xyz", GQ, (6g, 6g, 6g), f1, [(QI(9)//2 * g, z, z), (QI(9)//2 * g, QI(5)//2 * g, QI(5)//2 * g), (QI(9)//2 * g, QI(9)//2 * g, QI(5)//2 * g)]))
        # (2) mixed pair: target coarser on x, source coarser on y; both orientations (the swap columns and the swapped group)
        println(io); println(io, "# (2) mixed pair sT = (4g, g, g), sS = (g, 2g, g), b = (2.5, 1.5, 1) g; then the swapped pair as its own group"); hdr()
        sTm = (4g, g, g); sSm = (g, 2g, g)
        Rm = [(QI(5)//2 * g + g, g//2, z), (QI(5)//2 * g + 2g, g//2, z), (QI(5)//2 * g + 4g, g//2, z), (QI(7)//2 * g, z, z),
              (g//2, QI(3)//2 * g + g, z), (g//2, QI(3)//2 * g + 2g, z), (QI(7)//2 * g, QI(5)//2 * g, z), (QI(7)//2 * g, g//2, g)]
        acc(xGroup(io, RX, "mix", sTm, sSm, f1, Rm))
        acc(xGroup(io, RX, "mixswap", sSm, sTm, f1, [ntuple(d -> -R[d], 3) for R in Rm]))
        # (3) non-integer ratios: must refuse clearly
        println(io); println(io, "# (3) non-integer ratios (3/64 vs 1/32 on x: ratio 3/2, gcd 1/64; 1/32 vs 1/48: ratio 3/2, gcd 1/96)")
        for (nm, sT, sS, R) in (("ni3_64", (QI(3)//64, g, g), GQ, (QI(5)//128 + QI(1)//64, z, z)),
                                ("ni48", GQ, (QI(1)//48, QI(1)//48, QI(1)//48), (QI(5)//192 + QI(1)//96, QI(1)//192, QI(1)//192)))
            for (what, fn) in (("farSetupX", () -> farSetupX(sT, sS, f1)), ("farTensorX", () -> farTensorX(R, sT, sS, f1)),
                               ("farBlockX!", () -> farBlockX!(zeros(ComplexF64, 3, 3, 1), [R], sT, sS, f1)))
                try
                    r = fn(); println(io, rp(nm, 14), rp(what, 12), "NO ERROR: returned ", typeof(r))
                catch e
                    println(io, rp(nm, 14), rp(what, 12), "error: ", emsg(e))
                end
            end
            haskey(RX, (R, sT, sS, (1.0, 0.0))) && println(io, rp(nm, 14), "a 220-bit reference exists for R/g(gcd) = ", xRg(R, sT, sS), ": Re G11 = ", @sprintf("%.6e", Float64(real(RX[(R, sT, sS, (1.0, 0.0))][1, 1]))))
        end
        # (4) lambda/2 and lambda coarse cells at f = 1 against the fine lambda/32 cell
        println(io); println(io, "# (4) lambda/2 (16 g) and lambda (32 g) coarse cells, f = 1; the lambda-cube lateral-0 offsets are off the lattice (nS even) -> route 3"); hdr()
        acc(xGroup(io, RX, "l2rod", GQ, (16g, g, g), f1, [(QI(17)//2 * g + k * g, z, z) for k in (1, 2, 4, 8, 32)]; flip = false))
        acc(xGroup(io, RX, "l2cube", GQ, (16g, 16g, 16g), f1, [(QI(17)//2 * g + k * g, QI(15)//2 * g, QI(15)//2 * g) for k in (1, 8, 32)]; flip = false))
        acc(xGroup(io, RX, "l2cube_off", GQ, (16g, 16g, 16g), f1, [(QI(17)//2 * g + k * g, z, z) for k in (1, 32)]; flip = false, swap = false))
        acc(xGroup(io, RX, "lamrod", GQ, (QI(1), g, g), f1, [(QI(33)//64 + k * g, z, z) for k in (1, 4, 32)]; flip = false))
        acc(xGroup(io, RX, "lamcube", GQ, (QI(1), QI(1), QI(1)), f1, [(QI(33)//64 + g, QI(31)//64, QI(31)//64), (QI(33)//64 + 32g, z, z)]; flip = false, swap = false))
        # (5) complex f
        println(io); println(io, "# (5) complex f = 1+i and 0.5+2i on r4x and r4xyz"); hdr()
        for f in (1.0 + 1.0im, 0.5 + 2.0im)
            acc(xGroup(io, RX, "r4x_" * fstr(f), GQ, (4g, g, g), f, [(QI(5)//2 * g + k * g, z, z) for k in (1, 2, 4, 8)]))
            acc(xGroup(io, RX, "r4xyz_" * fstr(f), GQ, (4g, 4g, 4g), f, [(QI(5)//2 * g + k * g, QI(3)//2 * g, QI(3)//2 * g) for k in (1, 2, 8)]))
            acc(xGroup(io, RX, "r4xyz_" * fstr(f), GQ, (4g, 4g, 4g), f, [(QI(7)//2 * g, z, z)]))
        end
        # (6) offsets with negative and zero components mixed
        println(io); println(io, "# (6) mixed-sign offsets (new references; the flip column compares with the mirrored evaluation)"); hdr()
        acc(xGroup(io, RX, "neg_r4xyz", GQ, (4g, 4g, 4g), f1, [(-QI(7)//2 * g, QI(3)//2 * g, -QI(3)//2 * g), (-QI(11)//2 * g, -QI(3)//2 * g, QI(3)//2 * g), (QI(3)//2 * g, -QI(7)//2 * g, -QI(3)//2 * g)]))
        acc(xGroup(io, RX, "neg_r16x", GQ, (16g, g, g), f1, [(g//2, 2g, z), (-QI(19)//2 * g, z, z), (-QI(19)//2 * g, -g, z), (z, -3g, z)]))
        # (7) Float32
        println(io); println(io, "# (7) Float32: farTensorX with f::ComplexF32 vs the Float64 result (and the reference where cached); farBlockX! in Float32 bitwise vs farTensorX")
        println(io, rp("case", 14), rp("R/g", 17), rp("rt", 3), rp("|G32-G64|/|G|", 15), rp("G32 vs ref", 12), rp("G64 vs ref", 12), rp("c32/G", 10), rp("c32/e32", 9), "blk==tns")
        f32 = ComplexF32(1); w32 = 0.0
        for (tag, sS, Rs) in (("f32_r4x", (4g, g, g), [(QI(5)//2 * g + k * g, z, z) for k in (1, 2, 3, 4, 8, 32)]),
                              ("f32_r4xyz", (4g, 4g, 4g), [(QI(7)//2 * g, z, z), (QI(7)//2 * g, QI(3)//2 * g, QI(3)//2 * g), (QI(13)//2 * g, QI(3)//2 * g, QI(3)//2 * g), (QI(7)//2 * g, QI(7)//2 * g, QI(3)//2 * g)]))
            fx32 = farSetupX(GQ, sS, f32; offs = Rs); fx64 = farSetupX(GQ, sS, f1; offs = Rs)
            Gb = zeros(ComplexF32, 3, 3, length(Rs)); farBlockX!(Gb, Rs, GQ, sS, f32; fs = fx32)
            for (i, R) in enumerate(Rs)
                G32, c32 = farTensorX(R, GQ, sS, f32; fs = fx32, cert = true); G64 = farTensorX(R, GQ, sS, f1; fs = fx64)
                kind, _ = farRouteX(fx32, R)
                d = Float64(maximum(abs, ComplexF64.(G32) .- G64) / maximum(abs, G64)); w32 = max(w32, d)
                Gr = get(RX, (R, GQ, sS, (1.0, 0.0)), nothing)
                e32 = Gr === nothing ? NaN : errs(ComplexF64.(G32), Gr)[1]; e64 = Gr === nothing ? NaN : errs(G64, Gr)[1]
                ce = Gr === nothing ? NaN : c32 / absErr(ComplexF64.(G32), Gr)
                Gv = view(Gb, :, :, i)
                bw = all(reinterpret(UInt32, real(G32[j])) == reinterpret(UInt32, real(Gv[j])) && reinterpret(UInt32, imag(G32[j])) == reinterpret(UInt32, imag(Gv[j])) for j in 1:9)
                println(io, rp(tag, 14), rp(xRg(R, GQ, sS), 17), rp(kind, 3), rp(fm(d), 15), rp(fm(e32), 12), rp(fm(e64), 12),
                        rp(fm(c32 / Float64(maximum(abs, G32))), 10), rp(fmt1(ce), 9), bw ? "y" : "N", "  eltype ", eltype(G32))
            end
        end
        println(io, "worst |G32 - G64| / max|G| ", fm(w32), " (eps(Float32) = ", fm(Float64(eps(Float32))), ")")
        # (8) a FrqSetX reused outside the offsets it was built for
        println(io); println(io, "# (8) FrqSetX built for other offsets: near-only set (nBlk 4, Lw = LDEF) asked for far offsets; far-only set asked for a near offset")
        let sS = (4g, g, g), Rn = (QI(5)//2 * g + g, z, z), Rf = (QI(5)//2 * g + 32g, z, z), Rm = (QI(5)//2 * g + 4g, z, z), Rvf = (QI(5)//2 * g + 100g, z, z)
            fxn = farSetupX(GQ, sS, f1; offs = (Rn,), nBlk = 4)
            println(io, "near-only set: Lw ", fxn.Lw, " thr[1] (radius of L = 0) / g ", round(fxn.thr[1] / Float64(g), digits = 2), " rHi / g ", round(4 * 4 * sqrt(sum(fxn.b6 .^ 2)) / Float64(g), digits = 1))
            for (nm, R) in (("kap 4", Rm), ("kap 32", Rf), ("kap 100 (beyond rHi)", Rvf))
                try
                    kind, L = farRouteX(fxn, R); G = farTensorX(R, GQ, sS, f1; fs = fxn)
                    Gr = get(RX, (R, GQ, sS, (1.0, 0.0)), nothing)
                    G0 = farTensorX(R, GQ, sS, f1)
                    println(io, rp(nm, 22), "route ", kind, " L ", L, "  vs fresh set ", fm(maximum(abs, G .- G0) / maximum(abs, G0)),
                            Gr === nothing ? "  (no reference)" : string("  vs ref ", fm(errs(G, Gr)[1])))
                catch e
                    println(io, rp(nm, 22), "error: ", emsg(e))
                end
            end
            fxf = farSetupX(GQ, sS, f1; offs = (Rf,))
            G = farTensorX(Rn, GQ, sS, f1; fs = fxf); Gr = RX[(Rn, GQ, sS, (1.0, 0.0))]
            println(io, rp("far-only set, kap 1", 22), "route ", farRouteX(fxf, Rn)[1], "  vs ref ", fm(errs(G, Gr)[1]), "  fine set nBlk ", fxf.nBlk)
        end
        # (9) touching, overlapping and barely separated pairs
        println(io); println(io, "# (9) touching / overlapping / barely separated (r2x pair, b = (1.5, 1, 1) g)")
        let sS = (2g, g, g), fx = farSetupX(GQ, sS, f1)
            for (nm, R) in (("overlap R = 0", (z, z, z)), ("face touch |Rx| = b", (QI(3)//2 * g, z, z)), ("edge touch", (QI(3)//2 * g, g, z)),
                            ("corner touch", (-QI(3)//2 * g, -g, g)), ("inside on x, out on y", (g, 2g, z)))
                try
                    println(io, rp(nm, 24), "route ", farRouteX(fx, R))
                catch e
                    println(io, rp(nm, 24), "error: ", emsg(e))
                end
            end
            R = (QI(3)//2 * g + g // 1000, z, z)
            try
                t0 = time(); kind, L = farRouteX(fx, R); G, c = farTensorX(R, GQ, sS, f1; fs = fx, cert = true)
                println(io, rp("gap g/1000", 24), "route ", kind, "  ", @sprintf("%.1f s", time() - t0), "  Re G11 ", @sprintf("%.6e", real(G[1, 1])), "  c/G ", fm(c / maximum(abs, G)),
                        haskey(fx.ksrI, R) ? @sprintf("  N %d Lam %.2g", fx.ksrI[R]...) : "")
            catch e
                println(io, rp("gap g/1000", 24), "error: ", emsg(e))
            end
        end
        println(io)
        println(io, "offsets evaluated ", cnt[1], " (with reference ", cnt[2], "); worst eMx ", fm(tot[1]), " pEn ", fm(tot[2]), " (", fmt1(digs(tot[2])),
                " digits); swap identity worst ", fm(tot[3]), "; flip worst ", fm(tot[4]), "; certificate violations ", cnt[3], "; load ", Sys.loadavg())
    end
end

# ===== (l) routing and lattice logic, thread determinism ======================
"xgeom's r xyz test block (fine 2r^3 cells of g face to face with a coarse 2^3 of r g): R/g = (k1, k2, k3) + 1/2, k1 in r/2 .. 7r/2 - 1, k2, k3 in -3r/2 .. 3r/2 - 1 (27 r^3 offsets, contact included)."
xBlock(r::Int) = [(QI(2k1 + 1)//64, QI(2k2 + 1)//64, QI(2k3 + 1)//64) for k3 in (-3r÷2):(3r÷2 - 1) for k2 in (-3r÷2):(3r÷2 - 1) for k1 in (r÷2):(7r÷2 - 1)]
xTouch(R, b) = all(abs(R[d]) <= b[d] for d in 1:3)
"Independent lattice coordinate and smallest sub-offset max-norm of R for the pair (sT, sS)."
function xLatChk(R, sT, sS)
    gq = min.(sT, sS); nT = Int.(sT ./ gq); nS = Int.(sS ./ gq)
    m = ntuple(d -> R[d] / gq[d] + QI(nS[d] - nT[d]) // 2, 3)
    all(isinteger, m) || return (nothing, -1)
    mi = ntuple(d -> Int(m[d]), 3)
    # sub-offsets m + t, t in (1 - nS):(nT - 1) per axis: per-axis smallest |coordinate|, then the max over axes
    lo = ntuple(d -> (a = mi[d] + 1 - nS[d]; b = mi[d] + nT[d] - 1; a > 0 ? a : b < 0 ? -b : 0), 3)
    (mi, maximum(lo))
end
function partL()
    g = GQ[1]; f1 = ComplexF64(1); nth = Threads.nthreads()
    tee("l_route.txt") do io
        println(io, "(l) farRouteX on xgeom's r = 2, 4, 8, 16 xyz test blocks (fine g target vs coarse (r g)^3 source at 0;")
        println(io, "    27 r^3 offsets each, all with 2R/g odd).  Independent checks: touching = |R_d| <= b_d on every axis")
        println(io, "    (must raise), lattice m = R/g + (nS - nT)/2 integer (all are), smallest sub-offset max-norm over")
        println(io, "    the N_t N_s pieces (route 2 needs >= 2), and route 3 never (nothing is off the lattice).")
        println(io, "    rho = |b|/|R|.  load average at start ", Sys.loadavg(), "; threads ", nth)
        println(io)
        println(io, rp("r", 3), rp("offsets", 9), rp("touch", 7), rp("raised", 7), rp("r1", 8), rp("r2", 7), rp("r3", 4), rp("offlat", 7),
                rp("m!=xLat", 8), rp("near<2", 7), rp("minNear", 8), rp("rho1 max", 9), rp("rho2 min", 9), rp("L1 max", 7), rp("setup s", 8), rp("route s", 8), "L histogram")
        for r in (2, 4, 8, 16)
            sS = (r * g, r * g, r * g); b = ntuple(d -> (GQ[d] + sS[d]) // 2, 3)
            Rs = xBlock(r); tch = [xTouch(R, b) for R in Rs]
            sep = Rs[.!tch]
            t0 = time(); fx = farSetupX(GQ, sS, f1; offs = sep); tS = time() - t0
            nrs = 0
            for R in Rs[tch]
                try
                    farRouteX(fx, R)
                catch e
                    nrs += 1
                end
            end
            cnt = zeros(Int, 3); off = 0; mism = 0; near2 = 0; mn = typemax(Int); rho1 = 0.0; rho2 = Inf; L1 = 0
            hst = Dict{Int,Int}()
            t0 = time()
            for R in sep
                kind, L = farRouteX(fx, R); cnt[kind] += 1
                m, nr = xLatChk(R, GQ, sS)
                m === nothing && (off += 1)
                m == xLat(fx, R) || (mism += 1)
                rho = sqrt(sum(Float64.(b) .^ 2)) / sqrt(sum(Float64.(R) .^ 2))
                if kind == 1
                    rho1 = max(rho1, rho); L1 = max(L1, L); hst[L] = get(hst, L, 0) + 1
                elseif kind == 2
                    rho2 = min(rho2, rho); nr < 2 && (near2 += 1); mn = min(mn, nr)
                    nr == xNear(fx, m) || (mism += 1)
                end
            end
            tR = time() - t0
            println(io, rp(r, 3), rp(length(Rs), 9), rp(count(tch), 7), rp(nrs, 7), rp(cnt[1], 8), rp(cnt[2], 7), rp(cnt[3], 4), rp(off, 7),
                    rp(mism, 8), rp(near2, 7), rp(mn, 8), rp(round(rho1, digits = 3), 9), rp(round(rho2, digits = 3), 9), rp(L1, 7),
                    rp(fmt1(tS), 8), rp(@sprintf("%.2f", tR), 8), join([string(k, ":", v) for (k, v) in sort(collect(hst))], " "))
            flush(io)
        end
        println(io)
        println(io, "# off-lattice control: r = 4 xyz pair at lateral-0 and integer R/g offsets (not produced by Gila)")
        sS = (4g, 4g, 4g); fx = farSetupX(GQ, sS, f1)
        for R in ((QI(7)//2 * g, QI(0), QI(0)), (QI(9)//2 * g, QI(3)//2 * g, QI(0)), (4g, QI(0), QI(0)), (4g, 2g, 2g), (QI(7)//2 * g, QI(3)//2 * g, QI(1)//2 * g), (QI(13)//2 * g, QI(0), QI(0)))
            kind, L = farRouteX(fx, R); m, nr = xLatChk(R, GQ, sS)
            println(io, rp(xRg(R, GQ, sS), 17), "route ", kind, " L ", rp(L, 4), " lattice ", m === nothing ? "off" : string(m, " near ", nr),
                    kind == 3 && m !== nothing && nr >= 2 ? "  <-- route 3 on a lattice offset" : "")
        end
    end
    # farBlockX! vs farTensorX per offset, and the thread-determinism dump of the r = 4 block
    tee("l_thread_t$(nth).txt") do io
        println(io, "(l) farBlockX! against farTensorX per offset (bitwise), and the r = 4 block dumped for the thread")
        println(io, "    determinism check; threads ", nth, "; load ", Sys.loadavg())
        for r in (2, 4, 8)
            sS = (r * g, r * g, r * g); b = ntuple(d -> (GQ[d] + sS[d]) // 2, 3)
            Rs = [R for R in xBlock(r) if !xTouch(R, b)]
            fx = farSetupX(GQ, sS, f1; offs = Rs)
            G = zeros(ComplexF64, 3, 3, length(Rs))
            farBlockX!(G, Rs, GQ, sS, f1; fs = fx)
            t0 = time(); rt = farBlockX!(G, Rs, GQ, sS, f1; fs = fx); tB = time() - t0
            nb = count(bitsEq(farTensorX(Rs[i], GQ, sS, f1; fs = fx), view(G, :, :, i)) for i in eachindex(Rs))
            println(io, "r ", r, ": ", length(Rs), " offsets, routes ", count(==(1), rt), "/", count(==(2), rt), "/", count(==(3), rt),
                    ", farBlockX! warm ", @sprintf("%.2f", tB), " s, farTensorX bitwise equal on ", nb, " of ", length(Rs))
            if r == 4
                fn = joinpath(DIR, "l_blk_r4_t$(nth).bin")
                open(fn, "w") do bio; write(bio, G); end
                for other in sort(filter(x -> startswith(x, "l_blk_r4_t") && endswith(x, ".bin"), readdir(DIR)))
                    H = Array{ComplexF64}(undef, size(G)); read!(joinpath(DIR, other), H)
                    nd = count(reinterpret(UInt64, real(G[i])) != reinterpret(UInt64, real(H[i])) || reinterpret(UInt64, imag(G[i])) != reinterpret(UInt64, imag(H[i])) for i in eachindex(G))
                    println(io, "   vs ", other, ": ", nd == 0 ? "bitwise identical" : string(nd, " of ", length(G), " entries differ, max rel ", fm(maximum(abs, G .- H) / maximum(abs, G))))
                end
            end
        end
    end
end

# ===== (m) Gila before / after and the realistic block ========================
"Gila's egoFunOut! (the genEgoCrcExt! call) for the pair: (G, class, sep, ord, seconds)."
function gilaX(R, sT, sS, f)
    opt = GV.CPUKerOpt{Float64}(f, 48, false, GV.CPU())
    sTr = Rational{Int}.(sT); sSr = Rational{Int}.(sS)
    trgFac = Float64.(GV.cubFac(sTr)); srcFac = Float64.(GV.cubFac(sSr)); fp36 = GV.facPar()
    grd = SVector{3,Float64}(Float64.(R))
    G = zeros(ComplexF64, 3, 3)
    t = @elapsed GV.egoFunOut!(G, grd, sTr, sSr, trgFac, srcFac, fp36, opt)
    t = min(t, @elapsed GV.egoFunOut!(G, grd, sTr, sSr, trgFac, srcFac, fp36, opt))
    sclMax = Float64.(max.(sT, sS)); sep = maximum(round.(Int, abs.(Float64.(R)) ./ sclMax))
    ord = sep <= 1 ? 0 : GV.quadOrd(sep, 2pi * abs(f) * maximum(sclMax))
    (G, sep <= 1 ? "adp" : "fxd", sep, ord, t)
end
relX(A, B) = Float64(maximum(abs, A .- B) / maximum(abs, B))
function partM()
    RX = loadRefX(); g = GQ[1]; f1 = ComplexF64(1); nth = Threads.nthreads()
    if nth == 1
        tee("m_gila.txt") do io
            println(io, "(m) Gila's egoFunOut! today (adaptive hcubature for sep <= 1, fixed Gauss-Legendre quadOrd(sep) beyond)")
            println(io, "    against farTensorX and the 220-bit references on xgeom's r = 4 xyz block (every non-touching")
            println(io, "    offset) and a stratified sample of the r = 16 xyz block.  class = adaptive by face gap kap (fine")
            println(io, "    cells along x) or fixed by order; G-L = Gila vs farTensorX (max-entry, the library is within 5e-15")
            println(io, "    of the reference in (j)); G-ref / L-ref vs the reference where cached; ms = Gila per offset,")
            println(io, "    us = farTensorX per offset (fs held; route 2 = N_t N_s equal-cell pieces), blk us = farBlockX! per")
            println(io, "    offset of the whole block.  load ", Sys.loadavg(), "; threads ", nth)
            for r in (4, 16)
                sS = (r * g, r * g, r * g); b = ntuple(d -> (GQ[d] + sS[d]) // 2, 3)
                Rall = [R for R in xBlock(r) if !xTouch(R, b)]
                fx = farSetupX(GQ, sS, f1; offs = Rall)
                Gb = zeros(ComplexF64, 3, 3, length(Rall)); farBlockX!(Gb, Rall, GQ, sS, f1; fs = fx)
                t0 = time(); farBlockX!(Gb, Rall, GQ, sS, f1; fs = fx); tB = time() - t0
                cls(R) = (sep = maximum(round.(Int, abs.(Float64.(R)) ./ Float64(r * g)));
                          sep <= 1 ? string("adp kap ", Int(R[1] / g - QI(r + 1) // 2)) : string("fxd ord ", GV.quadOrd(sep, 2pi * Float64(r * g))))
                if r == 4
                    Rs = Rall
                else
                    # stratified: every cached reference in the block, plus up to 120 random offsets per class (seeded)
                    rng = Random.MersenneTwister(20260908)
                    byc = Dict{String,Vector{Int}}()
                    for (i, R) in enumerate(Rall); push!(get!(byc, cls(R), Int[]), i); end
                    pick = Set{Int}(i for (i, R) in enumerate(Rall) if haskey(RX, (R, GQ, sS, (1.0, 0.0))))
                    for (c, ix) in byc; for i in Random.shuffle(rng, ix)[1:min(120, length(ix))]; push!(pick, i); end; end
                    Rs = Rall[sort(collect(pick))]
                end
                idx = Dict(R => i for (i, R) in enumerate(Rall))
                println(io); println(io, "## r = ", r, " xyz block: ", length(Rall), " non-touching offsets, ", length(Rs), " evaluated with Gila; farBlockX! warm ",
                        @sprintf("%.2f", tB), " s = ", @sprintf("%.1f", 1e6 * tB / length(Rall)), " us per offset; routes 1/2/3 ",
                        join([count(==(k), [farRouteX(fx, R)[1] for R in Rall]) for k in 1:3], "/"))
                println(io, rp("R/g", 17), rp("class", 12), rp("rt", 3), rp("G-L", 10), rp("G-ref", 10), rp("L-ref", 10), rp("Gila ms", 9), "lib us")
                stat = Dict{String,Vector{Vector{Float64}}}()
                for R in Rs
                    Gg, c, sep, ord, tg = gilaX(R, GQ, sS, f1)
                    tl = @elapsed Gl = farTensorX(R, GQ, sS, f1; fs = fx)
                    tl = min(tl, @elapsed farTensorX(R, GQ, sS, f1; fs = fx))
                    kind, _ = farRouteX(fx, R)
                    Gr = get(RX, (R, GQ, sS, (1.0, 0.0)), nothing)
                    egl = relX(Gg, Gl); egr = Gr === nothing ? NaN : errs(Gg, Gr)[1]; elr = Gr === nothing ? NaN : errs(Gl, Gr)[1]
                    k = cls(R); s = get!(stat, k, [Float64[], Float64[], Float64[], Float64[], Float64[]])
                    push!(s[1], egl); push!(s[4], 1e3 * tg); push!(s[5], 1e6 * tl)
                    Gr === nothing || (push!(s[2], egr); push!(s[3], elr))
                    (Gr !== nothing || r == 4 && (R[2] == QI(3)//2 * g && R[3] == QI(3)//2 * g)) &&
                        println(io, rp(xRg(R, GQ, sS), 17), rp(k, 12), rp(kind, 3), rp(fm(egl), 10), rp(fm(egr), 10), rp(fm(elr), 10), rp(@sprintf("%.2f", 1e3 * tg), 9), @sprintf("%.1f", 1e6 * tl))
                end
                println(io)
                println(io, rp("class", 12), rp("n", 6), rp("G-L min", 10), rp("G-L med", 10), rp("G-L max", 10), rp("nref", 5), rp("G-ref max", 10), rp("L-ref max", 10), rp("Gila ms med", 12), rp("Gila ms max", 12), rp("lib us med", 11), "lib us max")
                for k in sort(collect(keys(stat)); by = x -> (startswith(x, "fxd"), parse(Int, split(x)[3]) * (startswith(x, "fxd") ? -1 : 1)))
                    s = stat[k]
                    println(io, rp(k, 12), rp(length(s[1]), 6), rp(fm(minimum(s[1])), 10), rp(fm(med(s[1])), 10), rp(fm(maximum(s[1])), 10), rp(length(s[2]), 5),
                            rp(isempty(s[2]) ? "-" : fm(maximum(s[2])), 10), rp(isempty(s[3]) ? "-" : fm(maximum(s[3])), 10),
                            rp(@sprintf("%.2f", med(s[4])), 12), rp(@sprintf("%.2f", maximum(s[4])), 12), rp(@sprintf("%.1f", med(s[5])), 11), @sprintf("%.1f", maximum(s[5])))
                end
                flush(io)
            end
        end
    end
    tee("m_block_t$(nth).txt") do io
        println(io, "(m) farBlockX! on the realistic block: coarse 4^3 cells of 16 g (lambda/2) against fine 64^3 cells of g")
        println(io, "    sharing a face, R = ((2k+1)/64), k_x in 8..119, k_y, k_z in -56..55, the 324 touching offsets excluded;")
        println(io, "    f = 1; threads ", nth, "; load at start ", Sys.loadavg())
        sS = (16g, 16g, 16g); b = ntuple(d -> (GQ[d] + sS[d]) // 2, 3)
        Rs = [(QI(2k1 + 1)//64, QI(2k2 + 1)//64, QI(2k3 + 1)//64) for k3 in -56:55 for k2 in -56:55 for k1 in 8:119]
        filter!(R -> !xTouch(R, b), Rs)
        println(io, "offsets ", length(Rs))
        G = zeros(ComplexF64, 3, 3, length(Rs)); GC.gc()
        rt = Int[]; cs = Float64[]
        for (run, crt) in (("cold (pair table and fine set built or read; no FrqSetX held)", false), ("warm", false), ("warm, cert = true", true))
            tms = Dict{Symbol,Float64}()
            t0 = time(); res = farBlockX!(G, Rs, GQ, sS, f1; cert = crt, tms = tms); tt = time() - t0
            rt = crt ? res[1] : res; crt && (cs = res[2])
            println(io, rp(run, 62), @sprintf("%7.2f s", tt), "  phases ", join([@sprintf("%s %.2f", k, v) for (k, v) in sort(collect(tms))], ", "),
                    "  maxrss ", @sprintf("%.2f", Sys.maxrss() / 1e9), " GB  load ", Sys.loadavg()[1])
        end
        fx = farSetupX(GQ, sS, f1; offs = Rs)
        t0 = time(); farBlockX!(G, Rs, GQ, sS, f1; fs = fx); tt = time() - t0
        println(io, rp("warm, FrqSetX held (fs = fx)", 62), @sprintf("%7.2f s", tt), "  load ", Sys.loadavg()[1])
        println(io, "routes 1/2/3 ", join([count(==(k), rt) for k in 1:3], "/"))
        rel = [cs[i] / maximum(abs, view(G, :, :, i)) for i in eachindex(cs)]
        for k in 1:2
            ix = [i for i in eachindex(rt) if rt[i] == k]
            println(io, "route ", k, " cert/max|G| min ", fm(minimum(rel[ix])), " median ", fm(med(rel[ix])), " max ", fm(maximum(rel[ix])), "; above tol ", count(>(TOL), rel[ix]), " of ", length(ix))
        end
        nr = 0; wr = 0.0; viol = 0
        for (i, R) in enumerate(Rs)
            Gr = get(RX, (R, GQ, sS, (1.0, 0.0)), nothing); Gr === nothing && continue
            nr += 1; e = errs(view(G, :, :, i), Gr); wr = max(wr, e[1]); viol += cs[i] < absErr(view(G, :, :, i), Gr)
        end
        println(io, "references in the block ", nr, "; worst max-entry error ", fm(wr), "; certificate violations ", viol, "; load ", Sys.loadavg())
    end
end

const T0 = time()
PART in ("a", "all") && partA()
PART in ("b", "all") && partB()
PART in ("c", "all") && partC()
PART in ("d", "all") && partD()
PART in ("e", "all") && partE()
PART in ("f", "all") && partF()
PART in ("g", "all") && partG()
PART in ("h", "all") && partH()
PART in ("i", "all") && partI()
PART in ("j", "all") && partJ()
PART in ("k", "all") && partK()
PART in ("l", "all") && partL()
PART in ("m", "all") && partM()
println("part ", PART, ": ", round(time() - T0, digits = 1), " s")
