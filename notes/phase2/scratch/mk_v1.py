import re, sys
src = open("../../farfield/farfield.jl").read()
orig = src

def rep(old, new, n=1):
    global src
    c = src.count(old)
    assert c == n, f"count {c} != {n} for:\n{old[:200]}"
    src = src.replace(old, new)

# ---- 1. dead: wgtT / wgtTrpT
rep("""# triangle weight on [-s,s]: mu_n = 2 s^{n+2}/((n+1)(n+2)) even n, 0 odd
wgtT(n::Int, s::S) where {S} = (n < 0 || isodd(n)) ? zero(S) : 2 * s^(n + 2) / S((n + 1) * (n + 2))
# affine""", "# affine")
rep("wgtTrpT(s::S, dM::Int) where {S} = momTrp(S[wgtT(n, s) for n in 0:dM])\n", "")
rep("a = 0 is wgtT bit for bit (x - 0 == x)", "a = 0 is the triangle bit for bit (x - 0 == x)")

# ---- 2. dead: jbnd
rep("""# |j_l(z)| <= |z|^l e^{|z|^2/(4l+6)} / (2l+1)!!
function jbnd(l::Int, z::T) where {T}
    df = one(T); for j in 0:l; df *= T(2j + 1); end
    z^l * exp(z^2 / T(4l + 6)) / df
end
""", "")

# ---- 3. dead: hrecS, hFac (hm collapses to the proven majorant hbndS)
rep('''"|h_l(z)| |k|^l/(2l+1)!! from the scaled upward recurrence: accurate in practice, but not proven."
function hrecS(L::Int, z::C, ak::T) where {T,C}
    hs = zeros(C, L + 1)
    hs[1] = -im * exp(im * z) / z
    L == 0 && return abs.(hs)
    hs[2] = hs[1] * (inv(z) - im) * ak / T(3)
    for l in 1:(L - 1)
        hs[l + 2] = T(2l + 1) * ak / (T(2l + 3) * z) * hs[l + 1] -
                    ak * ak / (T(2l + 1) * T(2l + 3)) * hs[l]
    end
    abs.(hs)
end

"The |h_l| factor of the bounds: :bd the proven majorant hb (default), :ex the recurrence value."
hFac(L::Int, z::C, ak::T, hm::Symbol) where {T,C} =
    hm === :bd ? hbndS(L, z, ak) : hrecS(L, z, ak)

''', "")

# ---- 4. dead: lamW
rep("""# lam_n(s) = int_{-s}^{s} t^n dt, the flat measure left on the two axes carrying a derivative
lamW(n::Int, s::T) where {T} = 2 * s^(n + 1) / T(n + 1)

""", "")

# ---- 5. hm out of bndCum
rep("""function bndCum(ls::Vector{Int}, t::NTuple{6,Vector{T}}, k::C, rr::T,
                hm::Symbol = :bd) where {T,C}
    n = length(ls)
    hs = hFac(ls[n], k * rr, abs(k), hm)""",
    """function bndCum(ls::Vector{Int}, t::NTuple{6,Vector{T}}, k::C, rr::T) where {T,C}
    n = length(ls)
    hs = hbndS(ls[n], k * rr, abs(k))""")

# ---- 6. hm out of whlThr
rep("""function whlThr(ls::Vector{Int}, t::NTuple{6,Vector{T}}, k::C, scf, tol::T,
                rLo::T, rHi::T, lTop::Int, hm::Symbol) where {T,C}
    n = count(<=(lTop), ls)
    ok(r, i) = (cum = bndCum(ls, t, k, r, hm); cum[i] <= tol * scf(r))""",
    """function whlThr(ls::Vector{Int}, t::NTuple{6,Vector{T}}, k::C, scf, tol::T,
                rLo::T, rHi::T, lTop::Int) where {T,C}
    n = count(<=(lTop), ls)
    ok(r, i) = (cum = bndCum(ls, t, k, r); cum[i] <= tol * scf(r))""")

# ---- 7. hm out of FrqSet / FrqSetX / farSetup / farSetupX / fineSet and the bound calls
rep("    hm::Symbol                      # :bd (proven hb majorant) or :ex the h_l recurrence\n", "", 0) # tolerate
rep("    hm::Symbol                      # :bd (proven hb majorant) or :ex (the h_l recurrence)\n", "")
rep("""    wT::NTuple{6,Vector{Float64}}
    hm::Symbol
    tol::Float64""", """    wT::NTuple{6,Vector{Float64}}
    tol::Float64""")

rep("""                  lDef::Int = LDEF, offs = nothing, scl::Symbol = :est, hm::Symbol = :bd,
                  disk::Bool = true""",
    """                  lDef::Int = LDEF, offs = nothing, scl::Symbol = :est,
                  disk::Bool = true""")
rep("    thr = whlThr(wLs, wT, k6, scf, tol, rLo, rHi, L, hm)",
    "    thr = whlThr(wLs, wT, k6, scf, tol, rLo, rHi, L)")
rep("                q = pickCut(oLs, bndCum(oLs, oT, k6, vr, hm), bud, L)",
    "                q = pickCut(oLs, bndCum(oLs, oT, k6, vr), bud, L)")
rep("""                hf, thr, wLs, wT, oLs, oT, hm, e0, eT, scl, tol, nCut, nNd, tb.nMax, rNc, rHi,""",
    """                hf, thr, wLs, wT, oLs, oT, e0, eT, scl, tol, nCut, nNd, tb.nMax, rNc, rHi,""")
rep("""                   L::Int = LMAX, nMax::Int = NMAX, nBlk::Int = 128, lDef::Int = LDEF, offs = (),
                   hm::Symbol = :bd, disk::Bool = true, nCap::Int = NCAP,""",
    """                   L::Int = LMAX, nMax::Int = NMAX, nBlk::Int = 128, lDef::Int = LDEF, offs = (),
                   disk::Bool = true, nCap::Int = NCAP,""")
rep("    thr = whlThr(wLs, wT, k6, r -> est(r, k6, f6, vs6), tol, rLo, rHi, L, hm)",
    "    thr = whlThr(wLs, wT, k6, r -> est(r, k6, f6, vs6), tol, rLo, rHi, L)")
rep("""                 thr, wLs, wT, hm, tol, nCut, nBlk, rNc, rHi, disk, String(dir),""",
    """                 thr, wLs, wT, tol, nCut, nBlk, rNc, rHi, disk, String(dir),""")
rep("""            (fx.fine[] = farSetup(fx.gQ, fx.frq; tol = fx.tol, nBlk = max(fx.nBlk, nBlk), hm = fx.hm,
                                  disk = fx.disk, dir = fx.dir))""",
    """            (fx.fine[] = farSetup(fx.gQ, fx.frq; tol = fx.tol, nBlk = max(fx.nBlk, nBlk),
                                  disk = fx.disk, dir = fx.dir))""")
rep("""bndWhl(fs::FrqSet{T}, rr::T, L::Int) where {T} =
    (c = bndCum(fs.wLs, fs.wT, ComplexF64(fs.k), Float64(rr), fs.hm); i = div(L, 2) + 1;""",
    """bndWhl(fs::FrqSet{T}, rr::T, L::Int) where {T} =
    (c = bndCum(fs.wLs, fs.wT, ComplexF64(fs.k), Float64(rr)); i = div(L, 2) + 1;""")
rep("""bndWhlX(fx::FrqSetX, rr::Real, L::Int) =
    (c = bndCum(fx.wLs, fx.wT, ComplexF64(fx.k), Float64(rr), fx.hm); i = div(L, 2) + 1;""",
    """bndWhlX(fx::FrqSetX, rr::Real, L::Int) =
    (c = bndCum(fx.wLs, fx.wT, ComplexF64(fx.k), Float64(rr)); i = div(L, 2) + 1;""")
rep("        Lc[q] = pickCut(fs.oLs, bndCum(fs.oLs, fs.oT, k6, vr, fs.hm), bud, fs.L)",
    "        Lc[q] = pickCut(fs.oLs, bndCum(fs.oLs, fs.oT, k6, vr), bud, fs.L)")
rep("        c += bndCum(fs.oLs, fs.oT, k6, vr, fs.hm)[Lc[q] + 1]",
    "        c += bndCum(fs.oLs, fs.oT, k6, vr)[Lc[q] + 1]")

# ---- 8. dead: farRouteStat
i0 = src.index('"Counts of offsets routed to (i), (ii), (iii)')
i1 = src.index("# ---- unequal cells ---")
src = src[:i0] + src[i1:]

# ---- 9. xKey inlined into shpFile
rep('''const ZQ3 = (zero(QI), zero(QI), zero(QI))
xKey(ta::NTuple{3,QI}) = all(iszero, ta) ? "" :
    string("_a", ta[1].num, "_", ta[1].den, "-", ta[2].num, "_", ta[2].den, "-", ta[3].num, "_", ta[3].den)
shpFile(s::NTuple{3,QI}, nMax::Int, dir::AbstractString = TABDIR[]) = shpFile(ZQ3, s, nMax, dir)
shpFile(ta::NTuple{3,QI}, tb::NTuple{3,QI}, nMax::Int, dir::AbstractString = TABDIR[]) =
    joinpath(dir, shpKey(tb, 0, nMax) * xKey(ta) * "_seg.txt")''',
'''const ZQ3 = (zero(QI), zero(QI), zero(QI))
shpFile(s::NTuple{3,QI}, nMax::Int, dir::AbstractString = TABDIR[]) = shpFile(ZQ3, s, nMax, dir)
shpFile(ta::NTuple{3,QI}, tb::NTuple{3,QI}, nMax::Int, dir::AbstractString = TABDIR[]) =
    joinpath(dir, shpKey(tb, 0, nMax) *
             (all(iszero, ta) ? "" :
              string("_a", ta[1].num, "_", ta[1].den, "-", ta[2].num, "_", ta[2].den,
                     "-", ta[3].num, "_", ta[3].den)) * "_seg.txt")''')

# ---- 10. route (iii): moments come from the same module, no runtime include
rep('''const MOMJL = Ref(joinpath(@__DIR__, "..", "moments", "moments.jl"))
const KSRDIR''', '''const KSRDIR''')
rep("""const MOMLK = ReentrantLock()           # guards MOMC and the one-time load of moments.jl
const MOMF = Ref{Any}(nothing)          # (facePair, pairMoments), fetched once in the latest world

# The two functions are read once, through invokelatest, and held here: a plain getglobal after a
# run-time Base.include reads a binding in a world prior to its definition world, which Julia 1.12
# warns about and a later version will make an error.
function needMom()
    lock(MOMLK) do
        MOMF[] === nothing || return MOMF[]
        if !(isdefined(Main, :pairMoments) && isdefined(Main, :facePair))
            isfile(MOMJL[]) || error("route (iii) needs notes/moments/moments.jl; set MOMJL[]")
            Base.include(Main, MOMJL[])
        end
        MOMF[] = (Base.invokelatest(getglobal, Main, :facePair),
                  Base.invokelatest(getglobal, Main, :pairMoments))
    end
end

\"srfSum! of glaVacOprMemGen.jl on srf[fp] = (raw face-pair integral)/V_t, fp = 6(F-1)+F'.\"
function srfSum(s::AbstractVector{C}) where {C}
    G = zeros(C, 3, 3)
    G[1, 1] = s[15] - s[16] - s[21] + s[22] + s[29] - s[30] - s[35] + s[36]
    G[2, 1] = -s[13] + s[14] + s[19] - s[20]
    G[3, 1] = -s[25] + s[26] + s[31] - s[32]
    G[1, 2] = -s[3] + s[4] + s[9] - s[10]
    G[2, 2] = s[1] - s[2] - s[7] + s[8] + s[29] - s[30] - s[35] + s[36]
    G[3, 2] = -s[27] + s[28] + s[33] - s[34]
    G[1, 3] = -s[5] + s[6] + s[11] - s[12]
    G[2, 3] = -s[17] + s[18] + s[23] - s[24]
    G[3, 3] = s[1] - s[2] - s[7] + s[8] + s[15] - s[16] - s[21] + s[22]
    return G
end

""", "const MOMLK = ReentrantLock()           # guards MOMC and MOMCX\n\n")

rep("""    fp0, pm0 = needMom()
    fp(a...) = Base.invokelatest(fp0, a...)
    pm(a...) = Base.invokelatest(pm0, a...)
    out = setprecision""", "    out = setprecision")
rep("        pnB = [map(bf, fp(D, F, Fp, sQ)) for F in 1:6 for Fp in 1:6]",
    "        pnB = [map(bf, parFac(D, F, Fp, sQ)) for F in 1:6 for Fp in 1:6]")
rep("""    _, pm0 = needMom()
    pm(a...) = Base.invokelatest(pm0, a...)
    out = setprecision""", "    out = setprecision")
rep("        lam = sum(pm(p[1], p[2], -1)[1] for p in pnB)",
    "        lam = sum(parMom(p[1], p[2], -1)[1] for p in pnB)", 2)
rep("            mm = (N + 1, [pm(p[1], p[2], N + 1) for p in pnB])",
    "            mm = (N + 1, [parMom(p[1], p[2], N + 1) for p in pnB])", 2)
rep("""        (srfSum(srf), N, Float64(amp))""",
    """        gb = zeros(CB, 3, 3); srfSum!(gb, srf)
        (gb, N, Float64(amp))""", 2)

# ---- 11. boxFaceQ / FACESX -> boxFace / FACES of glaVacIntMom.jl
rep('''const FACESX = ((1, -1), (1, 1), (2, -1), (2, 1), (3, -1), (3, 1))
boxFaceQ(c::Int, sig::Int, ctr, s) =
    ntuple(d -> d == c ? (ctr[d] + sig * s[d] // 2, ctr[d] + sig * s[d] // 2) :
                         (ctr[d] - s[d] // 2, ctr[d] + s[d] // 2), 3)
"Face F of the target (edges sT at R) and face F' of the source (edges sS at 0) as panels."
facePairX(RQ, sTQ, sSQ, F::Int, Fp::Int) =
    (boxFaceQ(FACESX[F]..., RQ, sTQ), boxFaceQ(FACESX[Fp]..., ntuple(_ -> zero(RQ[1]), 3), sSQ))''',
'''"Face F of the target (edges sT at R) and face F' of the source (edges sS at 0) as panels."
facePairX(RQ::NTuple{3,QI}, sTQ::NTuple{3,QI}, sSQ::NTuple{3,QI}, F::Int, Fp::Int) =
    (boxFace(FACES[F]..., RQ, sTQ), boxFace(FACES[Fp]..., ZQ3, sSQ))''')

# ---- 12. boundL / boundLX collapse
rep('''"Whole-box L for offset radius rr from the tabulated thresholds; -1 if L > Lmax is needed."
function boundL(fs::FrqSet{T}, rr::T) where {T}
    thr = fs.thr; r6 = Float64(rr)
    @inbounds for i in eachindex(thr)
        r6 >= thr[i] && return 2 * (i - 1)
    end
    return -1
end''',
'''"Whole-box L for offset radius rr from the tabulated thresholds; -1 if L > Lmax is needed."
function boundL(thr::Vector{Float64}, rr::Real)
    r6 = Float64(rr)
    @inbounds for i in eachindex(thr)
        r6 >= thr[i] && return 2 * (i - 1)
    end
    return -1
end
boundL(fs, rr::Real) = boundL(fs.thr, rr)   # FrqSet or FrqSetX''')
rep('''"Whole-box L at radius rr from the trapezoid thresholds; -1 if L > Lmax is needed."
function boundLX(fx::FrqSetX, rr::Real)
    thr = fx.thr; r6 = Float64(rr)
    @inbounds for i in eachindex(thr)
        r6 >= thr[i] && return 2 * (i - 1)
    end
    return -1
end

''', "")
rep("    L = boundLX(fx, rr)", "    L = boundL(fx, rr)")
# the two local bL closures now go through the same body
rep("    bL(rr) = (for i in eachindex(thr); rr >= thr[i] && return 2 * (i - 1); end; -1)\n    Lw = min(lDef, L); Lo = 0; rNc = rHi",
    "    Lw = min(lDef, L); Lo = 0; rNc = rHi")
rep("        Lc = bL(rr)\n        if Lc >= 0", "        Lc = boundL(thr, rr)\n        if Lc >= 0")
rep("""    bL(rr) = (for i in eachindex(thr); rr >= thr[i] && return 2 * (i - 1); end; -1)
    # the table is sized""", "    # the table is sized")
rep("        Lc = bL(rr)\n        Lc >= 0 &&", "        Lc = boundL(thr, rr)\n        Lc >= 0 &&")

# ---- 13. farRoute: offset-vector body, lattice method
rep('''@inline ckRad(fs::FrqSet, rLo::Float64, rUp::Float64, D::NTuple{3,Int}) =
    (fs.rNc * (1 - 1e-12) <= rLo && rUp <= fs.rHi * (1 + 1e-12)) ? nothing :
    error("farRoute: offset $D has radius $rLo..$rUp outside [$(fs.rNc), $(fs.rHi)], where this ",
          "set's n-cut is certified; build the set with this offset in offs")

"Route for one offset: (kind, L, Lc, cost); kind 1 = whole box, 2 = octants, 3 = k-series."
function farRoute(fs::FrqSet{T}, D::NTuple{3,Int}; cst::Bool = false) where {T}
    R = ntuple(d -> T(D[d]) * fs.s[d], 3)
    rr = sqrt(sum(R[i]^2 for i in 1:3))
    L = boundL(fs, rr)
    if L >= 0
        L <= fs.Lw || error("farRoute: offset $D needs whole-box l = $L, table holds $(fs.Lw)")
        ckRad(fs, Float64(rr), Float64(rr), D)
        return (1, L, Int[], cst ? costWhl(fs.whl, L) : 0)
    end
    Lc = boundLoct(fs, R)
    any(<(0), Lc) && return (3, -1, Lc, 0)
    maximum(Lc) <= fs.Lo || error("farRoute: offset $D needs octant l = $(maximum(Lc)), table holds $(fs.Lo)")
    vs = extrema(sqrt(sum(Float64(sg[d] * R[d] + fs.ctr[d])^2 for d in 1:3)) for sg in SGN8)
    ckRad(fs, vs[1], vs[2], D)
    (2, -1, Lc, cst ? costOct(fs.oct, Lc) : 0)
end''',
'''@inline ckRad(fs::FrqSet, rLo::Float64, rUp::Float64, R) =
    (fs.rNc * (1 - 1e-12) <= rLo && rUp <= fs.rHi * (1 + 1e-12)) ? nothing :
    error("farRoute: offset $R has radius $rLo..$rUp outside [$(fs.rNc), $(fs.rHi)], where this ",
          "set's n-cut is certified; build the set with this offset in offs")

"Route for one offset: (kind, L, Lc, cost); kind 1 = whole box, 2 = octants, 3 = k-series."
function farRoute(fs::FrqSet{T}, R::NTuple{3,T}; cst::Bool = false) where {T}
    rr = sqrt(sum(R[i]^2 for i in 1:3))
    L = boundL(fs, rr)
    if L >= 0
        L <= fs.Lw || error("farRoute: offset $R needs whole-box l = $L, table holds $(fs.Lw)")
        ckRad(fs, Float64(rr), Float64(rr), R)
        return (1, L, Int[], cst ? costWhl(fs.whl, L) : 0)
    end
    Lc = boundLoct(fs, R)
    any(<(0), Lc) && return (3, -1, Lc, 0)
    maximum(Lc) <= fs.Lo || error("farRoute: offset $R needs octant l = $(maximum(Lc)), table holds $(fs.Lo)")
    vs = extrema(sqrt(sum(Float64(sg[d] * R[d] + fs.ctr[d])^2 for d in 1:3)) for sg in SGN8)
    ckRad(fs, vs[1], vs[2], R)
    (2, -1, Lc, cst ? costOct(fs.oct, Lc) : 0)
end
farRoute(fs::FrqSet{T}, D::NTuple{3,Int}; cst::Bool = false) where {T} =
    farRoute(fs, ntuple(d -> T(D[d]) * fs.s[d], 3); cst = cst)''')

# ---- 14. farRouteX: integer-lattice body, exact-offset method
rep('''"Route for one exact offset: (kind, L); 1 whole trapezoid box at l = L, 2 gcd average, 3 k-series."
function farRouteX(fx::FrqSetX{T}, RQ::NTuple{3,QI}) where {T}
    # selection in Float64; the touching test is exact, behind a Float64 screen (rational arithmetic
    # on Rational{BigInt} allocates, and a block has 1e6 offsets of which a few hundred touch)
    R6 = ntuple(d -> Float64(RQ[d]), 3)
    all(abs(R6[d]) <= fx.b6[d] * (1 + 1e-9) for d in 1:3) &&
        all(abs(RQ[d]) <= fx.tbQ[d] for d in 1:3) &&
        error("farRouteX: the cells at R = $RQ touch or overlap (Gila's contact path)")
    rr = sqrt(sum(R6[i]^2 for i in 1:3))
    L = boundL(fx, rr)
    if L >= 0
        L <= fx.Lw || error("farRouteX: offset $RQ needs whole-box l = $L, table holds $(fx.Lw)")
        # the k-series cut of the table is certified on [rNc, rHi] only (D13: a set built for other
        # offsets gave 1.6e-11 at a nearer one); a set from farTensorX/farBlockX! covers its offsets
        fx.rNc * (1 - 1e-12) <= rr <= fx.rHi * (1 + 1e-12) ||
            error("farRouteX: |R| = $rr is outside [$(fx.rNc), $(fx.rHi)], where this set's n-cut is ",
                  "certified; build the set with this offset in offs")
        return (1, L)
    end
    m = xLat(fx, RQ)
    (m !== nothing && xNear(fx, m) >= 2) ? (2, -1) : (3, -1)
end
farRouteX(fx::FrqSetX, R::NTuple{3,<:Rational}) = farRouteX(fx, ntuple(d -> QI(R[d]), 3))''',
'''"The cells touch or overlap when |R_d| <= b_d on every axis; exact in both representations."
tchX(fx::FrqSetX, m::NTuple{3,Int}) =
    all(abs(2 * m[d] + fx.nT[d] - fx.nS[d]) <= fx.nT[d] + fx.nS[d] for d in 1:3)
# the rational test is behind a Float64 screen: Rational{BigInt} arithmetic allocates, and a block
# has 1e6 offsets of which a few hundred touch
tchX(fx::FrqSetX, RQ::NTuple{3,QI}) =
    all(abs(Float64(RQ[d])) <= fx.b6[d] * (1 + 1e-9) for d in 1:3) &&
    all(abs(RQ[d]) <= fx.tbQ[d] for d in 1:3)

"Route for one offset: (kind, L); 1 whole trapezoid box at l = L, 2 gcd average, 3 k-series."
function farRouteX(fx::FrqSetX, R6::NTuple{3,Float64}, m::Union{Nothing,NTuple{3,Int}})
    rr = sqrt(sum(R6[i]^2 for i in 1:3))
    L = boundL(fx, rr)
    if L >= 0
        L <= fx.Lw || error("farRouteX: offset $R6 needs whole-box l = $L, table holds $(fx.Lw)")
        # the k-series cut of the table is certified on [rNc, rHi] only (D13: a set built for other
        # offsets gave 1.6e-11 at a nearer one); a set from farTensorX/farBlockX! covers its offsets
        fx.rNc * (1 - 1e-12) <= rr <= fx.rHi * (1 + 1e-12) ||
            error("farRouteX: |R| = $rr is outside [$(fx.rNc), $(fx.rHi)], where this set's n-cut is ",
                  "certified; build the set with this offset in offs")
        return (1, L)
    end
    (m !== nothing && xNear(fx, m) >= 2) ? (2, -1) : (3, -1)
end
farRouteX(fx::FrqSetX, m::NTuple{3,Int}) =
    (tchX(fx, m) && errTchX(m); farRouteX(fx, xPos(fx, m), m))
farRouteX(fx::FrqSetX, RQ::NTuple{3,QI}) =
    (tchX(fx, RQ) && errTchX(RQ); farRouteX(fx, ntuple(d -> Float64(RQ[d]), 3), xLat(fx, RQ)))
farRouteX(fx::FrqSetX, R::NTuple{3,<:Rational}) = farRouteX(fx, ntuple(d -> QI(R[d]), 3))
errTchX(x) = error("farRouteX: the cells at R = $x touch or overlap (Gila's contact path)")''')

# xPos beside xLat
rep('''"Multiplicity of the sub-offset difference t = j - j' over j in 1:nT, j' in 1:nS."''',
'''"Float64 offset vector of the integer gcd-lattice coordinate m; the inverse of xLat."
xPos(fx::FrqSetX, m::NTuple{3,Int}) = ntuple(d -> m[d] * fx.g6[d] + fx.o6[d], 3)
"Multiplicity of the sub-offset difference t = j - j' over j in 1:nT, j' in 1:nS."''')

# g6, o6 fields
rep("""    gQ::NTuple{3,QI}                # the gcd cell min(sT, sS) per axis""",
    """    gQ::NTuple{3,QI}                # the gcd cell min(sT, sS) per axis
    g6::NTuple{3,Float64}
    o6::NTuple{3,Float64}           # xPos: R = m g + o, o = (nT - nS) g/2""")
rep("""    FrqSetX{T,E}(sT, sS, sTQ, sSQ, tbQ, b6, gQ, nT, nS, frq,""",
    """    g6 = ntuple(d -> Float64(gQ[d]), 3)
    o6 = ntuple(d -> Float64((nT[d] - nS[d]) * gQ[d] // 2), 3)
    FrqSetX{T,E}(sT, sS, sTQ, sSQ, tbQ, b6, gQ, g6, o6, nT, nS, frq,""")

open("far_v1.jl", "w").write(src)
print("written", len(src.splitlines()), "lines (was", len(orig.splitlines()), ")")
