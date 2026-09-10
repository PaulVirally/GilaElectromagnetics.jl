# Bitwise comparison of notes/farfield/farfield.jl (OLD) against the port (NEW).
# usage: julia --project battery.jl <path-to-new-file> <tag>
const NEWSRC = length(ARGS) >= 1 ? ARGS[1] : "far_v1.jl"
const TAG    = length(ARGS) >= 2 ? ARGS[2] : "v1"
const HERE   = @__DIR__
const ROOT   = normpath(joinpath(HERE, "..", "..", ".."))
const NF     = joinpath(ROOT, "notes", "farfield")

t0 = time()
using GilaElectromagnetics
const GV = GilaElectromagnetics.GilaVacuum

module FarOld
include(joinpath(Main.NF, "farfield.jl"))
end
module FarNew
const parFac  = Main.GV.parFac
const parMom  = Main.GV.parMom
const boxFace = Main.GV.boxFace
const FACES   = Main.GV.FACES
const srfSum! = Main.GV.srfSum!
include(joinpath(Main.HERE, Main.NEWSRC))
end
snap() = Dict(f => filesize(joinpath(TABD0, f)) for f in readdir(TABD0))
const TABD0 = joinpath(NF, "shapetab")
const SNAP0 = Dict(f => filesize(joinpath(TABD0, f)) for f in readdir(TABD0))
pr(x...) = (println(x...); flush(stdout))
println("loaded in ", round(time() - t0, digits = 1), " s, threads = ", Threads.nthreads())

# name mapping OLD -> NEW (the port renames; fall back to the old name)
nw(s::Symbol) = isdefined(FarNew, s) ? getfield(FarNew, s) : error("FarNew has no $s")
const MAP = Dict(:farSetup => :farSet, :farSetupX => :farSetX, :farBlock! => :farBlk!,
                 :farBlockX! => :farBlkX!, :farTensor => :farTns, :farTensorX => :farTnsX,
                 :farRoute => :farRte, :farRouteX => :farRteX, :farShape => :farShp,
                 :boundL => :whlLvl, :FarWs => :FarWrk, :boxSumX! => :boxSum!,
                 :tnsGcdX! => :tnsGcd!, :fineSet => :finSet, :xLat => :latInd)
old(s::Symbol) = getfield(FarOld, s)
new(s::Symbol) = (t = get(MAP, s, s); isdefined(FarNew, t) ? getfield(FarNew, t) : getfield(FarNew, s))

const TABD = joinpath(NF, "shapetab")
FarOld.TABDIR[] = TABD; new(:TABDIR)[] = TABD
mkpath(joinpath(HERE, "ksrOld")); mkpath(joinpath(HERE, "ksrNew"))
for f in readdir(joinpath(NF, "ksrcache"))
    for d in ("ksrOld", "ksrNew")
        cp(joinpath(NF, "ksrcache", f), joinpath(HERE, d, f); force = true)
    end
end
FarOld.KSRDIR[] = joinpath(HERE, "ksrOld"); new(:KSRDIR)[] = joinpath(HERE, "ksrNew")

# ---- bitwise comparison ------------------------------------------------------
const NCMP = Ref(0); const NBAD = Ref(0); const BAD = String[]
same(a::BigFloat, b::BigFloat) = isequal(a, b) && precision(a) == precision(b)
same(a::Complex{BigFloat}, b::Complex{BigFloat}) = same(real(a), real(b)) && same(imag(a), imag(b))
same(a::AbstractArray, b::AbstractArray) =
    size(a) == size(b) && all(same(a[i], b[i]) for i in eachindex(a))
same(a::Tuple, b::Tuple) = length(a) == length(b) && all(same(a[i], b[i]) for i in eachindex(a))
same(a, b) = a === b
function chk(lbl, a, b)
    NCMP[] += 1
    same(a, b) && return true
    NBAD[] += 1
    length(BAD) < 40 && push!(BAD, string(lbl, ": ", a, " != ", b))
    false
end
chkA(lbl, A, B) = (chk(string(lbl, ".size"), size(A), size(B)) &&
                   (for i in eachindex(A); chk(string(lbl, "[", i, "]"), A[i], B[i]); end); nothing)
chkT(lbl, a::Tuple, b::Tuple) = (for i in eachindex(a); chk(string(lbl, "[", i, "]"), a[i], b[i]); end)

function chkGeo(lbl, g, h)
    for f in (:L, :nMax, :cnc); chk("$lbl.$f", getfield(g, f), getfield(h, f)); end
    for f in (:dLm, :dLv); chkA("$lbl.$f", getfield(g, f), getfield(h, f)); end
    chkA("$lbl.Ad", g.Ad, h.Ad); chkA("$lbl.Bd", g.Bd, h.Bd)
    for c in 1:3
        chkA("$lbl.oLm$c", g.oLm[c], h.oLm[c]); chkA("$lbl.oLv$c", g.oLv[c], h.oLv[c])
        chkA("$lbl.Ao$c", g.Ao[c], h.Ao[c])
    end
end
function chkFbx(lbl, g, h)
    chk("$lbl.L", g.L, h.L); chk("$lbl.cnc", g.cnc, h.cnc)
    chkA("$lbl.dLm", g.dLm, h.dLm); chkA("$lbl.dLv", g.dLv, h.dLv); chkA("$lbl.Qd", g.Qd, h.Qd)
    for c in 1:3
        chkA("$lbl.oLm$c", g.oLm[c], h.oLm[c]); chkA("$lbl.oLv$c", g.oLv[c], h.oLv[c])
        chkA("$lbl.Qo$c", g.Qo[c], h.Qo[c])
    end
end
function chkSet(lbl, a, b)
    for f in (:frq, :k, :prf, :L, :Lw, :Lo, :tol, :nNd, :nMx, :rNc, :rHi, :e0, :scl)
        chk("$lbl.$f", getfield(a, f), getfield(b, f))
    end
    chkT("$lbl.s", a.s, b.s); chkT("$lbl.ctr", a.ctr, b.ctr)
    for f in (:thr, :wLs, :oLs, :eT, :nCut); chkA("$lbl.$f", getfield(a, f), getfield(b, f)); end
    for r in 1:6; chkA("$lbl.wT$r", a.wT[r], b.wT[r]); chkA("$lbl.oT$r", a.oT[r], b.oT[r]); end
    chkFbx("$lbl.whl", a.whl, b.whl); chkFbx("$lbl.oct", a.oct, b.oct)
end
function chkSetX(lbl, a, b)
    for f in (:frq, :k, :prf, :L, :Lw, :tol, :rNc, :rHi, :nBlk)
        chk("$lbl.$f", getfield(a, f), getfield(b, f))
    end
    chkT("$lbl.sT", a.sT, b.sT); chkT("$lbl.sS", a.sS, b.sS)
    chkT("$lbl.nT", a.nT, b.nT); chkT("$lbl.nS", a.nS, b.nS); chkT("$lbl.b6", a.b6, b.b6)
    for f in (:thr, :wLs, :nCut); chkA("$lbl.$f", getfield(a, f), getfield(b, f)); end
    for r in 1:6; chkA("$lbl.wT$r", a.wT[r], b.wT[r]); end
    chkFbx("$lbl.whl", a.whl, b.whl)
end

# ---- the battery -------------------------------------------------------------
QO = FarOld.QI; QN = new(:QI)
q(t) = (QO(t[1]), QO(t[2]), QO(t[3]))
const C32 = (1//32, 1//32, 1//32); const C8 = (1//8, 1//8, 1//8)
const C4  = (1//4, 1//4, 1//4);    const SL = (1//32, 1//32, 1//512)
const F7 = (ComplexF64(1), ComplexF64(1, 0.1), ComplexF64(0.37), ComplexF64(0.25),
            ComplexF64(1, 1), ComplexF64(0.5, 2), ComplexF64(2))
# |k| r_d <= ~3 keeps the k-series order inside the nMax = 12 tables already on disk
frqOf(s) = begin
    rd = sqrt(sum(Float64(x)^2 for x in s))
    Tuple(f for f in F7 if 2 * pi * abs(f) * rd <= 3.0)
end

const RTC = Dict{Int,Int}()   # route histogram over everything routed
function tally!(fs, ids, s)
    for i in ids
        k, = old(:farRoute)(fs, (i[1] - 1, i[2] - 1, i[3] - 1))
        RTC[k] = get(RTC, k, 0) + 1
    end
end

pr("\n== equal cells: farBlock! blocks ==")
for (nm, s) in (("c32", C32), ("c8", C8), ("c4", C4), ("sl", SL))
    for f in frqOf(s)
        dim = nm == "sl" ? (5, 5, 7) : (6, 6, 6)
        t = time()
        eo = zeros(ComplexF64, 3, 3, dim...); en = zeros(ComplexF64, 3, 3, dim...)
        so = old(:farSetup)(q(s), f; nBlk = maximum(dim))
        sn = new(:farSetup)(q(s), f; nBlk = maximum(dim))
        chkSet("set[$nm,$f]", so, sn)
        old(:farBlock!)(eo, s, f; fs = so); new(:farBlock!)(en, s, f; fs = sn)
        chkA("blk[$nm,$f]", eo, en)
        ids = [(i1, i2, i3) for i3 in 1:dim[3], i2 in 1:dim[2], i1 in 1:dim[1]
               if max(i1, i2, i3) >= 3]
        tally!(so, ids, s)
        for i in ids
            D = (i[1] - 1, i[2] - 1, i[3] - 1)
            chkT("rte[$nm,$f,$D]", old(:farRoute)(so, D)[1:3], new(:farRoute)(sn, D)[1:3])
            chkT("rteC[$nm,$f,$D]", old(:farRoute)(so, D; cst = true), new(:farRoute)(sn, D; cst = true))
        end
        pr("  $nm f=$f  ", round(time() - t, digits = 1), " s  bad=", NBAD[])
    end
end

pr("\n== equal cells: farTensor, incl. Float32 and a bigger block ==")
for (nm, s, f, Ds) in (("c32", C32, ComplexF64(1), ((2,0,0), (5,3,1), (12,0,0), (40,17,3))),
                       ("c4",  C4,  ComplexF64(1), ((2,2,2), (3,0,1), (9,4,0))),
                       ("sl",  SL,  ComplexF64(1, 0.1), ((0,0,4), (2,0,0), (0,0,32))))
    for D in Ds
        go = old(:farTensor)(D, s, f); gn = new(:farTensor)(D, s, f)
        chkA("tns[$nm,$D]", go, gn)
    end
    pr("  $nm done, bad=", NBAD[])
end
for D in ((3, 0, 0), (8, 5, 2))
    go = old(:farTensor)(D, C32, ComplexF32(1)); gn = new(:farTensor)(D, C32, ComplexF32(1))
    chkA("tns32[$D]", go, gn)
end
pr("  Float32 done, bad=", NBAD[])

pr("\n== equal cells: cold route (iii) ==")
for d in ("ksrCold1", "ksrCold2"); rm(joinpath(HERE, d); force = true, recursive = true); end
mkpath(joinpath(HERE, "ksrCold1")); mkpath(joinpath(HERE, "ksrCold2"))
let so = old(:farSetup)(q(SL), ComplexF64(1); nBlk = 8, offs = ((0, 0, 3),)),
    sn = new(:farSetup)(q(SL), ComplexF64(1); nBlk = 8, offs = ((0, 0, 3),))
    FarOld.KSRDIR[] = joinpath(HERE, "ksrCold1"); new(:KSRDIR)[] = joinpath(HERE, "ksrCold2")
    t = time()
    ko, kn = old(:farRoute)(so, (0, 0, 3))[1], new(:farRoute)(sn, (0, 0, 3))[1]
    chk("coldRte", ko, kn); pr("  route = $ko")
    go = old(:farTensor)((0, 0, 3), SL, ComplexF64(1); fs = so)
    gn = new(:farTensor)((0, 0, 3), SL, ComplexF64(1); fs = sn)
    chkA("cold3", go, gn)
    pr("  cold (iii) ", round(time() - t, digits = 1), " s, bad=", NBAD[])
    FarOld.KSRDIR[] = joinpath(HERE, "ksrOld"); new(:KSRDIR)[] = joinpath(HERE, "ksrNew")
end

pr("\n== cross-scale ==")
const XP = (((1//16, 1//16, 1//16), C32), ((1//8, 1//8, 1//8), C32),
            (C32, (1//8, 1//8, 1//8)), ((1//8, 1//8, 1//32), C32),
            ((1//4, 1//4, 1//4), C32))
const FX = (ComplexF64(1), ComplexF64(1, 0.1), ComplexF64(0.37))
for (sT, sS) in XP, f in FX
    rd = sqrt(sum(Float64((sT[d] + sS[d]) // 2)^2 for d in 1:3))
    2 * pi * abs(f) * rd <= 3.0 || continue
    gQ = ntuple(d -> min(QO(sT[d]), QO(sS[d])), 3)
    nT = ntuple(d -> Int(QO(sT[d]) / gQ[d]), 3); nS = ntuple(d -> Int(QO(sS[d]) / gQ[d]), 3)
    Rs = NTuple{3,QO}[]
    for m3 in -3:5, m2 in -2:4, m1 in -2:6
        m = (m1, m2, m3)
        all(abs(2 * m[d] + nT[d] - nS[d]) <= nT[d] + nS[d] for d in 1:3) && continue
        push!(Rs, ntuple(d -> QO(2 * m[d] + nT[d] - nS[d]) * gQ[d] // 2, 3))
    end
    # a few off the gcd lattice
    for k in 0:8
        R = ntuple(d -> QO(3 * (k + d) + 1) * gQ[d] // 3, 3)
        all(abs(R[d]) <= QO(sT[d] + sS[d]) // 2 for d in 1:3) && continue
        push!(Rs, R)
    end
    t = time()
    Go = zeros(ComplexF64, 3, 3, length(Rs)); Gn = zeros(ComplexF64, 3, 3, length(Rs))
    ro, co = old(:farBlockX!)(Go, Rs, sT, sS, f; cert = true)
    rn, cn = new(:farBlockX!)(Gn, Rs, sT, sS, f; cert = true)
    chkA("blkX[$sT/$sS,$f]", Go, Gn); chkA("rtX[$sT/$sS,$f]", ro, rn)
    chkA("crtX[$sT/$sS,$f]", co, cn)
    for k in ro; RTC[10 + k] = get(RTC, 10 + k, 0) + 1; end
    pr("  $sT/$sS f=$f n=", length(Rs), "  ", round(time() - t, digits = 1),
            " s routes=", (count(==(1), ro), count(==(2), ro), count(==(3), ro)), " bad=", NBAD[])
end

pr("\n== cross-scale: farTensorX, both orientations, cert ==")
for (sT, sS) in (((1//16, 1//16, 1//16), C32), (C32, (1//16, 1//16, 1//16)),
                 ((1//8, 1//8, 1//32), C32))
    gQ = ntuple(d -> min(QO(sT[d]), QO(sS[d])), 3)
    nT = ntuple(d -> Int(QO(sT[d]) / gQ[d]), 3); nS = ntuple(d -> Int(QO(sS[d]) / gQ[d]), 3)
    for m in ((4, 0, 0), (2, 2, 1), (-5, 3, 0), (20, 0, 0))
        all(abs(2 * m[d] + nT[d] - nS[d]) <= nT[d] + nS[d] for d in 1:3) && continue
        R = ntuple(d -> QO(2 * m[d] + nT[d] - nS[d]) * gQ[d] // 2, 3)
        go, c1 = old(:farTensorX)(R, sT, sS, ComplexF64(1); cert = true)
        gn, c2 = new(:farTensorX)(R, sT, sS, ComplexF64(1); cert = true)
        chkA("tnsX[$sT/$sS,$m]", go, gn); chk("certX[$sT/$sS,$m]", c1, c2)
    end
    pr("  $sT/$sS done, bad=", NBAD[])
end

pr("\n== cross-scale: cold route 3 ==")
let sT = (1//8, 1//8, 1//8), sS = C32, f = ComplexF64(1)
    R = (QO(7) // 96, QO(11) // 96, QO(13) // 96)     # off the gcd lattice
    FarOld.KSRDIR[] = joinpath(HERE, "ksrCold1"); new(:KSRDIR)[] = joinpath(HERE, "ksrCold2")
    t = time()
    xo = old(:farSetupX)(q(sT), q(sS), f; nBlk = 16, offs = (R,))
    xn = new(:farSetupX)(q(sT), q(sS), f; nBlk = 16, offs = (R,))
    chkSetX("setX", xo, xn)
    chkT("rteX3", old(:farRouteX)(xo, R), new(:farRouteX)(xn, R))
    pr("  route = ", old(:farRouteX)(xo, R)[1])
    go = old(:farTensorX)(R, sT, sS, f; fs = xo); gn = new(:farTensorX)(R, sT, sS, f; fs = xn)
    chkA("coldX3", go, gn)
    pr("  cold X3 ", round(time() - t, digits = 1), " s, bad=", NBAD[])
    FarOld.KSRDIR[] = joinpath(HERE, "ksrOld"); new(:KSRDIR)[] = joinpath(HERE, "ksrNew")
end

pr("\n== shape tables ==")
for (a, b) in ((FarOld.ZQ3, q(C32)), (FarOld.ZQ3, q(C8)), (FarOld.ZQ3, q(SL)),
               (q((1//64, 1//64, 1//64)), q((3//64, 3//64, 3//64))))
    to = old(:farShape)(a, b); tn = new(:farShape)(a, b)
    chk("shp.Lw", to.Lw, tn.Lw); chk("shp.Lo", to.Lo, tn.Lo); chk("shp.nMax", to.nMax, tn.nMax)
    chkGeo("shp.whl$b", to.whl, tn.whl); chkGeo("shp.oct$b", to.oct, tn.oct)
end
pr("  bad=", NBAD[])

println("\n================ $TAG ================")
let s1 = snap(); ch = [k for k in union(keys(s1), keys(SNAP0)) if get(s1,k,-1) != get(SNAP0,k,-1)];
    println("shapetab files changed during the run: ", isempty(ch) ? "none" : ch); end
println("comparisons: ", NCMP[], "   mismatches: ", NBAD[])
println("equal-cell route histogram (1,2,3): ",
        (get(RTC, 1, 0), get(RTC, 2, 0), get(RTC, 3, 0)))
println("cross-scale route histogram (1,2,3): ",
        (get(RTC, 11, 0), get(RTC, 12, 0), get(RTC, 13, 0)))
for b in BAD; pr("  ", b); end
NBAD[] == 0 || exit(1)
