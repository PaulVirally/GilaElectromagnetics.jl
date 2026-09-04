# Fixed Gauss-Legendre quadrature for cell pairs that are not in contact
# References are built in-test: an adaptive hcubature run far tighter than the
# production tolerances, and an order 24 tensor rule written out by hand, which
# also checks egoSrfFxd! against an independent implementation of the same rule.
import GilaElectromagnetics.GilaVacuum: quadOrd, egoSrfFxd!, egoSrfAdp!,
    egoFunOut!, srfKer, srfSum!, srfScl, cubFac, facPar, gauQud

const qdOpt = CPUKerOpt{Float64}()
const qdFpr = facPar()
const qdVac = GilaElectromagnetics.GilaVacuum

qdSep(sep, dir, scl) = Float64.(sep .* dir .* scl)
qdKScl(scl) = 2π * Float64(maximum(scl))

function qdAsm(srfVec)
    ego = zeros(ComplexF64, 3, 3)
    srfSum!(ego, srfVec)
    return ego
end

# face pair vector from the library fixed rule
function qdFxd(sep, scl, ord)
    fac = Float64.(cubFac(scl))
    srfVec = zeros(ComplexF64, 36)
    egoSrfFxd!(sep[1], sep[2], sep[3], srfVec, fac, fac, 1:36, qdFpr,
        srfScl(Float64.(scl), Float64.(scl)), qdOpt, ord)
    return srfVec
end

# face pair vector from the library adaptive rule, at production tolerances
function qdAdp(sep, scl)
    fac = Float64.(cubFac(scl))
    srfVec = zeros(ComplexF64, 36)
    egoSrfAdp!(sep[1], sep[2], sep[3], srfVec, fac, fac, 1:36, qdFpr,
        srfScl(Float64.(scl), Float64.(scl)), qdOpt)
    return srfVec
end

# adaptive reference, tolerances well below the accuracy under test
function qdRef(sep, scl; rtol=1e-11, atol=1e-13, prs=1:36)
    fac = Float64.(cubFac(scl))
    sclVec = srfScl(Float64.(scl), Float64.(scl))
    srfVec = zeros(ComplexF64, 36)
    Threads.@threads for fp ∈ prs
        ker = ordVec -> srfKer(ordVec, sep[1], sep[2], sep[3], fp, fac, fac,
            qdFpr, qdOpt)
        srfVec[fp] = sclVec[fp] * qdVac.hcubature(ker,
            qdVac.SVector(0.0, 0.0, 0.0, 0.0),
            qdVac.SVector(1.0, 1.0, 1.0, 1.0); rtol=rtol, atol=atol)[1]
    end
    return srfVec
end

# tensor Gauss-Legendre reference, written out independently of egoSrfFxd!
function qdGlRef(sep, scl, ord; prs=1:36)
    qud = gauQud(ord)
    pos = (qud[:, 1] .+ 1) ./ 2
    wgt = qud[:, 2] ./ 2
    fac = Float64.(cubFac(scl))
    sclVec = srfScl(Float64.(scl), Float64.(scl))
    srfVec = zeros(ComplexF64, 36)
    Threads.@threads for fp ∈ prs
        acc = zero(ComplexF64)
        for itrI ∈ 1:ord, itrJ ∈ 1:ord, itrK ∈ 1:ord, itrL ∈ 1:ord
            acc += (wgt[itrI] * wgt[itrJ] * wgt[itrK] * wgt[itrL]) *
                srfKer(qdVac.SVector(pos[itrI], pos[itrJ], pos[itrK],
                    pos[itrL]), sep[1], sep[2], sep[3], fp, fac, fac, qdFpr,
                    qdOpt)
        end
        srfVec[fp] = acc * sclVec[fp]
    end
    return srfVec
end

@testset "Quadrature order schedule" begin
    kFin = qdKScl((1//32, 1//32, 1//32))
    @test [quadOrd(sep, kFin) for sep ∈ (2, 3, 4, 5, 6, 7, 16, 17, 32)] ==
        [9, 7, 7, 6, 6, 5, 5, 4, 4]
    # cell size enters only through the floor
    @test quadOrd(32, qdKScl((1//16, 1//16, 1//16))) == 4
    @test quadOrd(32, qdKScl((1//8, 1//8, 1//8))) == 5
    @test quadOrd(32, qdKScl((1//6, 1//6, 1//6))) == 6
    @test quadOrd(32, qdKScl((1//4, 1//4, 1//4))) == 6
    @test quadOrd(2, qdKScl((1//4, 1//4, 1//4))) == 9
    # coarser than λ/4 warns, it does not error
    kCrs = qdKScl((1//2, 1//2, 1//2))
    @test (@test_logs (:warn, r"coarser than λ/4") match_mode=:any quadOrd(32, kCrs)) == 6
end

@testset "Coarse cell warning on build" begin
    # a separated pair of 1//2 λ cells, so every cell pair takes the fixed rule
    trgVol = GlaVol((2,2,2), (1//2,1//2,1//2), (0//1,0//1,0//1))
    srcVol = GlaVol((2,2,2), (1//2,1//2,1//2), (4//1,0//1,0//1))
    lgr = Test.TestLogger()
    mem = Base.CoreLogging.with_logger(lgr) do
        GlaVacOprMem(qdOpt, trgVol, srcVol)
    end
    @test count(rec -> occursin("coarser than λ/4", rec.message), lgr.logs) == 1
    @test all(isfinite, first(mem.egoFur))
end

@testset "Fixed rule against adaptive reference" begin
    #= The production adaptive tolerances stop after a single Genz-Malik step
    at these separations, which is where the old 1e-6 level error came from, so
    the fixed rule is also required to be far closer to the reference than the
    path it replaces. =#
    scl = (1//32, 1//32, 1//32)
    for (sep, dir) ∈ ((4, (1,0,0)), (8, (1,0,0)), (8, (1,1,1)), (16, (1,0,0)))
        off = qdSep(sep, dir, scl)
        ref = qdAsm(qdRef(off, scl))
        errFxd = frbErr(qdAsm(qdFxd(off, scl, quadOrd(sep, qdKScl(scl)))), ref)
        errAdp = frbErr(qdAsm(qdAdp(off, scl)), ref)
        @test errFxd < 1e-10
        @test errFxd < errAdp / 100
    end
end

@testset "Fixed rule against high order rule" begin
    ord(sep, scl) = quadOrd(sep, qdKScl(scl))
    for scl ∈ ((1//32, 1//32, 1//32), (1//4, 1//4, 1//4))
        for (sep, dir) ∈ ((2, (1,0,0)), (3, (1,0,0)), (3, (1,1,1)))
            off = qdSep(sep, dir, scl)
            @test frbErr(qdAsm(qdFxd(off, scl, ord(sep, scl))),
                qdAsm(qdGlRef(off, scl, 24))) < 1e-10
        end
    end
    # egoSrfFxd! reproduces the same rule written out by hand, up to the
    # summation order, which srfSum! cancellation lifts above round-off
    scl = (1//32, 1//32, 1//32)
    off = qdSep(6, (1,1,0), scl)
    @test frbErr(qdAsm(qdFxd(off, scl, 6)), qdAsm(qdGlRef(off, scl, 6))) < 1e-11
end

@testset "egoFunOut! quadrature routing" begin
    scl = (1//32, 1//32, 1//32)
    fac = Float64.(cubFac(scl))
    ego = zeros(ComplexF64, 3, 3)
    # separated cells take the fixed rule
    off = qdVac.SVector(qdSep(6, (1,1,0), scl)...)
    egoFunOut!(ego, off, scl, scl, fac, fac, qdFpr, qdOpt)
    @test frbErr(ego, qdAsm(qdFxd(off, scl, quadOrd(6, qdKScl(scl))))) < 1e-14
    # cells in contact keep the adaptive rule
    off = qdVac.SVector(qdSep(1, (1,0,0), scl)...)
    egoFunOut!(ego, off, scl, scl, fac, fac, qdFpr, qdOpt)
    @test frbErr(ego, qdAsm(qdAdp(off, scl))) < 1e-14
end

# Phase 7: the uncorrected face pairs of the touching shell take the fixed rule
import GilaElectromagnetics.GilaVacuum: egoFunSng!, cntOrd
import GilaElectromagnetics.GilaVolumes: sepGrd

# minimum distance between the two faces of a pair, the faces being axis aligned
# rectangles whose difference set is a box
function qdFacGap(fp, scl, off)
    fac = Float64.(cubFac(scl))
    dst = 0.0
    for dir ∈ 1:3
        trg = extrema(fac[dir, :, qdFpr[1, fp]])
        src = extrema(fac[dir, :, qdFpr[2, fp]])
        dst += max(0.0, trg[1] - src[2] + off[dir], src[1] - trg[2] - off[dir])^2
    end
    return sqrt(dst)
end

# the face pairs egoFunSng! leaves to quadrature are exactly those whose faces
# do not touch, which is what the mask tables encode
qdUnm(scl, off) = [fp for fp ∈ 1:36 if qdFacGap(fp, scl, off) > 0]

# assemble only the pairs in prs
function qdAsmPrs(srfVec, prs)
    slc = zeros(ComplexF64, 36)
    for fp ∈ prs; slc[fp] = srfVec[fp]; end
    return qdAsm(slc)
end

# assembled contribution of the uncorrected pairs alone: zero weights send every
# corrected pair to zero
function qdSngUnm(vol, posInd)
    ego = zeros(ComplexF64, 3, 3)
    wZro = zeros(ComplexF64, 9)
    egoFunSng!(ego, posInd, wZro, wZro, wZro, sepGrd(vol, vol, 0), vol,
        Float64.(cubFac(vol.scl)), Float64.(cubFac(vol.scl)), qdFpr, qdOpt)
    return ego
end

@testset "Touching shell uncorrected pairs" begin
    vol = mkVol((4,4,4))
    scl = vol.scl
    for posInd ∈ CartesianIndices((1:2, 1:2, 1:2))
        off = (Tuple(posInd) .- 1) .* Float64.(scl)
        prs = qdUnm(scl, off)
        # every uncorrected pair sits a full cell away, which is what fixes the order
        @test minimum(qdFacGap(fp, scl, off) for fp ∈ prs) ≈ Float64(scl[1])
        libEgo = qdSngUnm(vol, posInd)
        # the library uses the fixed rule at cntOrd on exactly these pairs
        @test frbErr(libEgo, qdAsmPrs(qdFxd(off, scl, cntOrd), prs)) < 1e-14
        # and lands on a high order reference well inside the old 1e-6 accuracy
        @test frbErr(libEgo, qdAsmPrs(qdGlRef(off, scl, 20; prs=prs), prs)) < 1e-9
    end
    # adaptive reference for the cheapest offset, six pairs at rtol 1e-10
    off = (0.0, 0.0, 0.0)
    prs = qdUnm(scl, off)
    @test length(prs) == 6
    @test frbErr(qdSngUnm(vol, CartesianIndex(1,1,1)),
        qdAsmPrs(qdRef(off, scl; rtol=1e-10, atol=1e-13, prs=prs), prs)) < 1e-9
end

@testset "Operator build with the fixed rule" begin
    for mem ∈ (_selfMem4, _extMem4)
        @test all(all(isfinite, fur) for fur ∈ mem.egoFur)
        @test all(isfinite, egoOpr!(mem, ones(ComplexF64, 4, 4, 4, 3)))
    end
end

# Phase 6b: batched fixed-rule fill on a KernelAbstractions backend. Only the
# CPU backend runs here, this machine having no working CUDA; the CUDA launch is
# the same call with a CUDABackend, so what is checked below is the kernel, the
# offset classification and all of the host assembly.
import GilaElectromagnetics.GilaVacuum: egoSrfFxdBat!, egoSrfFxdAsm!, egoSlfSpl,
    egoExtSpl, egoFunExt!, egoFunExtCnt!, genEgoExt!, genCntVol, genEgoSlf!,
    thrCubFil!
import GilaElectromagnetics.GilaVolumes: sepGrd
using KernelAbstractions

# run the batch on the CPU backend and hand back the host 36×N block
function qdBat(batSep, batOrd, sclTrg, sclSrc; chnSze=65536)
    trgFac = Float64.(cubFac(sclTrg))
    srcFac = Float64.(cubFac(sclSrc))
    out = KernelAbstractions.allocate(CPU(), ComplexF64, 36, length(batOrd))
    egoSrfFxdBat!(out, batSep, batOrd, trgFac, srcFac, qdFpr,
        srfScl(Float64.(sclTrg), Float64.(sclSrc)), qdVac.frqPhz(qdOpt), CPU();
        chnSze=chnSze)
    KernelAbstractions.synchronize(CPU())
    return Array(out)
end

# largest assembled disagreement between the batch and the per-cell fixed rule
function qdBatErr(batSep, batOrd, scl; chnSze=65536)
    fac = Float64.(cubFac(scl))
    sclVec = srfScl(Float64.(scl), Float64.(scl))
    outHst = qdBat(batSep, batOrd, scl, scl; chnSze=chnSze)
    errs = Float64[]
    for k ∈ eachindex(batOrd)
        srfVec = zeros(ComplexF64, 36)
        egoSrfFxd!(batSep[1,k], batSep[2,k], batSep[3,k], srfVec, fac, fac, 1:36,
            qdFpr, sclVec, qdOpt, batOrd[k])
        push!(errs, frbErr(qdAsm(view(outHst, :, k)), qdAsm(srfVec)))
    end
    return maximum(errs)
end

@testset "Batched fill against the per-cell rule" begin
    scl = stdScl
    slfVol = mkVol((4,4,4))
    slfGrd = sepGrd(slfVol, slfVol, 0)
    slfSpc = CartesianIndices((1:4, 1:4, 1:4))
    batPos, batSep, batOrd, hstPos = egoSlfSpl(slfSpc, slfGrd, scl, qdVac.frqPhz(qdOpt))
    # the split covers every position exactly once, and the batch never touches
    # a cell that egoFunSng! writes
    @test sort(vcat(batPos, hstPos)) == sort(vec(collect(slfSpc)))
    @test length(batPos) + length(hstPos) == 64
    @test all(pos -> pos[1] > 2 || pos[2] > 2 || pos[3] > 2, batPos)
    @test hstPos == vec(collect(CartesianIndices((1:2, 1:2, 1:2))))
    @test qdBatErr(batSep, batOrd, scl) < 1e-12
    # a chunk size well below the offset count forces several launches
    @test qdBat(batSep, batOrd, scl, scl; chnSze=7) ==
        qdBat(batSep, batOrd, scl, scl)

    trgVol = mkVol((4,4,4); org=(8//32, 0//1, 0//1))
    sepTrg = sepGrd(trgVol, slfVol, 0)
    sepSrc = sepGrd(trgVol, slfVol, 1)
    extSpc = CartesianIndices((1:8, 1:8, 1:8))
    batPos, batSep, batOrd, hstPos = egoExtSpl(trgVol.cel, extSpc, sepTrg,
        sepSrc, trgVol.scl, slfVol.scl, qdVac.frqPhz(qdOpt), false)
    @test sort(vcat(batPos, hstPos)) == sort(vec(collect(extSpc)))
    # the zero padding planes are never batched
    @test all(pos -> pos[1] != 5 && pos[2] != 5 && pos[3] != 5, batPos)
    @test qdBatErr(batSep, batOrd, scl) < 1e-12
end

# reproduce the order of operations of the device branch of genEgoCrcExt! on the
# CPU backend, so the classification and assembly are exercised end to end
function qdExtDev(trgVol, srcVol, cntSpl)
    egoCrc = Array{ComplexF64}(undef, 3, 3, (2 .* trgVol.cel)...)
    trgFac = Float64.(cubFac(trgVol.scl))
    srcFac = Float64.(cubFac(srcVol.scl))
    sepTrg = sepGrd(trgVol, srcVol, 0)
    sepSrc = sepGrd(trgVol, srcVol, 1)
    itrSpc = CartesianIndices(axes(egoCrc)[3:5])
    batPos, batSep, batOrd, hstPos = egoExtSpl(trgVol.cel, itrSpc, sepTrg,
        sepSrc, trgVol.scl, srcVol.scl, qdVac.frqPhz(qdOpt), cntSpl)
    out = KernelAbstractions.allocate(CPU(), ComplexF64, 36, length(batPos))
    egoSrfFxdBat!(out, batSep, batOrd, trgFac, srcFac, qdFpr,
        srfScl(Float64.(trgVol.scl), Float64.(srcVol.scl)), qdVac.frqPhz(qdOpt), CPU())
    if cntSpl
        cntVol = genCntVol(trgVol, srcVol)
        egoCrcCnt = Array{ComplexF64}(undef, 3, 3, (2 .* cntVol.cel)...)
        genEgoSlf!(egoCrcCnt, cntVol, qdOpt)
        thrCubFil!(hstPos) do posItr
            egoFunExtCnt!(cntVol, view(egoCrc, :, :, posItr), egoCrcCnt, posItr,
                trgVol.cel, sepTrg, sepSrc, trgVol.scl, srcVol.scl, trgFac,
                srcFac, qdFpr, qdOpt)
        end
    else
        thrCubFil!(hstPos) do posItr
            egoFunExt!(view(egoCrc, :, :, posItr), posItr, trgVol.cel, sepTrg,
                sepSrc, trgVol.scl, srcVol.scl, trgFac, srcFac, qdFpr, qdOpt)
        end
    end
    egoSrfFxdAsm!(egoCrc, out, batPos, CPU())
    return egoCrc
end

@testset "Batched external fill matches the host fill" begin
    srcVol = mkVol((2,2,2))
    # separated pair, and a pair sharing a face so the contact branch is taken
    for (trgVol, cntSpl) ∈ ((mkVol((2,2,2); org=(8//32, 0//1, 0//1)), false),
            (mkVol((2,2,2); org=(2//32, 0//1, 0//1)), true))
        ref = Array{ComplexF64}(undef, 3, 3, (2 .* trgVol.cel)...)
        genEgoExt!(ref, trgVol, srcVol, qdOpt)
        @test frbErr(qdExtDev(trgVol, srcVol, cntSpl), ref) < 1e-12
    end
end

@testset "GPU build against CPU build" begin
    if CUDA.functional()
        vol = mkVol((4,4,4))
        memGpu = GlaVacOprMem(GPUKerOpt{Float64}(), vol)
        for (furGpu, furCpu) ∈ zip(memGpu.egoFur, _selfMem4.egoFur)
            @test frbErr(Array(furGpu), furCpu) < 1e-12
        end
    else
        @test_skip "no functional CUDA device"
    end
end

# Phase 8: the weak integrals fold terms that repeat. wekSDir replaces the four
# congruent self triangles of a rectangular face by one evaluation, and on a
# square face the two edge terms of the second diagonal split by those of the
# first; the three characteristic faces of a cubic cell are the same face.
import GilaElectromagnetics.GilaVacuum: wekS, wekE, wekV, wekSDir, wekEDir,
    wekVDir, wekSInt, wekEInt, wekGrdPts!, wekTrp

const qdOrdFld = 8
const qdSclCub = (1//32, 1//32, 1//32)
const qdSclAni = (1//32, 1//16, 1//24)
# two equal in-plane scales, a different normal scale: the x-normal face is
# square, the other two are not
const qdSclSqr = (1//16, 1//32, 1//32)

# the unfolded wekSDir: four self terms and four edge terms, halved
function qdSDirRef(dir, scl, glQud, fld::Bool)
    grdPts = Array{Float64}(undef, 3, 18)
    wekGrdPts!(dir, scl, grdPts)
    sInt(a, b, c) = wekSInt(hcat(grdPts[:,a], grdPts[:,b], grdPts[:,c]), glQud,
        qdOpt)
    eInt(a, b, c, d, e, f) = wekEInt(hcat(grdPts[:,a], grdPts[:,b], grdPts[:,c],
        grdPts[:,d], grdPts[:,e], grdPts[:,f]), glQud, qdOpt)
    grpA = ((sInt(1,2,5) + sInt(1,5,4)) + eInt(1,2,5,1,5,4)) + eInt(1,5,4,1,2,5)
    # the fold reuses the first split's edge terms for the second
    grpB = fld ? grpA :
        ((sInt(4,1,2) + sInt(4,2,5)) + eInt(4,1,2,4,2,5)) + eInt(4,2,5,4,1,2)
    return (grpA + grpB) / 2.0
end

@testset "Weak integral folding" begin
    glQud = gauQud(qdOrdFld)
    for scl ∈ (qdSclCub, qdSclAni, qdSclSqr)
        grdPts = Array{Float64}(undef, 3, 18)
        # the fold is a congruence, so it holds to rounding and not bitwise
        for dir ∈ 1:3
            ref = qdSDirRef(dir, scl, glQud, false)
            @test abs(wekSDir(dir, scl, grdPts, glQud, qdOpt) - ref) /
                abs(ref) < 1e-14
        end
    end
    # the edge fold is gated on a square face because it is wrong otherwise
    @test abs(qdSDirRef(1, qdSclAni, glQud, true) -
        qdSDirRef(1, qdSclAni, glQud, false)) /
        abs(qdSDirRef(1, qdSclAni, glQud, false)) > 1e-7

    # a cubic cell sees the same grid points in every direction, so the three
    # directional values are bitwise equal and only one is evaluated
    grdPts = Array{Float64}(undef, 3, 18)
    grdPtsB = Array{Float64}(undef, 3, 18)
    @test wekSDir(1, qdSclCub, grdPts, glQud, qdOpt) ==
        wekSDir(2, qdSclCub, grdPtsB, glQud, qdOpt) ==
        wekSDir(3, qdSclCub, grdPts, glQud, qdOpt)
    @test wekEDir(1, qdSclCub, grdPts, glQud, qdOpt) ==
        wekEDir(2, qdSclCub, grdPtsB, glQud, qdOpt) ==
        wekEDir(3, qdSclCub, grdPts, glQud, qdOpt)
    @test wekVDir(1, qdSclCub, grdPts, glQud, qdOpt) ==
        wekVDir(2, qdSclCub, grdPtsB, glQud, qdOpt) ==
        wekVDir(3, qdSclCub, grdPts, glQud, qdOpt)
    # the anisotropic cell keeps all three
    @test wekSDir(1, qdSclAni, grdPts, glQud, qdOpt) !=
        wekSDir(2, qdSclAni, grdPtsB, glQud, qdOpt)
    # the fold is picked from the in-plane Rationals: the x-normal face of
    # qdSclSqr is square, the z-normal face, which shares a scale with its own
    # normal, is not
    @test abs(wekSDir(3, qdSclSqr, grdPts, glQud, qdOpt) -
        qdSDirRef(3, qdSclSqr, glQud, true)) /
        abs(qdSDirRef(3, qdSclSqr, glQud, true)) < 1e-14
    @test abs(qdSDirRef(1, qdSclSqr, glQud, true) -
        qdSDirRef(1, qdSclSqr, glQud, false)) /
        abs(qdSDirRef(1, qdSclSqr, glQud, false)) > 1e-7
end

# regularized kernel against a BigFloat evaluation of (exp(i z) - 1) / (4π d f²)
import GilaElectromagnetics.GilaVacuum: sclEgoN
setprecision(BigFloat, 256)
function qdEgoNRef(dst, frq)
    dstBig = BigFloat(dst)
    frqBig = Complex{BigFloat}(frq)
    phs = 2 * BigFloat(π) * dstBig * frqBig
    return (exp(im * phs) - 1) / (4 * BigFloat(π) * dstBig * frqBig^2)
end

@testset "Regularized kernel" begin
    dsts = (1e-14, 1e-12, 1e-10, 1e-8, 0.99e-7, 1e-7, 1.01e-7, 1e-6, 1e-4,
        1e-2, 0.1, 0.3)
    frqs = (1.0+0.0im, 0.5+0.0im, 2.0+0.0im, 1.0+0.1im, 0.7+0.3im)
    for dst ∈ dsts, frq ∈ frqs
        ref = qdEgoNRef(dst, frq)
        @test abs(sclEgoN(dst, frq) - ref) / abs(ref) < 8 * eps(Float64)
    end
    # the removable singularity: sinc(0) = 1 leaves im / (2 f)
    @test sclEgoN(0.0, 1.0+0.0im) == im / 2
    @test sclEgoN(0.0, 1.0+0.1im) ≈ im / (2 * (1.0+0.1im)) rtol=eps(Float64)
    # no branch left at the old 1e-7 cut
    for frq ∈ frqs
        @test abs(sclEgoN(1e-7 * (1 - 1e-12), frq) -
            sclEgoN(1e-7 * (1 + 1e-12), frq)) < 1e-14
    end
end

@testset "Cached weak triple" begin
    optFld = CPUKerOpt{Float64}(1.0+0.0im, qdOrdFld, false, CPU())
    key = (qdSclAni, qdOrdFld, ComplexF64(1.0+0.0im))
    empty!(qdVac.wekMem)
    trpCld = wekTrp(qdSclAni, optFld)
    # a hit returns the stored objects themselves
    @test wekTrp(qdSclAni, optFld) === trpCld
    @test haskey(qdVac.wekMem, key)
    # the memo holds what a direct evaluation gives
    glQud = gauQud(qdOrdFld)
    celInv = ^(prod(Float64.(qdSclAni)), -1)
    @test trpCld[1] == celInv .* wekS(qdSclAni, glQud, optFld)
    @test trpCld[2] == celInv .* wekE(qdSclAni, glQud, optFld)
    @test trpCld[3] == celInv .* wekV(qdSclAni, glQud, optFld)
    # the triple is deterministic, so a cleared memo recomputes the same values
    empty!(qdVac.wekMem)
    @test wekTrp(qdSclAni, optFld) == trpCld
    # a contact build that reuses the memo is a build that recomputes it
    srcVol = mkVol((2,2,2))
    trgVol = mkVol((2,2,2); org=(2//32, 0//1, 0//1))
    egoRun() = (ego = Array{ComplexF64}(undef, 3, 3, (2 .* trgVol.cel)...);
        genEgoExt!(ego, trgVol, srcVol, optFld); ego)
    empty!(qdVac.wekMem)
    egoCld = egoRun()
    @test haskey(qdVac.wekMem, (stdScl, qdOrdFld, ComplexF64(1.0+0.0im)))
    @test egoRun() == egoCld
end

# panel integrals against a BigFloat evaluation of the original closed forms
import GilaElectromagnetics.GilaVacuum: rSrfSlf, rSrfEdgCrn, rSrfEdgFlt, sclEgo
setprecision(BigFloat, 256)
qdPi(x) = oftype(float(real(x)), π)

function qdSlfRef(la, lb, frq)
    return (1 / (48 * qdPi(la) * frq^2)) * (8 * la^3 + 8 * lb^3
    - 8 * la^2 * sqrt(la^2 + lb^2) - 8 * lb^2 * sqrt(la^2 + lb^2) -
    3 * la^2 * lb * (2 * log(la) + 2 * log(la + lb - sqrt(la^2 + lb^2)) +
    log(sqrt(la^2 + lb^2) - lb) - 5 * log(lb + sqrt(la^2 + lb^2)) -
    2 * log(lb - la + sqrt(la^2 + lb^2)) -
    2 * log(la + 2 * lb - sqrt(la^2 + 4 * lb^2)) +
    log(sqrt(la^2 + 4 * lb^2) - 2 * lb) +
    2 * log(la - 2 * lb + sqrt(la^2 + 4 * lb^2)) +
    log(2 * lb + sqrt(la^2 + 4 * lb^2)) +
    2 * log(2 * lb - la + sqrt(la^2 + 4 * lb^2)) -
    2 * log(la + 2 * lb + sqrt(la^2 + 4 * lb^2))) + 6 * la * lb^2 *
    (log(64 * one(la)) + 4 * log(lb) + 2 * log(sqrt(la^2 + lb^2) - la) +
    3 * log(la + sqrt(la^2 + lb^2)) - 3 * log(sqrt(la^2 + 4 * lb^2) - la) -
    3 * log(sqrt(la^4 + 5 * la^2 * lb^2 + 4 * lb^4) +
    la * (sqrt(la^2 + lb^2) - la - sqrt(la^2 + 4 * lb^2)))))
end

function qdCrnRef(la, lb, lc, frq)
    return (1 / (48 * qdPi(la) * frq^2)) * (8 * lb * lc *
    sqrt(lb^2 + lc^2) - 8 * lb * lc * sqrt(la^2 + lb^2 + lc^2) - 12 * la^3 *
    acot(la * lc / (la^2 + lb^2 - lb * sqrt(la^2 + lb^2 + lc^2))) +
    12 * la^3 * atan(la / lc) -
    12 * la * lc^2 * atan(la * lb / (lc * sqrt(la^2 + lb^2 + lc^2))) -
    12 * la * lb^2 * atan(la * lc / (lb * sqrt(la^2 + lb^2 + lc^2))) -
    16 * la^3 * atan(lb * lc / (la * sqrt(la^2 + lb^2 + lc^2))) +
    6 * lc^3 * atanh(lb / sqrt(lb^2 + lc^2)) -
    6 * lc * (la^2 + lc^2) * atanh(lb / sqrt(la^2 + lb^2 + lc^2)) -
    15 * la^2 * lc * log(la^2 + lc^2) - lc^3 * log(la^2 + lc^2) +
    2 * lc^3 * log(lc / (lb + sqrt(lb^2 + lc^2))) +
    6 * la^2 * lc * log(sqrt(la^2 + lb^2 + lc^2) - lb) +
    24 * la^2 * lc * log(sqrt(la^2 + lb^2 + lc^2) + lb) +
    2 * lc^3 * log(sqrt(la^2 + lb^2 + lc^2) + lb) +
    6 * la * lb * (-2 * la * log(la^2 + lb^2) -
    lc * log((lb^2 + lc^2) * (sqrt(la^2 + lb^2 + lc^2) - la)) +
    3 * lc * log(la + sqrt(la^2 + lb^2 + lc^2)) +
    la * log(sqrt(la^2 + lb^2 + lc^2) - lc) +
    3 * la * log(sqrt(la^2 + lb^2 + lc^2) + lc)) +
    2 * lb^3 * (
    log((sqrt(la^2 + lb^2 + lc^2) - lc) / (lc + sqrt(la^2 + lb^2 + lc^2))) +
    log(1 + (2 * lc * (lc + sqrt(lb^2 + lc^2))) / lb^2)))
end

# the second block of the original carries the missing 1 / frq^2
function qdFltRef(la, lb, frq)
    return (1 / (12 * qdPi(la) * frq^2)) * (-la^3 + 2 * lb^2 *
    (3 * lb + sqrt(la^2 + lb^2) - 2 * sqrt(la^2 + 4 * lb^2)) + la^2 *
    (2 * sqrt(la^2 + lb^2) - sqrt(la^2 + 4 * lb^2))) +
    (1 / (64 * qdPi(la) * frq^2)) * la * lb * (lb * (-62 * log(2 * one(la)) -
    5 * log(-la + sqrt(la^2 + lb^2)) +
    4 * log(8 * lb^2 * (-la + sqrt(la^2 + lb^2))) -
    33 * log(la + sqrt(la^2 + lb^2)) + 17 * log(-la + sqrt(la^2 + 4 * lb^2)) -
    24 * log(lb * (-la + sqrt(la^2 + 4 * lb^2))) +
    57 * log(la + sqrt(la^2 + 4 * lb^2))) +
    4 * la * (-8 * asinh(lb / la) + 6 * asinh(2 * lb / la) +
    6 * atanh(lb / sqrt(la^2 + lb^2)) + 12 * log(la) -
    13 * log(-lb + sqrt(la^2 + lb^2)) + log((-lb + sqrt(la^2 + lb^2)) / la) +
    log(la / (lb + sqrt(la^2 + lb^2))) - 7 * log(lb + sqrt(la^2 + lb^2)) -
    2 * log((lb + sqrt(la^2 + lb^2)) / la) -
    3 * log(-(((lb + sqrt(la^2 + lb^2)) *
    (2 * lb - sqrt(la^2 + 4 * lb^2))) / (la^2))) -
    3 * log((-lb + sqrt(la^2 + lb^2)) / (-2 * lb + sqrt(la^2 + 4 * lb^2))) +
    11 * log(-2 * lb + sqrt(la^2 + 4 * lb^2)) -
    3 * log((lb + sqrt(la^2 + lb^2)) / (2 * lb + sqrt(la^2 + 4 * lb^2))) +
    log(2 * lb + sqrt(la^2 + 4 * lb^2)) +
    9 * log((2 * lb + sqrt(la^2 + 4 * lb^2)) / (lb + sqrt(la^2 + lb^2))) -
    2 * log(la^2 + 2 * lb * (lb - sqrt(la^2 + lb^2)))))
end

qdOne = Complex{BigFloat}(1)
qdOptFrq(frq) = CPUKerOpt{Float64}(frq, qdOrdFld, false, CPU())

@testset "Panel Integrals" begin
    opt = qdOptFrq(1.0+0.0im)
    lb = 1 / 32
    # the new forms lose under a digit over nine decades of aspect ratio, so
    # the BigFloat reference is matched to a few eps
    for rat ∈ 10.0 .^ (-4:1:4)
        la = lb * rat
        @test rSrfSlf(la, lb, opt) ≈
            qdSlfRef(BigFloat(la), BigFloat(lb), qdOne) rtol=1e-14
        @test rSrfEdgFlt(la, lb, opt) ≈
            qdFltRef(BigFloat(la), BigFloat(lb), qdOne) rtol=1e-14
        for cat ∈ 10.0 .^ (-4:2:4)
            lc = lb * cat
            @test rSrfEdgCrn(la, lb, lc, opt) ≈
                qdCrnRef(BigFloat(la), BigFloat(lb), BigFloat(lc), qdOne) rtol=1e-14
            # wekE calls both orderings of the free edges
            @test rSrfEdgCrn(la, lb, lc, opt) ≈
                rSrfEdgCrn(la, lc, lb, opt) rtol=1e-14
        end
    end
    # every panel integrand carries a single 1 / frq^2
    la, lc = lb, lb / 32
    for frq ∈ (1.0+0.1im, 0.7+0.7im)
        optFrq = qdOptFrq(frq)
        @test rSrfSlf(la, lb, optFrq) * frq^2 ≈
            rSrfSlf(la, lb, opt) rtol=4*eps(Float64)
        @test rSrfEdgFlt(la, lb, optFrq) * frq^2 ≈
            rSrfEdgFlt(la, lb, opt) rtol=4*eps(Float64)
        @test rSrfEdgCrn(la, lb, lc, optFrq) * frq^2 ≈
            rSrfEdgCrn(la, lb, lc, opt) rtol=4*eps(Float64)
    end
    # far kernel: cispi reduces the argument exactly, so 100 wavelengths of
    # phase cost nothing. A strongly complex frq is checked near the origin
    # instead: the decay exp(-2π dst Im frq) inherits the rounding of the
    # product, which is not something the kernel can undo.
    for (dst, frq) ∈ ((100.0, 1.0+0.0im), (100.0, 2.0+0.0im),
        (100.0, 1.0+0.01im), (1.0, 1.0+0.1im), (1.0, 0.5+0.5im))
        dstBig = BigFloat(dst)
        frqBig = Complex{BigFloat}(frq)
        ref = cispi(2 * dstBig * frqBig) / (4 * BigFloat(π) * dstBig * frqBig^2)
        @test abs(sclEgo(dst, frq) - ref) / abs(ref) < 8 * eps(Float64)
    end
end
