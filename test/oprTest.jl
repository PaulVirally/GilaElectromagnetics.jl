# GilaOperators tests
import GilaElectromagnetics.GilaOperators: ovrChk, mskRng, rszSus, setSus!

const oprOvrOrg = ntuple(i -> Rational(4) * stdScl[i] // 2, 3)
const oprVolOvr = GlaVol((4,4,4), stdScl, oprOvrOrg)
#= A slab and a box sharing interior, which the constructor sends through the
union route, so the operator carries a source and a target mask. =#
const oprSlb = GlaVol((2,4,4), scl16, stdOrg)
const oprBox = GlaVol((2,2,2), scl16, (1//16, 0//1, 0//1))
const oprMsk = GlaOprVac{Float64}(oprSlb, oprBox)

@testset "Constructors & predicates" begin
    gOvr = GlaOprVac{Float64}(oprVolOvr, _vol4) # Shifted two cells, so it overlaps
    for (opr, slf, ext, ovr) in ((_g0(), true, false, false), (_gExt(), false, true, false),
        (gOvr, false, false, true), (_invSct(), true, false, false),
        (_sct(), true, false, false), (_gla(), true, false, false))
        @test isselfoperator(opr) == slf
        @test isexternaloperator(opr) == ext
        @test isoverlappingoperator(opr) == ovr
        @test !isadjoint(opr)
        @test !isgpu(opr)
    end
    # Only the overlapping route carries masks
    @test !all(==(0:0), gOvr.srcMsk)
    @test !all(==(0:0), gOvr.trgMsk)
end

@testset "isgpu follows cmpInf" begin
    # useGpu!/useCpu! swap mem.cmpInf between GPUKerOpt and CPUKerOpt, so isgpu
    # only needs to check the type of the options. Swapping the options by hand
    # avoids needing a GPU; nothing here applies the operator. Fresh mem so the
    # shared _selfMem4 is not mutated.
    mem = GlaVacOprMem(CPUKerOpt{Float64}(), _vol4)
    opr = GlaOprVac(mem)
    @test !isgpu(opr)
    @test occursin("CPU", sprint(show, opr))

    mem.cmpInf = GPUKerOpt{Float64}()
    @test isgpu(opr)
    @test occursin("GPU", sprint(show, opr))

    mem.cmpInf = CPUKerOpt{Float64}()
    @test !isgpu(opr)
end

@testset "Hermitian part constructors" begin
    gSelf = _g0()
    @test asym(gSelf) isa AsyGlaOprVac
    @test glaSym(gSelf) isa SymGlaOprVac
    for opr in (_asy(), _sym())
        @test isselfoperator(opr)
        @test !isadjoint(opr)
    end
    # Neither part is defined for an operator between two different volumes
    @test_throws ArgumentError AsyGlaOprVac(_gExt())
    @test_throws ArgumentError SymGlaOprVac(_gExt())
end

@testset "Cross-constructors" begin
    invSct = _invSct()
    sct    = _sct()
    gla    = _gla()

    @test GlaOprVac(invSct) === invSct.oprVac
    @test GlaOprVac(sct)    === sct.invSctOpr.oprVac
    @test GlaOprVac(gla)    === gla.sctOpr.invSctOpr.oprVac
    @test SctOpr(gla)       === gla.sctOpr
    # Source bug: GlaOpr(::InvSctOpr) calls SctOpr(opr) without a solver,
    # but SctOpr(::InvSctOpr, ::GlaSlv) requires an explicit solver argument.
    @test_throws MethodError GlaOpr(sct.invSctOpr)

    @test InvSctOpr(sct)    === sct.invSctOpr
    @test InvSctOpr(gla)    === gla.sctOpr.invSctOpr
end

@testset "size / glaSze / eltype" begin
    n = prod((4,4,4)) * 3
    for opr in (_g0(), _gExt(), _invSct(), _sct(), _gla())
        @test eltype(opr) == ComplexF64
        @test size(opr) == (n, n)
        @test size(opr, 1) == n
        @test size(opr, 2) == n
        @test glaSze(opr, 1) == ((4,4,4)..., 3)
        @test glaSze(opr, 2) == ((4,4,4)..., 3)
    end
end

@testset "Composition identities" begin
    v = rand(ComplexF64, prod((4,4,4)) * 3)
    @test _invSct() * (_sct() * v) ≈ v
    @test _glaVac() * v ≈ _g0() * v
end

@testset "adjoint" begin
    for opr in (_g0(), _asy(), _sym(), _invSct(), _sct(), _gla())
        mat = dnsMat(opr)
        @test dnsMat(adjoint(opr)) ≈ mat'
        v = rand(ComplexF64, size(opr, 2))
        @test adjoint(adjoint(opr)) * v ≈ opr * v
    end

    # adjoint! toggles and restores (fresh mutable operator)
    g = _g0()
    adjoint!(g)
    @test isadjoint(g)
    adjoint!(g)
    @test !isadjoint(g)

    # Both Hermitian parts are their own adjoint
    for opr in (_asy(), _sym())
        @test adjoint!(opr) === opr
        mat = dnsMat(opr)
        @test mat ≈ mat'
    end
end

@testset "ovrChk / mskRng" begin
    v1 = _vol4
    # Non-overlapping
    v2 = mkVol((4,4,4); org=extOrg)
    @test !ovrChk(v1, v2)
    # Overlapping
    v3 = GlaVol((4,4,4), stdScl, (2//32, 2//32, 2//32))
    @test ovrChk(v1, v3)
    # Touching: edges meet but the volumes share no interior, so this is not
    # overlap. The external construction corrects for cell contact directly.
    v4 = GlaVol((4,4,4), stdScl, (4//32, 0//1, 0//1))
    @test !ovrChk(v1, v4)
    @test isexternaloperator(GlaOprVac{Float64}(v1, v4))
    # mskRng
    bigVol = GlaVol((8,4,4), stdScl, (2//32, 0//1, 0//1))
    rng = mskRng(v1, bigVol)
    @test length(rng) == 3
    # Misaligned sub-volume throws AssertionError
    badSub = GlaVol((4,4,4), stdScl, (1//64, 0//1, 0//1))
    @test_throws AssertionError mskRng(badSub, bigVol)
end

@testset "Touching with mismatched extents" begin
    # A small box flush against the face of a wider slab: the external contact
    # correction covers this shape, so both orientations take the external route
    # and reproduce the union of the two volumes
    box = GlaVol((2,2,2), scl16, (2//16, 0//1, 0//1))
    uni = uniVol(oprSlb, box)
    uniMat = dnsMat(GlaOprVac{Float64}(uni))
    li = LinearIndices((uni.cel..., 3))
    dofIdx(v) = (r = mskRng(v, uni); vec([li[i,j,k,d] for i in r[1], j in r[2], k in r[3], d in 1:3]))
    slbDof, boxDof = dofIdx(oprSlb), dofIdx(box)
    for (trg, src, rowDof, colDof) in ((oprSlb, box, slbDof, boxDof), (box, oprSlb, boxDof, slbDof))
        opr = GlaOprVac{Float64}(trg, src)
        @test isexternaloperator(opr)
        @test !isoverlappingoperator(opr)
        @test frbErr(dnsMat(opr), uniMat[rowDof, colDof]) < 1.5e-15
    end
    # A corner-fitting touching pair still takes the external route
    @test isexternaloperator(GlaOprVac{Float64}(_vol4, GlaVol((4,4,4), stdScl, (4//32, 0//1, 0//1))))
end

@testset "Masked operator adjoint" begin
    @test isoverlappingoperator(oprMsk)
    fwdMat = dnsMat(oprMsk)
    # The adjoint has to exchange the two masks along with the volumes
    adjMat = dnsMat(oprMsk')
    @test size(adjMat) == reverse(size(fwdMat))
    @test frbErr(adjMat, fwdMat') < 5e-16
    # In place, and the round trip back
    adjOpr = adjoint!(deepcopy(oprMsk))
    @test glaSze(adjOpr, 2) == glaSze(oprMsk, 1)
    @test frbErr(dnsMat(adjOpr), fwdMat') < 5e-16
    @test dnsMat(adjoint!(adjOpr)) == fwdMat
end

@testset "GlaOprVac on a field" begin
    opr = _g0()
    fld = discretize!(zerofield(Float64, opr.mem.srcVol), tstDns)
    out = opr * fld
    @test out isa GlaFld
    @test nregions(out.cvol) == 1
    @test regions(out.cvol)[1] == opr.mem.trgVol
    @test out.dat ≈ opr * fld.dat
    # A field over another volume, or over a tiling of several regions
    @test_throws ArgumentError opr * zerofield(Float64, GlaVol((2,2,2), stdScl, stdOrg))
    @test_throws ArgumentError opr * zerofield(Float64, GlaCmpVol(
        [GlaVol((2,2,2), stdScl, (-1//32, 0//1, 0//1)),
         GlaVol((2,2,2), stdScl, (1//32, 0//1, 0//1))]))
    # The masked route reads its input through a mask, so it takes no field
    @test_throws ArgumentError oprMsk * zerofield(Float64, oprBox)
end

@testset "rszSus" begin
    cel   = (4,4,4)
    sus3d = rand(ComplexF64, cel...)
    # 3D passthrough, 1D reshape
    @test rszSus(sus3d, cel) === sus3d
    @test rszSus(vec(sus3d), cel) == sus3d
    # Wrong length, and a rank the reshape does not cover
    @test_throws ArgumentError rszSus(zeros(ComplexF64, 5), cel)
    @test_throws ArgumentError rszSus(rand(ComplexF64, 4, 16), cel)
end

@testset "setSus!" begin
    newSus = mkSus((4,4,4); val=1.0+0.1im)
    invSct = _invSct()
    setSus!(invSct, newSus)
    @test invSct.sus == newSus
    # Wrong size throws
    @test_throws ArgumentError setSus!(invSct, mkSus((2,2,2)))
    # Propagates through SctOpr and GlaOpr
    sct = _sct()
    setSus!(sct, newSus)
    @test sct.invSctOpr.sus == newSus
    gla = _gla()
    setSus!(gla, newSus)
    @test gla.sctOpr.invSctOpr.sus == newSus
end

@testset "MulRegGlaOprVac" begin
    vols = [_vol4, _trgV4]
    op   = MulRegGlaOprVac{Float64}(vols, vols)
    n    = prod((4,4,4)) * 3

    # The diagonal is self, the off-diagonal external
    @test size(op.oprMat) == (2, 2)
    @test isselfoperator(op.oprMat[1,1])
    @test isselfoperator(op.oprMat[2,2])
    @test isexternaloperator(op.oprMat[1,2])
    @test isexternaloperator(op.oprMat[2,1])

    @test size(op) == (2n, 2n)
    @test size(op, 1) == 2n
    @test size(op, 2) == 2n

    x  = rand(ComplexF64, 2n)
    y  = op * x
    x1 = x[1:n]; x2 = x[n+1:2n]
    @test y ≈ vcat(op.oprMat[1,1] * x1 + op.oprMat[1,2] * x2,
                   op.oprMat[2,1] * x1 + op.oprMat[2,2] * x2)

    # Block-vector form
    yBlk = op * [reshape(x1, (4,4,4,3)), reshape(x2, (4,4,4,3))]
    @test vec(yBlk[1]) ≈ op.oprMat[1,1] * x1 + op.oprMat[1,2] * x2
    @test vec(yBlk[2]) ≈ op.oprMat[2,1] * x1 + op.oprMat[2,2] * x2

    D = dnsMat(op)
    @test D * x ≈ y
    @test dnsMat(adjoint(op)) ≈ D'
    @test size(adjoint(op)) == size(op)

    str = sprint(show, op)
    @test occursin("multi-region", str)
    @test occursin("2", str)
end

@testset "show" begin
    for (opr, keys) in ((_g0(),     ["Self",     "CPU", "G₀"]),
                        (_gExt(),   ["External", "CPU", "G₀"]),
                        (_asy(),    ["Self",     "CPU", "Asym(G₀)"]),
                        (_sym(),    ["Self",     "CPU", "Sym(G₀)"]),
                        (_invSct(), ["Self",     "CPU", "(I - XG₀)"]),
                        (_sct(),    ["Self",     "CPU", "(I - XG₀)⁻¹"]),
                        (_gla(),    ["Self",     "CPU", "G₀(I - XG₀)⁻¹"]))
        str = sprint(show, opr)
        for key in keys
            @test occursin(key, str)
        end
    end
    @test occursin("Adjoint", sprint(show, adjoint(_g0())))
end

@testset "Operator CPU/GPU parity" begin
    if CUDA.functional()
        sus  = _sus4
        susg = CuArray(sus)
        vcpu = rand(ComplexF64, size(_g0(), 2))
        vgpu = CuArray(vcpu)

        pairs = [
            (_g0(),                        GlaOprVac{Float64}(_vol4; useGpu=true)),
            (InvSctOpr{Float64}(_vol4, sus),        InvSctOpr{Float64}(_vol4, susg; useGpu=true)),
            (SctOpr{Float64}(_vol4, sus),           SctOpr{Float64}(_vol4, susg; useGpu=true)),
            (GlaOpr{Float64}(_vol4, sus),           GlaOpr{Float64}(_vol4, susg; useGpu=true)),
        ]
        for (cOpr, gOpr) in pairs
            @test cOpr * vcpu ≈ Array(gOpr * vgpu)
            # useGpu! on cpu opr
            useGpu!(cOpr)
            @test isgpu(cOpr)
            @test cOpr * vgpu ≈ gOpr * vgpu
            # useCpu! on gpu opr
            useCpu!(gOpr)
            @test !isgpu(gOpr)
            # No-ops
            useCpu!(gOpr)
            @test !isgpu(gOpr)
            useCpu!(cOpr)  # after useGpu!
            useGpu!(gOpr)
        end
    end
end
