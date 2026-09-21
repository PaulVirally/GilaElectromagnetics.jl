# Serialization round-trips of the operator types
#= A round-trip has to reproduce the operator entry by entry, since nothing in
the written format is recomputed, and the copy has to apply, which only works if
the FFTW plans were rebuilt on load rather than read back as raw pointers. =#
using Test, GilaElectromagnetics, LinearAlgebra, Serialization

const serSus = 0.5 + 0.05im

#= A coarse region face to face with a region refined in x alone, so both
cross-scale blocks run the contact quadrature and are a fine mesh block. =#
const serCvl = refine(GlaCmpVol(GlaVol((4, 2, 2), scl16, stdOrg)),
    ((-1//16, 0//1, 0//1), (1//8, 1//8, 1//8)); factor=(2, 1, 1))
const serOpr = GlaCmpOprVac{Float64}(serCvl)
const serFarVol = GlaVol((2, 2, 2), scl16, (1//1, 0//1, 0//1))
#= Two volumes sharing interior, so construction folds them into their union and
the masks become the only record of the sub-volumes. =#
const serOvrA = GlaVol((2, 2, 2), scl16, stdOrg)
const serOvrB = GlaVol((2, 2, 2), scl16, (1//16, 1//16, 1//16))
# A gapped tiling: two regions a wavelength apart in one composite volume
const serGapCvl = GlaCmpVol([GlaVol((2, 2, 2), scl16, stdOrg),
    GlaVol((2, 2, 2), scl16, extOrg)])
serFld() = discretize!(zerofield(Float64, serCvl), tstDns)

@testset "Vacuum operator serialization" begin
    for opr in (_g0s(), _gExt(), _asys(), SymGlaOprVac(_g0s()))
        serChk(opr)
    end
    # An operator nested in a container takes the same route
    tmpFil = tempname()
    try
        open(tmpFil, "w") do io; serialize(io, [_g0s()]); end
        @test isnothing(findfirst(codeunits("FFTW"), read(tmpFil)))
        innVec = rand(ComplexF64, size(_g0s(), 2))
        @test only(open(deserialize, tmpFil)) * innVec ≈ _g0s() * innVec
    finally
        rm(tmpFil; force=true)
    end
end

@testset "Overlapping operator serialization" begin
    ovrOpr = GlaOprVac{Float64}(serOvrA, serOvrB)
    @test isoverlappingoperator(ovrOpr)
    # Without the masks the copy reads back as the self operator on the union
    @test size(serRnd(ovrOpr)) == size(ovrOpr)
    desOpr = serChk(ovrOpr)
    @test isoverlappingoperator(desOpr)
    @test (desOpr.srcMsk, desOpr.trgMsk) == (ovrOpr.srcMsk, ovrOpr.trgMsk)
end

@testset "Composite operator serialization" begin
    @test count(blk -> blk isa GlaSnd, serOpr.blkMat) == 2
    desOpr = serChk(serOpr)
    @test nregions(desOpr.srcCvl) == 2
    @test desOpr.srcCvl == serCvl
    @test isselfoperator(desOpr)
    @test count(blk -> blk isa GlaSnd, desOpr.blkMat) == 2
    # A field on the original tiling still applies, the tilings compare equal
    @test (desOpr * serFld()).dat == (serOpr * serFld()).dat
    # Two bodies, so the block matrix is not square
    serChk(GlaCmpOprVac{Float64}(serCvl, GlaCmpVol(serFarVol)))
    # One tiling holding two separated regions
    serChk(GlaCmpOprVac{Float64}(serGapCvl))
    # No raw FFTW plan pointers reach the written composite operator either
    tmpFil = tempname()
    try
        open(tmpFil, "w") do io; serialize(io, serOpr); end
        @test isnothing(findfirst(codeunits("FFTW"), read(tmpFil)))
    finally
        rm(tmpFil; force=true)
    end
end

#= The parts hold the transformed Fourier coefficients of their blocks, so the
reader must not take the part again. =#
@testset "Composite Hermitian part serialization" begin
    for opr in (asym(serOpr), glaSym(serOpr))
        desOpr = serChk(opr)
        @test GilaElectromagnetics.adjoint!(desOpr) === desOpr
        desDns = dnsMat(desOpr)
        @test frbErr(desDns, desDns') < 5e-16
    end
    # The imaginary part taken twice is not the imaginary part
    @test frbErr(dnsMat(serRnd(asym(serOpr))), asymMat(dnsMat(serOpr))) < 5e-15
end

# The kind of vacuum operator is tagged in the stream, so every kind reads back
@testset "Inverse scattering operator kinds" begin
    for invSct in (_invScts(), InvSctOpr(_asys(), _sus2s),
        InvSctOpr(serOpr, serSus))
        desInv = serChk(invSct)
        @test desInv.oprVac isa typeof(invSct.oprVac)
        @test sus(desInv).sus == sus(invSct).sus
    end
end

@testset "Scattering operator serialization" begin
    for invSct in (_invScts(), InvSctOpr(serOpr, serSus))
        serChk(SctOpr(invSct, BiCGStabSolver()))
        serChk(GlaOpr(SctOpr(invSct, BiCGStabSolver())))
    end
    # The composite application path, on a field and on the flat vector
    invSct = InvSctOpr(serOpr, serSus)
    desInv = serRnd(invSct)
    @test (desInv * serFld()).dat == (invSct * serFld()).dat
    fldDat = collect(serFld().dat)
    @test desInv * fldDat == invSct * fldDat
    # No raw FFTW plan pointers reach the written scattering operator either
    tmpFil = tempname()
    try
        open(tmpFil, "w") do io; serialize(io, SctOpr(invSct, BiCGStabSolver())); end
        @test isnothing(findfirst(codeunits("FFTW"), read(tmpFil)))
    finally
        rm(tmpFil; force=true)
    end
end

@testset "Susceptibility operator serialization" begin
    isoOpr = SusOpr{Float64}(_vol2s, _sus2s)
    tenSus = zeros(ComplexF64, 2, 2, 2, 3, 3)
    for dir in 1:3
        tenSus[:, :, :, dir, dir] .= _sus2s
    end
    aniOpr = SusOpr{Float64}(_vol2s, tenSus)
    @test isoOpr.sus isa Vector
    @test aniOpr.sus isa Array{ComplexF64, 3}
    serChk(isoOpr)
    serChk(aniOpr)
    # Nested in a container takes the same route as the top level
    tmpFil = tempname()
    try
        open(tmpFil, "w") do io; serialize(io, [isoOpr, aniOpr]); end
        desIso, desAni = open(deserialize, tmpFil)
        @test desIso isa typeof(isoOpr) && desIso.sus == isoOpr.sus
        @test desAni isa typeof(aniOpr) && desAni.sus == aniOpr.sus
    finally
        rm(tmpFil; force=true)
    end
end
