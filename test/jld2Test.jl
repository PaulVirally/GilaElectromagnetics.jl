# JLD2 round-trips of the operator types
#= Unlike the Serialization stdlib, JLD2 is meant to be read back by a different
Julia; the round-trip therefore has to go through a real file, reproduce the
operator entry by entry, and leave no FFTW plan pointer on disk. =#
using Test, GilaElectromagnetics, LinearAlgebra, JLD2

const jldCvl = GlaCmpVol([GlaVol((2, 2, 2), scl16, stdOrg), GlaVol((2, 2, 2), scl16, extOrg)])
const jldCmp = GlaCmpOprVac{Float64}(jldCvl)

# an operator held as a field, to check the nested route against the top level
struct JldBox
    opr::Any
end

function jldRnd(obj)
    fil = tempname() * ".jld2"
    try
        jldsave(fil; obj)
        return load(fil, "obj")
    finally
        rm(fil; force=true)
    end
end

# round trip through a file and compare entrywise, not just through one matvec
function jldChk(opr; tol = 1e-12)
    desOpr = jldRnd(opr)
    @test desOpr isa typeof(opr)
    @test frbErr(dnsMat(desOpr), dnsMat(opr)) < tol
    innVec = rand(ComplexF64, size(opr, 2))
    @test norm(desOpr * innVec - opr * innVec) < tol * norm(opr * innVec)
    return desOpr
end

@testset "JLD2 extension loading" begin
    @test !isnothing(Base.get_extension(GilaElectromagnetics, :GilaJLD2Ext))
end

@testset "Vacuum operator JLD2 round-trip" begin
    for opr in (_g0s(), _gExt(), _asys(), SymGlaOprVac(_g0s()))
        jldChk(opr)
    end
    # nothing is recomputed from the file, so the Fourier data must be exact
    @test all(jldRnd(_g0s()).mem.egoFur .== _g0s().mem.egoFur)
    ovrOpr = GlaOprVac{Float64}(GlaVol((2, 2, 2), scl16, stdOrg),
        GlaVol((2, 2, 2), scl16, (1//16, 1//16, 1//16)))
    @test isoverlappingoperator(ovrOpr)
    desOvr = jldChk(ovrOpr)
    @test isoverlappingoperator(desOvr)
    @test (desOvr.srcMsk, desOvr.trgMsk) == (ovrOpr.srcMsk, ovrOpr.trgMsk)
end

@testset "Composite operator JLD2 round-trip" begin
    desOpr = jldChk(jldCmp)
    @test desOpr.srcCvl == jldCvl
    @test isselfoperator(desOpr)
    for opr in (asym(jldCmp), glaSym(jldCmp))
        desHrm = jldChk(opr)
        desDns = dnsMat(desHrm)
        @test frbErr(desDns, desDns') < 5e-16
    end
end

@testset "Scattering operator JLD2 round-trip" begin
    jldChk(_invScts())
    jldChk(_scts())
    jldChk(_glas())
end

@testset "Susceptibility operator JLD2 round-trip" begin
    isoOpr = SusOpr{Float64}(_vol2s, _sus2s)
    tenSus = zeros(ComplexF64, 2, 2, 2, 3, 3)
    for dir in 1:3
        tenSus[:, :, :, dir, dir] .= _sus2s
    end
    aniOpr = SusOpr{Float64}(_vol2s, tenSus)
    @test isoOpr.sus isa Vector
    @test aniOpr.sus isa Array{ComplexF64, 3}
    for opr in (isoOpr, aniOpr)
        @test jldChk(opr).sus == opr.sus
    end
end

@testset "Nested JLD2 round-trip" begin
    innVec = rand(ComplexF64, size(_g0s(), 2))
    refVec = _g0s() * innVec
    @test only(jldRnd([_g0s()])) * innVec == refVec
    @test jldRnd((box = JldBox(_g0s()),)).box.opr * innVec == refVec
    @test jldRnd(Dict("g" => _glas())) isa Dict{String, <:GlaOpr}
end

@testset "No FFTW plans on disk" begin
    for opr in (_g0s(), jldCmp, _scts(), SusOpr{Float64}(_vol2s, _sus2s))
        fil = tempname() * ".jld2"
        try
            jldsave(fil; opr)
            @test isnothing(findfirst(codeunits("FFTW"), read(fil)))
        finally
            rm(fil; force=true)
        end
    end
end

@testset "Kernel options survive JLD2" begin
    qssOpr = GlaOprVac{Float64}(_vol2s; qssApx=true)
    @test isquasistatic(jldChk(qssOpr))
    frqOpr = GlaOprVac{Float64}(_vol2s; frqPhz=1.0+0.1im)
    @test jldRnd(frqOpr).mem.cmpInf.frqPhz == 1.0+0.1im
    @test jldRnd(frqOpr).mem.cmpInf.genPrc === Float64
end
