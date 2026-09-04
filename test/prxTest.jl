# Proximity warning tests
# The bound is six cells of the coarser interface scale, where the measured
# error drops under 1e-8. Touching pairs go through the contact quadrature and
# are exempt, so only a strictly positive gap under six cells warns.
const prxScl16 = (1//16, 1//16, 1//16)
const prxScl32 = (1//32, 1//32, 1//32)
const prxScl64 = (1//64, 1//64, 1//64)
const prxOrg0 = (0//1, 0//1, 0//1)

# Two (2,2,2) cubes at 1//32 whose facing faces sit gap cells apart
prxPar(gap) = (GlaVol((2,2,2), prxScl32, prxOrg0),
    GlaVol((2,2,2), prxScl32, ((2 + gap)//32, 0//1, 0//1)))

# A two region tiling of a 4×2×2 box at 1//16, its near face at lwr
prxCvl(lwr) = GlaCmpVol([GlaVol((2,2,2), prxScl16, (lwr + 1//16, 0//1, 0//1)),
    GlaVol((2,2,2), prxScl16, (lwr + 3//16, 0//1, 0//1))])

@testset "Proximity warning, plain pair" begin
    @test_logs (:warn, r"separated by 1 cell of") GlaOprVac{Float64}(prxPar(1)...)
    @test_logs (:warn, r"separated by 5 cells of") GlaOprVac{Float64}(prxPar(5)...)
    # The message names the mechanism and the remedy
    @test_logs (:warn, r"quadrature limited.*`refine`") GlaOprVac{Float64}(prxPar(2)...)
    @test_nowarn GlaOprVac{Float64}(prxPar(6)...)
    @test_nowarn GlaOprVac{Float64}(prxPar(7)...)
end

@testset "Proximity warning, exempt pairs" begin
    # Contact is handled exactly, and a self operator has no separation at all
    @test_nowarn GlaOprVac{Float64}(prxPar(0)...)
    @test_nowarn GlaOprVac{Float64}(GlaVol((2,2,2), prxScl32, prxOrg0))
end

@testset "Proximity warning counts coarse cells" begin
    #= A 3//32 gap is three cells of a 1//32 interface and six cells of a 1//64
    one, so the same physical separation warns against the coarse volume and
    passes between two fine ones. =#
    volCrs = GlaVol((2,2,2), prxScl32, prxOrg0)
    volFin = GlaVol((4,4,4), prxScl64, (5//32, 0//1, 0//1))
    @test_logs (:warn, r"by 3 cells of the coarser 1//32") GlaOprVac{Float64}(volCrs, volFin)
    @test_logs (:warn, r"by 3 cells of the coarser 1//32") GlaOprVac{Float64}(volFin, volCrs)
    finNer = GlaVol((4,4,4), prxScl64, prxOrg0)
    @test_nowarn GlaOprVac{Float64}(finNer, volFin)
end

@testset "Proximity warning, composite pair" begin
    # Four cross pairs, one warning, at the smallest gap of the four
    trgCvl, srcCvl = prxCvl(-1//8), prxCvl(1//4)
    opr = @test_logs (:warn, r"by 2 cells of the coarser 1//16") GlaCmpOprVac{Float64}(trgCvl, srcCvl)
    @test size(opr.blkMat) == (2, 2)
    @test_logs (:warn, r"by 2 cells") GlaOprVac{Float64}(trgCvl, srcCvl)
    # A compliant pair, and the self operator of a tiling whose regions touch
    @test_nowarn GlaCmpOprVac{Float64}(trgCvl, prxCvl(1//2))
    @test_nowarn GlaCmpOprVac{Float64}(trgCvl)
end

@testset "Proximity warning, adjoints" begin
    # An adjoint reuses the memory of the operator, so it must not warn again
    opr = GlaOprVac{Float64}(prxPar(2)...; prxWrn=false)
    @test_nowarn adjoint(opr)
    cmpOpr = GlaCmpOprVac{Float64}(GlaCmpVol(GlaVol((2,2,2), prxScl16, prxOrg0)),
        GlaCmpVol(GlaVol((2,2,2), prxScl16, (1//4, 0//1, 0//1))); prxWrn=false)
    @test_nowarn adjoint(cmpOpr)
end
