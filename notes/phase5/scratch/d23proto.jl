using Test, GilaElectromagnetics, CUDA
const GVF = GilaElectromagnetics.GilaVacuum
const Q = Rational{BigInt}
@testset "proto" begin
    o = CPUKerOpt{Float64}()
    @test_throws ArgumentError CPUKerOpt{Float64}(1.0 + 0.0im, Float32, false, o.bckEnd)
    @test_throws ArgumentError (o.genPrc = Float32)
    o.genPrc = Float64
    o.frqPhz = 1.0 + 0.1im
    @test o.genPrc === Float64 && o.frqPhz == 1.0 + 0.1im
    g = GilaElectromagnetics.GilaVacuum.useGpu(o)
    @test_throws ArgumentError (g.genPrc = Float32)

    sT = (Q(3, 64), Q(1, 32), Q(1, 32)); sS = ntuple(d -> Q(1, 32), 3)
    @test_throws "integer multiples" GVF.farTnsX((Q(1), Q(0), Q(0)), sT, sS, 1.0 + 0.0im)
    @test_throws "integer multiples" GVF.farSetX(sT, sS, 1.0 + 0.0im)
end
