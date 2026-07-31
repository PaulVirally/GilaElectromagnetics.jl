using Test, Random, GilaElectromagnetics, LinearAlgebra
using LinearMaps, LinearOperators, SciMLOperators, Serialization, CUDA
import GilaElectromagnetics.GilaVolumes: uniVol

Random.seed!(0x67696c61)
include("tstHlp.jl")

@testset "GilaElectromagnetics" begin
    include("volTest.jl")
    include("vacTest.jl")
    include("slvTest.jl")
    include("oprTest.jl")
    include("linAlgTest.jl")
    include("extOpsTest.jl")
    include("physTest.jl")
end
