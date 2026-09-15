using Test

@testset "Vacuum Tests" begin
    include("momTest.jl")
    include("farTest.jl")
    include("qssTest.jl")
    include("posDefTest.jl")
    include("cpuGpuTest.jl")
    include("serializationTest.jl")
end
