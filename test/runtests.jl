using Random
Random.seed!(0xdeadbeef)

include("vacuum/runtests.jl")
include("operators.jl")
