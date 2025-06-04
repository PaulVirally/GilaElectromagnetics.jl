# Run from the `root` directory
# julia --startup-file=no --threads=auto --project=. benchmark/runbenchmarks.jl 
include("bmk.jl")
include("pltBmk.jl")
