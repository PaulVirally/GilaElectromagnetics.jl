#=
Compare two benchmark result files. Usage:

    julia --project=. benchmark/cmpBmk.jl <old>.(json|jld2) <new>.(json|jld2) [--tolerance=0.05]

For every benchmark present in both files, prints the BenchmarkTools.judge of
the median estimates (new relative to old) and flags time regressions. Exits
with a non-zero status when any time regression is found.
=#
include("bmkEnv.jl")

using BenchmarkTools
using JLD2
using Printf

tolArg = 0.05
pthLst = String[]
for arg ∈ ARGS
    if startswith(arg, "--tolerance=")
        global tolArg = parse(Float64, last(split(arg, "=")))
    else
        push!(pthLst, arg)
    end
end
if length(pthLst) != 2
    error("usage: julia --project=. benchmark/cmpBmk.jl <old>.json <new>.json [--tolerance=0.05]")
end
oldPth, newPth = pthLst

function loadGrp(pth::AbstractString)
    if endswith(pth, ".jld2")
        return load(pth, "results")
    end
    return only(BenchmarkTools.load(pth))
end

oldGrp = loadGrp(oldPth)
newGrp = loadGrp(newPth)

oldLvs = Dict(join(keyPth, "/") => tri for (keyPth, tri) ∈ BenchmarkTools.leaves(oldGrp))
newLvs = Dict(join(keyPth, "/") => tri for (keyPth, tri) ∈ BenchmarkTools.leaves(newGrp))
shrKey = sort!(collect(intersect(keys(oldLvs), keys(newLvs))))
if isempty(shrKey)
    error("the two result files have no benchmarks in common")
end

println("old: ", oldPth)
println("new: ", newPth)
println("tolerance: ", tolArg, "\n")
@printf("%-52s %12s %12s %8s %-12s %8s %-12s\n", "benchmark", "old", "new",
    "t-ratio", "time", "m-ratio", "memory")
regLst = String[]
for key ∈ shrKey
    oldMed = median(oldLvs[key])
    newMed = median(newLvs[key])
    jdg = judge(newMed, oldMed; time_tolerance = tolArg,
        memory_tolerance = tolArg)
    @printf("%-52s %12s %12s %8.3f %-12s %8.3f %-12s\n", key,
        BenchmarkTools.prettytime(time(oldMed)),
        BenchmarkTools.prettytime(time(newMed)),
        ratio(jdg).time, string(time(jdg)),
        ratio(jdg).memory, string(memory(jdg)))
    if time(jdg) == :regression
        push!(regLst, key)
    end
end

for (nam, mssLst) ∈ (("old", setdiff(keys(newLvs), keys(oldLvs))),
    ("new", setdiff(keys(oldLvs), keys(newLvs))))
    if !isempty(mssLst)
        println("\nBenchmarks missing from the $nam file:")
        foreach(key -> println("    ", key), sort!(collect(mssLst)))
    end
end

if isempty(regLst)
    println("\nNo time regressions (tolerance = $tolArg).")
else
    println("\nTime regressions (tolerance = $tolArg):")
    foreach(key -> println("    ", key), regLst)
    exit(1)
end
