#=
Plot creation / application times from runBmk.jl. Usage:

    julia --project=. benchmark/pltBmk.jl [result.(json|jld2)]

Defaults to the newest .json in benchmark/results/. Plots time versus size for
the cubic self operators, CPU and GPU overlaid when both are present, and
saves benchmark/bmk.png.
=#
include("bmkEnv.jl")

using BenchmarkTools
using JLD2
using Plots
using Statistics

function loadGrp(pth::AbstractString)
    if endswith(pth, ".jld2")
        return load(pth, "results")
    end
    return only(BenchmarkTools.load(pth))
end

resPth = if isempty(ARGS)
    resDir = joinpath(@__DIR__, "results")
    candLst = isdir(resDir) ?
        filter(pth -> endswith(pth, ".json") && !endswith(pth, ".meta.json"),
            readdir(resDir; join = true)) : String[]
    if isempty(candLst)
        error("no result files in $resDir; run benchmark/runbenchmarks.jl " *
            "first or pass a result file explicitly")
    end
    last(sort!(candLst; by = mtime))
else
    first(ARGS)
end
println("Plotting ", resPth)
resGrp = loadGrp(resPth)

# (n, mean seconds, std seconds) for the cubic self benchmarks of one device
function slfSrs(resGrp::BenchmarkGroup, opNam::String, devNam::String)
    (haskey(resGrp, opNam) && haskey(resGrp[opNam], devNam) &&
        haskey(resGrp[opNam][devNam], "self")) || return nothing
    ptsLst = Tuple{Int,Float64,Float64}[]
    for (key, tri) ∈ resGrp[opNam][devNam]["self"]
        dimLst = parse.(Int, split(key, "x"))
        allequal(dimLst) || continue # line plot is cubic sizes only
        tmStd = length(tri.times) > 1 ? std(tri.times) / 1e9 : 0.0
        push!(ptsLst, (first(dimLst), mean(tri.times) / 1e9, tmStd))
    end
    isempty(ptsLst) && return nothing
    return sort!(ptsLst; by = first)
end

function pltOpr(resGrp::BenchmarkGroup, opNam::String, titNam::String)
    plt = plot(xscale = :log2, yscale = :log10, ylabel = "Time [s]",
        title = titNam * " Benchmarks", legend = :topleft)
    for (devNam, mrk) ∈ (("cpu", :circle), ("gpu", :utriangle))
        srs = slfSrs(resGrp, opNam, devNam)
        isnothing(srs) && continue
        nLst = first.(srs)
        plot!(plt, nLst, getindex.(srs, 2); ribbon = getindex.(srs, 3),
            m = mrk, label = uppercase(devNam) * " " * titNam,
            xticks = (nLst, string.(nLst)))
    end
    return plt
end

crtPlt = pltOpr(resGrp, "create", "Creation")
appPlt = pltOpr(resGrp, "apply", "Application")
xlabel!(appPlt, "Self operator size: (n, n, n)")
plt = plot(crtPlt, appPlt; layout = (2, 1), size = (800, 600))

pngPth = joinpath(@__DIR__, "bmk.png")
savefig(plt, pngPth)
println("Plot saved as ", pngPth)
if isinteractive()
    display(plt)
end
