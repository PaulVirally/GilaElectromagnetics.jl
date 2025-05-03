# Run from the `root` directory
# julia --threads=auto --project=. benchmark/pltBmk.jl 

using JLD2, Plots, Statistics, BenchmarkTools, LaTeXStrings

# Load results from JLD2 file
@load "benchmark/bmkResults.jld2" crtStatsCpu crtStatsGpu actStatsCpu actStatsGpu volSiz

# Extract sizes for plotting
volSizFlat = [vol[1] for vol in volSiz]

# Extract CPU creation statistics
cpuCrtMeans = [mean(crtStatsCpu[vol].times) / 1e9 for vol in volSiz]  # Convert to seconds
cpuCrtStds = [std(crtStatsCpu[vol].times) / 1e9 for vol in volSiz]    # Convert to seconds

# Extract GPU creation statistics (if available)
gpuCrtMeans = []
gpuCrtStds = []
if !isempty(crtStatsGpu)
    gpuCrtMeans = [mean(crtStatsGpu[vol].times) / 1e9 for vol in volSiz if haskey(crtStatsGpu, vol)]
    gpuCrtStds = [std(crtStatsGpu[vol].times) / 1e9 for vol in volSiz if haskey(crtStatsGpu, vol)]
end

# Extract CPU application statistics
cpuActMeans = [mean(actStatsCpu[vol].times) / 1e9 for vol in volSiz]  # Convert to seconds
cpuActStds = [std(actStatsCpu[vol].times) / 1e9 for vol in volSiz]    # Convert to seconds

# Extract GPU application statistics (if available)
gpuActMeans = []
gpuActStds = []
if !isempty(actStatsGpu)
    gpuActMeans = [mean(actStatsGpu[vol].times) / 1e9 for vol in volSiz if haskey(actStatsGpu, vol)]
    gpuActStds = [std(actStatsGpu[vol].times) / 1e9 for vol in volSiz if haskey(actStatsGpu, vol)]
end

# Plot creation times
crtPlt = plot(
    volSizFlat, cpuCrtMeans,
    ribbon=cpuCrtStds, label="CPU Creation", xscale=:log2,
    m=:circle,
    xticks=([1<<i for i in 1:5], [L"2^{%$i}" for i in 1:5]),
    ylabel="Time [s]", legend=:topleft, title="Creation Benchmarks"
)
if !isempty(gpuCrtMeans)
    plot!(crtPlt,
        m=:triangle,
        volSizFlat, gpuCrtMeans,
        ribbon=gpuCrtStds, label="GPU Creation"
    )
end

# Plot application times
appPlt = plot(
    volSizFlat, cpuActMeans,
    ribbon=cpuActStds, label="CPU Application", xscale=:log2,
    m=:circle,
    xticks=([1<<i for i in 1:5], [L"2^{%$i}" for i in 1:5]),
    xlabel="Operator Size: (n, n, n)",
    ylabel="Time [s]", legend=:topleft, title="Application Benchmarks"
)
if !isempty(gpuActMeans)
    plot!(appPlt,
        m=:triangle,
        volSizFlat, gpuActMeans,
        ribbon=gpuActStds, label="GPU Application"
    )
end

plt = plot(crtPlt, appPlt, layout=(2, 1), size=(800, 600))

# Save plot
savefig(plt, "benchmark/bmk.png")
println("Plot saved as benchmark/bmk.png")

display(plt)
println("Press Enter to exit...")
readline()
