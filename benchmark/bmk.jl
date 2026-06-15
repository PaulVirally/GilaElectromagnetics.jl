# Run from the `root` directory
# julia --startup-file=no --threads=auto --project=. benchmark/bmk.jl 

using BenchmarkTools, GilaElectromagnetics, CUDA, JLD2, Base.Threads

# Check if all threads are being used
function check_threads()
    available_threads = Sys.CPU_THREADS
    active_threads = Threads.nthreads()
    if active_threads < available_threads
        @warn "Julia is using only $active_threads out of $available_threads available threads.\n" *
              "Consider starting Julia with more threads using the `-t` or `--threads` flag.\n" *
              "Example: julia --startup-file=no --threads=auto --project=. benchmark/bmk.jl"
    else
        println("All available threads ($active_threads) are being used.")
    end
end

# Run thread check
check_threads()

# Test volume sizes
volSiz = [(n, n, n) for n in 2:2:32]
sclArr = (1//32, 1//32, 1//32)

# Store results
crtStatsCpu = Dict{Tuple{Int, Int, Int}, BenchmarkTools.Trial}()
crtStatsGpu = Dict{Tuple{Int, Int, Int}, BenchmarkTools.Trial}()
actStatsCpu = Dict{Tuple{Int, Int, Int}, BenchmarkTools.Trial}()
actStatsGpu = Dict{Tuple{Int, Int, Int}, BenchmarkTools.Trial}()

println("Benchmarking vacuum Green function operators...")

# CPU Benchmarks
println("CPU Benchmarks:")
for volDim in volSiz
    println("Benchmarking for volume size: $(volDim)")
    volObj = GlaVol(volDim, sclArr, (0//1, 0//1, 0//1))

    # Benchmark creation of the operator
    trialCpuCrt = @benchmark GlaOprVac($volObj) samples=5 seconds=10000 evals=1 gcsample=true
    crtStatsCpu[volDim] = trialCpuCrt

    # Use the created operator for the application benchmark
    oprVac = GlaOprVac(volObj)
    vecOne = ones(ComplexF64, prod(volDim) * 3)
    trialCpuAct = @benchmark $oprVac * $vecOne
    actStatsCpu[volDim] = trialCpuAct
end

# GPU Benchmarks
if CUDA.functional()
    println("GPU Benchmarks:")
    for volDim in volSiz
        println("Benchmarking for volume size: $(volDim)")
        volObj = GlaVol(volDim, sclArr, (0//1, 0//1, 0//1))

        # Benchmark creation of the operator
        trialGpuCrt = @benchmark GlaOprVac($volObj; useGpu=true) samples=5 seconds=10000 evals=1 gcsample=true
        crtStatsGpu[volDim] = trialGpuCrt

        # Use the created operator for the application benchmark
        oprVacGpu = GlaOprVac(volObj; useGpu=true)
        vecOneGpu = CUDA.ones(ComplexF64, prod(volDim) * 3)
        trialGpuAct = @benchmark $oprVacGpu * $vecOneGpu
        actStatsGpu[volDim] = trialGpuAct
    end
else
    @warn "CUDA is not functional. Skipping GPU benchmarks."
end

# Save results in a single JLD2 file
@save "benchmark/bmkResults.jld2" crtStatsCpu crtStatsGpu actStatsCpu actStatsGpu volSiz
