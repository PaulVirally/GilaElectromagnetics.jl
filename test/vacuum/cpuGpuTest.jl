# The CPU and GPU builds of an operator must agree, and a built one must move
# between the two backends. Everything here is skipped without CUDA.
using Test, GilaElectromagnetics, CUDA

# widest difference between the two backends applied to the same vector
function gpuDif(mkMem, celSrc)
    innVec = rand(ComplexF64, celSrc..., 3)
    outCpu = egoOpr!(mkMem(CPUKerOpt{Float64}()), deepcopy(innVec))
    outGpu = egoOpr!(mkMem(GPUKerOpt{Float64}()), CuArray(innVec))
    return maximum(abs, outCpu .- Array(outGpu))
end

@testset "CPU-GPU consistency" begin
    if CUDA.functional()
        for celDim in volSizes
            srcVol = mkVol(celDim)
            trgVol = mkVol(celDim; org=extOrg)
            @test gpuDif(opt -> GlaVacOprMem(opt, srcVol), celDim) < 1e-6
            @test gpuDif(opt -> GlaVacOprMem(opt, trgVol, srcVol), celDim) < 1e-6
        end
    end
end

@testset "CPU-GPU conversion" begin
    if CUDA.functional()
        vol = mkVol((4,4,4))
        memCpu = GlaVacOprMem(CPUKerOpt{Float64}(), vol)
        memGpu = GlaVacOprMem(GPUKerOpt{Float64}(), vol)
        cpuVec = ones(ComplexF64, vol.cel..., 3)
        gpuVec = CUDA.ones(ComplexF64, vol.cel..., 3)
        @test egoOpr!(memCpu, deepcopy(cpuVec)) ≈ Array(egoOpr!(memGpu, deepcopy(gpuVec)))
        # moving a mem to the backend it is already on is a no-op
        useCpu!(memGpu)
        useCpu!(memCpu)
        @test egoOpr!(memCpu, deepcopy(cpuVec)) ≈ egoOpr!(memGpu, deepcopy(cpuVec))
        useGpu!(memCpu)
        useGpu!(memGpu)
        @test egoOpr!(memCpu, deepcopy(gpuVec)) ≈ egoOpr!(memGpu, deepcopy(gpuVec))
    end
end
