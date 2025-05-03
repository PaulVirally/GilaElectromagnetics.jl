using Test, GilaElectromagnetics, CUDA

@testset "CPU-GPU Consistency Tests for Green's Operator" begin
    volSizes = [
        (2, 2, 2),
        (4, 4, 4),
        (6, 6, 6),
        (8, 8, 8),
        (6, 4, 8),
        (8, 2, 10)
    ]
    sclArr = (1//32, 1//32, 1//32)
    orgSrc = (0//1, 0//1, 0//1)
    orgTrg = (1//1, 1//1, 1//1)  # Ensure non-overlapping target volume

    for volDim in volSizes
        println("Testing volume size: ", volDim)

        # Self Green's function
        volObj = GlaVol(volDim, sclArr, orgSrc)
        oprMemCpu = GlaVacOprMem(CPUKerOpt(), volObj)
        randVecCpu = rand(ComplexF64, oprMemCpu.srcVol.cel..., 3)
        outVecCpu = egoOpr!(oprMemCpu, randVecCpu)

        if CUDA.functional()
            println("Running GPU consistency tests for self Green's function...")

            oprMemGpu = GlaVacOprMem(GPUKerOpt(), volObj)
            randVecGpu = CUDA.zeros(ComplexF64, oprMemGpu.srcVol.cel..., 3)
            copyto!(randVecGpu, randVecCpu)
            outVecGpu = egoOpr!(oprMemGpu, randVecGpu)

            # Transfer GPU result back to CPU for comparison
            outVecGpuCpu = Array(outVecGpu)

            # Compute the maximum difference between CPU and GPU results
            diff = maximum(abs.(outVecCpu .- outVecGpuCpu))
            println("Maximum difference for self Green's function: ", diff)
            @test diff < 1e-6
        else
            println("CUDA is not functional. Skipping GPU consistency tests for self Green's function.")
        end

        # External Green's function
        volTrg = GlaVol(volDim, sclArr, orgTrg)
        oprMemExtCpu = GlaVacOprMem(CPUKerOpt(), volTrg, volObj)
        randVecExtCpu = rand(ComplexF64, oprMemExtCpu.srcVol.cel..., 3)
        outVecExtCpu = egoOpr!(oprMemExtCpu, randVecExtCpu)

        if CUDA.functional()
            println("Running GPU consistency tests for external Green's function...")

            oprMemExtGpu = GlaVacOprMem(GPUKerOpt(), volTrg, volObj)
            randVecExtGpu = CUDA.zeros(ComplexF64, oprMemExtGpu.srcVol.cel..., 3)
            copyto!(randVecExtGpu, randVecExtCpu)
            outVecExtGpu = egoOpr!(oprMemExtGpu, randVecExtGpu)

            # Transfer GPU result back to CPU for comparison
            outVecExtGpuCpu = Array(outVecExtGpu)

            # Compute the maximum difference between CPU and GPU results
            diffExt = maximum(abs.(outVecExtCpu .- outVecExtGpuCpu))
            println("Maximum difference for external Green's function: ", diffExt)
            @test diffExt < 1e-6
        else
            println("CUDA is not functional. Skipping GPU consistency tests for external Green's function.")
        end
    end
end