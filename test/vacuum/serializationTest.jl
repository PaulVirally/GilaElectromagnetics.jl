#= The GPU round trip: the arrays are written from device memory and have to come
back as device memory that still applies. The CPU round trip is checked entrywise
by serTest.jl. =#
using Test, Serialization, GilaElectromagnetics, CUDA

@testset "GlaVacOprMem serialization GPU" begin
    if CUDA.has_cuda()
        vol = mkVol((4, 4, 4))
        mem = GlaVacOprMem(GPUKerOpt{Float64}(), vol, vol)
        tmpFil = tempname()
        try
            open(io -> serialize(io, mem), tmpFil, "w")
            desMem = open(deserialize, tmpFil, "r")
            useGpu!(desMem)
            for fld in (:genPrc, :frqPhz, :adjMod)
                @test getfield(desMem.cmpInf, fld) == getfield(mem.cmpInf, fld)
            end
            @test typeof(desMem.cmpInf.bckEnd) == typeof(mem.cmpInf.bckEnd)
            for fld in (:minScl, :trgDiv, :srcDiv, :trgCel, :srcCel, :trgPar, :srcPar)
                @test getfield(desMem.mixInf, fld) == getfield(mem.mixInf, fld)
            end
            for fld in (:trgVol, :srcVol, :egoFur, :phzInf)
                @test getfield(desMem, fld) == getfield(mem, fld)
            end
            actVec = CUDA.ones(ComplexF64, (4, 4, 4, 3))
            @test egoOpr!(mem, deepcopy(actVec)) ≈ egoOpr!(desMem, deepcopy(actVec))
        finally
            isfile(tmpFil) && rm(tmpFil)
        end
    end
end
