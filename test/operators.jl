using Test, GilaElectromagnetics, LinearAlgebra, CUDA

# Test volume sizes and susceptibility
volSizes = [
    (2, 2, 2),
    (4, 4, 4),
    (6, 6, 6),
    (8, 8, 8),
    (6, 4, 8),
    (8, 2, 10)
]
sclArr = (1//32, 1//32, 1//32)

# Helper function for dense matrix extraction
function dnsMat(G::GlaOprVac)
    mat = zeros(eltype(G), size(G, 1), size(G, 2))
    for i in 1:size(G, 2)
        v = zeros(eltype(G), size(G, 2))
        v[i] = one(eltype(G))
        mat[:, i] .= G * v
    end
    return mat
end

@testset "Gila Operator Tests" begin
    @testset "GlaOprVac Tests" begin
        for volDim in volSizes
            # println("Testing simple GlaOprVac functionality for volume size: $(volDim)")
            volObj = GlaVol(volDim, sclArr, (0//1, 0//1, 0//1))

            # Test self Green's function operator
            oprVac = GlaOprVac(volObj)
            @test oprVac isa GlaOprVac
            @test isselfoperator(oprVac)
            @test !isexternaloperator(oprVac)

            # Test external Green's function operator
            trgVol = GlaVol(volDim, sclArr, (1//1, 1//1, 1//1))
            oprVacExt = GlaOprVac(trgVol, volObj)
            @test oprVacExt isa GlaOprVac
            @test !isselfoperator(oprVacExt)
            @test isexternaloperator(oprVacExt)

            # Test GPU operator if CUDA is functional
            if CUDA.functional()
                oprVacGpu = GlaOprVac(volObj; useGpu=true)
                @test oprVacGpu isa GlaOprVac
                @test oprVacGpu.mem.cmpInf isa GPUKerOpt
            end
        end
    end

    @testset "Operator Consistency Tests" begin
        for volDim in volSizes
            # println("Testing operator consistency for volume size: $(volDim)")
            volObj = GlaVol(volDim, sclArr, (0//1, 0//1, 0//1))
            vac = zeros(ComplexF64, volDim...)  # Zero susceptibility
            sus = ones(ComplexF64, volDim...) * (0.5 + 0.05im)  # Non-zero susceptibility
            vecOnes = ones(ComplexF64, prod(volDim) * 3)  # Vector of ones

            # Test GlaOpr behaves like GlaOprVac for zero susceptibility
            glaOprVac = GlaOprVac(volObj)
            glaOpr = GlaOpr(volObj, vac)
            @test glaOpr * vecOnes ≈ glaOprVac * vecOnes

            # Test scattering operator and inverse scattering operator consistency
            invSctOpr = InvSctOpr(volObj, sus)
            sctOpr = SctOpr(volObj, sus)
            @test invSctOpr * (sctOpr * vecOnes) ≈ vecOnes

            # Test adjoint of adjoint consistency
            adjOprVac = adjoint(adjoint(glaOprVac))
            adjInvSctOpr = adjoint(adjoint(invSctOpr))
            adjSctOpr = adjoint(adjoint(sctOpr))
            adjGlaOpr = adjoint(adjoint(glaOpr))
            @test adjOprVac * vecOnes ≈ glaOprVac * vecOnes
            @test adjInvSctOpr * vecOnes ≈ invSctOpr * vecOnes
            @test adjSctOpr * vecOnes ≈ sctOpr * vecOnes
            @test adjGlaOpr * vecOnes ≈ glaOpr * vecOnes

            # Test GPU operators if CUDA is functional
            if CUDA.functional()
                sus = CuArray(sus)
                glaOprGpu = GlaOpr(volObj, sus; useGpu=true)
                glaOprVacGpu = GlaOprVac(volObj; useGpu=true)
                vecOnesGpu = CUDA.ones(ComplexF64, prod(volDim) * 3)

                # Test GlaOpr behaves like GlaOprVac for zero susceptibility
                @test glaOprGpu * vecOnesGpu ≈ glaOprVacGpu * vecOnesGpu

                # Test scattering operator and inverse scattering operator consistency
                invSctOprGpu = InvSctOpr(volObj, sus; useGpu=true)
                sctOprGpu = SctOpr(volObj, sus; useGpu=true)
                @test invSctOprGpu * (sctOprGpu * vecOnesGpu) ≈ vecOnesGpu

                # Test adjoint of adjoint consistency
                adjOprVacGpu = adjoint(adjoint(glaOprVacGpu))
                adjInvSctOprGpu = adjoint(adjoint(invSctOprGpu))
                adjSctOprGpu = adjoint(adjoint(sctOprGpu))
                adjGlaOprGpu = adjoint(adjoint(glaOprGpu))
                @test adjOprVacGpu * vecOnesGpu ≈ glaOprVacGpu * vecOnesGpu
                @test adjInvSctOprGpu * vecOnesGpu ≈ invSctOprGpu * vecOnesGpu
                @test adjSctOprGpu * vecOnesGpu ≈ sctOprGpu * vecOnesGpu
                @test adjGlaOprGpu * vecOnesGpu ≈ glaOprGpu * vecOnesGpu
            end
        end
    end

    @testset "Operator Consistency and Repeated Application Tests" begin
        for volDim in volSizes
            # println("Testing repeated operator use for volume size: $(volDim)")
            volObj = GlaVol(volDim, sclArr, (0//1, 0//1, 0//1))
            sus = ones(ComplexF64, volDim...) * (0.5 + 0.05im)  # Non-zero susceptibility
            vecOnes = ones(ComplexF64, prod(volDim) * 3)  # Vector of ones

            # Test GlaOprVac
            glaOprVac = GlaOprVac(volObj)
            result1 = glaOprVac * vecOnes
            result2 = glaOprVac * vecOnes
            @test result1 ≈ result2
            @test all(vecOnes .== one(eltype(vecOnes))) 

            # Test InvSctOpr
            invSctOpr = InvSctOpr(volObj, sus)
            result1 = invSctOpr * vecOnes
            result2 = invSctOpr * vecOnes
            @test result1 ≈ result2
            @test all(vecOnes .== one(eltype(vecOnes)))

            # Test SctOpr
            sctOpr = SctOpr(volObj, sus)
            result1 = sctOpr * vecOnes
            result2 = sctOpr * vecOnes
            @test result1 ≈ result2
            @test all(vecOnes .== one(eltype(vecOnes)))

            # Test GlaOpr
            glaOpr = GlaOpr(volObj, sus)
            result1 = glaOpr * vecOnes
            result2 = glaOpr * vecOnes
            @test result1 ≈ result2
            @test all(vecOnes .== one(eltype(vecOnes)))

            # Test GPU operators if CUDA is functional
            if CUDA.functional()
                sus = CuArray(sus)
                vecOnesGpu = CUDA.ones(ComplexF64, prod(volDim) * 3)

                # Test GlaOprVac on GPU
                glaOprVacGpu = GlaOprVac(volObj; useGpu=true)
                result1 = glaOprVacGpu * vecOnesGpu
                result2 = glaOprVacGpu * vecOnesGpu
                @test result1 ≈ result2
                @test all(vecOnesGpu .== one(eltype(vecOnesGpu)))

                # Test InvSctOpr on GPU
                invSctOprGpu = InvSctOpr(volObj, sus; useGpu=true)
                result1 = invSctOprGpu * vecOnesGpu
                result2 = invSctOprGpu * vecOnesGpu
                @test result1 ≈ result2
                @test all(vecOnesGpu .== one(eltype(vecOnesGpu)))

                # Test SctOpr on GPU
                sctOprGpu = SctOpr(volObj, sus; useGpu=true)
                result1 = sctOprGpu * vecOnesGpu
                result2 = sctOprGpu * vecOnesGpu
                @test result1 ≈ result2
                @test all(vecOnesGpu .== one(eltype(vecOnesGpu)))

                # Test GlaOpr on GPU
                glaOprGpu = GlaOpr(volObj, sus; useGpu=true)
                result1 = glaOprGpu * vecOnesGpu
                result2 = glaOprGpu * vecOnesGpu
                @test result1 ≈ result2
                @test all(vecOnesGpu .== one(eltype(vecOnesGpu)))
            end
        end
    end

    @testset "Solver Tests" begin
        for volDim in volSizes
            # println("Testing solvers for volume size: $(volDim)")
            volObj = GlaVol(volDim, sclArr, (0//1, 0//1, 0//1))
            oprSlf = GlaOprVac(volObj)

            # Create a random right-hand side vector
            rhs = ones(ComplexF64, size(oprSlf, 1))

            # Test BiCGStabSolver
            solver = BiCGStabSolver()
            solution = solve(oprSlf, rhs, solver)
            @test size(solution) == (size(oprSlf, 2),)
            # Verify that the solution satisfies A * x ≈ b
            residual = norm(oprSlf * solution - rhs)
            @test residual < sqrt(eps(real(eltype(solution)))) * norm(rhs)

            # Test GMRESSolver
            solver = GMRESSolver()
            solution = solve(oprSlf, rhs, solver)
            @test size(solution) == (size(oprSlf, 2),)
            # Verify that the solution satisfies A * x ≈ b
            residual = norm(oprSlf * solution - rhs)
            @test residual < sqrt(eps(real(eltype(solution)))) * norm(rhs)

            # Test GPU solvers if CUDA is functional
            if CUDA.functional()
                oprSlfGpu = GlaOprVac(volObj; useGpu=true)
                rhsGpu = CUDA.ones(ComplexF64, size(oprSlfGpu, 1))

                # Test BiCGStabSolver on GPU
                solver = BiCGStabSolver()
                solutionGpu = solve(oprSlfGpu, rhsGpu, solver)
                @test size(solutionGpu) == (size(oprSlfGpu, 2),)
                # Verify that the solution satisfies A * x ≈ b
                residualGpu = norm(oprSlfGpu * solutionGpu - rhsGpu)
                @test residualGpu < sqrt(eps(real(eltype(solutionGpu)))) * norm(rhsGpu)

                # Test GMRESSolver on GPU
                solver = GMRESSolver()
                solutionGpu = solve(oprSlfGpu, rhsGpu, solver)
                @test size(solutionGpu) == (size(oprSlfGpu, 2),)
                # Verify that the solution satisfies A * x ≈ b
                residualGpu = norm(oprSlfGpu * solutionGpu - rhsGpu)
                @test residualGpu < sqrt(eps(real(eltype(solutionGpu)))) * norm(rhsGpu)
            end
        end
    end
end
