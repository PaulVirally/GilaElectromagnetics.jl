# GilaSolvers tests

@testset "Solver constructors" begin
    gmr = GMRESSolver()
    @test isnothing(gmr.rstItr)
    @test isnothing(gmr.maxItr)
    @test isnothing(gmr.absTol)
    @test isnothing(gmr.relTol)

    bcg = BiCGStabSolver()
    @test isnothing(bcg.maxItr)
    @test isnothing(bcg.absTol)
    @test isnothing(bcg.relTol)

    rfn = MixPrcRfn(Float32)
    @test rfn isa MixPrcRfn{Float32}
    @test rfn.innSlv isa GMRESSolver
    @test isnothing(rfn.oprLo)
    @test isnothing(rfn.maxItr)
    @test isnothing(rfn.absTol)
    @test isnothing(rfn.relTol)
    @test MixedPrecisionRefinement === MixPrcRfn
    @test MixPrcRfn(Float32; innSlv=BiCGStabSolver()).innSlv isa BiCGStabSolver
end

@testset "ini! defaults" begin
    n = 100
    v = zeros(ComplexF64, n)

    gmr = GMRESSolver()
    ini!(gmr, v)
    @test gmr.rstItr == min(20, n)
    @test gmr.maxItr == max(5000, n)
    @test gmr.absTol == zero(Float64)
    @test gmr.relTol == sqrt(eps(Float64))

    # Pre-set fields are not overwritten
    gmr2 = GMRESSolver(5, 1000, 1e-10, 1e-8)
    ini!(gmr2, v)
    @test gmr2.rstItr == 5
    @test gmr2.maxItr == 1000
    @test gmr2.absTol == 1e-10
    @test gmr2.relTol == 1e-8

    bcg = BiCGStabSolver()
    ini!(bcg, v)
    @test bcg.maxItr == n
    @test bcg.absTol == zero(Float64)
    @test bcg.relTol == sqrt(eps(Float64))

    # Pre-set fields are not overwritten
    bcg2 = BiCGStabSolver(500, 1e-9, 1e-7)
    ini!(bcg2, v)
    @test bcg2.maxItr == 500
    @test bcg2.absTol == 1e-9
    @test bcg2.relTol == 1e-7

    rfn = MixPrcRfn(Float32)
    ini!(rfn, v)
    @test rfn.maxItr == 20
    @test rfn.absTol == zero(Float64)
    @test rfn.relTol == sqrt(eps(Float64))

    # Pre-set fields are not overwritten
    rfn2 = MixPrcRfn(Float32; maxItr=7, absTol=1e-9, relTol=1e-10)
    ini!(rfn2, v)
    @test rfn2.maxItr == 7
    @test rfn2.absTol == 1e-9
    @test rfn2.relTol == 1e-10
end

@testset "MixPrcRfn precision guards" begin
    opr32 = GlaOprVac{Float32}(_g0s()) # Converted, so no integration cost
    n = size(opr32, 2)
    # Refining fp32 with fp32, and any operator/right hand side mismatch, throw
    @test_throws ArgumentError solve(opr32, zeros(ComplexF32, n), MixPrcRfn(Float32))
    @test_throws ArgumentError solve(opr32, zeros(ComplexF64, n), MixPrcRfn(Float32))
end

# G₀ is indefinite where (I - XG₀) is not, so both are worth a residual
@testset "solve residuals" begin
    for opr in (_g0(), _invSct())
        rhs = ones(ComplexF64, size(opr, 1))
        for slv in (BiCGStabSolver(), GMRESSolver())
            sol = solve(opr, rhs, slv)
            @test size(sol) == (size(opr, 2),)
            @test norm(opr * sol - rhs) < sqrt(eps(Float64)) * norm(rhs)
        end
    end
end

@testset "lstSqrHss" begin
    # Build a random upper-Hessenberg (n+1)×n matrix and rhs = β*e₁
    n = 5
    H = zeros(ComplexF64, n+1, n)
    for j in 1:n, i in 1:(j+1)
        H[i, j] = randn(ComplexF64)
    end
    β = randn(ComplexF64)
    rhs = zeros(ComplexF64, n+1)
    rhs[1] = β

    H_ref  = copy(H)
    rhs_ref = copy(rhs)
    GilaElectromagnetics.GilaSolvers.lstSqrHss(H, rhs)
    y_test = rhs[1:n]

    # Reference: dense least squares via QR
    y_ref = H_ref \ rhs_ref
    @test y_test ≈ y_ref[1:n] atol=1e-10
end

@testset "solve GPU" begin
    if CUDA.functional()
        vol = mkVol((4,4,4))
        opr = GlaOprVac{Float64}(vol; useGpu=true)
        rhs = CUDA.ones(ComplexF64, size(opr, 1))

        sol_bcg = solve(opr, rhs, BiCGStabSolver())
        @test size(sol_bcg) == (size(opr, 2),)
        @test norm(opr * sol_bcg - rhs) < sqrt(eps(Float64)) * norm(rhs)

        sol_gmr = solve(opr, rhs, GMRESSolver())
        @test size(sol_gmr) == (size(opr, 2),)
        @test norm(opr * sol_gmr - rhs) < sqrt(eps(Float64)) * norm(rhs)
    end
end
