# GilaSolvers tests
import GilaElectromagnetics.GilaSolvers: slvPrm

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
    @test rfn.cch[] == (nothing, nothing)
    @test isnothing(rfn.maxItr)
    @test isnothing(rfn.absTol)
    @test isnothing(rfn.relTol)
    @test MixedPrecisionRefinement === MixPrcRfn
    @test MixPrcRfn(Float32; innSlv=BiCGStabSolver()).innSlv isa BiCGStabSolver
end

@testset "slvPrm defaults" begin
    n = 100
    v = zeros(ComplexF64, n)

    @test slvPrm(GMRESSolver(), v) == (rstItr=min(20, n), maxItr=max(5000, n),
        absTol=zero(Float64), relTol=sqrt(eps(Float64)))
    @test slvPrm(BiCGStabSolver(), v) ==
        (maxItr=n, absTol=zero(Float64), relTol=sqrt(eps(Float64)))
    @test slvPrm(MixPrcRfn(Float32), v) ==
        (maxItr=20, absTol=zero(Float64), relTol=sqrt(eps(Float64)))

    # Set fields are passed through
    @test slvPrm(GMRESSolver(5, 1000, 1e-10, 1e-8), v) ==
        (rstItr=5, maxItr=1000, absTol=1e-10, relTol=1e-8)
    @test slvPrm(BiCGStabSolver(500, 1e-9, 1e-7), v) ==
        (maxItr=500, absTol=1e-9, relTol=1e-7)
    @test slvPrm(MixPrcRfn(Float32; maxItr=7, absTol=1e-9, relTol=1e-10), v) ==
        (maxItr=7, absTol=1e-9, relTol=1e-10)
end

#= A solver instance carries no problem: the settings used to be written back
into the struct, and MixPrcRfn's low precision copy used to be kept across a
change of operator, so a reused solver answered the second problem with the
first one's tolerances or the first one's Green function. =#
@testset "solver reuse" begin
    opr32 = InvSctOpr(GlaOprVac{Float32}(_g0s()), ComplexF32.(_sus2s))
    opr64 = _invSct()
    rhs32 = normalize(ones(ComplexF32, size(opr32, 1)))
    rhs64 = normalize(ones(ComplexF64, size(opr64, 1)))
    for mkSlv in (BiCGStabSolver, GMRESSolver)
        slv = mkSlv()
        solve(opr32, copy(rhs32), slv)
        @test solve(opr64, copy(rhs64), slv) ≈ solve(opr64, copy(rhs64), mkSlv()) rtol=1e-12
    end

    oprAsy = InvSctOpr(_asy(), _sus4) # Same size and precision, different kernel
    rfn = MixPrcRfn(Float32)
    solve(opr64, copy(rhs64), rfn)
    @test solve(oprAsy, copy(rhs64), rfn) ≈
        solve(oprAsy, copy(rhs64), MixPrcRfn(Float32)) rtol=1e-12
end

@testset "MixPrcRfn precision guards" begin
    opr32 = GlaOprVac{Float32}(_g0s()) # Converted, so no integration cost
    n = size(opr32, 2)
    # Refining fp32 with fp32 throws; an operator/right hand side mismatch has no method
    @test_throws ArgumentError solve(opr32, zeros(ComplexF32, n), MixPrcRfn(Float32))
    @test_throws MethodError solve(opr32, zeros(ComplexF64, n), MixPrcRfn(Float32))
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
