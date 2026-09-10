# glaMatFreExtOps tests
# Uses _g0s(), _invScts(), _scts(), _glas() (2,2,2) from tstHlp.jl for cheap loops
import GilaElectromagnetics.GilaOperators: invMul!, invMulAdj!

@testset "matrix-free wrappers" begin
    for wrp in (FunctionOperator, LinearOperator, LinearMap),
        opr in (_g0s(), _invScts(), _scts(), _glas())
        wrpOpr = wrp(opr)
        n, m = size(opr, 2), size(opr, 1)
        inp, out = rand(ComplexF64, n), rand(ComplexF64, m)
        @test size(wrpOpr) == (m, n)
        @test eltype(wrpOpr) == ComplexF64

        fwd = similar(out)
        mul!(fwd, wrpOpr, inp)
        @test fwd ≈ opr * inp

        adj = similar(inp)
        mul!(adj, wrpOpr', out)
        @test adj ≈ adjoint(opr) * out
    end
    # Per-wrapper extras
    lnOpr = LinearOperator(_g0s())
    @test !lnOpr.symmetric
    @test !lnOpr.hermitian
    @test Matrix(LinearMap(_g0s())) ≈ dnsMat(_g0s())
end

@testset "invMul! and invMulAdj!" begin
    v = rand(ComplexF64, size(_invScts(), 2))

    # SctOpr: invMul! solves sctOpr * w = v → checks opr*w ≈ v
    sct = _scts()
    w   = zeros(ComplexF64, length(v))
    invMul!(w, sct, v, one(ComplexF64), zero(ComplexF64))
    @test sct * w ≈ v

    # Neither inverse path leaves its operator adjoint
    sct2 = _scts()
    invMulAdj!(zeros(ComplexF64, length(v)), sct2, v, one(ComplexF64), zero(ComplexF64))
    @test !isadjoint(sct2)

    gla = _glas()
    invMul!(zeros(ComplexF64, length(v)), gla, v, one(ComplexF64), zero(ComplexF64))
    @test !isadjoint(gla)
end
