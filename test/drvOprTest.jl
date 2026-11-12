# Operators of the field and its derivatives at the cell centres (ctr, der) against the direct sum of the kernels
import GilaElectromagnetics: drvNer!, drvGam, QI

# unequal edges, so a swapped axis or a wrong reflection sign shows
const drvScl = (1//32, 1//24, 1//40)
const drvSrc = GlaVol((4, 3, 5), drvScl, stdOrg)
# touching the source along z, a quarter cell off its lattice along x
const drvTrg = GlaVol((4, 3, 5), drvScl, (1//128, 0//1, 1//8))

# E_i = sum_j K(c_i - c_j) p_j, each separation from the near route
function drvSum(trg::GlaVol, src::GlaVol, der::Int, p::Array{ComplexF64,4})
    nGam = length(drvGam(der))
    ctrTrg, ctrSrc = vec(collect(Iterators.product(trg.grd...))), vec(collect(Iterators.product(src.grd...)))
    Ds = unique([QI.(t .- s) for t in ctrTrg, s in ctrSrc])
    K = drvNer!(Array{ComplexF64}(undef, 3, 3, nGam, length(Ds)), Ds, QI.(src.scl), 1.0, der)
    pos = Dict(D => i for (i, D) in enumerate(Ds))
    pj = reshape(p, :, 3)
    E = zeros(ComplexF64, length(ctrTrg), 3, nGam)
    for (i, t) in enumerate(ctrTrg), (j, s) in enumerate(ctrSrc), g in 1:nGam
        E[i, :, g] .+= view(K, :, :, g, pos[QI.(t .- s)]) * pj[j, :]
    end
    return reshape(E, trg.cel..., 3, nGam)
end

@testset "Centre values and derivatives" begin
    p = randn(ComplexF64, drvSrc.cel..., 3)
    for (trg, der) in ((drvSrc, 0), (drvSrc, 1), (drvSrc, 2), (drvTrg, 0), (drvTrg, 2))
        opr = GlaOprVac{Float64}(trg, drvSrc; der, ctr = true, prxWrn = false)
        E = drvSum(trg, drvSrc, der, p)
        @test glaSze(opr, 1) == size(E)[1:(der == 0 ? 4 : 5)]
        @test norm(vec(opr * p) - vec(E)) <= 1e-12 * norm(E)
    end
end

@testset "Centre operator throws" begin
    @test_throws ArgumentError GlaOprVac{Float64}(_vol2s; der = 1, ctr = false)
    @test_throws ArgumentError GlaOprVac{Float64}(_vol2s; der = -1)
    @test_throws ArgumentError GlaOprVac{Float64}(_vol2s; ctr = true, qssApx = true)
    @test_throws ArgumentError GlaOprVac{Float64}(_vol2s; ctr = true, blc = (ax = (1,), P = (4,), θ = (0.0,)))
    @test_throws ArgumentError GlaOprVac{Float64}(GlaVol((4, 4, 4), scl16, (1//2, 0//1, 0//1)), _vol4; ctr = true, prxWrn = false)
    drv = GlaOprVac{Float64}(_vol2s; der = 1)
    v = randn(ComplexF64, size(drv, 2))
    for f in (adjoint!, asym, glaSym, o -> InvSctOpr(o, _sus2s), o -> SctOpr(o, _sus2s), o -> GlaOpr(o, _sus2s), o -> o \ v,
        o -> solve(o, v, GMRESSolver()), o -> solve(o, v, BiCGStabSolver()), o -> solve(o, v, GCRODRSolver()))
        @test_throws ArgumentError f(drv)
    end
end

@testset "Default is the cell average" begin
    @test GlaOprVac{Float64}(_vol2s; der = 0, ctr = false).mem.egoFur == _mem2s.egoFur
end
