using Test, Random, GilaElectromagnetics, LinearAlgebra, Printf
import GilaElectromagnetics.GilaVolumes: uniVol
Random.seed!(0x67696c61)
cd(joinpath(@__DIR__, "..", "..", ".."))
const stdOrg = (0//1, 0//1, 0//1)
function dnsMat(opr)
    n = size(opr, 2)
    M = zeros(ComplexF64, size(opr, 1), n)
    for i in 1:n
        v = zeros(ComplexF64, n); v[i] = 1
        M[:, i] .= opr * v
    end
    M
end
sclCrs = (1//16, 1//16, 1//16); sclFin = (1//32, 1//32, 1//32); sep = (1//2, 0//1, 0//1)
volCrs = GlaVol((2,2,2), sclCrs, stdOrg); volRef = GlaVol((4,4,4), sclFin, stdOrg)
volFin = GlaVol((4,4,4), sclFin, sep)
function agrMap(celCrs)
    celFin = celCrs .* 2
    iC = LinearIndices((celCrs..., 3)); iF = LinearIndices((celFin..., 3))
    A = zeros(prod(celCrs) * 3, prod(celFin) * 3)
    for d in 1:3, c in CartesianIndices(celCrs), o in CartesianIndices((0:1,0:1,0:1))
        A[iC[Tuple(c)..., d], iF[(2 .* Tuple(c) .- 1 .+ Tuple(o))..., d]] = 1.0
    end
    A
end
relFro(a, b) = norm(a - b) / norm(b)
agr = agrMap((2,2,2)); inj = collect(transpose(agr))
t = @elapsed begin
mCT = dnsMat(GlaOprVac{Float64}(volCrs, volFin)); mFT = dnsMat(GlaOprVac{Float64}(volFin, volCrs))
rCT = dnsMat(GlaOprVac{Float64}(volRef, volFin)); rFT = dnsMat(GlaOprVac{Float64}(volFin, volRef))
end
@printf("builds %.1f s\n", t)
@printf("l56 coarse target vs mean of fine rows      %.3e\n", relFro(mCT, (agr ./ 8) * rCT))
@printf("l69 fine target vs injected coarse column   %.3e\n", relFro(mFT, rFT * inj))
@printf("l78 volume-weighted reciprocity             %.3e\n", relFro(prod(sclCrs) .* mCT, prod(sclFin) .* transpose(mFT)))
@printf("l80 same-scale reference transpose          %.3e\n", relFro(rCT, transpose(rFT)))
volTch = GlaVol((4,4,4), sclFin, (4//32, 0//1, 0//1))
mTch = dnsMat(GlaOprVac{Float64}(volCrs, volTch))
@printf("touching cross-scale vs remeshed reference  %.3e  finite %s\n",
    relFro(mTch, (agr ./ 8) * dnsMat(GlaOprVac{Float64}(volRef, volTch))), all(isfinite, mTch))
volAni = GlaVol((4,4,4), (1//64, 1//32, 1//32), stdOrg)
volIso = GlaVol((4,4,4), sclFin, (5//32, 0//1, 0//1))
mAI = dnsMat(GlaOprVac{Float64}(volAni, volIso))
mIA = dnsMat(GlaOprVac{Float64}(volIso, volAni))
@printf("anisotropic reciprocity                     %.3e\n",
    relFro(prod(volAni.scl) .* mAI, transpose(prod(volIso.scl) .* mIA)))
