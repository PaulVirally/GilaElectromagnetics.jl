using Test, Random, GilaElectromagnetics, LinearAlgebra, CUDA, Printf
Random.seed!(0x67696c61)
cd("/Users/pvirally/.julia/dev/GilaElectromagnetics")
include("/Users/pvirally/.julia/dev/GilaElectromagnetics/test/tstHlp.jl")
include("/Users/pvirally/.julia/dev/GilaElectromagnetics/test/cmpOprTest.jl")
srcCvl = GlaCmpVol(GlaVol((2, 2, 2), cmpScl16, cmpOrg0))
srcRef = GlaVol((4, 4, 4), cmpScl32, cmpOrg0)
for (lbl, org) in (("near (3//16)", (3//16, 0//1, 0//1)), ("far (3//8)", (3//8, 0//1, 0//1)))
    cvl = GlaCmpVol(GlaVol((4, 4, 4), cmpScl32, org))
    mat = dnsMat(GlaCmpOprVac{Float64}(cvl, srcCvl))
    ref = dnsMat(GlaOprVac{Float64}(regions(cvl)[1], srcRef))
    @printf("%-14s %.3e\n", lbl, cmpRelFro(mat, cmpAgrRef(cvl, regions(cvl)[1], srcCvl, srcRef, ref)))
end
@printf("%-14s %.3e\n", "self composite", cmpRelFro(mnyMat, mnyAgr))
