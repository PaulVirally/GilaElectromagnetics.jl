# Top singular values of the external vacuum Green operator, coarse grid vs a
# 2x refined grid at matched physical geometry, swept over the gap.
using GilaElectromagnetics, LinearAlgebra, Printf
const GV = GilaElectromagnetics

function dns(mem)
    T = eltype(first(mem.egoFur))
    nn = prod(mem.srcVol.cel) * 3
    mm = prod(mem.trgVol.cel) * 3
    A = zeros(T, mm, nn)
    for i in 1:nn
        v = zeros(T, mem.srcVol.cel..., 3)
        v[i] = one(T)
        A[:, i] .= vec(egoOpr!(mem, v))
    end
    A
end

# body side 1//4 wavelength either way; gap is gap//16 of a wavelength
function sigs(nCel, scl, gapNum, kTop)
    org = (0//1, 0//1, 0//1)
    volT = GlaVol((nCel, nCel, nCel), (scl, scl, scl), org)
    volS = GlaVol((nCel, nCel, nCel), (scl, scl, scl), (1//4 + gapNum, 0//1, 0//1))
    mem = GlaVacOprMem(CPUKerOpt{Float64}(), volT, volS)
    svdvals(dns(mem))[1:kTop]
end

function main()
    kTop = 20
    @printf("%4s  %10s  %12s  %12s  %12s\n",
            "gap", "d/lambda", "sig1 rel", "top5 rel", "top20 rel")
    for g in 1:10
        gapNum = g // 16
        sC = sigs(4, 1//16, gapNum, kTop)
        sF = sigs(8, 1//32, gapNum, kTop)
        rel = abs.(sC .- sF) ./ sF
        @printf("%4d  %10.4f  %12.3e  %12.3e  %12.3e\n",
                g, float(gapNum), rel[1], maximum(rel[1:5]), maximum(rel))
        flush(stdout)
    end
end

main()
