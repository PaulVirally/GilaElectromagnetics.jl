# Same sweep one octave finer (lambda/32 vs lambda/64) to test whether the
# gap-independent floor scales as (s/lambda)^2.
using GilaElectromagnetics, LinearAlgebra, Printf

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

function sigs(cel, scl, xOrg, kTop)
    volT = GlaVol(cel, (scl, scl, scl), (0//1, 0//1, 0//1))
    volS = GlaVol(cel, (scl, scl, scl), (xOrg, 0//1, 0//1))
    svdvals(dns(GlaVacOprMem(CPUKerOpt{Float64}(), volT, volS)))[1:kTop]
end

function main()
    kTop = 20
    @printf("%4s  %10s  %12s  %12s  %12s\n",
            "gap", "d/lambda", "sig1 rel", "top5 rel", "top20 rel")
    for g in 1:12
        xOrg = (4 + g) // 32
        sC = sigs((4, 4, 4), 1//32, xOrg, kTop)
        sF = sigs((8, 8, 8), 1//64, xOrg, kTop)
        rel = abs.(sC .- sF) ./ sF
        @printf("%4d  %10.5f  %12.3e  %12.3e  %12.3e\n",
                g, g / 32, rel[1], maximum(rel[1:5]), maximum(rel))
        flush(stdout)
    end
end

main()
