# As sig.jl, but 8 cells across the body instead of 4, and swept further out,
# to separate the gap-dependent error from the body-resolution floor.
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

# slab body 8x8x2 coarse cells, so 8 cells across the two long axes
function sigs(cel, scl, xOrg, kTop)
    org = (0//1, 0//1, 0//1)
    volT = GlaVol(cel, (scl, scl, scl), org)
    volS = GlaVol(cel, (scl, scl, scl), (xOrg, 0//1, 0//1))
    mem = GlaVacOprMem(CPUKerOpt{Float64}(), volT, volS)
    svdvals(dns(mem))[1:kTop]
end

function main()
    kTop = 20
    @printf("%4s  %10s  %12s  %12s  %12s\n",
            "gap", "d/lambda", "sig1 rel", "top5 rel", "top20 rel")
    for g in 1:12
        xOrg = (8 + g) // 16
        sC = sigs((8, 8, 2), 1//16, xOrg, kTop)
        sF = sigs((16, 16, 4), 1//32, xOrg, kTop)
        rel = abs.(sC .- sF) ./ sF
        @printf("%4d  %10.4f  %12.3e  %12.3e  %12.3e\n",
                g, g / 16, rel[1], maximum(rel[1:5]), maximum(rel))
        flush(stdout)
    end
end

main()
