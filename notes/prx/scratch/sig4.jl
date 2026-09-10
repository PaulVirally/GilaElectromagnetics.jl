# Body fixed at lambda/4 x lambda/4 x lambda/16; only the grid pair changes.
# Pair A sees a gap of j coarse cells where pair B sees 2j, so a cells-based
# crossover lands at half the j for B and a distance-based one at the same j.
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
    @printf("%3s %9s | %11s %11s | %11s %11s\n",
            "j", "d/lambda", "A sig1", "A top5", "B sig1", "B top5")
    for j in 1:8
        xOrg = 1//4 + j//16
        aC = sigs((4, 4, 1), 1//16, xOrg, kTop)
        aF = sigs((8, 8, 2), 1//32, xOrg, kTop)
        bC = aF
        bF = sigs((16, 16, 4), 1//64, xOrg, kTop)
        ra = abs.(aC .- aF) ./ aF
        rb = abs.(bC .- bF) ./ bF
        @printf("%3d %9.4f | %11.3e %11.3e | %11.3e %11.3e\n",
                j, j / 16, ra[1], maximum(ra[1:5]), rb[1], maximum(rb[1:5]))
        flush(stdout)
    end
end

main()
