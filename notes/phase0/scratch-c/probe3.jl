include(joinpath(@__DIR__, "..", "..", "moments", "moments.jl"))
using Printf
setprecision(BigFloat, 128)
# one contact offset = 36 face pairs; the full block is 8 of these.
function run()
    for (nm, sQ, mMax) in (("c32", (1//32,1//32,1//32), 22),
                           ("c8",  (1//8,1//8,1//8),   42),
                           ("c4",  (1//4,1//4,1//4),   63))
        s64 = ntuple(d -> Float64(sQ[d]), 3)
        sBF = ntuple(d -> BigFloat(sQ[d].num)/BigFloat(sQ[d].den), 3)
        faceMoments((1,1,1), 1, 4, s64, 4); faceMoments((1,1,1), 1, 4, sBF, 4)
        t64 = 0.0; tBF = 0.0
        for F in 1:6, Fp in 1:6            # worst offset (1,1,1), 36 pairs
            t64 += @elapsed faceMoments((1,1,1), F, Fp, s64, mMax)
            tBF += @elapsed faceMoments((1,1,1), F, Fp, sBF, mMax)
        end
        @printf("%-4s mMax=%2d  36 pairs: f64 %7.3f s  b128 %8.3f s  ratio %5.1f | est 288-pair block: f64 %6.2f s  b128 %8.1f s\n",
                nm, mMax, t64, tBF, tBF/t64, t64*8, tBF*8)
        flush(stdout)
    end
end
run()
