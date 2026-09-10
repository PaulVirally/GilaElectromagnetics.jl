include(joinpath(@__DIR__, "..", "..", "moments", "moments.jl"))
using Printf
setprecision(BigFloat, 128)
const OFFS8 = [(d1,d2,d3) for d1 in 0:1, d2 in 0:1, d3 in 0:1][:]

function run()
    s64 = (1/32, 1/32, 1/32)
    sBF = ntuple(d -> BigFloat(1)/BigFloat(32), 3)
    faceMoments((1,1,1), 1, 4, s64, 4); faceMoments((1,1,1), 1, 4, sBF, 4)
    @printf("%-10s %12s %12s %8s\n", "offset", "f64 tot s", "b128 tot s", "ratio")
    gt64 = 0.0; gtBF = 0.0; wt = 0.0; wd = ""
    for D in OFFS8
        t64 = 0.0; tBF = 0.0
        for F in 1:6, Fp in 1:6
            t64 += @elapsed faceMoments(D, F, Fp, s64, 22)
            e = @elapsed faceMoments(D, F, Fp, sBF, 22)
            tBF += e
            if e > wt; wt = e; wd = "D=$D F=$F Fp=$Fp"; end
        end
        gt64 += t64; gtBF += tBF
        @printf("%-10s %12.4f %12.4f %8.1f\n", string(D), t64, tBF, tBF/t64); flush(stdout)
    end
    @printf("\nTOTAL 288 pairs: f64 %.3f s   b128 %.3f s   ratio %.1f\n", gt64, gtBF, gtBF/gt64)
    @printf("worst single pair: %.4f s at %s\n", wt, wd)
end
run()
