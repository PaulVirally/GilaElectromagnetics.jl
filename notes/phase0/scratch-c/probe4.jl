include(joinpath(@__DIR__, "..", "..", "moments", "moments.jl"))
using Printf
setprecision(BigFloat, 128)
# cost scaling in mMax: a fixed 6-pair sample, both precisions, three shapes.
const PRS = [(1,4),(2,5),(3,6),(1,1),(2,3),(6,1)]
function run(io)
    @printf(io, "%-4s %5s %11s %11s %7s %14s\n",
            "shp","mMax","f64 6pr s","b128 6pr s","ratio","b128 288pr s")
    for (nm, sQ, mMax) in (("c32",(1//32,1//32,1//32),22), ("c8",(1//8,1//8,1//8),42),
                           ("c4",(1//4,1//4,1//4),63))
        s64 = ntuple(d -> Float64(sQ[d]), 3)
        sBF = ntuple(d -> BigFloat(numerator(sQ[d]))/BigFloat(denominator(sQ[d])), 3)
        faceMoments((1,1,1),1,4,s64,4); faceMoments((1,1,1),1,4,sBF,4)
        t64 = 0.0; tBF = 0.0
        for (F,Fp) in PRS
            t64 += @elapsed faceMoments((1,1,1),F,Fp,s64,mMax)
            tBF += @elapsed faceMoments((1,1,1),F,Fp,sBF,mMax)
        end
        @printf(io, "%-4s %5d %11.4f %11.4f %7.1f %14.1f\n",
                nm, mMax, t64, tBF, tBF/t64, tBF*288/length(PRS))
        flush(io)
    end
end
open(io -> run(io), joinpath(@__DIR__, "scale.txt"), "w")
