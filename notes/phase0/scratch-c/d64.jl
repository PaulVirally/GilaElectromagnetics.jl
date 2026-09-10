using DoubleFloats, Printf
include(joinpath(@__DIR__, "..", "..", "moments", "moments.jl"))
setprecision(BigFloat, 128)
const PRS = [(1,4),(2,5),(3,6),(1,1),(2,3),(6,1)]

function run(io)
    # (A) does it even run, and is it accurate?
    println(io, "=== (A) Double64 correctness vs 256-bit BigFloat, faceMoments + momentSeries ===")
    for (nm, sQ, mMax) in (("c32",(1//32,1//32,1//32),22), ("c4",(1//4,1//4,1//4),63))
        s64 = ntuple(d -> Float64(sQ[d]), 3)
        sD  = ntuple(d -> Double64(numerator(sQ[d]))/Double64(denominator(sQ[d])), 3)
        ok = true; wrst = 0.0
        try
            for (F,Fp) in PRS
                vD = faceMoments((1,1,1),F,Fp,sD,mMax)
                vR = setprecision(BigFloat, 256) do
                    sB = ntuple(d -> BigFloat(numerator(sQ[d]))/BigFloat(denominator(sQ[d])), 3)
                    faceMoments((1,1,1),F,Fp,sB,mMax)
                end
                for i in eachindex(vD)
                    iszero(vR[i]) && continue
                    e = Float64(abs(BigFloat(vD[i]) - vR[i]) / abs(vR[i]))
                    e > wrst && (wrst = e)
                end
            end
        catch err
            ok = false
            @printf(io, "%-4s FAILED: %s\n", nm, sprint(showerror, err)[1:min(end,220)])
        end
        ok && @printf(io, "%-4s mMax=%2d  worst rel err vs 256-bit = %.3e  (eps(Double64)=%.2e)\n",
                      nm, mMax, wrst, Float64(eps(Double64)))
        flush(io)
    end
    # (B) cost
    println(io, "\n=== (B) cost, 6 face pairs at offset (1,1,1), serial ===")
    @printf(io, "%-4s %5s %11s %11s %11s %9s %9s\n","shp","mMax","f64 s","D64 s","b128 s","D64/f64","b128/D64")
    for (nm, sQ, mMax) in (("c32",(1//32,1//32,1//32),22), ("c8",(1//8,1//8,1//8),42),
                           ("c4",(1//4,1//4,1//4),63))
        s64 = ntuple(d -> Float64(sQ[d]), 3)
        sD  = ntuple(d -> Double64(numerator(sQ[d]))/Double64(denominator(sQ[d])), 3)
        sB  = ntuple(d -> BigFloat(numerator(sQ[d]))/BigFloat(denominator(sQ[d])), 3)
        faceMoments((1,1,1),1,4,s64,4); faceMoments((1,1,1),1,4,sD,4); faceMoments((1,1,1),1,4,sB,4)
        t64=0.0; tD=0.0; tB=0.0
        for (F,Fp) in PRS
            t64 += @elapsed faceMoments((1,1,1),F,Fp,s64,mMax)
            tD  += @elapsed faceMoments((1,1,1),F,Fp,sD, mMax)
            tB  += @elapsed faceMoments((1,1,1),F,Fp,sB, mMax)
        end
        @printf(io, "%-4s %5d %11.4f %11.4f %11.4f %9.1f %9.1f\n", nm, mMax, t64, tD, tB, tD/t64, tB/tD)
        flush(io)
    end
end
open(io -> run(io), joinpath(@__DIR__, "d64.txt"), "w")
