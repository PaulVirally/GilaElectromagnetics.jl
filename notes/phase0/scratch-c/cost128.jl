# The number Phase 0-B did not measure: the contact block at 128 bits (decision 8).
# 288 face pairs = 8 contact offsets x 36 face pairs, at the mMax each scale needs.
using Printf
include(joinpath(@__DIR__, "..", "..", "moments", "moments.jl"))

const OFFS8 = [(d1,d2,d3) for d1 in 0:1, d2 in 0:1, d3 in 0:1][:]
# (name, edge lengths as Rational, mMax that scale actually needs per Phase 0-B item 2)
const CASES = [("c32", (1//32,1//32,1//32), 22), ("sl", (1//32,1//32,1//512), 22),
               ("c8", (1//8,1//8,1//8), 42), ("c4", (1//4,1//4,1//4), 63)]

blk(s, mMax) = (a = zero(real(eltype(s))); for D in OFFS8, F in 1:6, Fp in 1:6
                    a += abs(faceMoments(D, F, Fp, s, mMax)[1]); end; a)

function blkThr(s, mMax)
    prs = [(D,F,Fp) for D in OFFS8 for F in 1:6 for Fp in 1:6]
    acc = zeros(Float64, Threads.maxthreadid())
    Threads.@threads for i in eachindex(prs)
        D, F, Fp = prs[i]
        acc[Threads.threadid()] += Float64(abs(faceMoments(D, F, Fp, s, mMax)[1]))
    end
    sum(acc)
end

@printf("nthreads = %d, BigFloat precision = 128\n\n", Threads.nthreads())
@printf("%-5s %5s  %11s %11s   %11s %11s   %6s\n",
        "shape","mMax","f64 ser s","f64 thr s","b128 ser s","b128 thr s","ratio")
setprecision(BigFloat, 128)
for (nm, sQ, mMax) in CASES
    s64 = ntuple(d -> Float64(sQ[d]), 3)
    sBF = ntuple(d -> BigFloat(sQ[d]), 3)
    blk(s64, 4); blkThr(s64, 4); blk(sBF, 4); blkThr(sBF, 4)      # compile
    t1 = @elapsed blk(s64, mMax);    t2 = @elapsed blkThr(s64, mMax)
    t3 = @elapsed blk(sBF, mMax);    t4 = @elapsed blkThr(sBF, mMax)
    @printf("%-5s %5d  %11.4f %11.4f   %11.4f %11.4f   %6.1f\n",
            nm, mMax, t1, t2, t3, t4, t3/t1)
    flush(stdout)
end
