# Cross-scale adversarial sweep: cell-ratio combinations, orientations and frequencies that appear
# in no reference file — in particular a target COARSER than the source, which farx.txt never has.
using GilaElectromagnetics
const GV = GilaElectromagnetics.GilaVacuum
include(joinpath(@__DIR__, "refgen.jl"))
pr(x...) = (println(x...); flush(stdout))
mx(A) = Float64(maximum(abs, A))
const ORD = parse(Int, get(ENV, "RGORD", "24"))
const Q = Rational{BigInt}

function prb(sT, sS, f, R; ksr = false)
    sTQ = ntuple(d -> Q(sT[d]), 3); sSQ = ntuple(d -> Q(sS[d]), 3)
    RQ = ntuple(d -> Q(R[d]), 3)
    fx = GV.farSetX(sTQ, sSQ, ComplexF64(f);
                    nBlk = GV.finBlk((RQ,), ntuple(d -> min(sTQ[d], sSQ[d]), 3),
                                     ntuple(d -> Int(sTQ[d] / min(sTQ[d], sSQ[d])), 3),
                                     ntuple(d -> Int(sSQ[d] / min(sTQ[d], sSQ[d])), 3)),
                    offs = (RQ,))
    kind, L = GV.farRteX(fx, RQ)
    G, c = GV.farTnsX(RQ, sTQ, sSQ, ComplexF64(f); fs = fx, cert = true)
    Gr = rgTns(RQ, sTQ, sSQ, ComplexF64(f); ord = ORD)
    m = mx(Gr)
    e = mx(G .- Gr) / m
    alt = String[]
    mI = GV.latInd(fx, RQ)
    if kind != 1
        Lw = GV.whlLvl(fx, sqrt(sum(Float64(RQ[d])^2 for d in 1:3)))
        Lw < 0 && Lw != -1 && nothing
        if Lw < 0
            G1 = zeros(ComplexF64, 3, 3)
            GV.tnsWhlX!(G1, fx, GV.FarWrk(fx.Lw, Float64), ntuple(d -> Float64(RQ[d]), 3), fx.Lw)
            push!(alt, string("r1@Lw=", fx.Lw, " ", mx(G1 .- Gr) / m))
        end
    end
    if kind != 2 && mI !== nothing && GV.subSep(fx, mI) >= 2
        fg = GV.finSet(fx)
        G2 = zeros(ComplexF64, 3, 3)
        GV.tnsGcd!(G2, fx, fg, GV.FarWrk(max(fg.Lw, fg.Lo), Float64), mI,
                   zeros(ComplexF64, 3, 3), false)
        push!(alt, string("r2 ", mx(G2 .- Gr) / m))
    end
    if ksr && kind != 3
        G3 = copy(GV.ksrCchX!(fx, RQ))
        push!(alt, string("r3 ", mx(G3 .- Gr) / m))
    end
    pr("  sT=", Float64.(sT), " sS=", Float64.(sS), " f=", f, " R=", Float64.(R),
       " kind=", kind, " maxT=", m, "  err=", e, "  cert=", c / m,
       isempty(alt) ? "" : "  | " * join(alt, "  "))
end

const CASES = [
    ((1//32,1//32,1//32), (5//32,1//32,1//32), 2.0 + 2.0im, ((9//32,0//1,0//1),)),
    ((1//48,1//48,1//48), (1//8,1//16,1//48), 1.0 + 0.1im, ((1//4,0//1,0//1),)),
    ((1//64,1//32,1//16), (3//64,1//8,1//16), 1.0 + 0.0im, ((1//4,1//4,0//1),)),
    ((3//32,3//32,3//32), (1//32,1//32,1//32), 1.0 + 0.0im, ((7//32,0//1,0//1),)),
    ((1//4,1//32,1//32), (1//32,1//32,1//32), 1.0 + 0.1im, ((9//32,0//1,0//1), (0//1,5//32,0//1))),
    ((1//8,1//8,1//8), (1//32,1//32,1//32), 0.5 + 2.0im, ((5//16,0//1,0//1),)),
]

pr("threads = ", Threads.nthreads())
for (sT, sS, f, Rs) in CASES
    pr("\n=== sT=", Float64.(sT), " sS=", Float64.(sS), " f=", f, " ===")
    for R in Rs
        try
            prb(sT, sS, f, R)
        catch e
            pr("  R=", Float64.(R), " FAILED: ", sprint(showerror, e)[1:min(end, 400)])
        end
    end
end
pr("ALLDONE")
