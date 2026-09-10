include(joinpath(@__DIR__, "..", "..", "moments", "moments.jl"))
using Printf
const CNT = Ref(0); const ORD = Int[]
const gauLegOrig = gauLeg
gauLeg(n::Int, ::Type{T}) where {T} = (CNT[] += 1; push!(ORD, n); gauLegOrig(n, T))

s64 = (1/32, 1/32, 1/32)
setprecision(BigFloat, 128)
sBF = ntuple(d -> BigFloat(1)/BigFloat(32), 3)

for (nm, s, mMax) in (("Float64", s64, 22), ("BigFloat128", sBF, 22))
    faceMoments((1,1,1), 1, 4, s, 4)                      # compile
    CNT[] = 0; empty!(ORD)
    t = @elapsed faceMoments((1,1,1), 1, 4, s, mMax)
    @printf("%-12s one face pair: %8.4f s, gauLeg calls = %5d, orders %s\n",
            nm, t, CNT[], isempty(ORD) ? "-" : "$(minimum(ORD))..$(maximum(ORD))")
end

@printf("\ngauLeg cost alone:\n")
for n in (11, 28, 60, 120, 240)
    t64 = @elapsed gauLegOrig(n, Float64)
    tBF = @elapsed gauLegOrig(n, BigFloat)
    @printf("  n=%3d  Float64 %9.6f s   BigFloat128 %9.6f s   ratio %7.1f\n", n, t64, tBF, tBF/t64)
end
