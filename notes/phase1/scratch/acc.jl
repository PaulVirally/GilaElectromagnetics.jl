module Prt; include("/Users/pvirally/.julia/dev/GilaElectromagnetics/src/vacuum/glaVacOprMemMom.jl"); end
using DoubleFloats
setprecision(BigFloat, 256)
function run()
worstD = 0.0; worstB = 0.0; nD = 0
for s0 in ((1//32,1//32,1//32), (1//32,1//32,1//512), (1//4,1//4,1//4)),
    D in ((0,0,0),(1,0,0),(1,1,0),(1,1,1)), F in 1:6, Fp in 1:6
    sD = (Double64(s0[1]), Double64(s0[2]), Double64(s0[3]))
    sB = (BigFloat(s0[1]), BigFloat(s0[2]), BigFloat(s0[3]))
    sF = (Float64(s0[1]), Float64(s0[2]), Float64(s0[3]))
    a = Prt.facMom(D, F, Fp, sD, 22)
    r = Prt.facMom(D, F, Fp, sB, 22)
    f = Prt.facMom(D, F, Fp, sF, 22)
    for k in eachindex(r)
        iszero(r[k]) && continue
        worstD = max(worstD, Float64(abs(BigFloat(a[k]) - r[k]) / abs(r[k])))
        worstB = max(worstB, Float64(abs(BigFloat(f[k]) - r[k]) / abs(r[k])))
    end
    nD += 1
end
println("face pairs: $nD;  worst rel Double64 vs 256-bit: $worstD;  worst rel Float64: $worstB")
end
run()
