module Ref; include("/Users/pvirally/.julia/dev/GilaElectromagnetics/notes/moments/moments.jl"); end
module Prt; include("/Users/pvirally/.julia/dev/GilaElectromagnetics/src/vacuum/glaVacOprMemMom.jl"); end
using DoubleFloats
bitEq(a::BigFloat, b::BigFloat) = isequal(a, b) && precision(a) == precision(b)
bitEq(a, b) = a === b
function run()
    n = 0; bad = 0
    Ds = ((2,0,0),(3,1,0),(2,2,2),(5,0,1),(1,4,2),(7,3,3))
    for s0 in ((1//32,1//32,1//32), (1//8,1//8,1//8), (1//32,1//32,1//512),
               (1//7,3//11,5//13))
        for T in (s0[1] == 1//7 ? (Float64,) : (Float64, Double64))
            s = (T(s0[1]), T(s0[2]), T(s0[3]))
            for D in Ds, F in 1:6, Fp in 1:6, mMax in (12, 22)
                a = Ref.faceMoments(D, F, Fp, s, mMax)
                b = Prt.facMom(D, F, Fp, s, mMax)
                n += 1
                all(bitEq.(a, b)) || (bad += 1; bad <= 3 && println("  DIFF $T $s0 $D $F $Fp $mMax"))
            end
        end
    end
    println("separated pairs: $n comparisons, $bad differ")
    g = 0; gb = 0
    for T in (Float64, Double64, BigFloat), k in (3, 7, 11, 14, 28, 40)
        x1, w1 = Ref.gauLeg(k, T); x2, w2 = Prt.gauLeg(k, T)
        g += 1
        all(bitEq.(x1, x2)) && all(bitEq.(w1, w2)) || (gb += 1)
    end
    println("gauLeg: $g comparisons, $gb differ")
end
run()
