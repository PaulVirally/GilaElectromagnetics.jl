module Ref; include("/Users/pvirally/.julia/dev/GilaElectromagnetics/notes/moments/moments.jl"); end
module Prt; include("/Users/pvirally/.julia/dev/GilaElectromagnetics/src/vacuum/glaVacOprMemMom.jl"); end
using DoubleFloats

fmRef = Ref.faceMoments
msRef = Ref.momentSeries
fpRef = Ref.facePair
fmPrt = isdefined(Prt, :faceMoments) ? Prt.faceMoments : Prt.facMom
msPrt = isdefined(Prt, :momentSeries) ? Prt.momentSeries : Prt.momSer
fpPrt = isdefined(Prt, :facePair) ? Prt.facePair : Prt.parFac

bitEq(a::BigFloat, b::BigFloat) = isequal(a, b) && precision(a) == precision(b)
bitEq(a, b) = a === b
eqv(a, b) = length(a) == length(b) && all(bitEq.(a, b))

function battery(::Type{T}, shapes, mMaxs; label) where {T}
    n = 0; bad = 0
    for s in shapes, Dx in 0:1, Dy in 0:1, Dz in 0:1, F in 1:6, Fp in 1:6, mMax in mMaxs
        D = (Dx, Dy, Dz)
        a = fmRef(D, F, Fp, s, mMax)
        b = fmPrt(D, F, Fp, s, mMax)
        n += 1
        if !eqv(a, b)
            bad += 1
            bad <= 5 && println("  DIFF $label s=$s D=$D F=$F Fp=$Fp mMax=$mMax")
        end
    end
    println("$label: $n comparisons, $bad differ")
    return n, bad
end

shp(::Type{T}) where {T} = ((T(1//32), T(1//32), T(1//32)),
                            (T(1//8), T(1//8), T(1//8)),
                            (T(1//4), T(1//4), T(1//4)),
                            (T(1//32), T(1//32), T(1//512)))

tot = 0; totBad = 0
n, b = battery(Float64, shp(Float64), (12, 22, 30); label = "faceMoments Float64")
tot += n; totBad += b

# momentSeries battery, Float64
let n = 0, bad = 0
    for s in shp(Float64), Dx in 0:1, Dy in 0:1, Dz in 0:1, F in 1:6, Fp in 1:6,
        f in (1.0, 1.0 + 0.1im, 0.37), mMax in (12, 22)
        D = (Dx, Dy, Dz)
        pA, pB = fpRef(D, F, Fp, s)
        qA, qB = fpPrt(D, F, Fp, s)
        (qA === pA && qB === pB) || (bad += 1)
        a = msRef(pA, pB, f, mMax)
        c = msPrt(qA, qB, f, mMax)
        n += 1
        (a[1] === c[1] && a[2] === c[2]) || (bad += 1;
            bad <= 5 && println("  DIFF momSer s=$s D=$D F=$F Fp=$Fp f=$f mMax=$mMax"))
    end
    println("momentSeries Float64: $n comparisons, $bad differ")
    global tot += n; global totBad += bad
end

# Double64 sample: all 8 offsets x 36 face pairs, lambda/32 and slender, mMax = 22
n, b = battery(Double64, (shp(Double64)[1], shp(Double64)[4]), (22,); label = "faceMoments Double64")
tot += n; totBad += b

# BigFloat sample at 128 bits
setprecision(BigFloat, 128) do
    n, b = battery(BigFloat, (shp(BigFloat)[1],), (22,); label = "faceMoments BigFloat128")
    global tot += n; global totBad += b
end

println("TOTAL: $tot comparisons, $totBad differ")
