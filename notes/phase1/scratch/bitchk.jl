# Re-proof of Phase 1a's bitwise identity after the parFac/boxFace/momSer edits.
module Ref; include("/Users/pvirally/.julia/dev/GilaElectromagnetics/notes/moments/moments.jl"); end
using GilaElectromagnetics, DoubleFloats
const Prt = GilaElectromagnetics.GilaVacuum

bitEq(a::BigFloat, b::BigFloat) = isequal(a, b) && precision(a) == precision(b)
bitEq(a, b) = a === b
eqv(a, b) = length(a) == length(b) && all(bitEq.(a, b))

function battery(::Type{T}, shapes, mMaxs; label) where {T}
    n = 0; bad = 0
    for s in shapes, Dx in 0:1, Dy in 0:1, Dz in 0:1, F in 1:6, Fp in 1:6, mMax in mMaxs
        D = (Dx, Dy, Dz)
        a = Ref.faceMoments(D, F, Fp, s, mMax)
        b = Prt.facMom(D, F, Fp, s, mMax)
        n += 1
        eqv(a, b) || (bad += 1; bad <= 5 && println("  DIFF $label s=$s D=$D F=$F Fp=$Fp mMax=$mMax"))
    end
    println("$label: $n comparisons, $bad differ")
    return n, bad
end

function serChk(::Type{T}, shapes, frqs) where {T}
    n = 0; bad = 0
    for s in shapes, Dx in 0:1, Dy in 0:1, Dz in 0:1, F in 1:6, Fp in 1:6, f in frqs
        pA, pB = Ref.facePair((Dx, Dy, Dz), F, Fp, s)
        a = Ref.momentSeries(pA, pB, f, 12)
        b = Prt.momSer(pA, pB, f, 12)
        n += 1
        (bitEq(a[1], b[1]) && bitEq(a[2], b[2])) || (bad += 1)
    end
    println("momSer: $n comparisons, $bad differ")
    return n, bad
end

tot = 0; bd = 0
for T in (Float64, Double64)
    # the reference throws Rational{BigInt}(::Double64) on non-dyadic scales, the
    # trap cnvQ fixes in src; those shapes are Float64-only here
    shp = [(T(1)/32, T(1)/32, T(1)/32), (T(1)/8, T(1)/8, T(1)/8),
           (T(1)/32, T(1)/32, T(1)/512), (T(1)/4, T(1)/4, T(1)/4)]
    T === Float64 && append!(shp, [(T(1)/10, T(1)/10, T(1)/10), (T(1)/7, T(1)/3, T(1)/5)])
    n, b = battery(T, shp, [-1, 0, 4, 12, 22]; label = string(T))
    global tot += n; global bd += b
end
n, b = serChk(Float64, [(1/32, 1/32, 1/32), (1/32, 1/32, 1/512)],
              [1.0 + 0.0im, 1.0 + 0.1im, 0.37 + 0.0im])
tot += n; bd += b
println("total $tot comparisons, $bd differ")
