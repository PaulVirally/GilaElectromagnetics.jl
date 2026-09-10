using GilaElectromagnetics, Printf
const GV = GilaElectromagnetics.GilaVacuum
const QI = Rational{BigInt}
s = ntuple(d -> QI(1, 16), 3); f = ComplexF64(1)
volA = GlaVol((2,2,2), (1//16,1//16,1//16), (0//1,0//1,0//1))
volB = GlaVol((2,2,2), (1//16,1//16,1//16), (9//64, 1//32, 0//1))
sgT = GV.sepGrd(volA, volB, 0); sgS = GV.sepGrd(volA, volB, 1)
Rs = NTuple{3,QI}[]
for p in CartesianIndices((4,4,4))
    any(d -> p[d] == 3, 1:3) && continue
    sep = ntuple(d -> QI(GV.grdSep(p[d], 2, d, sgT, sgS)), 3)
    all(d -> abs(sep[d]) <= QI(1,16), 1:3) && (println("touching ", Float64.(sep)); continue)
    push!(Rs, sep)
end
println(length(Rs), " separated offsets")
fx = GV.farSetX(s, s, f; offs = Rs, nBlk = GV.finBlk(Rs, s, (1,1,1), (1,1,1)))
rt = [GV.farRteX(fx, R)[1] for R in Rs]
println("farBlkX! routes 1/2/3 = ", (count(==(1), rt), count(==(2), rt), count(==(3), rt)))
q = findfirst(==(3), rt)
if q !== nothing
    println("first route-3 offset ", Float64.(Rs[q]), " |R| = ", sqrt(sum(Float64(Rs[q][d])^2 for d in 1:3)))
    t = @elapsed G = GV.farTnsX(Rs[q], s, s, f; fs = fx)
    @printf("route 3 cost %.2f s, max|G| = %.3e, finite %s\n", t, maximum(abs, G), all(isfinite, G))
end
