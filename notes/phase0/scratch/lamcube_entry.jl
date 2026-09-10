include("/Users/pvirally/.julia/dev/GilaElectromagnetics/notes/farfield/farfield.jl")

sT = (QI(1), QI(1), QI(1))
sS = (QI(1,32), QI(1,32), QI(1,32))
f  = ComplexF64(1.0)
R  = (QI(35,64), QI(31,64), QI(31,64))   # = (17.5, 15.5, 15.5) * g, g = 1/32

G, cert = farTensorX(R, sT, sS, f; cert=true)
println("R = ", R, "  (in g=1/32 units: ", R ./ (QI(1,32)), ")")
println("G = ")
for i in 1:3
    println("  ", G[i,:])
end
mx = maximum(abs, G)
println("max|G| = ", mx)
println("cert = ", cert, "  cert/max|G| = ", cert/mx)
println()
println("entry magnitudes relative to max|G|:")
for i in 1:3, j in 1:3
    println("  G[$i,$j] = ", G[i,j], "   |G|/max = ", abs(G[i,j])/mx)
end

gQ = ntuple(d -> min(sT[d], sS[d]), 3)
nT = ntuple(d -> Int(sT[d] / gQ[d]), 3); nS = ntuple(d -> Int(sS[d] / gQ[d]), 3)
fx = farSetupX(sT, sS, f; nBlk = nBlkX((R,), gQ, nT, nS), offs = (R,))
kind, L = farRouteX(fx, R)
println("route kind = ", kind, "  L = ", L, "  (1=whole trapezoid box, 2=gcd box sum, 3=k-series)")
