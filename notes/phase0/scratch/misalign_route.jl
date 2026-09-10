include("/Users/pvirally/.julia/dev/GilaElectromagnetics/notes/farfield/farfield.jl")

# Production equal-cell shape, lambda/32 cube, f = 1.
sQ = (QI(1,32), QI(1,32), QI(1,32))
f  = ComplexF64(1.0)
fs = farSetup(sQ, f; disk=false)
s6 = ntuple(d -> Float64(sQ[d]), 3)

function routeOf(fs, R::NTuple{3,Float64})
    rr = sqrt(sum(R[i]^2 for i in 1:3))
    L = boundL(fs, rr)
    L >= 0 && return (1, L)
    Lc = boundLoct(fs, R)
    any(<(0), Lc) && return (3, Lc)
    return (2, maximum(Lc))
end

println("shape s = ", s6, "  f = ", f)
println()
println("Aligned integer-lattice offsets, D from (0,0,0)-adjacent out to (3,3,3):")
for D in [(0,0,1),(0,0,2),(1,0,0),(1,1,0),(1,1,1),(2,0,0),(2,1,0),(2,1,1),(2,2,0),
          (2,2,1),(2,2,2),(3,0,0),(3,1,1)]
    R = ntuple(d -> Float64(D[d]) * s6[d], 3)
    println("  D=", D, "  R=", R, "  route=", routeOf(fs, R))
end

println()
println("Same offsets with a generic non-integer-cell fractional perturbation")
println("(delta = 0.37 of a cell added to each nonzero-D axis, simulating a")
println("misaligned Δorg -- physical R shifts by ~0.37 cell, well within what")
println("an arbitrary Rational Δorg could produce):")
delta = 0.37
for D in [(0,0,1),(0,0,2),(1,0,0),(1,1,0),(1,1,1),(2,0,0),(2,1,0),(2,1,1),(2,2,0),
          (2,2,1),(2,2,2),(3,0,0),(3,1,1)]
    R = ntuple(d -> (Float64(D[d]) + (D[d] != 0 ? delta : 0.0)) * s6[d], 3)
    println("  D=", D, "  R=", R, "  route=", routeOf(fs, R))
end

println()
println("Slender needle shape for comparison (route-3 territory, per registry.tex):")
sQn = (QI(1,32), QI(1,32), QI(1,2))
fsn = farSetup(sQn, f; disk=false)
s6n = ntuple(d -> Float64(sQn[d]), 3)
for n in [8, 16, 20, 22, 23, 24, 32]
    D = (0,0,n)
    R = ntuple(d -> Float64(D[d]) * s6n[d], 3)
    Rm = ntuple(d -> (Float64(D[d]) + (D[d] != 0 ? delta : 0.0)) * s6n[d], 3)
    println("  n=", n, "  aligned route=", routeOf(fsn, R), "   +0.37-cell route=", routeOf(fsn, Rm))
end
