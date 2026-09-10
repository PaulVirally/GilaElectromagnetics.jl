include("/Users/pvirally/.julia/dev/GilaElectromagnetics/notes/farfield/farfield.jl")

const GQ = (QI(1,32), QI(1,32), QI(1,32))
const f1 = ComplexF64(1.0)
g = GQ[1]
sS = (16g, 16g, 16g)
b  = ntuple(d -> (GQ[d] + sS[d]) // 2, 3)

println("building the realistic block offset list ...")
t0 = time()
Rs = [(QI(2k1 + 1)//64, QI(2k2 + 1)//64, QI(2k3 + 1)//64) for k3 in -56:55 for k2 in -56:55 for k1 in 8:119]
# xTouch replicate (touching test used by verify.jl); farRouteX itself raises on touching, so filter first
xTouch(R, b) = all(abs(R[d]) <= b[d] for d in 1:3)
filter!(R -> !xTouch(R, b), Rs)
println("offsets: ", length(Rs), "  (", time()-t0, " s)")

println("building FrqSetX (this sizes tables to the block) ...")
t0 = time()
fx = farSetupX(GQ, sS, f1; offs = Rs)
println("farSetupX: ", time()-t0, " s;  Lw=", fx.Lw, " (table L ceiling)")

# rho_whl for the trapezoid: r_d = |b| (half-diagonal of the trapezoid box), rho = r_d/|R|
b6 = ntuple(d -> Float64(b[d]), 3)
rd6 = sqrt(sum(b6[i]^2 for i in 1:3))
vs6 = prod(ntuple(d -> Float64(GQ[d]), 3))
k6 = ComplexF64(fx.k)

nRoute1 = 0; nRoute2 = 0
band = Int[]   # indices with rho in [0.55,0.60]
bandTight = 0  # of those, margin in [1,4]
margins = Float64[]
for (i, R) in enumerate(Rs)
    R6 = ntuple(d -> Float64(R[d]), 3)
    rr = sqrt(sum(R6[i]^2 for i in 1:3))
    rho = rd6 / rr
    L = boundLX(fx, rr)
    if L >= 0
        global nRoute1 += 1
        if 0.55 <= rho <= 0.60
            bnd = bndWhlX(fx, rr, L)
            est6 = est(rr, k6, ComplexF64(fx.frq), vs6)
            margin = TOL * est6 / bnd
            push!(margins, margin)
            push!(band, i)
            (1.0 <= margin <= 4.0) && (global bandTight += 1)
        end
    else
        global nRoute2 += 1
    end
end
println("route 1 (whole trapezoid box): ", nRoute1)
println("route 2 (gcd box sum):         ", nRoute2)
println("offsets with rho_whl in [0.55,0.60] (route 1, by definition since route2 has no L): ", length(band))
println("  of those, bound/tol margin in [1,4]x: ", bandTight)
if !isempty(margins)
    println("  margin min/median/max: ", minimum(margins), " / ", margins[div(end,2)+1], " / ", maximum(margins))
end

# also report the L at which these sit (should mostly be 56, the last shell)
Lvals = [boundLX(fx, sqrt(sum(Float64(Rs[i][d])^2 for d in 1:3))) for i in band]
println("  L values on the band (should be dominated by 56): ", sort(unique(Lvals)))
