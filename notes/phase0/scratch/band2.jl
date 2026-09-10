include("/Users/pvirally/.julia/dev/GilaElectromagnetics/notes/farfield/farfield.jl")

const GQ = (QI(1,32), QI(1,32), QI(1,32))
const f1 = ComplexF64(1.0)
g = GQ[1]
sS = (16g, 16g, 16g)
b  = ntuple(d -> (GQ[d] + sS[d]) // 2, 3)

Rs = [(QI(2k1 + 1)//64, QI(2k2 + 1)//64, QI(2k3 + 1)//64) for k3 in -56:55 for k2 in -56:55 for k1 in 8:119]
xTouch(R, b) = all(abs(R[d]) <= b[d] for d in 1:3)
filter!(R -> !xTouch(R, b), Rs)
fx = farSetupX(GQ, sS, f1; offs = Rs)

b6 = ntuple(d -> Float64(b[d]), 3)
rd6 = sqrt(sum(b6[i]^2 for i in 1:3))

band = [R for R in Rs if let rr = sqrt(sum(Float64(R[d])^2 for d in 1:3)); 0.55 <= rd6/rr <= 0.60; end]
println("band size: ", length(band))

function scan(band, fx)
    ratios = Float64[]
    tight = 0
    Ldist = Dict{Int,Int}()
    for R in band
        R6 = ntuple(d -> Float64(R[d]), 3); rr = sqrt(sum(R6[i]^2 for i in 1:3))
        L = boundLX(fx, rr)
        Ldist[L] = get(Ldist, L, 0) + 1
        G = zeros(ComplexF64, 3, 3)
        tnsWhlX!(G, fx, FarWs(fx.Lw, Float64), R6, L)
        mx = maximum(abs, G)
        bnd = bndWhlX(fx, rr, L)
        ratio = bnd / (TOL * mx)
        push!(ratios, ratio)
        (1.0 <= ratio <= 4.0) && (tight += 1)
    end
    return ratios, tight, Ldist
end
ratios, tight, Ldist = scan(band, fx)
println("L distribution on band: ", Ldist)
println("ratio bound/(tol*max|G|) : min ", minimum(ratios), " median ", ratios[div(end,2)], " max ", maximum(ratios))
println("count with ratio in [1,4]: ", tight, " of ", length(band))
println("count with ratio > 1 (i.e. certificate looser than nominal tol given actual maxG): ", count(>(1.0), ratios))
println("count with ratio in [0.9,4] (near/at 1x-4x): ", count(r -> 0.9 <= r <= 4.0, ratios))

# restrict to offsets sitting exactly at the last shell L = 56 (LMAX): the table
# genuinely has no more shells for these, vs L=52/54 which have headroom within L<=56
idxL56 = [i for i in eachindex(band) if boundLX(fx, sqrt(sum(Float64(band[i][d])^2 for d in 1:3))) == 56]
println("\noffsets at L = 56 exactly (in the 0.55-0.60 band): ", length(idxL56))
ratios56 = ratios[idxL56]
println("of those, ratio in [1,4]: ", count(r -> 1.0 <= r <= 4.0, ratios56), " of ", length(ratios56))
println("ratio56 min/median/max: ", minimum(ratios56), " / ", ratios56[div(end,2)], " / ", maximum(ratios56))

# also: how many offsets in the WHOLE block (not just rho 0.55-0.60) sit at L=56 with ratio in [1,4]?

# worst-margin offset in the L=56 sub-band: smallest ratio
wi = idxL56[argmin(ratios56)]
Rw = band[wi]
println("\nworst-margin offset: R = ", Rw, "  ratio(L=56) = ", ratios[wi])
rrW = sqrt(sum(Float64(Rw[d])^2 for d in 1:3))
println("rho_whl = ", rd6/rrW)

println("\n--- raising LMAX: rebuild the table at higher L for this shape/pair, no offs ---")
for Ltry in (56, 64, 72, 80)
    tB = @elapsed fxL = farSetupX(GQ, sS, f1; offs = (Rw,), L = Ltry)
    Lgot = boundLX(fxL, rrW)
    G = zeros(ComplexF64, 3, 3)
    if Lgot >= 0
        tnsWhlX!(G, fxL, FarWs(fxL.Lw, Float64), ntuple(d->Float64(Rw[d]),3), Lgot)
        mx = maximum(abs, G)
        bnd = bndWhlX(fxL, rrW, Lgot)
        r = bnd/(TOL*mx)
        println("L cap = ", Ltry, "  selected L = ", Lgot, "  ratio = ", r, "  table build (this offset only) = ", round(tB,digits=2), " s")
    else
        println("L cap = ", Ltry, "  still route 2 (no whole-box L found)  table build = ", round(tB,digits=2), " s")
    end
end

println("\n--- forcing a larger L directly (bypassing boundLX's greedy smallest-L search) ---")
t0 = @elapsed fx80 = farSetupX(GQ, sS, f1; offs = (Rw,), L = 80, lDef = 80)
println("table build at L cap 80: ", round(t0,digits=2), " s")
G = zeros(ComplexF64, 3, 3)
for Lforce in (52, 54, 56, 58, 60, 64, 70, 80)
    tnsWhlX!(G, fx80, FarWs(fx80.Lw, Float64), ntuple(d->Float64(Rw[d]),3), Lforce)
    mx = maximum(abs, G)
    bnd = bndWhlX(fx80, rrW, Lforce)
    r = bnd/(TOL*mx)
    println("  L = ", Lforce, "  bound = ", bnd, "  ratio bound/(tol*maxG) = ", r)
end
