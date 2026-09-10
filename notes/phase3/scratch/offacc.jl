# an off-lattice pair with a gap of one cell or more: accuracy of the shifted farFil! path
# against the exact-rational farTnsX, and reachability of the two sizing fixes
using GilaElectromagnetics, Printf
const GV = GilaElectromagnetics.GilaVacuum
const QI = Rational{BigInt}
scl = (1//16, 1//16, 1//16); org = (0//1, 0//1, 0//1)
frq = ComplexF64(1); tol = GV.farTol(Float64)
sQ = ntuple(d -> QI(scl[d]), 3)

function setup(orgB)
    volA = GlaVol((2,2,2), scl, org); volB = GlaVol((2,2,2), scl, orgB)
    sgT = GV.sepGrd(volA, volB, 0); sgS = GV.sepGrd(volA, volB, 1)
    lat = GV.extOff((4,4,4), volA.cel, volA.scl, sgT, sgS)
    Ds = NTuple{3,Int}[]; Rs = NTuple{3,QI}[]
    for p in CartesianIndices((4,4,4))
        any(d -> p[d] == 3, 1:3) && continue
        push!(Rs, ntuple(d -> QI(GV.grdSep(p[d], 2, d, sgT, sgS)), 3))
        push!(Ds, ntuple(d -> lat[1][d][p[d]], 3))
    end
    (Ds, Rs, ntuple(d -> Float64(lat[2][d]), 3))
end

for orgB in ((3//16, 1//32, 0//1), (1//4, 1//32, 0//1), (5//16, 3//64, 1//32))
    Ds, Rs, off = setup(orgB)
    G = Array{ComplexF64}(undef, 3, 3, length(Ds))
    GV.farFil!(Ds, sQ, frq, off; tol = tol, disk = false) do q; view(G, :, :, q); end
    fx = GV.farSetX(sQ, sQ, frq; offs = Rs, tol = tol,
        nBlk = GV.finBlk(Rs, sQ, (1,1,1), (1,1,1)))
    wrs = 0.0
    for q in eachindex(Rs)
        X = GV.farTnsX(Rs[q], sQ, sQ, frq; fs = fx)
        wrs = max(wrs, maximum(abs, view(G, :, :, q) .- X) / maximum(abs, X))
    end
    @printf("%-24s off %-30s %2d offsets  max rel %.2e\n", string(orgB), string(off),
        length(Ds), wrs)
end

# the nerOff fix: size the table on the unshifted lattice and route the shifted offsets
Ds, Rs, off = setup((1//4, 1//32, 0//1))
st = GV.farSet(sQ, frq; tol = tol, nBlk = maximum(D -> maximum(abs, D), Ds) + 1,
                nMin = 1, off = GV.ZR3, disk = false)
bad = 0
for D in Ds
    try; GV.farRte(st, ntuple(d -> Float64(D[d]) * st.s[d] + off[d], 3)); catch e
        bad += 1
        bad == 1 && println("nerOff unshifted: ", first(split(sprint(showerror, e), '\n')))
    end
end
println("nerOff unshifted sizing fails on $bad of $(length(Ds)) offsets")
