# is the nerOff shift-aware sizing reachable from a live (non-refused) pair? sweep off-lattice
# separated pairs and compare a table sized on the unshifted lattice against the shifted one
using GilaElectromagnetics, Printf
const GV = GilaElectromagnetics.GilaVacuum
const QI = Rational{BigInt}
scl = (1//16, 1//16, 1//16); frq = ComplexF64(1); tol = GV.farTol(Float64)
sQ = ntuple(d -> QI(scl[d]), 3)

function pair(cel, orgB)
    volA = GlaVol(cel, scl, (0//1,0//1,0//1)); volB = GlaVol(cel, scl, orgB)
    sgT = GV.sepGrd(volA, volB, 0); sgS = GV.sepGrd(volA, volB, 1)
    nCrc = ntuple(d -> 2 * cel[d], 3)
    lat = GV.extOff(nCrc, volA.cel, volA.scl, sgT, sgS)
    lat === nothing && return nothing
    cntLim = (volA.scl .+ volB.scl) .// 2
    Ds = NTuple{3,Int}[]; nCnt = 0
    for p in CartesianIndices(nCrc)
        any(d -> p[d] == volA.cel[d] + 1, 1:3) && continue
        sep = ntuple(d -> QI(GV.grdSep(p[d], volA.cel[d], d, sgT, sgS)), 3)
        all(d -> abs(sep[d]) <= cntLim[d], 1:3) ? (nCnt += 1) :
            push!(Ds, ntuple(d -> lat[1][d][p[d]], 3))
    end
    (Ds, ntuple(d -> Float64(lat[2][d]), 3), nCnt)
end

fails(st, Ds, off) = count(Ds) do D
    try; GV.farRte(st, ntuple(d -> Float64(D[d]) * st.s[d] + off[d], 3)); false
    catch; true; end
end

nBad = 0; nLive = 0; msg = ""
for cel in ((2,2,2), (3,2,2)), gx in 4:2:10, sy in (0//1, 1//64, 1//32, 3//64),
    sz in (0//1, 1//32)
    orgB = ((8 + gx)//64, sy, sz)
    p = pair(cel, orgB); p === nothing && continue
    Ds, off, nCnt = p
    nB = maximum(D -> maximum(abs, D), Ds) + 1
    stFix = GV.farSetMem(sQ, frq; tol = tol, nBlk = nB, nMin = all(iszero, off) ? 2 : 1,
        off = ntuple(d -> abs(off[d]), 3), disk = false)
    rte = [GV.farRte(stFix, ntuple(d -> Float64(D[d]) * stFix.s[d] + off[d], 3))[1] for D in Ds]
    3 in rte && continue                    # refused pair, not a live path
    global nLive += 1
    stBad = GV.farSetMem(sQ, frq; tol = tol, nBlk = nB, nMin = 2, off = GV.ZR3, disk = false)
    nf = fails(stBad, Ds, off)
    if nf > 0
        global nBad += 1
        if isempty(msg)
            try; GV.farRte(stBad, ntuple(d -> Float64(Ds[1][d]) * stBad.s[d] + off[d], 3))
            catch; end
            for D in Ds
                try; GV.farRte(stBad, ntuple(d -> Float64(D[d]) * stBad.s[d] + off[d], 3))
                catch e; global msg = first(split(sprint(showerror, e), '\n')); break; end
            end
        end
        @printf("cel %s orgB %-26s off %-28s %d/%d offsets fail unshifted\n", string(cel),
            string(orgB), string(off), nf, length(Ds))
    end
end
println("live off-lattice pairs swept: $nLive; unshifted sizing fails on $nBad of them")
isempty(msg) || println("example: ", msg)
