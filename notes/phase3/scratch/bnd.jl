# refusal boundary: sweep the x gap of an off-lattice equal-cell pair, report the route
# histogram farFil! would see and whether the live build succeeds
using GilaElectromagnetics, Printf
const GV = GilaElectromagnetics.GilaVacuum
const QI = Rational{BigInt}

scl = (1//16, 1//16, 1//16)
org = (0//1, 0//1, 0//1)
frq = ComplexF64(1)
tol = GV.farTol(Float64)

function probe(orgB)
    volA = GlaVol((2,2,2), scl, org)
    volB = GlaVol((2,2,2), scl, orgB)
    sgT = GV.sepGrd(volA, volB, 0); sgS = GV.sepGrd(volA, volB, 1)
    nCrc = (4,4,4)
    lat = GV.extOff(nCrc, volA.cel, volA.scl, sgT, sgS)
    lat === nothing && return (:noLat, nothing, nothing)
    cntLim = (volA.scl .+ volB.scl) .// 2
    Ds = NTuple{3,Int}[]; nCnt = 0
    for p in CartesianIndices(nCrc)
        any(d -> p[d] == volA.cel[d] + 1, 1:3) && continue
        sep = ntuple(d -> QI(GV.grdSep(p[d], volA.cel[d], d, sgT, sgS)), 3)
        if all(d -> abs(sep[d]) <= cntLim[d], 1:3)
            nCnt += 1
        else
            push!(Ds, ntuple(d -> lat[1][d][p[d]], 3))
        end
    end
    off = ntuple(d -> Float64(lat[2][d]), 3)
    st = GV.farSetMem(ntuple(d -> QI(volA.scl[d]), 3), frq; tol = tol,
        nBlk = maximum(D -> maximum(abs, D), Ds) + 1, nMin = all(iszero, off) ? 2 : 1,
        off = ntuple(d -> abs(off[d]), 3), disk = false)
    rte = zeros(Int, 3)
    for D in Ds
        rte[GV.farRte(st, ntuple(d -> Float64(D[d]) * st.s[d] + off[d], 3))[1]] += 1
    end
    (off, nCnt, rte)
end

println("orgB                       shift            cnt  routes 1/2/3   build")
for j in 0:12
    orgB = ((8 + j)//64, 1//32, 0//1)
    o, nCnt, rte = probe(orgB)
    bld = try
        m = GlaVacOprMem(CPUKerOpt{Float64}(), GlaVol((2,2,2), scl, org),
            GlaVol((2,2,2), scl, orgB))
        all(v -> all(isfinite, v), m.egoFur) ? "ok" : "nonfinite"
    catch e
        "ERR " * first(split(sprint(showerror, e), '\n'))[1:min(48, end)]
    end
    @printf("%-26s %-16s %3s  %-14s %s\n", string(orgB), string(o),
        o === :noLat ? "-" : string(nCnt), o === :noLat ? "-" : string(rte), bld)
end
