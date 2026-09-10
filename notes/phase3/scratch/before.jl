using GilaElectromagnetics, StaticArrays, Printf
const GV = GilaElectromagnetics.GilaVacuum
const QI = Rational{BigInt}
prsQ(s) = (p = split(s, "//"); QI(parse(BigInt, p[1]), parse(BigInt, p[2])))

grp = Dict{Any,Vector{Any}}()
for ln in eachline(joinpath(@__DIR__, "..", "..", "farfield", "refcache", "reftensors_x.txt"))
    p = split(strip(ln), '|')
    (length(p) == 7 && p[1] == "vol" && p[6] == "p=220") || continue
    v = split(p[7]); length(v) == 9 || continue
    R = Tuple(prsQ.(split(p[2][3:end], ',')))
    sT = Tuple(prsQ.(split(p[3][4:end], ','))); sS = Tuple(prsQ.(split(p[4][4:end], ',')))
    fr = parse.(Float64, split(p[5][3:end], ';'))
    G = reshape([complex(parse(BigFloat, x[1]), parse(BigFloat, x[2])) for x in split.(v, ';')], 3, 3)
    push!(get!(grp, (sT, sS, complex(fr[1], fr[2])), []), (R, G))
end
println(sum(length, values(grp)), " reference rows in ", length(grp), " groups")

function gilaOld(R, sT, sS, f)
    opt = GV.CPUKerOpt{Float64}(f, Float64, false, GV.CPU())
    sTr = Rational{Int}.(sT); sSr = Rational{Int}.(sS)
    tf = Float64.(GV.cubFac(sTr)); sf = Float64.(GV.cubFac(sSr))
    G = zeros(ComplexF64, 3, 3)
    GV.egoFunOut!(G, SVector{3,Float64}(Float64.(R)), sTr, sSr, tf, sf, GV.facPar, opt)
    (G, maximum(round.(Int, abs.(Float64.(R)) ./ Float64.(max.(sT, sS)))))
end
rel(A, B) = Float64(maximum(abs, ComplexF64.(A) .- B) / maximum(abs, B))
srt(v) = isempty(v) ? (NaN, NaN, NaN) : (minimum(v), sort(v)[max(1, div(length(v), 2))], maximum(v))

setprecision(BigFloat, 220)
eAdp = Float64[]; eFxd = Float64[]; lAdp = Float64[]; lFxd = Float64[]; rte = zeros(Int, 3)
for (ky, cas) in grp
    (sT, sS, f) = ky
    fx = GV.farSetXMem(sT, sS, ComplexF64(f); nBlk = GV.finBlk(first.(cas),
        ntuple(d -> min(sT[d], sS[d]), 3), ntuple(d -> Int(sT[d] / min(sT[d], sS[d])), 3),
        ntuple(d -> Int(sS[d] / min(sT[d], sS[d])), 3)), offs = first.(cas))
    for (R, Gr) in cas
        (G, sep) = gilaOld(R, sT, sS, f)
        all(isfinite, G) || continue
        L = GV.farTnsX(R, sT, sS, ComplexF64(f); fs = fx)
        rte[GV.farRteX(fx, R)[1]] += 1
        if sep <= 1
            push!(eAdp, rel(G, Gr)); push!(lAdp, rel(L, Gr))
        else
            push!(eFxd, rel(G, Gr)); push!(lFxd, rel(L, Gr))
        end
    end
end
@printf("near band (Gila adaptive, sep<=1): n=%d  Gila %.2e / %.2e / %.2e   library %.2e / %.2e / %.2e\n",
    length(eAdp), srt(eAdp)..., srt(lAdp)...)
@printf("separated (Gila fixed rule, sep>=2): n=%d  Gila %.2e / %.2e / %.2e   library %.2e / %.2e / %.2e\n",
    length(eFxd), srt(eFxd)..., srt(lFxd)...)
println("library routes 1/2/3 = ", rte)
