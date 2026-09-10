using GilaElectromagnetics, Printf
const GV = GilaElectromagnetics.GilaVacuum
const QI = Rational{BigInt}
prsQ(s) = (p = split(s, "//"); QI(parse(BigInt, p[1]), parse(BigInt, p[2])))
setprecision(BigFloat, 220)
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
rel(A, B) = Float64(maximum(abs, ComplexF64.(A) .- B) / maximum(abs, B))
srt(v) = isempty(v) ? (NaN, NaN, NaN) : (minimum(v), sort(v)[max(1, div(length(v), 2))], maximum(v))
eAdp = Float64[]; eFxd = Float64[]; rte = zeros(Int, 3); nSkp = 0
for (ky, cas) in sort(collect(grp); by = e -> string(e[1]))
    (sT, sS, f) = ky
    gQ = ntuple(d -> min(sT[d], sS[d]), 3)
    fx = try
        GV.farSetXMem(sT, sS, ComplexF64(f); offs = first.(cas),
            nBlk = GV.finBlk(first.(cas), gQ, ntuple(d -> Int(sT[d] / gQ[d]), 3),
                ntuple(d -> Int(sS[d] / gQ[d]), 3)))
    catch e
        global nSkp += length(cas); println("skip ", Float64.(sT), " ", Float64.(sS), ": ",
            sprint(showerror, e)[1:min(90, end)]); continue
    end
    for (R, Gr) in cas
        kind = GV.farRteX(fx, R)[1]
        rte[kind] += 1
        kind == 3 && (global nSkp += 1; continue)
        e = rel(GV.farTnsX(R, sT, sS, ComplexF64(f); fs = fx), Gr)
        push!(maximum(round.(Int, abs.(Float64.(R)) ./ Float64.(max.(sT, sS)))) <= 1 ?
            eAdp : eFxd, e)
    end
end
@printf("near band (sep<=1): n=%d  min/med/max %.2e %.2e %.2e\n", length(eAdp), srt(eAdp)...)
@printf("separated (sep>=2): n=%d  min/med/max %.2e %.2e %.2e\n", length(eFxd), srt(eFxd)...)
println("routes 1/2/3 = ", rte, "; skipped ", nSkp)
