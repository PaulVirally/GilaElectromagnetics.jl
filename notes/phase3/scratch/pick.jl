using GilaElectromagnetics
const GV = GilaElectromagnetics.GilaVacuum
const QI = Rational{BigInt}
prsQ(s) = (p = split(s, "//"); QI(parse(BigInt, p[1]), parse(BigInt, p[2])))
grp = Dict{Any,Vector{Any}}()
for ln in eachline("notes/farfield/refcache/reftensors_x.txt")
    p = split(strip(ln), '|')
    (length(p) == 7 && p[1] == "vol" && p[6] == "p=220") || continue
    length(split(p[7])) == 9 || continue
    push!(get!(grp, (p[3][4:end], p[4][4:end], p[5][3:end]), []), p[2][3:end])
end
fine = "1//32,1//32,1//32"
want = String[]
for c in ("16", "8", "4", "2")
    push!(want, "1//$c,1//32,1//32", "1//$c,1//$c,1//32", "1//$c,1//$c,1//$c")
end
open("notes/phase3/scratch/keep.txt", "w") do io
    for (ky, offs) in sort(collect(grp); by = e -> (e[1][2], e[1][3]))
        (sTs, sSs, fs) = ky
        (sTs == fine && sSs ∈ want) || continue
        sT = ntuple(d -> prsQ(split(sTs, ',')[d]), 3); sS = ntuple(d -> prsQ(split(sSs, ',')[d]), 3)
        fr = parse.(Float64, split(fs, ';')); f = complex(fr[1], fr[2])
        g = ntuple(d -> min(sT[d], sS[d]), 3)
        Rs = [ntuple(d -> prsQ(split(o, ',')[d]), 3) for o in offs]
        fx = GV.farSetX(sT, sS, f; offs = Rs, nBlk = GV.finBlk(Rs, g,
            ntuple(d -> Int(sT[d] / g[d]), 3), ntuple(d -> Int(sS[d] / g[d]), 3)))
        ok = sort([o for (o, R) in zip(offs, Rs) if GV.farRteX(fx, R)[1] != 3])
        length(ok) > 12 && (ok = [ok[1 + div((i - 1) * length(ok), 12)] for i in 1:12])
        for o in ok; println(io, sTs, "|", sSs, "|", fs, "|", o); end
        println(sSs, " ", fs, " kept ", length(ok), " of ", length(offs))
    end
end
