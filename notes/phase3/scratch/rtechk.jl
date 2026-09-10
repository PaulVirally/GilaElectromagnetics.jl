using GilaElectromagnetics, Printf
const GV = GilaElectromagnetics.GilaVacuum
const QI = Rational{BigInt}
sq(w) = ntuple(d -> (p = split(split(w, ',')[d], "//"); QI(parse(BigInt, p[1]), parse(BigInt, p[2]))), 3)
grp = Dict{Any,Vector{Any}}()
for ln in eachline("test/ref/farx.txt")
    (isempty(strip(ln)) || startswith(ln, "#")) && continue
    w = split(ln)
    push!(get!(grp, (sq(w[1]), sq(w[2]), complex(parse(Float64, w[3]), parse(Float64, w[4]))), []), sq(w[5]))
end
tot = zeros(Int, 3); tSet = 0.0
for (ky, offs) in grp
    (sT, sS, f) = ky
    g = ntuple(d -> min(sT[d], sS[d]), 3)
    t = @elapsed fx = GV.farSetX(sT, sS, f; offs = offs, nBlk = GV.finBlk(offs, g,
        ntuple(d -> Int(sT[d] / g[d]), 3), ntuple(d -> Int(sS[d] / g[d]), 3)))
    global tSet += t
    r = zeros(Int, 3)
    for R in offs; r[GV.farRteX(fx, R)[1]] += 1; end
    tot .+= r
    println(rpad(string(Float64.(sS)), 34), " f=", f, " routes ", r, @sprintf("  set %.1fs", t))
end
println("total routes 1/2/3 = ", tot, @sprintf("; set build %.1f s", tSet))
