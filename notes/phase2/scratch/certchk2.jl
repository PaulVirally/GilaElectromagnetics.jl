using GilaElectromagnetics
const GVF = GilaElectromagnetics.GilaVacuum
const farQI = Rational{BigInt}
farSql(w) = ntuple(dir -> (p = split(split(w, ',')[dir], "//");
    farQI(parse(BigInt, p[1]), parse(BigInt, p[2]))), 3)
grp = Dict{Any,Any}()
for lin ∈ eachline(joinpath(@__DIR__, "..", "..", "..", "test", "ref", "far.txt"))
    (isempty(strip(lin)) || startswith(lin, "#")) && continue
    w = split(lin)
    key = (farSql(w[1]), complex(parse(Float64, w[2]), parse(Float64, w[3])))
    push!(get!(grp, key, []), (ntuple(d -> parse(Int, w[3+d]), 3),
        reshape([(z = split(x, ';'); complex(parse(Float64, z[1]), parse(Float64, z[2]))) for x ∈ w[7:15]], 3, 3)))
end
worst = Ref{Any}((0.0, nothing))
for ((sQ, f), cas) in grp
    fs = GVF.farSet(sQ, f; offs = first.(cas))
    for (D, Tr) in cas
        Tn = GVF.farTns(D, sQ, f; fs = fs)
        e = maximum(abs, Tn .- Tr)
        k, L, Lc, _ = GVF.farRte(fs, D)
        c = GVF.certEq(fs, D, k, L, Lc); m = maximum(abs, Tn)
        r = e / (c + eps() * m)
        r > worst[][1] && (worst[] = (r, (Float64.(sQ), f, D, k, e, c, m, e / (eps() * m))))
    end
end
println("worst err/(cert + eps max|T|) = ", worst[][1])
println("  (scl, f, D, route, err, cert, maxT, err/(eps maxT)) = ", worst[][2])
