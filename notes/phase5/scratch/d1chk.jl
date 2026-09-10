# D1: are the f = 1 + 0.1i rows of test/ref/far.txt evaluated at the exact decimal 0.1
# instead of at Float64(0.1), the frequency Gila is actually handed?
const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(ROOT, "notes", "farfield", "ref.jl"))
const QQ = Rational{BigInt}
prsQ(s) = (p = split(s, "//"); QQ(parse(BigInt, p[1]), parse(BigInt, p[2])))
pcx(s) = (u = split(s, ';'); Complex{BigFloat}(parse(BigFloat, u[1]), parse(BigFloat, u[2])))

function rows(prc = 300)
    out = []
    setprecision(BigFloat, prc) do
        for ln in eachline(joinpath(ROOT, "test", "ref", "far.txt"))
            startswith(ln, '#') && continue
            isempty(strip(ln)) && continue
            w = split(ln)
            s = ntuple(d -> prsQ(split(w[1], ',')[d]), 3)
            f = (parse(Float64, w[2]), parse(Float64, w[3]))
            D = ntuple(d -> parse(Int, w[3 + d]), 3)
            push!(out, (s, f, D, reshape([pcx(w[6 + q]) for q in 1:9], 3, 3)))
        end
    end
    out
end

const PRC = 300
const ORD = 36
fDec = setprecision(() -> Complex{BigFloat}(BigFloat(1), parse(BigFloat, "0.1")), BigFloat, PRC)
fF64 = setprecision(() -> Complex{BigFloat}(BigFloat(1), BigFloat(0.1)), BigFloat, PRC)

R = rows()
sel = [r for r in R if r[2] == (1.0, 0.1)]
ctl = [r for r in R if r[2] != (1.0, 0.1)][1:2:end]
println("rows at 1+0.1i: ", length(sel), "  control rows: ", length(ctl)); flush(stdout)

res = Vector{Any}(undef, length(sel))
Threads.@threads for i in eachindex(sel)
    s, f, D, G = sel[i]
    A = volTensor(D, s, fDec; prec = PRC, ord = ORD, cache = false)
    B = volTensor(D, s, fF64; prec = PRC, ord = ORD, cache = false)
    m = maximum(abs, B)
    res[i] = (s, D, Float64(maximum(abs, G .- A) / m), Float64(maximum(abs, G .- B) / m),
              Float64(maximum(abs, A .- B) / m))
end
for (s, D, ed, e6, ab) in res
    println(rpad(join(string.(s), ","), 26), " D=", rpad(string(D), 16),
            "  vs dec ", ed, "   vs f64 ", e6, "   |dec-f64| ", ab)
end
flush(stdout)
println("--- control (other frequencies), vs the file's own f ---")
for (s, f, D, G) in ctl
    fB = setprecision(() -> Complex{BigFloat}(BigFloat(f[1]), BigFloat(f[2])), BigFloat, PRC)
    B = volTensor(D, s, fB; prec = PRC, ord = ORD, cache = false)
    println(rpad(join(string.(s), ","), 26), " f=", f, " D=", rpad(string(D), 16),
            "  ", Float64(maximum(abs, G .- B) / maximum(abs, B)))
    flush(stdout)
end
println("ALLDONE")
