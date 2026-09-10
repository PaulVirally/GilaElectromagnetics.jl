# Regenerate the test/ref/far.txt rows named on the command line (1-based indices into the
# non-comment rows) at f = 1 + Float64(0.1)i, the frequency Gila is handed.  The value is the
# 36-face-pair reference at 220 bits, cross-checked against the independent volume quadrature.
const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(ROOT, "notes", "farfield", "ref.jl"))
const QQ = Rational{BigInt}
prsQ(s) = (p = split(s, "//"); QQ(parse(BigInt, p[1]), parse(BigInt, p[2])))
const FAR = joinpath(ROOT, "test", "ref", "far.txt")
const PRC = 220
const ORD = 44

lins = readlines(FAR)
idx = [i for (i, l) in enumerate(lins) if !startswith(l, '#') && !isempty(strip(l))]
tgt = Set(parse.(Int, ARGS))
fF64 = setprecision(() -> Complex{BigFloat}(BigFloat(1), BigFloat(0.1)), BigFloat, PRC)

job = [(j, i) for (j, i) in enumerate(idx) if j ∈ tgt]
out = Vector{Any}(undef, length(job))
Threads.@threads for q in eachindex(job)
    j, i = job[q]
    w = split(lins[i])
    s = ntuple(d -> prsQ(split(w[1], ',')[d]), 3)
    D = ntuple(d -> parse(Int, w[3 + d]), 3)
    G = refTensor(D, s, fF64; prec = PRC, ord = ORD, cache = false)
    V = volTensor(D, s, fF64; prec = PRC, ord = 32, cache = false)
    old = setprecision(BigFloat, 300) do
        reshape([(u = split(x, ';');
            Complex{BigFloat}(parse(BigFloat, u[1]), parse(BigFloat, u[2]))) for x in w[7:15]], 3, 3)
    end
    cut(x) = string(BigFloat(x; precision = 100))
    lin = join(vcat(w[1:6], [string(cut(real(z)), ";", cut(imag(z))) for z in vec(G)]), " ")
    out[q] = (j, i, lin, Float64(maximum(abs, G .- V) / maximum(abs, G)),
              Float64(maximum(abs, G .- old) / maximum(abs, G)), Float64(maximum(abs, G)),
              join(string.(s), ","), D)
end
for (j, i, lin, dv, dold, mx, ss, D) in sort(out; by = first)
    lins[i] = lin
    println("row ", j, "  ", ss, "  D=", D, "  pairs vs volume ", dv,
            "   new vs old ", dold, "   max|T| ", mx); flush(stdout)
end
open(io -> foreach(l -> println(io, l), lins), FAR, "w")
println("rewrote ", length(job), " rows")
println("ALLDONE")
