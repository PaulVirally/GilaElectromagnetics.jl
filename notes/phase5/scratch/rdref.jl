# parsers for test/ref/{far,farx}.txt
const QI = isdefined(Main, :QI) ? Main.QI : Rational{BigInt}
const REFROOT = normpath(joinpath(@__DIR__, "..", "..", "..", "test", "ref"))

pq(s) = (u = split(s, "//"); QI(parse(BigInt, u[1]), parse(BigInt, u[2])))
ptrip(s) = ntuple(d -> pq(split(s, ',')[d]), 3)
pcx(s) = (u = split(s, ';'); Complex{BigFloat}(parse(BigFloat, u[1]), parse(BigFloat, u[2])))

function rdFar(prc::Int = 260)
    out = Tuple{NTuple{3,QI},ComplexF64,NTuple{3,Int},Matrix{Complex{BigFloat}}}[]
    setprecision(BigFloat, prc) do
        for ln in eachline(joinpath(REFROOT, "far.txt"))
            startswith(ln, '#') && continue
            w = split(ln)
            isempty(w) && continue
            s = ptrip(w[1]); f = ComplexF64(parse(Float64, w[2]), parse(Float64, w[3]))
            D = ntuple(d -> parse(Int, w[3 + d]), 3)
            G = reshape([pcx(w[6 + q]) for q in 1:9], 3, 3)
            push!(out, (s, f, D, G))
        end
    end
    out
end

function rdFarX(prc::Int = 260)
    out = Tuple{NTuple{3,QI},NTuple{3,QI},ComplexF64,NTuple{3,QI},Matrix{Complex{BigFloat}}}[]
    setprecision(BigFloat, prc) do
        for ln in eachline(joinpath(REFROOT, "farx.txt"))
            startswith(ln, '#') && continue
            w = split(ln)
            isempty(w) && continue
            sT = ptrip(w[1]); sS = ptrip(w[2])
            f = ComplexF64(parse(Float64, w[3]), parse(Float64, w[4]))
            R = ptrip(w[5])
            G = reshape([pcx(w[5 + q]) for q in 1:9], 3, 3)
            push!(out, (sT, sS, f, R, G))
        end
    end
    out
end
