# Curate test/ref/far.txt from notes/farfield/refcache/reftensors.txt: one line per
# (shape, frequency, offset), the tensor already assembled from the 36 face pairs where that is
# the reference kind.  Run: julia --project notes/phase2/scratch/mkref.jl
const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
using GilaElectromagnetics
const GV = GilaElectromagnetics.GilaVacuum
const QI = Rational{BigInt}
prsQ(s) = (p = split(s, "//"); QI(parse(BigInt, p[1]), parse(BigInt, p[2])))

# only the clean production shapes: the random audit shapes have one offset each and would each
# cost a fresh geometry table
const KEEP = ["1//32,1//32,1//32", "1//4,1//4,1//4", "1//8,1//8,1//8",
              "1//32,1//32,1//512", "1//512,1//32,1//32"]
const NPER = 12                 # offsets kept per (shape, frequency)

R = Dict{Any,Any}(); O = Dict{Any,Int}()
for ln in eachline(joinpath(ROOT, "notes", "farfield", "refcache", "reftensors.txt"))
    (isempty(strip(ln)) || startswith(ln, "#")) && continue
    p = split(strip(ln), '|'); length(p) >= 7 || continue
    p[3][3:end] ∈ KEEP || continue
    fr = parse.(Float64, split(p[4][3:end], ';'))
    ky = (String(p[1]), p[2][3:end], p[3][3:end], (fr[1], fr[2]))
    ord = parse(Int, p[6][3:end])
    ord < get(O, ky, -1) && continue
    O[ky] = ord
    R[ky] = [Complex(parse(BigFloat, x[1]), parse(BigFloat, x[2])) for x in split.(split(p[7]), ';')]
end

cases = Tuple{String,String,Tuple{Float64,Float64}}[]
for grp in sort(unique([(k[3], k[4]) for k in keys(R)]))
    ds = sort(unique([k[2] for k in keys(R) if (k[3], k[4]) == grp]))
    ste = max(1, cld(length(ds), NPER))
    append!(cases, [(d, grp[1], grp[2]) for d in ds[1:ste:end]])
end
open(joinpath(ROOT, "test", "ref", "far.txt"), "w") do io
    println(io, "# Separated-pair tensors T_ab of the vacuum Green operator, curated from")
    println(io, "# notes/farfield/refcache/reftensors.txt (220-bit references, notes/farfield/ref.jl).")
    println(io, "# Columns: cell scale, Re f, Im f, offset in cells, then T11 T21 T31 T12 ... T33 as")
    println(io, "# re;im.  Read only by tests, never by a build.")
    n = 0
    setprecision(BigFloat, 220) do
        for (D, s, f) in cases
            v, kind = if haskey(R, ("pairs", D, s, f))
                G = zeros(Complex{BigFloat}, 3, 3)
                GV.srfSum!(G, R[("pairs", D, s, f)]); (vec(G), "pairs")
            elseif haskey(R, ("vol", D, s, f))
                (R[("vol", D, s, f)], "vol")
            else
                k = findfirst(k -> k[2] == D && k[3] == s && k[4] == f &&
                              startswith(k[1], "tns"), collect(keys(R)))
                k === nothing ? (nothing, "") : (R[collect(keys(R))[k]], "tns")
            end
            v === nothing && continue
            n += 1
            cut(x) = string(BigFloat(x; precision = 100))
            println(io, s, " ", f[1], " ", f[2], " ", replace(D, "," => " "), " ",
                    join([string(cut(real(z)), ";", cut(imag(z))) for z in v], " "))
        end
    end
    println("wrote ", n, " tensors")
end
