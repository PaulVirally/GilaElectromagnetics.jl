# Which entries of test/ref/*.txt are genuine high-precision decimals and which are the 32-digit
# expansion of a Float64 (i.e. lost their precision somewhere in the curation)?
const T5 = normpath(joinpath(@__DIR__, "..", "..", "..", "test", "ref"))
pr(x...) = (println(x...); flush(stdout))
setprecision(BigFloat, 260)

function scan(fil, skip::Int)
    n = 0; f64 = 0; big = 0; zro = 0
    bad = String[]
    for ln in eachline(joinpath(T5, fil))
        startswith(ln, '#') && continue
        w = split(ln)
        isempty(w) && continue
        vals = String[]
        for t in w[(skip + 1):end]
            append!(vals, split(t, ';'))
        end
        rows64 = 0; rowsbg = 0
        for v in vals
            x = parse(BigFloat, v)
            n += 1
            if iszero(x); zro += 1; continue; end
            d = abs(x - BigFloat(Float64(x))) / abs(x)
            if d < 1e-30; f64 += 1; rows64 += 1 else big += 1; rowsbg += 1 end
        end
        rows64 > 0 && push!(bad, string(join(w[1:min(skip,4)], " "), "  [", rows64, "/", rows64 + rowsbg, "]"))
    end
    pr(fil, ": ", n, " numbers, ", zro, " zero, ", big, " genuine >Float64, ", f64,
       " indistinguishable from a Float64 round trip")
    pr("   rows with a Float64 round-trip entry: ", length(bad))
    for b in unique(bad)[1:min(end, 20)]; pr("     ", b); end
end

scan("far.txt", 6)
scan("farx.txt", 5)
scan("mom.txt", 5)
pr("done")
