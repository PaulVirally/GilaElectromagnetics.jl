using GilaElectromagnetics, StaticArrays, Printf
const GV = GilaElectromagnetics.GilaVacuum
const QI = Rational{BigInt}
prsQ(s) = (p = split(s, "//"); QI(parse(BigInt, p[1]), parse(BigInt, p[2])))
REF = "/Users/pvirally/.julia/dev/GilaElectromagnetics/notes/farfield/refcache/reftensors_x.txt"

setprecision(BigFloat, 220)
rows = []
for ln in eachline(REF)
    p = split(strip(ln), '|')
    (length(p) == 7 && p[1] == "vol" && p[6] == "p=220") || continue
    v = split(p[7]); length(v) == 9 || continue
    R = Tuple(prsQ.(split(p[2][3:end], ',')))
    sT = Tuple(prsQ.(split(p[3][4:end], ','))); sS = Tuple(prsQ.(split(p[4][4:end], ',')))
    fr = parse.(Float64, split(p[5][3:end], ';'))
    G = reshape([complex(parse(BigFloat, x[1]), parse(BigFloat, x[2])) for x in split.(v, ';')], 3, 3)
    push!(rows, (R, sT, sS, complex(fr[1], fr[2]), G))
end
println(length(rows), " reference rows")
fp = isa(GV.facPar, Function) ? GV.facPar() : GV.facPar
opt(f) = try GV.CPUKerOpt{Float64}(f, 48, false, GV.CPU()) catch
    GV.CPUKerOpt{Float64}(f, Float64, false, GV.CPU()) end
rel(A, B) = Float64(maximum(abs, ComplexF64.(A) .- B) / maximum(abs, B))
srt(v) = isempty(v) ? (NaN, NaN, NaN) : (minimum(v), sort(v)[max(1, div(length(v), 2))], maximum(v))
eAdp = Float64[]; eFxd = Float64[]; nBad = 0
for (R, sT, sS, f, Gr) in rows
    sTr = Rational{Int}.(sT); sSr = Rational{Int}.(sS)
    tf = Float64.(GV.cubFac(sTr)); sf = Float64.(GV.cubFac(sSr))
    G = zeros(ComplexF64, 3, 3)
    try
        GV.egoFunOut!(G, SVector{3,Float64}(Float64.(R)), sTr, sSr, tf, sf, fp, opt(f))
    catch e
        global nBad += 1; continue
    end
    all(isfinite, G) || (global nBad += 1; continue)
    push!(maximum(round.(Int, abs.(Float64.(R)) ./ Float64.(max.(sT, sS)))) <= 1 ? eAdp : eFxd,
        rel(G, Gr))
end
@printf("near band  (adaptive hcubature, sep<=1): n=%d  min/med/max %.2e %.2e %.2e\n", length(eAdp), srt(eAdp)...)
@printf("separated  (fixed GL rule,      sep>=2): n=%d  min/med/max %.2e %.2e %.2e\n", length(eFxd), srt(eFxd)...)
println("skipped ", nBad)
