# Equal-cell adversarial sweep: shapes, aspect ratios, offsets and frequencies that appear in no
# existing test or reference file.  Every offset is answered by all the routes that are available
# there and each is scored against a fresh 220-bit reference computed by refgen.jl.
using GilaElectromagnetics
const GV = GilaElectromagnetics.GilaVacuum
include(joinpath(@__DIR__, "refgen.jl"))

pr(x...) = (println(x...); flush(stdout))
mx(A) = Float64(maximum(abs, A))
const ORD = parse(Int, get(ENV, "RGORD", "32"))


# shapes nothing in test/ref uses: coprime anisotropy, non-dyadic bases, extreme aspect ratios,
# and the coarse lambda/2 and lambda cells
const SHP = Dict(
    :aniso  => (3//64, 5//64, 7//64),        # coprime edges, ratio 2.33
    :aniso2 => (1//8, 1//24, 1//48),         # ratio 6, non-dyadic
    :rod    => (1//6, 1//96, 1//96),         # ratio 16, non-dyadic base
    :plate  => (1//10, 1//10, 1//10000),     # aspect 1000, non-dyadic
    :needle => (1//1000, 1//40, 1//13),      # all three edges different and non-dyadic
    :half   => (1//2, 1//2, 1//2),           # lambda/2 cube
    :one    => (1//1, 1//1, 1//1),           # lambda cube
    :thick  => (1//3, 1//5, 1//7))           # coarse, coprime

const FRQ = Dict(:f3 => 3.0 + 0.0im, :fA => 0.5 + 2.0im, :fB => 2.0 + 2.0im,
                 :f1 => 1.0 + 0.0im, :fC => 1.0 + 1.0im, :fD => 1.0 + 0.1im,
                 :f37 => 0.37 + 0.0im)

# sweep the route kind along a few rays and return the offsets straddling every kind change,
# plus the two extremes; a wrong route is invisible from the interior of a route's own region
function bndOff(fs, rays, rmax::Int)
    ks = Dict{NTuple{3,Int},Int}()
    out = NTuple{3,Int}[]
    for u in rays
        prv = 0
        for n in 2:rmax
            D = ntuple(d -> u[d] * n, 3)
            maximum(abs, D) >= 2 || continue
            k = try
                first(GV.farRte(fs, D))
            catch
                -1
            end
            ks[D] = k
            if k != prv && prv != 0
                push!(out, ntuple(d -> u[d] * (n - 1), 3)); push!(out, D)
            end
            prv = k
        end
    end
    unique!(out), ks
end

function run(shp::Symbol, frq::Symbol; extra = NTuple{3,Int}[], rmax = 40, ksr = false)
    s = SHP[shp]; f = FRQ[frq]
    sQ = ntuple(d -> Rational{BigInt}(s[d]), 3)
    rays = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1), (1, 1, 0), (2, 1, 0)]
    pr("\n=== ", shp, " ", Float64.(s), "  f = ", f, " ===")
    fs0 = try
        GV.farSet(sQ, f; nBlk = 4 * rmax)
    catch e
        pr("  farSet FAILED: ", sprint(showerror, e)[1:min(end, 300)]); return
    end
    offs, ks = bndOff(fs0, rays, rmax)
    hst = Dict{Int,Int}(); for (_, k) in ks; hst[k] = get(hst, k, 0) + 1; end
    pr("  route histogram over ", length(ks), " ray offsets: ", sort(collect(hst)))
    append!(offs, extra)
    unique!(offs)
    filter!(D -> maximum(abs, D) >= 2, offs)
    isempty(offs) && (pr("  no offsets"); return)
    fs = GV.farSet(sQ, f; nBlk = 4 * rmax, offs = Tuple(offs))
    ws = GV.FarWrk(max(fs.Lw, fs.Lo), Float64)
    for D in sort(offs)
        R = ntuple(d -> Float64(D[d]) * fs.s[d], 3)
        kind, L, Lc, _ = GV.farRte(fs, D)
        Gr = try
            rgTns(ntuple(d -> Rational{BigInt}(D[d]) * sQ[d], 3), sQ, sQ, ComplexF64(f);
                  ord = ORD)
        catch e
            pr("  D=", D, " ref FAILED ", sprint(showerror, e)[1:min(end, 200)]); continue
        end
        mr = mx(Gr)
        G = zeros(ComplexF64, 3, 3)
        kind == 1 ? GV.tnsWhl!(G, fs, ws, R, L) :
        kind == 2 ? GV.tnsOct!(G, fs, ws, R, Lc) : copyto!(G, GV.ksrCch!(fs, D))
        e0 = mx(G .- Gr) / mr
        crt = GV.certEq(fs, D, kind, L, Lc) / mr
        # every alternative route that is available at this offset
        alt = String[]
        if kind != 1
            Lw = GV.whlLvl(fs, sqrt(sum(R[i]^2 for i in 1:3)))
            if Lw < 0
                G1 = zeros(ComplexF64, 3, 3); GV.tnsWhl!(G1, fs, ws, R, fs.Lw)
                push!(alt, string("r1@Lw=", fs.Lw, " ", mx(G1 .- Gr) / mr))
            end
        end
        if kind != 2
            Lc2 = GV.octLvl(fs, R)
            if !any(<(0), Lc2) && maximum(Lc2) <= fs.Lo
                G2 = zeros(ComplexF64, 3, 3); GV.tnsOct!(G2, fs, ws, R, Lc2)
                push!(alt, string("r2 ", mx(G2 .- Gr) / mr))
            end
        end
        if ksr && kind != 3
            G3 = copy(GV.ksrCch!(fs, D))
            push!(alt, string("r3 ", mx(G3 .- Gr) / mr))
        end
        pr("  D=", D, " kind=", kind, " maxT=", mr, "  err=", e0, "  cert=", crt,
           isempty(alt) ? "" : "  | " * join(alt, "  "))
    end
end
