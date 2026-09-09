# 220-bit reference for Gila's far-field 3x3 tensor at one relative offset.
# Two independent routes: the 36 face-pair integrals of g (pairKer) assembled by
# the srfSum! signs, and the volume-volume integral of (d_a d_b - delta_ab lap) g
# over the difference box with the triangle weight. No Gila dependency.

const REFDIR = @__DIR__
const REPODIR = normpath(joinpath(REFDIR, "..", ".."))
isdefined(@__MODULE__, :facePair) ||
    include(joinpath(REPODIR, "notes", "moments", "moments.jl"))
isdefined(@__MODULE__, :pairKer) ||
    include(joinpath(REPODIR, "notes", "gen", "ref", "mom.jl"))

const CACHEDIR = joinpath(REFDIR, "refcache")
const CACHEFILE = joinpath(CACHEDIR, "reftensors.txt")
const CACHE = Dict{String,Vector{Complex{BigFloat}}}()
const CACHEPOS = Ref(0)

fldStr(t::NTuple{3,Int}) = join(t, ",")
fldStr(t::NTuple{3,Rational{BigInt}}) = join((string(numerator(x)) * "//" * string(denominator(x)) for x in t), ",")
fldStr(z::Complex{BigFloat}) = string(real(z)) * ";" * string(imag(z))

refKey(knd, D, s, f, prec, ord) =
    string(knd, "|D=", fldStr(D), "|s=", fldStr(s), "|f=", fldStr(f),
           "|p=", prec, "|n=", ord)

function parCpx(str::AbstractString)
    j = findfirst(';', str)
    return Complex{BigFloat}(parse(BigFloat, str[1:j-1]), parse(BigFloat, str[j+1:end]))
end

# read every line appended since the last sync; a torn trailing line is left for
# the next call, so a concurrent appender cannot corrupt the in-memory cache
function cacheSync!(prec::Int)
    isfile(CACHEFILE) || return nothing
    open(CACHEFILE, "r") do io
        seek(io, CACHEPOS[])
        dat = read(io, String)
        i = findlast('\n', dat)
        i === nothing && return nothing
        CACHEPOS[] += i
        setprecision(BigFloat, prec) do
            for ln in split(dat[1:i], '\n'; keepempty = false)
                j = findlast('|', ln)
                j === nothing && continue
                try
                    CACHE[ln[1:j-1]] = [parCpx(v) for v in split(ln[j+1:end], ' '; keepempty = false)]
                catch
                end
            end
        end
    end
    return nothing
end

# one line per record, written by a single buffered write flushed at close
function cachePut(key::String, vals::AbstractVector{Complex{BigFloat}})
    mkpath(CACHEDIR)
    str = key * "|" * join((fldStr(v) for v in vals), " ") * "\n"
    open(CACHEFILE, "a") do io
        write(io, str)
    end
    CACHE[key] = collect(vals)
    return nothing
end

# srfMat[fp] = (int_F int_F' g dS' dS) / V_t for the 36 pairs, fp = 6(F-1)+F'
function refPairs(D::NTuple{3,Int}, s::NTuple{3,Rational{BigInt}}, f::Complex{BigFloat};
                  prec::Int = 220, ord::Int = 44, cache::Bool = true)
    key = refKey("pairs", D, s, f, prec, ord)
    if cache
        haskey(CACHE, key) || cacheSync!(prec)
        haskey(CACHE, key) && return CACHE[key]
    end
    val = setprecision(BigFloat, prec) do
        bf(p) = map(iv -> (BigFloat(iv[1]), BigFloat(iv[2])), p)
        vt = prod(BigFloat.(s))
        fB = Complex{BigFloat}(f)
        [begin
             pA, pB = facePair(D, F, Fp, s)
             pairKer(bf(pA), bf(pB), fB; ordN = ord, ordX = ord, ordE = ord) / vt
         end for F in 1:6 for Fp in 1:6]
    end
    cache && cachePut(key, val)
    return val
end

# Gila's srfSum! signs applied to the 36 scaled face-pair values
function refSum(sm::AbstractVector{<:Complex})
    G = zeros(eltype(sm), 3, 3)
    G[1,1] = sm[15] - sm[16] - sm[21] + sm[22] + sm[29] - sm[30] - sm[35] + sm[36]
    G[2,1] = -sm[13] + sm[14] + sm[19] - sm[20]
    G[3,1] = -sm[25] + sm[26] + sm[31] - sm[32]
    G[1,2] = -sm[3] + sm[4] + sm[9] - sm[10]
    G[2,2] = sm[1] - sm[2] - sm[7] + sm[8] + sm[29] - sm[30] - sm[35] + sm[36]
    G[3,2] = -sm[27] + sm[28] + sm[33] - sm[34]
    G[1,3] = -sm[5] + sm[6] + sm[11] - sm[12]
    G[2,3] = -sm[17] + sm[18] + sm[23] - sm[24]
    G[3,3] = sm[1] - sm[2] - sm[7] + sm[8] + sm[15] - sm[16] - sm[21] + sm[22]
    return G
end

# the face pairs summed into each tensor entry, for the amplification metric
const SUMIDX = permutedims(reshape(Vector{Int}[
    [15,16,21,22,29,30,35,36], [13,14,19,20], [25,26,31,32],
    [3,4,9,10], [1,2,7,8,29,30,35,36], [27,28,33,34],
    [5,6,11,12], [17,18,23,24], [1,2,7,8,15,16,21,22]], 3, 3))

refTensor(D::NTuple{3,Int}, s::NTuple{3,Rational{BigInt}}, f::Complex{BigFloat};
          prec::Int = 220, ord::Int = 44, cache::Bool = true) =
    refSum(refPairs(D, s, f; prec = prec, ord = ord, cache = cache))

# Gauss-Legendre nodes and weights on [-1, 1], Newton on the Legendre recursion
function glNodes(n::Int, ::Type{T}) where {T}
    x = Vector{T}(undef, n)
    w = Vector{T}(undef, n)
    for i in 1:n
        z = cos(T(pi) * (4i - 1) / (4n + 2))
        p0 = one(T); p1 = z; dp = one(T)
        for _ in 1:300
            p0 = one(T); p1 = z
            for k in 2:n
                p0, p1 = p1, ((2k - 1) * z * p1 - (k - 1) * p0) / k
            end
            dp = n * (z * p1 - p0) / (z^2 - 1)
            dz = -p1 / dp
            z += dz
            abs(dz) <= 4 * eps(T) * abs(z) && break
        end
        p0 = one(T); p1 = z
        for k in 2:n
            p0, p1 = p1, ((2k - 1) * z * p1 - (k - 1) * p0) / k
        end
        dp = n * (z * p1 - p0) / (z^2 - 1)
        x[i] = z
        w[i] = 2 / ((1 - z^2) * dp^2)
    end
    return x, w
end

# composite GL rule on [lo, hi], panels doubling in width away from the endpoint
# closest to t; a single panel when the near singularity at distance dmn is farther
# than the interval is long
function grdRule(lo::T, hi::T, t::T, dmn::T, x0::Vector{T}, w0::Vector{T}) where {T}
    L = hi - lo
    bps = T[lo, hi]
    h = dmn / 2
    h > 0 || error("grdRule: singularity inside the box, dmn = 0")
    if h < L
        cts = T[]
        c = h
        while c < L
            push!(cts, c)
            c *= 2
        end
        bps = abs(lo - t) <= abs(hi - t) ? vcat(T[lo], lo .+ cts, T[hi]) :
                                           vcat(T[lo], hi .- reverse(cts), T[hi])
        sort!(bps)
        unique!(bps)
    end
    nds = T[]
    wts = T[]
    for j in 1:(length(bps) - 1)
        a, b = bps[j], bps[j + 1]
        b <= a && continue
        m = (a + b) / 2
        hw = (b - a) / 2
        append!(nds, m .+ hw .* x0)
        append!(wts, hw .* w0)
    end
    return nds, wts
end

# (1/V_t) int_D w(delta) [(d_a d_b - delta_ab lap) g](R + delta) d delta over the
# difference box D = prod [-s_i, s_i], split at 0 (the weight has a kink there) and
# graded towards the corner nearest the singularity at delta = -R
function volTensor(D::NTuple{3,Int}, s::NTuple{3,Rational{BigInt}}, f::Complex{BigFloat};
                   prec::Int = 220, ord::Int = 32, cache::Bool = true)
    key = refKey("vol", D, s, f, prec, ord)
    if cache
        haskey(CACHE, key) || cacheSync!(prec)
        haskey(CACHE, key) && return reshape(CACHE[key], 3, 3)
    end
    G = setprecision(BigFloat, prec) do
        T = BigFloat
        F = Complex{T}
        sB = ntuple(i -> T(s[i]), 3)
        R = ntuple(i -> T(D[i]) * sB[i], 3)
        fB = F(f)
        k = 2 * T(pi) * fB
        k2 = k^2
        cst = inv(4 * T(pi) * fB^2)
        x0, w0 = glNodes(ord, T)
        acc = zeros(F, 3, 3)
        for sgn in Iterators.product((-1, 1), (-1, 1), (-1, 1))
            lo = ntuple(i -> sgn[i] < 0 ? -sB[i] : zero(T), 3)
            hi = ntuple(i -> sgn[i] < 0 ? zero(T) : sB[i], 3)
            np = ntuple(i -> min(max(-R[i], lo[i]), hi[i]), 3)
            dmn = sqrt(sum(ntuple(i -> (np[i] + R[i])^2, 3)))
            ax = ntuple(i -> grdRule(lo[i], hi[i], -R[i], dmn, x0, w0), 3)
            n1, n2, n3 = length(ax[1][1]), length(ax[2][1]), length(ax[3][1])
            for i1 in 1:n1
                d1 = ax[1][1][i1]; a1 = ax[1][2][i1] * (sB[1] - abs(d1)); x1 = R[1] + d1
                for i2 in 1:n2
                    d2 = ax[2][1][i2]; a2 = a1 * ax[2][2][i2] * (sB[2] - abs(d2)); x2 = R[2] + d2
                    for i3 in 1:n3
                        d3 = ax[3][1][i3]; a3 = a2 * ax[3][2][i3] * (sB[3] - abs(d3))
                        x3 = R[3] + d3
                        r = sqrt(x1^2 + x2^2 + x3^2)
                        ri = inv(r)
                        g = cst * exp(im * k * r) * ri
                        gp = g * (im * k - ri)
                        gpp = gp * (im * k - ri) + g * ri^2
                        q = a3 * (gpp - gp * ri) * ri^2
                        h = a3 * gp * ri
                        dg = a3 * k2 * g
                        acc[1,1] += q * x1 * x1 + h + dg
                        acc[2,2] += q * x2 * x2 + h + dg
                        acc[3,3] += q * x3 * x3 + h + dg
                        acc[1,2] += q * x1 * x2
                        acc[1,3] += q * x1 * x3
                        acc[2,3] += q * x2 * x3
                    end
                end
            end
        end
        acc[2,1] = acc[1,2]; acc[3,1] = acc[1,3]; acc[3,2] = acc[2,3]
        acc ./ prod(sB)
    end
    cache && cachePut(key, vec(G))
    return G
end
