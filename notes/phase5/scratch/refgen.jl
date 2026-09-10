# Independent BigFloat reference for the vacuum far tensor of a separated cell pair, equal or
# unequal cells.  No shared code with src/ and none with notes/farfield/ref.jl: the difference-box
# form is rederived here from the definition, the weight being the per-axis convolution of the two
# cell indicators (a trapezoid; a triangle when the cells are equal).
#
#   T_ab(R) = (1/V_t) int_D prod_i w_i(d_i) [(da db + dab k^2) g](R + d) dd,
#   w_i(t) = min(min(sT_i, sS_i), b_i - |t|),  b_i = (sT_i + sS_i)/2,  g = e^{ikr}/(4 pi f^2 r).
#
# Cut at every kink of w and, on an axis whose singular projection -R_i falls inside the support,
# graded geometrically towards it.  Cached append-only in refs.txt.

const QI = Rational{BigInt}
const RGDIR = @__DIR__
const RGFILE = joinpath(RGDIR, "refs.txt")
const RGCACHE = Dict{String,Matrix{Complex{BigFloat}}}()
const RGPOS = Ref(0)

qstr(x::QI) = string(numerator(x), "//", denominator(x))
tstr(t) = join(qstr.(QI.(t)), ",")

function rgKey(R, sT, sS, f, ord, lvl, prc)
    string("R=", tstr(R), "|sT=", tstr(sT), "|sS=", tstr(sS),
           "|f=", real(f), ";", imag(f), "|o=", ord, "|g=", lvl, "|p=", prc)
end

function rgSync!(prc::Int)
    isfile(RGFILE) || return nothing
    open(RGFILE, "r") do io
        seek(io, RGPOS[])
        dat = read(io, String)
        i = findlast('\n', dat)
        i === nothing && return nothing
        RGPOS[] += i
        setprecision(BigFloat, prc) do
            for ln in split(dat[1:i], '\n'; keepempty = false)
                j = findlast('|', ln)
                j === nothing && continue
                vs = split(ln[j+1:end], ' '; keepempty = false)
                length(vs) == 9 || continue
                try
                    G = Matrix{Complex{BigFloat}}(undef, 3, 3)
                    for q in 1:9
                        u = split(vs[q], ';')
                        G[q] = Complex{BigFloat}(parse(BigFloat, u[1]), parse(BigFloat, u[2]))
                    end
                    RGCACHE[ln[1:j-1]] = G
                catch
                end
            end
        end
    end
    nothing
end

function rgPut(key::String, G, prc::Int)
    ln = setprecision(BigFloat, prc) do
        key * "|" * join((string(real(G[q])) * ";" * string(imag(G[q])) for q in 1:9), " ")
    end
    open(io -> println(io, ln), RGFILE, "a")
    RGCACHE[key] = collect(G)
    nothing
end

# Gauss-Legendre nodes and weights on [-1,1] in T, Newton on the three-term recursion
function glr(n::Int, ::Type{T}) where {T}
    x = Vector{T}(undef, n); w = Vector{T}(undef, n)
    for k in 1:n
        z = cos(T(pi) * (4k - 1) / (4n + 2)); dp = zero(T)
        for _ in 1:400
            p0 = one(T); p1 = zero(T)
            for j in 1:n
                p2 = p1; p1 = p0
                p0 = ((2 * T(j) - 1) * z * p1 - (T(j) - 1) * p2) / T(j)
            end
            dp = n * (z * p0 - p1) / (z^2 - 1)
            dz = p0 / dp; z -= dz
            abs(dz) < 4 * eps(T) * max(abs(z), one(T)) && break
        end
        x[k] = z; w[k] = 2 / ((1 - z^2) * dp^2)
    end
    x, w
end

# panel breakpoints on one axis: the support ends, the two trapezoid kinks, the origin, and
# panels doubling in width away from the singular projection t0 = -R_i when it lies inside, the
# first of them half the distance dmn from the singularity to the box
function rgCut(bQ::QI, aQ::QI, t0::QI, dmn::QI)
    cs = QI[-bQ, -aQ, zero(QI), aQ, bQ]
    if dmn > 0 && -bQ <= t0 <= bQ
        push!(cs, t0)
        h = dmn // 2
        while h < 2 * bQ
            t0 + h < bQ && push!(cs, t0 + h)
            t0 - h > -bQ && push!(cs, t0 - h)
            h *= 2
        end
    end
    sort!(unique!(cs))
end

# (da db + dab k^2) g at y
function rgKer!(G, y::NTuple{3,T}, k::Complex{T}, f::Complex{T}) where {T}
    r2 = y[1]^2 + y[2]^2 + y[3]^2
    r = sqrt(r2)
    u = exp(im * k * r) / r
    c1 = im * k / r - 1 / r2
    c2 = -k^2 - 3 * im * k / r + 3 / r2
    p = 1 / (4 * T(pi) * f^2)
    @inbounds for a in 1:3, b in 1:3
        G[a, b] = p * u * ((a == b) * (c1 + k^2) + y[a] * y[b] * c2 / r2)
    end
    G
end

function rgTns(R, sT, sS, f::Complex{Float64};
               ord::Int = 32, grd::Bool = true, prc::Int = 256, cache::Bool = true)
    RQ = ntuple(d -> QI(R[d]), 3); sTQ = ntuple(d -> QI(sT[d]), 3); sSQ = ntuple(d -> QI(sS[d]), 3)
    bT = ntuple(d -> (sTQ[d] + sSQ[d]) // 2, 3)
    # the grading of an axis whose singular projection sits exactly on the box face was added
    # after the first sweep, so those keys are versioned apart from the cached ones
    key = rgKey(RQ, sTQ, sSQ, f, ord, grd ? 1 : 0, prc) *
          (any(abs(RQ[d]) == bT[d] for d in 1:3) ? "|v2" : "")
    if cache
        haskey(RGCACHE, key) || rgSync!(prc)
        haskey(RGCACHE, key) && return RGCACHE[key]
    end
    bQ = ntuple(d -> (sTQ[d] + sSQ[d]) // 2, 3)
    aQ = ntuple(d -> abs(sTQ[d] - sSQ[d]) // 2, 3)
    mnQ = ntuple(d -> min(sTQ[d], sSQ[d]), 3)
    all(abs(RQ[d]) <= bQ[d] for d in 1:3) && error("rgTns: the cells overlap at R = $RQ")
    # squared distance from the singularity at -R to the difference box, exactly
    d2 = sum((min(max(-RQ[d], -bQ[d]), bQ[d]) + RQ[d])^2 for d in 1:3)
    dmn = grd ? QI(rationalize(BigInt, sqrt(Float64(d2)); tol = 1e-6)) : zero(QI)
    G = setprecision(BigFloat, prc) do
        T = BigFloat; C = Complex{T}
        k = 2 * T(pi) * C(f); fB = C(f)
        Rb = ntuple(d -> T(RQ[d]), 3)
        mn = ntuple(d -> T(mnQ[d]), 3); bb = ntuple(d -> T(bQ[d]), 3)
        ck = ntuple(d -> rgCut(bQ[d], aQ[d], -RQ[d], dmn), 3)
        nd = ntuple(d -> [T(c) for c in ck[d]], 3)
        gx, gw = glr(ord, T)
        np = ntuple(d -> length(nd[d]) - 1, 3)
        jobs = [(j1, j2, j3) for j1 in 1:np[1] for j2 in 1:np[2] for j3 in 1:np[3]]
        acc = [zeros(C, 3, 3) for _ in 1:Threads.maxthreadid()]
        Threads.@threads :static for jj in jobs
            tot = acc[Threads.threadid()]; Gk = zeros(C, 3, 3)
            lo = ntuple(d -> nd[d][jj[d]], 3); hi = ntuple(d -> nd[d][jj[d] + 1], 3)
            hw = ntuple(d -> (hi[d] - lo[d]) / 2, 3); md = ntuple(d -> (hi[d] + lo[d]) / 2, 3)
            pt = ntuple(d -> [md[d] + hw[d] * gx[q] for q in 1:ord], 3)
            wt = ntuple(d -> [hw[d] * gw[q] * min(mn[d], bb[d] - abs(md[d] + hw[d] * gx[q]))
                              for q in 1:ord], 3)
            for q1 in 1:ord, q2 in 1:ord
                w12 = wt[1][q1] * wt[2][q2]
                y1 = Rb[1] + pt[1][q1]; y2 = Rb[2] + pt[2][q2]
                for q3 in 1:ord
                    rgKer!(Gk, (y1, y2, Rb[3] + pt[3][q3]), k, fB)
                    ww = w12 * wt[3][q3]
                    @inbounds for a in 1:3, b in 1:3
                        tot[a, b] += ww * Gk[a, b]
                    end
                end
            end
        end
        sum(acc) ./ prod(ntuple(d -> T(sTQ[d]), 3))
    end
    cache && rgPut(key, G, prc)
    G
end
