# Independent BigFloat reference for the volume form, with no shared code with ref.jl and no
# dependence on the face-pair machinery: a tensor Gauss-Legendre rule on the difference box with
# the exact triangle weight,
#   T_ab(R) = (1/V_t) int_D w(d) [(da db + dab k^2) g](R + d) dd,  g = e^{ikr}/(4 pi f^2 r).
# Every axis is cut at 0 (the kink of w) and, on an axis with |D_a| <= 1, graded geometrically
# towards the point R_a + d_a = 0, which is where the singularity at d = -R touches the box: with
# uniform cuts that point is a panel endpoint and the rule stalls at 1e-12 (unify measured 5.9e-12
# at the slender (1,0,2)), with the grading it reaches 1e-45.
# API:  refQuad(D, s, f; ord = 32, lvl = 8, rat = 2, prc = 256, cache = true)
# Cached as kind `tns:refQuad` in refcache/reftensors.txt, the file verify.jl reads.
const QI = Rational{BigInt}
const RQDIR = @__DIR__

# Gauss-Legendre nodes and weights on [-1,1], Newton on the Legendre recursion, in T
function glRule(n::Int, ::Type{T}) where {T<:AbstractFloat}
    x = Vector{T}(undef, n); w = Vector{T}(undef, n)
    for k in 1:n
        z = cos(T(pi) * (T(k) - T(1) / 4) / (T(n) + T(1) / 2))
        dp = zero(T)
        for _ in 1:200
            p0 = one(T); p1 = zero(T)
            for j in 1:n
                p2 = p1; p1 = p0
                p0 = ((2 * T(j) - 1) * z * p1 - (T(j) - 1) * p2) / T(j)
            end
            dp = n * (z * p0 - p1) / (z^2 - 1)
            dz = p0 / dp
            z -= dz
            abs(dz) < eps(T) * 4 && break
        end
        x[k] = z; w[k] = 2 / ((1 - z^2) * dp^2)
    end
    return x, w
end

# (da db + dab k^2) g at y
function kerTen!(G, y::NTuple{3,T}, k::Complex{T}, frq::Complex{T}) where {T<:AbstractFloat}
    r2 = y[1]^2 + y[2]^2 + y[3]^2
    r = sqrt(r2)
    u = exp(im * k * r) / r
    c1 = im * k / r - 1 / r2
    c2 = -k^2 - 3 * im * k / r + 3 / r2
    for a in 1:3, b in 1:3
        G[a, b] = u * ((a == b) * (c1 + k^2) + y[a] * y[b] / r2 * c2) / (4 * T(pi) * frq^2)
    end
    return G
end

"Cuts of [-s,s]: 0 always, and geometric grading towards -D s when |D| <= 1."
function rqCut(s::QI, D::Int, lvl::Int, rat::QI)
    abs(D) <= 1 || return QI[-s, 0, s]
    t0 = -QI(D) * s
    wt = [rat^(j - 1) for j in 1:lvl]; tot = sum(wt)
    cs = QI[zero(QI)]
    for j in 1:lvl; push!(cs, cs[end] + wt[j] // tot); end
    lo = [t0 - (t0 + s) * c for c in cs if t0 - (t0 + s) * c >= -s]
    hi = [t0 + (s - t0) * c for c in cs if t0 + (s - t0) * c <= s]
    return sort(unique(vcat(lo, hi, QI[-s, 0, s])))
end

function refQuad(D::NTuple{3,Int}, s::NTuple{3,QI}, frq::Complex{Float64};
                 ord::Int = 32, lvl::Int = 8, rat::Int = 2, prc::Int = 256, cache::Bool = true)
    G = setprecision(BigFloat, prc) do
        T = BigFloat; C = Complex{T}
        k = 2 * T(pi) * C(frq)
        fq = C(frq)
        R = ntuple(i -> T(QI(D[i]) * s[i]), 3)
        sT = ntuple(i -> T(s[i]), 3)
        ck = ntuple(i -> rqCut(s[i], D[i], lvl, QI(rat)), 3)
        gx, gw = glRule(ord, T)
        tot = zeros(C, 3, 3); Gk = zeros(C, 3, 3)
        nd = ntuple(i -> [T(c) for c in ck[i]], 3)
        for j1 in 1:(length(ck[1]) - 1), j2 in 1:(length(ck[2]) - 1), j3 in 1:(length(ck[3]) - 1)
            jj = (j1, j2, j3)
            lo = ntuple(i -> nd[i][jj[i]], 3); hi = ntuple(i -> nd[i][jj[i] + 1], 3)
            hw = ntuple(i -> (hi[i] - lo[i]) / 2, 3); md = ntuple(i -> (hi[i] + lo[i]) / 2, 3)
            pt = ntuple(i -> [md[i] + hw[i] * gx[q] for q in 1:ord], 3)
            wt = ntuple(i -> [hw[i] * gw[q] * (sT[i] - abs(md[i] + hw[i] * gx[q])) for q in 1:ord], 3)
            for q1 in 1:ord, q2 in 1:ord
                w12 = wt[1][q1] * wt[2][q2]
                y1 = R[1] + pt[1][q1]; y2 = R[2] + pt[2][q2]
                for q3 in 1:ord
                    kerTen!(Gk, (y1, y2, R[3] + pt[3][q3]), k, fq)
                    ww = w12 * wt[3][q3]
                    for a in 1:3, b in 1:3
                        tot[a, b] += ww * Gk[a, b]
                    end
                end
            end
        end
        tot ./ prod(sT)
    end
    cache && rqPut(D, s, frq, ord, prc, G)
    return G
end

"Append one record to refcache/reftensors.txt in the format verify.jl reads."
function rqPut(D, s, frq, ord, prc, G)
    dir = joinpath(RQDIR, "refcache"); mkpath(dir)
    ln = string("tns:refQuad|D=", join(D, ","), "|s=",
                join((string(numerator(x)) * "//" * string(denominator(x)) for x in s), ","),
                "|f=", real(frq), ";", imag(frq), "|p=", prc, "|n=", ord, "|",
                join((setprecision(() -> string(real(G[i])) * ";" * string(imag(G[i])),
                                   BigFloat, prc) for i in 1:9), " "))
    open(io -> println(io, ln), joinpath(dir, "reftensors.txt"), "a")
    return nothing
end
