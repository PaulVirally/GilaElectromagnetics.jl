# High precision evaluator of panel-pair moments I_p = int_A int_B r^p dA dA'.
#
# Reduction: for axis-aligned rectangles A, B the 4D integral collapses onto the
# separation w = rA - rB.  Along each axis the weight is the box-box convolution
# (overlap length), a trapezoid; an axis where one panel is degenerate gives an
# indicator; an axis where both are degenerate is frozen and only adds a
# constant to r^2.  The domain is thus a union of axis-aligned boxes carrying a
# multilinear weight.  After splitting at 0 and reflecting into the positive
# octant, a box is singular only when every lower corner coordinate is 0 and
# there is no frozen offset; those are handled by a Duffy transform which makes
# the radial integrand an exact polynomial, everything else by tensor
# Gauss-Legendre on an analytic integrand.

# --- Gauss-Legendre nodes/weights in arbitrary precision ---------------------
function glNds(n::Integer, T = BigFloat)
    x = Vector{T}(undef, n)
    w = Vector{T}(undef, n)
    for i in 1:n
        z = cos(T(π) * (i - T(1) / 4) / (n + T(1) / 2))
        for _ in 1:200
            pm1 = one(T); p = z
            for k in 2:n
                pm2 = pm1; pm1 = p
                p = ((2k - 1) * z * pm1 - (k - 1) * pm2) / k
            end
            n == 1 && (p = z; pm1 = one(T))
            dp = n * (z * p - pm1) / (z^2 - 1)
            dz = -p / dp
            z += dz
            abs(dz) < eps(T) * 8 && break
        end
        pm1 = one(T); p = z
        for k in 2:n
            pm2 = pm1; pm1 = p
            p = ((2k - 1) * z * pm1 - (k - 1) * pm2) / k
        end
        n == 1 && (p = z; pm1 = one(T))
        dp = n * (z * p - pm1) / (z^2 - 1)
        x[i] = z
        w[i] = 2 / ((1 - z^2) * dp^2)
    end
    return x, w
end
# nodes/weights mapped to [a, b]
function glMap(nds, wts, a, b)
    return (b + a) / 2 .+ (b - a) / 2 .* nds, ((b - a) / 2) .* wts
end

const NDCH = Dict{Tuple{Int,Int},Tuple{Vector{BigFloat},Vector{BigFloat}}}()
glGet(n) = get!(NDCH, (n, precision(BigFloat))) do
    glNds(n, BigFloat)
end

# --- per-axis convolution weight --------------------------------------------
# A = (p, q), B = (r, s).  Returns (:frozen, offset) or (:pieces, [(lo, hi, c, d)])
# with weight c + d * w on [lo, hi].
function axsWgt(A, B)
    p, q = A
    r, s = B
    if p == q && r == s
        return (:frozen, p - r)
    elseif p == q
        return (:pieces, [(p - s, p - r, one(p), zero(p))])
    elseif r == s
        return (:pieces, [(p - r, q - r, one(p), zero(p))])
    end
    dns(w) = max(zero(p), min(q, s + w) - max(p, r + w))
    lo = p - s; hi = q - r
    bps = sort(unique(filter(x -> lo <= x <= hi, [p - s, q - s, p - r, q - r])))
    pcs = Tuple{typeof(p),typeof(p),typeof(p),typeof(p)}[]
    for i in 1:(length(bps) - 1)
        α, β = bps[i], bps[i + 1]
        β <= α && continue
        dα, dβ = dns(α), dns(β)
        d = (dβ - dα) / (β - α)
        c = dα - d * α
        push!(pcs, (α, β, c, d))
    end
    return (:pieces, pcs)
end

# --- box list for a panel pair ----------------------------------------------
# pnl = ((x1, x2), (y1, y2), (z1, z2)) with one degenerate axis.
# Returns (boxes, off2) where each box is (lo::Vector, hi::Vector, c::Vector,
# d::Vector) in the positive octant.
function pairBox(pA, pB)
    T = typeof(pA[1][1])
    off2 = zero(T)
    axs = Vector{Vector{NTuple{4,T}}}()
    for i in 1:3
        kind, val = axsWgt(pA[i], pB[i])
        if kind == :frozen
            off2 += val^2
        else
            push!(axs, val)
        end
    end
    d = length(axs)
    boxes = Tuple{Vector{T},Vector{T},Vector{T},Vector{T}}[]
    for combo in Iterators.product(axs...)
        # split at 0 per axis then reflect into the positive octant
        splt = Vector{Vector{NTuple{4,T}}}()
        for (lo, hi, c, dd) in combo
            sub = NTuple{4,T}[]
            if lo < 0 < hi
                push!(sub, (lo, zero(T), c, dd))
                push!(sub, (zero(T), hi, c, dd))
            else
                push!(sub, (lo, hi, c, dd))
            end
            push!(splt, sub)
        end
        for sc in Iterators.product(splt...)
            lov = Vector{T}(undef, d); hiv = similar(lov)
            cv = similar(lov); dv = similar(lov)
            for (i, (lo, hi, c, dd)) in enumerate(sc)
                if hi <= 0
                    lov[i] = -hi; hiv[i] = -lo; cv[i] = c; dv[i] = -dd
                else
                    lov[i] = lo; hiv[i] = hi; cv[i] = c; dv[i] = dd
                end
            end
            any(hiv .<= lov) && continue
            push!(boxes, (lov, hiv, cv, dv))
        end
    end
    return boxes, off2
end

# --- integration ------------------------------------------------------------
# accumulate wgt * r^p for p in -1:pMax into acc (acc[p + 2])
@inline function accPow!(acc, wgt, r, pMax)
    rp = inv(r)
    @inbounds for j in 1:(pMax + 2)
        acc[j] += wgt * rp
        rp *= r
    end
    return nothing
end

# composite rule on [lo, hi] graded away from the near corner: the integrand
# r^p has a near singularity at distance dmn, so panels double in width from lo
function grdInt(ordN, lo, hi, dmn)
    T = typeof(lo)
    bps = [lo]
    h = dmn / 2
    while lo + h < hi && h < (hi - lo)
        push!(bps, lo + h)
        h *= 2
    end
    push!(bps, hi)
    nds = T[]; wts = T[]
    n0, w0 = glGet(ordN)
    for i in 1:(length(bps) - 1)
        n, w = glMap(n0, w0, bps[i], bps[i + 1])
        append!(nds, n); append!(wts, w)
    end
    return nds, wts
end

function boxReg!(add!, box, off2, ordN)
    lov, hiv, cv, dv = box
    d = length(lov)
    dmn = sqrt(sum(x -> x^2, lov) + off2)
    xs = [grdInt(ordN, lov[i], hiv[i], dmn) for i in 1:d]
    n1 = length(xs[1][1]); n2 = length(xs[2][1])
    if d == 2
        @inbounds for i1 in 1:n1
            u1 = xs[1][1][i1]; w1 = xs[1][2][i1] * (cv[1] + dv[1] * u1)
            for i2 in 1:n2
                u2 = xs[2][1][i2]; w2 = w1 * xs[2][2][i2] * (cv[2] + dv[2] * u2)
                add!(w2, sqrt(u1^2 + u2^2 + off2))
            end
        end
    else
        n3 = length(xs[3][1])
        @inbounds for i1 in 1:n1
            u1 = xs[1][1][i1]; w1 = xs[1][2][i1] * (cv[1] + dv[1] * u1)
            for i2 in 1:n2
                u2 = xs[2][1][i2]; w2 = w1 * xs[2][2][i2] * (cv[2] + dv[2] * u2)
                for i3 in 1:n3
                    u3 = xs[3][1][i3]
                    w3 = w2 * xs[3][2][i3] * (cv[3] + dv[3] * u3)
                    add!(w3, sqrt(u1^2 + u2^2 + u3^2 + off2))
                end
            end
        end
    end
    return nothing
end

# composite rule on [0, 1], graded towards 0 so a boundary layer of width ρ is
# resolved; needed when the box aspect ratio is far from 1
function grdRul(ordE, ρ, T)
    bps = [zero(T), one(T)]
    if ρ < 1 // 2
        h = T(ρ) / 2
        bps = [zero(T)]
        while h < 1 // 2
            push!(bps, h)
            h *= 2
        end
        push!(bps, one(T))
    end
    nds = T[]; wts = T[]
    n0, w0 = glGet(ordE)
    for i in 1:(length(bps) - 1)
        n, w = glMap(n0, w0, bps[i], bps[i + 1])
        append!(nds, n); append!(wts, w)
    end
    return nds, wts
end

function boxDuf!(add!, box, ordX, ordE)
    lov, hiv, cv, dv = box
    d = length(lov)
    β = hiv
    T = eltype(β)
    ndX, wtX = glMap(glGet(ordX)..., zero(T), one(T))
    ρ = minimum(β) / maximum(β)
    ndE, wtE = grdRul(ordE, ρ, T)
    ordE = length(ndE)
    prdβ = prod(β)
    u = Vector{T}(undef, d)
    for k in 1:d
        oth = filter(!=(k), 1:d)
        for eidx in Iterators.product(ntuple(_ -> 1:ordE, d - 1)...)
            wE = one(T)
            q = β[k]^2
            for (m, j) in enumerate(oth)
                wE *= wtE[eidx[m]]
                q += (β[j] * ndE[eidx[m]])^2
            end
            sq = sqrt(q)
            for ix in 1:ordX
                ξ = ndX[ix]
                u[k] = β[k] * ξ
                for (m, j) in enumerate(oth)
                    u[j] = β[j] * ξ * ndE[eidx[m]]
                end
                wgt = wtX[ix] * wE * prdβ * ξ^(d - 1)
                for i in 1:d
                    wgt *= (cv[i] + dv[i] * u[i])
                end
                add!(wgt, ξ * sq)
            end
        end
    end
    return nothing
end

function pairInt(add!, pA, pB, ordN, ordX, ordE)
    boxes, off2 = pairBox(pA, pB)
    for bx in boxes
        if off2 == 0 && all(bx[1] .== 0)
            boxDuf!(add!, bx, ordX, ordE)
        else
            boxReg!(add!, bx, off2, ordN)
        end
    end
    return nothing
end

# I[p + 2] = int int r^p, p = -1 .. pMax
function pairMom(pA, pB, pMax; ordN = 44, ordX = 30, ordE = 44)
    T = typeof(pA[1][1])
    acc = zeros(T, pMax + 2)
    pairInt(pA, pB, ordN, ordX, ordE) do wgt, r
        accPow!(acc, wgt, r, pMax)
    end
    return acc
end

# direct high precision int int exp(2 pi i f r) / (4 pi r f^2)
function pairKer(pA, pB, frq; ordN = 44, ordX = 44, ordE = 44)
    T = typeof(pA[1][1])
    F = Complex{T}
    f = F(frq)
    acc = Ref(zero(F))
    pairInt(pA, pB, ordN, ordX, ordE) do wgt, r
        acc[] += wgt * exp(2 * T(π) * im * f * r) / (4 * T(π) * r * f^2)
    end
    return acc[]
end

# --- geometry construction, mirroring wekGrdPts! ----------------------------
# dir -> (gridX, gridY, gridZ)
function grdScl(dir, scl)
    dir == 1 && return (scl[1], scl[2], scl[3])
    dir == 2 && return (scl[3], scl[1], scl[2])
    dir == 3 && return (scl[2], scl[3], scl[1])
    error("bad dir")
end
# panel A: grid points 1, 2, 5, 4
pnlA(gX, gY, gZ) = ((zero(gX), gX), (zero(gY), gY), (zero(gZ), zero(gZ)))
# the eight panel pairs Gila needs, keyed by name
function pnlPair(nam, gX, gY, gZ)
    z = zero(gX)
    A = pnlA(gX, gY, gZ)
    nam == :slf && return (A, A)
    # edge adjacent, coplanar, shared edge along y (length gY), widths gX
    nam == :edgFltX && return (A, ((gX, 2gX), (z, gY), (z, z)))
    # edge adjacent, coplanar, shared edge along x (length gX), widths gY
    nam == :edgFltY && return (A, ((z, gX), (gY, 2gY), (z, z)))
    # edge adjacent, cornered: B at x = gX, spans y in [0, gY], z in [0, gZ]
    nam == :edgCrnX && return (A, ((gX, gX), (z, gY), (z, gZ)))
    # edge adjacent, cornered: B at y = gY, spans x in [0, gX], z in [0, gZ]
    nam == :edgCrnY && return (A, ((z, gX), (gY, gY), (z, gZ)))
    # vertex adjacent, coplanar
    nam == :vtxCop && return (A, ((gX, 2gX), (gY, 2gY), (z, z)))
    # vertex adjacent, perpendicular, B at x = gX
    nam == :vtxPrpX && return (A, ((gX, gX), (gY, 2gY), (z, gZ)))
    # vertex adjacent, perpendicular, B at y = gY
    nam == :vtxPrpY && return (A, ((gX, 2gX), (gY, gY), (z, gZ)))
    error("bad name")
end

# --- series -----------------------------------------------------------------
# partial sums S_N = (1 / (4 pi f^2)) sum_{n=0}^{N} (2 pi i f)^n I_{n - 1} / n!
function serPar(mom, frq, nMax)
    T = eltype(mom)
    F = Complex{T}
    z = 2 * T(π) * im * F(frq)
    prt = Vector{F}(undef, nMax + 1)
    trm = Vector{F}(undef, nMax + 1)
    acc = zero(F)
    cof = one(F)
    for n in 0:nMax
        t = cof * mom[n + 1]
        acc += t
        trm[n + 1] = t
        prt[n + 1] = acc / (4 * T(π) * F(frq)^2)
        cof *= z / (n + 1)
    end
    return prt, trm ./ (4 * T(π) * F(frq)^2)
end
