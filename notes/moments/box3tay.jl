# 3D centred Taylor series for W = int_box prod_i (cw_i + dw_i u_i) (u^2+v^2+w^2)^{m/2} du dv dw,
# the analogue of box2Tay. With p = u - u_m etc. and rho^2 = u_m^2 + v_m^2 + w_m^2,
#   r^m = (rho^2 + 2 u_m p + 2 v_m q + 2 w_m s + p^2 + q^2 + s^2)^{m/2} = sum c_{ijk} p^i q^j s^k,
# g d_p f = (m/2) f d_p g with f = g^{m/2} gives, for i >= 1 (level n = i+j+k from levels n-1, n-2),
#   rho^2 i c_{ijk} = (m-2(i-1)) u_m c_{i-1,j,k} + (m-i+2) c_{i-2,j,k}
#                     - 2 v_m i c_{i,j-1,k} - 2 w_m i c_{i,j,k-1} - i c_{i,j-2,k} - i c_{i,j,k-2},
# the q-mirror for i = 0, j >= 1, the s-mirror for i = j = 0, c_000 = rho^m. Weight moments per
# axis, w = alpha + beta p with alpha = cw + dw u_m, beta = dw, on [-h, h]:
#   mu_n = alpha 2h^{n+1}/(n+1) (n even),  beta 2h^{n+2}/(n+2) (n odd).
# Absolute convergence of the t-expansion, t = 2 u.x + |x|^2 >= 0 on the octant box, holds iff
# t(h) < rho^2, i.e. lam > 1 with lam the positive root of |h|^2 lam^2 + 2 (u_m.h) lam - rho^2 = 0;
# the attempt is gated at lam > 1/2 as in box2Tay and stopped by the level ratios.
# Returns (W, levels, condition) or nothing.
function box3Tay(m::Integer, lo, hi, cw, dw; nMax::Int = 140, tol = nothing, cndMax = 4)
    T = eltype(lo)
    tl = tol === nothing ? eps(T) / 16 : T(tol)
    h0 = [(hi[i] - lo[i]) / 2 for i in 1:3]
    c0 = [(hi[i] + lo[i]) / 2 for i in 1:3]
    scl = maximum(c0)
    (scl > 0 && all(>(0), h0)) || return nothing
    aa = sum(abs2, h0); bb = 2 * sum(c0[i] * h0[i] for i in 1:3); rr = sum(abs2, c0)
    lam = (-bb + sqrt(bb^2 + 4 * aa * rr)) / (2 * aa)
    lam > 1//2 || return nothing
    e = exponent(scl)
    h = [ldexp(x, -e) for x in h0]; cm = [ldexp(x, -e) for x in c0]
    al = [ldexp(cw[i] + dw[i] * c0[i], -e) for i in 1:3]; be = [T(dw[i]) for i in 1:3]
    rsq = sum(abs2, cm)
    mom = [T[] for _ in 1:3]                     # mom[i][n+1] = mu_n on axis i
    cf = Vector{Matrix{T}}()                     # cf[n+1][i+1, j+1] = c_{i, j, n-i-j}
    lvs = T[]; runA = zero(T); relPr = one(T); relPr2 = one(T); nUse = -1
    for n in 0:nMax
        for i in 1:3
            push!(mom[i], iseven(n) ? al[i] * 2 * h[i]^(n + 1) / T(n + 1) : be[i] * 2 * h[i]^(n + 2) / T(n + 2))
        end
        cn = zeros(T, n + 1, n + 1)
        if n == 0
            cn[1, 1] = rsq^(T(m) / 2)
        else
            for i in 0:n, j in 0:(n - i)
                k = n - i - j
                if i >= 1
                    s = T(m - 2 * (i - 1)) * cm[1] * cf[n][i, j + 1]
                    i >= 2 && (s += T(m - i + 2) * cf[n - 1][i - 1, j + 1])
                    j >= 1 && (s -= 2 * cm[2] * T(i) * cf[n][i + 1, j])
                    k >= 1 && (s -= 2 * cm[3] * T(i) * cf[n][i + 1, j + 1])
                    j >= 2 && (s -= T(i) * cf[n - 1][i + 1, j - 1])
                    k >= 2 && (s -= T(i) * cf[n - 1][i + 1, j + 1])
                    cn[i + 1, j + 1] = s / (rsq * T(i))
                elseif j >= 1
                    s = T(m - 2 * (j - 1)) * cm[2] * cf[n][1, j]
                    j >= 2 && (s += T(m - j + 2) * cf[n - 1][1, j - 1])
                    k >= 1 && (s -= 2 * cm[3] * T(j) * cf[n][1, j + 1])
                    k >= 2 && (s -= T(j) * cf[n - 1][1, j + 1])
                    cn[1, j + 1] = s / (rsq * T(j))
                else
                    s = T(m - 2 * (k - 1)) * cm[3] * cf[n][1, 1]
                    k >= 2 && (s += T(m - k + 2) * cf[n - 1][1, 1])
                    cn[1, 1] = s / (rsq * T(k))
                end
            end
        end
        push!(cf, cn)
        L = zero(T); A = zero(T)
        for i in 0:n, j in 0:(n - i)
            k = n - i - j
            c = cn[i + 1, j + 1]
            iszero(c) && continue
            t = c * mom[1][i + 1] * mom[2][j + 1] * mom[3][k + 1]
            L += t; A += abs(t)
        end
        push!(lvs, L); runA += A
        isfinite(runA) || return nothing
        iszero(runA) && return nothing
        rel = A / runA
        # THREE consecutive small levels: odd levels vanish identically for constant weights, and at
        # m = -1 (1/r harmonic) level 2 vanishes exactly whenever the centre is on a diagonal u_m = (a,a,a)
        # (the three pure second derivatives are equal and sum to zero), so two levels are not enough:
        # measured 11-13 digits lost on the [2,3]^3 and [1,2]x[0,3]x[0,3] boxes with the 2D rule.
        if n >= 4 && rel <= tl && relPr <= tl && relPr2 <= tl
            nUse = n; break
        end
        if n >= 10 && n < nMax
            r = max(rel, eps(T)^2)^(one(T) / T(n))
            (r >= 1 || log(tl) / log(r) > nMax) && return nothing
        end
        relPr2 = relPr; relPr = rel
    end
    nUse < 0 && return nothing
    W = zero(T)
    for n in nUse:-1:0
        W += lvs[n + 1]
    end
    (W > 0 && runA <= cndMax * W) || return nothing
    return (ldexp(W, e * (m + 6)), nUse, runA / W, lam)
end

# the quadratic's root as a convergence diagnostic alone (no series built)
function tayLam(lo, hi)
    h0 = [(hi[i] - lo[i]) / 2 for i in 1:3]; c0 = [(hi[i] + lo[i]) / 2 for i in 1:3]
    aa = sum(abs2, h0); bb = 2 * sum(c0[i] * h0[i] for i in 1:3); rr = sum(abs2, c0)
    return (-bb + sqrt(bb^2 + 4 * aa * rr)) / (2 * aa)
end
