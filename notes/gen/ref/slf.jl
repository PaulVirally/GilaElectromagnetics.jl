setprecision(BigFloat, 220)
include("../series/mom.jl")

# --- reduced 1D form for the self panel -------------------------------------
# I(k) = int_F int_F exp(i k r)/r dA dA', F = [0,p] x [0,q]
# region integrand: exp(i k p sec t) * [(q cos t - p sin t)/a^2 + 2 cos t sin t/a^3]
function glSeg(n, a, b)
    x, w = glGet(n)
    glMap(x, w, a, b)
end

function regInt(p, q, k, n)
    T = typeof(p); F = Complex{T}
    a = im * F(k)
    th0 = atan(q / p)
    nds, wts = glSeg(n, zero(T), th0)
    acc = zero(F)
    for i in eachindex(nds)
        t = nds[i]; c = cos(t); s = sin(t)
        acc += wts[i] * exp(a * p / c) * ((q * c - p * s) / a^2 + 2 * c * s / a^3)
    end
    return 4 * acc
end

function slfRed(p, q, k; n = 240)
    T = typeof(p); F = Complex{T}
    a = im * F(k)
    ele = 4 * (-p * q * T(π) / (2a) - (p + q) / a^2 - 1 / a^3)
    return ele + regInt(p, q, k, n) + regInt(q, p, k, n)
end

# reference: direct 4D
function slfRef(p, q, k; ordN = 44, ordX = 44, ordE = 44)
    T = typeof(p)
    f = k / (2 * T(π))
    A = ((zero(p), p), (zero(q), q), (zero(p), zero(p)))
    ker = pairKer(A, A, f; ordN = ordN, ordX = ordX, ordE = ordE)
    return ker * (Complex{T}(k)^2 / T(π))
end

rel(a, b) = abs(a - b) / abs(b)

for (p, q) in [(big(1)/32, big(1)/32), (big(1)/4, big(1)/4), (big(1)/32, big(1)/512)]
    for k in [2big(π), 2big(π) * (1 + im/10)]
        r1 = slfRed(p, q, k; n = 160)
        r2 = slfRed(p, q, k; n = 240)
        ref = slfRef(p, q, k)
        println("p=", Float64(p), " q=", Float64(q), " k=", ComplexF64(k))
        println("  red(160) = ", ComplexF64(r1))
        println("  red(240) = ", ComplexF64(r2))
        println("  ref(4D)  = ", ComplexF64(ref))
        println("  self-conv rel = ", Float64(rel(r1, r2)))
        println("  vs ref  rel = ", Float64(rel(r2, ref)))
    end
end
