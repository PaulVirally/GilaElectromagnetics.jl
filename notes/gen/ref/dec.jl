setprecision(BigFloat, 220)
include("../series/mom.jl")
using SpecialFunctions

glSeg(n, a, b) = glMap(glGet(n)..., a, b)

# F_m(z, Y) = int_1^Y exp(i z y) y^{-m} (y^2-1)^{-1/2} dy, done in theta: y = sec t
function Fm(m, z, Y; n = 400)
    T = real(typeof(z))
    th0 = acos(one(T) / Y)
    nds, wts = glSeg(n, zero(T), th0)
    acc = zero(z)
    for i in eachindex(nds)
        c = cos(nds[i])
        acc += wts[i] * exp(im * z / c) * c^(m - 1)
    end
    acc
end
# P_m(z, Y) = int_1^Y exp(i z y) y^{-m} dy
function Pm(m, z, Y; n = 400)
    T = real(typeof(z))
    nds, wts = glSeg(n, one(T), Y)
    sum(wts[i] * exp(im * z * nds[i]) * nds[i]^(-m) for i in eachindex(nds))
end
# incomplete-Hankel-type: int_0^T exp(i z cosh t) cosh(nu t) dt
function ihk(nu, z, T0; n = 400)
    T = real(typeof(z))
    nds, wts = glSeg(n, zero(T), T0)
    sum(wts[i] * exp(im * z * cosh(nds[i])) * cosh(nu * nds[i]) for i in eachindex(nds))
end

# E_1 by series (complex, BigFloat)
function e1ser(x)
    T = real(typeof(x))
    γ = T(big"0.57721566490153286060651209008240243104215933593992359880576723488486772677")
    s = zero(x); t = one(x)
    for n in 1:400
        t *= -x / n
        s += -t / n
        abs(t / n) < eps(T) * abs(s) / 8 && n > 5 && break
    end
    return -γ - log(x) + s
end
Em(m, x) = m == 1 ? e1ser(x) : (exp(-x) - x * Em(m - 1, x)) / (m - 1)

rel(a, b) = abs(a - b) / max(abs(b), eps(BigFloat))

function regInt(p, q, k, n)
    T = typeof(p); F = Complex{T}; a = im * F(k)
    th0 = atan(q / p); nds, wts = glSeg(n, zero(T), th0)
    acc = zero(F)
    for i in eachindex(nds)
        t = nds[i]; c = cos(t); s = sin(t)
        acc += wts[i] * exp(a * p / c) * ((q * c - p * s) / a^2 + 2 * c * s / a^3)
    end
    4 * acc
end

for (p, q) in [(big(1)/32, big(1)/32), (big(1)/4, big(1)/4), (big(1)/32, big(1)/512)]
    for k in [Complex{BigFloat}(2big(π)), 2big(π) * (1 + im/10)]
        D = sqrt(p^2 + q^2); a = im * k; z = k * p; Y = D / p; T0 = acosh(Y)
        asm = 4 * ((q / a^2) * Fm(2, z, Y) - (p / a^2) * Pm(2, z, Y) + (2 / a^3) * Pm(3, z, Y))
        println("p=", Float64(p), " q=", Float64(q), " k=", ComplexF64(k))
        println("  regInt vs F2/P2/P3 assembly : ", Float64(rel(asm, regInt(p, q, k, 400))))
        # F2 = e^{izY} sqrt(Y^2-1)/Y + i z (F1 - F_{-1})
        idn = exp(im * z * Y) * sqrt(Y^2 - 1) / Y + im * z * (Fm(1, z, Y) - Fm(-1, z, Y))
        println("  F2 recursion identity       : ", Float64(rel(idn, Fm(2, z, Y))))
        # F_{-1} = int_0^T e^{iz cosh t} cosh t dt ; F_0 = int_0^T e^{iz cosh t} dt
        println("  F_-1 = ihk(1)               : ", Float64(rel(ihk(1, z, T0), Fm(-1, z, Y))))
        println("  F_0  = ihk(0)               : ", Float64(rel(ihk(0, z, T0), Fm(0, z, Y))))
        # P_m via exponential integral E_m
        for m in 2:3
            cf = Em(m, -im * z) - Y^(1 - m) * Em(m, -im * z * Y)
            println("  P_$m = E_$m(-iz) - Y^(1-$m)E_$m(-izY) : ", Float64(rel(cf, Pm(m, z, Y))))
        end
    end
end

# complete limit check: int_0^inf e^{i z cosh t} dt = (i pi/2) H_0^(1)(z), Im z > 0
for z in [1.0 + 0.3im, 4.0 + 1.0im, 0.3 + 0.2im]
    v = quadgkval = let
        # integrate to large T (decays like exp(-Im z cosh t))
        setprecision(BigFloat, 220)
        zz = Complex{BigFloat}(z); Tm = BigFloat(30)
        ihk(0, zz, Tm; n = 3000)
    end
    h = (im * big(π) / 2) * besselh(0, 1, z)
    println("z=", z, "  int_0^inf e^{izcosh t}dt = ", ComplexF64(v), "  (i pi/2)H0^(1) = ", ComplexF64(h), "  rel=", Float64(abs(v - h) / abs(h)))
end
