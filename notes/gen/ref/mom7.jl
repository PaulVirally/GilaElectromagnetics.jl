# Coplanar self-panel moments  mom(m) = int_0^p int_0^q (p-u)(q-v)(u^2+v^2)^(m+1/2)
# Three implementations: Mathematica's raw output, the general closed form, and a
# cancellation-free regrouping.  All generic in the number type.

dfl(n) = n <= 0 ? big(1) : reduce(*, big(n):-2:1)          # n!!
cSum(j) = dfl(2j - 2) // dfl(2j - 1)                        # (2j-2)!!/(2j-1)!!
bCof(m) = dfl(2m + 1) // (dfl(2m + 2) * (2m + 3) * (2m + 4))

# --- general closed form ----------------------------------------------------
function momGen(m::Integer, p::T, q::T) where {T}
    h = sqrt(p^2 + q^2)
    s = zero(T)
    for j in 1:(m + 1)
        s += T(cSum(j)) * h^(2j - 1) * (p^(2 * (m + 1 - j)) + q^(2 * (m + 1 - j)))
    end
    ash = p * q^(2m + 4) * asinh(p / q) + q * p^(2m + 4) * asinh(q / p)
    return T(bCof(m)) * (p^2 * q^2 * s + ash) -
        (h^(2m + 5) - p^(2m + 5) - q^(2m + 5)) / T((2m + 3) * (2m + 4) * (2m + 5))
end

# --- cancellation-free regrouping ------------------------------------------
# h^n - p^n - q^n cancels to O(min^2 max^(n-2)) when p << q; write
# h^n - q^n = (h-q) * sum_i h^i q^(n-1-i) with h - q = p^2/(h+q).
function momSaf(m::Integer, p::T, q::T) where {T}
    a, b = minmax(p, q)                                     # a <= b
    h = hypot(p, q)
    n = 2m + 5
    gs = zero(T)                                            # sum_{i=0}^{n-1} h^i b^(n-1-i)
    for i in 0:(n - 1)
        gs += h^i * b^(n - 1 - i)
    end
    dif = (a^2 / (h + b)) * gs - a^n                        # = h^n - p^n - q^n
    s = zero(T)
    for j in 1:(m + 1)
        s += T(cSum(j)) * h^(2j - 1) * (a^(2 * (m + 1 - j)) + b^(2 * (m + 1 - j)))
    end
    ash = a * b^(2m + 4) * asinh(a / b) + b * a^(2m + 4) * asinh(b / a)
    return T(bCof(m)) * (a^2 * b^2 * s + ash) - dif / T((2m + 3) * (2m + 4) * (2m + 5))
end

# --- Mathematica output, transcribed verbatim -------------------------------
function momMma(m::Integer, p::T, q::T) where {T}
    h = sqrt(p^2 + q^2)
    A(x) = asinh(x)
    L(x) = log(x)
    if m == 0
        return (2 * (p^5 - p^4 * h + 3 * p^2 * q^2 * h + q^4 * (q - h)) +
            5 * p * q^4 * (2 * A(p / q) + L((-p + h) / q)) -
            5 * p^4 * q * L(p / (q + h))) / T(120)
    elseif m == 1
        return ((4 * (-8 * p^8 + 17 * p^6 * q^2 + 50 * p^4 * q^4 + 17 * p^2 * q^6 +
                8 * p^7 * h + 8 * q^7 * (-q + h))) / h +
            21 * p * q * (q^5 * (-15 * A(p / q) + 5 * L(q) - 12 * L(-p + h) + 7 * L(p + h)) +
                4 * p^5 * L((q + h) / p))) / T(6720)
    elseif m == 2
        return (-32 * p^10 + 86 * p^8 * q^2 + 238 * p^6 * q^4 + 238 * p^4 * q^6 +
            86 * p^2 * q^8 + 32 * p^9 * h + 32 * q^9 * (-q + h) +
            45 * p * q * h * (q^7 * (11 * A(p / q) - 7 * L(q) + L((p - h)^8 / (p + h))) +
                2 * p^7 * L((q + h) / p))) / (T(16128) * h)
    elseif m == 3
        return (-384 * p^12 + 1293 * p^10 * q^2 + 3623 * p^8 * q^4 + 3892 * p^6 * q^6 +
            3623 * p^4 * q^8 + 1293 * p^2 * q^10 + 384 * p^11 * h + 384 * q^11 * (-q + h) -
            1155 * p * q * h * (q^9 * L(q / (p + h)) + p^9 * L(p / (q + h)))) /
            (T(380160) * h)
    elseif m == 4
        return (-1280 * p^14 + 5249 * p^12 * q^2 + 15227 * p^10 * q^4 + 18666 * p^8 * q^6 +
            18666 * p^6 * q^8 + 15227 * p^4 * q^10 + 5249 * p^2 * q^12 + 1280 * p^13 * h +
            1280 * q^13 * (-q + h) -
            4095 * p * q * h * (q^11 * L(q / (p + h)) + p^11 * L(p / (q + h)))) /
            (T(2196480) * h)
    elseif m == 5
        return (-1024 * p^16 + 4983 * p^14 * q^2 + 15101 * p^12 * q^4 + 21118 * p^10 * q^6 +
            24048 * p^8 * q^8 + 21118 * p^6 * q^10 + 15101 * p^4 * q^12 + 4983 * p^2 * q^14 +
            1024 * p^15 * h + 1024 * q^15 * (-q + h) -
            3465 * p * q * h * (q^13 * L(q / (p + h)) + p^13 * L(p / (q + h)))) /
            (T(2795520) * h)
    elseif m == 6
        return (-215040 * p^18 + 1215675 * p^16 * q^2 + 3862505 * p^14 * q^4 +
            6103126 * p^12 * q^6 + 7910408 * p^10 * q^8 + 7910408 * p^8 * q^10 +
            6103126 * p^6 * q^12 + 3862505 * p^4 * q^14 + 1215675 * p^2 * q^16 +
            215040 * p^17 * h + 215040 * q^17 * (-q + h) -
            765765 * p * q * h * (q^15 * L(q / (p + h)) + p^15 * L(p / (q + h)))) /
            (T(877363200) * h)
    end
    error("no Mathematica form for m = $m")
end

# --- independent reference: exact radial integral, numerical angular --------
# polar over the rectangle, R(t) = p sec t on [0, atan(q/p)], q csc t after.
function momPol(m::Integer, p::T, q::T; ordN = 200) where {T}
    nds, wts = glGet(ordN)
    t0 = atan(q, p)
    N = 2m + 3
    f = function (t)
        c = cos(t); s = sin(t)
        R = t <= t0 ? p / c : q / s
        return p * q * R^N / N - (p * s + q * c) * R^(N + 1) / (N + 1) +
            c * s * R^(N + 2) / (N + 2)
    end
    acc = zero(T)
    for (lo, hi) in ((zero(T), t0), (t0, T(π) / 2))
        n, w = glMap(nds, wts, lo, hi)
        for i in eachindex(n)
            acc += w[i] * f(n[i])
        end
    end
    return acc
end
