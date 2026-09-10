PI(x) = oftype(float(real(x)), π)

# Type-generic variants: pi and log(64 * one(la)) forced to the working precision.
# copies of the three closed-form panel integrals from
# src/vacuum/glaVacOprMemInt.jl, generic in (la, lb[, lc], f), no return annotation.

function rSrfSlfG(la, lb, f)
    return (1 / (48 * PI(la) * f^2)) * (8 * la^3 + 8 * lb^3
    - 8 * la^2 * sqrt(la^2 + lb^2) - 8 * lb^2 * sqrt(la^2 + lb^2) -
    3 * la^2 * lb * (2 * log(la) + 2 * log(la + lb - sqrt(la^2 + lb^2)) +
    log(sqrt(la^2 + lb^2) - lb) - 5 * log(lb + sqrt(la^2 + lb^2)) -
    2 * log(lb - la + sqrt(la^2 + lb^2)) -
    2 * log(la + 2 * lb - sqrt(la^2 + 4 * lb^2)) +
    log(sqrt(la^2 + 4 * lb^2) - 2 * lb) +
    2 * log(la - 2 * lb + sqrt(la^2 + 4 * lb^2)) +
    log(2 * lb + sqrt(la^2 + 4 * lb^2)) +
    2 * log(2 * lb - la + sqrt(la^2 + 4 * lb^2)) -
    2 * log(la + 2 * lb + sqrt(la^2 + 4 * lb^2))) + 6 * la * lb^2 *
    (log(64 * one(la)) + 4 * log(lb) + 2 * log(sqrt(la^2 + lb^2) - la) +
    3 * log(la + sqrt(la^2 + lb^2)) - 3 * log(sqrt(la^2 + 4 * lb^2) - la) -
    3 * log(sqrt(la^4 + 5 * la^2 * lb^2 + 4 * lb^4) +
    la * (sqrt(la^2 + lb^2) - la - sqrt(la^2 + 4 * lb^2)))))
end

function rSrfEdgCrnG(la, lb, lc, f)
    return (1 / (48 * PI(la) * f^2)) * (8 * lb * lc *
    sqrt(lb^2 + lc^2) - 8 * lb * lc * sqrt(la^2 + lb^2 + lc^2) - 12 * la^3 *
    acot(la * lc / (la^2 + lb^2 - lb * sqrt(la^2 + lb^2 + lc^2))) +
    12 * la^3 * atan(la / lc) -
    12 * la * lc^2 * atan(la * lb / (lc * sqrt(la^2 + lb^2 + lc^2))) -
    12 * la * lb^2 * atan(la * lc / (lb * sqrt(la^2 + lb^2 + lc^2))) -
    16 * la^3 * atan(lb * lc / (la * sqrt(la^2 + lb^2 + lc^2))) +
    6 * lc^3 * atanh(lb / sqrt(lb^2 + lc^2)) -
    6 * lc * (la^2 + lc^2) * atanh(lb / sqrt(la^2 + lb^2 + lc^2)) -
    15 * la^2 * lc * log(la^2 + lc^2) - lc^3 * log(la^2 + lc^2) +
    2 * lc^3 * log(lc / (lb + sqrt(lb^2 + lc^2))) +
    6 * la^2 * lc * log(sqrt(la^2 + lb^2 + lc^2) - lb) +
    24 * la^2 * lc * log(sqrt(la^2 + lb^2 + lc^2) + lb) +
    2 * lc^3 * log(sqrt(la^2 + lb^2 + lc^2) + lb) +
    6 * la * lb * (-2 * la * log(la^2 + lb^2) -
    lc * log((lb^2 + lc^2) * (sqrt(la^2 + lb^2 + lc^2) - la)) +
    3 * lc * log(la + sqrt(la^2 + lb^2 + lc^2)) +
    la * log(sqrt(la^2 + lb^2 + lc^2) - lc) +
    3 * la * log(sqrt(la^2 + lb^2 + lc^2) + lc)) +
    2 * lb^3 * (
    log((sqrt(la^2 + lb^2 + lc^2) - lc) / (lc + sqrt(la^2 + lb^2 + lc^2))) +
    log(1 + (2 * lc * (lc + sqrt(lb^2 + lc^2))) / lb^2)))
end

# verbatim: first block 1/(12 pi f^2), second block 1/(64 pi) with no f^2
function rSrfEdgFltG(la, lb, f)
    return (1 / (12 * PI(la) * f^2)) * (-la^3 + 2 * lb^2 *
    (3 * lb + sqrt(la^2 + lb^2) - 2 * sqrt(la^2 + 4 * lb^2)) + la^2 *
    (2 * sqrt(la^2 + lb^2) - sqrt(la^2 + 4 * lb^2))) +
    (1 / (64 * PI(la))) * la * lb * (lb * (-62 * log(2 * one(la)) -
    5 * log(-la + sqrt(la^2 + lb^2)) +
    4 * log(8 * lb^2 * (-la + sqrt(la^2 + lb^2))) -
    33 * log(la + sqrt(la^2 + lb^2)) + 17 * log(-la + sqrt(la^2 + 4 * lb^2)) -
    24 * log(lb * (-la + sqrt(la^2 + 4 * lb^2))) +
    57 * log(la + sqrt(la^2 + 4 * lb^2))) +
    4 * la * (-8 * asinh(lb / la) + 6 * asinh(2 * lb / la) +
    6 * atanh(lb / sqrt(la^2 + lb^2)) + 12 * log(la) -
    13 * log(-lb + sqrt(la^2 + lb^2)) + log((-lb + sqrt(la^2 + lb^2)) / la) +
    log(la / (lb + sqrt(la^2 + lb^2))) - 7 * log(lb + sqrt(la^2 + lb^2)) -
    2 * log((lb + sqrt(la^2 + lb^2))/la) -
    3 * log(-(((lb + sqrt(la^2 + lb^2)) *
    (2 * lb - sqrt(la^2 + 4 * lb^2))) / (la^2))) -
    3 * log((-lb + sqrt(la^2 + lb^2)) / (-2 * lb + sqrt(la^2 + 4 * lb^2))) +
    11 * log(-2 * lb + sqrt(la^2 + 4 * lb^2)) -
    3 * log((lb + sqrt(la^2 + lb^2)) / (2 * lb + sqrt(la^2 + 4 * lb^2))) +
    log(2 * lb + sqrt(la^2 + 4 * lb^2)) +
    9 * log((2 * lb + sqrt(la^2 + 4 * lb^2)) / (lb + sqrt(la^2 + lb^2))) -
    2 * log(la^2 + 2 * lb * (lb - sqrt(la^2 + lb^2)))))
end

# fixed variant: second block also carries 1/f^2
function rSrfEdgFltFixG(la, lb, f)
    blk1 = (1 / (12 * PI(la) * f^2)) * (-la^3 + 2 * lb^2 *
    (3 * lb + sqrt(la^2 + lb^2) - 2 * sqrt(la^2 + 4 * lb^2)) + la^2 *
    (2 * sqrt(la^2 + lb^2) - sqrt(la^2 + 4 * lb^2)))
    blk2 = (1 / (64 * PI(la) * f^2)) * la * lb * (lb * (-62 * log(2 * one(la)) -
    5 * log(-la + sqrt(la^2 + lb^2)) +
    4 * log(8 * lb^2 * (-la + sqrt(la^2 + lb^2))) -
    33 * log(la + sqrt(la^2 + lb^2)) + 17 * log(-la + sqrt(la^2 + 4 * lb^2)) -
    24 * log(lb * (-la + sqrt(la^2 + 4 * lb^2))) +
    57 * log(la + sqrt(la^2 + 4 * lb^2))) +
    4 * la * (-8 * asinh(lb / la) + 6 * asinh(2 * lb / la) +
    6 * atanh(lb / sqrt(la^2 + lb^2)) + 12 * log(la) -
    13 * log(-lb + sqrt(la^2 + lb^2)) + log((-lb + sqrt(la^2 + lb^2)) / la) +
    log(la / (lb + sqrt(la^2 + lb^2))) - 7 * log(lb + sqrt(la^2 + lb^2)) -
    2 * log((lb + sqrt(la^2 + lb^2))/la) -
    3 * log(-(((lb + sqrt(la^2 + lb^2)) *
    (2 * lb - sqrt(la^2 + 4 * lb^2))) / (la^2))) -
    3 * log((-lb + sqrt(la^2 + lb^2)) / (-2 * lb + sqrt(la^2 + 4 * lb^2))) +
    11 * log(-2 * lb + sqrt(la^2 + 4 * lb^2)) -
    3 * log((lb + sqrt(la^2 + lb^2)) / (2 * lb + sqrt(la^2 + 4 * lb^2))) +
    log(2 * lb + sqrt(la^2 + 4 * lb^2)) +
    9 * log((2 * lb + sqrt(la^2 + 4 * lb^2)) / (lb + sqrt(la^2 + lb^2))) -
    2 * log(la^2 + 2 * lb * (lb - sqrt(la^2 + lb^2)))))
    return blk1 + blk2
end
