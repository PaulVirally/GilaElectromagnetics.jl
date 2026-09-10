# Independent references for the three panel-pair integrals of 1/(4 pi |r-r'|).
#
# Reduction: for two axis-aligned panels that are translates along an axis, the
# 4D integral collapses onto the difference variable with a triangular density
# (box-box convolution): for A=[0,a], B=[c,c+a] the density of u=x'-x is
# a-|u-c| on [c-a,c+a].
#
# Slf: A=B=[0,la]x[0,lb]        -> 4 * int_0^la int_0^lb (la-u)(lb-v)/r
# Flt: A=[0,la]x[0,lb], B=[0,la]x[lb,2lb]
#                               -> 2 * int_0^la int_0^2lb (la-u)(lb-|v-lb|)/r
# Crn: A={x in[0,la], y in[0,lb], z=0}, B={x in[0,la], y=0, z in[0,lc]}
#                               -> 2 * int_0^la (la-u) int_0^lb int_0^lc 1/r
#
# 2D moments M_ij(p,q) = int_0^p int_0^q u^i v^j / sqrt(u^2+v^2) du dv have
# exact closed forms (polar coordinates, analytic radial integral):
#   M00 = p*asinh(q/p) + q*asinh(p/q)
#   M10 = (p^2/2)asinh(q/p) + q(h-q)/2,  M01 = (q^2/2)asinh(p/q) + p(h-p)/2
#   M11 = (h^3-p^3-q^3)/3,  h = hypot(p,q)
# All are used only in the reference, evaluated in BigFloat.

using HCubature, LinearAlgebra

hyp(p, q) = sqrt(p^2 + q^2)
M00(p, q) = p * asinh(q / p) + q * asinh(p / q)
# M10 = (p^2/2)*asinh(q/p) + q*(h-q)/2 ; h-q written cancellation-free
M10(p, q) = (p^2 / 2) * asinh(q / p) + q * (p^2 / (hyp(p, q) + q)) / 2
M01(p, q) = (q^2 / 2) * asinh(p / q) + p * (q^2 / (hyp(p, q) + p)) / 2
M11(p, q) = (hyp(p, q)^3 - p^3 - q^3) / 3

# 4 * int_0^la int_0^lb (la-u)(lb-v)/r
refSlf(la, lb) = (1 / (4 * oftype(float(la), π))) * 4 *
    (la * lb * M00(la, lb) - lb * M10(la, lb) - la * M01(la, lb) + M11(la, lb))

function refFlt(la, lb)
    # region v in [0,lb], weight (la-u)*v
    a1 = la * M01(la, lb) - M11(la, lb)
    # region v in [lb,2lb], weight (la-u)*(2lb-v), by inclusion-exclusion
    g(q) = 2 * lb * la * M00(la, q) - 2 * lb * M10(la, q) - la * M01(la, q) + M11(la, q)
    a2 = g(2 * lb) - g(lb)
    return (1 / (4 * oftype(float(la), π))) * 2 * (a1 + a2)
end

# int_0^q int_0^s dy dz / sqrt(u^2+y^2+z^2)  (closed form, u>0)
function pot2(u, q, s)
    r = sqrt(u^2 + q^2 + s^2)
    return q * asinh(s / sqrt(u^2 + q^2)) + s * asinh(q / sqrt(u^2 + s^2)) -
        u * atan(q * s / (u * r))
end

# 2 * int_0^la (la-u) * pot2(u,lb,lc) du, adaptive
function refCrn(la, lb, lc; rtol = 1e-30)
    T = typeof(la)
    fi(x) = (la - x) * pot2(x, lb, lc)
    v, _ = hquadrature(fi, zero(T), T(la); rtol = rtol, atol = zero(T), maxevals = 10^7)
    return (1 / (4 * T(π))) * 2 * v
end

# fully independent brute-force checks -------------------------------------
# Slf by 2D adaptive integration in polar coordinates over the rectangle
function refSlfPolar(la, lb; rtol = 1e-12)
    th1 = atan(lb, la)
    # wedge 1: theta in [0,th1], r in [0, la/cos]
    f1(th) = begin
        R = la / cos(th)
        c = cos(th); s = sin(th)
        la * lb * R - (la * s + lb * c) * R^2 / 2 + c * s * R^3 / 3
    end
    f2(th) = begin
        R = lb / sin(th)
        c = cos(th); s = sin(th)
        la * lb * R - (la * s + lb * c) * R^2 / 2 + c * s * R^3 / 3
    end
    v1, _ = hquadrature(f1, 0.0, th1; rtol = rtol, maxevals = 10^7)
    v2, _ = hquadrature(f2, th1, π / 2; rtol = rtol, maxevals = 10^7)
    return (1 / (4 * π)) * 4 * (v1 + v2)
end

# Slf by 4D Duffy-type HCubature on the raw 4D integral (crude, low accuracy)
function refSlf4D(la, lb; rtol = 1e-6)
    f(x) = begin
        u = x[1]; v = x[2]
        # difference-variable form, both quadrants folded
        (la - u) * (lb - v) / sqrt(u^2 + v^2)
    end
    # singularity at origin: substitute u = la*s^2? use polar-free tanh-sinh-ish
    # instead integrate with substitution u = la*a, v = lb*b and let hcubature cope
    g(x) = begin
        a = x[1]; b = x[2]
        # map (a,b) in [0,1]^2 -> u=la*a^2, v=lb*b^2, jac = 4*la*lb*a*b
        u = la * a^2; v = lb * b^2
        4 * la * lb * a * b * (la - u) * (lb - v) / sqrt(u^2 + v^2)
    end
    v, e = hcubature(g, [0.0, 0.0], [1.0, 1.0]; rtol = rtol, maxevals = 10^7)
    return (1 / (4 * π)) * 4 * v, e
end

# Crn by raw 3D HCubature with u=la*a^2 style grading
function refCrn3D(la, lb, lc; rtol = 1e-8)
    g(x) = begin
        a, b, c = x[1], x[2], x[3]
        u = la * a; y = lb * b; z = lc * c
        la * lb * lc * (la - u) / sqrt(u^2 + y^2 + z^2)
    end
    v, e = hcubature(g, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]; rtol = rtol, maxevals = 10^8)
    return (1 / (4 * π)) * 2 * v, e
end

# Flt by 2D adaptive in polar over [0,la]x[0,2lb] with the kink at v=lb
function refFltPolar(la, lb; rtol = 1e-12)
    # integrand in cartesian, w(u,v) = (la-u)*(lb-|v-lb|)
    w(u, v) = (la - u) * (lb - abs(v - lb))
    # split the rectangle [0,la]x[0,2lb] into angular wedges at the corners
    # (la,0) direction theta=0 ... use two sub-rectangles and inclusion of kink:
    # rect1 [0,la]x[0,lb], rect2 [0,la]x[lb,2lb]; rect2 has no singularity ->
    # plain hcubature.  rect1 in polar.
    th1 = atan(lb, la)
    f1(th) = begin
        R = la / cos(th); c = cos(th); s = sin(th)
        # w = (la - r c)*(r s) since v<=lb in rect1
        la * s * R^2 / 2 - c * s * R^3 / 3
    end
    f2(th) = begin
        R = lb / sin(th); c = cos(th); s = sin(th)
        la * s * R^2 / 2 - c * s * R^3 / 3
    end
    v1, _ = hquadrature(f1, 0.0, th1; rtol = rtol, maxevals = 10^7)
    v2, _ = hquadrature(f2, th1, π / 2; rtol = rtol, maxevals = 10^7)
    v3, _ = hcubature(x -> w(x[1], x[2]) / sqrt(x[1]^2 + x[2]^2),
        [0.0, lb], [la, 2 * lb]; rtol = rtol, maxevals = 10^7)
    return (1 / (4 * π)) * 2 * (v1 + v2 + v3)
end
