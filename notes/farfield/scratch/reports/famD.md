# Family (d): subdivision of the difference box — mathematics, bound, and measurements

Author: famD. Work dir `notes/farfield/scratch/work/famD/`.
Code: `famD.jl` (moments, splits, BigFloat base expansion, a priori bound),
`refTen.jl` + `refRun.jl` (220-bit references), `chkDer.jl`, `chkSgn.jl`,
`geoStudy.jl` (rho tables), `convStudy.jl` (degrees, work, cancellation), `gradStudy.jl` (graded splits).
Tables in `work/famD/tab/`.

## 0. What is computed, and the two verifications that license everything below

Target (sign convention confirmed, see 0.2):

    T_ab(R) = (1/V_t) int_D w(delta) [(d_a d_b + delta_ab k^2) g](R + delta) d delta,
    g(r) = e^{ikr}/(4 pi f^2 r),  k = 2 pi f,  D = prod_i [-s_i, s_i],  w = prod_i (s_i - |delta_i|).

Subdivision: cut each axis, D = union_j D_j, D_j a box with centre c_j and half-widths h^{(j)},
X_j = R + c_j, H_j = |h^{(j)}| (half-diagonal), rho_j = H_j/|X_j|. Per sub-box, plain Cartesian
Taylor about X_j graded by total degree:

    T_j = (1/V_t) sum_alpha (d^{alpha+e_a+e_b} + delta_ab k^2 d^alpha) g (X_j) mu^{(j)}_alpha / alpha!,
    mu^{(j)}_alpha = int_{D_j} w(delta) (delta - c_j)^alpha d delta = prod_i M_i(j_i, alpha_i),
    M_i(j, n) = int_{a}^{b} (s_i - |t|) (t - c)^n dt  with c = (a+b)/2, [a,b] the i-th sub-interval,
              = (s_i - c) [(v^{n+1}-u^{n+1})/(n+1)] - [(v^{n+2}-u^{n+2})/(n+2)]   for a >= 0,
              = (s_i + c) [(v^{n+1}-u^{n+1})/(n+1)] + [(v^{n+2}-u^{n+2})/(n+2)]   for b <= 0,
              u = a-c, v = b-c; intervals straddling 0 are split at 0.
All M_i exact in Rational{BigInt} (`wgtMom` in famD.jl); the sums are formed in BigFloat, 256 bits.

FACT (cost consequence, stated up front): for the undivided box, mu_alpha vanishes for every odd
alpha_i, so only 1/8 of the multi-indices survive. A sub-interval kills its odd moments only when it
is symmetric about 0. For a uniform split into m > 1 pieces, no sub-interval except the (odd m)
central one is symmetric, so subdivision loses the odd-moment vanishing: the term count per sub-box
at fixed degree p goes up by a factor of about 8, on top of the m^3 sub-boxes.

### 0.1 Derivative table (verified)

`kerTay(X, k, N)` returns U[alpha] = d^alpha u(X)/alpha!, u = e^{ikr}/r, by a graded recursion on
r^2 d_i u = -x_i u + i k x_i e^{ikr} differentiated alpha times (two coupled tables u and e^{ikr}).
Checked in 512-bit BigFloat against central finite differences (h = 1e-18) for
alpha in {(1,0,0)...(3,1,1)} and against the closed form of d_a d_b u, at four points
(`chkDer.jl`, output `out_chkDer.txt`):

    X = (2,0,0)/32,      f = 1        worst relative error 1.28e-33
    X = (3,1,1)/32,      f = 1+0.1i   worst relative error 6.53e-34
    X = (1/4,-1/8,1/2),  f = 1        worst relative error 1.04e-35
    X = (2/512,1/32,0),  f = 1+0.1i   worst relative error 5.99e-33

(target 1e-30). The BigFloat is real: U[4,2,2] at 128 vs 256 bits differs in the 37th digit.

### 0.2 Sign convention (pinned, no flip)

At D = (2,1,1), s = (1/32)^3, f = 1, the m = 4, p = 32 subdivision expansion reproduces the
220-bit face-pair reference (36 `pairKer` values through `srfSum!`) with

    max|T - G_ref| / max|G_ref| = 0.0 (Float64 rounding),   max|T + G_ref| / max|G_ref| = 2.0,

entrywise ratio T/G_ref = 1 to 1e-16 on all nine entries (`chkSgn.jl`, `out_chkSgn.txt`).
So the convention above is Gila's with coefficient +1; no sign flip.

## 1. The a priori remainder bound

Fix a sub-box, X = X_j, H = H_j, rho = H/|X| < 1. Let delta' in D_j - c_j, |delta'| <= H, and
phi(t) = F_ab(X + t delta') with F_ab = (d_a d_b + delta_ab k^2) g.

**Analyticity radius.** g is singular exactly where y . y = 0 (complex cone). With y = X + t delta',
y.y = |X|^2 + 2 t (X.delta') + t^2 |delta'|^2 has roots t_pm with t_+ t_- = |X|^2/|delta'|^2 and,
by Cauchy-Schwarz, a non-positive discriminant, hence |t_pm| = |X|/|delta'| exactly. So phi is
analytic in |t| < |X|/|delta'|, and uniformly in delta' in |t| < |X|/H = 1/rho.

**Bounds on the circle.** Put e in (H, |X|), beta = e/|X| in (rho,1), and take the Cauchy circle
|t| = e/|delta'| (radius e in length units). Then, on that circle,
  |r| <= U := |X|(1+beta)   (from |y.y| <= (sum(|X_i| + sigma|delta'_i|))^2 <= (|X|+e)^2),
  |r| >= L := |X|(1-beta)   (from y.y = |delta'|^2 (t-t_+)(t-t_-) and |t - t_pm| >= |X|/|delta'| - e/|delta'|),
  |r - |X|| <= e U/L        (from r' = (X.delta' + t|delta'|^2)/r, |r'| <= |delta'| U/L, path length e/|delta'|),
so |e^{ikr}| <= exp(-Im(k)|X| + |k| e U/L).
With d_a d_b u = e^{ikr}[ delta_ab (ik/r - 1/r^2) + (y_a y_b/r^2)(-k^2 - 3ik/r + 3/r^2) ]/r and
|y_a y_b / r^2| <= (U/L)^2,

    |F_ab| <= M(beta) := exp(-Im(k)|X| + |k| beta |X| U/L) / (4 pi |f|^2 L)
              * [ |k|^2 (1 + (U/L)^2) + (|k|/L)(1 + 3 (U/L)^2) + (1/L^2)(1 + 3 (U/L)^2) ].

**Tail.** Cauchy gives |degree-n Taylor term at delta'| <= M(beta) (|delta'|/e)^n <= M(beta) q^n,
q := rho/beta < 1, so with W_j = int_{D_j} w = mu^{(j)}_0 (exact rational),

    | T_j - T_j^{(p)} |  <=  (W_j / V_t) * M(beta) * q^{p+1} / (1 - q),      q = rho_j / beta,

minimized over beta in (rho_j, 1). This is the promised C rho^{p+1}/(1-rho) form: letting beta -> 1
gives literally C rho_j^{p+1}/(1-rho_j) with C = (W_j/V_t) M(1^-), but M blows up as beta -> 1, so
the useful statement is the minimum over beta, which is what `bndPre`/`bndDeg` compute (800-point
beta grid; the optimum is interior). The required degree follows in closed form per beta:
p+1 >= log(tol (1-q)/((W_j/V_t) M)) / log q.

Budget rule used throughout: to certify 1e-13 of max_ab|T_ab| for the total, each sub-box is given
tol_j = 1e-13 * max_ab|T_ab| / nBox.

## 2. Reference set (both references are independent of each other and of the expansion)

Two 220-bit/256-bit references were built for every case, and they agree with each other:

- **Face-pair reference** (`refTen.jl`, `refRun.jl`): 36 `pairKer` integrals at ordN = ordX = ordE = 44,
  divided by V_t, assembled with the `srfSum!` signs. 48 cubic cases (12 offsets x 2 scales x 2
  frequencies), 19-140 s each. It stalls on the slender cell (one offset took > 15 min), so
  slender cases were done by the second reference.
- **Volume Gauss-Legendre reference** (`refQuad.jl`): BigFloat Legendre nodes by Newton, tensor rule
  on every sub-box of a cut that includes the weight kinks at delta_i = 0, integrand
  w(delta) (d_a d_b + delta_ab k^2) g(R+delta) in closed form.

Cross-check of the two (`chkQuad.jl`, `chkSld.jl`), max|Q - G_ref|/max|G_ref|:

| case | n = 24 | n = 40 | n = 56 |
|---|---|---|---|
| cube 1/32 (2,0,0) f=1 | 1.41e-32 | 5.56e-54 | 1.36e-64 |
| cube 1/32 (2,1,1) f=1 | 1.48e-33 | 2.08e-55 | 1.93e-63 |
| cube 1/32 (3,1,1) f=1+0.1i | 3.65e-45 | 1.10e-63 | 1.10e-63 |
| cube 1/32 (8,0,0) f=1 | 1.19e-63 | 1.19e-63 | 1.19e-63 |
| cube 1/4 (2,2,2) f=1+0.1i | 7.71e-47 | 1.11e-64 | 1.11e-64 |
| cube 1/4 (4,0,0) f=1 | 8.11e-53 | 2.94e-65 | 2.94e-65 |
| slender (2,0,0) f=1 | - | 1.87e-44 (n=30) | 1.43e-62 (n=42) |

The floor at 1e-63 to 1e-65 is the face-pair reference's own accuracy (the BRIEF quotes ordN 44 vs
60 agreeing to 1.5e-63). So the volume form, its normalization against `srfScl`/`srfSum!`, and the
sign are confirmed to 1e-63, not merely to Float64. For the slender cases the GL reference was used
with transverse cuts `ctrCut(s_i, 6, 3)` and orders 30 and 42; the difference between the two orders
(the reported floor) is 1.1e-31 for the worst case (0,0,2), 1.1e-38 (0,0,8), 1.5e-63 (0,0,32),
2.4e-66 (4,0,0), 1.2e-51 (2,2,0), 2.7e-47 (1,1,24) — all far below the 1e-13 being tested.

**Expansion vs reference.** Across all 414 (shape, scale, offset, frequency, split) rows of
`tab/conv.tsv`, every row that reached its truncation cap converged to the reference; the largest
final error over the 364 converged rows is **1.76e-15** (relative to the largest tensor entry), and
typical final errors are 1e-18 to 1e-30. The subdivision expansion is therefore correct at every
offset, both cell shapes, both scales, both frequencies.

## 3. (i) Per-sub-box ratios rho_j

`tab/geo.tsv`. rho_j = H_j/|R + c_j|; the table gives the worst and the best sub-box. rho is scale
free for cubes, so the 1/32 and 1/4 cubes have identical rho. "u m" is the uniform m x m x m split;
"a m1xm2xm3" is the greedy anisotropic split (`aniSplitN`: cut the axis with the largest current
half-edge, within a box budget).

**Cubes, rho_max (rho_min in parentheses):**

| offset | m=1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| (2,0,0) | 0.8660 | 0.5222 (0.333) | 0.4330 (0.204) | 0.3333 (0.147) | 0.2425 (0.094) | 0.1901 (0.069) |
| (2,1,0) | 0.7746 | 0.5222 (0.293) | 0.4201 (0.180) | 0.3333 (0.129) | 0.2425 (0.083) | 0.1901 (0.061) |
| (2,1,1) | 0.7071 | 0.5222 (0.264) | 0.4083 (0.162) | 0.3333 (0.117) | 0.2425 (0.075) | 0.1901 (0.055) |
| (2,2,0) | 0.6124 | 0.3974 | 0.3062 | 0.2425 | 0.1741 | 0.1357 |
| (2,2,2) | 0.5000 | 0.3333 | 0.2500 | 0.2000 | 0.1429 | 0.1111 |
| (3,0,0) | 0.5774 | 0.3333 | 0.2474 | 0.1901 | 0.1325 | 0.1015 |
| (3,1,1) | 0.5222 | 0.3333 | 0.2425 | 0.1901 | 0.1325 | 0.1015 |
| (3,3,3) | 0.3333 | 0.2000 | 0.1429 | 0.1111 | 0.0769 | 0.0588 |
| (4,0,0) | 0.4330 | 0.2425 | 0.1732 | 0.1325 | 0.0909 | 0.0692 |
| (4,4,4) | 0.2500 | 0.1429 | 0.1000 | 0.0769 | 0.0526 | 0.0400 |
| (6,0,0) | 0.2887 | 0.1562 | 0.1082 | 0.0823 | 0.0558 | 0.0422 |
| (8,0,0) | 0.2165 | 0.1150 | 0.0787 | 0.0597 | 0.0403 | 0.0304 |

For a cube and an axis offset R = n s e_1 the worst sub-box is the one hugging the origin side, at
c = (-s + s/m, +-s/m, +-s/m), so

    rho_max(m, n) = (sqrt(3)/m) / sqrt((n - 1 + 1/m)^2 + 2/m^2)  ~  sqrt(3)/(m(n-1) + 1).

Check at n = 2: sqrt(3)/(m+1) = 0.577, 0.433, 0.346, 0.247, 0.192 for m = 2,3,4,6,8 against the
measured 0.522, 0.433, 0.333, 0.243, 0.190. **This is the central fact of family (d):** because the
worst sub-box moves towards the origin as it shrinks, rho falls only like 1/(m(n-1)+1), i.e. by a
factor m(n-1)/n relative to the undivided box — a factor m/2 at two cells, not m.

Anisotropic splits equal the uniform ones for cubes at every budget tested (8, 27, 64, 216, 512);
`aniSplitN` returns (m,m,m) exactly. They differ only for the slender cell.

**Slender cell (1/32, 1/32, 1/512), rho_max:**

| offset | m=1 | u2 | u3 | u4 | u6 | u8 | a4x2x1 (8) | a8x8x1 (64) | a23x22x1 (506) |
|---|---|---|---|---|---|---|---|---|---|
| (2,0,0) | 0.7078 | 0.4476 | 0.3539 | 0.2776 | 0.2002 | 0.1563 | 0.4178 | 0.1656 | 0.0849 |
| (0,0,2) | 11.325 | 0.9923 | **5.6624** | 0.9774 | 0.9563 | 0.9301 | 0.9820 | 0.8660 | 0.6667 |
| (0,0,8) | 2.8312 | 0.8343 | **1.0295** | 0.6158 | 0.4661 | 0.3693 | 0.7500 | 0.3536 | 0.1766 |
| (0,0,32) | 0.7078 | 0.3383 | 0.2409 | 0.1783 | 0.1202 | 0.0906 | 0.2709 | 0.0934 | 0.0443 |
| (4,0,0) | 0.3539 | 0.2002 | 0.1416 | 0.1086 | 0.0744 | 0.0566 | 0.1711 | 0.0600 | 0.0291 |
| (2,2,0) | 0.5005 | 0.3336 | 0.2502 | 0.2002 | 0.1430 | 0.1112 | 0.2881 | 0.1178 | 0.0600 |
| (1,1,24) | 0.6867 | 0.4342 | 0.3079 | 0.2366 | 0.1608 | 0.1215 | 0.3514 | 0.1241 | 0.0591 |

Two things to read off. (1) **Odd m is a trap for offsets with a zero component:** an odd split leaves
a sub-box centred at c_perp = 0, whose half-width is s_perp/m while |X| is only the (small) offset,
so rho explodes — 5.66 at u3 for (0,0,2) against 0.99 at u2, and 1.03 at u3 for (0,0,8) against 0.83
at u2. Any automatic split selection must use even m on every axis whose offset component is zero.
(2) For the slender cell the anisotropic split is strictly better per sub-box: a8x8x1 (64 boxes)
beats u8 (512 boxes) at every slender offset, e.g. 0.354 vs 0.369 at (0,0,8) and 0.0934 vs 0.0906
at (0,0,32) with 8x fewer boxes.

## 4. (ii)+(iii) Truncation degree and work

**Work model.** Per sub-box the cost is (a) the derivative table of u = e^{ikr}/r to total degree
p+2, C(p+5,3) coefficients at about 12 complex flops each in `kerTay` (each coefficient is one
graded recursion step: 2 complex mults with X_i plus 6 adds, in both the u and the e^{ikr} table),
and (b) the moment contraction, one multi-index per non-vanishing moment, about 24 complex flops for
all six independent tensor components. So

    flops(split, p) = sum_j [ 12 C(p_j+5, 3) + 24 N_j(p_j) ],
    N_j(p) = #{alpha : |alpha| <= p, alpha_i even on every axis whose sub-interval is symmetric}.

For the undivided box every axis is symmetric, so N(p) = C(floor(p/2)+3, 3) — one eighth of the
multi-indices. **Subdivision destroys that**: for m > 1 only the middle interval of an odd split is
symmetric, so N_j(p) = C(p+3, 3) for essentially every sub-box. That factor of about 8 in term count
is paid on top of the m^3 sub-boxes, and it is why the m = 1 rows below are so much cheaper whenever
they converge at all.

The tables give **pMeas** (the smallest total degree, applied uniformly to all sub-boxes, at which
max|T(p) - T_ref| / max|T_ref| <= 1e-13 against the reference) and the flops at that degree.
"-1, -" means 1e-13 was not reached within the computed cap (the cap and the error actually reached
are in `tab/conv.tsv`, columns Pmax and errAtP; e.g. cube (2,0,0) m=1 reaches only 7.3e-9 at p = 44).
Per-entry degrees pMeasEnt (relative to each individual entry rather than the largest) are at most 4
higher than pMeas anywhere in the sweep, and equal in most rows; they are the pMeasEnt column of
`tab/conv.tsv`.

#### cube s=1//32, f = 1.0 + 0.0im   (pMeas | work in complex flops)
| offset | rho(m=1) | u1 | u2 | u3 | u4 | u6 | u8 | best |
|---|---|---|---|---|---|---|---|---|
| (2,0,0) | 0.8660 | -1, - | 36, 2.78e+06 | 28, 3.51e+06 | 21, 5.11e+06 | 16, 8.47e+06 | 12, 9.77e+06 | u2 @ 2.78e+06 |
| (2,1,0) | 0.7746 | -1, - | 37, 3.00e+06 | 27, 3.18e+06 | 21, 5.11e+06 | 14, 6.04e+06 | 12, 9.77e+06 | u2 @ 3.00e+06 |
| (2,1,1) | 0.7071 | -1, - | 38, 3.23e+06 | 26, 2.87e+06 | 20, 4.49e+06 | 14, 6.04e+06 | 12, 9.77e+06 | u3 @ 2.87e+06 |
| (2,2,0) | 0.6124 | 36, 1.60e+05 | 27, 1.26e+06 | 20, 1.44e+06 | 16, 2.51e+06 | 11, 3.34e+06 | 10, 6.31e+06 | u1 @ 1.60e+05 |
| (2,2,2) | 0.5000 | 30, 9.81e+04 | 24, 9.12e+05 | 17, 9.46e+05 | 14, 1.79e+06 | 11, 3.34e+06 | 10, 6.31e+06 | u1 @ 9.81e+04 |
| (3,0,0) | 0.5774 | 30, 9.81e+04 | 23, 8.14e+05 | 16, 8.13e+05 | 13, 1.49e+06 | 10, 2.66e+06 | 10, 6.31e+06 | u1 @ 9.81e+04 |
| (3,1,1) | 0.5222 | 30, 9.81e+04 | 22, 7.22e+05 | 16, 8.13e+05 | 12, 1.22e+06 | 11, 3.34e+06 | 9, 4.94e+06 | u1 @ 9.81e+04 |
| (3,3,3) | 0.3333 | 20, 3.45e+04 | 16, 3.14e+05 | 12, 4.02e+05 | 10, 7.89e+05 | 9, 2.08e+06 | 8, 3.78e+06 | u1 @ 3.45e+04 |
| (4,0,0) | 0.4330 | 24, 5.48e+04 | 17, 3.67e+05 | 13, 4.87e+05 | 10, 7.89e+05 | 10, 2.66e+06 | 9, 4.94e+06 | u1 @ 5.48e+04 |
| (4,4,4) | 0.2500 | 16, 1.99e+04 | 13, 1.86e+05 | 10, 2.63e+05 | 10, 7.89e+05 | 8, 1.60e+06 | 7, 2.83e+06 | u1 @ 1.99e+04 |
| (6,0,0) | 0.2887 | 16, 1.99e+04 | 13, 1.86e+05 | 10, 2.63e+05 | 10, 7.89e+05 | 8, 1.60e+06 | 8, 3.78e+06 | u1 @ 1.99e+04 |
| (8,0,0) | 0.2165 | 14, 1.45e+04 | 10, 9.86e+04 | 10, 2.63e+05 | 9, 6.17e+05 | 8, 1.60e+06 | 7, 2.83e+06 | u1 @ 1.45e+04 |

#### cube s=1//4, f = 1.0 + 0.0im   (pMeas | work in complex flops)
| offset | rho(m=1) | u1 | u2 | u3 | u4 | u6 | u8 | best |
|---|---|---|---|---|---|---|---|---|
| (2,0,0) | 0.8660 | -1, - | 34, 2.37e+06 | 27, 3.18e+06 | 21, 5.11e+06 | 13, 5.02e+06 | 11, 7.91e+06 | u2 @ 2.37e+06 |
| (2,1,0) | 0.7746 | -1, - | 35, 2.57e+06 | 25, 2.58e+06 | 20, 4.49e+06 | 13, 5.02e+06 | 11, 7.91e+06 | u2 @ 2.57e+06 |
| (2,1,1) | 0.7071 | -1, - | 35, 2.57e+06 | 24, 2.32e+06 | 18, 3.40e+06 | 13, 5.02e+06 | 11, 7.91e+06 | u3 @ 2.32e+06 |
| (2,2,0) | 0.6124 | 34, 1.37e+05 | 25, 1.02e+06 | 17, 9.46e+05 | 15, 2.13e+06 | 11, 3.34e+06 | 9, 4.94e+06 | u1 @ 1.37e+05 |
| (2,2,2) | 0.5000 | 26, 6.74e+04 | 21, 6.38e+05 | 15, 6.91e+05 | 12, 1.22e+06 | 10, 2.66e+06 | 9, 4.94e+06 | u1 @ 6.74e+04 |
| (3,0,0) | 0.5774 | 30, 9.81e+04 | 21, 6.38e+05 | 15, 6.91e+05 | 12, 1.22e+06 | 10, 2.66e+06 | 9, 4.94e+06 | u1 @ 9.81e+04 |
| (3,1,1) | 0.5222 | 26, 6.74e+04 | 21, 6.38e+05 | 14, 5.84e+05 | 12, 1.22e+06 | 10, 2.66e+06 | 9, 4.94e+06 | u1 @ 6.74e+04 |
| (3,3,3) | 0.3333 | 18, 2.65e+04 | 14, 2.24e+05 | 12, 4.02e+05 | 11, 9.89e+05 | 10, 2.66e+06 | 9, 4.94e+06 | u1 @ 2.65e+04 |
| (4,0,0) | 0.4330 | 20, 3.45e+04 | 16, 3.14e+05 | 12, 4.02e+05 | 11, 9.89e+05 | 10, 2.66e+06 | 9, 4.94e+06 | u1 @ 3.45e+04 |
| (4,4,4) | 0.2500 | 18, 2.65e+04 | 14, 2.24e+05 | 12, 4.02e+05 | 11, 9.89e+05 | 10, 2.66e+06 | 9, 4.94e+06 | u1 @ 2.65e+04 |
| (6,0,0) | 0.2887 | 16, 1.99e+04 | 14, 2.24e+05 | 12, 4.02e+05 | 11, 9.89e+05 | 10, 2.66e+06 | 8, 3.78e+06 | u1 @ 1.99e+04 |
| (8,0,0) | 0.2165 | 16, 1.99e+04 | 14, 2.24e+05 | 12, 4.02e+05 | 11, 9.89e+05 | 10, 2.66e+06 | 8, 3.78e+06 | u1 @ 1.99e+04 |

#### slender s=1//32, f = 1.0 + 0.0im   (pMeas | work in complex flops)
| offset | rho(m=1) | u1 | u2 | u3 | u4 | u6 | u8 | a4x2x1 | a8x8x1 | a23x22x1 | best |
|---|---|---|---|---|---|---|---|---|---|---|---|
| (2,0,0) | 0.7078 | -1, - | 34, 2.37e+06 | 25, 2.58e+06 | 20, 4.49e+06 | 14, 6.04e+06 | 12, 9.77e+06 | 29, 1.07e+06 | 11, 7.42e+05 | 8, 2.87e+06 | a8x8x1 @ 7.42e+05 |
| (0,0,2) | 11.3250 | -1, - | -1, - | -1, - | -1, - | -1, - | -1, - | -1, - | -1, - | -1, - | - @ - |
| (0,0,8) | 2.8312 | -1, - | -1, - | -1, - | -1, - | -1, - | -1, - | -1, - | 22, 4.12e+06 | 11, 5.82e+06 | a8x8x1 @ 4.12e+06 |
| (0,0,32) | 0.7078 | -1, - | 26, 1.13e+06 | 18, 1.10e+06 | 14, 1.79e+06 | 11, 3.34e+06 | 10, 6.31e+06 | 20, 4.02e+05 | 9, 4.72e+05 | 7, 2.17e+06 | a4x2x1 @ 4.02e+05 |
| (4,0,0) | 0.3539 | 20, 3.45e+04 | 17, 3.67e+05 | 12, 4.02e+05 | 11, 9.89e+05 | 9, 2.08e+06 | 8, 3.78e+06 | 15, 1.95e+05 | 8, 3.66e+05 | 6, 1.60e+06 | u1 @ 3.45e+04 |
| (2,2,0) | 0.5005 | 34, 1.37e+05 | 26, 1.13e+06 | 20, 1.44e+06 | 16, 2.51e+06 | 13, 5.02e+06 | 11, 7.91e+06 | 21, 4.57e+05 | 11, 7.42e+05 | 8, 2.87e+06 | u1 @ 1.37e+05 |
| (1,1,24) | 0.6867 | -1, - | 32, 2.00e+06 | 22, 1.84e+06 | 17, 2.93e+06 | 11, 3.34e+06 | 11, 7.91e+06 | 25, 7.22e+05 | 10, 5.97e+05 | 8, 2.87e+06 | a8x8x1 @ 5.97e+05 |


Complex frequency changes nothing structurally: at f = 1 + 0.1i every pMeas in the cubic tables above
moves by at most 2 (e.g. cube 1/32 (2,1,0) u2: 37 -> 38; (8,0,0) u2: 10 -> 11), and the winning split
is the same except that (2,1,0) at lambda/32 moves from u2 to u3 (3.00e6 -> 3.18e6, within the noise
of the model). The full f = 1 + 0.1i tables are the second half of `tab/conv.tsv`.

### Optimum m per band (measured, 1e-13, f = 1)

| band | shape/scale | worst offset in band | best split | flops | best split at m>1 forced | m=1 flops |
|---|---|---|---|---|---|---|
| 2 cells | cube 1/32 | (2,0,0) | **u2** | 2.78e6 | u2 | fails (7.3e-9 at p=44) |
| 2 cells | cube 1/32 | (2,1,1) | **u3** | 2.87e6 | u3 | fails |
| 2 cells | cube 1/32 | (2,2,2) | **u1** | 9.8e4 | u2 9.1e5 | 9.8e4 |
| 2 cells | cube 1/4 | (2,0,0) | **u2** | 2.37e6 | u2 | fails |
| 2 cells | slender | (2,0,0) | **a8x8x1** | 7.4e5 | a8x8x1 | fails |
| 3 cells | cube 1/32 | (3,0,0) | **u1** | 9.8e4 | u3 8.1e5 | 9.8e4 |
| 3 cells | cube 1/4 | (3,0,0) | **u1** | 9.8e4 | u3 6.9e5 | 9.8e4 |
| 4 cells | cube 1/32 | (4,0,0) | **u1** | 5.5e4 | u2 3.7e5 | 5.5e4 |
| 4 cells | cube 1/4 | (4,0,0) | **u1** | 3.4e4 | u2 3.1e5 | 3.4e4 |
| 6 cells | cube 1/32 | (6,0,0) | **u1** | 2.0e4 | u2 1.9e5 | 2.0e4 |
| 8 cells | cube 1/32 | (8,0,0) | **u1** | 1.45e4 | u2 9.9e4 | 1.45e4 |

**Subdivision is worth doing at exactly one place: the two-cell offsets whose undivided ratio exceeds
about 0.6** — (2,0,0), (2,1,0), (2,1,1) for cubes, (2,0,0), (0,0,n) and (1,1,24) for the slender cell.
Everywhere else it costs 6x to 60x more than the undivided expansion. The reason is the scaling law
of section 3: work ~ m^3 p(m)^3 with p(m) = A/ln(m(n-1)+1) x const, so
d log(work)/d log m = 3 - 3/ln(m(n-1)+1) > 0 as soon as m(n-1) > e - 1, i.e. for every m >= 2 at
n >= 3. At n = 2 the logarithm is small enough that m = 2 or 3 wins, and only barely.

### A priori bound versus measurement

Over the 364 rows where both a finite a priori degree and a measured degree exist:

- **the bound is never violated**: pMeas <= pAp in 364 of 364 rows;
- pAp / pMeas: min 1.75, median **2.29**, max 7.74. The loosest cases are the lambda/4 cube with
  rho near 0.6 ((2,2,0) u1: pAp 263 vs pMeas 34; (3,0,0) u1: 224 vs 30), where the
  exp(|k| e U/L) factor of M(beta) dominates; the tightest are small-rho sub-boxes
  (cube (6,0,0) u8: 14 vs 8).
- Since work ~ p^3, a degree factor of 2.3 is a **flop factor of about 12**: sizing a production
  scheme from this bound alone costs an order of magnitude. The bound is usable as a certificate,
  not as a term-count selector.

## 5. (iv) Cancellation

Two measures, both in 256-bit BigFloat, per tensor entry, worst entry reported.

**Between sub-boxes**, sum_j |T_j| / |sum_j T_j| (`cancMax` in `tab/conv.tsv`). Over all 414 rows the
range is **1.000 to 3.997**. It is 1.00 for axis offsets (all sub-boxes contribute with the same
sign), 1.2-1.6 for mixed offsets, and its maximum 4.00 occurs at the slender (0,0,2) with the
506-box anisotropic split — the only place where sub-boxes on opposite sides of the origin plane
carry opposite-sign contributions of comparable size. Subdivision therefore costs at most
log10(4) = 0.6 digits.

**Between degrees**, sum_j sum_d |T_j(d) - T_j(d-1)| / |sum_j T_j| (`cancDeg`): range **1.000 to
10.43** over all rows; the 10.43 is the pathological u3 split of the slender (0,0,2) (rho = 5.66, a
divergent sub-box). Excluding rows with rho_max > 1, the maximum is 7.08 (0.85 digits).

**Between individual monomials** (`cancStudy.jl`, `tab/canc.tsv`), sum_{j,alpha} |term| / |T_ab|:

| case | m=1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| cube 1/32 (2,0,0) f=1, P=36 | 3.40 | 2.13 | 1.46 | 1.25 | 1.11 | 1.06 |
| cube 1/32 (2,1,1) f=1, P=38 | 1.73 | 4.15 | 1.89 | 1.68 | 1.45 | 1.38 |
| cube 1/32 (2,0,0) f=1+0.1i | 3.37 | 2.12 | 1.46 | 1.25 | 1.10 | 1.06 |

The largest monomial cancellation anywhere in `tab/canc.tsv` is 4.15, i.e. **0.62 digits**. This
refutes, for this organization of the sum, the round-0 prediction of about 0.145 p digits of
monomial cancellation: the absolute series is dominated by its first few terms, not by the terms
near the truncation degree, so the ratio sum|term|/|sum| stays O(1) as long as the *absolute*
convergence ratio (|delta_1| + |delta_perp|)/|R| is below 1, which subdivision enforces even where
the undivided box violates it. Consequence: the expansion of family (d) can be summed in Float64;
the accuracy limit is the derivative table, not the summation.

## 6. The three specific questions

### 6.1 Does m = 8 make two cells work (rho_j < 0.25), and at what cost?

Yes, and m = 6 already does. At the worst two-cell offset (2,0,0) of a cube, rho_max is 0.3333 (m=4),
**0.2425 (m=6)**, 0.1901 (m=8): m = 6 is the first uniform split with every rho_j < 0.25, m = 8 gives
0.19. The measured degrees and costs at (2,0,0), lambda/32, f = 1:

| split | nBox | rho_max | pMeas | flops |
|---|---|---|---|---|
| u1 | 1 | 0.8660 | not reached at p=44 (err 7.3e-9) | - |
| u2 | 8 | 0.5222 | 36 | **2.78e6** |
| u3 | 27 | 0.4330 | 28 | 3.51e6 |
| u4 | 64 | 0.3333 | 21 | 5.11e6 |
| u6 | 216 | 0.2425 | 16 | 8.47e6 |
| u8 | 512 | 0.1901 | 12 | 9.77e6 |

So m = 8 buys rho = 0.19 and a degree of only 12, but costs **9.77e6 flops, 3.5x more than m = 2**.
Driving rho below 0.25 is the wrong objective: the term count per sub-box falls like p^3 = O(1/ln^3 m)
while the sub-box count grows like m^3. The cheapest two-cell subdivision is m = 2 or m = 3, at about
3e6 complex flops per offset.

### 6.2 Is a non-uniform split (finer on the origin side) better than uniform at equal cost?

No, not materially. `gradStudy.jl` sweeps, for every case, m in {1,2,3,4,6,8} against a 5 x 5 grid of
grading ratios (q_A on axes with a non-zero offset component, refining towards the origin side;
q_P on axes with a zero component, refining towards delta_i = 0, which is where the nearest point of
the sub-box to the singularity lies), scored by the a priori flop model at 1e-13. Best gain over
uniform, per case, over the whole sweep:

| case | best gain | where |
|---|---|---|
| cube 1/32 (2,0,0), (2,1,1), (2,2,2), (3,0,0), (4,0,0), (6,0,0) | 1.00-1.09 | q_A = 5/4 or 3/2 at m = 2 |
| cube 1/4 (2,0,0), (2,1,1), (3,0,0), (4,0,0) | 1.00-1.04 | q_A = 3/2 at m = 2 |
| slender (2,0,0), (0,0,2), (0,0,32), (1,1,24), (2,2,0) | 1.00-1.06 | - |
| slender (0,0,8) | **1.24-1.50** | q_A = 3, q_P = 5/4 at m = 4-8 |

Measured confirmation (`extraStudy.jl`, `tab/extra.tsv`, cube lambda/32, f = 1, graded q = 2):

| case | uniform pMeas / flops | graded q=2 pMeas / flops |
|---|---|---|
| (2,0,0) m=2 | 36 / 2.15e6 | 37 / 2.31e6 |
| (2,0,0) m=3 | 28 / 1.64e6 | 27 / 2.11e6 |
| (2,0,0) m=4 | 21 / 2.89e6 | 19 / 4.24e6 |
| (2,1,1) m=2 | 38 / 1.30e6 | 29 / 1.39e6 |
| (2,1,1) m=3 | 26 / 1.20e6 | 23 / 2.32e6 |
| (2,1,1) m=4 | 20 / 2.24e6 | 21 / 3.84e6 |

(flops here are the per-sub-box allocation of `tab/extra.tsv`, column workPerBox.) Grading does lower
the worst rho and the maximum degree — (2,1,1) m = 2 drops from p = 38 to p = 29 — but the sub-boxes
it makes small are exactly the ones that were already cheap, so the total does not improve. The
reason grading fails is geometric: on any axis whose offset component is zero, a tensor cut with a
node at delta_i = 0 gives the innermost sub-box |c_i| = h_i exactly, so its rho contribution from
that axis is 1 no matter how fine the grading. Refining only shrinks h_i and |c_i| together.
**Only refining an axis whose offset component is non-zero helps, and that is one axis out of three.**

### 6.3 The slender cell along (0,0,n), n < 23

`tab/needle.tsv` (`needle2.jl`) searches, per n, over uniform and centre-refined transverse cuts
(m in {1,2,4,...,64} per transverse axis, m3 in {1,2,4,8}, q in {1, 3/2, 2, 3, 4}) for the smallest
box count reaching a given rho_max. rho_max(m=1) = 22.65/n.

| n | rho(m=1) | first rho<1 | first rho<0.5 | first rho<0.25 |
|---|---|---|---|---|
| 1 | 22.65 | **impossible** | - | - |
| 2 | 11.325 | 4 boxes (2x2x1), rho 0.9886 | 200 (10x10x2, q=2), rho 0.406 | 1600 (20x20x4, q=3/2), rho 0.216 |
| 3 | 7.550 | 4, rho 0.9704 | 64 (8x8x1, q=2), rho 0.425 | 512 (16x16x2, q=3/2), rho 0.229 |
| 4 | 5.662 | 4, rho 0.9465 | 36 (6x6x1, q=2), rho 0.445 | 288 (12x12x2, q=3/2), rho 0.247 |
| 6 | 3.775 | 4, rho 0.8869 | 36 (6x6x1, q=3/2), rho 0.400 | 144 (12x12x1, q=3/2), rho 0.227 |
| 8 | 2.831 | 4, rho 0.8197 | 16 (4x4x1, q=2), rho 0.446 | 100 (10x10x1, q=3/2), rho 0.225 |
| 12 | 1.888 | 4, rho 0.6887 | 16 (4x4x1), rho 0.433 | 64 (8x8x1), rho 0.243 |
| 16 | 1.416 | 4, rho 0.5796 | 16 (4x4x1), rho 0.339 | 36 (6x6x1), rho 0.237 |
| 20 | 1.133 | 4, rho 0.4943 | 4 (2x2x1), rho 0.4943 | 36 (6x6x1), rho 0.192 |
| 22 | 1.030 | 4, rho 0.4591 | 4, rho 0.4591 | 36 (6x6x1), rho 0.175 |
| 23 | 0.9848 | 1 (none needed) | 4 (2x2x1), rho 0.4431 | 16 (4x4x1), rho 0.243 |

**n = 1 is a touching pair** (R_3 = s_3 and delta_3 ranges over [-s_3, s_3], so y = R + delta reaches
the singularity inside the domain): no subdivision of any kind converges, and none can. For n >= 2,
"every rho_j < 1" is reached at once by the trivial 2x2x1 split (transverse halving), but at
rho = 0.99 that is worthless — the required degree is 13/log10(1/0.9886) = 2600. The honest thresholds
are the rho < 0.5 and rho < 0.25 columns.

Measured cost at 1e-13 (`extraStudy.jl`, f = 1 and f = 1+0.1i, identical to 3 digits):

| offset | split | nBox | pMeas | flops |
|---|---|---|---|---|
| (0,0,2) | 10x10x2 graded q=2 | 200 | **25** | **2.7e7** |
| (0,0,2) | 16x16x2 uniform | 512 | not reached at p=22 (err 2.2e-5) | - |
| (0,0,8) | 8x8x1 graded q=2 | 64 | 21 | 3.1e6 |
| (0,0,8) | a8x8x1 uniform-in-each | 64 | 22 | 4.1e6 |
| (0,0,32) | a4x2x1 | 8 | 20 | 4.0e5 |

So the needle at n = 2 costs about 2.7e7 complex flops per offset, 10^5 times the "few hundred flops"
budget, and about 3e6 at n = 8. The k-series of `notes/moments` owns that needle (kD ~ 0.6 there);
subdivision reaches it but at a cost nobody would pay.

## 7. What subdivision cannot fix

1. **It cannot beat the m^3 term-count growth.** Work ~ m^3 p(m)^3 with p(m) ~ A/ln(rho_1^{-1} m
   (n-1)+...) : the sub-box count grows cubically while the degree falls only logarithmically. The
   crossover is at m(n-1) ~ e, so beyond two cells the optimum is always m = 1 and subdivision costs
   6x to 60x extra (section 4 table). Its whole domain of usefulness is the handful of offsets with
   rho_1 > 0.6, and there it buys about a factor of 2-3 over the (divergent-in-practice) undivided
   expansion, not orders of magnitude.

2. **It loses the odd-moment vanishing.** The undivided box has mu_alpha = 0 for every odd alpha_i;
   any sub-interval not symmetric about 0 has all moments non-zero. That is a fixed factor of about 8
   in the multi-index count per sub-box, paid before the first sub-box is added. The m = 1 -> m = 2
   step therefore costs 8 x 8 = 64 in term count and buys only rho 0.866 -> 0.522.

3. **rho_j falls like 1/(m(n-1)+1), not like 1/m.** The worst sub-box is always the one against the
   origin side; as it shrinks it also moves closer to the singularity. At two cells (n = 2) an m-fold
   split buys a factor m/2 in rho, so the first split buys nothing (m=2 -> factor 1) and the gain is
   entirely in the second and third. This is the reason m = 8 gives only rho = 0.19 at two cells
   rather than 0.108.

4. **Grading cannot be used on the axes that matter.** On any axis whose offset component is zero,
   the innermost sub-box has |c_i| = h_i for every tensor cut with a node at 0, so its contribution
   to rho is fixed at 1 whatever the grading. Measured gain over uniform, across the whole
   (m, q_A, q_P) sweep: 1.00-1.09 for cubes, 1.00-1.50 for the slender cell (section 6.2).

5. **It cannot help a touching pair.** The slender (0,0,1) offset (and every touching offset) has the
   singularity inside D; rho >= 1 for every sub-box that contains it, and no split removes it. The
   contact and touching-shell integrals of `notes/moments` are the only tool there.

6. **It does not bring any band inside the cost target.** The best measured cost per offset at 1e-13
   is 1.45e4 complex flops at eight cells (undivided), 9.8e4 at three cells, and 2.8e6 at two cells
   (m = 2). The task's budget is "a few hundred floating-point operations per offset". Family (d)
   makes the two-cell band **converge** where the plain centre expansion does not, but leaves it
   10^4 times over budget, and it makes every other band worse. If a cell-size expansion is to meet
   the budget it must come from a cheaper base expansion (fewer terms per degree, or a cheaper
   derivative table), not from splitting the box.

7. **The dominant cost is the derivative table, not the moments.** At p = 14, 12 C(19,3) = 11628 of
   the 14508 flops (80%) are `kerTay`; subdivision multiplies exactly that part by m^3, because each
   sub-box needs its own table at its own centre. Any scheme that shares derivative information
   between sub-boxes would change these numbers; plain subdivision shares none.

## 8. Honest limits of this measurement

- The work numbers are flop counts from the model of section 4, not measured runtimes; the constants
  12 and 24 come from reading `kerTay` and the contraction loop, not from a benchmark. Term counts
  (`wTrm`, `wDer` in `tab/conv.tsv`) are given raw so the model can be reweighted.
- Per-sub-box degree allocation with tol_j = tol/nBox gives no consistent saving over a uniform
  degree: the ratio (per-box work)/(uniform-p work) over 344 unsaturated rows is 0.40 to 1.40, median
  0.94. A smarter allocation (weight the budget by W_j, or by the measured |T_j|) was not explored.
- Everything is computed in 256-bit BigFloat by design, so the Float64 conditioning of the derivative
  recursion is untested here. The cancellation measurements of section 5 say the *summation* is safe
  in Float64 (at most 0.62 digits lost over all monomials); they say nothing about `kerTay` in
  Float64, which is item 6 of the groundwork and the business of family (g).
- Only the plain Cartesian Taylor base was subdivided, as instructed. Composing subdivision with a
  different base expansion would change the per-sub-box constants but not the rho_j geometry of
  section 3, which is base-independent and is what limits family (d).
- Anisotropic splits were chosen by the greedy rule `aniSplitN` (cut the axis with the largest
  current half-edge). It is not proven optimal; for the slender cell it beat the uniform split at
  every offset and every budget tested, and for cubes it coincides with the uniform split.
- Cell shapes tested: (1/32)^3, (1/4)^3, (1/32,1/32,1/512). Aspect ratios beyond 16 were not tested;
  the needle table (section 6.3) is the aspect-ratio-limited case and it scales with s_perp/s_3.

## 9. Files

    work/famD/famD.jl        kerTay (Taylor table of e^{ikr}/r), wgtMom/axsMom (exact sub-box moments),
                             uniCut/grdCut/ctrCut/axsCut/aniSplitN (splits), subBoxes, boxTay,
                             subTensor, boxGeo, bndPre/remBnd/bndDeg (a priori bound), trmCnt/derCnt,
                             splitCost
    work/famD/refTen.jl      220-bit face-pair reference (pairKer + srfSum! signs), cache I/O
    work/famD/refRun.jl      builds the 48 cubic references (ARGS: half 1 or 2)
    work/famD/refQuad.jl     BigFloat Gauss-Legendre volume reference (glRule, kerTen!, volRef)
    work/famD/refQuadRun.jl  builds the 14 slender references
    work/famD/chkDer.jl      derivative verification vs finite differences and closed forms
    work/famD/chkSgn.jl      sign convention pinned against the reference
    work/famD/chkQuad.jl     GL reference vs face-pair reference (cubes)
    work/famD/chkSld.jl      GL reference vs face-pair reference (slender)
    work/famD/geoStudy.jl    rho tables
    work/famD/convStudy.jl   degrees, work, cancellation vs reference (ARGS: half 1 or 2)
    work/famD/gradStudy.jl   graded vs uniform splits, a priori flop model
    work/famD/needle2.jl     slender needle thresholds
    work/famD/cancStudy.jl   monomial-level cancellation
    work/famD/extraStudy.jl  measured degrees for arbitrary cuts (needle, graded)
    work/famD/sum.py         renders the summary tables
    work/famD/tab/           geo.tsv, needle.tsv, conv.tsv (=conv_1+conv_2), grad.tsv, canc.tsv,
                             extra.tsv, report_tables.txt
    work/famD/refcache_big_*.txt   62 reference tensors, full 220/256-bit text
