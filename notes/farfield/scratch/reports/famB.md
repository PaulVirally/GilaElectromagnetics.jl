# Family (b): spherical addition theorem for the separated far field

Library: `SCRATCH/work/famB/famB.jl` (standalone, generic in the number type).
Scripts: `t1_check.jl` (identities), `t3_bessel.jl` (special-function conditioning),
`t4_conv.jl` (convergence in l, conditioning), `t5_acc.jl` (accuracy vs the 220-bit
reference, symmetries), `t6_cost.jl` (flops and timing), `ref.jl` + `refrun.jl`
(220-bit reference builder with a disk cache in `work/famB/out/refcache_*.txt`).
Raw output of every table is in `work/famB/out/`.

## 1. Derivation and normalisation

### 1.1 The target and the sign convention (pinned numerically)

    T_ab(R) = (1/V_t) int_D w(d) [(da db + dab k^2) g](R + d) dd,
    g(r) = e^{ikr} / (4 pi f^2 r),  k = 2 pi f,  D = prod [-s_i, s_i],  w = prod (s_i - |d_i|).

The sign is `+` with no extra factor: at D = (5,1,0), s = (1/32)^3, f = 1 the formula above
reproduces the 220-bit reference tensor (36 `pairKer` face pairs assembled by the `srfSum!`
signs of the brief) to the last bit of Float64 in every non-zero entry:

    entrywise ratio famB / reference = 1.0 + 0.0im for (1,1), (1,2), (2,1), (2,2), (3,3)
    max |G - G_ref| / max|G_ref| = 0.0        max |G + G_ref| / max|G_ref| = 2.0

(script `t2_sign.jl`; the reference row is `vol|D=5,1,0|...|n=40` of the shared cache, itself
checked against ordN = 16, 24, 32, 40, stable from the 40th digit.)

### 1.2 The addition theorem, with the sign the task statement omits

For |d| < |R|, with r = R + d,

    e^{ik|R+d|} / |R+d| = ik sum_l (2l+1) (-1)^l j_l(k|d|) h_l^{(1)}(k|R|) P_l(Rhat . dhat)
                        = 4 pi ik sum_{l,m} (-1)^l h_l(kR) Y_lm(Rhat) j_l(k|d|) Y_lm(dhat).

The factor (-1)^l is REQUIRED in this form and is missing from the statement in the task
prompt: the classical theorem carries P_l(cos gamma) with gamma the angle between the two
vectors of r = r1 - r2, and here r2 = -d, so cos gamma = -Rhat.dhat.  Measured (BigFloat 300
bit, real Y_lm, L = 40, script `t1_check.jl`):

    R = (1,2,3), d = (0.3,-0.2,0.1), f = 1        without (-1)^l : rel err 6.66e-01
                                                  with    (-1)^l : rel err 1.06e-41

The constant 4 pi i k is therefore verified to 1.1e-41 (f = 1) and 1.1e-40 (f = 1+0.1i) at
rho = 0.1, i.e. far better than the 1e-30 asked for.  At larger rho the residual is pure
l-truncation: rho = 0.5165 gives 1.02e-13 at L = 40 and rho = 0.8067 gives 3.85e-12 at L = 90.

**It does not matter for family (b).**  Section 1.4 shows that the triangle weight kills every
odd l, so (-1)^l = +1 on every surviving term.  The code carries no sign.

### 1.3 Real solid harmonics with rational coefficients

Real harmonics, no Condon-Shortley phase:

    Y_l0 = N_l0 P_l(cos th),  Y_lm = sqrt2 N_lm P_l^m cos(m ph),  Y_l,-m = sqrt2 N_lm P_l^m sin(m ph),
    N_lm = sqrt( (2l+1) (l-m)! / (4 pi (l+m)!) ),   sum_m Y_lm(a) Y_lm(b) = ((2l+1)/4pi) P_l(a.b).

The regular solid harmonic r^l Y_lm(rhat) is an INTEGER polynomial times one square root:

    r^l Y_lm = ncoef(l,m) * C_l^m(z, r^2) * A_m(x,y),
    A_m = Re (x+iy)^m  (m >= 0 entries),   A_m = Im (x+iy)^m  (m < 0 entries),
    ncoef(l,m) = sqrt( (2l+1)(2 - dm0) / (4 pi (l-m)! (l+m)!) ),
    C_m^m = (2m-1)!!,  C_{m+1}^m = (2m+1)!! z,
    C_l^m = (2l-1) z C_{l-1}^m - (l+m-1)(l-m-1) r^2 C_{l-2}^m     (integers; note r^2, not x^2+y^2).

C_l^m = (l-m)! * [r^l P_l^m(z/r) stripped of (x+iy)^m]; the (l-m)! is what makes the recursion
integral, and it cancels against ncoef.  Stored in the parity-compact array P[a+1,b+1] = the
coefficient of x^{2a+p1} y^{2b+p2} z^{deg-i-j}, so multiplication by z is free (the degree
label carries it) and multiplication by r^2 is `dst[a,b] = src[a,b]+src[a-1,b]+src[a,b-1]`.
Checked against the `shrm!` normalised-Legendre recursion at a generic point, all (l,m) with
l <= 20, BigFloat 300 bit: **max relative difference 2.8e-87** (`t1_check.jl`, item 1).

### 1.4 What the weight kills

Each monomial x^i y^j z^p of r^{2n} r^l Y_lm has a fixed parity triple determined by (l,m) and
by whether the entry is the cos or the sin family:

    parity(x) = (m - t) mod 2, parity(y) = t mod 2, parity(z) = (l - m) mod 2,
    t even for the cos family, t odd for the sin family.

w(d) is even in every coordinate, so int_D w d^alpha != 0 only for alpha all even. Applying
da db shifts the parity of alpha_a and alpha_b. Hence exactly four classes survive:

    entry 11, 22, 33 :  cos family, m even >= 0, l - m even   -> l even
    entry 12         :  sin family, m even >= 2, l - m even   -> l even
    entry 13         :  cos family, m odd,       l - m odd    -> l even
    entry 23         :  sin family, m odd,       l - m odd    -> l even

**Every surviving l is even**, which is why (-1)^l never appears.  Survivor counts:

    L     all (l,m)   diag (each of 11,22,33)   12      13 = 23    stored numbers
    10      121              21                 15        15          144
    20      441              66                 55        55          484
    30      961             136                120       120         1024
    40     1681             231                210       210         1764

so the geometry table holds 3*n1 (three diagonal second derivatives) + n1 (the k^2 moment)
+ n2 + 2*n3 numbers per n, about the same as one full (l,m) table but with the three diagonal
entries and both off-diagonal families already inside it.

### 1.5 The expansion actually evaluated

    j_l(k|d|) Y_lm(dhat) = sum_n c_ln(k) |d|^{2n} [ |d|^l Y_lm(dhat) ],
    c_ln(k) = (-1)^n k^{l+2n} / (2^n n! (2l+2n+1)!!),

so with Q^{ab}_{lmn} = (1/V_t) int_D w(d) (da db + dab k^2 -> geometry part only)
[|d|^{2n} |d|^l Y_lm(dhat)] dd, split into

    Ad[a,n,lm] = (1/V_t) int_D w da^2 [ |d|^{2n+l} Y_lm ],     (a = 1,2,3)
    Bd[n,lm]   = (1/V_t) int_D w      [ |d|^{2n+l} Y_lm ],
    Ao[c,n,lm] = (1/V_t) int_D w da db [ |d|^{2n+l} Y_lm ],    (ab = 12, 13, 23)

every one an EXACT rational in the edge lengths times ncoef(l,m), because
int_D w d^alpha = prod_i mu_{alpha_i}(s_i), mu_n(s) = 2 s^{n+2}/((n+1)(n+2)) for even n, and
da^2, da db only re-index that product with the factors alpha_a(alpha_a-1), alpha_a alpha_b.
Then, once per frequency,

    Qt^{aa}_{lm}(k) = sum_n c_ln(k) ( Ad[a,n,lm] + k^2 Bd[n,lm] ),
    Qt^{ab}_{lm}(k) = sum_n c_ln(k) Ao[ab,n,lm],

and per offset only

    T_ab(R) = (i k / f^2) sum_{l even <= L} sum_m h_l(k|R|) Y_lm(Rhat) Qt^{ab}_{lm}(k).

### 1.6 Internal checks of the geometry table (script `t1_check.jl`, item 3)

lap[ |d|^{2n} H_lm ] = 2n(2n+2l+1) |d|^{2n-2} H_lm, hence sum_a Ad[a,n] = 2n(2n+2l+1) Bd[n-1].
Exact-rational table, l <= 20, 1 <= n <= 6, BigFloat 300 bit output:

    max relative violation                       1.73e-89
    n = 0 row, sum_a Ad[a,0] (harmonicity)       0.0     (exactly zero in rational arithmetic)
    max cancellation sum|term|/|acc| inside the exact contraction   2.14e5  (L = 20, nMax = 6)

The last number is the reason the geometry moments are formed in exact rational arithmetic and
not in Float64: the integer solid-harmonic coefficients alternate in sign, and 2.1e5 at L = 20
already costs 5 digits; the same quantity at L = 40 is reported in section 4.

## 2. Stable evaluation of h_l(kR), j_l and Y_lm  (script `t3_bessel.jl`, raw `out/t3_bessel.txt`)

Float64 versus BigFloat(300), l up to 40, kR from 0.3 to 300, real and complex.  `rec` is the
upward recurrence h_{l+1} = ((2l+1)/z) h_l - h_{l-1} from h_0 = -i e^{iz}/z; `sum` is the finite
closed form h_l = (-i)^{l+1}(e^{iz}/z) sum_{s<=l} i^s (l+s)!/(s!(l-s)!(2z)^s); `Miller` is the
downward recurrence for j_l normalised on the larger of j_0, j_1.

    z              l    |h_l|      err(h rec)  err(h sum)   err(Re h rec)  err(j Miller)
    0.3            10   3.7e14     2.1e-16     1.3e-16      1.27e+13       4.3e-16
    0.3            40   2.2e80     6.4e-16     8.0e-16      1.67e+74       1.7e-15
    1.0            10   6.7e8      1.5e-16     3.5e-17      3.9e+02        6.9e-17
    3.0            20   3.4e13     8.7e-16     6.9e-16      2.66e+13       6.1e-16
    10.0           20   1.2e3      1.4e-15     2.4e-14      4.0e-07        1.9e-15
    10.0           40   1.5e18     2.3e-15     7.7e-14      1.4e+24        2.7e-15
    30.0           40   1.1e1      1.3e-15     6.3e-10      1.9e-10        1.0e-15
    100.0          40   1.05e-2    3.9e-16     5.8e-14      3.2e-16        1.8e-15
    300.0          40   3.3e-3     4.4e-16     3.8e-16      4.6e-16        6.8e-15
    3.0 + 0.3i     40   1.9e39     9.0e-16     7.5e-16         -           4.7e-16
    30.0 + 3.0i    40   8.2e0      2.7e-15     1.9e-10         -           4.0e-15
    100.0 + 10.0i  40   1.1e-6     3.2e-16     2.6e-14         -           3.6e-16
    300.0 + 30.0i  40   4.1e-16    7.8e-17     6.1e-16         -           1.2e-16

Three facts, all measured:

- **The upward recurrence for h_l is accurate to 3e-15 relative everywhere tested**, l <= 40,
  |z| in [0.3, 300], Im z in [0, 30].  This includes the case that looks dangerous, Im z > 0
  where h_l is the recessive solution of the recurrence at fixed l: at z = 300+30i,
  j_0/h_0 = 1.8e10/3.1e-16 = 6e25, and the recurrence still returns h_40 to 7.8e-17.  The
  reason is that the contamination injected at l = 0 is eps*|h_0| in ABSOLUTE terms and then
  propagates as j_l/j_0, which decreases with l, while |h_l| increases; the relative
  contamination therefore shrinks with l rather than growing.  No closed form is needed.
- **The finite closed form is the worse of the two in Float64** and its error is worst when
  |z| <~ l (6.3e-10 at z = 30, l = 40; 1.9e-10 at z = 30+3i, l = 40; 2.6e-14 at z = 100+10i),
  where the terms (l+s)!/(s!(l-s)!(2|z|)^s) grow before they decay.  For |z| >> l it is as good
  as the recurrence (3.8e-16 at z = 300, l = 40).  Family (b) uses the recurrence.
- **The radiative part is the one that needs care.**  For real k, Re h_l = j_l and the upward
  recurrence destroys it as soon as l > ~|z|: err(Re h rec) is 1.3e13 at z = 0.3, l = 10 and
  1.4e24 at z = 10, l = 40.  Miller's downward recurrence returns j_l to relative accuracy at
  every (z,l) tested, worst 6.8e-15 (worst case 6.0e-13 at z = 300, l = 5, which sits on a zero
  of j_5).  `hnkFix!` therefore replaces Re h_l by the Miller j_l whenever Im z = 0.
  **Answer to the question posed: yes, j_l(kR) is obtained to relative accuracy for l > kR,
  by Miller and only by Miller.**  Section 5 measures whether it matters for the tensor: for
  real f the imaginary part of T is (k/f^2) sum_l j_l Y Q while the real part is
  -(k/f^2) sum_l y_l Y Q, so the absolute error induced in Im T by a corrupted j_l is
  eps * |Re T|, and the relative damage is eps |Re T| / |Im T| ~ eps * 3 (kR)^{-3}.

Real spherical harmonics from the normalised associated-Legendre recursion (`shrm!`), l <= 40,
Float64 versus BigFloat:

    direction              max abs err   max rel err (|Y| > 1e-12)
    (1,0,0)                4.98e-16      9.6e-16
    (0,0,1)                6.21e-15      2.8e-15
    (1,1,0)/sqrt2          5.18e-16      1.3e-15
    (1,1,1)/sqrt3          1.10e-15      1.3e-13
    (0.3,-0.5,0.8123)      1.64e-15      4.8e-14

The relative outliers sit on near-zeros of individual Y_lm; the absolute error, which is what
enters sum_m Y_lm Qt_lm, never exceeds 6.2e-15.

## 3. Convergence in l, and an a priori remainder bound

### 3.1 The two elementary bounds (proofs)

**|j_l(z)| <= |z|^l e^{|z|^2/(4l+6)} / (2l+1)!!.**  From
j_l(z) = (z^l/(2l+1)!!) sum_n (-z^2/2)^n / (n! prod_{q=1..n} (2l+2q+1)) and
(2l+2q+1) >= (2l+3) for q >= 1,
|j_l(z)| <= (|z|^l/(2l+1)!!) sum_n (|z|^2/2)^n/(n! (2l+3)^n) = (|z|^l/(2l+1)!!) e^{|z|^2/(2(2l+3))}. QED

**|h_l^{(1)}(z)| <= (e^{-Im z}/|z|) sum_{s=0}^{l} (l+s)!/(s!(l-s)!(2|z|)^s).**  Triangle
inequality on the classical finite sum h_l^{(1)}(z) = (-i)^{l+1}(e^{iz}/z) sum_s i^s
(l+s)!/(s!(l-s)!(2z)^s) (Abramowitz-Stegun 10.1.16), |e^{iz}| = e^{-Im z}. QED
Both are checked numerically inside `remBndAll`/`t4_conv.jl` (the bound is never violated in
any row of any table below).

### 3.2 Effect of da db, and the bound actually used

Move the derivatives onto the SINGULAR side: since d/dR_a g(R+d) = d/dd_a g(R+d),

    T_ab = (ik/(f^2 V_t)) sum_{l,m} [ (da db + dab k^2) ( h_l(k|R|) Y_lm(Rhat) ) ] * I_lm,
    I_lm = int_D w(d) j_l(k|d|) Y_lm(dhat) dd.

Each Cartesian first derivative of a singular solid harmonic h_l Y_lm is k times a combination
of at most four terms h_{l+-1} Y_{l+-1, m or m+-1} with coefficients of modulus <= 1 (the
standard differentiation relations a_lm = sqrt((l-m)(l+m)/((2l-1)(2l+1))) etc.), so two
derivatives give at most 16 such terms with l' in {l-2, l, l+2}, and the dab k^2 term adds one
more: the constant 17.  Using |Y_lm| <= sqrt((2l+1)/4pi), sum_m |Y_lm| <= (2l+1)/sqrt(4pi),
hbnd increasing in l, and |j_l(k|d|)| <= |k|^l |d|^l e^{(|k| r_d)^2/(4l+6)}/(2l+1)!| POINTWISE
in |d| (so that the exact radial moment absorbs the geometry),

    |Rem_L| <= (17 |k|^3 / (4 pi |f|^2 V_t)) sum_{l > L, even} (2l+1) sqrt(2l+5)
               * |k|^l e^{(|k| r_d)^2/(4l+6)} / (2l+1)!! * W_l * hb(l+2, |kR|),
    W_l = int_D w(d) |d|^l dd = sum_{i+j+p=l/2} (l/2)!/(i!j!p!) mu_{2i}(s1) mu_{2j}(s2) mu_{2p}(s3),
    r_d = sqrt(s1^2+s2^2+s3^2).

W_l is an exact positive sum of box moments (no cancellation), and using it instead of the
crude V r_d^l is what makes the bound usable: with V r_d^l the measured bound/actual ratio is
1e5 to 1e12; with W_l it is **1.8e2 to 4e6** over the whole cubic-1/32 table (`t4_conv.jl`
column `bound(L13)/act`).  It is still an over-estimate, by 2 to 6 orders, which costs roughly
a factor 1.3 to 1.6 in the L it selects; the residual looseness is the constant 17, the
Cauchy-Schwarz sum_m |Y_lm| <= (2l+1)/sqrt(4pi), and |P_l| <= 1.

### 3.3 Measured convergence, cube lambda/32 (script `t4_conv.jl`, `out/t4_c32_L40.txt`)

L(eps) = the smallest even L for which |sum_{l>L} t_l| / |T_ab| < eps, computed in BigFloat
from the per-l terms; -1 = not reached within L = 40.  Cancellation columns:
`c_l` = sum_l |t_l| / |T_ab|, `c_lm` = sum_{l,m} |t_lm| / |T_ab|.

    offset       rho     kR      ent  L(1e-8) L(1e-11) L(1e-13) L(1e-15)  c_l    c_lm   bnd/act
    (2,0,0)      0.866   0.393   11     -1      -1       -1       -1      1.06   1.06     -
    (2,0,0)      0.866   0.393   22     -1      -1       -1       -1      1.28   2.45     -
    (3,0,0)      0.577   0.589   11     16      28       32       -1      1.03   1.03   4.3e6
    (3,0,0)      0.577   0.589   22     16      28       32       -1      1.49   2.71   8.8e6
    (4,0,0)      0.433   0.785   11     14      20       26       30      1.04   1.04   3.2e4
    (6,0,0)      0.289   1.178   11     10      16       18       22      1.14   1.14   4.3e3
    (8,0,0)      0.217   1.571   11     10      12       16       18      1.29   1.29   1.4e4
    (16,0,0)     0.108   3.142   11      6      10       12       14      2.18   2.18   1.9e3
    (32,0,0)     0.054   6.283   11      6       8       10       10      4.22   4.22   3.0e3
    (64,0,0)     0.027  12.566   11      6       8        8       10      8.39   8.39   1.9e2
    (2,2,0)      0.612   0.555   11     22      32       38       -1      1.04   1.04   3.1e5
    (2,2,0)      0.612   0.555   12     22      30       36       -1      1.00   1.00   5.1e5
    (3,3,0)      0.408   0.833   11     12      18       22       28      1.09   1.09   -
    (4,4,0)      0.306   1.111   11     10      16       18       22      1.16   1.16   -
    (6,6,0)      0.204   1.666   11      8      12       14       16      1.30   1.30   -
    (8,8,0)      0.153   2.221   11      8      10       12       14      1.40   1.40   -
    (16,16,0)    0.077   4.443   11      6       8       10       10      1.58   1.58   -
    (32,32,0)    0.038   8.886   11      6       6        8       10      1.64   1.64   -
    (64,64,0)    0.019  17.772   11      6       6        8       10      1.66   1.66   -
    (2,2,2)      0.500   0.680   11     10      16       22       26      1.00   1.01   7.7e7
    (2,2,2)      0.500   0.680   12     18      26       32       36      1.00   1.00   2.0e4
    (3,3,3)      0.333   1.020   11      8      12       16       18      1.00   1.00   -
    (4,4,4)      0.250   1.360   11      6      10       12       16      1.00   1.00   -
    (6,6,6)      0.167   2.041   11      4       8       10       12      1.00   1.00   -
    (8,8,8)      0.125   2.721   11      4       6        8       10      1.00   1.00   2.6e6
    (16,16,16)   0.063   5.441   11      4       6        6        8      1.00   1.00   3.3e6
    (32,32,32)   0.031  10.883   11      4       6        6        8      1.00   1.00   6.5e5
    (64,64,64)   0.016  21.766   11      4       6        6        8      1.00   1.00   6.5e5

f = 1+0.1i changes no L by more than 2 (full table in the output file).

Two things worth stating exactly:

- **Convergence is faster than rho^l, by a fixed factor in the ratio.**  10^{-13/L(1e-13)} is
  0.394 at rho = 0.577, 0.317 at 0.433, 0.190 at 0.289, 0.153 at 0.217, 0.081 at 0.108: the
  effective ratio is 0.68 to 0.71 times rho at every separation.  The reason is that the corner
  radius r_d = sqrt(3) s that defines rho is attained on a set where the triangle weight
  vanishes cubically; the weighted radial mean is (int w |d|^2/int w)^{1/2} = s/sqrt2 = 0.41 r_d.
  So the practical rule for cubes is **L(1e-13) ~ 13 / log10(1/(0.7 rho))**.
- **The l-sum has no cancellation.**  sum_l |t_l| / |T| never exceeds 8.4 in the whole table,
  and sum over (l,m) never exceeds 8.4 either; on the body diagonal it is 1.000.  The largest
  value, 8.39, is entry 11 at (64,0,0), where kR = 12.6 and the leading terms alternate with
  the phase of h_l.  Family (b) therefore loses at most one digit to the l-sum.

### 3.4 Coarse cells and the slender cell (`out/t4_c8_L40.txt`, `t4_c4_L40.txt`, `t4_sl_L40.txt`)

Cube lambda/8 (k d = 1.360) and lambda/4 (k d = 2.721), entry 11 unless stated, f = 1:

    offset      rho     kR(l/8)  L13(l/8)  c_l(l/8)   kR(l/4)  L13(l/4)  c_l(l/4)
    (2,0,0)     0.866    1.571     -1        1.33      3.142      -1       2.19
    (3,0,0)     0.577    2.356     32        1.71      4.712      32       3.17
    (4,0,0)     0.433    3.142     24        2.18      6.283      24       4.21
    (6,0,0)     0.289    4.712     16        3.19      9.425      18       6.30
    (8,0,0)     0.217    6.283     16        4.22     12.566      16       8.40
    (16,0,0)    0.108   12.566     12        8.39     25.133      16      16.8
    (32,0,0)    0.054   25.133     12       16.8      50.265      16      33.6
    (64,0,0)    0.027   50.265     12       33.5     100.531      16      67.2
    (2,2,0)     0.612    2.221     36        1.40      4.443      36       1.58
    (4,4,4)     0.250    5.441     14        1.00     10.883      16       1.00
    (8,8,8)     0.125   10.883     12        1.00     21.766      14       1.00

**L is set by rho, not by kR**: the L(1e-13) column is the same to within +-2 for lambda/32,
lambda/8 and lambda/4 at the same offset in cells, even though kR differs by a factor 8.  What
kR controls is the cancellation of the l-sum for the axial 11 entry, which grows linearly:
c_l = 8.4 at kR = 12.6, 16.8 at 25.1, 33.6 at 50.3, 67.2 at 100.5 (i.e. c_l ~ 0.67 kR).  That
is 1.8 digits at kR = 100 and is the only place in family (b) where the l-sum loses anything.
Every other entry and every diagonal direction has c_l <= 1.7.

Slender cell (1/32, 1/32, 1/512), r_d = 0.04421, k d = 0.278, f = 1:

    offset       rho      kR      L(1e-8)  L(1e-11)  L(1e-13)   c_l    c_lm
    (2,0,0)     0.7078   0.393     26        -1        -1       1.00   1.04
    (3,0,0)     0.4719   0.589     18        26        30       1.02   1.02
    (4,0,0)     0.3539   0.785     14        18        22       1.04   1.04
    (8,0,0)     0.1769   1.571     10        12        14       1.29   1.29
    (16,0,0)    0.0885   3.142      6        10        12       2.18   2.18
    (0,0,24)    0.9437   0.295     -1        -1        -1       3.01   3.01
    (0,0,32)    0.7078   0.393     36        -1        -1       2.01   2.01
    (0,0,64)    0.3539   0.785     16        22        26       1.95   1.95
    (2,2,0)     0.5005   0.555     22        30        36       1.03   1.03
    (4,4,0)     0.2502   1.111     12        16        20       1.16   1.16
    (2,2,2)     0.5000   0.556     20        28        34       1.03   1.03
    (8,8,8)     0.1250   2.224      8        10        12       1.41   1.41

**Along the short axis rho > 1 until 23 cells** (rho = r_d/(n/512) = 22.63/n), so family (b)
does not converge there at all for n <= 22; it is not slow, it diverges.  At n = 24 (rho =
0.944) it converges but not to 1e-8 within L = 40; at n = 32 (rho = 0.708) it reaches 1e-8 at
L = 36 and no further within 40; only from n = 64 (rho = 0.354) does it reach 1e-13, at L = 26.
That band belongs to the k-series (k D along the needle is 0.29 at n = 24), not to family (b).
The in-plane offsets of the slender cell behave like the cube with the same rho: (2,0,0) has
rho = 0.708 and is as bad as the cube's (2,2,0); everything from 3 cells outward is normal.

### 3.5 Truncation of the k-series in n (script `t7_nmax.jl`, `out/t7_nmax.txt`)

Relative size of the n-th term of Qtilde_lm, maximised over l <= 24, m and entry type:

    shape          k d      n=0    n=2      n=4      n=6      n=8      n=10     n=12
    cube 1/32      0.340    2.0    2.5e-3   1.9e-9   4.6e-16  3.9e-23  1.5e-30  3.1e-38
    cube 1/8       1.360    2.1    4.2e-2   8.4e-6   5.1e-10  1.1e-14  1.1e-19  5.8e-25
    cube 1/4       2.721    2.4    2.0e-1   6.8e-4   6.4e-7   2.2e-10  3.4e-14  2.9e-18
    slender        0.278    2.0    1.8e-3   1.1e-9   1.7e-16  9.6e-24  2.5e-31  3.3e-39

f = 1+0.1i shifts every entry by less than 10%.  For 1e-16 relative truncation:
**nMax = 6 at lambda/32 and for the slender cell, 10 at lambda/8, 12 at lambda/4** — the n
series is a non-issue, and its own cancellation sum_n|term|/|sum| is 3.07 (lambda/32), 3.29
(lambda/8), 4.14 (lambda/4).  Only the n = 0 entry exceeds 1 (2.0 to 2.4), i.e. the series is
alternating with a first term about twice the total.

## 4. Conditioning

Three separate places can lose digits; all three were measured.

1. **The geometry contraction** (integer solid-harmonic coefficients against box moments).
   sum|term|/|acc| = 2.1e5 at L = 20 and **5.8e8 at L = 40** for the cubes, 1.1e7 for the
   slender cell.  That is 8.8 digits, so the geometry table CANNOT be built in Float64.  It is
   built once per cell shape in exact rational arithmetic (`geoTab` with Rational{BigInt} edge
   lengths and BigInt polynomial coefficients) and rounded to Float64 at the end; a
   BigFloat(1024) path is available and agrees.
2. **The n-sum (k-series inside Qtilde)**: 3.07 / 3.29 / 4.14 for lambda/32, lambda/8,
   lambda/4.  Half a digit.  Also done once per frequency, not per offset.
3. **The per-offset l-sum**: sum_l |t_l|/|T| <= 8.4 over the entire lambda/32 table, and
   sum_{l,m} |t_lm|/|T| is the same to 3 digits.  The only growth is the axial 11 entry, where
   c_l ~ 0.67 kR: 16.8 at kR = 25, 33.6 at kR = 50, 67.2 at kR = 100 (lambda/4, 64 cells).
   **At most 1.8 digits, and under 1 digit everywhere at lambda/32.**

So family (b) is conditioned entirely by the one-time geometry table; the per-offset evaluation
is essentially cancellation-free, which is the property the Cartesian monomial organisation of
family (a) was predicted to lack.

## 6. Cost (script `t6_cost.jl`, `t8_agg.jl`; `out/t6_cost.txt`, `out/t8_agg.txt`)

Flops per offset (real flops, complex mul = 6, add = 2; Y_lm recursion + h_l recursion +
the four contracted sums):

    L      nY      n1(diag)  n2(xy)  n3(xz)  flops/offset
     6      49       10        6       6        914
    10     121       21       15      15       2042
    14     225       36       28      28       3618
    20     441       66       55      55       6822
    26     729      105       91      91      11034
    30     961      136      120     120      14402
    40    1681      231      210     210      24782

Measured Float64, single thread, M3 Pro, zero allocations:

    L     fix=false   fix=true (Miller j_l for real f)
    10     391 ns      1104 ns
    20    1250 ns      2028 ns
    30    2731 ns      3599 ns
    40    4972 ns      5820 ns

i.e. t(L) = 3.1 L^2 + 80 ns, and the Miller repair of the radiative part costs a flat 850 ns
(a fixed ~110 extra downward steps).  One-time geometry table per cell shape, exact rational:
1.27 s (L = 20, nMax = 6), 3.51 s (L = 30, nMax = 8), **14.65 s (L = 40, nMax = 12)**, 20202
stored numbers.  One frequency contraction (`qTab`) costs 31 ms and is shared by all offsets.

Aggregate over Gila's actual offset set (egoToe, N^3 offsets, dropping max|n| <= 1 which is the
contact and touching shell of the k-series), choosing L per offset from the measured rule
L = 13/log10(1/(0.7 rho)):

    N      offsets    total time, fix=true    fix=false   L histogram
    32      32768        0.04 s                0.01 s     8:35 10:24598 12:6221 14:1212 ...
    64     262144        0.31 s                0.08 s     8:188977 10:65032 12:6221 ...
    128   2097152        2.37 s                0.59 s     6:88342 8:1935643 10:65032 12:6221 ...

**2.4 seconds single-threaded for the whole 128^3 far field at lambda/32**, against the ten to
thirty minutes of the quadrature path quoted in the prompt.  The L histogram shows why: 96.5% of
the offsets need L <= 8, and only 12 offsets per octant (those with |n| <= 2.56, i.e. the
classes (2,0,0), (2,1,0), (2,1,1)) need L > 40 and are not reached by this family at 1e-13.

## 5. Accuracy against the 220-bit reference (script `t5_acc.jl`, `t9_deep.jl`)

Reference: the 36 face-pair integrals from `pairKer` at 220 bits, ordN = 44, assembled by the
`srfSum!` signs of the brief, divided by V_t.  Its own convergence was checked here, not
assumed: **ordN 44 versus ordN 60 agree to 2.46e-64 at D = (2,0,0) and 2.62e-63 at (2,2,0)**
(`t10_refchk.jl`), so the reference is exact at every level discussed below.
`eF`/`eB` = max over the 9 entries of |G - G_ref| / max|G_ref| in Float64 / BigFloat at the
same L; `pF` = worst per-entry relative error in Float64 over entries above 1e-4 of the largest;
`dig` = log10(pF/eps).  `Lb` is the L the a priori bound of 3.2 selects for 1e-13; `L` is what
was used (capped at the table's L).

### 5.1 Cube lambda/32, both frequencies, all three directions (`out/t5_c32.txt`, 48 rows)

    offset       f          rho     kR      Lb   L    eF        eB        pF        dig
    (2,0,0)      1          0.866   0.393  178   40   3.5e-8    3.5e-8    4.1e-8    8.3
    (2,2,0)      1          0.612   0.555   58   40   4.0e-14   4.0e-14   7.2e-14   2.5
    (2,2,2)      1          0.500   0.680   44   40   1.7e-15   3.5e-17   1.7e-15   0.9
    (3,0,0)      1          0.577   0.589   52   40   3.5e-15   3.4e-15   4.1e-15   1.3
    (3,3,0)      1          0.408   0.833   34   34   1.4e-15   1.2e-18   1.4e-15   0.8
    (3,3,3)      1          0.333   1.020   28   28   3.9e-16   2.4e-18   4.0e-16   0.3
    (4,0,0)      1          0.433   0.785   36   36   3.9e-16   8.5e-19   7.7e-16   0.5
    (4,4,0)      1          0.306   1.111   26   26   4.9e-16   2.7e-18   5.4e-16   0.4
    (4,4,4)      1          0.250   1.360   22   22   5.1e-16   5.9e-18   5.1e-16   0.4
    (6,0,0)      1          0.289   1.178   24   24   1.8e-16   7.7e-18   4.1e-16   0.3
    (6,6,6)      1          0.167   2.041   18   18   2.5e-16   8.9e-19   2.5e-16   0.0
    (8,0,0)      1          0.217   1.571   20   20   2.7e-16   3.7e-18   2.7e-16   0.1
    (8,8,8)      1          0.125   2.721   16   16   3.8e-16   5.4e-19   6.1e-16   0.4
    (16,0,0)     1          0.108   3.142   14   14   4.1e-16   8.0e-18   4.4e-16   0.3
    (16,16,16)   1          0.063   5.441   12   12   6.5e-16   2.2e-19   9.9e-16   0.6
    (32,0,0)     1          0.054   6.283   12   12   3.2e-16   2.1e-19   9.6e-16   0.6
    (32,32,32)   1          0.031  10.883   10   10   1.0e-15   1.0e-18   1.0e-15   0.7
    (64,0,0)     1          0.027  12.566   10   10   9.7e-16   6.4e-19   2.1e-15   1.0
    (64,64,0)    1          0.019  17.772   10   10   5.6e-16   1.5e-19   6.0e-16   0.4
    (64,64,64)   1          0.016  21.766   10   10   1.7e-15   4.3e-19   1.7e-15   0.9

f = 1+0.1i (24 more rows) is the same to within a factor 2 in every row.  **Every offset from
three cells outward, in every direction, at both frequencies, is at 2.1e-15 or better in
Float64, losing at most one digit** (dig <= 1.0).  Separating real and imaginary parts, the
worst per-entry |ReErr| is 5.4e-14 and |ImErr| 2.2e-14, both at 64 cells where the smaller
part of an entry is 1e-2 of the larger.  BigFloat at the same L is 2e-20 to 1e-17, i.e. the
Float64 numbers are ROUNDING-limited, not truncation-limited, everywhere except at 2 cells.

### 5.2 The 2-cell band, pushed to L = 100 (`out/t9_deep.txt`)

Geometry table to L = 100 with BigFloat(1024) moments (173 s); checked against the exact
rational table for l <= 40: max relative difference 1.6e-66.

    offset     f          L=40      L=60      L=80      L=94      L=100     dig at best L
    (2,0,0)    1          3.5e-8    3.4e-10   4.8e-12   5.7e-14   8.1e-14      2.5
    (2,0,0)    1+0.1i     3.6e-8    3.4e-10   4.9e-12   5.6e-14   7.9e-14      2.5
    (2,2,0)    1          4.0e-14   7.9e-16   7.9e-16   7.9e-16   7.9e-16      0.5
    (2,2,2)    1          1.7e-15   1.7e-15   1.7e-15   1.7e-15   1.7e-15      0.9
    (3,0,0)    1          3.5e-15   3.7e-16   3.7e-16   3.7e-16   3.7e-16      0.6

(eB tracks eF at (2,0,0) to two digits at every L, and falls to 2e-20 at (2,2,0), (2,2,2),
(3,0,0) once L exceeds the truncation requirement: so 5.7e-14 at (2,0,0) is TRUNCATION, not
rounding, and the reference is exact to 2.5e-64 there.)

**Verdict for the 2-cell axial class:** family (b) converges — rho = 0.866 < 1 — but needs
L = 94 for 6.6e-14 per entry, and within L = 100 it does not reach 1e-15.  Cost at L = 94 is
27 us per offset; there are 12 such offsets per octant (the classes (2,0,0), (2,1,0), (2,1,1),
|n| <= 2.56) whatever N, so the total extra cost is 0.33 ms and the L = 100 geometry table
costs 173 s once per shape.  It is affordable but it is the one place where the family does not
reach the 1e-13 target comfortably, and it is truncation, not conditioning: sum_l|t_l|/|T| is
1.06 to 1.28 there.  Subdivision (family d) or the k-series is the natural fix for those 12
offsets; at lambda/32 k D = 0.68 there, well inside the k-series.

### 5.3 Symmetries and homogeneity (`t5_acc.jl`, bottom of `out/t5_c32.txt`)

    T(-R) = T(R)                                        max violation 0.0 (bit-exact)
    reflection of axis c flips entries with one index c  max violation 0.0 (bit-exact)
    T(aR, a s; f/a) = a^2 T(R, s; f), a = 1/4            max violation 0.0 (bit-exact)

The symmetries are exact rather than approximate because they are properties of the (l,m)
parity classes themselves: T(-R) = T(R) because only even l survive, and the axis reflections
act on the stored Y_lm indices by a sign.

## 7. Where family (b) stands, band by band

Cubic cells, both frequencies, all three directions, Float64, per-entry relative error against
the 220-bit reference, with the L each band needs:

    band (cells)   rho      L for 1e-13   Float64 achieved   status
    2 (axis)       0.866       94          6.6e-14           converges, 2.5 digits lost, needs L=94
    2 (face diag)  0.612       38          7.9e-16           clean at L = 60
    2 (body diag)  0.500       22          1.7e-15           clean
    3 (axis)       0.577       32          3.7e-16           clean at L = 60
    4              0.433       26          7.7e-16           clean
    6              0.289       18          4.1e-16           clean
    8              0.217       16          2.7e-16           clean
    16             0.108       12          4.4e-16           clean
    32             0.054       10          9.6e-16           clean
    64             0.027        8          2.1e-15           clean
    slender, short axis n <= 22 : rho > 1, DIVERGES (rho = 22.63/n)
    slender, short axis n = 24  : rho = 0.944, 1.3e-4 at L = 40 (11.8 digits lost)
    slender, short axis n = 32  : rho = 0.708, 1.5e-9  at L = 40
    slender, short axis n = 64  : rho = 0.354, 3.4e-16 at L = 32
    slender, in-plane n >= 3    : same as the cube at equal rho

Everything from three cells outward, in every direction, at lambda/32, lambda/8 and lambda/4,
at f = 1 and f = 1+0.1i, is at or below 2.1e-15 in Float64 with at most one digit lost, and the
error is ROUNDING, not truncation (BigFloat at the same L is 1e-17 to 1e-20).  The only two
failures are:

- **the 12 offsets per octant with |n| <= 2.56** (classes (2,0,0), (2,1,0), (2,1,1)), where the
  failure is TRUNCATION: rho = 0.87 needs L = 94 for 6.6e-14 and does not reach 1e-15 by
  L = 100.  Conditioning there is perfect (sum_l|t_l|/|T| = 1.06 to 1.28), so this is a
  convergence-rate failure and nothing else.  It is fixed by subdivision or by the k-series
  (k D = 0.68 there at lambda/32), not by more precision.
- **the slender cell along its short axis inside 23 cells**, where rho > 1 and the expansion
  does not converge at all.  Same cause (geometry), same remedy (the k-series: k D = 0.29 at
  n = 24).

## 8. What is not settled, and where the numbers are weakest

- **The a priori bound over-selects L by 1.5 to 2.0.**  Measured: Lb (bound) versus L (measured
  need) for 1e-13 at lambda/32: 178 vs 94 at (2,0,0), 58 vs 38 at (2,2,0), 44 vs 22 at (2,2,2),
  52 vs 32 at (3,0,0), 36 vs 26 at (4,0,0), 20 vs 16 at (8,0,0), 10 vs 8 at (64,0,0).  The cost
  penalty is (Lb/L)^2 = 1.6 to 4.  Tightening it means replacing sum_m |Y_lm| <= (2l+1)/sqrt(4pi)
  and the constant 17 by something that sees the actual angular structure; I did not do it.
- **The constant 17 in the derivative bound is argued, not proved here.**  It rests on the
  standard differentiation relations for solid harmonics having coefficients of modulus <= 1
  (a_lm = sqrt((l-m)(l+m)/((2l-1)(2l+1))) and its m+-1 partners).  Every measured tail is far
  below the bound (bound/actual 1.8e2 to 4e6 at lambda/32, up to 2.8e8 at lambda/4), so the
  bound is never violated in any run here, but a reader who wants a theorem should check those
  coefficients rather than take the 17 on trust.
- **The L = 100 geometry table was built with BigFloat(1024) moments, not exact rationals**
  (173 s versus an exact-rational run that had not finished in 20 minutes).  It agrees with the
  exact table to 1.6e-66 for l <= 40.  For a cubic cell many (l,m) entries vanish identically by
  octahedral symmetry and come out as ~1e-300 noise instead of 0 in that path; they are harmless
  at 1e-300 relative but they make the cancellation diagnostic report Inf.
- **The measured per-offset timings use a workspace sized to the L actually used.**  A
  production implementation that allocates one workspace at the maximum L and evaluates at a
  smaller Lcut would pay the full (L+1)^2 cost of the Y_lm recursion; `shrm!` must be given the
  band's L, not the table's.
- **No GPU, no threading, no wiring into `GlaVacOprMem`.**  The 2.37 s figure for 128^3 is
  a per-offset benchmark times the offset count, not a measured Gila build.
- **Only the offsets and shapes listed were verified.**  In particular no random lattice
  direction and no aspect ratio between 1 and 16 other than the cube and the 1:16 slender cell.

## 9. Comparison with the Cartesian organisation (family a), from these measurements alone

The two are the same expansion; the difference is where the cancellation is parked.  In the
(l,m) organisation:

- the cancellation is entirely inside the geometry table (5.8e8 at L = 40), which is
  frequency-independent, computed once per cell shape in exact rational arithmetic, and rounded
  to Float64 only at the end;
- the per-offset sum has sum_{l,m}|term|/|total| <= 8.4 over the whole lambda/32 table and
  <= 67 at lambda/4, 64 cells, i.e. at most 1.8 digits, so Float64 per-offset evaluation reaches
  the rounding floor (2e-15 relative to the largest entry, one digit lost);
- the truncation is controlled by a single index L with a proven bound, and the number of terms
  is the same as the Cartesian count only up to the parity pruning of section 1.4 (which removes
  every odd l and three quarters of the (l,m) pairs per entry type).

A Cartesian monomial organisation with the same geometry would have to carry the same 5.8e8
cancellation into the per-offset sum, because there the cancellation is between monomials whose
coefficients depend on R.  That is the structural argument for (b); this report does not measure
(a) and does not claim its numbers.

## 10. Summary of the numbers

    addition theorem constant 4 pi i k, real Y_lm, BigFloat 300 bit   1.06e-41 (rho = 0.1)
    (-1)^l required in the Rhat.dhat form; every surviving l is even, so it never appears
    integer solid harmonic vs shrm! recursion, l <= 20                2.81e-87
    Helmholtz identity on the geometry table, l <= 20, n <= 6         1.73e-89
    harmonicity of H_lm (n = 0 row)                                   0.0 exactly
    sign convention: T = +(1/V_t) int w (da db + dab k^2) g           bit-exact vs reference
    h_l Float64 upward recurrence, l <= 40, |z| in [0.3,300]          <= 3e-15 relative
    h_l Float64 finite closed form                                    up to 6.3e-10 (|z| ~ l)
    j_l Float64 Miller downward recurrence                            <= 7e-15 relative
    Re h_l from the upward recurrence for l > |z|                     destroyed (1e13 to 1e74)
    real Y_lm Float64, l <= 40                                        <= 6.2e-15 absolute
    geometry contraction cancellation, L = 40 cube                    5.84e8  (exact arithmetic)
    n-series cancellation, lambda/32 / lambda/8 / lambda/4            3.07 / 3.29 / 4.14
    per-offset l-sum cancellation, lambda/32, all offsets             <= 8.4
    per-offset l-sum cancellation, lambda/4, 64 cells, entry 11       67.2
    a priori bound / measured tail                                    1.8e2 to 4e6 (lambda/32)
    L(1e-13) rule for cubes                                           13 / log10(1/(0.7 rho))
    n_max for 1e-16, lambda/32 / lambda/8 / lambda/4                  6 / 10 / 12
    Float64 accuracy vs the 220-bit reference, >= 3 cells             <= 2.1e-15, <= 1 digit lost
    Float64 accuracy vs the reference at 2 cells (axis), L = 94       6.6e-14, 2.5 digits lost
    reference self-check ordN 44 vs 60 at (2,0,0)                     2.46e-64
    T(-R) = T(R) and axis reflections                                 0.0 (bit-exact)
    T(aR, a s; f/a) = a^2 T(R, s; f), a = 1/4                         0.0 (bit-exact)
    per-offset Float64 time, L = 10 / 20 / 30 / 40                    391 / 1250 / 2731 / 4972 ns
    the same with the Miller repair                                   1104 / 2028 / 3599 / 5820 ns
    geometry table, exact rational, L = 40 nMax = 12                  14.65 s, 20202 numbers
    geometry table, BigFloat(1024), L = 100 nMax = 6                  173 s
    whole 128^3 far field at lambda/32, single thread                 2.37 s (0.59 s without repair)

## 11. Accuracy for the coarse cubes and the slender cell (added after section 5)

All Float64, L chosen by the a priori bound and capped at 40; f = 1 rows shown, f = 1+0.1i is
the same to within a factor 3 (full tables `out/t5_c8.txt`, `out/t5_c4.txt`, `out/t5_sl.txt`).

Cube lambda/8 (`t5_c8.txt`):

    offset      rho     kR      Lb   L    eF        eB        pF        dig
    (2,0,0)     0.866   1.571  182   40   2.1e-8    2.1e-8    2.1e-8    8.0
    (2,2,0)     0.612   2.221   60   40   1.5e-14   1.5e-14   1.9e-14   1.9
    (2,2,2)     0.500   2.721   44   40   7.1e-16   8.3e-18   7.2e-16   0.5
    (3,0,0)     0.577   2.356   54   40   1.6e-15   1.7e-15   1.6e-15   0.9
    (4,0,0)     0.433   3.142   36   36   8.1e-16   2.5e-19   8.1e-16   0.6
    (6,0,0)     0.289   4.712   26   26   4.4e-16   2.6e-20   9.8e-16   0.6
    (8,8,8)     0.125  10.883   18   18   9.5e-16   8.3e-22   1.2e-15   0.7
    (16,16,16)  0.063  21.766   14   14   2.7e-15   8.1e-19   2.7e-15   1.1

Cube lambda/4 (`t5_c4.txt`):

    (2,0,0)     0.866   3.142  188   40   9.9e-9    9.9e-9    1.4e-8    7.8
    (2,2,0)     0.612   4.443   62   40   4.2e-15   4.4e-15   7.0e-15   1.5
    (2,2,2)     0.500   5.441   46   40   7.8e-16   2.7e-18   1.5e-15   0.8
    (3,0,0)     0.577   4.712   56   40   8.5e-16   4.9e-16   1.9e-15   0.9
    (4,4,4)     0.250  10.883   26   26   9.7e-16   1.5e-21   1.4e-15   0.8
    (8,0,0)     0.217  12.566   24   24   8.0e-16   8.7e-22   2.1e-15   1.0
    (16,16,16)  0.063  43.531   18   18   4.2e-15   8.6e-20   4.2e-15   1.3

Slender (1/32, 1/32, 1/512) (`t5_sl.txt`):

    (0,0,24)    0.944   0.295  200   40   1.2e-4    1.2e-4    1.3e-4   11.8   short axis, rho -> 1
    (0,0,32)    0.708   0.393   92   40   1.2e-9    1.2e-9    1.5e-9    6.8
    (0,0,64)    0.354   0.785   32   32   2.6e-16   8.2e-18   3.4e-16   0.2
    (2,0,0)     0.708   0.393   92   40   4.5e-10   4.5e-10   1.1e-9    6.7
    (2,2,0)     0.501   0.555   46   40   4.1e-15   3.7e-15   4.6e-15   1.3
    (2,2,2)     0.500   0.556   46   40   2.5e-15   2.4e-15   3.6e-14   2.2
    (3,0,0)     0.472   0.589   42   40   5.9e-16   4.2e-17   5.9e-16   0.4
    (4,0,0)     0.354   0.785   32   32   1.8e-15   2.7e-18   1.8e-15   0.9
    (8,8,8)     0.125   2.224   16   16   1.3e-16   2.3e-17   3.7e-16   0.2
    (16,16,16)  0.063   4.447   12   12   3.7e-16   1.3e-17   5.6e-16   0.4

At lambda/8 and lambda/4 the picture is identical to lambda/32: everything from three cells
outward is at 4.2e-15 or better with at most 1.3 digits lost, and eB at the same L is 1e-19 to
1e-22, so Float64 is rounding-limited.  The kR = 43.5 case (lambda/4, 16 cells body diagonal)
is the largest kR verified and is at 4.2e-15.  The slender cell obeys the same rho rule with no
extra penalty, once rho is computed from its own r_d.

## 12. Does the Miller repair of the radiative part matter? (`t12_fix.jl`, `out/t12_fix.txt`)

Float64, f = 1, lambda/32, comparing the tensor computed with `fix = true` (Re h_l from Miller)
and `fix = false` (Re h_l from the upward recurrence, i.e. garbage for l > kR):

    offset      L    |Im/Re|   ImErr fix    ImErr no fix   ReErr (identical either way)
    (3,0,0)     40   0.147     9.3e-17      1.1e-15        4.1e-15
    (4,0,0)     36   0.345     2.4e-16      2.4e-16        8.1e-16
    (6,0,0)     24   0.861     1.2e-16      2.8e-16        5.4e-16
    (8,0,0)     20   0.934     3.4e-16      3.4e-16        3.2e-16
    (16,0,0)    14   3.14      6.6e-16      3.2e-16        8.1e-16
    (32,0,0)    12   6.28      7.4e-16      1.6e-15        6.0e-15
    (64,0,0)    10  12.6       1.2e-14      5.9e-15        2.6e-14
    (4,4,4)     22   4.68      1.8e-16      1.8e-15        2.1e-15
    (8,8,8)     16   0.448     4.4e-16      1.0e-15        7.7e-16

The real part is bit-identical either way, as it must be (it comes from y_l, which the upward
recurrence gets right).  The imaginary part is improved by up to a factor 11 (at (3,0,0) and
(4,4,4)) and is never worse than 5.9e-15 even without the repair, so at lambda/32 **the repair
buys the last digit of the radiative part but is not needed for 1e-13**.  It becomes necessary
when |Im/Re| falls further: that ratio scales as (kR)^3/3, so at lambda/128 and two cells
(kR = 0.098) it is 3e-4 and the unrepaired imaginary part would be at 1e-12 relative.  The
repair costs 850 ns per offset, i.e. it triples the cost at L = 10; it is worth switching off
for the offsets where |Im/Re| > 1 and on elsewhere.

## 13. Identically zero (l,m) entries (`t11_zeros.jl`, `out/t11_zeros.txt`)

Exact-rational table, L = 30, counting (l,m) entries whose whole n-column is exactly zero:

    shape                        diagonal class   xy class   xz, yz classes   total nonzero
    cube (1/32)^3                415/544 = 0.76   64/120     120/120          719/904 = 0.795
    slender (1/32,1/32,1/512)    416/544 = 0.77   64/120     120/120          720/904 = 0.796
    generic (1/32,1/17,1/23)     544/544 = 1.00   120/120    120/120          904/904 = 1.000

The square cross-section (s1 = s2) kills 20% of the table, half of the xy class in particular;
a generic box kills nothing.  Compressing the tables would buy about 20% of the per-offset cost
for cubes and nothing for a general cell shape, so it is not worth the bookkeeping.

## 14. Files

    work/famB/famB.jl        the library: geoTab (exact geometry table), qTab (per frequency),
                             farTns! (per offset, zero allocations), shrm!, hnk!, bslj!,
                             hnkFix!, farTerms (diagnostics), jbnd, hbnd, wRadAll, remBndAll
    work/famB/t1_check.jl    harmonic polynomial vs shrm!, addition theorem, Helmholtz identity
    work/famB/t2_sign.jl     sign convention pinned against the reference at (5,1,0)
    work/famB/t3_bessel.jl   h_l, j_l, Y_lm Float64 vs BigFloat
    work/famB/t4_conv.jl     convergence in l and conditioning (args: shape L nMax [bf])
    work/famB/t5_acc.jl      accuracy vs the 220-bit reference, symmetries, homogeneity
    work/famB/t6_cost.jl     flops, timings, geometry-table cost
    work/famB/t7_nmax.jl     k-series truncation in n
    work/famB/t8_agg.jl      aggregate cost over Gila's egoToe offset set
    work/famB/t9_deep.jl     the 2-cell band at L up to 100
    work/famB/t10_refchk.jl  reference self-check, ordN 44 vs 60
    work/famB/t11_zeros.jl   identically zero (l,m) entries
    work/famB/t12_fix.jl     effect of the Miller repair on the radiative part
    work/famB/ref.jl,
    work/famB/refrun.jl      220-bit reference builder; 163 tensors cached in
                             work/famB/out/refcache_{1,2,3,chk}.txt
    work/famB/out/*.txt      raw output of every table above
