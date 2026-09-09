# Family (a): Cartesian Taylor expansion about the centre separation, and item 6

Author: sub-agent famA. Work dir `notes/farfield/scratch/work/famA/`.
Code: `famA.jl` (standalone, generic in the number type). Scripts: `part1.jl` (item 6),
`p2bnd.jl` (a priori term counts), `p2ver.jl` + `p2n2.jl` (measurement vs the 220-bit reference),
`sym.jl` (divergence, overflow, symmetry, homogeneity), `gcmp.jl` (cost vs Gila), `bld.jl`/`ref.jl`
(reference cache `refcache.txt`), `tst.jl`/`chk.jl` (correctness). Outputs: `part1.txt`,
`part1cond.txt`, `part1sum.txt`, `p2bnd.txt`, `p2ver.txt`, `p2n2.txt`, `sym.txt`, `gcmp.txt`.

## 0. Sign and normalization, pinned against the reference

`chk.jl`, 220-bit reference from `pairKer` on the 36 face pairs assembled with the brief's
`srfSum!` signs. With

    T_ab(R) = (1/V_t) int_D w(delta) [(d_a d_b + delta_ab k^2) g](R + delta) d delta,
    g(r) = e^{ikr}/(4 pi f^2 r),  k = 2 pi f,  V_t = s1 s2 s3,
    D = prod[-s_i, s_i],  w = prod (s_i - |delta_i|)

the sign is **plus**, no flip. Truncating the expansion at total degree 40, in BigFloat:

    D=(5,1,0) s=1/32 f=1      max|+T - Gref|/max|Gref| = 8.03e-26   ( |-T - Gref| = 2.0 )
    D=(5,2,1) s=1/32 f=1                                 1.88e-27
    D=(4,0,0) s=1/32 f=1                                 3.23e-22
    D=(8,8,0) s=1/32 f=1+0.1i                            1.03e-40

The moment coefficient used throughout is exact:

    mu_alpha/(alpha! V_t) = 8 prod_i s_i^{alpha_i+1}/(alpha_i+2)!   (even alpha; 0 otherwise)

since mu_n(s) = 2 s^{n+2}/((n+1)(n+2)) and mu_n/n! = 2 s^{n+2}/(n+2)!.

## 1. Item 6: the two derivative routes and their conditioning

Route (i), **Hobson**: g^{(n)}(r) from r g' = (ikr-1) g, i.e.
`d[n+2] = ((ikr - 1 - n) d[n+1] + ikn d[n])/r`; then T_m = (r^{-1} d/dr)^m g from
`T_{m+1} = -((2m+1) T_m + k^2 T_{m-1})/r^2`; then

    d^alpha g(R) = sum_{2nu <= alpha} alpha!/(nu! (alpha-2nu)! 2^{|nu|}) R^{alpha-2nu} T_{|alpha|-|nu|}.

A second variant (`radTeeB`) builds T_m from the integer table b_{m+1,n} = b_{m,n-1} + (n-2m) b_{m,n}
and the g^{(n)}; it behaves the same.

Route (ii), **coupled Cartesian recurrence** (`derRec`). With u = g, w = r g, the two identities
r^2 d_j u + x_j u - i k x_j w = 0 and d_j w = i k x_j u give, on differentiating alpha times and
solving for the highest index (j = argmax_i |R_i| among i with gamma_i > 0),

    r^2 d^gamma u = i k (R_j d^alpha w + alpha_j d^{alpha-e_j} w) - R_j d^alpha u - alpha_j d^{alpha-e_j} u
                    - sum_i [ 2 alpha_i R_i d^{alpha-e_i+e_j} u + alpha_i(alpha_i-1) d^{alpha-2e_i+e_j} u ],
    d^gamma w = i k (R_j d^alpha u + alpha_j d^{alpha-e_j} u),   gamma = alpha + e_j.

Verification (`tst.jl`, 256 bits, R = (5/32, 1/32, 0), f = 1, P = 12): the two routes agree to
`max rel |rec - hob| = 3.83e-75`; the Helmholtz residual `max |sum_j d^{alpha+2e_j}g + k^2 d^alpha g|`
relative to the degree scale is `8.28e-77`; 256 vs 512 bits moves the answer at `7.1e-77`, so the
BigFloat reference is genuinely BigFloat. Both routes were then validated end-to-end against the
220-bit face-pair reference (section 0), for axis, face-diagonal and generic offsets.

### Digits lost in the individual derivatives (`part1.txt`, `part1cond.txt`)

digits lost = log10( max_{|alpha|=p} |d^alpha g_64 - d^alpha g_B| / (eps * max_{|alpha|=p} |d^alpha g_B|) ),
256-bit BigFloat reference, |alpha| = 0..32, seven kR, two frequencies, four directions.

    dir     f        kR      p=10        p=20        p=25        p=30
                             rec  hob    rec  hob    rec  hob    rec  hob
    axis    1        0.3    1.04 2.51   1.67 5.45   1.85 7.34   1.99 8.29
    axis    1        1.0    1.05 1.89   1.29 5.27   1.39 7.76   1.54 8.09
    axis    1        3.0    1.36 2.34   1.68 5.84   1.79 6.99   1.89 9.55
    axis    1       10.0    0.84 1.23   3.40 4.91   4.59 8.38   4.87 10.41
    axis    1       30.0    1.09 1.18   1.07 1.28   1.11 2.38   1.09 2.72
    axis    1      100.0    0.84 0.86   0.76 0.97   0.83 0.95   0.82 0.98
    axis    1      300.0    1.31 1.30   1.30 1.31   1.31 1.31   1.31 1.34
    face    1        1.0    0.88 1.91   1.12 4.49   1.17 5.57   1.18 7.35
    face    1       10.0    0.38 0.72   1.99 4.20   2.77 5.91   2.83 7.31
    body    1        1.0    0.40 1.52   0.66 3.69   0.85 4.62   0.88 6.38
    body    1       10.0    0.67 0.58   1.48 3.28   2.24 4.93   2.54 6.18
    gen521  1        1.0    0.32 2.50   0.38 6.13   0.38 7.45   0.60 9.30
    gen521  1       10.0    0.68 0.61   2.20 4.12   3.37 7.61   3.67 9.08
    gen521  1+0.1i   1.0    0.78 2.72   0.50 6.18   0.92 7.85   0.55 9.34
    gen521  1+0.1i  10.0    0.14 0.62   2.25 5.48   2.82 7.50   3.20 9.58

(full 56-block table with every p in `part1.txt`, all 4 dirs x 2 f x 7 kR in `part1cond.txt`.)

**Route (i) is unusable and route (ii) is fine.** Hobson loses about one digit per 3.5 orders and is
at 7.3-8.4 digits by p = 25 for kR <= 3; the loss is the alternating cancellation inside the nu-sum,
where R^{alpha-2nu} T_{p-|nu|} terms of comparable size subtract. The coupled recurrence loses
**1.4 digits at p = 25, kR = 1 on the axis** and less off-axis (1.17 face, 0.85 body, 0.38 generic).

**Answer to the design question of item 6: yes, the Float64 recursion is stable to order 25 at
kR ~ 1, provided it is route (ii); it loses 0.4-1.4 digits.** Route (i) at the same point loses
4.6-7.8 digits and must not be used.

**Where route (ii) is not stable.** The worst band is kR ~ 10 (R ~ 1.6 lambda), where the loss rises
to 4.6 digits at p = 25 and 4.9 at p = 30 (axis, f = 1); 3.4 at p = 20. This is the crossover where
the e^{ikr} phase and the algebraic 1/r^n growth are comparable and the recursion's two contributions
cancel. Below kR = 3 the loss is flat at ~1.5-2 digits; above kR = 30 it drops back to ~1 digit
(0.8-2.4 over all directions and both frequencies at every p up to 32), because the exponential term
then dominates the recursion and nothing cancels. Complex f = 1+0.1i behaves identically (within
0.3 digits everywhere). Direction matters by about a digit, axis worst, body diagonal best.

**Float64 dynamic range** is the other order limit (`sym.txt`). |d^alpha g| ~ p!/|R|^{p+3}, so the
Float64 exponent range caps p:

    |R| (lambda)  kR      largest p with all |d^alpha g| finite in Float64
    1/32          0.196    90
    2/32          0.393   100
    4/32          0.785   120
    8/32          1.571   130
    1.0           6.283   170
    4.0          25.13    230

For every separation that family (a) actually needs (p <= 36, section 4) this is not binding.

### Digits lost in the quantity that matters (`part1sum.txt`, `part1sumcond.txt`)

S_p = sum_{|alpha|=p} d^alpha g(R) mu_alpha/alpha!, cubic cell of edge s set by rho = sqrt3 s/|R|,
168 blocks over 4 directions x 2 frequencies x 7 kR x rho in {0.87, 0.43, 0.22}. Physically
ks = k sqrt3 s = rho*kR <= 2.72 (cells at most lambda/4); blocks with ks > 2.72 are unphysical and
are where the expansion itself blows up (at kR = 300, rho = 0.22 the terms grow to |S_32|/|S_0| =
8e14 before turning over, since the series only starts decaying past p ~ k|delta| = 66).

Physical region, at p = 30:

    dir     f      kR    rho    ks     |S30|/|S0|  lossS30  lossCum  canc
    axis    1      1.0   0.87   0.870   1.26e-9     4.80    -0.56    1.169
    axis    1      1.0   0.43   0.430   8.27e-19    4.23    -0.96    1.033
    axis    1      3.0   0.87   2.610   1.44e-9     3.63     0.12    1.691
    axis    1     10.0   0.22   2.200   4.76e-27    4.56     0.68    1.344
    face    1      1.0   0.87   0.870   5.91e-11    5.94    -0.40    1.100
    face    1      3.0   0.87   2.610   1.47e-10    5.36    -0.31    1.580
    body    1      1.0   0.87   0.870   3.23e-9     4.43     0.12    1.062
    body    1      3.0   0.87   2.610   3.36e-9     4.58     0.25    1.489
    gen521  1      1.0   0.87   0.870   8.95e-10    3.85    -0.83    1.118

The individual S_p lose 3.2-6.3 digits **relative to their own magnitude**, but |S_30|/|S_0| is
1e-9 to 1e-29, so the absolute error they contribute is negligible. The accumulated sum loses
**between -1.06 and +0.68 digits over the whole physical grid** (max +0.68). The reason is in the
last column: sum_p |S_p| / |sum_p S_p| = 1.00 to 1.74, i.e. **there is essentially no cancellation
between degrees**. The a priori amplification 1/(1-rho) = 7.7 at rho = 0.87 is itself a factor 4-7
pessimistic. This is the single most important conditioning result of family (a): the derivative
recursion is what could lose digits, and the moment weights kill exactly the orders where it does.

## 2. The a priori remainder bound, with explicit constants

Let z = R + w, w in C^3, u = sqrt(z.z) (principal branch continued from u(R) = |R|),
h_ab(z) = (d_a d_b + delta_ab k^2) g(z). Explicitly, with G1 = e^{iku}(iku-1)/(4 pi f^2 u^3) and
G2 = e^{iku}(3 - 3iku - k^2u^2)/(4 pi f^2 u^5),

    h_ab = delta_ab (G1 + k^2 g) + z_a z_b G2,   and by Helmholtz h_aa = -(d_b^2 + d_c^2) g.

**(i) Analyticity radius.** g is analytic wherever z.z != 0. Minimising |z.z| over |w| <= rho
(with a = R+u_r, v = Im w, |z.z|^2 = (|a|^2-|v|^2)^2 + 4(a.v)^2, the minimum taken at v perp a and
u_r anti-parallel to R) gives, exactly,

    min_{|w|<=rho} |u|  =  |R| - rho                for rho <= |R|/2,
                        =  sqrt(|R|^2/2 - rho^2)    for |R|/2 < rho < |R|/sqrt2,          (uLo)

and 0 at rho = |R|/sqrt2. So dist(R, {z.z = 0}) = |R|/sqrt2, and the closed ball of any
rho < |R|/sqrt2 is a convex, cone-free domain, hence simply connected, so u = sqrt(z.z) is
single valued and h is analytic there. Also |u| <= |R| + rho.

**(ii) The exponential.** |e^{iku}| = e^{-Im(k)|R| - Im(k(u-|R|))} <= e^{-Im(k)|R| + |k| du} with
du = |R| - sqrt(|R|^2 - (2|R|rho + rho^2)), which is provable whenever 2|R|rho + rho^2 <= |R|^2,
i.e. rho <= (sqrt2 - 1)|R| [from |xi|(2|R| - |xi|) <= |xi(xi + 2|R|)| = |2R.w + w.w|]. Beyond that
radius the code falls back on the unconditional |e^{iku}| <= e^{|k|(|R|+rho)}; for Gila's
frequencies this costs at most a factor e^{2 pi (0.1) |R|} ~ 2 (it is exactly 1 for real f).

**(iii) The constant.** With C0 = E(rho)/(4 pi |f|^2), E the bound of (ii), lo = uLo, hi = |R|+rho,
z_a = |R_a| + rho, A0 = C0/lo, A1 = C0(|k| hi + 1)/lo^3, A2 = C0(3 + 3|k| hi + |k|^2 hi^2)/lo^5,

    M_ab(rho) = z_a z_b A2                                                       (a != b)
    M_aa(rho) = min( A1 + |k|^2 A0 + z_a^2 A2 ,  2 A1 + (z_b^2 + z_c^2) A2 )     (a = b)

is an upper bound for |h_ab| on the closed ball of radius rho (the second branch is the Helmholtz
form, which is much tighter for axis offsets).

**(iv) The tail.** Inscribe the polydisc of radii r_i = s_i rho/d, d = sqrt(sum s_i^2), in the ball
(sum r_i^2 = rho^2). Multivariate Cauchy gives |d^alpha h_ab(R)| <= alpha! M_ab(rho)/prod r_i^{alpha_i}.
Only even alpha survive the moments, and mu_alpha/(alpha! V_t) * alpha! = mu_alpha/V_t, so with
tau = d/rho the degree-q contribution obeys

    |term_q| <= M_ab(rho) sum_{|alpha|=q, even} mu_alpha/(V_t prod r_i^{alpha_i})
             =  V_t M_ab(rho) * 8 w_q tau^q,    w_q = sum_{|alpha|=q even} prod_i 1/((alpha_i+1)(alpha_i+2)),

with w_0 = 1/8 and w_q <= 1/8 monotonically decreasing (w_2 = 1/16, 8w_4 = 0.283). Hence, after
total degree p (even),

    |T_ab(R) - T_ab^{(p)}(R)|  <=  V_t M_ab(rho) [ 8 sum_{q=p+2,p+4,...,Q} w_q tau^q + tau^{Q+2}/(1-tau^2) ]
                               <=  C tau^{p+2}/(1 - tau^2),   C = V_t M_ab(rho),                    (BND)

valid for every rho in (d, |R|/sqrt2), and minimised numerically over rho (`bndMin`, 600 points).
The w_q factor is a free gain of 1-2 orders at large p; the crude form C tau^{p+2}/(1-tau^2) is the
statement asked for, with C = V_t M_ab and tau = d/rho.

**(v) Convergence criterion.** (BND) is finite iff some rho in (d, |R|/sqrt2) exists, i.e.

    **sqrt(s1^2 + s2^2 + s3^2)  <  |R| / sqrt2**,      best ratio tau_min = sqrt2 d / |R|.       (CVG)

This is not an artefact of the bound: for real delta in the corner of D with |delta| > |R|/sqrt2 the
Taylor series of h about R genuinely diverges, and since w(delta) > 0 there on a set of positive
measure, the degree-by-degree integrated series diverges too. Section 5 shows this happening.

For a cubic cell (d = sqrt3 s) at offset n cells: axis needs n > sqrt6 = 2.449 (n >= 3),
face diagonal n > sqrt3 = 1.732 (n >= 2), body diagonal n > sqrt2 = 1.414 (n >= 2).
**Family (a) does not converge at the first non-touching axis offset (2,0,0) of a cubic grid.**

## 3. Term counts from the bound (`p2bnd.jl` -> `p2bnd.txt`, `p2bndcond.txt`)

Normalization: p is the smallest even total degree with (BND) <= 1e-13 * nrm, nrm =
max_ab V_t |(d_a d_b + delta_ab k^2) g|(R), the degree-0 tensor entry (it agrees with
max_ab |Gref_ab| to within a factor 1.03-1.4 at every offset measured). "-" means (BND) does not
reach the tolerance below p = 800.

    #        --- s = 1/32 ---   ---- s = 1/8 ----   ---- s = 1/4 ----     tau (all cubes)
    #  n     axis face body     axis face body      axis face body        axis  face  body
      2       -  276  112      -  300  122      -  324  136     1.2247   0.866  0.7071
      3     190   70   52    208   78   58    226   88   66     0.8165  0.5774  0.4714
      4      76   46   38     86   52   44     96   60   52     0.6124   0.433  0.3536
      6      42   32   28     48   38   34     58   46   44     0.4082  0.2887  0.2357
      8      32   26   22     38   32   30     48   42   38     0.3062  0.2165  0.1768
      16     20   18   18     28   24   24     34   30   30     0.1531  0.1083  0.0884
      32     16   14   14     22   20   20     28   26   26     0.0765  0.0541  0.0442
      64     14   12   12     18   18   18     24   24   24     0.0383  0.0271  0.0221

f = 1+0.1i differs by at most 4 in p. Cell size enters only through nrm and |k|hi: p grows by
6-10 from lambda/32 to lambda/4 at fixed n, because the tensor is smaller relative to the
derivative bound at large kR.

Slender cell (1/32, 1/32, 1/512), d = 0.044237, so (CVG) needs |R| > 0.06256 lambda:

    #  dir     n=2                n=4                n=8                n=16
    #          rho    tau    p    rho    tau    p    rho    tau    p    rho    tau    p
      e1      0.708  1.001   -   0.354  0.501  54   0.177  0.250  26   0.089  0.125  18
      e2      0.708  1.001   -   0.354  0.501  54   0.177  0.250  26   0.089  0.125  18
      e3     11.325 16.016   -   5.662  8.008   -   2.831  4.004   -   1.416  2.002   -
      d12     0.501  0.708 112   0.250  0.354  36   0.125  0.177  22   0.063  0.089  16
      d13     0.706  0.999   -   0.353  0.500  54   0.177  0.250  26   0.088  0.125  18
      d23     0.706  0.999   -   0.353  0.500  54   0.177  0.250  26   0.088  0.125  18
      d123    0.500  0.707 110   0.250  0.354  36   0.125  0.177  22   0.063  0.088  16

**The slender cell breaks family (a) along its short axis, and not marginally.** Offsets along e3
are multiples of s3 = 1/512 while the difference box still spans +-1/32 in x and y, so
rho = 0.0442/|R| = 22.6/n and (CVG) demands **n >= 33** along e3 (n > sqrt2*0.044237/(1/512) = 32.0).
At n = 16 along e3, tau = 2.00: the expansion diverges. Along e1/e2/d13/d23 the criterion is
n >= 3 (n > 2.002); along d12/d123, n >= 2. An isotropic cell-size expansion cannot see that
delta_3 only ranges over +-1/512, and this is a structural defect of family (a), not a constant.

## 4. Measured against the 220-bit reference (`p2ver.jl` -> `p2ver.txt`, `p2vercond.txt`)

Reference: `pairKer` at 220 bits on each of the 36 face pairs, ordN = 44, assembled with the
brief's `srfSum!` signs; 185 offsets cached in `refcache.txt` (13-170 s each). nrm = max_ab |Gref_ab|.
pTr = smallest even p with |G_big(p) - Gref|/nrm <= 1e-13 (truncation alone);
p64 = same with the Float64 evaluation; dL = log10(|G_64(pTr) - G_big(pTr)|/(eps*nrm));
pB = the a priori degree of section 3.

                   s=1/32 f=1        s=1/32 f=1+.1i     s=1/8  f=1         s=1/4  f=1
     dir  n     pTr p64 dL   pB   pTr p64 dL   pB    pTr p64 dL   pB   pTr p64 dL   pB
     axis 2      -   -  1.2    -    -   -  1.3    -    -   -  0.88   -    -   -  1.21   -
     axis 3     30  30 0.74  190   30  30 0.58  190   30  30 0.81  208   30  30 0.57  226
     axis 4     24  24 0.88   76   24  24 0.43   76   22  22 0.34   86   20  20 0.27   96
     axis 6     16  16 0.87   42   16  16 0.65   42   14  14 0.58   48   16  16 0.56   58
     axis 8     14  14 0.65   32   14  14 0.40   32   12  12 0.16   38   16  16 0.43   48
     axis 16    10  10 0.23   20   10  10 0.52   22   12  12 0.71   28   16  16 0.68   34
     face 2     36  36 0.66  276   36  36 0.62  276   34  34 0.99  300   34  34 0.94  324
     face 3     22  22 0.27   70   22  22 0.42   70   20  20 0.50   78   20  20 0.69   88
     face 4     16  16 0.54   46   16  16 0.25   46   16  16 0.42   52   16  16 0.48   60
     face 6     12  12 0.77   32   14  14 0.42   32   12  12 0.72   38   16  16 0.96   46
     face 8     12  12 0.31   26   12  12 0.69   26   12  12 0.32   32   16  16 0.61   42
     face 16     8   8 -0.11  18    8   8 0.39   18   12  12 0.58   24   16  16 0.89   30
     body 2     30  28 1.12  112   30  30 1.12  112   26  26 0.78  122   26  26 0.71  136
     body 3     20  20 0.77   52   20  20 0.76   52   16  16 0.65   58   18  18 0.58   66
     body 4     16  16 0.70   38   16  16 0.99   38   14  14 0.45   44   18  18 0.85   52
     body 6     10  10 0.71   28   12  12 0.67   28   12  12 0.29   34   18  18 0.90   44
     body 8     10  10 0.45   22   10  10 0.57   22   12  12 0.78   30   18  18 0.94   38
     body 16     8   8 0.48   18    8   8 0.24   18   12  12 1.09   24   18  18 1.26   30

Generic offsets: (5,1,0) pTr = 18, dL = 0.49; (5,2,1) pTr = 18, dL = 0.34 (s = 1/32, f = 1).

**(i) Actual error vs the bound.** pB/pTr = 2.0 to 7.7, settling at ~2.2 beyond six cells and
blowing up at the near separations (6.3 at axis n=3, 7.7 at face n=2). In flops that is a factor
5 to 60. Two mechanisms: the ratio and the constant. Measured per-degree ratio ratMs vs the
Cauchy tau = sqrt2 d/|R| (s = 1/32, f = 1):

     dir  n    rho     tau     ratMs   ratMs/tau  flr64     nTrm@pTr  flops@pTr
     axis 2    0.866   1.2247  0.7297    0.596    2.3e-12    -         -
     axis 3    0.5774  0.8165  0.4561    0.558    1.2e-15    4896      147186
     axis 4    0.433   0.6124  0.3332    0.544    3.0e-16    2730       82152
     axis 6    0.2887  0.4082  0.1709    0.419    9.4e-16     990       29880
     axis 8    0.2165  0.3062  0.1390    0.454    6.6e-16     720       21762
     axis 16   0.1083  0.1531  0.0600    0.392    4.5e-16     336       10206
     face 2    0.6124  0.866   0.5074    0.586    1.1e-15    7980      239760
     face 4    0.3062  0.433   0.2140    0.494    8.5e-16     990       29880
     face 16   0.0765  0.1083  0.0253    0.234    5.3e-16     210        6408
     body 2    0.5     0.7071  0.4176    0.590    9.9e-16    4896      147186
     body 4    0.25    0.3536  0.1768    0.500    1.1e-15     990       29880
     body 16   0.0625  0.0884  0.0258    0.292    7.6e-16     210        6408

So the true ratio is 0.23-0.60 of the Cauchy tau (about 0.55 tau near in, 0.25-0.40 far out; it is
below the naive rho = d/|R| at every offset, and below s/|R| for the diagonals). The Cauchy estimate
cannot see two things: the triangle weight concentrates delta near 0, and the box is anisotropic
relative to R (for an axis offset only d_1 is large and the box extent along 1 is s, not sqrt3 s).
The remaining factor is M(rho), which is minimised at a radius where uLo has already fallen well
below |R|, costing 1e2-1e3 in the constant.

**(ii) Float64 digits lost at the converged p.** dL = -0.11 to +1.84 over all 185 measured offsets, three
cell sizes, both frequencies, all directions. **p64 = pTr in every case but one** (body n=2,
s=1/32, f=1, where Float64 got there two degrees earlier by luck). The Float64 error floor is
3.0e-16 to 3.5e-15 relative to max|Gref| at every converged offset. **Float64 is sufficient
everywhere family (a) converges; no double-double, no 128-bit build path is needed.**

**(iii) Terms and flops per offset.** nTrm = 6 * #{even alpha, |alpha| <= p} tensor terms;
flops ~ 18 * nDer + 36 * nTrm complex operations (nDer = (p+3)(p+4)(p+5)/6 derivatives). At the
measured pTr, s = 1/32, f = 1: 10206 flops at 16 cells (body: 6408), 21762 at 8, 29880 at 6,
82152 at 4, 147186 at 3 axis. Measured wall time of the in-place Float64 path `farTenF!`
(Apple M3 Pro, single thread, median of 400 calls) and the comparison with Gila's current
`egoSrfFxd!` + `srfSum!` for the same offset (`gcmp.jl` -> `gcmp.txt`, cubic 1/32, f = 1, axis):

      n   quadOrd  kernel evals  gila_us   p   flops   famA_us  speedup
      2      9       236196      22985.6   36  239760   174.33   131.8x
      3      7        86436       8223.4   30  147186   108.75    75.6x
      4      7        86436       8207.0   24   82152    60.08   136.6x
      6      6        46656       2863.3   16   29880    22.12   129.4x
      8      5        22500       1353.3   14   21762    17.17    78.8x
     16      5        22500       1426.5   10   10206     7.71   185.0x
     32      4         9216        567.6   10   10206     7.83    72.5x
     64      4         9216        579.9   10   10206    16.88    34.4x

(caveat: `gila_us` is the scalar, un-batched, single-threaded call path, so it is an upper bound on
Gila's real per-offset cost; the flop ratio, 236196 complex exponentials vs 239760 complex flops at
n = 2 and 22500 vs 10206 at n = 16, is the machine-independent statement.) The prompt's target of
"a few hundred flops per offset" is **not** met by family (a) alone: 10^4 flops at 16 cells, 10^5 at
4 cells. 7.7 us per offset x 1.7e7 offsets of a 128^3 doubled grid = 130 s single-threaded, ~15 s on
12 threads. That is seconds-not-minutes, but it is 30x above the stated flop budget.

**(iv) Where Float64 fails to reach 1e-13, and why.** Only at **axis n = 2** (all three cell sizes,
both frequencies). The failure is **truncation, not cancellation**: at that offset dL = 0.88-1.3,
i.e. the Float64 sum tracks the BigFloat sum to 1e-15, and both stall at the same place. Section 5
gives the mechanism. Everywhere else the three error sources separate cleanly as
truncation(pTr) <= 1e-13, rounding <= 3.5e-15, reference <= 1.5e-63 (ordN 44 vs 60).

## 5. The divergence at the first axis offset (`p2n2.jl`, `sym.jl`)

D = (2,0,0), cubic: d/|R| = sqrt3/2 = 0.866 > 1/sqrt2 = 0.7071, so the eight corners of the
difference box (|delta| = sqrt3 s = 0.0541) lie outside the Taylor ball of radius |R|/sqrt2 = 0.0442.
The series is a divergent series whose early terms are small, and the observed behaviour is exactly
that of an asymptotic series: it descends to a floor and then turns around.
Float64, error relative to max|Gref|, s = 1/32, f = 1 (`sym.txt`):

     p     0     10      20      30      40      50      60      70      80      90     100    110
     err 2.6e-2 1.5e-4 9.7e-7 6.0e-8 7.9e-9 1.0e-9 1.5e-10 2.1e-11 3.2e-12 5.2e-13 7.7e-14 overflow

and in BigFloat (`p2n2.txt`, same offset, to p = 120) the truncation error goes
7.9e-12 (p=72), 2.3e-12 (76), 3.2e-12 (80), 6.2e-13 (88), 5.6e-14 (92), 9.4e-14 (96), 1.0e-14 (108),
1.2e-15 (112), 3.3e-15 (116), 2.2e-15 (120) - non-monotone from p = 72 onward, and it never becomes
a convergent tail. The best attainable at this offset is about **1e-15 in BigFloat at p ~ 112, and
7.7e-14 in Float64 at p = 100**, at which point 1.4e6 derivatives and 9.1e5 sum terms are needed;
past p ~ 100 the Float64 derivatives overflow (|d^alpha g| ~ p!/|R|^{p+3} > 1e308 at |R| = 1/16).
For contrast, D = (2,2,0) at the same cell (tau = 0.866 < 1) is genuinely convergent and reaches
5.5e-14 at p = 36, 7.7e-17 at p = 48, 1.2e-22 at p = 72 and 8.5e-34 at p = 120.

**So family (a) reaches 1e-13 at (2,0,0) only by accident of where the asymptotic floor sits, at a
cost of ~3e6 flops, and it cannot be made better by more terms.** Any production use must either
subdivide the difference box there (family (d): halving each edge takes d to sqrt3 s/2 and
d/|R| to 0.433 < 0.707, restoring convergence with 8 sub-boxes) or hand (2,0,0) to another family.
This offset is not a corner case: on a cubic grid it is the six nearest non-touching neighbours,
and by (CVG) the same happens for every offset with sqrt(sum s_i^2) >= |R|/sqrt2.

## 6. Symmetries and homogeneity (`sym.jl` -> `sym.txt`)

Because T_ab depends on R only through the even function u(z) and the monomials R_a R_b, the two
symmetries hold **identically in the implementation**, not to within a tolerance. Verified at 220
bits, f = 1+0.1i, s = 1/32, p = 30, max over the nine entries of |T(sigma R) - sigma_a sigma_b T(R)|/nrm:

     D           -D        e1-refl   e2-refl   e3-refl
     (5,2,1)     0.00e+00  0.00e+00  0.00e+00  0.00e+00
     (4,0,0)     0.00e+00  0.00e+00  0.00e+00  0.00e+00
     (3,3,0)     0.00e+00  0.00e+00  0.00e+00  0.00e+00
     (2,2,2)     0.00e+00  0.00e+00  0.00e+00  0.00e+00
     (16,4,1)    0.00e+00  0.00e+00  0.00e+00  0.00e+00

(exact zeros, in Float64 too). Family (a) therefore needs only the offsets with non-negative
components, which is the factor 8 of groundwork item 2.

Homogeneity: T(cR, cs, f/c) = c^2 T(R, s, f). Tested against the reference with c = 4, i.e.
Gref(D, s = 1/8, f = 1) vs 16 * T(D, s = 1/32, f = 4) - a genuine cross-check, since the two sides
use different cells, different frequencies and (on the left) the independent face-pair evaluator:

     D           rel dev (BigFloat, p=40)   rel dev (Float64, p=40)
     (4,0,0)     1.87e-22                   1.03e-15
     (3,3,3)     1.79e-26                   1.22e-15
     (6,6,0)     3.66e-36                   2.00e-15
     (8,8,8)     9.42e-47                   8.81e-16
     (16,0,0)    1.34e-47                   1.34e-15

Reference-side check of the same two symmetries (`symref.jl`; the negative offsets have their own
220-bit references, built independently through `facePair`):

     D             |mine(D) - Gref(D)|/nrm    |Gref(D) - sig_a sig_b Gref(D0)|/nrm
     (-5,1,0)      4.39e-24                   5.77e-64
     (5,-1,0)      4.39e-24                   2.70e-64
     (-5,-1,0)     4.39e-24                   6.45e-64
     (-5,-2,-1)    6.06e-25                   1.73e-63
     (5,2,-1)      6.06e-25                   1.73e-63

## 7. The far plateau: p saturates in kR and is set by k s, not by rho

Measured pTr at n = 32 and 64 (references built for this; `p2ver.txt`):

     s      dir    n=16  n=32  n=64    rho(n=64)  ratMs(n=64)  ks = k s
     1/32   axis    10     8     8      0.0271     0.0248       0.196
     1/32   face     8     8     8      0.0191     0.0267       0.196
     1/32   body     8     8     8      0.0156     0.0278       0.196
     1/8    axis    12    12    12      0.0271     0.0696       0.785
     1/8    body    12    12    12      0.0156     0.0890       0.785
     1/4    axis    16    16    16      0.0271     0.1196       1.571
     1/4    body    18    18    18      0.0156     0.1416       1.571

**Beyond about 16 cells the measured per-degree ratio stops following rho and saturates at a value
set by the cell size in wavelengths** (ratMs -> 0.025, 0.072, 0.13 for s = 1/32, 1/8, 1/4;
ratMs/ks = 0.14, 0.09, 0.08). The reason is that once kR >> 1 the tensor is dominated by the
radiative 1/R term e^{ikR}, whose delta-expansion has ratio k|delta|, not |delta|/|R|. Consequence:
**the far field costs a constant number of terms per offset, independent of separation**:
p = 8 (210 terms, 6408 flops) at lambda/32, p = 12 (504, 15264) at lambda/8, p = 16-18
(990-1320, 29880-39798) at lambda/4. The Float64 floor at that plateau is 2e-16 to 2e-15 at
lambda/32 and lambda/8, and 4e-15 to 1.5e-14 at lambda/4, n = 64, body (dL = 1.84) - still under
1e-13 but with only one decade of margin, and that is the worst Float64 case found anywhere.

## 8. The slender cell (1/32, 1/32, 1/512), measured

References built for six directions x n in {2,4,8,16}, f = 1 and 1+0.1i (`bld.jl C`).
d = 0.044237, and (CVG) is |R| > 0.06256.

     dir       n=2            n=4            n=8            n=16
              tau  pTr  dL   tau  pTr  dL   tau  pTr  dL   tau  pTr  dL
     e1      1.001  52  0.39 0.501 20 0.41 0.250 12 -0.05 0.125  8  0.31
     e2      1.001  52  0.44 0.501 20 0.41 0.250 12 -0.16 0.125  8  0.31
     d12     0.708  34  0.28 0.354 18 0.28 0.177 12  0.58 0.089  8  0.04
     d13     0.999  56  1.10 0.500 20 0.18 0.250 12  0.12 0.125  8  0.57
     d23     0.999  56  1.10 0.500 20 0.13 0.250 12  0.12 0.125  8  0.57
     d123    0.707  34  0.48 0.354 18 0.15 0.177 12  0.28 0.089  8  0.18
     e3     16.02   -    -   8.008  -   -   4.004  -   -   2.002  -   -

Float64 floors 1.0e-16 to 3.0e-15; dL <= 1.10. So in the long directions the slender cell is
**easier** than the cube at the same n (pTr = 20 vs 24 at n = 4 axis), because d is smaller.
At n = 2 along e1/e2/d13/d23 tau is 0.999-1.001, exactly on the convergence boundary, and the
measured behaviour is the asymptotic one: it descends to 6e-16 around p = 52-56 (6.6e5 to 8.1e5
flops) and then stops improving. That is usable but 40x the cost of n = 4.

**Along the short axis e3 family (a) is dead** for every separation Gila uses. (CVG) needs
n > sqrt2 * 0.044237/(1/512) = 32.0, i.e. n >= 33, |R| > 0.0626 lambda. References were built at
n = 8, 16, 33, 64 along e3 (the near ones cost 100-200 s each; `bld.jl F`) and the criterion is
confirmed to the offset:

     n(e3)   |R|      tau      best |G_64 - Gref|/nrm    pTr   flops
       8     0.01562  4.0039   3.9    (390 %)             -      -
      16     0.03125  2.0020   0.95   (95 %)              -      -
      33     0.06445  0.9706   1.6e-14                   58    893358
      64     0.125    0.5005   7.5e-16                   22     65754

At n = 8 the sum diverges by 18 orders (dL = 18.15). The difference box is 64 times longer in x and
y than the separation. No amount of terms fixes this; the cure has to be anisotropic (family (c),
one exact axis) or subdivision by a factor >= 32 in x and y (family (d), 1024 sub-boxes,
not viable).

## 9. How pessimistic the bound is, and why

Splitting pB/pTr into the part explained by the ratio (log(1/ratMs)/log(1/tau)) and the residue,
which is the constant C = V_t M_ab(rho) (s = 1/32, f = 1):

     dir  n   tau     ratMs   pB/pTr  from ratio  from constant
     axis 3   0.8165  0.4561    6.33       3.87        1.64
     axis 4   0.6124  0.3332    3.17       2.24        1.41
     axis 6   0.4082  0.1709    2.62       1.97        1.33
     axis 8   0.3062  0.1390    2.29       1.67        1.37
     axis 16  0.1531  0.0600    2.00       1.50        1.33
     face 2   0.866   0.5074    7.67       4.72        1.63
     face 4   0.433   0.2140    2.88       1.84        1.56
     face 16  0.1083  0.0253    2.25       1.65        1.36
     body 2   0.7071  0.4176    3.73       2.52        1.48
     body 4   0.3536  0.1768    2.38       1.67        1.42
     body 16  0.0884  0.0258    2.25       1.51        1.49

The constant costs a steady factor 1.33-1.64 in p. The ratio is the whole story near in, and it is
irreducible within a Cauchy framework: tau = sqrt2 d/|R| is the true radius of convergence of the
Taylor series at the worst corner of the box, and the bound must respect it, while the actual
integral only feels that corner through the triangle weight, which vanishes there to third order.

I tried to recover the anisotropy by replacing the ball with a Cartesian polydisc of radii r_i.
The sharp min-modulus over such a polydisc is, decomposing w = w_par Rhat + w_perp (an orthogonal
splitting in the Hermitian norm because Rhat is real),

    min |u|^2 = |R|^2 - 2|R| t* + 2 t*^2 - sum_i r_i^2,   t* = min( sum_i |R_i| r_i/|R| , |R|/2 ),

which for an axis offset reduces to the elementary (|R| - r_1)^2 - r_2^2 - r_3^2 and does let r_1
grow well past the isotropic radius. It buys nothing for these shapes: the tail needs x_i = s_i/r_i
< 1 in every axis, so r_perp > s always, and for a cubic cell at n = 4 the best achievable
max_i x_i is 0.69-0.77 against the isotropic tau = 0.61. The polydisc route is therefore recorded
as tried and not useful here. Closing the remaining factor needs a bound that uses the weight
(a Cauchy estimate on the weighted integral, not on the integrand), which I did not obtain.

## 10. Where family (a) stands, separation band by separation band

Cubic cells, the tolerance is 1e-13 relative to max_ab |Gref_ab|, Float64, measured:

     band                          status
     touching (n = 1)              out of scope (the delta term and the singularity are present)
     (2,0,0) and axis n = 2        **FAILS**. tau = 1.2247 > 1: the expansion diverges. Best
                                   attainable 7.7e-14 in Float64 at p = 100 (3e6 flops, at the edge
                                   of Float64 overflow) and 1.2e-15 in BigFloat at p = 112, both
                                   asymptotic floors, not convergence. Needs subdivision (2^3
                                   sub-boxes take tau to 0.61) or another family.
     (2,2,0), (2,2,2)              works: p = 26-36, 3.4e5-2.4e5 flops, floor 3e-16..2e-15.
                                   Expensive but provably convergent (tau = 0.87, 0.71).
     (3,0,0)                       works: p = 30, 1.5e5 flops, floor 1.2e-15 at all three scales.
     n = 3 diagonals               p = 16-22, 3e4-6.6e4 flops.
     n = 4                         p = 14-24, 2.2e4-8.2e4 flops.
     n = 6, 8                      p = 10-18, 1.0e4-4.0e4 flops.
     n >= 16                       p = 8 (lambda/32), 12 (lambda/8), 16-18 (lambda/4), constant in
                                   n; 6408, 15264, 29880-39798 flops; 7.7 us per offset.
     Float64 everywhere it converges: 0.11-1.84 digits lost, floor 3e-16 to 1.5e-14.

Slender (1/32, 1/32, 1/512):

     e1, e2, d13, d23 at n = 2     boundary case tau = 0.999-1.001; floor 6e-16 at p = 52-56
                                   (6.6e5-8.1e5 flops). Usable, barely, and not provable.
     e1, e2, d12, d13, d23, d123   n >= 4: p = 8-20, easier than the cube at the same n.
     e3 at every n <= 32           **FAILS structurally**, tau = 32/n; needs n >= 33.

Frequency: f = 1 and f = 1+0.1i are indistinguishable (pTr differs by at most 2, dL by at most 0.4)
and the exact homogeneity T(cR, cs, f/c) = c^2 T(R, s, f) was verified against the reference to
1e-22..1e-47 (BigFloat) and 1e-15 (Float64), so a single geometry table serves every frequency
after rescaling.

## 10b. Subdivision repairs (2,0,0), at a price (`farTenSub`, `sub.jl` -> `sub.txt`)

Splitting the difference box into its 8 half-boxes and expanding each about its own centre. The
restricted triangle weight has closed-form moments about the half-box centre, h = s/2,

    m_n(sigma) = int over the half of (h -+ v) v^n dv = 2 h^{n+2}/(n+1)  (n even)
                                                     = -sigma 2 h^{n+2}/(n+2)  (n odd),

so odd degrees now contribute and the term count per degree triples. The nearest sub-centre is at
|R + sigma s/2| = 1.658 s, so tau drops from 1.2247 to **0.7385** and the expansion is provably
convergent. Measured at D = (2,0,0), s = 1/32, f = 1, against the 220-bit reference:

     p       0      8      16      24      32      36      44      52      60
     eBig  1.3e-1 1.5e-4 1.6e-7  1.5e-9  3.5e-12 5.3e-14 1.7e-15 1.4e-18 3.2e-20
     e64   1.3e-1 1.5e-4 1.6e-7  1.5e-9  3.4e-12 5.5e-14 2.4e-15 2.7e-15 3.1e-15

pTr = 36 at s = 1/32 and 1/8 (both frequencies), 34 at s = 1/4; Float64 floor 2.4e-15, rounding
error 2.5e-15, i.e. again no digits lost. **This closes the one cubic band where family (a) fails**,
but at 8 x (18 x 10660 + 36 x 54834) = 1.7e7 flops per offset - 2700x the cost of a far offset at
lambda/32, and 6x the cost of the (divergent) plain expansion pushed to its floor. At lambda/32
that band is 6 offsets out of ~1.7e7, so the total cost is irrelevant; at lambda/4, where the
near-field k-series does not reach, it is the whole transition band and 1.7e7 flops per offset is
not acceptable. A 3^3 split gives tau = 0.50 and would need p ~ 24, roughly 27 x 4e6 = 1.1e8: worse.
Subdivision is a correctness fix, not a cost fix.

## 11. Honest summary

What family (a) settles:
- The **coupled Cartesian recurrence** (route ii) is the right derivative engine: it is stable to
  order 30 in Float64 (worst case 4.9 digits at kR = 10, typically 1-2), whereas the Hobson route
  loses 7-10 digits by order 25-30 and must be discarded. This answers item 6.
- The **quantity that matters loses under one digit**: the accumulated sum loses -1.06 to +0.68
  digits over the whole physical (kR, rho, direction, frequency) grid, because
  sum_p |S_p|/|sum_p S_p| = 1.00-1.74. Float64 suffices; no higher-precision build path is needed.
- **A proven remainder bound** (BND) with explicit constants and the exact convergence criterion
  sqrt(sum s_i^2) < |R|/sqrt2, confirmed by the measured divergence at (2,0,0) and the measured
  convergence at (2,2,0).
- The **sign and normalization** of the volume form against Gila's `srfSum!`: plus, verified to
  8e-26 at (5,1,0) and 1e-40 at (8,8,0).
- **Exact symmetry** under offset -> -offset and the three axis reflections, in the implementation
  and on the reference side (5.8e-64), so only the non-negative octant of offsets is needed.

What family (a) does not settle:
- **(2,0,0) on a cubic grid**, i.e. the six nearest non-touching neighbours, where the expansion
  provably diverges. Subdivision into 2^3 sub-boxes is the obvious repair (tau 1.2247 -> 0.6124,
  cost 8 x p ~ 24) but I did not measure it.
- **The short axis of the slender cell** at every separation Gila uses. Isotropic cell-size
  expansion cannot work there; this is family (c)'s or (d)'s problem.
- **The flop budget.** 6408 flops at lambda/32 far offsets is 20-30x the "few hundred" the task
  asks for, although it is 3-200x below the current fixed Gauss rules (7.7 us vs 570-1400 us per
  offset in the scalar call path). The p^3 derivative table is the cost; only about
  (p/2)^2/2 x 6 of those derivatives are actually used, so a route that builds only the needed
  even-plus-two multi-indices would cut it, and I did not build one.
- **The bound is 2.0-7.7x too pessimistic in p** (5-60x in flops), so a priori band selection from
  (BND) alone would cost several times what the method actually needs. The gap decomposes into
  1.5-4.7 from the convergence ratio and 1.3-1.6 from the constant; the polydisc sharpening I
  derived does not close it.

## 12. Deliverables and how to reproduce

All under `notes/farfield/scratch/work/famA/`, run as
`JULIA_NUM_THREADS=1 julia --startup-file=no --project=$ENV <script>.jl`.

- `famA.jl` - the implementation, standalone, generic in the number type, constants typed.
  `radDer`, `radTee`, `radTeeB`, `derHob` (route i); `derRec`, `derRec!` (route ii);
  `farTen` (cumulative by degree, both routes), `farTenF!` (in-place, allocation-free after the
  workspace), `farTenSub` (2^3 subdivision); `uLo`, `expBnd`, `supH`, `wgtSeq`, `tailSum`,
  `bndEnt`, `bndMin`, `bndTen`, `bndOrd` (the bound); `leadTen`, `cost`, `momCof`, `subMom`.
- `ref.jl` + `bld.jl` - 220-bit reference builder and the text cache `refcache.txt`
  (185 offsets: sets A cubic 1/32, B cubic 1/8 and 1/4, C slender, D negative offsets,
  E n = 32 and 64, F slender short axis). `bld.jl <set>`; each offset is 13-200 s.
- `tst.jl` (route agreement, Helmholtz residual, precision sanity), `chk.jl` (sign),
  `fin.jl` (all three evaluation paths against the reference, final check: 1.9e-15 Float64,
  1.0e-21 BigFloat, 4.2e-26 subdivided).
- `part1.jl` -> `part1.txt`, `part1sum.txt`; condensed by hand into `part1cond.txt`,
  `part1sumcond.txt` (item 6).
- `p2bnd.jl` -> `p2bnd.txt`, `p2bndcond.txt` (a priori term counts).
- `p2ver.jl` -> `p2ver.txt`, `p2vercond.txt`; `p2n2.jl` -> `p2n2.txt` (measurement).
- `sym.jl` -> `sym.txt`, `symref.jl` (divergence, overflow, symmetry, homogeneity, timing).
- `gcmp.jl` -> `gcmp.txt` (cost against Gila's `egoSrfFxd!`).
- `sub.jl` -> `sub.txt` (subdivision at (2,0,0)).

Final end-to-end check (`fin.jl`), all three code paths against the 220-bit reference:

    D=(5,2,1) s=1/32  f=1     p=30:  farTenF!(Float64) 1.88e-15  farTen(Big) 1.03e-21  farTenSub 4.20e-26
    D=(4,4,0) s=1/8   f=1+.1i p=24:  farTenF!(Float64) 2.29e-16  farTen(Big) 5.49e-19  farTenSub 1.34e-22
    D=(8,0,8) slender f=1     p=20:  farTenF!(Float64) 2.86e-16  farTen(Big) 1.12e-20  farTenSub 7.29e-24
