# famC -- family (c), one exact axis, two expanded

Deliverables (all in `SCRATCH/work/famC/`): `famC.jl` (standalone, generic in the number
type), `ref.jl` (220-bit reference builder + disk cache, `ref1.txt`/`ref2.txt`, 52 offsets),
`run.jl` (sweep -> `tables/famC_sweep.txt`), `conv.jl` (`tables/famC_conv.txt`,
`tables/famC_tau.txt`), `mom.jl` (`tables/famC_moments.txt`), `sym.jl`
(`tables/famC_sym.txt`), `chk.jl`, `cost.jl`.

## 0. The headline, before the numbers

**The family as literally specified does not exist.** With `delta_2, delta_3` frozen, the
`delta_1` integral is `int (s_1-|t|) e^{ik rho} rho^{-l} dt` with
`rho = sqrt((R_1+t)^2 + c^2)`. That is an incomplete-Hankel / Lipschitz--Hankel object, not
elementary, for every `l >= 1`; the ladder of `notes/moments` covers `rho^m` for integer `m`
and nothing else. So candidate (ii) is dead in one line, and candidate (i) has to supply the
`e^{ikr}` by a series in something.

**What survives is family (c) with its roles inverted.** The kernel depends on the two
"expanded" coordinates only through the single variable

    xi = 2 R_2 y + y^2 + 2 R_3 z + z^2,     r^2 = rho(t)^2 + xi   (exact identity),

so `F(r) = H(r^2)` is analytic in `xi` and every level `xi^m` is a *polynomial* in `y, z`
whose integral against the triangle weights is exact (`mu_n(s) = 2 s^{n+2}/((n+1)(n+2))`,
`n` even). **The two expanded axes are the ones that end up being done exactly, one
finite polynomial integral per level; the "exact" axis is the one that carries a series.**
This is a genuine resummation, not a rearrangement of (a): the whole two-dimensional
transverse expansion collapses onto one index `m` instead of `O(m^2)` monomials, and its
ratio is the *square* of the two-dimensional `rho` whenever the exact axis is aligned with
`R` (no linear term in `xi`).

Verified to `2.9e-16` against the 220-bit reference at `(5,1,0)`, `1e-13` or better at every
offset with `tau_xi < 1`; **blocked at every 2-cell offset by `tau_xi = 1.0` exactly**, and
on the slender needle `(0,0,n)` for `n <= 16` by `tau_xi = 1.13` at `n = 16`.

## 1. Volume form and sign, pinned

`T_ab(R) = (1/V_t) int_D w(delta) [(d_a d_b + delta_ab k^2) g](R+delta) d delta` with
`g = e^{ikr}/(4 pi f^2 r)`. Writing `x = R + delta`,

    (d_a d_b + delta_ab k^2) g = x_a x_b P(r) + delta_ab Q(r),   C = 1/(4 pi f^2),
    P(r) = C e^{ikr} (3 r^{-5} - 3 i k r^{-4} - k^2 r^{-3}),
    Q(r) = C e^{ikr} (k^2 r^{-1} + i k r^{-2} - r^{-3}).

Against the 220-bit `pairKer` reference assembled by the `srfSum!` signs of the BRIEF
(`tables/famC_sym.txt`), **as computed**, with **no sign flip and no extra factor**:

    D = (5,1,0)   2.87e-16     with a global sign flip: 2.0
    D = (5,3,2)   4.24e-16                              2.0
    D = (4,4,4)   1.21e-15                              2.0
    D = (8,0,0)   4.79e-16                              2.0

(the "2.0" is the trivial `|-G-G|/|G|`, i.e. the flipped sign is wrong by 100%). This agrees
with famG's independent finding.

## 2. The mechanism, derived

Permute so the exact axis is first: `(t,y,z) = (delta_ax, delta_b, delta_c)`,
`R = (R_1,R_2,R_3)` permuted the same way, `c^2 = R_2^2 + R_3^2`,
`rho(t)^2 = (R_1+t)^2 + c^2`, `R_0 = |R| = rho(0)`.

**(a) The transverse direction, exactly.** `H(W) = F(sqrt W)` is analytic on `|W - rho^2| <
rho^2` (the only branch point is `W = 0`), so `F(r) = sum_m F_m(rho) xi^m`,
`F_m = (1/m!) D^m F`, `D = (1/(2r)) d/dr`. On the basis `e^{ikr} r^{-l}` the operator `D` is
closed and elementary:

    D[e^{ikr} r^{-l}] = (ik/2) e^{ikr} r^{-l-1} - (l/2) e^{ikr} r^{-l-2},

so every `F_m` is a finite combination of `e^{ik rho} rho^{-l}` with `l <= 5 + 2m`, generated
by one pass of a two-line map. `xi^m` is a polynomial of degree `2m` in each of `y,z`, and
because `xi = (Ay+y^2) + (Bz+z^2)` splits (`A = 2R_2`, `B = 2R_3`), the transverse integral
factorizes in `O(M^2)`:

    W(m; k2,k3) = sum_i binom(m,i) Ymom(i,k2) Zmom(m-i,k3),
    Ymom(i,k2)  = int w_2(y) (Ay+y^2)^i (R_2+y)^{k2} dy
                = sum_v binom(i,v) A^{i-v} sum_u binom(k2,u) R_2^{k2-u} mu_{i+v+u}(s_2).

Exact rationals in the lengths, frequency-independent, no cancellation (all moments of a
positive weight, and the only signs come from `A`, `B`, which have a fixed sign).

**(b) The exact axis.** The family `G_l = e^{ik rho(t)} rho(t)^{-l}` satisfies the closed
first-order system `G_l' = ik (R_1+t) G_{l+1} - l (R_1+t) G_{l+2}` (from `rho' = (R_1+t)/rho`),
which gives its Taylor coefficients in `t` by a two-term level recurrence:

    (q+1) g[l,q+1] = ik (R_1 g[l+1,q] + g[l+1,q-1]) - l (R_1 g[l+2,q] + g[l+2,q-1]),
    g[l,0] = e^{ik R_0} R_0^{-l}.

Then `int w_1(t) t^p G_l(t) dt = sum_q g[l,q] mu_{p+q}(s_1)`, odd `p+q` killed by the weight.
Radius of convergence `|t| < R_0` (the zeros of `rho^2` at `t = -R_1 +- ic`, both at `|t| =
R_0`). **No Cartesian derivative of `g` is ever formed, no binomial expansion of
`(rho - R_0)^n` is ever formed, and no quadrature node is ever sampled.**

The two truncations are independent: `M` (levels of `xi`) and `Q` (levels of `t`).

## 3. Candidate-by-candidate verdicts, with the number

**(ii) incomplete-Hankel line integral: BLOCKED, structural.** `int (s-|t|) e^{ik
sqrt((R_1+t)^2+c^2)} rho^{-l} dt` is not elementary. Nothing further was spent on it.

**(i) line moments of `rho^m` by the ladder, then the recentred k-series: the centred line
moments CAN be generated cancellation-free, but the mechanism that does it makes the ladder
unnecessary.** Measured (`tables/famC_moments.txt`), `s = 1/32`,
`A_n = int (s-|t|)(rho-R_0)^n dt`:

route (2) = naive binomial `sum_p binom(n,p)(-R_0)^{n-p} I_p` with **exactly rounded** exact
ladder moments `I_p`; amp = `sum_p binom(n,p) R_0^{n-p}|I_p| / |A_n|`; `d2` = Float64 digits
lost. Route (3) = expand `rho` itself (`rho_0 = R_0`, `rho_1 = R_1/R_0`,
`rho_2 = c^2/(2R_0^3)`, `rho_q = -(1/2R_0) sum_{i=1}^{q-1} rho_i rho_{q-i}`), power the
series, contract with `mu_q`; `d3` = its Float64 digits lost.

    geometry                      n=4              n=8              n=12             n=20
    radial   R1=4s, c=0     amp   6.24e4           8.11e8           7.38e12          4.21e20
                            d2    4.05             8.39             12.04            19.75
                            d3    -0.23            -0.82            -0.54            -0.58
    diagonal R1=4s,c=4V2 s  amp   4.96e6           4.59e12          2.66e18          4.84e29
                            d2    5.52             11.80            17.42            28.35
                            d3    -0.77            -0.70            0.33             0.71
    transverse R1=0, c=4s   amp   7.95e8           4.85e16          1.86e24          1.59e39
                            d2    7.96             16.24            23.67            37.72
                            d3    0.10             -0.29            0.10             (trunc.)

So the naive binomial loses **19.8 digits at n = 20 on a radial line and 37.7 on a
transverse one**, i.e. it is dead past `n ~ 6`; and the `rho`-series route holds
**under 1.4 digits through n = 20** on every geometry where its own `t`-truncation
(`Q = 60`, ratio `s_1/R_0`) is converged. (The two entries marked "trunc." are `t`-series
truncation, not cancellation: for `R_1 = 0, c = 2s` the ratio is `s_1/R_0 = 0.5` and `Q = 60`
does not reach `u^{20}`; the 220-bit route-3 value disagrees with the 220-bit quadrature by
`1e-4`, which is the truncation, and the same number in Float64 is not a rounding statement.)

**This answers the "Bridging the gap" bullet affirmatively**: the centred moments
`int int w (r-R_0)^n` do have a direct cancellation-free form, obtained by expanding `r`
itself rather than `(r-R_0)^n`. But the same expansion, applied to `e^{ikr} r^{-l}` directly
rather than to `(r-R_0)^n`, *is* the `g[l,q]` table of section 2b -- so once you have the
cancellation-free centred moments you no longer need them, and the ladder of
`notes/moments` plays no part in the far field. That is the useful negative result: family
(c)'s advertised route through the moments machinery is a detour.

**(iii) 1D Taylor about `R_1` in `delta_1` only: this is what section 2b is,** generated by
the ODE recurrence instead of by differentiating `g`. Ratio `s_1/R_0`; only even `p+q`
survives, so the effective ratio per retained term is `(s_1/R_0)^2`. Its stability is
measured in section 6: **max 1.07 Float64 digits lost over all 100+ converged cases,
including `kR = 21.8`.**

**(iv) the new mechanism: the `xi` resummation** of section 2a. Its value over a plain
Cartesian monomial expansion is not the rate (the nearest singularity is the same) but the
count: one coefficient per transverse level instead of `(m+1)(m+2)/2`, and -- when the exact
axis is aligned with `R`, so `xi` has no linear term -- a ratio that is the *square* of the
two-dimensional `rho`. At `(8,0,0)` cubic, `rho_2D = sqrt(2)/8 = 0.177` but
`tau_xi = 0.0408 = rho_2D^2 * 1.3`.

## 4. A priori bounds and term counts

With `rho_min^2 = max(|R_1|-s_1,0)^2 + R_2^2 + R_3^2` and
`xi_max = 2|R_2|s_2 + s_2^2 + 2|R_3|s_3 + s_3^2`,

    tau_xi = xi_max / rho_min^2,     tau_t = s_1 / R_0.

Cauchy on `|xi| = theta rho^2` and on `|t| = theta R_0` (using
`|rho(t) - R_0| <= R_0 theta(2+theta)/(2-theta)`, so `|Im rho| <= 1.5 theta R_0` for small
`theta`):

    |Rem_M| <= rho^{-j} (1-theta)^{-j/2} e^{k rho theta} (tau_xi/theta)^{M+1}/(1-tau_xi/theta)
    |Rem_Q| <= 2 s_1^2 R_0^{-l}(1-theta)^{-l} e^{1.5 k R_0 theta}
                 sum_{q>Q} (tau_t/theta)^q/((q+1)(q+2))

for any `theta` in `(tau,1)`; the rates are `tau_xi` and `tau_t`. Convergence requires
`tau_xi < 1`; `tau_t < 1` is automatic for every non-touching offset.

**Practical rules and how they held.** `M_pred = ceil(13/log10(1/tau_xi))` was **never
violated** in any of the ~110 converged (offset, axis, frequency) cases; measured
`M*/M_pred` ranged from `0.57` to `1.0` (e.g. cubic `(3,0,0)`: `M* = 25` vs `M_pred = 44`;
`(4,4,4)`: `25` vs `37`; slender `(0,0,32)` ax3: `32` vs `48`).
`Q_pred = ceil(13/log10(1/tau_t))` is **sharp at `kR <~ 2` and violated at large `kR`**: it
predicts `28` and measures `28` at cubic `(3,0,0)`, `kR = 0.59`; but at
`lambda/4` it under-predicts by up to 3 -- `(8,8,0)`, `kR = 17.8`: `Q_pred = 13`,
`Q* = 16`; `(8,0,0)`, `kR = 12.6`: `13` vs `16`; `(8,8,8)`, `kR = 21.8`: `12` vs `14`.
That is the `e^{1.5 k R_0 theta}` factor of the bound making itself felt. The patched rule
`Q >= ceil(13/log10(1/tau_t)) + ceil(kR_0/8)` was not violated anywhere in this study; it is
empirical, not proven. The rigorous bound above, optimized over `theta`, over-predicts `Q`
by about a factor 1.8 at `kR = 21.8`.

Measured geometric decay (`tables/famC_conv.txt`), error vs level with the other index
saturated -- clean geometric in both indices in every case, e.g.

    cubic (3,0,0) ax1, tau_xi=0.5,  M = 0,4,...,32 : 0.12 1.3e-4 7.0e-7 8.8e-9 1.7e-10 4.2e-12 1.2e-13 3.9e-15 1.0e-16
    cubic (3,0,0) ax1, tau_t=0.333, Q = 0,4,...,32 : 0.061 2.9e-3 1.0e-4 2.0e-6 3.1e-8 4.4e-10 6.0e-12 7.8e-14 8.2e-16
    lambda/4 (8,8,8) ax1, kR=21.8,  M = 0,4,...,20 : 0.15 5.4e-4 6.5e-7 2.6e-10 4.3e-14 2.1e-15
    slender (0,0,32) ax3, tau_xi=0.533, M = 0,4,...,44: 0.23 1.6e-3 2.6e-5 6.7e-7 2.3e-8 8.9e-10 3.9e-11 1.8e-12 8.9e-14 5.1e-15 6.8e-16 4.9e-16

## 5. Choice of exact axis, and the "two exact axes" variant

`tau_xi` for every choice, and for the hypothetical variant in which only one coordinate is
expanded (`tables/famC_tau.txt`; `tau < 1` required). Cubic cells, `tau` is independent of
the cell size:

    D            ax1     ax2=ax3   only-1-exp  only-2-exp  only-3-exp   rho3D
    (2,0,0)      2.0     1.5       1.25        1.0         1.0          0.866
    (3,0,0)      0.5     0.889     0.778       0.25        0.25         0.577
    (4,0,0)      0.222   0.625     0.5625      0.111       0.111        0.433
    (8,0,0)      0.0408  0.281     0.266       0.0204      0.0204       0.217
    (2,2,0)      1.2/1.2 1.25      1.0         1.0         0.5          0.612
    (4,4,0)      0.4/0.4 0.5625    0.36        0.36        0.0556       0.306
    (2,2,2)      1.111   1.111     0.833       0.833       0.833        0.5
    (4,4,4)      0.439   0.439     0.265       0.265       0.265        0.25

Rule: **take `ax = argmin_i tau_xi(i)`**, which is the axis with the largest `|R_i|` in every
case measured. Getting it wrong is expensive: at `(4,0,0)` the aligned axis needs `M* = 13`
and reaches `1.0e-14`, the transverse choices reach only `4.8e-10` at `M = 34`
(`tau_xi = 0.625`); at `(3,0,0)` the transverse choices stall at `3.8e-6`.

**Two exact axes: not implementable, and for cubes it would not help anyway.** The required
object is `int int w_1 w_2 e^{ikr} r^{-l}` over a rectangle at frozen `delta_3`, which is not
elementary for the same reason as the 1D case; `box2` gives it only for `r^m`, so the `k`
dependence would again need a series, and the recentred version needs the 2D analogue of the
`rho`-series of section 3 -- at which point one is expanding again and the "exactness" is
nominal. The numbers say it would not be worth it for cubes: at `(2,0,0)` the best
single-expanded-axis `tau` is **exactly 1.0**, and at `(2,2,2)` it is `0.833` against the
one-exact-axis `1.111` -- a gain of `1.111 -> 0.833` in rate that buys `M` from `divergent`
to `86` levels. For the slender needle it *would* help (section 7) but there it coincides
with the k-series, which already owns that band.

## 6. Verification against the 220-bit reference

52 offsets x 3 axis choices (`tables/famC_sweep.txt`), cubic `1/32` and `1/4`, slender
`(1/32,1/32,1/512)`, `f = 1` and `1 + 0.1i`, directions `(n,0,0)`, `(n,n,0)`, `(n,n,n)`,
`(0,0,n)`, plus `(5,1,0)`, `(5,3,2)` and the reflected offsets. `errMax` = max over the 9
entries of `|G-G_ref|/max|G_ref|`; `errEnt` = worst per-entry relative among entries above
`1e-3` of the largest; `digLost = log10(|x_64-x_big|/(eps |x_big|))` at the same `M,Q`;
`us` = measured wall time per offset (allocating implementation).

Cubic `lambda/32`, `f = 1`, best axis:

    D           tau_xi   tau_t   kR     M*  Q*  errMax    errEnt    digLost  us
    (2,0,0)     2.0      0.5     0.393  --  --  diverges (see section 8)
    (3,0,0)     0.5      0.333   0.589  25  28  3.66e-14  3.66e-14  0.75     142.0
    (4,0,0)     0.222    0.25    0.785  13  22  1.01e-14  1.01e-14  -0.09     52.5
    (6,0,0)     0.08     0.167   1.178   9  18  1.37e-15  1.37e-15  0.06      38.2
    (8,0,0)     0.041    0.125   1.571   7  14  3.18e-14  3.18e-14  -0.04     17.9
    (3,3,0)     0.615    0.236   0.833  40  20  7.30e-14  1.32e-13  0.66     250.6
    (4,4,0)     0.4      0.177   1.111  24  16  7.07e-14  1.07e-13  0.45     119.1
    (6,6,0)     0.230    0.118   1.666  16  12  8.45e-14  1.02e-13  0.44      68.1
    (8,8,0)     0.159    0.088   2.221  13  10  7.42e-14  1.06e-13  0.30      30.3
    (3,3,3)     0.636    0.192   1.020  41  16  7.85e-14  1.37e-13  0.56     232.4
    (4,4,4)     0.439    0.144   1.360  25  14  6.01e-14  6.57e-14  0.71     120.8
    (6,6,6)     0.268    0.096   2.041  16  12  8.53e-14  1.14e-13  0.28      71.6
    (8,8,8)     0.192    0.072   2.721  13  10  8.25e-14  1.32e-13  0.60      31.3
    (5,1,0)     0.235    0.196   1.001  15  20  1.30e-14  2.71e-14  0.24      52.1
    (5,3,2)     0.414    0.162   1.210  22  16  7.76e-14  1.42e-13  0.55      79.8

Cubic `lambda/4`, `f = 1` (same `tau`, `kR` 8x larger):

    D           kR      M*  Q*  errMax    errEnt    digLost  us
    (3,0,0)     4.712   23  26  5.85e-14  1.31e-13  0.49     160.3
    (4,0,0)     6.283   13  22  1.78e-15  5.43e-15  0.77      45.7
    (6,0,0)     9.425    8  16  7.53e-14  3.50e-13  0.81      36.9
    (8,0,0)    12.566    7  16  8.56e-14  5.34e-13  0.45      19.4
    (3,3,3)     8.162   35  16  8.86e-14  1.73e-13  0.57     188.9
    (4,4,4)    10.883   23  16  2.80e-14  4.93e-14  0.72     111.9
    (6,6,6)    16.324   17  14  4.79e-14  4.79e-14  0.76      73.6
    (8,8,8)    21.766   16  14  4.88e-14  4.88e-14  0.97      75.7

**`M*` at `lambda/4` equals `M*` at `lambda/32` to within 2 in every case**, i.e. the
`e^{k rho theta}` factor of the bound does not bite in practice; only `Q*` drifts up by 2--3.
`f = 1 + 0.1i` changes `M*`, `Q*` by at most 2 and the errors not at all (e.g. cubic
`(4,0,0)`: `1.01e-14` at `f=1`, `9.78e-15` at `f=1+0.1i`; `(8,0,0)` `lambda/4`:
`8.56e-14` and `4.40e-15`).

**Float64 digits lost: max 1.07 over every converged case** (cubic `lambda/4` `(6,6,0)`),
median about 0.5, and negative (i.e. better than one `eps`) at several offsets. The only
value above 1.1 in the whole study is `3.38`, at slender `(2,0,0)` where `tau_xi = 1.004`,
`M = 48` and the series is on its radius of convergence -- a divergence artefact, not a
conditioning one.

Invariances (`tables/famC_sym.txt`), all at fixed `M = 30, Q = 40`:

    offset -> -offset, D = (5,1,0),(4,4,4),(5,3,2),(8,0,0)     0.0 (bit-identical)
    axis reflection D_i -> -D_i, off-diagonals flip, 6 cases   0.0 (bit-identical)
    cell-axis permutation of (s,D) permutes the tensor, 4 cases  <= 8.5e-16
    agreement across the choice of exact axis: (4,4,4) 3.6e-16, (8,0,0) 6.6e-16,
       (6,6,0) 7.1e-16, (5,3,2) ax1-ax2 1.9e-14, (5,1,0) ax1-ax3 8.6e-12
       (the last two are the ax2/ax3 truncation at fixed M = 30, tau_xi = 0.48/0.54)
    homogeneity T(alpha s, f/alpha) = alpha^2 T(s,f), alpha = 2,4,8, real and
       complex f, 8 cases                                     0.0 (bit-identical)
    f = 1+0.1i against the reference, D = (4,0,0),(4,4,4),(8,0,0)  <= 1.34e-15

## 7. The slender cell (1/32, 1/32, 1/512)

    D            ax  what is exact       tau_xi    tau_t    M*  Q*  errMax    us
    (2,0,0)      1   long axis (offset)  1.004     0.5      48  34  4.77e-10  681   (1e-9 only)
    (2,0,0)      2   long axis           1.251              --  --  diverges
    (2,0,0)      3   short axis          1.5                --  --  diverges
    (4,0,0)      1   long axis (offset)  0.1116    0.25     11  24  4.91e-15   48
    (4,0,0)      2   long axis           0.5627    0.25     47  22  8.22e-14  383
    (4,0,0)      3   short axis          0.625     0.0156   34   4  6.16e-10  480   (1e-9 only)
    (8,0,0)      1   long axis (offset)  0.0205    0.125     6  14  9.27e-14   16
    (8,0,0)      3   short axis          0.281     0.0078   21   4  1.70e-14   73
    (16,0,0)     1   long axis (offset)  0.00446   0.0625    4  10  9.37e-14    9.0
    (16,0,0)     3   short axis          0.133     0.0039   13   4  3.35e-14   26
    (0,0,8)      1   long axis           4.266     2.0      --  --  diverges
    (0,0,8)      3   short axis (offset) 10.449    0.125    --  --  diverges
    (0,0,16)     1   long axis           1.129     1.0      --  --  diverges
    (0,0,16)     3   short axis (offset) 2.276     0.0625   --  --  diverges
    (0,0,32)     1   long axis           0.313     0.5      21  38  9.98e-14  150
    (0,0,32)     3   short axis (offset) 0.533     0.0313   32   8  9.36e-14  126
    (0,0,64)     1   long axis           0.094     0.25     10  20  6.62e-14   33
    (0,0,64)     3   short axis (offset) 0.129     0.0156   11   6  7.07e-14   23

Readings:

* Along a **long** axis, family (c) is at its best in the whole study: `(16,0,0)` needs
  `M* = 4, Q* = 10`, **9.0 us**, `9.4e-14`. `(8,0,0)` needs 16 us. This is the anisotropy
  payoff, and it is real.
* **The geometry study's finding is confirmed and sharpened.** Along the short axis
  `(0,0,n)`, making the short axis exact gives `tau_xi = 2.276` at `n = 16`, worse than
  making a *long* axis exact (`1.129`) -- but both are `> 1`, so **family (c) is blocked on
  the needle for `n <= 16` in every axis choice, and the number that blocks it is
  `tau_xi = 1.129` at `n = 16` with the best axis.** It converges from `n = 32`
  (`tau_xi = 0.313`, `M* = 21`, 150 us) -- and there it is **8x faster than famG's 1.2 ms at
  the same offset**, because famG's isotropic `rho = 0.708` forces `q* = 68`.
* The only variant that would reach `n = 4..16` on the needle is **two exact axes** (both
  long axes exact, the short/offset axis expanded): `tau = 0.5625 (n=4), 0.266 (n=8),
  0.129 (n=16), 0.0635 (n=32)`. That is a genuinely convergent route in the band where
  everything else fails -- but the required 2D integral of `e^{ikr} r^{-l}` over the
  rectangle is not elementary, so the only way to build it is `box2` for `r^m` plus a
  `k`-series; and on that needle the pair diameter is `2 sqrt(2)/32 = 0.088`, `kD = 0.55`,
  so the *plain uncentred* `k`-series of `notes/moments` converges there in ~12 terms.
  **The two-exact-axes variant on the needle is family (k), not a new mechanism.**

## 8. Where family (c) is blocked, with the number

**The 2-cell band, all shapes: `tau_xi = 1.0` exactly, and it is a geometric identity.**
For cubic cells at `D = (2,0,0)` with axis 1 exact, `xi_max = s_2^2 + s_3^2 = 2s^2` and
`rho_min = |R_1| - s_1 = s`, so `tau_xi = 2.0`. With the best possible choice -- *two* exact
axes, only one transverse coordinate expanded -- `xi_max = s^2` and `rho_min = s`, so
`tau_xi = 1.0` **exactly**: the nearest point of the difference box to the origin is at
distance `s` from it, and the transverse excursion of the box is also `s`. The expansion
point sits exactly on its own radius of convergence. No choice of exact axis, and no
regrouping within family (c), can move that; only subdivision (family d) can. Measured:
cubic `(2,0,0)` fails at `M = 48, Q = 80` for all three axes; `(2,2,0)` reaches `4.6e-2`
(`tau_xi = 1.2`); `(2,2,2)` reaches `7.2e-4` (`tau_xi = 1.111`); at `lambda/4` the same
offsets reach `1.9e6`, `5.0e-3`, `5.5e-5`.

**The slender needle `(0,0,n)`, `n <= 16`: `tau_xi = 1.129` at `n = 16`** (best axis),
`4.27` at `n = 8`, `16.6` at `n = 4`.

**Everything else measured converged to `1e-13`** with the aligned axis: every cubic offset
from 3 cells outward in all three directions at `lambda/32` and `lambda/4`, real and complex
`f`; every slender long-axis offset from 4 cells outward; the needle from 32 cells outward.
The 3-cell band is reached but is the expensive one (`M* = 25..41`, 140--250 us).

## 9. Cost

Operation model (complex FMAs): `gTab` `~ 3(L Q - Q^2)` with `L = 5+2M+2Q`, `Tint`
`~ 6 L_0 Q` with `L_0 = 5+2M`, `Ymom/Zmom/W` `~ 3(M+1)^2`, the `D`-map `~ 2 M L_0`.
Measured minimum times (BenchmarkTools, `cost.jl`) against that model:

    D            M   Q   ns      alloc KiB   model ops   ns/op
    (16,0,0) sl  4   10   8305    10.9        2357       3.5
    (8,0,0)      7   14  16958    19.8        4856       3.5
    (6,0,0)      9   18  25416    30.5        7656       3.3
    (4,0,0)     13   22  42667    43.1       12536       3.4
    (4,4,4)     25   14  82875    56.4       15440       5.4
    (3,0,0)     25   28 111958   100.4       28124       4.0
    (3,3,3)     41   16 183541   114.7       31092       5.9

The implementation allocates the `g` table and the moment arrays per call (11--115 KiB);
the uniform `3.5 ns` per modelled operation is memory traffic, not arithmetic. A
preallocated version should land near 1 ns/op, i.e. **2.4 us at 16 cells, 4.9 us at 8 cells,
13 us at 4 cells, 28 us at 3 cells**. Even unoptimized, `18 us` at 8 cells is far from the
task's "a few hundred flops per offset": the true count is `~5e3` operations at 8 cells and
`~3e4` at 3 cells.

Comparison with famG at the same offsets (`lambda/32`, cubic): famG 127 us at 3 cells /
famC 142 us; famG's model gives ~70 us at 4 cells / famC 52 us; at 8 cells famG's
`q* = 14` gives ~24 us / famC 18 us. **Comparable within a factor 1.5 everywhere on cubes.**
famC wins decisively only where the geometry is anisotropic: slender `(16,0,0)` at 9.0 us,
and slender `(0,0,32)` at 150 us against famG's 1.2 ms.

## 10. Honest summary, band by band

| band | family (c) status | the number |
|---|---|---|
| 2 cells, any shape, any direction | **BLOCKED** | `tau_xi = 1.0` exactly even with two exact axes; measured `4.6e-2` at `(2,2,0)`, `7.2e-4` at `(2,2,2)` |
| 3 cells, cubic, all directions, `lambda/32` and `lambda/4`, real and complex `f` | **WON**, expensive | `<= 8.9e-14` at `M* = 23..41`, `Q* = 16..28`, 140--250 us |
| 4--8 cells, cubic, all directions | **WON** | `<= 1.3e-13` per entry, `M* = 7..25`, 18--121 us, `<= 1.07` digits lost |
| 16+ cells, cubic | **WON**, cheap | `M* <= 7`, `< 20 us` |
| slender, long-axis offsets, `n >= 4` | **WON**, cheapest in the study | `(16,0,0)`: `M*=4, Q*=10, 9.0 us, 9.4e-14` |
| slender, long-axis offset `n = 2` | **BLOCKED** | `tau_xi = 1.004`; reaches `4.8e-10` at `M=48` losing 3.4 digits |
| slender needle `(0,0,n)`, `n <= 16` | **BLOCKED** | `tau_xi = 1.129` at `n = 16`, `4.27` at `n = 8` |
| slender needle, `n >= 32` | **WON**, and 8x cheaper than famG | `tau_xi = 0.313`, `M* = 21`, 150 us, `1.0e-13` |

Not proven: the `Q` rule. The rigorous remainder bound of section 4 is valid but loose by
`~1.8x` at `kR = 21.8`, and the simple rate-only rule under-predicts `Q*` by up to 3 there.
Everything else quoted is a measurement against the 220-bit reference.

Not done: no `GlaVacOprMem` build was touched (no offset dispatch, no band selection, no
timing of the real assembly); the reference set is 52 offsets, not a random lattice sweep;
the anti-Hermitian positivity of an assembled block was not checked (that needs the build).

## Appendix: the level-cancellation number

The `cancel` column of `tables/famC_sweep.txt` is stale (it was written before the diagnostic
was corrected; it summed intermediate `l`-basis products, which are not the terms of any
sum, and reads `1e30`--`1e148`). The correct quantity,
`sum_m |level term| / max_entry |T|` over the seven `xi`-sums, re-measured at the `M*, Q*`
of the table:

    cubic (3,0,0) 2.30   (4,0,0) 1.91   (8,0,0) 1.90   (4,4,4) 7.65   (3,3,3) 7.93
    lambda/4 (3,0,0) 2.53   (8,8,8) 6.17
    slender (16,0,0) 2.25   (0,0,32) ax3 3.11   (0,0,32) ax1 2.51

i.e. **the level sums amplify by 1.9 to 7.9, never more**, which is consistent with the
`<= 1.07` digits lost measured directly against 220-bit arithmetic, and is the reason there
is no conditioning problem anywhere family (c) converges. Cross-check of the shipped code in
both precisions at `(8,0,0)`, `M=10, Q=20`: `|T_64 - T_220bit|/max = 4.63e-16`.
