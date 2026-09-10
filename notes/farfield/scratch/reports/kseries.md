# k-series outward: term counts, cancellation, band edges, and the shifted exponential

Agent `kseries`. All scripts in `SCRATCH/work/kseries/`, all tables written incrementally:

| script | run as | writes |
|---|---|---|
| `kseries.jl` | `include`d | reusable pieces: `facePairs`, `serTerms`, `serVal`, `serOrd`, `serCnc`, `srfSum`, `panSpan`, `truncBnd`, `truncOrd`, `refSrf` (cached 220-bit reference), `tenErr`, `digLost` |
| `grid.jl` | `include`d | shapes, offsets, frequencies of the sweep |
| `refbuild.jl` | `refbuild.jl c32 sl c8 c4` | `refcache/` (36 `pairKer` values per (D, s, f), 220 bits, ordN 44; 200+ offsets) |
| `scope.jl` | `scope.jl` | `t1_scope.txt` (item 1: 233 rows) |
| `sweep.jl` | `sweep.jl f c32 sl c8 c4` | `t2_float.txt` (items 2, 3, 4a: 208 rows) |
| `sweep.jl` | `sweep.jl b c32:2,0,0 ...` | `t4_big.txt` (item 4b: BigFloat(128) moments) |
| `cost2.jl` | `cost2.jl` | `t5_cost.txt` (item 5) |
| `centred.jl` | `centred.jl` | `t6_centred.txt` (shifted exponential, naive vs `q` route) |
| `centred2.jl` | `centred2.jl` | `t8_shift.txt` (plain vs shifted term counts, volume level) |
| `bands.jl` | `bands.jl` | `t7_bands.txt` (band edges from the bound) |
| `gilachk.jl` | `gilachk.jl` | `t9_gila.txt` (Gila vs series vs 220-bit reference) |
| `gilaben.jl` | `gilaben.jl` | `t10_gilacost.txt` (cost per offset, Gila vs k-series) |

Run everything as `JULIA_NUM_THREADS=1 julia --startup-file=no --project=SCRATCH/env <script>`.
The `n64` column of `t7_bands.txt` is unreliable (it maxes over individual rows rather than
requiring every offset at a given max-norm to pass); the Float64 edges in Section 7(a) are
recomputed correctly from `t2_float.txt`.

## 1. The truncation bound, stated and proved

For a panel pair (A, B) write `I_m = int_A int_B |x-y|^m dS' dS` and
`S = int_A int_B g = (1/(4 pi f^2)) sum_{n>=0} t_n`, `t_n = (i k)^n I_{n-1} / n!`, `k = 2 pi f`.
Let `Dmax = max_{x in A, y in B} |x - y|` and `x = |k| Dmax`.

**Bound.** For every `n >= 0`, `r^{n-1} = r^{-1} r^n <= r^{-1} Dmax^n` pointwise on A x B, and
`r^{-1} > 0`, so `|I_{n-1}| <= Dmax^n I_{-1}`. Hence `|t_n| <= x^n I_{-1} / n!` and

        |R_N| = |S - (1/(4 pi f^2)) sum_{n<=N} t_n|
              <= (I_{-1} / (4 pi |f|^2)) * sum_{n>N} x^n / n!
              <= (I_{-1} / (4 pi |f|^2)) * x^{N+1}/(N+1)! * B(x, N),

        B(x, N) = min( e^x , 1/(1 - x/(N+2)) )   [second form only when x < N+2].

*Proof of B.* `sum_{n>N} x^n/n! = (x^{N+1}/(N+1)!) sum_{j>=0} x^j (N+1)!/(N+1+j)!`.
Since `(N+1)!/(N+1+j)! <= 1/j!` the sum is at most `e^x`; since
`(N+1)!/(N+1+j)! = 1/((N+2)...(N+1+j)) <= (N+2)^{-j}` it is at most `1/(1 - x/(N+2))` when
`x < N+2`. Both are valid, so their minimum is. QED.

`I_{-1}/(4 pi |f|^2)` is the modulus of the `n = 0` term, i.e. the static (Coulomb) value of the
same pair. The bound is therefore *relative to the static value*, and it is relative to the true
value up to the factor `C = (I_{-1}/(4 pi |f|^2)) / |S|`. Measured over the whole sweep
(`lam/amp` columns of `t2_float.txt`, summed over the 36 pairs): `C in [1.000, 16.46]`, with
`C = 1.00` to `1.01` at `f = 1` for every shape and offset and the whole excursion at
`f = 1 + 0.1i`, where `e^{-2 pi Im(f) r}` damps `|S|` but not `I_{-1}` (worst `c8 (24,24,0)`).
So the bound over-states the relative remainder by a factor 1 at real frequency and at most 15
at `f = 1 + 0.1i`.

**Assembly.** `srfMat[fp] = (raw face-pair integral)/V_t` and `G = srfSum!(srfMat)`; every entry is a
signed sum of at most 8 of the 36. Truncating every pair at the same `N`,

        max_ab |G - G_N|_ab / max_ab |G|_ab  <=  Lambda * x_max^{N+1}/(N+1)! * B(x_max, N),
        Lambda = ( sum_{fp} I_{-1}^{(fp)} ) / (4 pi |f|^2 V_t) / max_ab |G_ab| ,

`x_max = |k| max_fp Dmax^{(fp)}`. `Lambda` is column `lam` of `t2_float.txt`: 68.5 at
`c32 (2,0,0)`, 1405 at `c32 (n,n,n)`, 2.2e3 -> 6.5e4 for the slender cell, 15.4 -> 23.5 for
`c4`. It is the *static* assembly amplification and is 1.00 to 3.0 times the measured
`amp = sum_fp |srfMat| / max|G|`.

**Sharpness.** `Nbnd` (smallest N with the bound <= 1e-16) versus `N64` (measured: first term with
`|t_N| <= 1e-16 |sum|`), over all 233 rows of `t1_scope.txt`:
`N64 - Nbnd` in `[-3, +3]` (and identically `[-3, +3]` over the 208 rows of `t2_float.txt`):
the bound over-states the term count by at most 3 and under-states it by at most 3. The bound is tight to O(1) terms at every shape, offset and frequency measured.

## 2. Term counts (`t1_scope.txt`, 233 rows)

Worst over the 36 face pairs, `N` = last term below 1e-16 relative; `Nbnd` = bound's prediction.

| shape | offset | k*Dmax | N (f=1) | Nbnd | N (f=1+0.1i) | cancellation sum|t|/|sum| |
|---|---|---|---|---|---|---|
| c32 | (2,0,0)    | 0.65 | 16 | 15 | 16 | 1.8 / 1.9 |
| c32 | (8,8,8)    | 3.06 | 28 | 28 | 28 | 17.2 / 23.1 |
| c32 | (32,32,32) | 11.2 | 56 | 56 | 57 | 6.0e4 / 1.9e5 |
| c8  | (2,0,0)    | 2.60 | 26 | 26 | 26 | 11.0 / 14.2 |
| c8  | (24,24,24) | 34.0 | 120 | 122 | >120 | 2.6e14 |
| c4  | (2,0,0)    | 5.21 | 35 | 36 | 36 | 122 / 202 |
| c4  | (8,8,8)    | 24.5 | 92 | 95 | 94 | 9.4e9 / 1.0e11 |
| c4  | (12,12,12) | 35.4 | >120 | 125 | >120 | - |
| sl  | (0,0,2)    | 0.28 | 12 | 12 | 12 | 1.24 |
| sl  | (0,0,46)   | 0.64 | 16 | 15 | 16 | 1.83 |
| sl  | (32,32,32) | 9.17 | 50 | 49 | 50 | 8.4e3 / 2.2e4 |

The 120-term budget is exhausted at `c4 (12,12,12)` (12 cells on the body diagonal, 3 lambda),
`c4 (16,16,0)`, `c8 (24,24,24)` and beyond; every `c32` and `sl` offset up to 32 cells converges.
Measured cancellation `cnc` divided by `e^{k Dmax}` lies in `[0.218, 13.4]` over all 233 rows
(`c32 (32,32,32)`: `e^{11.22} = 7.4e4` vs measured 6.0e4). The upper factor is exactly `C`
above: `cnc = (sum|t_n|)/|sum| <= e^x I_{-1} / |S| = C e^x`, and `C <= 16.46` (worst at
`c8 (24,24,0)`, `f = 1 + 0.1i`, where `e^{-2 pi Im(f) r}` damps `|S|` but not `I_{-1}`).

## 3. Cancellation and digits lost, face-pair level (`t2_float.txt`, 208 rows)

Measured over every (shape, offset, frequency) whose series converges within 120 terms.

- Face-pair cancellation `cnc = sum_n |t_n| / |sum_n t_n|`, worst over the 36 pairs, satisfies
  `cnc / e^{k Dmax} in [0.218, 13.4]`; the proven upper bound is `C e^{k Dmax}` with `C <= 16.46`.
- Digits lost of the Float64 face-pair series (Float64 panels, Float64 moments, Float64 sum)
  against the 220-bit `pairKer` value, worst over the 36 pairs:

        dlSer = log10(cnc) + d,       d in [-1.19, +0.59]   over all 208 rows.

  So the face-pair loss is exactly the series cancellation, plus the moments' own ladder loss
  of ~0.3 digits, and nothing else. Extremes over the 208 rows: **0.109 digits** at
  `sl (0,0,46)`, `f = 1 + 0.1i`, and **12.643 digits** at `c4 (12,12,0)`,
  `f = 1 + 0.1i`, where `cnc = 2.6e13`.

## 4. The assembled tensor, and where the precision has to go (items 3 and 4)

Three evaluations of the same 36 face-pair series, all compared with the 220-bit `pairKer`
reference passed through the same `srfSum!` signs:

| variant | moments | sum | file |
|---|---|---|---|
| `ae64`   | Float64 | Float64 | `t2_float.txt` |
| `aeM64`  | Float64, rounded up | BigFloat(128) | `t2_float.txt` |
| `aeB128` | BigFloat(128) | BigFloat(128) | `t4_big.txt` |

**Result.** `aeB128` is `1.9e-31` to `1.6e-28` at every offset of the `t4_big.txt` subset:
BigFloat moments remove the whole problem. `aeM64/ae64` (BigFloat summation and assembly from
the *same* Float64 moments) is in `[0.5, 2]` for **160 of the 208** offsets and in `[0.2, 5]`
for 187: wherever the series cancels, the moment error dominates and higher-precision
summation buys nothing. The 21 outliers are all slender-cell offsets with `cnc` near 1, where
the Float64 summation of the 36 face-pair values and the `srfSum!` signed difference are the
larger error: `sl (1,1,24)` goes from `5.06e-13` to `8.49e-15` (60x), `sl (0,0,2)` from
`7.81e-15` to `1.91e-16` (41x), `sl (0,0,23)` from `2.21e-13` to `1.38e-14` (16x).

Sample (`max_ab |G - Gref| / max_ab |Gref|`):

| shape | offset | f | ae64 | aeM64 | aeB128 | dlMom |
|---|---|---|---|---|---|---|
| sl  | (0,0,8) | 1      | 8.02e-14 | 9.18e-14 | 4.03e-35 | 0.93 |
| sl  | (1,1,24)| 1      | 5.06e-13 | 8.49e-15 | - | - |
| sl  | (8,8,0) | 1      | 4.09e-12 | 6.11e-12 | 1.80e-29 | 1.01 |
| c32 | (2,0,0) | 1      | 2.60e-15 | 8.34e-16 | 6.05e-33 | 0.94 |
| c32 | (4,4,4) | 1      | 4.37e-14 | 2.19e-14 | 6.06e-30 | 0.88 |
| c32 | (8,8,8) | 1+0.1i | 1.18e-13 | 1.10e-13 | 1.28e-29 | 0.94 |
| c8  | (2,0,0) | 1      | 8.81e-16 | 1.14e-15 | 5.00e-31 | 2.65 |
| c4  | (2,0,0) | 1      | 4.33e-15 | 2.71e-15 | 1.86e-31 | 5.51 |

`dlMom` is the worst digits lost of the Float64 moments themselves against the BigFloat(128)
moments, over all m and all 36 pairs: **0.87 to 5.51** across the subset, which exceeds
`moments.tex` Sec. 6.8's `log10((m+1)/2) + 0.08` at the larger `mMax` (Section 9).

The measured Float64 tensor error is bounded by the product estimate and never exceeds it:

        ae64 = alpha * eps * cnc * Lambda,   alpha in [0.00136, 0.648]  over all 208 rows

(`eps = 2.22e-16`). `alpha < 1` because only 4 or 8 of the 36 pairs enter each entry and their
errors have random signs; `alpha` is largest at the small-amplification `c4` offsets.

### The slender cell, and where its loss actually sits
`Lambda` for `(1/32, 1/32, 1/512)` is 2.19e3 at `(2,0,0)` and 3.4e4 to 8.1e4 beyond 8 cells, so
`eps * Lambda` alone is `4.8e-13` to `1.8e-11` before any series cancellation. **In pure
Float64 no face-pair method — the k-series, Gila's Gauss rule, or anything else — reaches
1e-13 on the assembled tensor for the slender cell at any separation**: the best measured value
over the whole slender sweep is `7.8e-15` at `(0,0,2)`, `8.0e-14` at `(0,0,8)` (`1.7e-13` per
entry) and `2.2e-13` at `(0,0,23)`, and every offset with a non-zero x or y component is at
`2.5e-13` or worse at two cells and `9.4e-09` at 32 cells.

The loss splits differently along the two axis families, and the split is the useful part:

- **`(0,0,n)`, the short axis.** `cnc = 1.24` to `1.95`, `dlSer = 0.11` to `0.68` digits: the
  series is essentially exact and the whole error is the Float64 *summation and `srfSum!`
  assembly*. Doing only those in BigFloat(128), from unchanged Float64 moments, moves the band
  edge from **16 cells to beyond 46** (`(0,0,23)`: 2.21e-13 -> 1.38e-14; `(1,1,24)`:
  5.06e-13 -> 8.49e-15).
- **Offsets with a non-zero x or y.** `dlPan` (Float64 moments, exact sum) is `3.8e-16` at
  `(2,0,0)`, which times `Lambda = 2205` is `8.5e-13`: **moment-limited**, and a BigFloat
  assembly does not help (`aeM64 = 1.86e-13` against `ae64 = 2.48e-13`). Only BigFloat moments
  fix it (`sl (8,8,0)`: `4.09e-12 -> 1.80e-29`).

Either way the amplification is `srfSum!`'s, not the k-series', and this is the numerical case
for the volume formulation.

## 5. The shifted exponential and the centred moments (`centred.jl`, `t6_centred.txt`)

With `R` the cell-centre separation, `R0 = |R|`, `x - y = R + delta`, `r = |R + delta|`,
`u = r - R0`, and the triangle weight `w(delta) = prod_d (s_d - |delta_d|)` on
`D = prod_d [-s_d, s_d]`,

        int_Vt int_Vs g  =  (e^{i k R0} / (4 pi f^2)) * sum_n (i k)^n K_n / n! ,
        K_n = int_D w(delta) (r - R0)^n / r  d delta .

**Measurement method.** `K_n`, the plain moments `Itil_m = int_D w r^m` and `<q^p> = int_D w q^p`
with `q = (2 R.delta + |delta|^2)/R0^2` all come from one 512-bit tensor Gauss-Legendre rule on
the eight sub-boxes of `D` (24 nodes per axis per half-interval; `w` is a polynomial and `u` is
analytic on each sub-box). `u` is formed as `(2 R.delta + |delta|^2)/(r + R0)`, never as a
difference. The 16-node rule agrees with the 24-node rule to 1e-16 at two cells and 3e-21 at
eight. Because the naive binomial and the exact `K_n` are evaluated on the *same* grid, the
binomial identity is exact at grid level and the quadrature error cancels identically: the
digit-loss numbers below are pure arithmetic cancellation, independent of the quadrature.

**Scale invariance.** Every quantity in this section (cancellation, digits lost, `max|q|`,
`rho`, the q-series term counts) is *identical* at `s = 1/4` and `s = 1/32` for the same lattice
offset `D`. It depends only on `D`, not on the cell size and not on `k R0`. Only the outer
series length depends on the scale, through `k |s|`.

### (i) The naive binomial loses `n log10(2.3 N)` digits

`K_n = sum_j C(n,j) (-R0)^{n-j} Itil_{j-1}`. Cancellation `sum_j |term| / |K_n|` and digits lost
against the exact `K_n`, at cubic cells, axis offset of `N` cells (`c4` and `c32` identical):

| n | N=2 cnc | dl53 | dl128 | N=4 cnc | dl53 | N=8 cnc | dl53 |
|---|---|---|---|---|---|---|---|
| 2  | 1.01e2  | 1.06 | 1.23 | 3.89e2  | 1.63 | 1.54e3  | 2.30 |
| 4  | 4.31e3  | 2.91 | 3.04 | 6.34e4  | 4.11 | 9.91e5  | 3.25 |
| 8  | 3.86e6  | 5.69 | 5.75 | 8.12e8  | 6.93 | 1.97e11 | 9.97 |
| 12 | 2.37e9  | 8.31 | 8.70 | 7.01e12 | 11.55 | 2.64e16 | 15.32 |
| 16 | 1.23e12 | 10.51 | 11.08 | 5.04e16 | 15.54 | 2.93e21 | 20.71 |
| 20 | 5.68e14 | 13.55 | 13.17 | 3.24e20 | 19.73 | 2.93e26 | 24.33 |
| 26 | 4.92e18 | 16.81 | 17.61 | 1.47e26 | 24.49 | 8.13e33 | 32.82 |

The cancellation grows by a constant factor per order: `10^0.653 = 4.5` at `N = 2`,
`10^0.995 = 9.9` at `N = 4`, `10^1.283 = 19.2` at `N = 8`, i.e. a factor `~2.3 N = 2 R0/d_eff`
with `d_eff = 0.87 s`. Digits lost track it exactly:

        dl(n, N) = n * log10(2.3 N) - 1.2 ,   verified in every column above and at every
        precision (dl53, dl128, dl256 agree to +-1.5 digits: the loss is precision independent).

The loss does **not** depend on `k R0` or on the cell scale. The required `n` (below) is 11 for
`c32` and 20 for `c4`, which costs `11 * 0.653 = 7.2` digits at two cells and
`20 * 1.283 = 25.7` at eight. **The naive form is unusable in Float64 anywhere.**

### (ii) No cancellation-free generation from the `I_m` ladder

Three routes were examined.

*Binomial.* Above: loses `n log10(2.3 N)`.

*The divergence recurrence.* Applying Theorem 5.1 of `integrals.tex` /
eq. (2.2) of `moments.tex` to `phi = (r - R0)^n r^m` with `r = sqrt(|Y|^2 + c^2)`, `Y` in `N'`
dimensions, and using `T_{n-1}^{(m+1)} = T_n^{(m)} + R0 T_{n-1}^{(m)}`:

        (N' + m + n) T_n^{(m)} = B_n^{(m)} - n R0 T_{n-1}^{(m)} + n c^2 T_{n-1}^{(m-1)}
                                 + m c^2 T_n^{(m-2)} ,

`B` the `(N'-1)`-dimensional boundary sum. `T_n^{(m)}` is `O(d^n R0^m V)` while
`n R0 T_{n-1}^{(m)}` is `O(n (R0/d) d^n R0^m V)`: **the recurrence subtracts a quantity
`n R0/d` times larger than its own result at every step**, so it cancels at the same
exponential rate as the binomial (worse by the extra `n!`). This is `moments.tex` pitfall 5
("do not recover a centred moment by subtraction") in its three-dimensional form. Derived and
bounded, not measured numerically.

*The substitution `u = R0 (sqrt(1+q) - 1)`, `q = (2 R.delta + |delta|^2)/R0^2`.* Here
`u^n / r = R0^{n-1} psi_n(q)`, `psi_n(q) = (sqrt(1+q)-1)^n (1+q)^{-1/2} = sum_p c_{n,p} q^p`, so
`K_n = R0^{n-1} sum_p c_{n,p} <q^p>` with `<q^p>` an exact polynomial moment of the box
(products of `mu_j(s) = 2 s^{j+2}/((j+1)(j+2))`). This route is cancellation-free
(measured `sum|c<q^p>| / |sum|`: 1.01 to 26.6 at `N = 8`) **but it converges only where
`max_delta |q| < 1`**. That maximum is attained at the corner aligned with `R`:

        cubic cell, axis offset N:   max|q| = (2N+3)/N^2   -> 1.750 (N=2), 1.000 (N=3),
                                                              0.6875 (N=4), 0.2969 (N=8)
        cubic cell, body diagonal:   max|q| = (2N+1)/N^2   -> 1.250 (N=2), 0.778 (N=3)

so `N >= 4` on an axis and `N >= 3` on the body diagonal, with `N = 3` on the axis exactly on
the boundary. Measured, at `1e-16` relative, the number of q-terms needed:

| offset | max\|q\| | rho | p for n=0 | n=4 | n=8 | n=16 | n=26 | cancellation at n=26 |
|---|---|---|---|---|---|---|---|---|
| (2,0,0) | 1.744 | 0.866 | diverges | diverges | diverges | diverges | diverges | 3.96 |
| (4,0,0) | 0.685 | 0.433 | 53 | 69 | >70 (1.4e-14) | >70 (1.3e-9) | >70 (3.4e-4) | 4.7e3 |
| (8,0,0) | 0.297 | 0.217 | 22 | 30 | 37 | 49 | 64 | 26.6 |

### (iii) The radius, and what the shifted exponential is actually worth

The **outer** series `sum_n (ik)^n K_n/n!` needs, at `1e-16` relative, and with cancellation:

| shape | offset | k R0 | k max\|u\| | plain N | plain cancellation | shifted N | shifted cancellation |
|---|---|---|---|---|---|---|---|
| c32 | (2,0,0) | 0.393 | 0.258 | 15 | 1.49 | **11** | **1.007** |
| c32 | (4,0,0) | 0.785 | 0.234 | 18 | 2.21 | **11** | **1.006** |
| c32 | (8,0,0) | 1.571 | 0.217 | 22 | 4.84 | **11** | **1.006** |
| c32 | (2,2,2) | 0.680 | 0.339 | 17 | 1.99 | **11** | **1.007** |
| c4  | (2,0,0) | 3.142 | 2.061 | 33 | 34.8  | **20** | **1.502** |
| c4  | (4,0,0) | 6.283 | 1.873 | 43 | 8.08e2 | **20** | **1.508** |
| c4  | (8,0,0) | 12.57 | 1.739 | 63 | 4.33e5 | **19** | **1.510** |
| c4  | (2,2,2) | 5.441 | 2.713 | 41 | 3.48e2 | **21** | **1.518** |
| sl  | (8,0,0) | 1.571 | 0.207 | 22 | 4.83 | **11** | **1.008** |
| sl  | (0,0,8) | 0.098 | 0.200 | 12 | 1.15 | **10** | **1.042** |

(`t8_shift.txt`, volume-pair level, 512-bit Gauss, tolerance 1e-16 relative; `f = 1`, the
`f = 1 + 0.1i` rows differ by at most one term.) `max|u| <= |s| = sqrt(s1^2+s2^2+s3^2)`, so the
same proof as Section 1 gives a **separation-independent** truncation bound for the shifted
series, `(k|s|)^{N+1}/(N+1)! * B(k|s|, N)`: `k|s| = 0.34` at `c32` (N = 13 for 1e-16) and
`2.72` at `c4` (N = 27). The measured N (11 and 20) is smaller because `max|u|` is 25 to 35 %
below `|s|`. So the shifted exponential does deliver everything claimed for the *outer* series:
**a fixed 11 (c32) or 20 (c4) terms at every separation, with cancellation 1.006 to 1.51
instead of up to 1.9e5.**

The **inner** generation is the blocker. The exact statement of the radius:

- `z -> |R + z delta|` has its nearest singularity at `|z| = R0/|delta|` for *every* direction
  `delta` (`R0^2 + 2z(R.delta) + z^2|delta|^2 = 0` has roots of product `R0^2/|delta|^2` and
  non-positive discriminant, hence both of modulus `R0/|delta|`). So a Taylor generation of
  `u^n` in `delta` converges exactly for `rho = max|delta|/R0 < 1`: `rho = sqrt(3)/N` on an
  axis, `1/N` on the body diagonal for a cube, i.e. from two cells outward.
- The geometric term count for `1e-13` is `p >= 13/log10(1/rho)`: **208 terms at `N = 2`
  (`rho = 0.866`), 36 at `N = 4`, 20 at `N = 8`.**
- The `q` substitution, which is the only route measured here that is cancellation-free,
  converges only for `max|q| < 1`, i.e. `rho < sqrt(2) - 1 = 0.4142` in the worst direction,
  which is `N >= 4` on an axis and `N >= 3` on the body diagonal.

**Verdict.** The centred moments *are* generable without cancellation, but only by a
`delta`-expansion whose radius is `rho`, which is exactly the radius of the cell-size expansion
of family (a). The shifted exponential therefore does **not** extend the k-series inward: where
it can be generated (`rho <~ 0.43`, i.e. four cells and beyond for cubes) family (a) already
works, and where the k-series is needed (`rho` near 1, two and three cells at `lambda/4`) no
cancellation-free generation of `K_n` exists. The k-series and the cell-size expansion do not
become one object; they meet, and they meet at `rho ~ 0.43`.

What the shifted form *is* worth, and should be handed to family (a): if `K_n` is produced by
the same `delta`-Taylor machinery that family (a) uses, the frequency-dependent work per offset
collapses from 35-110 terms with `e^{k Dmax}` cancellation to 11-20 terms with cancellation
below 1.51, at every separation, and the geometry data (the `<q^p>` or the `delta`-moments) is
frequency independent.

## 6. Cost (item 5, `t5_cost.txt`, `t10_gilacost.txt`, sweep logs)

`pairMoments` at a fixed offset `(4,2,1)`, all 36 face pairs, Apple M3 Pro, one thread,
microseconds per face pair:

| mMax | c32 F64 | c4 F64 | sl F64 | BF128 (2D pairs only) | BF256 (2D pairs only) |
|---|---|---|---|---|---|
| 12 | 860 | 720 | 836 | 2.21e5 | 1.09e6 |
| 24 | 5.04e3 | 5.61e3 | 5.60e3 | 3.55e5 | 1.57e6 |
| 40 | 5.35e4 | 5.86e4 | 5.22e4 | 7.21e5 | 1.61e6 |
| 60 | 4.29e5 | 4.10e5 | 4.45e5 | 9.19e5 | 1.71e6 |

The Float64 columns scale as `mMax^2.6` from 12 to 24, `mMax^4.6` from 24 to 40 and
`mMax^5.1` from 40 to 60 (overall `mMax^3.9` over 12 to 60). The cause is structural:
`pairMoments` runs `for m in -1:mMax; out += box3(m, ...)`, i.e. it re-runs the whole
`K -> M00 -> J` ladder from its seed for every `m` on the three-dimensional (perpendicular)
face pairs. The two-dimensional (parallel and coplanar) pairs use `box2(m0, mMax, ...)`,
which ladders once over the whole range, and are `O(mMax)`: the BigFloat columns above
happen to sample two such pairs (the first two of the 36) and are therefore **not** a
36-pair average. From the 36-pair BigFloat(128) runs of `sweep.jl b`, the honest
BigFloat(128)/Float64 ratio at equal `mMax` is **104x (mMax 24), 625x (mMax 40)**, so a
BigFloat(128) moment build costs 10^2 to 10^3 times the Float64 one.

The **series sum itself is free**: 0.0 to 0.125 us per face pair per frequency in Float64,
10 to 52 us in BigFloat. Everything is in the moments, and the moments are frequency
independent.

### Cost per offset against Gila's present rule (`c32`, one 3x3 tensor, Float64)

| separation (cells) | Gila `quadOrd` order | Gila us/offset | k-series mMax | k-series us/offset | ratio |
|---|---|---|---|---|---|
| 2  | 9 | 13882 | 16 | 32218 | 2.3x |
| 4  | 7 | 4936  | 18 | 48452 | 9.8x |
| 8  | 5 | 1317  | 22 | 70871 | 54x |
| 16 | 5 | 1346  | 29 | 191383 | 142x |
| 32 | 4 | 549   | 40 | 670853 | 1222x |

**This is the number that decides the k-series as a far-field method.** It is already
2.3x more expensive than the present rule at two cells and 1200x at 32 cells, in Float64,
and 10^2 to 10^3 times that again in BigFloat(128). At a 120-term budget the Float64
moments alone cost `3.6e8 us = 362 s per offset`; a 128^3 self-volume has ~1.7e7 offsets.

## 7. Band edges

### (a) Float64: measured, `t2_float.txt`, criterion on `max_ab|G - Gref| / max_ab|Gref|`

| shape | direction | n (ae64) | n (per entry) | n (ae with BigFloat sum) | N terms | mMax | first failing n | ae64 there |
|---|---|---|---|---|---|---|---|---|
| c32 | axis | **12** | 12 | 12 | 27 | 31 | 16 | 1.01e-13 |
| c32 | face | **8**  | 8  | 8  | 26 | 30 | 12 | 2.91e-13 |
| c32 | body | **4**  | 4  | **6** | 22 | 26 | 6  | 1.14e-13 |
| c8  | axis | **6**  | 6  | 6  | 38 | 43 | 8  | 1.68e-13 |
| c8  | face | **4**  | 3  | 4  | 37 | 43 | 6  | 5.91e-13 |
| c8  | body | **3**  | 3  | 3  | 36 | 42 | 4  | 3.06e-13 |
| c4  | axis | **3**  | 3  | **4** | 41 | 46 | 4  | 1.05e-13 |
| c4  | face | **2**  | 2  | 2  | 40 | 47 | 3  | 2.25e-13 |
| c4  | body | **2**  | 0  | 2  | 44 | 52 | 3  | 2.40e-12 |
| sl  | (0,0,n) | **16** | 4 | **46** (all measured) | 13 | 17 | 23 | 2.21e-13 |
| sl  | axis / face / body | **0** | 0 | 0 | - | - | 2 | 2.48e-13 / 8.49e-13 / 5.47e-13 |

The fifth column is Float64 moments with the series sum *and* the `srfSum!` assembly in
BigFloat(128) — the cheap half of the precision upgrade, since the moments are untouched.

Taking the worst direction, the Float64 k-series reaches 1e-13 on the assembled tensor for
**max-norm separations up to 4 cells at lambda/32, 3 at lambda/8, 2 at lambda/4, and nowhere
at all for the slender cell** (except along its own short axis, where the offsets are
sub-wavelength and the amplification is 253 to 1080). With the sum and assembly in
BigFloat(128) and the moments still Float64 that becomes **6 / 3 / 2 / 0**. Number of lattice
offsets in the non-negative octant with max-norm <= n_max (`(n_max+1)^3`): **125 -> 343 (c32),
64 (c8), 27 (c4), 1 (sl)**. Cost per offset there (36 face pairs, Float64 moments), from the controlled `(4,2,1)`
scaling of Section 6: **0.26 s (c32, mMax 26), 2.4 s (c8, mMax 42), 6.5 s (c4, mMax 52)**.
The spread across offsets of the same `mMax` is a factor 2 to 3 either way, because the cost
depends on which `box3` branch each face pair takes (the directly measured `c32 (4,1,0)`
offset at mMax 18 costs 48 ms, against 88 ms from the scaling).

### (b) BigFloat(128) moments and BigFloat(128) sums: `t7_bands.txt`

With BigFloat(128) moments the measured tensor error is **1.9e-31 to 1.6e-28** at every
offset in the `t4_big.txt` subset (`c32 (2,0,0)` through `c4 (2,0,0)` and `sl (8,8,0)`), i.e.
17 to 20 orders below the target. Precision stops being the limit; the **term budget** takes
over. Solving `Lambda * x^{N+1}/(N+1)! * B(x,N) <= 1e-13` for `N <= 120` with the measured
`Lambda` gives the reach:

| shape | axis | face diagonal | body diagonal |
|---|---|---|---|
| c32 | 168 cells (N=120) | 118 (N=120) | 96 (N=120) |
| c8  | 41 (N=119)        | 29 (N=120)  | 23 (N=118) |
| c4  | 20 (N=118)        | 14 (N=119)  | 11 (N=117) |
| sl  | 163 (N=120)       | 115 (N=120) | 115 (N=120) |
| sl (0,0,n) | > 512 (N=41 at n=512) | - | - |

Octant offsets inside the worst-direction edge: **9.1e5 (c32), 1.4e4 (c8), 1.7e3 (c4),
1.6e6 (sl)**. **Cost per offset at those edges: 3.2e8 to 3.6e8 us of Float64 moments,
times the 10^2-10^3 BigFloat(128) factor, i.e. 10 to 100 hours per offset.** The BigFloat
band is unreachable in practice by three to five orders of magnitude; it is a statement
about the mathematics, not about a usable method.

## 8. End-to-end check against Gila (`t9_gila.txt`)

`egoSrfFxd!` + `srfSum!` at Gila's own `quadOrd` order, and the Float64 k-series, both against
the 220-bit `pairKer` reference through the same `srfSum!` signs:

| shape | offset | f | Gila ae / per entry | k-series ae / per entry | Gila worst face pair |
|---|---|---|---|---|---|
| c32 | (2,0,0) | 1      | 3.16e-13 / 7.30e-13 | **2.60e-15** / 4.49e-15 | 8.9e-14 |
| c32 | (5,1,0) | 1      | 6.43e-14 / 2.50e-13 | **4.16e-15** / 1.21e-14 | 3.7e-15 |
| c32 | (8,8,8) | 1      | 1.95e-13 / 3.11e-13 | 9.31e-14 / 1.49e-13 | 2.6e-15 |
| c8  | (4,0,0) | 1      | 2.29e-14 / 2.29e-14 | **9.28e-15** / 1.32e-14 | 6.6e-15 |
| c4  | (2,2,0) | 1      | **1.68e-14** / 3.05e-14 | 2.65e-14 / 4.76e-14 | 8.5e-15 |
| sl  | (3,3,3) | 1      | 1.33e-11 / 2.38e-11 | **5.80e-13** / 1.56e-12 | 4.3e-15 |
| sl  | (0,0,8) | 1      | **1.79e-3 / 3.70e-3** | **8.02e-14** / 1.66e-13 | 2.8e-5 |

The `c32 (5,1,0)` row reproduces the root's published 6.4e-14 for Gila against the same
reference, which validates the offset convention, the panel bookkeeping and the `srfSum!`
signs of everything in this report.

Two findings fall out of this table that are not about the k-series:

- **Gila's fixed rule is wrong by three digits on the slender cell at `(0,0,8)`**
  (1.79e-3 relative on the tensor, 2.8e-5 on the worst face pair, at `quadOrd(8, .) = 5`).
  The `(0,0,n)` offsets of `(1/32, 1/32, 1/512)` put two 1/32-by-1/32 faces `n/512` apart:
  the aspect ratio of the pair is 16n, the integrand is near-singular over the face, and the
  order-5 rule of the `sep >= 8` band cannot see it. The k-series is at 8.0e-14 on the same
  offset.
- At two cells and lambda/32 the k-series is **120x more accurate than the present order-9
  rule** (2.6e-15 against 3.16e-13) for 2.3x the cost.

## 9. Two side findings about the Float64 moments

`dlMom` in `t4_big.txt` is the worst digits lost, over all `m` and all 36 face pairs, of the
Float64 moments against the BigFloat(128) moments of the same panels:

| shape | offset | mMax | dlMom | `moments.tex` Sec. 6.8 prediction `log10((m+1)/2)+0.08` |
|---|---|---|---|---|
| c32 | (2,0,0) | 24 | 0.94 | 1.19 |
| c32 | (4,4,4) | 31 | 0.88 | 1.29 |
| c32 | (16,0,0)| 40 | 0.87 | 1.39 |
| sl  | (8,8,0) | 36 | 1.01 | 1.35 |
| c8  | (2,0,0) | 36 | **2.65** | 1.35 |
| c4  | (2,0,0) | 48 | **5.51** | 1.47 |

The published law holds for `c32` and `sl` but is exceeded by 1.3 digits at `c8` and by
**4.0 digits** at `c4`, at `mMax` of 36 and 48 — above the `m <= 30` range on which
Section 6.8 was measured, and on real face pairs rather than canonical panels. It does not
hurt the series here (the `m ~ 48` terms of a `c4` series are 1e-30 of the sum, and
`c4 (2,0,0)` still lands at `ae64 = 4.3e-15`), but any use of `pairMoments` above `m ~ 30`
in Float64 should be measured, not assumed to lose one digit.

Second: raising the precision of the *summation* is worthless. `aeM64` (Float64 moments,
BigFloat(128) sum) equals `ae64` to within a factor 2 at 160 of the 208 offsets, and the extra
precision has to be in the moments, where it costs 10^2 to 10^3 times as much. The exception,
worth 16x to 60x, is the slender cell at `cnc` near 1 (Section 4): there the Float64
summation and the `srfSum!` difference are the dominant error and a BigFloat(128) assembly
from unchanged Float64 moments moves the `(0,0,n)` band edge from 16 cells to beyond 46.

## 10. What is blocked, and by which number

1. **The k-series cannot be Gila's far-field method, on cost, not on accuracy.** At `lambda/32`
   it is 2.3x the present rule at two cells and **1222x at 32 cells** (`t10_gilacost.txt`),
   and the moment cost grows as `mMax^3.9` to `mMax^5.1` while `mMax` grows linearly with
   separation. The `120`-term budget costs `362 s` of Float64 moments per offset. The task's
   budget is "a few hundred flops per offset".
2. **Extending it outward in Float64 stops at 4 / 3 / 2 cells** (c32 / c8 / c4, worst
   direction) and **never** for the slender cell, blocked by
   `ae64 ~ alpha * eps * cnc * Lambda` with `alpha in [0.002, 0.65]`,
   `cnc/e^{k Dmax} in [0.22, 13.4]` and `Lambda` up to `8.1e4`.
3. **For the slender cell the blocker is `srfSum!`, not the series.** `eps * Lambda` alone is
   `4.8e-13` at two cells and `1.8e-11` beyond eight. Along `(0,0,n)` the series is essentially
   exact (`cnc = 1.24 to 1.95`, `dlSer = 0.11 to 0.68` digits) and the whole error is the
   Float64 summation and assembly: doing those two steps in BigFloat(128) from unchanged
   Float64 moments moves that band edge from 16 cells to beyond 46. Off that axis the Float64
   moments' own `3.8e-16` times `Lambda = 2205` is already `8.5e-13`, so nothing short of
   BigFloat moments works. No pure-Float64 face-pair method reaches 1e-13 on that cell. This is
   the strongest numerical argument in this report for the volume formulation.
4. **The shifted exponential does not bridge the near/far gap.** Its outer series is short and
   clean (11 terms at `lambda/32`, 19-21 at `lambda/4`, cancellation 1.006 to 1.52, at every
   separation), but the centred moments `K_n` have no cancellation-free generation from the
   `I_m` ladder: the binomial loses `n log10(2.3 N) - 1.2` digits (13.6 at `n = 20, N = 2`;
   24.3 at `n = 20, N = 8`), and the divergence recurrence subtracts a term `n R0/d` times
   larger than its own result at every step. The only cancellation-free generation is a
   `delta`-expansion, whose radius is `rho = max|delta|/R0 < 1` (proved) and whose only
   measured realisation, the `q` substitution, needs `max_delta |2 R.delta + |delta|^2| < R0^2`,
   i.e. `N >= 4` on an axis and `N >= 3` on the body diagonal for a cube (exactly `1.000` at
   `N = 3` on an axis). That is family (a)'s own radius, so the two expansions meet at
   `rho ~ 0.43`; they do not become one object.
5. **Not done.** (i) The divergence recurrence for `K_n` was derived and its cancellation
   bounded, but not implemented or measured. (ii) The `delta`-Taylor generation of `K_n`
   (radius `rho < 1`, term count between the rms rate `2 delta_rms/R0 ~ 0.82/N` and the
   worst-corner rate `rho`) was not implemented; that is family (a)'s object and my measured
   `q`-route term counts (53 to 69 at `rho = 0.433`, 22 to 64 at `rho = 0.217`) are an upper
   bound on it. (iii) Subdivision of the difference box, which would push both `rho` and
   `k Dmax` down and is the obvious next move for the two-and-three-cell band at `lambda/4`,
   was not measured. (iv) The digits-lost baseline of item 2 uses the 220-bit `pairKer` values
   rather than a 256-bit `BigFloat` series; the substitution is validated by `t4_big.txt`,
   where the BigFloat(128) series agrees with `pairKer` to `1.9e-31` to `1.6e-28` at every
   subset offset. (v) `c4` offsets beyond `(16,0,0)`, `c8` beyond `(24,24,0)` and all
   `(12,12,12)`-class `c4` offsets exceed 120 terms and were scoped (`t1_scope.txt`) but not
   swept.
6. **One number that should go to whoever owns Gila's fixed rule**: `egoSrfFxd!` at
   `quadOrd(8, .)` returns the `(1/32, 1/32, 1/512)` tensor at offset `(0,0,8)` with a
   relative error of **1.79e-3**.
