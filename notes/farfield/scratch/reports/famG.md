# famG — derivative-free Taylor coefficients of the far-field kernel

Author: famG. Work dir `SCRATCH/work/famG/`. All numbers on Apple M3 Pro, Julia 1.12.6, single thread.

Target, source cell at the origin, target cell at `R = (n1 s1, n2 s2, n3 s3)`:

    T_ab(R) = (1/V) ∫_D w(δ) [∂_a ∂_b + δ_ab k²] g(R+δ) dδ,   g(r) = e^{ikr}/(4π f² r),  k = 2π f,
    D = Π[-s_i, s_i],  w = Π (s_i - |δ_i|),  V = s1 s2 s3.

Everything below is built on the Taylor coefficients `c_α = ∂^α g(R)/α!`, written `a_α` in the code,
obtained without ever differentiating `g`.

## 0. Sign and normalization, pinned against the reference

With **no extra sign and no extra factor**, `T_ab` as written above equals Gila's `srfSum!` output built
from `srfMat[fp] = (raw face-pair integral)/V_t`. Verified against the 220-bit reference
(`pairKer` on the 36 face pairs, then the `srfSum!` signs) at **68 (shape, offset, frequency) cases**:
the BigFloat truncated sum reaches `max|T - T_ref| / max|T_ref| ≤ 9.2e-15` at every one of them, and the
opposite sign `|T + T_ref|` is O(1) (script `t2.jl`, table `ver.txt`). Scripts: `ref.jl` (cached 220-bit
reference, 69 tensors in `work/famG/refcache/`), `s_ver.jl`.

## 1. Route (i) — the (G, S) pair recurrence  [the one that works]

`G(δ) = g(R+δ)`, `S(δ) = |R+δ| G(δ) = e^{ik|R+δ|}/(4π f²)`, `P(δ) = |R+δ|² = ρ² + 2R·δ + |δ|²`, `ρ = |R|`.
Two first-order systems, both with polynomial coefficients:

    ∇S = i k (R+δ) G,          P ∇G = (R+δ)(i k S - G).

Writing `G = Σ a_α δ^α`, `S = Σ b_α δ^α`, and reading off the coefficient of `δ^α` in component `j`:

    b_{α+e_j} = i k ( R_j a_α + a_{α-e_j} ) / (α_j + 1)

    ρ² (α_j+1) a_{α+e_j} = R_j ( i k b_α - a_α ) + ( i k b_{α-e_j} - a_{α-e_j} )
                           - 2 R_j α_j a_α - (α_j - 1) a_{α-e_j}
                           - Σ_{i≠j} (α_j+1) [ 2 R_i a_{α-e_i+e_j} + a_{α-2e_i+e_j} ]

with `a_β = b_β = 0` whenever any component of `β` is negative. Seeds:

    a_0 = e^{ikρ} / (4π f² ρ),      b_0 = e^{ikρ} / (4π f²).

Level `n+1` uses only levels `n` and `n-1` — the same two-term structure as the `(ρ²+2up+2vq+p²+q²)^{m/2}`
recurrence of moments.tex §5.4, with the exponential carried by the auxiliary series `b`. The axis `j` is
free; the implementation picks `j = argmax_i (α+e_j)_i` (`piv` in `famG.jl`), the best-conditioned pivot.
Cost: one target coefficient per pivot choice, so `(p+1)(p+2)(p+3)/6` coefficients for total order `p`.

**Cross-checks** (`t1.jl`, 220-bit): route (i) against route (iii) AD `7.9e-66`; against route (ii)
1D ODE along three directions `2.2e-66 … 3.0e-65`; against route (iv) addition theorem `2.5e-66 … 6.2e-65`;
`b_0/a_0 - |R| = 4.0e-68`; Helmholtz residual of the produced coefficients `7.0e-64`.

**Float64 digits lost** (max over 4 directions × {f=1, f=1+0.1i}; `coef.txt`, script `s_coef.jl`):

    kR      n=10  n=20  n=30  n=40
    0.3     1.06  1.25  1.39  1.41
    1.0     0.87  1.11  1.31  1.39
    3.0     0.93  1.13  1.37  1.54
    10      1.01  2.07  2.04  2.02
    30      1.13  1.13  1.84  3.84
    100     1.66  1.67  1.68  1.70
    300     2.15  2.15  2.15  2.15

The flat 1.66/2.15 floors at kR = 100/300 are the **seed**, not the recurrence: `e^{ikρ}` in Float64 at
`kρ = 100` (resp. 300) carries `≈ kρ·eps` relative error, i.e. 2.0 (resp. 2.5) digits, and the value never
degrades further with order. The only genuine growth is at kR = 30 beyond order 30 (1.84 → 3.84 from n=30
to n=40). **Route (i) is stable to order 30 everywhere measured: ≤ 2.15 digits lost, and ≤ 1.4 digits for
kR ≤ 3.** The per-step amplification `Σ|terms|/|result|` peaks at `10^3.9` (kR = 0.3–10, body direction)
yet the realized loss is ~1.4 digits, i.e. the large-amplification entries are the coefficients that are
small by symmetry.

Note that the (high order, high kR) corner never occurs in the assembly: order is set by `ρ = |s|/|R|`, so
large `q` forces small `ρ` hence small `k|R|·ρ`; at λ/32 an offset with kR = 30 sits at 153 cells, `ρ = 0.011`,
`q* = 4`.

## 2. Route (i-b) — the Helmholtz march  [blocked]

    (α_1+2)(α_1+1) c_{α+2e_1} + (α_2+2)(α_2+1) c_{α+2e_2} + (α_3+2)(α_3+1) c_{α+2e_3} + k² c_α = 0,

marched in δ₃ from Cauchy data `c_{(i,j,0)}, c_{(i,j,1)}` (`helmMarch!`). Seeded with the **exact 220-bit**
planar data, so the numbers below are pure recurrence instability, not seed error:

    kR      n=10  n=20  n=30  n=40
    0.3     0.81  1.90  3.46  5.02
    1.0     0.66  2.46  3.70  4.98
    3.0     0.99  2.09  3.55  5.05
    10      1.46  2.30  3.77  5.10
    30      2.09  4.12  6.02  7.91
    100     2.47  5.66  8.23 10.48
    300     2.46  5.88  9.29 12.73

Growth is linear in `n` at roughly `0.12 n` digits at kR ≤ 10 and `0.32 n` at kR = 300. Even at kR = 0.3 the
march has lost 5 digits by order 40 and 3.5 by order 30. It also needs Cauchy data on a plane, which itself
has to come from somewhere. **Blocked**: it never beats route (i), which supplies the same data for free.

## 3. Route (ii) — the 1D polynomial-coefficient ODE plus angular recovery  [blocked on the recovery]

Along a direction `u` (not necessarily unit), with `A = |R|²`, `B = 2 R·u`, `Dq = |u|²`, `φ(t) = g(R+tu)`,
`ψ(t) = |R+tu| φ(t)`, `P(t) = A + Bt + Dq t²`:

    ψ' = i k (B/2 + Dq t) φ,        P φ' = (B/2 + Dq t)(i k ψ - φ),

    q_{n+1} = i k ( (B/2) p_n + Dq p_{n-1} ) / (n+1)
    A (n+1) p_{n+1} = (B/2)(i k q_n - p_n) + Dq (i k q_{n-1} - p_{n-1}) - B n p_n - Dq (n-1) p_{n-1}

seeds `p_0 = e^{ikρ}/(4π f² ρ)`, `q_0 = e^{ikρ}/(4π f²)`. `p_n(u) = H_n(u) = Σ_{|α|=n} a_α u^α`, checked to
`3e-65` against route (i).

The ODE itself is fine: digits lost (max over directions and frequency)

    kR      n=10  n=20  n=30  n=40
    0.3     1.13  1.49  1.84  2.09
    1.0     1.75  2.32  2.68  2.94
    3.0     1.43  1.80  2.13  2.32
    10      0.88  3.37  4.85  4.97
    30      1.10  1.19  1.28  2.39
    100     1.66  1.67  1.66  1.66
    300     2.15  2.16  2.16  2.16

The **recovery** of `a_α` from `H_n` on directions is what kills it. `H_n` restricted to `S²` determines the
degree-`n` homogeneous polynomial uniquely, but in the monomial basis the inversion is exponentially
ill-conditioned. Measured (`s_ang.jl`, `ang.txt`); `cond(Gram)` is the L²(S²) Gram matrix of the degree-`n`
monomials with exact entries `∫_{S²} u^α dΩ = 4π (α_1-1)!!(α_2-1)!!(α_3-1)!!/(|α|+1)!!`:

    n   dim  log10 cond(Gram)  log10 cond(normalised Gram)
    4    15       1.78              0.90
    8    45       4.35              2.00
    12   91       7.20              3.22
    16  153      10.08              4.44
    20  231      13.03              5.67
    24  325      16.02              6.89

and the end-to-end recovery with `M = 2 dim_n` Fibonacci-spiral directions, `R = (3,1,2)/32`, f = 1:

    n   dim   M    log10 cond(V)  digits lost (Float64 ODE)  digits lost (exact H, Float64 solve)
    4    15   30      1.00            0.40                       0.19
    8    45   90      2.32            0.72                       0.46
    12   91  182      3.73            1.04                       0.73
    16  153  306      5.17            1.48                       1.49
    20  231  462      6.62            2.07                       2.07

`log10 cond(V) = 0.33 n` and the loss is `0.10 n`; extrapolating to n = 30 gives `cond ≈ 10^10` and ~3 digits,
to n = 40 `cond ≈ 10^13`. Cost is the second blocker: `M = (n+1)(n+2)` directions, each one ODE.

    p    M    one ODE   M ODEs    one level solve   total
    12  182   0.29 µs    53 µs        16 µs         267 µs
    20  462   0.29 µs   135 µs        86 µs        1941 µs
    30  992   0.42 µs   413 µs       362 µs       11644 µs

Against route (i) at 30 µs (p=20) and 93 µs (p=30), route (ii) is **65× and 125× more expensive** and
loses more digits. **Blocked** — kept only as the independent cross-check that validated route (i) to 3e-65.

## 4. Route (iii) — truncated multivariate Taylor arithmetic  [correct, too slow]

`P(δ) = ρ² + 2R·δ + |δ|²` is entered exactly (10 nonzero coefficients), then `r = sqrt(P)`,
`E = exp(ik r)`, `g = E/(4π f² r)`, with the four truncated operations on the degree-`p` simplex
(`tmul`, `tsqrt`, `texp`, `tdiv` in `famG.jl`): `r_0 = sqrt(P_0)` and
`r_γ = (P_γ - Σ' r_β r_{γ-β}) / (2 r_0)`; `E_0 = exp(V_0)` and, along the largest axis `d` of `γ`,
`γ_d E_γ = Σ_{β ≤ γ-e_d} (β_d+1) V_{β+e_d} E_{γ-e_d-β}`; `Q_γ = (N_γ - Σ' D_β Q_{γ-β})/D_0`.

Agreement with route (i) at 220 bits: `7.9e-66`. Float64 digits lost (`ad30.txt`, `s_ad.jl`; max over 3
directions × 2 frequencies):

    kR    n=10  n=20  n=25  n=30      BigFloat(128) then rounded, n=10..30
    0.3   0.93  1.36  1.31  1.51           ≤ 0 (rounding only)
    3.0   1.23  1.46  1.48  1.55           ≤ 0
    30    1.20  1.61  1.94  2.62           ≤ 0
    300   1.64  1.71  1.68  1.69           ≤ 0

So AD in Float64 is as stable as route (i) (≤ 2.62 digits at order 30), and at BigFloat(128) it is exact to
Float64 after rounding, as expected. The problem is cost — the truncated products are convolutions over the
whole simplex, `O(p^6)`:

    p     Float64     BigFloat(128)     route (i) Float64
    20    1105 µs        184 ms             29.5 µs
    30    8727 µs       1.54 s              92.6 µs

37×/94× route (i) in Float64, 6200×/16600× in BigFloat(128). At the 1.7·10⁷ offsets of a doubled 128³ grid,
the "BigFloat once per offset then round" variant is 36 days. **Blocked on cost, not on accuracy.**

## 5. Route (iv) — the addition theorem  [blocked]

    g(R+δ) = (i k / 4π f²) Σ_l (2l+1) (-1)^l h_l^{(1)}(kρ) j_l(k|δ|) P_l(cos θ),   cos θ = R·δ/(ρ|δ|),

(the `(-1)^l` because `R+δ = R-(-δ)`). Extracting the homogeneous degree-`n` part with `n = l+2m` and
`j_l(z) = Σ_m (-1)^m z^{l+2m}/(2^m m! (2l+2m+1)!!)`, `|δ|^l P_l(cosθ) = Σ_j γ_{lj} ρ^{-(l-2j)} |δ|^{2j} (R·δ)^{l-2j}`
where `γ_{lj}` are the monomial coefficients of `P_l`:

    c^{(n)} = (i k/4π f²) Σ_{l ≡ n (2), l ≤ n} (2l+1)(-1)^{l+m} h_l(kρ) k^n / (2^m m! (l+n+1)!!)
              × Σ_{j=0}^{⌊l/2⌋} γ_{lj} ρ^{-(l-2j)} · [ monomials of |δ|^{2(m+j)} (R·δ)^{l-2j} ],  m = (n-l)/2,

expanded to Cartesian monomials by the two multinomial theorems (`taylorLeg`). Correct: `2.5e-66 … 6.2e-65`
against route (i) at 220 bits, all directions and kR tested.

The cancellation is exactly the predicted one. `P_l` in monomials, `Σ_j |γ_{lj}|` (with `P_l(1) = 1`):

    l      2     6    10    14    20    30    40
    Σ|γ|  2.0  41.0  1.09e3 3.15e4 5.23e6 2.88e10 1.68e14
    (Σ|γ|)^{1/l}  1.414 1.857 2.013 2.095 2.167 2.232 2.268     → (1+√2) = 2.4142

and the realized amplification of `taylorLeg` (max over directions and frequency, `log10 Σ|terms|/|coefficient|`)
and its Float64 digits lost:

    kR     digits lost n=4,8,12,16      log10 amplification n=4,8,12,16
    0.3    0.81  1.22  1.92  2.91        1.62  4.26  4.50  5.46
    1.0    0.59  1.11  1.99  2.77        1.42  3.22  4.61  7.00
    3.0    0.76  1.13  1.92  3.03        1.47  3.01  4.97  5.31
    10     0.66  0.99  1.65  3.00        1.29  1.66  3.76  5.04
    30     1.13  1.19  1.59  2.08        2.24  3.58  4.49  5.12
    100    1.66  1.66  1.86  2.55        3.28  5.67  7.63  9.31
    300    2.15  2.15  2.16  2.91        4.24  7.58 10.50 13.14

Amplification `≈ 10^{0.33 n}` at kR ≤ 30 and `10^{0.82 n}` at kR = 300 (the `h_l(kρ) ~ (2l-1)!!/(kρ)^{l+1}`
against `j_l` cancellation). Loss `≈ 0.18 n`. At order 16 it has already lost 3.0 digits; **at order 30 it is
past the Float64 floor** (projected amplification `10^{10}` at kR ≤ 30, `10^{25}` at kR = 300; the last is
already an overflow risk in Float64 at order 40, where `|h_40(0.3)| = 2.2·10^{80}`, `|h_30(0.3)| = 4.7·10^{56}`). Cost is also bad:
117 µs (p=10), 886 µs (p=16), 2886 µs (p=20) — 100× route (i). **Blocked**: this is the same expansion as
route (i) reorganized, and the reorganization is strictly worse conditioned.

## 6. The assembly, and the reference comparison

With `μ_n(s) = 2 s^{n+2}/((n+1)(n+2))` for even `n`, 0 for odd, and `μ_β = Π_i μ_{β_i}(s_i)`:

    T_ab = (1/V) Σ_{β even, |β| ≤ q} μ_β [ (β_a + 1 + δ_ab)(β_b + 1) c_{β+e_a+e_b} + δ_ab k² c_β ]

(`farTen`). Only even `β` survive, so the sum has `(q/2+1)(q/2+2)(q/2+3)/6` terms and needs `c_α` to `|α| = q+2`.

Route (i) + this assembly, against the 220-bit reference, 68 cases (`ver.txt`, `s_ver.jl`). `q*` is the first
even degree at which the **BigFloat** sum reaches `1e-14` relative to the largest tensor entry — that column is
the **truncation** error; `roundDig` is `log10(|T_Float64 - T_BigFloat|/(eps·max|T_ref|))`, the **rounding**
error; `cancel` is `Σ|terms|/|sum|` entrywise.

    shape / offset      ρ       q*   truncErr   roundDig  cancel   Float64 rel err
    c1/32 (3,0,0)     0.5774    30   9.1e-15      1.20     2.11      5.6e-15
    c1/32 (4,0,0)     0.4330    24   7.9e-15      0.78     2.37      8.4e-15
    c1/32 (6,0,0)     0.2887    18   1.5e-15      1.00     2.45      7.6e-16
    c1/32 (8,0,0)     0.2165    14   1.2e-15      0.57     2.05      2.1e-15
    c1/32 (16,0,0)    0.1083    10   7.9e-16      0.22     3.05      9.5e-16
    c1/32 (4,4,4)     0.2500    16   2.5e-15      0.59     2.04      2.4e-15
    c1/32 (16,16,16)  0.0625     8   7.4e-16      0.48     2.01      1.4e-15
    c1/4  (3,0,0)     0.5774    34   5.8e-15      0.82     7.43      5.3e-15
    c1/4  (8,0,0)     0.2165    18   4.8e-16      0.62    19.45      4.5e-16
    c1/4  (16,0,0)    0.1083    16   7.9e-15      0.67    38.49      8.9e-15
    sl    (4,0,0)     0.3539    24   7.2e-16      0.50     2.31      1.4e-15
    sl    (8,0,0)     0.1769    14   2.4e-15      0.54     2.04      1.9e-15
    sl    (2,2,0)     0.5005    38   3.7e-15      0.60     1.59      3.3e-15
    sl    (0,0,32)    0.7078    68   4.9e-15      1.13     2.16      1.9e-15

Over all 68 cases (both frequencies): max truncation `9.2e-15`, max **rounding 1.42 digits**, max cancellation
`38.7` (c1/4 at 16 cells), max Float64 tensor error `9.8e-15`. Per entry, over the 368 entries not suppressed
by symmetry, the relative error distribution is 2 at `1e-16`, 69 at `1e-15`, 283 at `1e-14`, 14 at `1e-13`;
**worst single entry 2.5e-14** (c1/4 (16,0,0)). `q*` is unchanged or shifts by ±2 between f = 1 and f = 1+0.1i.

`q*` scales as `q* ≈ 17/log10(1/ρ) + const` at λ/32, but at λ/4 it saturates at a floor of 18 driven by the
oscillation `k·2s = π` across one cell, not by `ρ`.

## 7. Cost per offset (`cost.txt`, `cost2.txt`; scripts `s_cost.jl`, `s_cost2.jl`, `s_cost3.jl`)

Route (i) costs a flat **17.0 ns per coefficient** from p = 10 to p = 32 (4.61 µs at p=10 / 286 coefficients,
111.8 µs at p=32 / 6545). `farTen` costs 17.6 ns per even-β term. So

    t(q) ≈ 17 ns × [ (q+3)(q+4)(q+5)/6 + (q/2+1)(q/2+2)(q/2+3)/6 ]

    q      8     10     12     14     16     18     20     24     30     38
    µs   5.62   8.75  12.96  18.21  25.00  33.46  43.87  70.00 127.04 229.71

Gila's current path for the same offset (`egoSrfFxd!` + `srfSum!`, identical timing with a Float32 or Float64
`CPUKerOpt`, so no type-instability artifact):

    offset     quadOrd   µs
    (2,0,0)       9    13607
    (3,0,0)       7     4982
    (4,0,0)       7     4996
    (5,1,0)       6     2672
    (6,0,0)       6     2673
    (8,0,0)       5     1284
    (16,0,0)      5     1319
    (32,0,0)      4      539

Speedup of route (i) at the measured `q*`, cubic λ/32: **(3,0,0) q*=30, 127 µs vs 4982 µs = 39×; (4,0,0) q*=24, 70 vs 4996 = 71×;
(6,0,0) q*=18, 33 vs 2673 = 80×; (8,0,0) q*=14, 18 vs 1284 = 71×; (16,0,0) q*=10, 8.8 vs 1319 = 151×.**
At λ/4: (3,0,0) q*=34, 175 µs = 28×; (4,0,0) q*=24, 70 µs = 71×; (8,0,0) q*=18, 33 µs = 38×;
(16,0,0) q*=16, 25 µs = 53×.

**The one case where it loses**: the slender cell `(1/32,1/32,1/512)` at offset `(0,0,32)`, where the separation
`32·(1/512) = 1/16` is smaller than the transverse cell extent, `ρ = 0.708`, `q* = 68`, cost 1189 µs against
Gila's 539 µs — **2.2× slower**. Float64 accuracy is still fine there (rounding 1.13 digits, tensor error 1.9e-15),
so this is a cost failure, not a stability failure, and it is what subdivision (family (d)) exists for.

This does not meet the prompt's "a few hundred flops per offset": at `q* = 10` route (i) already spends
~30 000 flops. It meets the accuracy target with margin.

## 8. Verdict

**Stable to order 30 in Float64: route (i), the (G,S) pair recurrence (≤ 2.15 digits lost, ≤ 1.4 for kR ≤ 3),
and route (iii), truncated Taylor arithmetic (≤ 2.62 digits).** Route (i) is 37–94× cheaper than route (iii)
and is the one to use. Assembled through the exact moments it reaches every entry of the reference tensor to
`≤ 2.5e-14` relative in Float64 across 68 cases, with the Float64 rounding contributing at most 1.42 digits
and the truncation controlled by the chosen `q*`.

**Not stable to order 30: route (i-b), the Helmholtz march (3.5 digits at order 30 even at kR = 0.3, 9.3 at
kR = 300, from exact Cauchy data); route (iv), the addition theorem (3.0 digits already at order 16, amplification
`10^{0.33 n}` from the Legendre-in-monomials growth `(1+√2)^l`, projected `10^{10}` at order 30); and the angular
recovery half of route (ii) (`cond = 10^{0.33 n}`, 2.07 digits at order 20, ~3 at order 30) — which is also
65–125× more expensive than route (i).**

Open / not done here: an a priori remainder bound (the `q*` above are measured, not proven — `C ρ^{q+1}/(1-ρ)`
is the shape but the constant is not pinned); subdivision for the slender `(0,0,32)` class; anything about the
actual `GlaVacOprMem` build time.

## Files

    work/famG/famG.jl     routes (i), (i-b), (ii), (iii), (iv), the moments and the assembly
    work/famG/ref.jl      220-bit reference per offset, cached in work/famG/refcache/ (69 tensors)
    work/famG/t1.jl       cross-checks of the four routes against each other at 220 bits
    work/famG/t2.jl       sign and normalization against the reference at (5,1,0)
    work/famG/s_coef.jl   → coef.txt   digits lost, routes (i),(i-b),(ii),(iii), order × kR × direction
    work/famG/s_ad.jl     → ad30.txt   route (iii) at order 30, Float64 and BigFloat(128)-rounded
    work/famG/s_leg.jl    → leg.txt    route (iv): correctness, amplification, Legendre-in-monomials growth
    work/famG/s_ang.jl    → ang.txt    route (ii): Gram/Vandermonde conditioning, recovery digits lost
    work/famG/s_ver.jl    → ver.txt    assembled tensor vs the 220-bit reference, 68 cases
    work/famG/s_cost*.jl  → cost.txt, cost2.txt   cost per offset, all routes, and Gila's fixed rule
