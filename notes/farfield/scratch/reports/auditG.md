# auditG — adversarial audit of famG route (i) (the (G,S) recurrence) and its assembly

Author: auditG. Work dir `SCRATCH/work/auditG/`. Nothing under `work/famG/` was modified; `fg.jl`
and `refbase.jl` are byte copies of `famG.jl` and `ref.jl`.

**Machine load warning, stated once and applying to every timing below.** The two-process limit was
respected by this agent throughout (never more than two `julia` processes owned by auditG), but other
agents did not: `ps` showed 18 concurrent `julia --startup-file=no` processes and load averages of
16–37 on a 12-core M3 Pro for the whole session. Every wall time below was taken at a load average
between 16 and 24, recorded in the output files. Accuracy numbers are unaffected.

Reference: the brief's 220-bit recipe (`myref.jl`, cache `work/auditG/refcache/`). Self-convergence
of the reference, measured before anything else (`s_probe.jl`):

    shape                      offset       f        ordN 44 vs 64   t(44)   t(64)
    cubic 1/32                 (7,-3,2)     1        9.4e-63         23.6 s  69.0 s
    aspect (0.00173,0.00309,0.242)  (-7,30,-14)  1   5.9e-61         27.2 s  89.9 s

so the 44-point reference is exact at the 1e-60 level even for a 140:1 aspect ratio. 25 s per tensor,
not the 14 s quoted in the brief.

---

## 1. Independent re-derivation of the (G,S) recurrence

### Symbolic

With `u = R+δ`, `r = |u|`, `G(δ) = g(r)`, `S(δ) = r G(δ) = e^{ikr}/(4π f²)`, `P(δ) = r² = ρ² + 2R·δ + |δ|²`:

    ∂_j S = ik (u_j/r) S = ik u_j G                                        (given)
    ∂_j G = (∂_j S)/r − S u_j/r³ = u_j (ik G/r − G/r²)  ⇒  P ∂_j G = u_j (ik S − G)   (given)

Write `G = Σ_α a_α δ^α`, `S = Σ_α b_α δ^α`, `w_α = ik b_α − a_α`.

*First equation.* Coefficient of `δ^α` in `∂_j S` is `(α_j+1) b_{α+e_j}`; in `ik(R_j+δ_j)G` it is
`ik(R_j a_α + a_{α−e_j})`. Hence

    b_{α+e_j} = ik ( R_j a_α + a_{α−e_j} ) / (α_j + 1).                          (A)

*Second equation.* With `d_β := (β_j+1) a_{β+e_j}` the coefficients of `∂_j G`, the coefficient of
`δ^α` in `P ∂_j G` is `ρ² d_α + Σ_i 2R_i d_{α−e_i} + Σ_i d_{α−2e_i}`, and on the right
`R_j w_α + w_{α−e_j}`. Splitting the `i = j` terms, `d_{α−e_j} = α_j a_α` and
`d_{α−2e_j} = (α_j−1) a_{α−e_j}`, while for `i ≠ j`, `d_{α−e_i} = (α_j+1) a_{α−e_i+e_j}` and
`d_{α−2e_i} = (α_j+1) a_{α−2e_i+e_j}`:

    ρ² (α_j+1) a_{α+e_j} = R_j (ik b_α − a_α) + (ik b_{α−e_j} − a_{α−e_j})
                           − 2 R_j α_j a_α − (α_j−1) a_{α−e_j}
                           − Σ_{i≠j} (α_j+1) [ 2 R_i a_{α−e_i+e_j} + a_{α−2e_i+e_j} ].   (B)

(A) and (B) are **term for term identical** to §1 of `famG.md` and to lines 37–45 of `famG.jl`
(`acc = R[j]*(im*k*gt(b,al) − gt(a,al)) + (im*k*gt(b,am) − gt(a,am))`, then
`acc -= 2*R[j]*aj*gt(a,al) + (aj−1)*gt(a,am)`, then the `i ≠ j` loop, then division by
`rho2*(aj+1)`), with `al = α = g − e_j`, `aj = α_j`, `am = α − e_j`. Seeds
`a_0 = e^{ikρ}/(4π f² ρ)`, `b_0 = e^{ikρ}/(4π f²)` also agree. No sign, index or factor differs.

The assembly was re-derived the same way. `∂_a∂_b Σ_α a_α δ^α` has coefficient
`(β_a+1+δ_ab)(β_b+1) a_{β+e_a+e_b}` at `δ^β` (for `a=b` this is `(β_a+2)(β_a+1)`), and
`∫_D w δ^β dδ = Π_i μ_{β_i}(s_i)` with `μ_n(s) = 2 s^{n+2}/((n+1)(n+2))` for even `n`, 0 for odd
(`∫_{-s}^{s}(s−|t|)t^n dt = 2[s t^{n+1}/(n+1) − t^{n+2}/(n+2)]_0^s`), so

    T_ab = (1/V) Σ_{β even, |β| ≤ q} μ_β [ (β_a+1+δ_ab)(β_b+1) a_{β+e_a+e_b} + δ_ab k² a_β ],

which is `farTen` exactly. My own implementation `farSeq` (`aud.jl`) recomputes this independently
and degree by degree; it agrees with `farTen` to the last bit at every `q` used below.

**One defect in `famG.jl` worth naming.** `gt(A,i,j,k)` returns zero for any index `> p` but returns
the *stored* (never written, therefore zero) entry when `|α| > p` with each component `≤ p`. So
`farTen(a, s, f, q)` with `q > size(a,1)−3` silently truncates the series instead of erroring:
the caller is responsible for having generated `a` to order `q+2`. `farSeq` asserts this instead.

### Numerical, against an independently written 256-bit evaluator

`mtay.jl` is my own multivariate truncated-Taylor arithmetic and shares no code with famG's
`tmul/tsqrt/texp/tdiv`: `1/r` by Newton degree-doubling on `y ← y(3 − P y²)/2`, `r = P·y`, and
`e^{ik(r−ρ)}` by the *terminating* power series `Σ_m u^m/m!` (`u` has no constant term), not by an
ODE recursion. A third, non-Taylor-arithmetic check extracts the same coefficients by the
multivariate Cauchy integral on the polydisc `|δ_m| = ρ/6`, sampled on a 64³ torus grid and
transformed by a separable DFT (`cauchyCoef` in `s_q1.jl`).

Max **per-coefficient relative** disagreement over `|α| ≤ 30` (`q1.txt`):

    case                          kR      precision   max rel. disagreement   argmax        max |Δ|/max|a|
    cubic 1/32 (3,0,0)   f=1      0.589   256 bit     5.38e-76                (24,0,0)      1.69e-76
    cubic 1/32 (3,0,0)   f=1      0.589   384 bit     3.39e-114               (30,0,0)      4.17e-115
    cubic 1/32 (7,-3,2)  f=1+0.1i 1.554   256 bit     3.58e-73                (9,7,11)      2.66e-76
    cubic 1/32 (30,20,10) f=3+0.3i 22.15  256 bit     1.49e-72                (17,11,2)     2.31e-73

The disagreement moves from the 76th to the 114th digit when the precision is raised from 256 to 384
bits, i.e. it is rounding, not a difference of formula. Cauchy extraction versus both:

    Cauchy (N=64, r0 = ρ/6) vs famG taylorSys:  max rel 5.07e-26 at (0,0,30),  max |Δ|/max|a| 2.34e-32
    Cauchy vs mtay:                              max rel 5.07e-26 at (0,0,30),  max |Δ|/max|a| 2.34e-32

(the 1e-26 floor is the `6^{30}` amplification of the polydisc radius, not a disagreement: the two
Taylor routes agree with each other 50 orders of magnitude better).

**Verdict on item 1: famG's coefficient formula is correct. No discrepancy found at any level.**

Scripts: `work/auditG/s_q1.jl` → `q1.txt`; `mtay.jl`; `aud.jl`.


---

## 2. Random lattice directions and shapes vs the 220-bit reference

### 2a. First, the near band — where route (i) does not converge

Before the random sweep: the truncation curve of the BigFloat sum itself, degree by degree, at the
first separated shells. `err(q) = max|T_q − T_100| / max|T_100|`, 200-bit, f = 1 (`s_q8.jl` → `q8.txt`).
`tail` is `|T_100 − T_98|/max`, i.e. the residual error of the last column; where `tail` is large the
`q(1e-13)` / `q(1e-15)` columns mean "> 100".

    shape   D          ρ       err(10)  err(20)  err(30)  err(40)  err(60)  err(80)  tail(100)  q(1e-13)  q(1e-15)  cancel  Float64 rounding
    c32     (2,0,0)    0.8660   1.5e-4   9.7e-7   6.0e-8   7.9e-9   1.5e-10  3.1e-12  1.6e-13    >100      >100      974.0   3.02e-14
    c32     (2,1,0)    0.7746   6.3e-5   4.8e-7   8.4e-9   1.6e-10  9.5e-14  4.7e-16  1.4e-18      60        78        2.79   6.73e-15
    c32     (2,1,1)    0.7071   2.2e-5   1.1e-7   7.1e-10  3.8e-12  1.2e-15  2.1e-19  2.5e-22      50        62        1.73   4.51e-15
    c32     (2,2,0)    0.6124   3.8e-6   1.2e-9   3.5e-12  3.7e-15  4.7e-20  1.2e-24  1.1e-28      36        46        1.55   2.59e-15
    c32     (2,2,1)    0.5774   7.4e-7   4.9e-10  4.4e-13  2.3e-16  3.5e-21  1.5e-26  1.6e-31      34        40        1.41   3.22e-15
    c32     (2,2,2)    0.5000   2.3e-7   4.1e-11  1.3e-14  5.3e-18  1.4e-24  5.1e-31  5.4e-37      30        34        1.52   3.33e-15
    c32     (3,0,0)    0.5774   1.5e-6   2.5e-10  9.1e-15  1.5e-16  1.3e-21  1.1e-26  3.6e-31      30        40        1.35   4.00e-15
    c32     (3,1,0)    0.5477   4.7e-7   1.2e-10  7.3e-14  1.2e-16  9.1e-23  1.3e-28  1.4e-33      30        38        1.35   9.34e-16
    c32     (3,2,1)    0.4629   6.2e-8   2.8e-12  2.9e-16  7.5e-20  2.6e-27  1.9e-34  6.5e-41      26        30        1.49   1.01e-15
    c32     (4,0,0)    0.4330   4.8e-8   5.1e-13  5.0e-18  3.2e-22  1.8e-29  5.1e-37  1.1e-43      24        28        1.17   1.32e-15
    c4      (2,0,0)    0.8660   2.3e-5   1.2e-7   2.4e-8   2.7e-9   4.4e-11  9.1e-13  4.5e-14    >100      >100      339.9   2.12e-14
    c4      (3,0,0)    0.5774   3.7e-8   1.1e-11  2.1e-14  4.2e-17  2.3e-22  1.6e-27  4.9e-32      30        36        1.82   1.54e-15
    c4      (2,2,2)    0.5000   3.0e-8   1.3e-12  6.9e-16  3.2e-19  9.3e-26  3.5e-32  3.6e-38      26        30        1.55   4.27e-15
    c4      (4,0,0)    0.4330   3.1e-8   1.3e-14  1.9e-18  1.7e-22  2.6e-30  5.5e-38  9.1e-45      20        26        1.72   1.88e-15
    slZ     (0,0,20)   1.1325   8.3e-1   7.9e-1   8.3e-1   7.7e-1   7.0e-1   4.7e-1   1.6e+0     never     never     362.9   7.84e-15
    slZ     (0,0,24)   0.9437   1.6e-2   1.8e-3   3.7e-4   9.0e-5   7.0e-6   5.4e-7   8.4e-8     >100      >100        2.87   1.01e-14
    slZ     (0,0,32)   0.7078   5.0e-4   3.4e-6   3.8e-8   5.3e-10  1.3e-13  3.4e-17  2.3e-20      62        72        1.74   2.29e-15
    slZ     (2,0,0)    0.7078   9.2e-5   2.2e-8   1.2e-8   6.8e-12  3.9e-15  2.4e-18  4.8e-20      52        64        2.09   2.94e-15

Two hard failures, neither of which appears in famG's report.

**(i) Two-cell separation.** At `D = (2,0,0)`, λ/32 cubes, `ρ = 0.866`, route (i) is at **7.9e-9** at
q = 40 and needs q ≈ 100 to reach 1.6e-13, at which degree the assembly's cancellation
`Σ|terms|/|sum|` has grown to **974** and the Float64 rounding error is 3.0e-14. So at two cells
route (i) at q ≤ 40 is **four orders worse than Gila's existing order-9 rule**, which delivers
3.16e-13 there (item 2e). Route (i) does beat it, but only at q ≈ 100, where it reaches 1.11e-13 at a
cost of about 4.4 ms (187 460 coefficients at 18 ns plus 23 426 assembly terms at 43 ns) against
Gila's measured 14.0 ms — 3.2× cheaper and 2.8× more accurate, but two orders of magnitude more
expensive than route (i) is anywhere else, and only at that one degree (item 2e shows the U-curve).
The
neighbouring classes `(2,1,0)` and `(2,1,1)` need q = 60 and q = 50 for 1e-13, also above 40.
The first offsets that reach 1e-13 within q ≤ 40 are `(2,2,0)` (q = 36) and `(3,0,0)` (q = 30).
At λ/4 the picture is the same or slightly worse: `(2,0,0)` needs q > 100.

**(ii) Slender cells separated along the short axis — route (i) diverges.** For
`s = (1/32, 1/32, 1/512)`, `|s| = 0.044196`; an offset `(0,0,n)` has `|R| = n/512`, so
`ρ = |s|/|R| > 1` for all `n ≤ 22`. The Taylor series about `R` has radius `|R|`, the difference box
reaches `|s| > |R|`, and the expansion **does not converge**: at `n = 20` the error is 0.83 at q = 10
and 1.6 at q = 100, i.e. the partial sums are still O(1) wrong and growing. At `n = 24` (ρ = 0.944)
it converges but at 8.4e-8 after q = 100. These are ordinary separated offsets that Gila computes
today with `quadOrd`; a slender self-volume of more than 23 cells along the short axis contains 21
divergent z-shells plus their neighbours. famG tested only `(0,0,32)` (ρ = 0.708) and reported it as
a *cost* failure; it is a *convergence* failure two shells further in, and famG's report does not
mention that a convergence boundary exists at all for this shape.

The convergence boundary, scanned (`s_q10.jl` → `q10.txt`, 200-bit, f = 1, `err(q) = |T_q − T_60|/max`):

    slender (1/32,1/32,1/512), offset (0,0,n):  |s| = 0.044196, |R| = n/512, ρ = |s|/|R|
    n     ρ        err(10)  err(20)  err(30)  err(40)  err(50)  tail(60)
    18    1.2583   1.0      0.99     1.0      0.96     1.2      1.7        diverges
    20    1.1325   0.44     0.29     0.43     0.24     0.56     0.67       diverges
    21    1.0785   0.10     0.0067   0.051    0.0030   0.051    0.052      diverges
    22    1.0295   0.046    0.011    0.0073   0.0019   0.0038   0.0032     diverges
    23    0.9848   0.026    0.0046   0.0015   4.4e-4   3.2e-4   2.1e-4
    24    0.9437   0.016    0.0018   3.7e-4   8.3e-5   3.2e-5   1.6e-5
    26    0.8711   0.0061   3.2e-4   2.9e-5   3.1e-6   4.4e-7   1.3e-7
    28    0.8089   0.0025   6.3e-5   2.7e-6   1.4e-7   8.9e-9   1.4e-9
    30    0.7550   0.0011   1.4e-5   3.0e-7   7.9e-9   2.4e-10  2.2e-11
    32    0.7078   5.0e-4   3.4e-6   3.8e-8   5.3e-10  8.3e-12  4.3e-13
    40    0.5662   3.4e-5   2.5e-8   3.0e-11  4.6e-14  7.6e-17  6.0e-19

The boundary is exactly `n = 512·|s| = 22.6`, i.e. `ρ = 1`. **Within q ≤ 40, no offset (0,0,n) with
n ≤ 39 reaches 1e-13 for this shape** — the first is n = 40 at 4.6e-14. That is 39 z-shells of a
slender self-volume, every one of which Gila computes today.

### 2b. What was actually run

296 220-bit reference tensors were built in two background processes (`mkref.jl`, shards 1 and 2),
25–31 s each for the far cases and 37–112 s for the near band (the graded reference quadrature needs
many panels at two cells), ≈ 2 h 10 min of wall time in two processes. Case list (`cases.jl`,
`MersenneTwister(20260906)`):

- 40 random cubic λ/32 offsets, integer components in [-40,40], `Σn² ≥ 9`, no duplicates,
  the first 8 forced to have exactly one zero component, the next 8 forced to have two equal
  components (e.g. (0,-38,-22), (39,15,0), (-17,0,28), (12,14,-39), (-34,-15,35), (31,31,-8)).
- 20 random shapes `s = (1/32)(10^{u1},10^{u2},10^{u3})`, `u_i ~ U[-1.5,1.5]` rounded to exact
  dyadic rationals, each with one random offset. **These reach cell edges of 0.96 λ and aspect
  ratios up to 188:1** — inside the prompt's specification, far outside Gila's λ/128–λ/4 range;
  they are separated out below because they fail for a reason that never occurs in Gila.
- `(1/32,1/32,1/512)` and `(1/512,1/32,1/32)`, 8 offsets each, including the short axis at n = 32,
  48, 64.
- 15 near-band cases: cubic λ/32 at (2,0,0) (2,1,0) (2,1,1) (2,2,0) (2,2,1) (2,2,2) (3,1,0) (3,2,1),
  cubic λ/4 at (2,0,0) (3,0,0) (2,2,2), slender at (0,0,20) (0,0,24) (2,0,0).

Frequencies: every case at f = 1 and f = 1+0.1i; a third rotating over {1+1i, 0.37, 3+0.3i} by case
index; and all of {1+1i, 0.37, 3+0.3i} on the 8 most extreme aspect ratios and all 16 slender cases.
**290 (case, frequency) comparisons in total**, 88 at f=1, 78 at 1+0.1i, 42 at 1+1i, 40 at 0.37,
42 at 3+0.3i. No case failed to produce a number.

`q*` is the smallest even degree at which the 220-bit partial sum is within 1e-15 of the converged
value, relative to the largest tensor entry; the Float64 tensor is then evaluated at that `q*`.
Errors relative to `max|T_ref|` unless stated; `maxEntry`, `maxRe`, `maxIm` are per-entry relative
errors over the entries with `|T_ref[i]|/max ≥ 1e-12` (respectively `|Re|`, `|Im|` above the same
threshold).

    class / frequency        n   maxTrunc   maxTens   maxEntry     maxRe     maxIm    maxCancel  max q*
    cubic 1/32 (40 offs)    120    1.0e-15   2.8e-15   7.5e-15   1.2e-13   1.3e-13       1.06    14
    random aspect (20)       76    9.7e-16   5.8e-10   1.4e-08   1.5e-08   1.5e-09   1.32e+07    72
    (1/32,1/32,1/512)        40    1.0e-15   4.8e-15   8.5e-15   6.5e-14   2.4e-12       1.74    74
    (1/512,1/32,1/32)        40    1.0e-15   4.8e-15   8.5e-15   6.5e-14   2.4e-12       1.74    74
    near band cubic 1/32      9    2.2e-15   1.1e-12   1.6e-12   1.6e-12   6.1e-13   1.79e+04   120
    near band cubic 1/4       4    8.4e-16   2.4e-14   2.6e-14   8.3e-14   6.7e-15        450   102
    near band slender         1    2.3e+00   2.3e+00   2.5e+00   2.5e+00   7.3e-15   6.83e+03   120

    f = 1                    88    2.3e+00   2.3e+00   2.5e+00   2.5e+00   4.0e-12   1.79e+04   120
    f = 1+0.1i               78    2.2e-15   1.1e-12   2.8e-12   2.8e-12   1.3e-12   1.77e+04   120
    f = 1+1i                 42    8.2e-16   2.3e-14   6.5e-14   2.9e-13   3.9e-14       11.8    74
    f = 0.37                 40    1.0e-15   1.9e-15   4.8e-15   6.4e-15   2.4e-12       2.27    72
    f = 3+0.3i               42    7.5e-16   5.8e-10   1.4e-08   1.5e-08   1.5e-09   1.32e+07    72

    the 200 rows on the two    200    1.0e-15   4.8e-15   8.5e-15   1.2e-13   2.4e-12       1.74    74
    Gila-range shapes (cubic
    λ/32 and both slender)

**The random directions themselves found nothing.** Over 120 cubic λ/32 rows spanning every
direction class (one zero component, two equal components, generic, offsets to |n| = 62 cells,
kR from 0.8 to 25), the assembled Float64 tensor is within **2.8e-15** of the 220-bit reference and
every entry within **7.5e-15**; `q* ≤ 14`; the assembly's cancellation never exceeds 1.06. Both
slender shapes give bit-identical error columns to each other (a further confirmation of the
permutation symmetry of item 3) and stay within 4.8e-15 / 8.5e-15.

Worst nine rows overall:

    class  D             s                                f          ρ       kR     k·2s   q*   trunc    Float64  perEntry   Re       Im       cancel
    nearZ  (0,0,20)      (0.03125,0.03125,0.001953)       1          1.1325    0.24   0.39  120  2.3e+00  2.3e+00  2.5e+00  2.5e+00  7.3e-15  6.83e+03
    asp    (21,-10,40)   (0.9639,0.005121,0.07885)        3+0.3i     0.0472  388.07  36.52   72  6.3e-17  5.8e-10  1.4e-08  1.5e-08  1.5e-09  1.32e+07
    asp    (26,19,14)    (0.001058,0.7764,0.0342)         3+0.3i     0.0527  279.59  29.41   62  6.8e-17  1.9e-11  2.6e-09  3.5e-09  6.3e-10  4.25e+05
    asp    (21,-10,40)   (0.9639,0.005121,0.07885)        1          0.0472  128.71  12.11   36  3.2e-16  3.4e-13  4.8e-12  1.1e-10  7.4e-13  4.33e+03
    asp    (21,-10,40)   (0.9639,0.005121,0.07885)        1+0.1i     0.0472  129.36  12.17   36  6.9e-17  9.6e-14  2.8e-12  2.8e-12  4.7e-13       778
    near   (2,0,0)       (0.03125,0.03125,0.03125)        1          0.8660    0.39   0.39  120  2.2e-15  9.7e-13  1.6e-12  1.6e-12  3.8e-16  1.79e+04
    near   (2,0,0)       (0.03125,0.03125,0.03125)        1+0.1i     0.8660    0.40   0.40  120  2.2e-15  1.1e-12  1.1e-12  1.1e-12  6.1e-13  1.77e+04
    asp    (-26,18,-30)  (0.3865,0.003946,0.07587)        3+0.3i     0.0382  195.21  14.64   40  3.0e-17  3.4e-14  5.9e-13  1.1e-13  1.2e-12  1.06e+03
    asp    (26,19,14)    (0.001058,0.7764,0.0342)         1          0.0527   92.74   9.76   30  9.3e-16  9.1e-15  2.0e-13  4.6e-13  4.0e-12       77

Nine of 290 rows exceed 1e-13 per entry, seven exceed 1e-12; six exceed 1e-13 on the whole tensor.
Every one of them is one of three failure modes, none of which is a random-direction effect:

**(A) `ρ ≥ 1` — divergence** (`nearZ (0,0,20)`): already covered in 2a.

**(B) `k·2s` of order 10 or more — Float64 cancellation in the assembly, not truncation.** The
BigFloat truncation error at these rows is 3e-17 to 9e-16, i.e. the *series is fine*; what fails is
the Float64 evaluation. The cancellation `Σ|terms|/|sum|` measured entrywise grows with the phase
across one cell:

    k·2s      1     3.1     7.0     9.8    12.1    14.6    29.4    36.5
    cancel   1.1     450    11.6      77   4.3e3   1.1e3   4.3e5   1.3e7

i.e. roughly `10^{0.2 k·2s}`, and the Float64 error tracks it: 5.8e-10 at cancel 1.3e7 is
`cancel · 4e-17`. **Gila never sees this**: its coarsest cell is λ/4, `k·2s ≤ π`, and the two rows
at `k·2s ≈ 3.1` (cubic λ/4 near band) show cancel 450 and error 2.4e-14. But any claim that route (i)
is general in cell size is false — it costs one digit per 5 radians of phase across a cell.

**(C) `ρ → 1` — cancellation grows with the degree needed.** At cubic λ/32 (2,0,0), `q* = 120` is
needed for 1e-15 truncation and the assembly's cancellation there is 1.79e4, giving a Float64 error
of 1.1e-12 — see 2c.

### 2c. Real and imaginary parts separately

The task asked for the radiative (imaginary) part to be checked in relative terms. Over all 290 rows
the product (relative Im error) × (|Im entry| / max|T|) has median 9.7e-17 and maximum 4.1e-13, and
over the 200 Gila-range rows its maximum is **1.7e-15**. In other words the *absolute* error of the
imaginary part is at the `eps · max|T|` level everywhere in Gila's range, and its *relative* error is
simply `≈ 10^{-16} / (|Im| / max|T|)`. Consequences:

    shape          D        f       |Im|/max|T|    relative Im error
    cubic 1/32   (2,32,-16)  1+0.1i   3.6e-3        1.33e-13
    cubic 1/32   (-37,23,16) 1+0.1i   2.0e-3        1.31e-13
    slender      (0,3,5)     0.37     1.6e-6        2.37e-12
    slender      (3,3,0)     0.37     5.9e-5        4.43e-13
    aspect       (26,19,14)  1        6.0e-5        3.99e-12

**So the answer to "is the radiative part right in relative terms" is: no, not to 1e-13, whenever the
imaginary part of an entry falls below about 1e-3 of the largest entry** — which happens routinely
at low frequency (f = 0.37) and for slender cells. The largest real-part relative error in Gila's
range is 1.2e-13 (cubic λ/32, D = (22,0,27), f = 3+0.3i), by the same mechanism.
This is a Float64 floor of the *formulation*, not of route (i): the tensor is assembled as a single
complex sum, so a real part 1e3 times larger than the imaginary part costs the imaginary part three
digits. Nothing in famG's report separates Re from Im, and its "per-entry worst 2.5e-14" is a
modulus, not a component.

### 2d. The truncation degree, and the rule for it

`q*` (1e-15 truncation, 220-bit) against `L = log10(1/ρ)`, from the 290 rows plus the near band:

    ρ      0.8660 0.7746 0.7078 0.7071 0.6124 0.5774 0.5477 0.5000 0.4693 0.4629 0.3539 0.2000 0.1525 0.0918 0.0560 0.0373
    L      0.0625 0.1109 0.1501 0.1505 0.2130 0.2385 0.2615 0.3010 0.3285 0.3345 0.4511 0.6990 0.8167 1.0372 1.2518 1.4283
    q*        120     78     74     62     46     40     38     34     32     30     28     18     14     10      8     12
    q*·L     7.50   8.65  11.11   9.33   9.80   9.54   9.94  10.24  10.51  10.04  12.63  12.58  11.43  10.37  10.01  17.14

`q*·L` is **not** constant: it drifts from 7.5 near ρ = 1 to 10–17 at small ρ (the spread at fixed ρ
is the frequency and shape dependence, ±4 in q*). A least-squares two-point fit gives

    q* ≈ 7 + 7.1 / log10(1/ρ)     (add 2 for margin, round up to even)

famG's quoted `q* ≈ 17/log10(1/ρ) + const` overshoots by 3.4× in cost (item 7); a pure `9.7/L` with
no constant undershoots by 2 at ρ ≤ 0.06, which is where 90 % of a block's offsets sit. Re-running
the 32³ sweep with the corrected rule: **1.85 s, 56.5 µs/offset**, q histogram
`14:761 16:28633 18:2602 20:464 22:139 24:58 26:33 28:24 …`, 15 offsets above q = 40, max 120.
Against Gila's 23.3 s this is a **12.6× block speedup**, not 18× and not 39–151×.

### 2e. Route (i) against the rule it is meant to replace

Gila's `egoSrfFxd!` + `srfSum!` at its `quadOrd` order, versus route (i), both against the same
220-bit reference, f = 1 (`s_q9.jl` → `q9.txt`; errors relative to `max|T_ref|`):

    shape  D          quadOrd   Gila       route(i) @ q=40   q*    route(i) @ q*
    c32    (2,0,0)      9       3.16e-13   7.87e-9           100   1.11e-13
    c32    (2,1,0)      9       8.19e-14   1.58e-10           70   1.40e-14
    c32    (2,1,1)      9       1.48e-13   3.81e-12           54   9.72e-15
    c32    (2,2,0)      9       7.32e-14   4.36e-15           40   4.36e-15
    c32    (2,2,1)      9       1.93e-13   3.36e-15           36   1.08e-14
    c32    (2,2,2)      9       1.80e-13   3.61e-15           32   9.91e-16
    c32    (3,1,0)      7       4.33e-14   1.10e-15           34   4.23e-15
    c32    (3,2,1)      7       7.92e-14   3.35e-15           28   3.41e-15
    c4     (2,0,0)      9       8.62e-14   2.69e-9           100   2.74e-14
    c4     (3,0,0)      7       1.03e-14   1.26e-15           34   5.25e-15
    c4     (2,2,2)      9       1.43e-14   6.95e-15           28   7.45e-15
    slZ    (0,0,20)     4       4.49e-6    1.84e-1           100   4.12       (route (i) diverges)
    slZ    (0,0,24)     4       1.85e-6    9.03e-5           100   3.57e-8
    slZ    (2,0,0)      9       3.21e-12   6.80e-12           60   6.62e-15

and the Float64 error of route (i) as a function of q at two cells, showing that it has a **minimum**
(truncation falling, cancellation rising) — `q11.txt`:

    c32 (2,0,0)   q=40 7.9e-9  q=60 1.5e-10  q=80 3.2e-12  q=100 1.1e-13  q=120 9.7e-13  q=140 2.3e-11  q=160 1.1e-9
    c32 (2,1,0)   q=40 1.6e-10 q=60 9.7e-14  q=80 8.2e-15  q=100 8.8e-15  q=120 1.0e-14  q=140 1.6e-14  q=160 7.8e-15
    c4  (2,0,0)   q=40 2.7e-9  q=60 4.4e-11  q=80 9.4e-13  q=100 2.7e-14  q=120 1.3e-13  q=140 2.7e-12  q=160 1.1e-10

**The best route (i) can ever do at cubic two-cell separation in Float64 is 1.1e-13, at q = 100**
(≈ 4 ms per offset against Gila's 14 ms), and it is worse than Gila's existing order-9 rule at every
q ≤ 80. `1e-13 per entry is therefore not reachable at (2,0,0) in Float64 by route (i) alone.`

**A finding about Gila, not about route (i).** For `(1/32,1/32,1/512)` at `(0,0,20)` and `(0,0,24)`
Gila's own fixed rule is **4.5e-6 and 1.9e-6** relative — six orders worse than the 1e-10 the
`quadOrd` docstring claims. `quadOrd` keys the order on the max-norm separation in *cells*
(20 cells → order 4), but for this shape 20 cells along the short axis is 1/16 λ while the pair's
transverse extent is 1/32 λ, so the face-pair integrand is *near*, not far. The whole slender
short-axis band is broken in Gila today, independently of anything in this task; the root should
know, because it changes what the far-field replacement has to beat there.

---

## 3. Symmetries

Route (i) + `farTen` in Float64, 151 cases: 4 shapes × 7 offsets × 3 frequencies × q ∈ {24, 40}
(cases with `ρ = |s|/|R| > 0.95` skipped, which removes the aspect shape at short offsets).
All errors relative to `max_ab |T_ab|`. Shapes: `cub1/32`, `cub1/4`, `slZ = (1/32,1/32,1/512)`,
`asp = (0.001732, 0.003089, 0.24191)`. Offsets (3,0,0) (5,1,0) (7,-3,2) (13,13,4) (31,-17,9) (4,4,4) (2,2,0).

    shape      frq        rows  max |T(-D)-T(D)|  max reflection  max |T_ab-T_ba|  max permutation
    cub1/32    1                14   0               0               0              3.12e-16
    cub1/32    1+0.1i           14   0               0               0              7.88e-16
    cub1/32    3+0.3i           14   0               0               0              1.76e-15
    cub1/4     1                14   0               0               0              1.32e-15
    cub1/4     1+0.1i           14   0               0               0              1.36e-15
    cub1/4     3+0.3i           14   0               0               0              1.09e-14
    slZ        1                14   0               0               0              8.85e-17
    slZ        1+0.1i           14   0               0               0              3.46e-16
    slZ        3+0.3i           14   0               0               0              5.54e-16
    asp        1                 8   0               0               0              4.66e-16
    asp        1+0.1i            8   0               0               0              6.54e-16
    asp        3+0.3i            8   0               0               0              5.88e-15

`T(-D) = T(D)`, the three axis reflections `T_ab(D with D_m → -D_m) = σ_a σ_b T_ab(D)` (σ_m = -1),
and `T_ab = T_ba` hold **bitwise**, all 151 cases, both q values, every shape and frequency.
This is not luck and it is not evidence of much: negating `R` (or one component of it) multiplies
`a_α` by `(-1)^{|α|}` (resp. `(-1)^{α_m}`) with every intermediate in the recurrence flipping sign
consistently, so the Float64 arithmetic reproduces it exactly; `farTen` builds `M[A,B]` and `M[B,A]`
by summing the identical term sequence, so the transpose is exact by construction. **These three
tests cannot detect an error in route (i) and should not be quoted as evidence that it is right.**

The only symmetry that is *not* built in is the axis permutation, because `piv(g)` breaks the tie
among equal components by lowest index, so a permuted multi-index can take a different pivot and a
different rounding path. Tested as `T(s∘σ, D∘σ)_{ab} = T(s, D)_{σ(a)σ(b)}` over the 5 non-identity
permutations, this is the honest floor of the method:

    worst permutation violation overall     1.09e-14  at cub1/4, D=(5,1,0), f=3+0.3i, q=24
    worst inside Gila's operating range     1.76e-15  at cub1/32, D=(3,0,0)/(5,1,0), f=3+0.3i
    worst at f = 1                          1.32e-15  (cub1/4)

1.09e-14 is at λ/4 cells driven at f = 3, i.e. cells of 0.75 λ — outside Gila's stated range
(λ/128 to λ/4). Inside the range the permutation floor is 1.8e-15 relative to the largest entry.

Script: `work/auditG/s_q3.jl` → `q3.txt`.

---

## 4. Homogeneity in f

Derived from `g` and the volume form. Scale lengths by `1/λ` and the frequency by `λ`
(`s → s/λ`, `R → R/λ`, `f → λf`, so `k' r' = k r` and the integer offset `n` is unchanged). Then
`g'(r') = e^{ikr}/(4π λ² f² · r/λ) = g(r)/λ`, `w' dδ' = w dδ / λ⁶`, `1/V' = λ³/V`,
`∂'_a∂'_b + δ_ab k'² = λ² (∂_a∂_b + δ_ab k²)`, so

    **T(λ f, s/λ, n) = λ^{-2} T(f, s, n).**

(Equivalently, term `n` of the k-series carries `f^{n-2}` once lengths are measured in wavelengths.)
Also, at the coefficient level, `a_α(λf, R/λ) = λ^{|α|-1} a_α(f, R)`.

Tested at q = 24 over 3 shapes × 5 offsets × 2 frequencies × λ ∈ {2, 3.7, 0.5} (`q4.txt`):

    λ      max rel. violation, Float64   max rel. violation, BigFloat(200)   max coefficient-law violation
    2      0                             0                                   0
    0.5    0                             0                                   0
    3.7    7.21e-15                      4.12e-59                            4.49e-56

λ = 2 and λ = 1/2 give exact zeros because every quantity changes only by an exponent — the test is
vacuous at powers of two. The informative row is λ = 3.7: the law holds to the BigFloat floor
(4.1e-59 at 200 bits) and to 7.2e-15 in Float64 (worst case `asp`, D = (0,0,32), f = 1+0.1i).

Same law applied to the 220-bit **reference** at λ = 2:

    D = (5,1,0):  |T_ref(f=2, s=1/64) − T_ref(f=1, s=1/32)/4| / max = 0
    D = (7,-3,2): same, 0

again exactly zero and therefore uninformative for the same reason (the reference's arithmetic
differs only in exponents); it does confirm that no spurious constant enters the reference builder.

Script: `work/auditG/s_q4.jl` → `q4.txt`.

---

## 5. Anti-Hermitian positivity of a 6³ block at λ/32

Built exactly as Gila builds a self volume (`gilaToe` in `s_q5.jl`): `GlaVol((6,6,6),(1//32)³)`,
`CPUKerOpt(f, 48, false, CPU())`, `egoFunInn!` over all Toeplitz offsets, `egoFunSng!` (i.e.
`wekTrp`/`wekS,wE,wV`) over the eight offsets with all indices ≤ 2, then `egoToe[a,a,1,1,1] -= 1/f²`.
The second build replaces every offset with max-norm separation ≥ 2 by route (i) + the exact-moment
assembly at `q = 2⌈(9.7/log10(1/ρ) + 8)/2⌉` capped at 40; contact and touching shell are identical in
both. The dense 648×648 matrix uses Gila's own rule read off `egoToeCrc!`: `M[(c,a),(c',b)] =
egoToe[a,b,|d|+1] · σ_a σ_b` with `d = c − c'`, `σ_m = sign(d_m)` (the `SMatrix(1, fj*fi, fk*fi, …)`
mask is exactly `σ_a σ_b`). `M` comes out complex symmetric to 4.2e-16, so `(M − M')/(2i) = Im M`.

    f          build      λ_min            λ_max      eps·λ_max   max|M|    # eigenvalues < 0
    1          Gila       -4.147626e-15    0.0772947  1.72e-17    0.32951   242 / 648
    1          route (i)  -1.328363e-15    0.0772947  1.72e-17    0.32951   150 / 648
    1+0.1i     Gila       +2.855081e-03    0.1881490  4.18e-17    0.32640     0 / 648
    1+0.1i     route (i)  +2.855081e-03    0.1881490  4.18e-17    0.32640     0 / 648

At f = 1 the operator is lossless, `Im M` is positive **semi**-definite with a large null space, so
most of the 648 eigenvalues are exact zeros scattered by rounding; the meaningful number is the most
negative one. Route (i) moves it from **-4.15e-15 to -1.33e-15**, a factor 3.1 closer to zero, and
cuts the number of negative eigenvalues from 242 to 150. Both remain 77–240× the Float64 floor
`eps·λ_max = 1.7e-17`, because the residual is dominated by the contact and touching-shell entries,
which are identical in the two builds and are not what route (i) replaces.
At f = 1+0.1i the identity term `-1/f²` dominates and λ_min = 2.855081e-3 is identical to 12 digits
in both builds.

**The failure this test exposed.** Same run, same script:

    worst truncation tail |T_q − T_{q-2}| / max|T|  at q = 40:  8.04e-12, at offset (0,0,2)
    max |Gila − route(i)| / max|T| over all separated offsets:  3.23e-12, at offset (2,0,0)

At two-cell separation `ρ = √3/2 = 0.866` and the series has **not** converged at q = 40 — it is
still moving in the 12th digit. The 3.2e-12 gap between the two builds at (2,0,0) is route (i)'s
truncation, not Gila's quadrature error. This is quantified further in item 2 below.

Script: `work/auditG/s_q5.jl` → `q5.txt`.

---

## 6. Stress: order 50 and 60, kR to 3000, strongly complex k, and the seed

Digits lost `= log10(|a_Float64 − a_256bit| / (eps · |a_256bit|))`, maximised over the three
directions axis / face-diagonal / body-diagonal. Two metrics: **lvl** is famG's, which normalises by
the largest coefficient *of that total order*; **perCoef** normalises each coefficient by itself,
restricted to coefficients within 1e-8 of the level maximum. `sys` is famG's `taylorSys`; `seed` is
the identical recurrence with the seed `e^{ikρ}` supplied at 256-bit accuracy (`taylorSysE`,
`seed.jl`) — the difference between the two rows is exactly the seed's contribution.

                     lvl metric                          perCoef metric
    f       kR       n=10  20    30    40    50    60     10    20    30    40    50    60
    1       0.3      0.69  0.90  1.08  1.15  1.24  1.35   1.83  4.25  1.96  3.12  5.44  2.73
    1       1        0.84  1.08  1.31  1.39  1.48  1.60   1.78  3.31  2.51  4.06  3.77  3.16
    1       3        0.90  1.13  1.36  1.54  1.61  1.64   1.62  2.82  2.34  3.46  4.80  2.83
    1       10       0.73  1.73  1.85  2.02  2.06  1.93   0.96  3.92  3.78  3.27  4.57  4.93
    1       30       1.08  1.03  1.67  3.52  6.24  6.69   1.13  1.18  2.24  4.10  7.07  8.66
    1       100      1.66  1.67  1.68  1.70  1.72  1.69   1.68  1.69  1.71  1.76  1.97  2.75
    1       300      2.15  2.15  2.15  2.14  2.15  2.13   2.15  2.15  2.16  2.16  2.17  2.18
    1       1000     2.99  2.99  2.99  2.99  2.99  2.99   2.99  2.99  2.99  2.99  2.99  2.99
    1       3000     3.35  3.35  3.35  3.35  3.35  3.35   3.35  3.35  3.35  3.35  3.35  3.35
    1+1i    0.3      0.97  1.13  1.21  1.42  1.42  1.49   2.45  4.81  2.21  3.49  5.34  2.92
    1+1i    1        0.74  0.94  1.19  1.25  1.39  1.49   1.42  2.95  1.98  2.77  3.93  2.27
    1+1i    3        0.33  0.85  0.94  1.12  1.19  1.25   0.74  2.26  1.80  2.27  2.98  2.10
    1+1i    10       0.96  1.24  1.35  1.46  1.54  1.57   1.04  1.80  1.84  1.76  2.02  1.94
    1+1i    30       1.27  1.36  1.42  1.52  1.56  1.58   1.31  1.38  1.98  2.18  2.63  2.70
    1+3i    0.3      0.64  0.75  0.98  1.06  1.15  1.21   2.16  4.17  1.84  3.20  4.92  2.51
    1+3i    1        0.97  1.17  1.38  1.56  1.59  1.67   2.27  3.29  2.08  3.26  4.34  2.79
    1+3i    3        0.54  0.76  0.90  0.98  1.03  1.22   0.94  2.17  1.81  2.32  3.09  2.31
    1+3i    10       0.95  1.00  1.16  1.08  1.20  1.31   1.03  1.62  1.50  1.60  2.05  2.11
    1+3i    30       1.03  0.96  0.96  1.02  1.07  1.06   1.21  1.25  1.41  2.01  2.48  1.95

Three things famG's report does not say.

1. **Order 50–60 at kR = 30 is not stable.** famG stopped at 40 and saw 3.84 digits; at 50 and 60 the
   level metric reaches 6.24 and 6.69 and the per-coefficient metric 7.07 and 8.66. Beyond order ~45
   at kR = 30 route (i) delivers **7 significant digits at best**. famG's headline "stable to order 30
   everywhere measured: ≤ 2.15 digits lost" is true only for the level metric and only to order 30.
2. **The per-coefficient loss is 2 to 4 digits worse than the level metric everywhere**, including at
   kR = 0.3 and 1, where famG reports ≤ 1.4 digits. Individual coefficients within a factor 1e-8 of
   the level maximum lose 4.2–5.4 digits already at orders 20 and 50. In the assembly these coefficients
   are weighted by small moments, which is why the assembled tensor is nevertheless good (item 2), but
   the claim "≤ 1.4 digits lost for kR ≤ 3" describes the level maximum, not the coefficients.
3. **Complex k does not destabilise the recurrence.** At Im k/Re k = 1 and 3 the losses are the same
   as or smaller than at real k — the damping `e^{-2π Im f ρ}` suppresses the oscillatory cancellation.
   Nothing was found here.

### The seed, and a fix that works

For lattice offsets `R = n∘s` with dyadic `s`, `ρ² = Σ (n_i s_i)²` is exactly representable, so
`ρ = sqrt(ρ²)` is correctly rounded and the *only* Float64 error in `e^{2πifρ}` is `2πfρ · ulp(ρ)`.
The fix: carry `ρ` and the argument `θ = 2 f ρ` in double-double (two Float64s; `ρ` from one Newton
correction on `sqrt` with `fma`), reduce `θ` mod 2 exactly in double-double, and evaluate
`cispi(t_hi)·cispi(t_lo)`; for complex `f`, `e^{2πifρ} = e^{-2π Im(f) ρ} cispi(2 Re(f) ρ)` with the
real exponent also carried in double-double (`phsDD`, `seed.jl`, 12 flops).

    n                     kρ        |Δ|/|E| naive     |Δ|/|E| double-double
    (3,0,0)      f=1      0.6       4.71e-17          4.71e-17
    (100,0,0)    f=1      19.6      7.92e-17          7.92e-17
    (500,300,100) f=1     116.2     3.79e-15          2.64e-17
    (1000,0,0)   f=1      196.3     2.01e-14          1.13e-75
    (5000,3000,1000) f=1  1161.6    6.63e-14          6.16e-17
    (20000,0,0)  f=1      3927.0    6.07e-14          1.38e-74
    (100000,70000,30000) f=1  24680.8  4.43e-13       1.08e-17
    (1000,0,0)   f=1+0.1i 197.3     2.01e-14          1.81e-15
    (5000,3000,1000) f=1+0.1i 1167.4 6.71e-14         1.09e-14
    (20000,0,0)  f=1+0.1i 3946.6    8.68e-14          3.71e-14
    (100000,70000,30000) f=1+0.1i 24803.9  1.0        1.0

The last row is not a failure of the fix: `e^{-2π·0.1·3948} = e^{-2481}` underflows Float64 to zero,
which is correct to every digit that Float64 can carry. For complex `f` the residual 1e-14 at
kρ ≈ 4000 is the error of `exp` on the *damping* exponent `2π Im(f) ρ ≈ 2482`, whose Float64
representation is already 2482·eps in absolute terms; only the phase is recoverable, and it is.

Propagated through the whole recurrence, on lattice offsets at s = 1/32, f = 1 (level metric):

    n                     kR       naive seed: n=10 20 30 40      dd seed: n=10 20 30 40
    (3,0,0)               0.6      -0.15 0.08 -0.09 -0.09          -0.15 0.08 -0.09 -0.09
    (16,0,0)              3.1       0.16 -0.07 0.15 0.11           -0.04 -0.15 0.11 0.04
    (100,0,0)             19.6      0.22 0.85 2.02 2.02             0.22 0.97 2.06 2.05
    (500,300,100)         116.2     1.24 1.26 1.30 1.39             0.36 0.74 1.07 1.11
    (1000,0,0)            196.3     1.96 1.96 1.96 1.96             0.58 0.69 0.54 0.67
    (5000,3000,1000)      1161.6    2.48 2.48 2.48 2.48             0.54 0.66 0.87 0.95

and on the generic (non-lattice) offsets of the main table, the `seed` rows above give
kR = 300: 2.15 → 0.57; kR = 1000: 2.99 → 0.78; kR = 3000: 3.35 → 0.88 at order 20.

**So famG's "flat floor is the seed, and the value never degrades further" is right about the
diagnosis and wrong to leave it there: the floor is removable for 12 flops per offset, and removing
it recovers 2.5 of the 3.35 digits lost at kR = 3000.** It does nothing at kR ≈ 20, where the loss
(2.0 digits at order 30) is the recurrence itself.

Scripts: `work/auditG/s_q6.jl` → `q6.txt`; `seed.jl`.

---

## 7. Cost, re-measured

BenchmarkTools, single thread, load average 18–24 during the run (`q7.txt` records it).

    p     coefficients   taylorSys (µs)   ns per coefficient
    10    286            11.21            39.19
    16    969            17.75            18.32
    20    1771           31.25            17.65
    24    2925           54.29            18.56
    32    6545           119.67           18.28
    42    14190          249.17           17.56
    62    43680          786.33           18.00

    q     even terms     farTen (µs)      ns per term
    8     35             0.71             20.2
    12    84             1.62             19.4
    16    165            7.33             44.4
    20    286            12.29            43.0
    24    455            20.54            45.2
    30    816            15.50            19.0
    40    1771           75.71            42.8
    60    5456           103.71           19.0

**famG's 17.0 ns per coefficient is confirmed at 17.6–18.6 ns for p ≥ 16 under a load average of 20**,
i.e. the claim is not an artefact of a quiet machine. Below p = 16 the allocation of the two
`(p+1)³` arrays dominates and the figure is 39 ns/coefficient — relevant, because the *most common*
truncation degree in a real sweep is q = 10 (see below), i.e. p = 12. `farTen`'s per-term cost is
bimodal (19 vs 43–45 ns) and not reproducible run to run under this load; famG's 17.6 ns is the
optimistic branch.

Full sweep of all 32760 separated offsets of a 32³ Toeplitz block, cubic λ/32, f = 1, single thread,
q chosen per offset from ρ = √3/|n|:

    rule                                    total   µs/offset   q histogram (top)                   q>30  q>40  max q
    q = 2⌈(9.7/log10(1/ρ)+2)/2⌉  (fit)      1.28 s  39.1        10:19989 12:10234 14:1674 16:461     43    21    120
    q = 2⌈(⌈17/log10(1/ρ)⌉+2)/2⌉ (famG's)   4.29 s  131.1       16:18991 18:8258 20:2526 22:970     231    79    120

famG's own `q*` measurements do **not** support the `17/log10(1/ρ)` rule it quotes: `q* log10(1/ρ)`
over famG's seven cubic rows is 7.16, 8.72, 9.71, 9.30, 9.65, 9.63, 9.63 — a constant near 9.7, not
17. The 17-rule costs 3.35× more for nothing. Everything below uses the fitted rule.

Gila's current path on the same machine and load (`s_q7b.jl` → `q7b.txt`), `egoSrfFxd!` over the 36
face pairs at the `quadOrd` order:

    offset      sep  quadOrd   µs        offset      sep  quadOrd   µs
    (2,0,0)      2    9        14046.7   (8,0,0)      8    5        1342.8
    (3,0,0)      3    7         5180.6   (16,0,0)    16    5        1375.8
    (4,0,0)      4    7         5162.0   (17,0,0)    17    4         566.8
    (5,1,0)      5    6         2770.2   (32,0,0)    32    4         563.2
    (6,0,0)      6    6         2772.0

Offsets per order in a 32³ block: 27855 at order 4, 4570 at 5, 218 at 6, 98 at 7, 19 at 9, giving
**23.3 s single-threaded for the whole block**. Against route (i)'s 1.28 s that is a **18.2× speedup
for the block**, not the 39–151× that famG's per-offset table suggests: 85 % of the offsets of a 32³
block sit beyond 17 cells where Gila already drops to a 4⁴-point rule at 563 µs while route (i) still
needs q = 10–12 at 39 µs. famG's largest quoted speedup (151× at (16,0,0)) is measured against
`quadOrd`'s order 5, and the block average is dominated by order 4. With the 17-rule the block
speedup is 5.4×.

Neither figure is "a few hundred flops per offset": at q = 10, route (i) computes 455 coefficients
plus 35 assembly terms, of the order of 5·10³ flops, and the sweep average is 39 µs.

Scripts: `work/auditG/s_q7.jl` → `q7.txt`; `s_q7b.jl` → `q7b.txt`.

---

## 8. Two checks the brief did not ask for

**Trace identity.** Because `∇²g = −k²g` holds level by level on the Taylor coefficients
(`Σ_a (β_a+2)(β_a+1) a_{β+2e_a} = −k² a_β` for every β), the assembly must satisfy, **exactly at
every truncation degree q**,

    tr T_q = Σ_a T_aa = 2 k² · (1/V) Σ_{β even, |β| ≤ q} μ_β a_β = 2 k² ⟨g⟩_q .

This is a sharp internal check of the recurrence and of `farTen` together, needing no reference.
Over 108 cases (4 shapes × 5 offsets × 3 frequencies × q ∈ {20, 40}, ρ ≤ 0.95):

    max |tr T − 2k²⟨g⟩| / |tr T|,  Float64        1.31e-14
    max |tr T − 2k²⟨g⟩| / |tr T|,  BigFloat(200)  2.45e-58

So the identity holds to the BigFloat floor — the recurrence really does produce Helmholtz-consistent
coefficients and `farTen` really does implement the stated sum — and the Float64 realisation of the
pair is good to 1.3e-14, an independent estimate of the assembled rounding error that agrees with
famG's `roundDig ≤ 1.42 digits`.

**`gt` silently truncates.** `farTen(a, s, f, q)` called with `q > size(a,1) − 3` reads unwritten
zeros instead of raising: `gt` only guards indices above `p` per axis, not total degree above `p`.
Every result in famG's report is generated by scripts that pass `q + 2`, so nothing there is wrong,
but the function is a trap for any later caller. My `farSeq` errors instead.
