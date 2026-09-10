# Where a coarse grid stops disagreeing with a refined one

Question (Paul, 2026-09-10): the deleted proximity warning was about **resolving
the current distribution**, not entry quality. Compare top singular values of the
external vacuum Green operator, coarse grid against a 2x refined one at matched
physical geometry, and find where they agree.

Scripts `scratch/sig{1..4}.jl` and `scratch/sweep.jl`; raw output
`scratch/sweep.log`. Dense `svdvals`. `frqPhz = 1` so a cell of `1//n` is
`lambda/n`. Singular values compare directly across grids with no reweighting:
Gila takes unit-density pulse coefficients and returns cell-averaged fields,
which is the matrix of the continuous operator in an L2-orthonormal basis.
"Crossover" below = the gap at which the coarse-vs-fine difference has fallen to
within 5 % of its wide-separation value, log-interpolated.

## 1. Two superposed effects

**A gap-independent part**, `= (pi s/lambda)^2 / 4` on sigma1, measured 9.46e-03
at lambda/16 and 2.37e-03 at lambda/32 against 9.64e-03 / 2.41e-03 predicted.
This is the pulse basis: source integrated over a cell, field averaged over a
cell, two `sinc(pi s/lambda)` factors.

**It converges — this is not a compactness failure.** The external kernel is
analytic on disjoint supports, so the operator is compact and smoothing. The
scheme is second order: same body, changing only the grid pair, sigma1 gives
9.46e-03 -> 2.37e-03 and top5 gives 6.185e-02 -> 1.550e-02, both ratio **3.99**.
By Richardson the lambda/16 grid's own sigma1 error is 1.26e-02 against
`(pi/16)^2/3 = 1.285e-02` predicted. What separation cannot fix is not the same
as what refinement cannot fix. Deeper singular values carry a larger constant
(top5 ~6.5x sigma1) at the same h^2 rate.

**A gap-dependent excess** above that. This is what the warning was about.

## 2. The crossover is a fixed physical distance, set by body size

Grid resolution does not move it; body size does. Same body at two resolutions:

| body | grid | crossover | in coarse cells |
|---|---|---|---|
| lambda/8 cube | lambda/16 | 0.3116 lambda | 4.99 |
| lambda/8 cube | lambda/32 | 0.3027 lambda | 9.69 |
| lambda/4 cube | lambda/16 | 0.2265 lambda | 3.62 |
| lambda/4 cube | lambda/32 | 0.2256 lambda | 7.22 |

The distances agree to 3 % and 0.4 %; the cell counts differ by exactly the
factor of 2 between the grids. A cells-based rule is wrong.

**Bigger bodies converge sooner**, in any direction (lateral 4x4, lambda/16):

| extent along gap | crossover | | lateral extent | crossover |
|---|---|---|---|---|
| lambda/16 | 0.334 | | lambda/8 | 0.249 |
| lambda/8 | 0.290 | | lambda/4 | 0.227 |
| lambda/4 | 0.227 | | lambda/8 x 3 | 0.187 |
| lambda/8 x 3 | 0.187 | | | |
| lambda/2 | 0.165 | | | |

Reading: a larger body's leading mode is a smoother, longer-wavelength current
that the coarse grid captures relatively better, while the proximity error is
local to the facing surfaces. **So the worst case is the SMALLEST body**, and a
threshold has to be set from that end: worst measured is **0.334 lambda** for
sigma1 and **0.547 lambda** for top5, both from the thinnest body.

## 3. Loss makes it HARDER, not easier (hypothesis refuted)

Cube of 4 cells at lambda/16, sweeping `Im(frqPhz)`:

| frqPhz | sigma1 floor | sigma1 crossover |
|---|---|---|
| 1.0 | 9.51e-03 | 0.227 lambda |
| 1 + 0.05i | 9.69e-03 | 0.256 |
| 1 + 0.1i | 9.92e-03 | 0.302 |
| 1 + 0.2i | 1.06e-02 | 0.391 |
| 1 + 0.5i | 1.37e-02 | 0.538 |
| 1 + 1i | 2.26e-02 | 0.576 |

Monotone in both columns, so **`Im(f) = 0` is the EASIEST case, not the
hardest.** Two mechanisms, both pushing the same way: loss raises `|k|`, and the
`exp(-2 pi gamma r)` factor concentrates the interaction on the facing surfaces,
which is the sharpest and worst-resolved part of it. At `1 + 1i` the decay length
is 0.16 lambda, only 2.5 cells at lambda/16. The measurement is conservative:
those curves have not fully plateaued by the end of the sweep, which biases the
crossover *early*. No simple rescaling collapses them — dividing by `|n|` makes
the spread worse (0.227 -> 0.815), so loss has to be handled explicitly or
covered by a conservative constant.

## 4. Anisotropic cells

Wired and correct; all three variants build and return finite values.

**The floor is set by the cell size along the coupling axis**, not by the largest
cell dimension:

| cells (gap axis first) | sigma1 floor | `(pi s_gap/lambda)^2/4` |
|---|---|---|
| lambda/8, lambda/32, lambda/32 | 3.70e-02 | 3.85e-02 |
| lambda/16, lambda/32, lambda/64 | 9.21e-03 | 9.64e-03 |
| lambda/32, lambda/8, lambda/8 | 4.89e-03 | 2.41e-03 |

The first two match the gap-axis prediction and ignore the transverse cells
entirely. The third is 2x its gap-axis prediction, so a coarse *transverse* cell
does contribute once the coupling axis is much finer, but weakly.

**Caveat: the anisotropic curves are non-monotone in gap** — the ratio dips below
1 mid-sweep (to 0.83-0.89) and returns, i.e. the coarse-fine difference passes
through a partial cancellation. A single crossover number is not meaningful for
those configurations and the ones in `sweep.log` should not be quoted.

## 5. What this implies for a restored warning

- Phrase it in **wavelengths, not cells**. `lambda/4` is 4 cells at lambda/16, 8
  at lambda/32, 16 at lambda/64; "6 cells" only coincides near `s = lambda/24`
  and grows too lenient as the user refines — backwards from what a warning
  should do.
- A lossless threshold of `d < lambda/3` covers every lossless configuration
  measured (worst 0.334 lambda) for sigma1. Covering top5 needs `lambda/2`, and
  covering `Im(f) = 1` needs `0.6 lambda`.
- Do not promise that separation buys accuracy. The `(pi s/lambda)^2` part is
  always present — ~1.3 % on sigma1 at lambda/16, and tens of percent on deep
  singular values — and only refinement reduces it.
