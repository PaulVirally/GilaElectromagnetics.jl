# Prompt: unequal cells in the far-field expansion (and one near-field regime)

For the agent that produced `notes/farfield/farfield.jl` and `notes/moments/moments.jl`. Gila is
about to wire both libraries into `src/`; the one case they do not yet cover is a pair of cells with
**different edge lengths** `s_t ≠ s_s` (composite volumes, `GlaCmpVol`, where a coarse region meets
a fine one). Touching unequal cells are already handled by Gila through the gcd cell
(`genCntVol`/`egoCntOut!` average fine-cell tensors), so only **separated** unequal pairs matter.

## What changes mathematically

Nothing in the mechanism; only the difference-box weight. For a target cell of edges `s_t` at
centre `R` and a source cell of edges `s_s` at the origin, the per-axis convolution of the two
intervals is a **trapezoid** instead of a triangle:

    w_i(δ) = min(s_t,i, s_s,i)                      for |δ| ≤ a_i = |s_t,i − s_s,i| / 2
           = b_i − |δ|                              for a_i ≤ |δ| ≤ b_i = (s_t,i + s_s,i) / 2
           = 0                                      beyond,

with `∫ w_i = s_t,i s_s,i` and kinks at `±a_i` (none at 0: the plateau is flat). The volume form
`G_ab(R) = (1/V_t) ∫_D w [(∂_a∂_b + δ_ab k²) g](R + δ) dδ` with `D = Π[−b_i, b_i]` and `V_t` the
target volume is otherwise unchanged (`srfMat[fp] = I_FF'/V_t` still holds: `A_F s_c(F) = V_t`).

## What to deliver

1. **Whole-box table**: replace the triangle 1D moments `wgtT(n, s)` by the trapezoid moments
   `∫_{−b}^{b} w(t) t^n dt` (exact rationals in `s_t,i, s_s,i`; odd `n` still vanish, so the parity
   argument and the four surviving `(l,m)` classes are unchanged). Table key becomes the pair
   `(s_t, s_s)` (ordered: the weight is symmetric so `(s_t, s_s)` and `(s_s, s_t)` share a table up
   to the `1/V_t` factor, say which).
2. **Sub-box split**: a piece may not straddle a kink. With kinks at `±a_i` the minimal legal
   split is three pieces per axis, `[−b,−a]`, `[−a,a]`, `[a,b]`: ramps with affine weight
   `α + βt` (existing `wgtTrpA` machinery) and a plateau with `β = 0`. Twenty-seven sub-boxes,
   reflection rule as before (the plateau piece is its own mirror). If `a_i = 0` on an axis
   (equal edges there) fall back to the two-piece split on that axis.
3. **Bounds**: Theorem A/B radial moments (`farMomW`, `farMomO`) and the scale `est(R)` for the
   trapezoid weight; `r_d = |b|` is the analyticity radius. State what changes in the proofs (I
   expect: nothing beyond the moment values, since both need only `w ≥ 0` and the product form).
4. **Routing** on a real offset vector `R` (external pairs are not on the source lattice), and
   the k-series route on arbitrary rectangle pairs (`pairMoments` already takes them).
5. **Near field, thin-indicator regime**: for a perpendicular pair whose indicator axis is narrow
   relative to its offset (`2(hi − lo) < hi`), `moments.jl` currently uses a precision-adaptive
   Gauss–Legendre rule along that axis (Section "The off-lattice thin-indicator branch"). Unequal
   cells reach this regime routinely (a fine face against a coarse perpendicular face a coarse cell
   away). The 3D analogue of the 2D centred Taylor series (which removed the same problem for the
   doubly shifted 2D box) would make the near field closed-form all the way down; deliver it if it
   exists, or a measured statement of the digits the current branch loses over the cross-scale
   geometries Gila produces (scale ratios 2, 4, 8, 16 per axis, all orientations).
6. **Verification** as before: exact-rational identity checks on polynomial kernels for the new
   weight, 220-bit references (`pairKer` on the 36 face pairs of unequal cells, the graded volume
   rule) at scale ratios 2, 4, 8, 16 in one, two and three axes, separations 2–32, `f ∈ {1, 1+0.1i,
   0.37}`, digits lost in Float64, and a timing of the table build against the equal-cell table.

Deliverables in `notes/farfield/` (a new `crossscale.jl` or an extension of `farfield.jl`, the
verification script and tables, and a short section appended to the document). Do not touch `src/`.
