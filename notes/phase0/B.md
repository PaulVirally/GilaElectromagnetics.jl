# Phase 0-B report: near-field measurements (moments.jl)

Repo `/Users/pvirally/.julia/dev/GilaElectromagnetics`, branch `paul-farfield`.
Baseline commit: `5965fdf0a6b5709bafd3c4e25ac76a3d7e0051d6` (2026-09-09 10:27:04 -0400,
"Save Claude's notes for the analytic integrals in Gila").

**Tree-cleanliness caveat.** `git status` at the start of this task already showed
`notes/PLAN_wiring.md` modified (not by this agent — no edit to that file was made
here; the diff was present before this agent's first command ran, presumably Paul's
own approval edit or Phase 0-A). Everything under `src/` and `test/` was clean for
the whole task. Files touched by this agent: everything under `notes/phase0/`
(`scratch-b/b12.jl`, `scratch-b/b3.jl`, `scratch-b/probe{,2,3}.jl`, their `*.stdout`
and `*_out.txt` outputs, `scratch-b/testsuite.stdout`, this file). Nothing under
`src/`, `test/` or `notes/farfield/` was read for modification purposes (only
`src/vacuum/glaVacOprMemGen.jl` and `glaVacOprMemInt.jl` were **read** to confirm
`srfSum!`/`facPar`/`srfScl`'s exact index conventions for item 3 — no edits).
Nothing was committed.

Not done, per the task's explicit exclusions: no `Double64` work (§3.9), no
precision-escalation constant fit (§3.8) — item 3 below is the one confirming
number only.

## 1. Face-pair cost: 288 pairs/shape, mMax = 12 and 30, serial and threaded

Command: `JULIA_NUM_THREADS=auto julia --project --startup-file=no
notes/phase0/scratch-b/b12.jl` (item 1 section). `Threads.nthreads() = 6` as seen
by the process (`JULIA_NUM_THREADS=auto` on this 12-logical-core, 6P+6E Apple
Silicon machine resolves the default thread pool to 6 — the same figure came back
from two independent processes on this machine with the same env var). Best of 3
repeats per cell, warm (post-compile). Shapes are `farfield/verify.jl`'s four
production shapes (`SHP`): `c32=(1/32)^3`, `c8=(1/8)^3`, `c4=(1/4)^3`,
`sl=(1/32,1/32,1/512)`. The eight contact offsets are `D_i in {0,1}`, i.e. the
domain `genEgoCrcSlf!` hands to `egoFunSng!` (`posItr` indices 1 or 2 per axis,
`src/vacuum/glaVacOprMemGen.jl:291-325`) — the only offsets `cntBlk!` will ever be
asked to fill for a self block, regardless of grid size (Toeplitz symmetry
supplies the rest).

| shape | mMax | serial (s) | serial (us/pair) | threaded (s) | threaded (us/pair) |
|---|---|---|---|---|---|
| c32 | 12 | 0.09138 | 317.27 | 0.02335 | 81.08 |
| c32 | 30 | 0.92274 | 3203.96 | 0.34480 | 1197.21 |
| c8  | 12 | 0.09183 | 318.86 | 0.02577 | 89.48 |
| c8  | 30 | 0.91767 | 3186.36 | 0.32495 | 1128.28 |
| c4  | 12 | 0.08984 | 311.94 | 0.02152 | 74.72 |
| c4  | 30 | 0.90070 | 3127.45 | 0.32119 | 1115.23 |
| sl  | 12 | 0.07602 | 263.96 | 0.01961 | 68.11 |
| sl  | 30 | 0.89123 | 3094.54 | 0.32233 | 1119.20 |

Per-pair serial cost at mMax=12 (264-319 us) matches `tables/e_timing.txt`'s
existing per-sub-code figures (84-942 us for cube/slender at mMax=12) — sanity
check passes. Cost at mMax=30 is ~10x the mMax=12 cost, not 2.5x: `box3`'s
internal recursion is not O(mMax) amortized across the ladder the way `box2lad`
is (see item 2's cost note below); this is a real, shape-independent effect, not
noise (all four shapes show the same ~10x ratio to within 3%).

Threading gives only 2.7-4.4x on 6 threads (best at mMax=12: 3.9-4.4x; at mMax=30:
2.7-2.9x), not 6x — expected for a 288-iteration loop whose total serial cost is
under 1s; per-shape build-budget relevance is moot regardless, since **this whole
288-pair cost is a one-time, grid-size-independent cost per (shape, mMax)**: only
eight offsets exist for a self block no matter how large `slfVol.cel` is, so it
does not compete with §7's 0.5s/128³ near-field budget in any way that scales —
at under 1s worst case (serial, mMax=30) for the *entire* near-field contact
assembly of one shape, it is not a factor in the 128³ budget at all. This does
not cover whatever "touching shell" work beyond these eight offsets exists for
composite/external contact (`egoCntOut!`/`genCntVol`, decision 10) — out of this
item's literal scope ("288 face pairs" = 8 x 36, exactly the eight offsets).

## 2. Series accuracy vs a 256-bit sum

Command: same `b12.jl` run, item 2 section. Five canonical touching-shell
sub-codes (S, Ef, Ec, Vc, Vp — same set as `moments.tex`'s "The weak integrals"
section) via `canonPanels`, at six cell scales (`l32, l8, l4, l2, l1` = cube
edge 1/32, 1/8, 1/4, 1/2, 1; `sl` = the slender production shape) and seven
frequencies `f in {1, 0.37, 3, 1+0.1i, 1+1i, 0.5+2i, 2+2i}` — 210 combinations.
mMax search: a priori guess from `serGss` (Gaussian-tail estimate of
`(2*pi*|f|*rMax)^n/n! < tol`) refined by up to 4 more `momentSeries` calls at
1.3x/1.7x/2x the guess, capped at mMax=100 (**not** a unit-step scan: a unit-step
scan to mMax~140 was measured at 95s for a single combination — `box3`'s
per-`m` cost is not memoized across `m`, roughly cubic in `mMax`, so a linear
scan is roughly quartic in the target order). 256-bit reference: `momentSeries` at
the **same** mMax the Float64 search found (not a separate search) — this is
valid because the 256-bit floor has ~200 extra bits of cancellation headroom over
Float64, so a term count sufficient for Float64's own truncation criterion is
enough for the 256-bit sum to be trustworthy to below the Float64 comparison
floor (worst observed 256-bit self-consistency indicator among "converged" rows:
6.04e-17, itself below `eps(Float64) = 2.22e-16` — see the l1/Ef/0.5+2i finding
below).

**Term counts** (min-max mMax found over the 5 subcodes x 7 freqs, i.e. a wider
frequency sweep than `moments.tex`'s 10-13/15-21/20-29 at f in {1, 1+0.1i}):

| scale | mMax range |
|---|---|
| l32 (lambda/32) | 10-22 |
| l8 (lambda/8) | 14-42 |
| l4 (lambda/4) | 18-63 |
| l2 (lambda/2) | 22-100 (capped) |
| l1 (lambda) | 30-100 (capped) |
| sl (slender, production) | 10-22 |

**Worst digits lost vs the 256-bit sum, per scale** (`log10(|v64-vBig|/(eps64*|vBig|))`,
max over the 5 subcodes x 7 freqs, negative values from converged rows clipped to
not affect the max — they represent sub-ulp agreement):

| scale | worst digits lost |
|---|---|
| l32 | 0.512 |
| sl  | 0.512 |
| l8  | 1.609 |
| l4  | 4.265 |
| l2  | 9.857 |
| l1  | 10.227 |

l32 and sl (0.512) are consistent with `moments.tex`'s own 0.91-digit `pairMoments`
bound (this measurement adds the series-summation step on top and stays under 1
digit at the two scales that matter for production, confirming the existing
number rather than contradicting it). **l4 is already the finding**: 4.265 digits
lost is well past the 1-digit bar `moments.tex` documents for `pairMoments` alone,
first appearing at `Vc`, `f=2+2i`, mMax=61. l2 and l1 lose up to ~10 digits — most
of a full Float64 mantissa.

**Mechanism (why, not just how much).** `pairMoments` itself is proven
cancellation-free (<=1.28 digits at m<=30, `moments.tex`), so the loss is not
there. It is in `momentSeries`'s own complex sum: `cof *= z/(n+1)` with
`z = 2*pi*i*f` grows the running term to a peak near `n ~= 2*pi*|f|*rMax` before
factorial decay brings it back down, and the terms rotate in phase (complex
`z^n`), so the **partial sum** transiently holds a value far larger in magnitude
than the converged answer, then cancels back down to it. Rounding error scales
with the partial sum's peak magnitude, not the final one — classic catastrophic
cancellation, amplified by how far the peak exceeds the final value (grows with
`2*pi*|f|*rMax`). Concretely: `l1, Ef, f=0.5+2i, mMax=100` has `momentSeries`'s
own convergence indicator `rel = |last term|/|partial sum| = 6.04e-17` — which
would normally be read as ~16 correct digits — yet the value is wrong by
**10.227 digits** against the 256-bit reference. **The built-in `rel` indicator
does not catch this failure mode**: it certifies term-decay (truncation), not
partial-sum conditioning (cancellation), and the two diverge exactly when the
peak/final ratio is large. This is a real hazard only if `momentSeries`'s own
`rel` were ever trusted as a correctness gate outside the regime it was measured
in (`moments.tex`: lambda/32-lambda/4, f in {1, 1+0.1i, 0.37}). It does not touch
decision 8 (contact block fixed at 128 bits, always small cells) or production
cell scales (lambda/32-lambda/8, comfortably under 1 digit lost here too), but it
is a genuine limit of the tool if ever reused at coarser scale/larger |f|.

**Non-convergence** (10 of 210 combinations did not reach `rel < 1e-16` by
mMax=100, all at `l1`, all higher-|f|): `(l1,Ef,3)`, `(l1,Ef,2+2i)`,
`(l1,Ec,3)`, `(l1,Ec,2+2i)`, `(l1,Vc,3)`, `(l1,Vc,0.5+2i)`, `(l1,Vc,2+2i)`,
`(l1,Vp,3)`, `(l1,Vp,0.5+2i)`, `(l1,Vp,2+2i)`. Same mechanism: at `f=3` or
`f=2+2i` on a full-wavelength cell, `2*pi*|f|*rMax` is large enough
(`rMax` up to `sqrt(6)` for Vp, so x up to ~55 at `f=3`, ~65 at `f=2+2i`) that
100 terms is not past the point where cancellation-limited `rel` can reach
1e-16 in Float64 at all — raising the cap would not obviously fix this, since the
mechanism is cancellation, not under-truncation (see l1/Ef/0.5+2i above, which
*did* nominally converge and was still wrong by 10 digits). **None of this
touches the production regime**: mesh cells at lambda scale or larger are never
used in Gila's near field (the contact block only ever sees the actual
discretization cell, always well under lambda), so this is a boundary-stress
finding about the tool's range of applicability, not a production defect.

## 3. Confirming number: srfSum! amplification Lambda on the eight contact offsets

Command: `julia --project --startup-file=no notes/phase0/scratch-b/b3.jl`
(single-threaded; loads `GilaElectromagnetics` read-only for `GV.facPar()` and
`GV.srfSum!`, `GV = GilaElectromagnetics.GilaVacuum`). `facPar()`'s `(target
face, source face)` -> `k=(i-1)*6+j` convention matches `moments.jl`'s
`facePair(D,F,Fp,s)` `F=target, Fp=source` exactly (confirmed by reading
`src/vacuum/glaVacOprMemInt.jl:308-321`'s comment, not by modifying anything), so
`momentSeries(facePair(D,F,Fp,s)..., f, mMax)` for `k=1..36` is a physically
scaled `srfMat` vector with no separate `srfScl` step needed (unlike Gila's
reference-domain quadrature, `moments.jl` evaluates directly in physical
coordinates). `srfSum!`'s own index sets (which 4 or 8 of the 36 `srfMat` entries
feed each of the 9 tensor entries) were read from
`src/vacuum/glaVacOprMemGen.jl:734-756` and reproduced verbatim in the scratch
script, not re-derived. Same four production shapes, same eight contact offsets
as item 1, all seven frequencies of item 2. `eps` at 128 bits (`setprecision(BigFloat,
128); eps(BigFloat)`) = **5.877471754111438e-39**.

Lambda := `(sum_fp |srfMat_fp|) / |G_ab|` over the pairs feeding the entry, per
(shape, offset, frequency, entry). 1053 (shape, offset, freq, entry) rows
computed; 45 of them have `|G_ab| < 1e4 * eps(Float64) * sum|srfMat_fp|`, i.e.
the entry is consistent with zero given the rounding floor of the terms that
feed it.

**These 45 rows are the loud finding, and they are an artifact, not a
precision risk.** They occur exclusively at off-diagonal entries for offsets
with a zero axis component (e.g. `D=(1,0,0)` entry `(2,1)`, `D=(1,1,0)` entry
`(3,2)`): the entry is an **exact structural zero** by mirror symmetry of the
offset (odd under a reflection that leaves the offset invariant), so
`sum|srfMat_fp| ~ 1e-6..1e-4` cancels to a residual of `~1e-22..1e-25` —
consistent with the rounding floor of the summed terms, not with the entry
itself carrying any information. Dividing by that residual gives Lambda up to
**1.47e18** (worst: `sl`, `D=(1,1,0)`, `f=1+0.1i`, entry `(3,2)`), which is
`Lambda` in the letter of the definition but not in its intent (Section 3.3 of
`groundwork.tex`/`join.tex` tabulate `amp` for entries that carry the physics,
not zero/zero ratios). Raw worst per shape: c32 3.50e17, c8 2.35e17, c4 2.35e17,
sl 1.47e18.

**Worst Lambda excluding the 45 structural-zero rows** (the number the decision
is actually about):

| shape | Lambda | at (offset, freq, entry a, b) |
|---|---|---|
| c32 | 2261.8 | D=(1,1,1), f=0.37, (1,1) |
| c8  | 141.65 | D=(1,1,1), f=0.37, (1,1) |
| c4  | 35.65  | D=(1,1,1), f=0.37, (1,1) |
| sl  | 2255.2 | D=(1,1,1), f=0.37, (1,1) |

Worst case is the fully-diagonal contact offset (self-touching corner neighbor),
lowest frequency tested. This is above the "10-1000" range item 3's brief quotes
for far-field separated pairs (`groundwork.tex`), by about 2-4x — not an
order-of-magnitude blowout, and the same order of magnitude as `farfield`'s own
route-3 figure of `Lambda <= 1002` (a different offset population: separated
pairs' 36-panel k-series sum, not the contact block's 8 offsets, so exact
agreement isn't expected). It is **not** in the same regime as the far-field
document's other quoted extreme, "1.67e5 on the slender cell's long axes"
(that number is for far-field route 3 at large separation, a different geometry
regime from the eight D in {0,1}^3 contact offsets measured here).

**eps_128 x Lambda**, both readings:

| reading | worst Lambda | eps_128 x Lambda |
|---|---|---|
| raw (incl. structural zeros) | 1.4709e18 (sl) | 8.645e-21 |
| excluding structural zeros | 2261.8 (c32) | 1.330e-35 |

Either way, `eps_128 * Lambda` is **21 to 35 orders of magnitude** below any
plausible target (1e-13, or even 1e-30) — decision 8's "beaten by precision"
claim is confirmed with enormous margin for the contact block specifically, well
beyond what the far-field route-3 figure (`eps64*Lambda <= 2.23e-13`, i.e. at
Float64, not 128-bit, precision) needed to justify escalating to 128 bits in the
first place. No fitting was done, per the task's instruction; this is the one
confirming number.

## 4. Test suite baseline

Verified `git status` clean under `src/` and `test/` before running (see the
tree-cleanliness caveat above for the pre-existing, out-of-scope
`notes/PLAN_wiring.md` diff). Baseline commit `5965fdf0a6b5709bafd3c4e25ac76a3d7e0051d6`.

Command: `JULIA_NUM_THREADS=auto julia --project -e 'using Pkg; Pkg.test()'`
(stdout at `notes/phase0/scratch-b/testsuite.stdout`).

**Result: 1819 pass, 3 broken, 1822 total, 3m20.3s (200.3s) wall time, exit 0.**

The 3 broken are pre-existing and accounted for by source inspection (not run
output, since Julia's `Test` summary does not name individual broken tests):
- `test/physTest.jl:69`, `@test_broken` — "Asym(G0) PSD — overlapping as union";
  comment states this is a known limitation (masked/union operators break the
  self-operator PSD property).
- `test/vacTest.jl:53`, `@test_skip` — "overlap guard not practically testable:
  hangs during integration".
- `test/quadTest.jl:333`, `@test_skip` — "no functional CUDA device" (this
  machine has no GPU; `Test`'s summary counts `@test_skip` under "Broken" too,
  which is why the total is 3, not 1).

**Thread count: not independently verified inside the `Pkg.test()` subprocess**
(the suite prints no thread count, and instrumenting `test/runtests.jl` to print
one is off-limits under this task's file restrictions). Inferred only: the same
`JULIA_NUM_THREADS=auto` environment variable on the same machine gave
`Threads.nthreads() = 6` in both item 1/2 and item 3's independent processes, and
per the plan (§4.1's note, repeated in this task's brief) the env var — not `-t`
on the parent — is what reaches the `Pkg.test` subprocess, so 6 is very likely
correct but is not a directly observed number for this specific run.

## What could not be verified, plainly

- **Item 2's mMax values are a-priori-guided upper estimates, not exhaustively
  verified minima.** A true unit-step minimum search was measured to cost
  ~95s for a single (l1, Vp, 2+2i)-scale combination and was abandoned as
  infeasible for a 210-combination grid; the geometric search (guess,
  1.3x, 1.7x, 2x, cap) can overshoot the true minimum by a modest factor. This
  does not affect the digits-lost or non-convergence findings (those use
  whatever mMax was actually found), only the precision of "N terms are
  needed" as a minimum-term claim.
- **The `l1`/`l2` non-convergent rows' true error is unknown**, only that
  `rel >= 1e-16` at the mMax=100 cap; I did not push further (per the mechanism
  established by the l1/Ef/0.5+2i case, higher mMax would not obviously help,
  and confirming that would need a much larger compute budget than this task's
  2-concurrent-process, single-session budget allows).
- **Item 1 and item 3's thread count (6) reflects this machine's
  `JULIA_NUM_THREADS=auto` resolution** (6P+6E Apple Silicon, auto picks the
  performance-core count for the default pool); a different machine could see a
  different number, and this was not cross-checked against a manually-set
  `JULIA_NUM_THREADS=12`.
- **Item 4's thread count is inferred, not observed**, as stated above.
- **"Touching shell" beyond the eight D in {0,1}^3 contact offsets** (whatever
  composite/external-volume touching work `cntBlk!`'s docstring implies beyond
  the self-block eight, per decision 10's `egoCntOut!`/`genCntVol` path) was not
  measured in any of items 1-3: those items' briefs specify "288 face pairs"
  and "the eight near (contact) offsets" exactly, which is what was measured.
- No `Double64` work and no precision-escalation constant fit, per explicit
  task exclusion (§3.9, §3.8) — not attempted, not reported as a gap.
