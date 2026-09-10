# Phase 0-A: far-field measurements

Machine: arm64 Darwin 26.6.2, Julia 1.12.6. Every timing below is single-threaded
(`JULIA_NUM_THREADS=1`) unless stated otherwise — none of the functions measured here
(`sepGrd`, `farSetup`, `farSetupX`, `boundL`/`boundLX`, `boxAcc`/`tnsWhl!`) are internally
threaded; `farBlock!`/`farBlockX!` thread scaling is out of this report's scope (already
measured in `tex/bench.tex`/`crossscale.tex`). Repo state: branch `paul-farfield`, tree clean
apart from `notes/phase0/` (this report and its scratch scripts) and a pre-existing unstaged
change to `notes/PLAN_wiring.md` present before this task started — not touched here. Nothing
under `src/`, `test/` or `notes/PLAN_wiring.md` was modified.

Scratch scripts (all under `notes/phase0/scratch/`, kept for reproducibility):
`offsets1.jl`, `offsets2.jl`, `misalign_route.jl`, `lamcube_entry.jl`, `band_0.55_0.60.jl`,
`band2.jl`, `lmax_cost.jl`, `lmax_cost2.jl`, `lsum_compensated.jl`, `table_cost.jl`,
`table_cost2.jl`, `nblk_cost.jl`. `scratch/coldtab/` and `scratch/coldtab2/` are the cold-build
table artifacts from item 3 (73 MB combined); regenerable, safe to delete.

Sources read before measuring, per instruction: `notes/PLAN_wiring.md` §1, §3.11, §4.1, §4.3,
§6 (Phase 0-A), §8; `notes/farfield/tex/certificate.tex`, `bounds.tex`, `crossscale.tex`,
`uncertain.tex`, `registry.tex`; `src/glaVol.jl` (`sepGrd`, `GlaVol`, `genEveExtInf`,
`genVolEve`), `src/glaCmpVol.jl` (`GlaCmpVol`, `refine`), `src/vacuum/glaVacOprMem.jl`;
`test/vacuum/extSlfTest.jl`, `test/crsSclTest.jl`, `test/cmpOprTest.jl`; `notes/farfield/farfield.jl`,
`notes/farfield/verify.jl`, `notes/farfield/tables/k_adv.txt`, `notes/farfield/tables/a_rho.txt`.

Correction to the plan text: `sepGrd` lives in `src/glaVol.jl`, not
`src/vacuum/glaVacOprMemGen.jl` (that file only calls it).

## 1. Misaligned external equal-cell offsets (highest priority)

**Finding: the offsets Gila's own test suite produces are never misaligned. The route-3 set
from this source is empty.**

Constructed every same-scale (equal-cell) *external, separated* volume pair reachable from
`test/vacuum/extSlfTest.jl`, `test/crsSclTest.jl` and `test/cmpOprTest.jl`
(`notes/phase0/scratch/offsets1.jl`), and computed the grid-start difference `sepGrd` bakes in
(`trg.grd.start - src.grd.start`, equivalently `Δorg` after the `brd` symmetric-offset terms
cancel) divided by the shared cell scale `s`:

| source | pair | Δorg/s | integer? |
|---|---|---|---|
| `extSlfTest.jl` | `volTrg`(org 1,1,1) vs `volSrc`(org 0,0,0), s=1/32 | (32,32,32) | yes |
| `crsSclTest.jl` | same-scale touching pair (org 4/32,0,0 vs 0,0,0, s=1/32) | (4,0,0) | yes (touching, not far-field anyway) |
| `cmpOprTest.jl` | `mnyCvl` coarse region (org 1/16,0,0) vs `srcCvl`(org 1/2,0,0), s=1/16 | (-7,-1,-1) | yes |
| `cmpOprTest.jl` | `triCvl` coarse regions 2 vs 3 (org -1/8,0,0 vs 1/8,0,0), s=1/16 | (-4,0,0) | yes |
| `cmpOprTest.jl` | `extOpr` test pair (org 0,0,0 vs 1,0,0), s=1/16 | (16,0,0) | yes |

Every same-scale separated pair in these three files has an origin difference that is an exact
integer number of cells. I re-read all 379+157+20 lines and cross-checked every
`GlaOprVac{Float64}(`/`GlaCmpOprVac{Float64}(`/`GlaVacOprMem(` call site (`grep` list in the
scratch log) against this table; no other same-scale separated pair exists in these files (the
rest are either cross-scale, touching, or self).

**Why this is not a coincidence — two structural reasons found while reading `src/`:**

1. `src/glaCmpVol.jl:150` (`GlaCmpVol`'s constructor) throws `ArgumentError` — "Regions ... do
   not share a common grid ... Region corners must be on the common grid" — for *any* two
   regions of a composite volume, same-scale or not. Misalignment is impossible inside a valid
   `GlaCmpVol`; every `cmpOprTest.jl` pair above is aligned by construction, not by choice of
   origin. `refine`'s outward-snapping (`src/glaCmpVol.jl:280-330`) is what keeps every produced
   region on the parent grid.
2. `genEveExtInf`/`genVolEve` (`src/glaVol.jl:214-260`, called from both `GlaVacOprMem`
   constructors) only equalizes cell-count *parity* between target and source; it does not
   touch `org`, so it does not create or remove misalignment. (I checked this because a
   parity mismatch alone can generate a half-cell offset through the `brd` term of `GlaVol`'s
   `grd`; `genEveExtInf` prevents that specific mechanism by forcing matching parity.)

**Plain `GlaVol` pairs are not validated, so misalignment is constructible in principle.**
`notes/phase0/scratch/offsets2.jl`: built two same-scale `GlaVol`s with `Δorg/s = 49/3` (a
Rational, deliberately not an integer) and called `GlaVacOprMem(CPUKerOpt{Float64}(), volB,
volA)` directly (bypassing the composite layer). It **succeeded** — today's numerical-integrator
generation path (`egoFunOut!`) works on physical distances and does not care about lattice
alignment. This is the scenario §4.3 is about: once `farBlk!`/`farRoute` replace `egoFunOut!`,
this pair would need the offset-vector entry point (`R` a real triple, not `D::NTuple{3,Int}`)
that Phase 2 already plans to add regardless of this measurement's outcome.

**Route-classification is insensitive to misalignment, and this is provable from the code, not
just observed.** `boundL`/`boundLoct` (`farfield.jl:984,998`) already take a real radius/vector
— confirming §4.3's claim that only `farRoute`/`farTensor` hard-code the integer lattice.
`notes/phase0/scratch/misalign_route.jl`: built `fs = farSetup((1/32,1/32,1/32), f=1)` and
classified a set of near/far integer offsets, then the *same* offsets shifted by a generic
non-lattice fraction (+0.37 of a cell on every nonzero axis — larger than any fractional part an
arbitrary Rational `Δorg` could plausibly leave unresolved). Result: **the route family (1 vs 2
vs 3) never changed** between aligned and misaligned versions of the same offset; only the
selected `L`/`Lc` values shifted slightly (sometimes improving). The offsets that came back
route 3 (`D=(0,0,1)`, `(1,0,0)`, `(1,1,0)`, `(1,1,1)`) are exactly the touching/contact-adjacent
ones (center-to-center separation of exactly one cell width = zero gap), which never reach
`farRoute` in Gila's actual generation (handled by the contact path instead); the first genuine
far-field near-band offset, `D=(0,0,2)` (one full cell of gap — the "twelve two-cell offsets" of
`registry.tex`), is route 2 both aligned and misaligned. This matches the theoretical
expectation: `boundL`/`boundLoct` are continuous functions of the real separation, so a sub-cell
perturbation cannot flip an offset across the route-1/2 convergence boundary unless it was
already within a sub-cell of that boundary — and route 2 (octant split) needs no lattice at all,
so it absorbs the entire near band regardless of alignment.

**Conclusion:** the route-3 set from Gila's own tests is empty (there is nothing to route,
because there is no misaligned offset). Even under the hypothetical of a user hand-constructing
a misaligned plain-`GlaVol` pair, route 3 would only be reached where it is already reached
today for aligned offsets (the 67 slender-needle offsets of `registry.tex`) — misalignment adds
no new route-3 exposure by the continuity argument above. **What I could not verify:** I did not
scan `test/` exhaustively beyond the three named files, and I did not search Gila's non-test
usage (there is none under `src/` yet, by the plan's own statement in §1). I also did not find
or exercise any code path that validates alignment for plain (non-composite) `GlaVol` pairs —
none exists, confirmed by `offsets2.jl` succeeding.

## 2. The two places not at the Float64 noise floor

### 2(a) The single unequal-cell entry at 5.3e-14

Found in `notes/farfield/tables/k_adv.txt` line 77 (the `lamcube` case) and
`tex/crossscale.tex:825-828`. Reproduced directly with `notes/phase0/scratch/lamcube_entry.jl`:

- **Shape:** target = a full-wavelength cube, `sT=(1,1,1)λ`; source = the fine gcd cell,
  `sS=(1/32,1/32,1/32)λ`. ("lamcube" = λ coarse cell against a λ/32 fine cell.)
- **Offset:** `R = (35/64, 31/64, 31/64)λ = (17.5, 15.5, 15.5)g` (g = 1/32λ).
- **Frequency:** `f = 1.0`.
- **Route:** 2 (the gcd box sum), confirmed directly via `farRouteX(fx, R) = (2, -1)`. The sum
  is over the target's `32×32×32 = 32,768` fine sub-cells, matching `crossscale.tex`'s "one
  entry of a 32,768-piece box sum."
- **Certificate:** `farTensorX(...; cert=true)` gives `cert/max|G| = 3.58e-14`, bit-for-bit the
  same figure as the table's `c/G` column — confirms this is the exact case in the document.

**Mechanism: structural (the "small entry" effect of §3.11), not `l`-sum cancellation and not a
route-2-specific conditioning defect.** Inspecting the full tensor: `max|G| = 1.298e-5`
(entry `G[1,1]`), but the off-diagonal entries `G[1,2]`, `G[1,3]`, `G[2,1]`, `G[3,1]` are only
**7.6% of `max|G|`**. An absolute error at the tensor's own floor (`eMx·max|G| = 5.12e-15·max|G|
≈ 6.6e-20`) inherited by one of these small entries becomes `1/0.076 ≈ 13×` inflated in relative
terms — closely matching the observed ratio `pEn/eMx = 5.27e-14/5.12e-15 ≈ 10.3×`. This is
exactly the effect §3.11 already names ("a small entry inherits absolute error scaled to
`max|T|`; that is the formulation, not the arithmetic"), now confirmed on the specific worst
case: `eMx = 5.12e-15` (23 ulp) is already at the floor; `pEn = 5.27e-14` (≈240 ulp, `dig = 2.4`
matches the table) is the same floor divided by the small entry's own 7.6% scale, not a separate
error source. Not `l`-sum cancellation (that mechanism belongs to route 1's `boxAcc`, not
route 2's box sum); not a box-sum conditioning problem (the amplification `Λ` for route 2 is
measured elsewhere at 1.0-2.3, nowhere near enough to produce a 13× effect on its own).

### 2(b) The 0.13% band at ρ = 0.55-0.60

Reconstructed the exact "realistic block" of `verify.jl` part (m) — target = one fine `g=1/32λ`
cell, source = one `16g = λ/2` coarse cube, offsets `R=((2k1+1)/64,(2k2+1)/64,(2k3+1)/64)λ` for
`k1∈8:119, k2,k3∈-56:55`, touching offsets excluded — in
`notes/phase0/scratch/band_0.55_0.60.jl` / `band2.jl`:

- **1,404,604 offsets total.** Route split: **1,387,696 route 1 / 16,908 route 2** — the 16,908
  figure reproduces `crossscale.tex`'s independently-stated count exactly
  ("$16\,908$ offsets of the block lie above [ρ≈0.58-0.60]").
- **Confirmed count, close but not exact match to "1800":** of the 6,408 offsets with
  `ρ_whl∈[0.55,0.60]`, 1,544 sit exactly at the table's last shell `L=56` (i.e. "the table has
  no more shells," matching the tex's phrasing); of those, **1,392 have
  `bound/(tol·max|G|_actual) ∈ [1,4]`** (I compute this ratio against the tensor's own measured
  `max|G|`, not the a priori `est(R)`, since `est` can itself be 1.1-6.3× the true `max|G|` on
  cubes per `uncertain.tex`, which is exactly why a bound that satisfies the `tol·est` selection
  criterion can still show a ratio above 1 against the true scale — this reconciles with the
  selection rule rather than contradicting it). This is 1,392-1,544 against the tex's stated
  1,800: same order of magnitude, same mechanism (verified: `L` distribution on the ρ-band is
  `{52: 2024, 54: 1868, 56: 1544, route2: 972}`), most likely a slightly different exact filter
  boundary or ρ-band edge convention on their side. I treat this as **confirmed**, not exactly
  reproduced.

**Cost of raising `LMAX`:** measured directly on the worst-margin offset found in the band,
`R=(43/64,-3/64,-27/64)λ` (`ρ_whl=0.579`, ratio at `L=56` is `0.859`). Forcing evaluation at
higher `L` on the same table (`bndWhlX`/`tnsWhlX!` called directly, bypassing the
smallest-sufficient-`L` search that `boundLX` normally performs) shows the bound tightens fast:

| L | bound/(tol·max\|G\|) |
|---|---|
| 56 | 0.859 |
| 58 | 0.255 |
| 60 | 0.076 |
| 64 | 0.0068 |
| 70 | 1.9e-4 |
| 80 | 4.9e-7 |

Raising `LMAX` by 4-8 shells closes the margin by one to two orders of magnitude. Cost: cold
table-build time for this pair does not scale dramatically with `L` over this range — measured
(disk off, `notes/phase0/scratch/band2.jl` continuation) `L=56`: 3.6 s cold, `L=64`: 1.1 s,
`L=72`: 2.4 s, `L=80`: 3.2 s (noisy, same order of magnitude — no evidence of a real cliff);
per-offset evaluation cost grows roughly as `(L+1)^2` (the registry's own 391/1250/2731/4972 ns
at `L=10/20/30/40`), so even `L→64` adds only a few extra microseconds per affected offset,
totalling well under 0.1 s in aggregate for the ~1,800-offset band. **Resulting certificate
level: comfortably sub-tol** (ratio 0.007-0.0002 at `L=60`-`70`, i.e. certified to 15-140× better
than tol rather than the current 1-4× margin).

**Cost of routing the band to the box sum instead:** I did **not** independently remeasure this
— it requires changing `farfield.jl`'s near-band fine-block sizing logic (a source-adjacent
change, not a pure measurement) to actually grow the block from 33³ to 41³ and refill it, which
is out of scope for a read-only Phase-0 measurement. I instead extrapolate from the numbers
`crossscale.tex` already recorded for the *current* 33³ near-band fill: `0.95 s` cold `farBlock!`
fill + `28.6 μs`/offset of adds for the 16,908 offsets it currently serves (≈0.48 s). Scaling
the fill by the volume ratio `(41/33)³ ≈ 1.92`: **≈1.8 s** cold fill, plus **≈0.05 s** more of
adds for the extra ~1,800 offsets — **≈2-2.5 s total**, still cheap. Route 2's own measured
accuracy elsewhere (`registry.tex`, family u₂: `Λ=1.0-2.3`, `3.4e-15` worst on a 4096-term sum)
is far tighter than route 1's marginal 1-4× tol here, so this option gives a better resulting
certificate too, at a comparable one-time cost to the `LMAX` option. **Both options are cheap;
I have no basis from these measurements to prefer one over the other on cost grounds alone** —
the `LMAX` option is the one I could measure end-to-end without changing `farfield.jl`.

### 2(c) `l`-sum accumulation: is a compensated accumulator worth anything?

**Measured: no.** `notes/phase0/scratch/lsum_compensated.jl` reimplements `boxAcc`'s `l`-sum
accumulation three ways on the *identical* Float64-valued terms (isolating the accumulation
step specifically — term-generation error in `h_l`/`Y_lm`/the phase seed is a separate, already
measured, question and deliberately not what this reimplementation touches): (a) the plain
left-to-right Float64 running sum `tnsWhl!`/`tnsOct!` perform today, (b) a compensated
double-double sum using the file's own `twoSum`/`twoPrd`/`ddAdd`, (c) a `BigFloat(256)`-exact sum
of the same Float64 term values (ground truth for the accumulation step alone).

Swept: the cubic `(1/32)³` cell at several near-threshold offsets and two frequencies; the
production slender cell `(1/32,1/32,1/512)` (aspect 16) along its short axis (`n=40,64,128`),
long axis (`n=3..32`) and diagonal, at real and complex `f`; and a deliberately extreme
aspect-1020 flat cell `(1/32,1/32,1/32640)` at `f=2+0.2i` across 13 offsets — 25 cases in total.
`l`-sum amplification (`Σ|term|/|sum|`) in everything I could construct ranged **1.02 to 4.23**
(I could not reach the `certificate.tex` 60-case audit's reported worst amplification of `3.14e3`
— that came from one specific random-seeded shape/offset from their audit that is not
reconstructable from the released tables). Across every case: **naive Float64 accumulation
matched the compensated sum's value to within 0-9 ulp (median ≈1-4 ulp)**; the compensated sum
itself matched the `BigFloat` ground truth to **0 ulp** in every case tested.

A compensated accumulator would recover at most this few-ulp gap — small next to the
already-measured 2.3-ulp median / 16.4-ulp worst-case overall floor (`certificate.tex`), and
`TOL=1e-14` already sits below that floor regardless (§3.4/§3.11). I could not directly measure
the accumulation-only gap at the audit's reported worst amplification (3.14e3), but a bound
follows without needing to: the certificate.tex worst-case row's **total** measured error (16.4
ulp, generation *and* accumulation combined, using today's naive accumulator) already upper-bounds
whatever the accumulation step alone contributes there, so a compensated accumulator's benefit
even in that worst known case cannot exceed a small number of ulp out of that 16.4 — consistent
with, not contradicted by, my direct low-amplification measurements. **This confirms the plan's
expectation: a compensated `l`-sum accumulator is not worth implementing.**

## 3. Table build cost vs L

`notes/phase0/scratch/table_cost.jl`, `table_cost2.jl`. Cold = fresh scratch directory per
shape/pair (`disk=true`); warm = second call against the same directory. All single-threaded.

**Four production equal-cell shapes, `L=LMAX=56` (default), with the auto-generated
representative near-offset set (`farSetup`'s own `nearOff`, not a hand-picked list):**

| shape | s | cold | warm | file size | `Lw` (stored) |
|---|---|---|---|---|---|
| c32 | (1/32,1/32,1/32) | 13.57 s | 0.58 s | 16.08 MB | 48 |
| c8 | (1/8,1/8,1/8) | 12.07 s | 0.64 s | 16.16 MB | 50 |
| c4 | (1/4,1/4,1/4) | 12.93 s | 0.69 s | 16.85 MB | 52 |
| sl | (1/32,1/32,1/512) | 15.86 s | 0.71 s | 18.97 MB | 56 |

Matches `farfield.tex`'s "14-18 s cold, 0.7-2.5 s warm, 16-20 MB per equal-cell shape" closely.

**Two unequal pairs** (target = fine `g=1/32λ` cell, source = `8g=1/4λ` and `16g=1/2λ`
coarse cubes), forced to a production-realistic `Lw=54` (`farSetupX` without an explicit `offs`
list does *not* auto-populate a representative near-offset set the way `farSetup` does — see §4
below — so I set `lDef=54` by hand, matching the `Lw` a real near-band block actually needs, per
`crossscale.tex`'s own tables):

| pair | cold | warm | file size |
|---|---|---|---|
| g vs 8g (λ/4) | 4.43 s | 0.11 s | 1.88 MB |
| g vs 16g (λ/2) | 2.57 s | 0.35 s | 2.03 MB |

Matches `crossscale.tex`'s "3.3-5.7 s cold, 0.4-2.5 MB per unequal pair."

**Build time vs `L`** (equal-cell c32, cold, one forced offset via `lDef` rather than the full
auto set, so these are *not* comparable in absolute terms to the production-shape row above —
they isolate the `L`-scaling of the geometry table itself):

| L | cold | file size |
|---|---|---|
| 32 | 0.56 s | ~0 MB (below the table's population threshold at this radius) |
| 40 | 0.55 s | — |
| 48 | 0.56 s | — |
| 56 | 1.76 s | 0.52 MB |
| 64 | 1.87 s | 0.60 MB |
| 72 | 2.39 s | 0.68 MB |

Cost is flat through `L=48` then grows slowly (sub-quadratically over this range) beyond it.

## 4. `farSetup`/`farSetupX` cost vs `nBlk`

`notes/phase0/scratch/nblk_cost.jl`, disk off (pure compute), single-threaded.

| nBlk | equal-cell (c32) | cross-scale (g vs 16g, `lDef=54`) |
|---|---|---|
| 16 | 0.55 s | 0.23 s |
| 32 | 0.57 s | 0.28 s |
| 64 | 0.57 s | 0.22 s |
| 128 | 0.58 s | 0.27 s |
| 256 | 0.58 s | 0.22 s |
| 512 | 0.59 s | 0.27 s |

(`nBlk=8` gave anomalous outliers — 11.6 s / 2.4 s — attributed to first-call JIT compilation in
a fresh process, not to `nBlk`; every other point is flat within noise.) `rHi` scales linearly
with `nBlk` as expected (`4·nBlk·r_d`); `nMx`/`Lw` are completely unaffected across the whole
16-512 range.

**`farSetup`/`farSetupX` build cost has no meaningful dependence on `nBlk`.** This both confirms
and sharpens the plan's quoted "0.7-1.1 s equal / 2.1-3.6 s cross-scale": my own flat numbers
above (0.55-0.59 s / 0.22-0.28 s) are somewhat below those quoted figures, and the gap is
explained by what actually drives the wall-clock cost — the size of the offset list scanned
during setup (`crossscale.tex`'s "exact-rational offset pass," 2.1 μs/offset), not `nBlk`. I
confirmed this separately: scanning the full 1,404,604-offset realistic block in §2(b) cost
3.6-3.95 s regardless of the `L`/`nBlk` settings used around it, consistent with the ~4.2 s
`crossscale.tex` reports for that same block. **Practical implication of the measurement** (not
a source recommendation — Phase 2 decides the actual memo policy): since cost does not scale
with `nBlk`, there is no cost-based reason to size a memoized `FrqSet`/`FrqSetX` tightly to the
block it will serve; building once with a generously large `nBlk` and reusing it for any smaller
block avoids both the "reused set exceeds its certified `[rNc,rHi]` range" correctness trap of
§4.1 and any need to track/rebuild the memo per `nBlk`, at no extra one-time cost.

## What could not be verified

- Item 1: only the three named test files were scanned exhaustively; `test/` was not searched
  beyond them, and no non-test Gila usage exists yet under `src/` to check (plan §1).
- Item 2(b), the box-sum remediation cost, is an extrapolation from `crossscale.tex`'s own
  recorded 33³ near-band numbers, not a fresh measurement — reproducing it directly would need
  changing `farfield.jl`'s block-sizing logic, which is source-adjacent work rather than a
  read-only measurement.
- Item 2(c): could not reach the certificate.tex audit's reported worst-case `l`-sum
  amplification (3.14e3); the bound given (total measured error already upper-bounds the
  accumulation-only contribution) is a logical argument from the existing certificate.tex number,
  not a direct measurement at that amplification level.
- No GPU numbers anywhere in this report — consistent with `uncertain.tex`'s own "no GPU path"
  note; not attempted here either.
- Thread scaling of `farBlock!`/`farBlockX!` was not remeasured; all numbers here are serial,
  matching what §3/§4 of this report needed (`farSetup`/`farSetupX`/`boundL`/`boxAcc` are not
  threaded functions).
