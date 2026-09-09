# Plan: wiring the analytic integrals into Gila

Branch `paul-farfield`, repo `/Users/pvirally/.julia/dev/GilaElectromagnetics`.
Status 2026-09-08: **ready to execute, awaiting Paul's approval.** Nothing under `src/` touched yet.

Read first: `notes/integrals.pdf` (where the integrals come from), `notes/moments/moments.pdf` (near
field, 15 touching sub-geometries, all moments), `notes/farfield/farfield.pdf` (far field, equal and
unequal cells, 83 pp).

### What is in `notes/`, and why each piece is kept

| path | why it is here |
|---|---|
| `PLAN_wiring.md` | this plan |
| `PROMPT_{integrals,farfield,crossscale}.md` | the task statements the three documents answer and cite |
| `integrals.{tex,pdf}` | where Gila's integrals come from; the elementarity theorem |
| `gen/` | the figures and generated tables the documents `\input`, and `gen/ref/mom.jl`, the 220-bit reference evaluator that `farfield/ref.jl` and `moments/verify.jl` load |
| `moments/moments.jl` | the near-field library to port (Phase 1) |
| `moments/box3tay.jl` | the 3D centred Taylor series; the drop-in named in §4.4, not currently used |
| `moments/{verify.jl,gilacmp.jl,refcache.txt,tables/}` | its verification, its Gila comparison, and their measured output |
| `moments/{moments.tex,moments.pdf,figs/}` | the near-field document |
| `farfield/farfield.jl` | the far-field library to port (Phases 2–3) |
| `farfield/{verify.jl,bench.jl,ref.jl,refquad.jl,crosscheck/famG.jl}` | its verification, its build benchmark, two independent 220-bit references and the independent Cartesian cross-check |
| `farfield/refcache/reftensors{,_x}.txt` | the 220-bit reference values; `test/ref/` is curated from these |
| `farfield/ksrcache/` | cached route-3 values, ~1.5 h of BigFloat work, append-only text |
| `farfield/tables/` | every table the far-field document quotes, as `verify.jl` prints it |
| `farfield/{farfield.tex,farfield.pdf,tex/}` | the far-field document and its section files |
| `farfield/gcd.md` | the three-hunk rational-gcd extension for non-integer cell ratios |
| `farfield/scratch/{REGISTRY.md,reports/}` | the registry of approach families with what was blocked and why, and the 31 agent reports: the audit trail behind every number quoted here |

Not kept: the agent working trees `farfield/scratch/{work,xwork}` (61 MB of scripts, logs and
intermediate caches) and `farfield/mkrefcache.jl` (a one-time migration from those caches). The
far-field document footnotes individual files inside them, so **those footnotes no longer resolve**;
the numbers they support are in the document's own tables, in `farfield/tables/` and in
`scratch/reports/`. `farfield/shapetab/` (110 MB of exact geometry tables) stays on disk and
gitignored: it is regenerable at 14–18 s per shape and is worth keeping for Phase 0.

## 1. What exists

Two standalone libraries, generic in the number type, no package dependencies, verified against
220/320-bit references. Neither has ever been called from `src/`.

**`notes/moments/moments.jl`** (1720 lines). Closed-form moments `I_m = ∫_F∫_F' |x−y|^m dS' dS` for
any axis-aligned rectangle pair, `m = −1, 0, 1, …`. Entry points `pairMoments(A, B, mMax)`,
`faceMoments(D, F, Fp, s, mMax)`, `momentSeries(A, B, f, mMax)`. The face-pair integral of
`g = e^{ikr}/(4πf²r)` is the entire series `(1/4πf²) Σ_n (2πif)^n I_{n−1}/n!`; the moments do not
depend on `f`. Worst error vs the 320-bit reference `1e-44` (its own floor); Float64 loses ≤ 0.91
digits for `m ≤ 12`, ≈ 1.3 at `m = 30`; 10–13 series terms at λ/32, 20–29 at λ/4. Cost 0.1–1 ms per
face pair for 14 moments. Reproduces Gila's `wekS/E/V` to their own error (6e-11 … 1.8e-9) with
Gila's order-48 and order-64 values converging toward the series.

**`notes/farfield/farfield.jl`** (2009 lines). The 36 face pairs of a separated pair are one integral
of `(∂_a∂_b + δ_ab k²) g` over the difference box against a non-negative product weight; the
spherical addition theorem splits it into an exact-rational, frequency-independent geometry table per
cell shape and a per-offset sum of elementary functions. Proven truncation bounds (Theorems A/A′/B)
choose the order per offset; every call can return an a posteriori certificate.

| | equal cells `s` | unequal cells `sT ≠ sS` |
|---|---|---|
| weight | triangle, `D = Π[−s_i, s_i]` | trapezoid, `a = |sT−sS|/2`, `b = (sT+sS)/2`, `D = Π[−b_i, b_i]` |
| setup | `farSetup(s, f; tol, nBlk, offs)` | `farSetupX(sT, sS, f; …)` |
| route 1 | whole box, even `l` only | whole trapezoid box (Theorem A′) |
| route 2 | 8 octants, affine weight, all `l` | gcd average of equal-cell tensors of `g = min(sT,sS)`, as a box sum over one fine `egoToe` |
| route 3 | BigFloat k-series on the 36 panels | same, on the 36 unequal panels |
| one offset | `farTensor(D::NTuple{3,Int}, s, f)` | `farTensorX(R::rational, sT, sS, f; cert)` |
| a block | `farBlock!(egoToe, s, f)` (Toeplitz layout) | `farBlockX!(G, Rs, sT, sS, f; tms)` (3×3×n) |

Measured: equal cells 8.6e-15 max-norm over 427 references, ≤ 1.59 digits lost, 128³ far fill at
λ/32 in 0.200 s on twelve threads against 200.8 s; unequal cells 3.42e-15 over 439 + 50 references,
0 certificate violations in 978 certified evaluations, routing exact on 125 256 enumerated offsets,
`farBlockX!` bitwise deterministic at 1/4/12 threads, 1.4e6-offset realistic block in 6.6 s serial
and 2.4 s on twelve threads with the set held. The equal-cell paths are bit-identical before and
after the cross-scale integration (120/120 tensors, two 12³ blocks, fresh tables byte-identical).

## 2. What Gila does today, and what it costs

`wekTrp` → DIRECTFN order 48 (`wekS/E/V`) plus the closed-form `rSrf*`; `egoFunSng!` masks with the
order-9 `cntOrd` rule on the touching shell; `egoFunInn!`/`egoFunOut!` → `egoSrfFxd!` with `quadOrd`;
`egoSrfAdp!` (hcubature, `cubRelTol = 1e-6`) for every pair at `sep ≤ 1`, which on a cross-scale block
is the whole near band; GPU batched `srfFxdKer!`.

Measured error against the 220-bit references:

| path | error today | library |
|---|---|---|
| separated equal cells, λ/32–λ/4 | 5e-14 … 9e-13 | 4e-15 … 9e-15 |
| separated slender cell, short axis | **O(1)** (0.79 at `(0,0,2)`) | 8.7e-15 |
| touching shell, slender cell | **117 %** | exact series |
| contact cell, slender | 12.7 % | exact series |
| cross-scale near band (`egoSrfAdp!`) | **1e-7 … 5e-4** | 3.4e-15 |
| cross-scale fixed rule | 2e-14 … 1.8e-12 | 3.4e-15 |

Two of these are below Paul's 1e-8 publishability bar. The order-9 shell rule alone makes the slender
operator's anti-Hermitian part indefinite at the percent level; with exact contact integrals that
operator is positive definite at `f = 1+0.1i` and PSD to the Float64 floor at `f = 1`. Cost: at 128³,
λ/32, twelve threads, 5.14 s of the 6.73 s build is `wekTrp`; per offset the library is 1000–2000×
cheaper than the fixed rule and 5–400× cheaper than the adaptive one.

## 3. Decisions (settled)

1. **Delete every numerical integrator**: DIRECTFN (`glaVacOprMemIntSup.jl`), `wekS/E/V`, `wekSDir`,
   `wekEDir`, `wekVDir`, `wekGrdPts!`, `rSrfSlf`, `rSrfEdgFlt`, `rSrfEdgCrn`, `egoFunSng!`'s masks,
   `cntOrd`, `quadOrd`, `egoSrfFxd!`, `egoSrfAdp!`, `srfKer`, `cubVecAltAdp`, `cubFac`, the GPU batch
   (`srfFxdKer!`, `egoSrfFxdBat!`, `egoSlfSpl`, `egoExtSpl`, `egoSrfFxdAsm!`), `gauQud`, and
   `intConTest.jl`'s order-convergence test. No quadrature fallback: route 3 (the k-series) is the
   universal fallback and needs no lattice, so nothing can fail to be computed. Drop `HCubature` and
   `FastGaussQuadrature` from `Project.toml`.
2. **Cross-scale is in scope now** (the trapezoid whole box, the gcd average, and the unequal-panel
   k-series all exist and are verified).
3. **Precision is a type.** Replace the now-meaningless `intOrd::Integer` field of
   `CPUKerOpt`/`GPUKerOpt` by `genPrc::Type{<:AbstractFloat}`, default `Float64`; serializers write
   it. Generation runs in `Complex{genPrc}` and narrows once into `Complex{T}` in `genEgoFur`, as
   today. `Double64` (DoubleFloats.jl, the user's own dependency; a test extra here) works because
   every new file is generic and every constant typed. **`genPrc = Float32` is refused** with a clear
   error: D14 (§4.2).
4. **Tolerance as tight as the type allows**: `tol = c · eps(genPrc)` with `c` fixed by the Phase-0
   measurement of the Float64 evaluation floor (the whole-box sum stops improving near 6.6e-14 at
   `L = 94` at the worst offset, so `c` cannot be 1; today's shipped `TOL = 1e-14` delivers 9.3e-15
   per entry for +0.7 % in terms over `1e-13`). One knob, one measurement, no per-shape tuning.
5. **Shape-table disk cache is a keyword on generation**, default off. On, it uses a Scratch.jl space
   (new dependency, already a test extra). Tables are 16–20 MB per equal-cell shape and 0.4–2.5 MB
   per unequal pair, 14–18 s to build cold, 0.7–2.5 s to read.
6. **Reference values live in `test/ref/`** and are read only by tests, never by a build.
7. **Generation moves to the CPU**, threaded. The FFT, the circulant embedding and every matvec stay
   on the device exactly as today. Rationale: the far fill is 0.200 s at 128³ and the near field is
   milliseconds, so there is nothing left worth batching, and the GPU rule is one of the numerical
   integrators being deleted.
8. **The near field never needs unequal cells.** Gila remeshes touching composite region pairs to the
   fine scale (`_sndBlk`, confirmed by `cmpOprTest.jl:100`), and a direct unequal touching pair goes
   through `genCntVol`/`egoCntOut!`, which averages self tensors of the gcd cell. Every contact
   evaluation is therefore equal-cell. `moments.jl` is wired for equal cells only.

### 3.9 What fp64 and fp32 actually get

`genPrc = Float64` is the configuration every number in §1 was measured in, so it is verified, not
projected. What it delivers per tensor entry, against the 220/320-bit references:

| piece | measured in Float64 | in units of eps |
|---|---|---|
| far field, equal cells, 427 refs | 8.6e-15 max-norm, 9.3e-15 per entry | ≈ 40 |
| far field, unequal cells, 489 refs | 3.4e-15 max-norm, 5.3e-14 worst entry | ≈ 15, one entry at 240 |
| far-field rounding, 60 random cases | ≤ 16.4 eps · max|T| | 16.4 |
| near-field series vs a 256-bit sum | 9.1e-16 | ≈ 4 |
| near-field moments, `m ≤ 12` | ≤ 0.91 digits lost | ≈ 8 |
| near-field moments, `m ≈ 25–30` (λ/4 cells) | 1.15–1.28 digits lost | ≈ 19 |
| a posteriori truncation certificate | ≤ 4.5e-14 | 200 |

So:

- **fp32 storage is completely covered.** Float64 generation leaves at least eight spare decimal
  digits everywhere, including the coarse and slender cases, and the single narrowing in `genEgoFur`
  is then the best fp32 representation obtainable. Nothing is open. This is strictly better than
  generating in fp32, which is refused anyway (§4.2).
- **fp64 storage is covered to about 1e-14 relative, i.e. tens of ulp, not the last bit.** That is
  five to six orders better than Gila today and six orders past the 1e-8 publishability bar, and it
  needs nothing we do not already have. The residual is genuine Float64 rounding in the sums, not
  truncation: the far-field bounds certify the truncation separately, and at the hardest offsets the
  whole-box sum was measured to stop improving near 6.6e-14 by `L = 94`, which is why the octant
  route exists for those twelve offsets and reaches 4.4e-15 there.
- **Last-bit fp64 needs `genPrc = Double64`**, and that is a measurement rather than new
  mathematics. Both libraries are generic in the number type with every constant typed, the shape
  tables are stored at 192 bits (Double64 carries 106), and the selection arithmetic is already
  forced to Float64 for reasons that do not change. What is unverified is whether every scalar
  function the two files call exists and is accurate for `Double64`: `cispi`, `sinc`, `asinh`,
  `atan`, `hypot`, `ldexp`, `exponent`, and the `BigFloat` conversions in the table reader. Phase 0-B
  answers exactly that. Expected outcome is that it works with at most a handful of small fixes.

The two things that would *not* be fixed by more generation precision, and are therefore worth
stating once: the assembly amplification is gone by construction (the volume form replaces the
signed sum of 36 face pairs, so the 10–1000× amplification of `srfSum!` no longer applies), and for
complex `f` a part of an entry carrying less than about 1e-3 of that entry's own modulus is accurate
only in absolute terms. The latter is a property of the formulation, not of the expansion; at real
`f` the real and imaginary parts are independent real sums and each keeps its own relative accuracy.

## 4. Hazards that will bite if ignored

### 4.1 The frequency set's certified radius interval (D13, D15)

`farSetup`/`farSetupX` certify the table's k-series cut on `[rNc, rHi]`, an interval fixed by the
offsets the set was **built** for (`rHi = 4·nBlk·r_d`). Reusing a set outside it returned a silently
wrong tensor under a valid-looking certificate: 3.2e-8 under a 6.5e-15 certificate on the slender
cell, a violation by 5e6. Both are now **fixed** — the interval is stored and `farRoute`/`farRouteX`
raise outside it — and the equal-cell paths are re-proved bit-identical. The default
`offs = nothing` path is safe, measured over 16 785 152 offsets with zero raises, and
`farTensor`/`farBlock!` called without a set build their own and cannot fire it.

**What this means for the wiring.** The natural optimization — memoize one set per (shape, frequency)
like `wekMem` does — is exactly the failure mode: a set built for a 32³ block and reused for a 128³
one exceeds `rHi`. The rule is: **key the memo on `(shape, frequency, tol, nBlk)`**, or store `nBlk`
in the entry and rebuild when a larger block arrives. `farSetup` costs 0.7–1.1 s (equal) and
2.1–3.6 s (cross-scale, dominated by the exact-rational offset pass), so this is worth memoizing and
worth getting right. The verification suite's own part (b) committed this bug and lost a factor 1.2
of accuracy for it; assume we will too unless the key is explicit.

### 4.2 Float32 returns NaN in the near band (D14, not fixed)

The upward `h_l(k|R|)` recursion overflows Float32 at `l ≈ 27`, so any offset whose selected `L`
reaches 28 returns NaN in all nine entries — that is every `|D| ≤ 4` at λ/32, and the whole
cross-scale near band through route 2. Pre-existing, equal-cell, left alone under the bit-identity
mandate. Consequences: generation in Float32 is refused (§3.3); if we ever want it, the fix is to run
`hFill!`/`shrm!` in Float64 and round, or to scale the recursion by `(2l−1)!!`, plus an `isfinite`
guard on the workspace. Storage in Float32 is unaffected — generation is Float64 and narrows at the
end, which is strictly better than generating in Float32.

### 4.3 Two interfaces have to change, and they are the cost, not the mathematics

- **Equal-cell external pairs need an offset-vector entry point.** `sepGrd` produces offsets
  `Δorg + n·s` with `Δorg` an arbitrary Rational, so a misaligned external pair is not on the integer
  lattice `farRoute(fs, D::NTuple{3,Int})` assumes. `boundL`, `boundLoct`, `tnsWhl!` and `tnsOct!`
  already take a real `R`; only `farRoute`/`farTensor` hard-code `R = D .* s`. This is ~10 lines of
  plumbing. The octant split needs no lattice, so the near band stays cheap; only route 3 would be
  reached, and only where neither expansion converges.
- **Cross-scale needs an integer-lattice entry point.** On the 1.4e6-offset realistic block the
  exact-rational offset vector is ~1 GB of the 2.95 GB peak, and the selection plus routing passes
  cost 4.2 s single-threaded against 4.4 s of actual expansion work. The document names the fix:
  pass `m` and the gcd cell instead of `R`. Gila's circulant index already *is* that lattice, so this
  is a strict simplification of the call site, and it removes both passes.

### 4.4 Honest limits

- **The thin-indicator branch in `moments.jl` is a quadrature, it stays, and here is why.** A box
  whose position along an axis exceeds half its width is integrated by a Gauss-Legendre rule along
  that axis against the 2D closed forms at the node heights, with the order taken from a proven
  analyticity radius (≥ 3+√8 by the gate itself, giving order 11–14 in Float64 and 28 at 128 bits).
  It never fires for the fifteen touching equal-cell sub-geometries, so the contact path is
  closed-form all the way down. It fires on 96–99 % of the perpendicular boxes of *separated* pairs,
  equal cells included, so far-field route 3 runs through it, at 128 bits, where it loses at most
  0.99 of 38.2 digits: 22 orders below that route's target and therefore invisible.

  The closed forms do exist there and are bypassed for conditioning, not existence: the divergence
  identities lose 3.96 digits with one narrow axis just above the gate and 6.90 with two, and in 2D
  this was *proven* intrinsic (the four moments are functionally independent, only one carries the
  arctangent, amplification exactly 16). The analytic cure is the 3D centred Taylor series
  `notes/moments/box3tay.jl` (90 lines, generic in T, exact and finitely truncated, elementary
  coefficients by a three-term recurrence, exact rational weight moments — not a quadrature). It was
  written, verified per box against a 320-bit reference, and deliberately **not** shipped:

  | policy | worst digits lost per face pair | cost vs the rule, median / p90 |
  |---|---|---|
  | rule everywhere (shipped) | 0.93 | 1.0 |
  | series on thin boxes, rule otherwise | 0.90 | 1.5 / 6.5 |
  | series gated at absolute convergence | 0.90 | 1.23 / 3.5 |
  | series on every box | 0.71 | 2.4 / 33 |

  Per box the series is far better (0.00 median, 0.35 worst, against 0.51–0.71 median and 0.96
  worst) but the face-pair worst case is pinned by corner boxes no series reaches, so the global gain
  is under one binary digit for up to 6.3× the aggregate cost and 2126× on one box. And the rule
  cannot be deleted in any case: 1.0 % of thin boxes fail the polydisc gate at every `m` and a
  further 2.3 % at high `m` (a thin axis beside a long corner or coarse axis), and there the closed
  form cancels catastrophically *and* the series does not converge, while the rule's accuracy is flat
  across every regime. **Conclusion: no change to `box3`'s dispatch.** If a Float64 near-field fill
  on separated pairs is ever wired and its budget is counted in eps, the drop-in is `box3tay.jl`
  gated on thin-axis *and* absolute convergence with the rule kept beneath it, and it must carry the
  three-level stopping rule: the two-level rule inherited from 2D loses 11–13 digits on `1/r` at
  diagonal box centres, because harmonicity makes the second level vanish identically.
- **Route 3 costs 6–21 s per offset** (BigFloat, `N = 13–44`). It is reached by 67 offsets of the
  slender needle and by nothing at all on a cross-scale block (0 of 1.4e6). It caches to disk. If a
  build ever hits many route-3 offsets it will be slow, not wrong.
- **0.13 % of the realistic cross-scale block is certified only to 1–4 × tol** (1800 offsets at
  ρ = 0.55–0.60 that exhaust `L = 56`); where references exist they are correct to 3e-15. Raising
  `LMAX` or routing that band to the box sum would tighten it.
- **The table writer is unlocked** across processes: two processes extending one file each append a
  duplicate segment. Harmless on read since D12 (`mrgGeo` drops columns already present), but it
  wastes disk. If the disk cache is enabled by default anywhere, add a lock.
- **`est` is a scale, not a bound.** It over-states `max|T|` by 1.1–6.3× on cubes (safe direction) and
  under-states by 2–4× on a ratio-16 rod. The a posteriori certificate discharges the hypothesis at
  no cost and evaluates to ≤ 4.5e-14 over the reference set; use it in the tests.
- **Non-integer cell ratios are refused**, matching `GlaExtInf`'s own rule. A three-line rational-gcd
  change makes them work (`notes/farfield/gcd.md`, verified) if Gila ever allows them.
- `setprecision` is process-global inside route 3; `farBlock!` sets it once around the threaded
  k-series pass. Anything else in the process that cares about BigFloat precision must not run
  concurrently.

## 5. Architecture

New files under `src/vacuum/`, both ports with the numerics carried over verbatim and proved
bit-identical to the notes implementation before any restyling:

- **`glaVacOprMemMom.jl`** (~1100 lines, from `moments.jl` minus the explicit cross-check forms and
  the reference loader): `pairBxs` with exact Rational breakpoints, `box2`/`box3`, `pairMom`, the
  panel bookkeeping, `momSer` with its `|last term|/|sum|` indicator, and `nearBlk!(egoToe, vol,
  cmpInf)` filling the eight contact offsets plus the touching shell. Moments memoized per
  `(scl, mMax, genPrc)` — the frequency-free successor of `wekMem`. Series order from the a priori
  bound `(kD)^{N+1}/(N+1)!·e^{kD}`; when `eps(genPrc)·e^{|k|D}·Λ` exceeds the target the offset is
  summed at higher precision instead of being truncated (cells above ≈ λ/2 for fp32 storage, ≈ λ/4
  for fp64).
- **`glaVacOprMemFar.jl`** (~1000 lines, from `farfield.jl` minus the verification hooks and the
  BigFloat-only caches): the exact table build (in-place GMP), the shape cache (memory always, disk
  behind the keyword), `farSet`/`farSetX` with the Theorem A/A′/B selector and the `[rNc, rHi]`
  guard, `tnsWhl!`/`tnsOct!`/`tnsGcd!`, route 3 delegating to `glaVacOprMemMom.jl`, and the two block
  fills. Both interface changes of §4.3 land here.
- `glaVacOprMemInt.jl` keeps only `sclEgo`/`sclEgoN` (still used by tests and the composite path) and
  `facPar`/`srfScl`/`srfSum!` if anything still needs them; `glaVacOprMemIntSup.jl` is deleted.
- `glaVacOprMemGen.jl`: `genEgoCrcSlf!` becomes `nearBlk!` + `farBlk!` + identity term + embedding;
  `genEgoCrcExt!` the same on the offset vector, keeping `egoCntOut!`/`genCntVol` for the contact
  class unchanged; `srfSum!`, `egoToeCrc!`, `sepGrd` glue unchanged.

Style for every agent: Paul's — abbreviated camelCase, one-line docstrings, terse comments, no banner
comments, no hollow helpers, delete knobs that do nothing. Commit nothing; leave the tree dirty.
Suite: `JULIA_NUM_THREADS=auto julia --project -e 'using Pkg; Pkg.test()'`. Reports carry files,
numbers and what could not be verified, never status. The Manifest was stale and has been resolved;
`using GilaElectromagnetics` works.

## 6. Phases

**Phase 0 — measurements (two sonnet agents in parallel, read-only on `src/`).**
A (far field): the Float64 evaluation floor as a function of `tol` and `L`, giving the constant `c` of
§3.4; table build time, peak RSS and file size vs `L` for the production shapes and two unequal
pairs; the offsets Gila's *external* equal-cell pairs actually produce (enumerate `Δorg/s` over
`extSlfTest.jl`, `crsSclTest.jl`, `cmpOprTest.jl`) and how many land in the route-2/route-3 band;
`farSetup` cost vs `nBlk` for the memo key of §4.1.
B (near field): `moments.jl` under `Double64` (what fails, accuracy vs 320 bits, cost); 288 face
pairs per shape at `mMax` 12 and 30, serial and threaded; series digits lost vs 256 bits for
`f ∈ {1, 0.37, 3, 1+0.1i, 1+i, 0.5+2i, 2+2i}` at λ/32, λ/8, λ/4, λ/2, λ and the slender cell, giving
the precision-escalation constants; `srfSum!` amplification `Λ` on the eight near offsets per shape;
the baseline full suite (pass count, time).

**Phase 1 — near field (opus).** Port, wire `nearBlk!`, reproduce `wekTrp`/`egoFunSng!` against
DIRECTFN at intOrd 48 and 64 (bracketing) *before* deleting them, then delete. Tests: even moments vs
exact rationals, homogeneity in `f`, Ec `b↔c` and Vp `a↔c` symmetry, `test/ref/` moments at 1e-13,
the slender shell offsets (the 117 % case), `posDefTest` tightened to the Float64 floor at `f = 1` and
positive definite at `f = 1+0.1i` including a 6×6×12 slender block, fp32 build bitwise equal to the
narrowed fp64 build.

**Phase 2 — far field, equal cells (opus).** Port, shape cache, `tol` from `genPrc`, the
offset-vector entry point of §4.3, the `(shape, f, tol, nBlk)` memo key of §4.1, route 3 in `genPrc`;
wire `farBlk!` into the self and external equal-cell builds; delete the fixed rule and the GPU batch.
Tests: `test/ref/` tensors across shapes, directions and the seven frequencies at 1e-13; the near/far
overlap at two cells (k-series vs expansion — a free end-to-end check of bookkeeping and
normalization); `M_ab(R) = M_ba(−R)`; thread-count determinism; the a posteriori certificate never
violated.

**Phase 3 — far field, cross-scale (opus).** Port the X path, the integer-lattice entry point of
§4.3, the fine equal-cell set and the box sum, wire into `genEgoCrcExt!` replacing the `egoFunOut!`
loop with `egoCntOut!` left in place. Tests: the cross-scale reference tensors at ratios 2, 4, 8, 16
on one, two and three axes at three frequencies; `crsSclTest.jl` retargeted from 1e-6 to 1e-12; the
swap identity `G(R; sT, sS) = (V_s/V_t) G(−R; sS, sT)`; the reflection identity; a composite operator
build end to end.

**Phase 4 — precision plumbing and cleanup (sonnet).** `genPrc` field, constructors, serializers,
the Float32 refusal, `egoToe::Complex{genPrc}` with the single conversion at embedding, the
scratch-space keyword; sweep `test/` for `intOrd`; a guarded `Double64` smoke test; delete every dead
function and constant; `Project.toml`.

**Phase 5 — adversarial verification (opus, independent of Phases 1–3).** New 220-bit references at
shapes nobody has used (aspect 1e-3…1e3, λ/2 and λ cells), the seven frequencies, fp32/fp64/Double64
builds, PSD of `asym(G₀)` on 6³ and 6×6×12 blocks, `egoOpr!` against a dense reference, before/after
build times at 32³/64³/128³ (λ/32) and 32³ (λ/8, λ/4) at one and twelve threads, the full suite. Every
number reported.

**Phase 6 — docs and final run (sonnet).** `docs/src/concepts.md` (where the integrals come from, the
two expansions, the routes and their gates, `genPrc`), `docs/src/usage.md` (precision choice, first-
build table cost, the disk-cache keyword), the important integral results moved from `notes/` into
`docs/`, full suite.

## 7. Acceptance

- Suite green; no quadrature reachable from any generation path except the thin-axis rule inside
  route 3, which is kept deliberately (§4.4) and runs at 128 bits.
- **`genPrc = Float64`, the default and the verified configuration:** every tensor entry within
  1e-13 of the 220-bit reference on the test matrix (equal and unequal cells, cubic, anisotropic,
  slender, coarse, fine; real and complex `f`). This is the fp64 target and it is met by what
  already exists; 1e-14 is the measured level and 1e-13 is the acceptance bar with margin.
- **`T = Float32` storage:** the build is bitwise equal to the narrowed `genPrc = Float64` build,
  and `genPrc = Float32` raises.
- **`genPrc = Double64`:** runs at all, and within 1e-20, on the shapes Phase 0-B clears. If it
  needs more than a handful of small fixes it is deferred to its own phase rather than blocking
  fp64, and the plan says so here so nobody treats it as a prerequisite.
- `asym(G₀)` PSD to the Float64 floor at real `f` and positive definite at `f = 1+0.1i` on the
  slender block.
- 128³ self build at λ/32: far fill ≤ 0.3 s, near field ≤ 0.5 s, total FFT-dominated.
- Cross-scale: the realistic 1.4e6-offset block in seconds, and `crsSclTest` at 1e-12 where it is at
  1e-6 today.

## 8. Open question for Paul

Nothing blocks fp64: §3.9 records that the default `genPrc = Float64` configuration is the one every
measurement in §1 was taken in, and it meets the acceptance bar with margin. Two items are flagged
rather than open, both measurements in Phase 0 and neither on the critical path.

The first is `Double64`, if last-bit fp64 is ever wanted; §3.9 lists the scalar functions to check.

The second: Phase 0-A enumerates the offsets Gila's **misaligned external equal-cell** pairs produce
(origins differing by a non-integer number of cells). Those are off the gcd lattice, so if any of
them sits in the near band where neither the whole box nor the octant split converges, the only route
left is the k-series at 6–21 s per offset. The measurement will say whether that set is empty (likely:
the octant split needs no lattice and covers ρ up to ≈ 0.87, i.e. two cells) or whether we want a
cheaper treatment for it. Not a blocker; flagged because it is the one place where the wiring could
turn out slow rather than wrong.
