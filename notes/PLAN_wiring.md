# Plan: wiring the analytic integrals into Gila

Branch `paul-farfield`, repo `/Users/pvirally/.julia/dev/GilaElectromagnetics`.
Status 2026-09-09: **executing. Phases 0, 0.5 and 1 are complete and independently verified.**
Phase 0 measurements in `notes/phase0/{A,B}.md`; Phase 1 bracketing in
`notes/phase1/scratch/brkt.txt`. Suite 1731 pass / 5 broken / 1736 (two of the broken are the
slender PSD assertions Phase 2 flips green, §4.8). `src/` is 7975 → 8085 lines; the large
deletions land in Phase 2. **Phase 2a done and verified 2026-09-09**: `integrals/glaVacIntFar.jl`, 1897 lines from 2009,
109 functions ported and 8 dead ones dropped, bitwise identical to `notes/farfield/farfield.jl`
across **2,540,737 comparisons, 0 mismatches**, proved twice (verbatim and after 63 renames plus
the comment rewrite), with identical route histograms both times — equal-cell (4361, 412, 140),
cross-scale (2907, 4239, 24). The X path is ported too, so Phase 3 only wires.
**Phase 2b done 2026-09-09**: far field wired into the equal-cell self and external builds, the
`farSet` memo keyed `(shape, T, frequency, tol, nBlk, nMin)` (§4.1), the shape-table disk cache behind
`shpCch = true` defaulting off, and **both slender PSD assertions flipped green** — `lamMin = +3.2501e-4`
at `f = 1+0.1i`, matching `merge.md`'s `+3.25e-4` to three digits, and −45.9 eps·lamMax at `f = 1`
(§4.8's prediction of −2.1 eps was optimistic, but the sign is right and it is 100× inside the bar).
Suite **1770 pass, 3 broken, 1773**; battery still 2,540,737 comparisons, 0 mismatches.
**Line accounting, honestly:** `src/` is 7975 → 9962, i.e. **+1987**. The two analytic libraries are
3170 lines and 1183 lines of old code have gone; the rest of the quadrature is still standing because
the cross-scale path needs it. Phase 3 is where the balance moves.
**Phase 3 in progress 2026-09-09**: the cross-scale wiring and the 2b deletions are in; `src/` is
7975 → 9473. Two decisions landed this round. **(a) A pair that is both off the cell lattice and
under a cell of gap is refused**, not routed to `farBlkX!`: `GlaCmpVol` already forbids
misalignment, so it is reachable only from a hand-built two-volume `GlaOprVac`, and it was
1e-6-accurate at best. The boundary is measured, not assumed — sweeping the x gap of a half-cell
off-lattice pair, route 3 appears at every sub-cell gap (0, ¼, ½, ¾ cell) and vanishes at one cell,
where all 27 circulant offsets take routes 1/2 and agree with the exact-rational `farTnsX` to
2.4e-15 max-norm (two of three probes bitwise). Both `extOff`'s half-cell rounding and `nerOff`'s
shift-aware sizing are load-bearing on live paths: half-cell shifts return `nothing` under
round-to-even, and 15 of 43 live off-lattice pairs raise "needs octant l = 54, table holds 51"
without the shifted sizing. **(b) The six-cell proximity warning is deleted** — `prxCelMin`,
`_prxGap`, both `prxChk` methods, the `prxWrn` keyword on every constructor, and `test/prxTest.jl`.
Its claim ("quadrature limited … about 1e-8 at a gap of 6 cells") is false at every gap now that
the expansion is exact: 6.5e-16 at one coarse cell. Suite **1764 pass, 0 fail, 0 error, 2 broken**
(1777/2 error before: −17 from `prxTest.jl`, +2 from the two `cntTest` errors turning into
`@test_throws`, +2 new off-lattice separated assertions).
Reviewed 2026-09-09 (hollow-abstraction audit, style audit, four decisions from Paul); the changes
from that pass are marked **[R]** where they overrule the original text.

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
   `intConTest.jl`'s order-convergence test. **[R] Also missing from this list and also deleted:**
   `wekTrp`, `wekMem`, `wekLck`, `cubRelTol`, `cubAbsTol`, the whole of `glaVacOprMemInt.jl` (§5),
   and — from the ported libraries — `jbnd`, `needMom`/`MOMJL`/`MOMF`/`MOMLK`, `farRouteStat` and
   farfield.jl's local `srfSum` (§5). No quadrature fallback: route 3 (the k-series) is the
   universal fallback and needs no lattice, so nothing can fail to be computed. Drop `HCubature` and
   `FastGaussQuadrature` from `Project.toml`.
2. **Cross-scale is in scope now** (the trapezoid whole box, the gcd average, and the unequal-panel
   k-series all exist and are verified).
3. **Precision is a type.** Replace the now-meaningless `intOrd::Integer` field of
   `CPUKerOpt`/`GPUKerOpt` by `genPrc::Type{<:AbstractFloat}`, default `Float64`; serializers write
   it. Generation runs in `Complex{genPrc}` and narrows once into `Complex{T}` in `genEgoFur`, as
   today. **`genPrc = Float32` is refused** with a clear error: D14 (§4.2).
   **[R]** This is a straight rename of an existing field, not a new mechanism. `intOrd::Integer` is
   already abstractly typed, so `Type{<:AbstractFloat}` costs nothing that is not already paid, it is
   read once per build and never in a matvec, and it rejects an `Int` at construction for free. A
   second type parameter `CPUKerOpt{T,G}` and a generation keyword were both considered and both cost
   more code (the keyword needs forwarding through twelve signatures). **No backward compatibility.**
   `intOrd` leaves the export list in `glaVac.jl:10` and the four serializers; old files fail loudly
   on load, which is the intended outcome, and no version tag or compatibility shim is added.
   `GlaVacOprMem`'s docstring, which hard-codes "generation is always performed in `Float64`", is
   updated with it.
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
8. **[R] The contact block runs at a fixed `Double64`** (106 bits, `DoubleFloats.jl`). Superseded
   2026-09-09: this decision first said "a fixed 128 bits, exactly as route 3 does", which was right
   about needing more than Float64 and wrong about how much and how to pay for it. The reason is it is the other place where the signed 36-face-pair sum survives, and
   there the assembly amplification `Λ` is beaten by precision rather than by construction (§3.11).
   There is still no adaptive escalation, no per-offset precision decision and no `Λ` fitting -- the
   machinery §5 originally described is deleted before it is written. `Double64` is a fixed working
   type inside `cntBlk!`, independent of `genPrc`.

   **Why `Double64` and not BigFloat-128.** The digit budget (13 for the `1e-13` bar, plus the series
   loss, plus 3.4 for `Λ`) needs 57 bits at λ/32, 69 at λ/4 and 89 at a λ cell. `Double64` carries 106
   and covers all of it; BigFloat-128 carries 38 digits where 27 is the worst case, and costs 54-69x
   Float64 for that unused headroom. Measured contact-block cost, 288 face pairs, serial:

   | shape | mMax | Float64 | **Double64** | BigFloat-128 |
   |---|---|---|---|---|
   | λ/32 | 22 | 0.35 s (measured) | **~3 s** | 24.1 s (measured) |
   | λ/8 | 42 | ~7 s | **~47 s** | ~6 min |
   | λ/4 | 63 | ~50 s | **~6 min** | ~47 min |

   288 face pairs, serial; divide by the thread count for a threaded fill. Everything but the two
   measured λ/32 entries is a 6-pair sample on the worst offset `(1,1,1)`, scaled by 48 and divided by
   the 2.9x offset bias measured on λ/32. The per-call ratios are stable across shapes: **`Double64`
   is 7.0-8.6x Float64, and BigFloat-128 is a further 7.5-8.0x on top of `Double64`** — that last
   factor is what this decision buys back, for headroom the digit budget says is unused.

   **`moments.jl` runs under `Double64` unmodified**, at 7.49e-32 relative against a 256-bit
   reference -- 1.5 ulp of `Double64` -- and *identically* at `mMax = 22` and `mMax = 63`, so the term
   count does not degrade it. Every scalar function it calls (`cispi`, `sinc`, `asinh`, `atan`,
   `hypot`) is accurate there. This was the one thing that had to be checked before Phase 1 and it is
   clean.

   **One `Double64` trap, found and fixed in Phase 1a.** `Rational{BigInt}(::Double64)` routes through
   `BigFloat` and throws `InexactError` on values it cannot scale to an integer at `2^106`, and
   `parBxs` converts every panel coordinate that way to get exact breakpoints. Measured: dyadic scales
   (`1//32`) are fine everywhere, `1//7` throws on separated pairs, and **`1//10` throws on the contact
   path itself** — and λ/10 is an ordinary discretization, so this is a likely case, not a corner.
   Fixed by `cnvQ`, which converts through the two `Float64` halves (`HI(x) + LO(x)`), exact and total
   because every finite `Float64` is a dyadic rational. Verified 12/12 shape × precision × offset
   combinations, and bitwise-identical on the 576 comparisons that already passed. **Do not simplify
   `cnvQ` back to `Q(x)`** — the comment above it says why.

   **Note the budget problem is older than this decision.** At λ/4 the contact block is ~55 s *in
   Float64*, because λ/4 needs `mMax = 63` and §1's "0.1-1 ms per face pair" was quoted for 14
   moments. §7's near-field acceptance is restated accordingly; it was unmeetable as written at any
   precision.

   **Measured in Phase 0-B (`notes/phase0/B.md`), and the margin is not close.** Two independent
   losses stack on the contact block, and Float64 does not survive their sum at production cell
   scales:

   | | digits lost, Float64 |
   |---|---|
   | assembly `Λ` (worst genuine entry, `Λ ≈ 2.3e3` at the corner offset `(1,1,1)`) | 3.4 |
   | series cancellation, λ/32 and slender cells | 0.51 |
   | series cancellation, λ/8 | 1.61 |
   | series cancellation, **λ/4 — a production and acceptance shape** | **4.27** |
   | series cancellation, λ/2 | 9.86 |
   | series cancellation, λ (10 of 35 sub-code/frequency cases do not converge at all) | 10.23 |

   So at λ/4 a Float64 contact block leaves `16 − 4.27 − 3.4 = 8.3` digits, about `5e-9` — which
   misses §7's per-entry `1e-13` bar by four orders and sits *on* the 1e-8 publishability bar. At λ
   cells it leaves 2.4 digits. At 128 bits the same worst case leaves 30 digits, and `eps₁₂₈·Λ` is
   `1.3e-35`. The adaptive branch this decision deleted would have had to fire on every production
   shape from λ/4 up; the fixed choice is not the conservative one, it is the only correct one.
   (`Λ` appears to reach 1.5e18 on 45 rows; every one has `|G_ab| ≤ 1.4e-20` — entries that are zero
   by symmetry, where the ratio divides by a zero and means nothing. Ignore them.)
9. **[R] `DoubleFloats.jl` is a real dependency, and `genPrc = Double64` is supported.** Superseded
   2026-09-09: this decision first put `Double64` out of scope. Decision 8 needs it anyway, so it is
   a `[deps]` entry, not an extra. `genPrc` **defaults to `Float64`** and that is the verified
   far-field configuration -- the far field is already at the Float64 noise floor (§3.11), so running
   it in `Double64` would cost 5-20x per operation for nothing and would break the measured 0.200 s
   far fill at 128³. `genPrc = Double64` is available for anyone who wants it and gets a smoke test,
   not a full verification matrix. The contact block is `Double64` regardless of `genPrc` (decision 8).

   **Dependency ledger for this work** (Paul asked to be told): **+`DoubleFloats`** (Jeffrey Sarnoff;
   `Double64` is a pair of `Float64`s, 106-bit significand, `eps = 4.93e-32`, `isbits`, 16 bytes, no
   heap traffic; its closure pulls in ~20 indirect packages including `GenericLinearAlgebra`,
   `GenericSchur`, `MatrixFactorizations`, `FillArrays`, `SpecialFunctions` and **`Quadmath`**, the
   only one with a binary artifact). **+`Scratch`** (§3.5; already a test extra).
   **−`HCubature`**, **−`FastGaussQuadrature`** (§3.1). Net direct: a wash.
10. **The near field never needs unequal cells.** Gila remeshes touching composite region pairs to the
   fine scale (`_sndBlk`, confirmed by `cmpOprTest.jl:100`), and a direct unequal touching pair goes
   through `genCntVol`/`egoCntOut!`, which averages self tensors of the gcd cell. Every contact
   evaluation is therefore equal-cell. `moments.jl` is wired for equal cells only.

### 3.11 What fp64 and fp32 actually get

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
- **[R] fp64 is already at the Float64 noise floor in max-norm.** Over the sixty random audit cases
  (aspect ratios to 1020, five frequencies), `actual / (eps * max|G_ref|)` is 0.636 min, **2.3
  median, 16.4 max** -- some cases below one ulp -- and the a posteriori rule
  `bound + 20 eps max|T|` is never exceeded (`farfield/tex/certificate.tex`). There is no reasonable
  headroom left to chase here. The per-entry `1e-14` figures elsewhere in this document are larger
  only because a small entry inherits absolute error scaled to `max|T|`; that is the formulation, not
  the arithmetic, and no precision fixes it. Note also that `TOL = 1e-14` already sits below the
  16.4-ulp floor of about 3.6e-15, so tightening `tol` adds terms without improving the answer --
  which is the same observation as §3.4's "`c` cannot be 1".
  Three things are genuinely *not* at the floor and are Phase 0-A's real targets: the single
  unequal-cell entry at 5.3e-14 (about 240 ulp, unexplained, one of 489 references); the 0.13 % band
  at `rho = 0.55-0.60` certified only to 1-4x tol; and route 3 plus the contact block at
  `eps*Lambda` up to 2.2e-13, which §3.8 closes by fixing them at 128 bits.
  **[R] All three are now measured (Phase 0-A, `notes/phase0/A.md`).** (i) The 5.3e-14 entry is
  located exactly — `tables/k_adv.txt:77`, the λ cube against a λ/32 cell at `R = (17.5,15.5,15.5)g`,
  `f = 1`, route 2 — and the mechanism is the structural small-entry effect of this section, not
  `l`-sum cancellation and not a route-2 defect. Nothing to fix. (ii) The `ρ = 0.55-0.60` band is
  1392-1544 offsets (the "1800" was an order-of-magnitude figure); raising `LMAX` by 4-8 shells
  closes the margin by one to two orders at negligible cost, and that is the cheap fix. (iii) A
  compensated `l`-sum accumulator is worth **0-9 ulp** over 25 swept cases — not worth having.
  Do not implement it.
- **Last-bit fp64 would need `genPrc = Double64`, and is out of scope (§3.11).** Recorded
  here only so nobody re-derives it: both libraries are generic with every constant typed, the shape
  tables are stored at 192 bits (Double64 carries 106), and the selection arithmetic is already
  forced to Float64 for reasons that do not change. What would be unverified is whether `cispi`,
  `sinc`, `asinh`, `atan`, `hypot`, `ldexp`, `exponent` and the table reader's `BigFloat` conversions
  are accurate for `Double64`. Given the bullet above -- max-norm is already at the floor -- there is
  currently nothing for it to buy.

Two things are worth stating once, because they would *not* be fixed by more generation precision.

**[R] There are two distinct amplifications, and only one is gone by construction.** Getting this
wrong in either direction is expensive, so:

- *Intra-moment cancellation*, the geometry-level inclusion-exclusion inside `pairMom`, **is
  eliminated entirely**. Every primitive is written on the shifted range `[lo, hi]` rather than as a
  difference of corner-box values, so every box enters with a positive weight; with the delta-grouping
  (1.31 → 1.03 digits) and van Oosterom-Strackee for the solid angle, the worst assembled loss is
  1.03 digits (`moments.tex` §"The three-dimensional box"). This is the principal achievement of the
  analytic-integral work and nothing here qualifies it.
- *Inter-face-pair amplification* `Λ`, the **signed** sum of 36 face-pair values into one tensor
  entry by `srfSum!`, is 10-1000x and reaches 1.67e5 on the slender cell's long axes. Far-field
  routes 1 and 2 **also** eliminate it by construction, because the volume form never forms that sum,
  and that takes all but a handful of offsets. But it **survives on route 3 and in the contact
  block**, the two places where the face-pair sum is still formed, and there it is beaten by
  precision: `Λ ≤ 1002`, `eps*Λ ≤ 2.23e-13`, which is why route 3 runs at 128 bits, and why §3.8 puts
  the contact block there too. **`srfSum!` is therefore not deletable**, and neither is `Λ`'s role in
  choosing that precision.

Second: for complex `f` a part of an entry carrying less than about 1e-3 of that entry's own modulus
is accurate only in absolute terms. The latter is a property of the formulation, not of the expansion; at real
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
worth getting right.

**[R] Five caches are in play; list them with their keys, because this hazard is about a key.**
`SHPC` (shape tables, `(a, b, nMax)`, precision-independent at 192 bits); the far-set memo
(`(shape, f, tol, nBlk)`, the subject of this section); `MOMC` and `MOMCX` (route-3 panel moments,
`(D, sQ)` and `(sT, sS, RQ)`, BigFloat at `KSRPRC`); and the new contact-moment memo (`(scl, mMax)`,
frequency-free), which replaces `wekMem`/`wekLck` — **neither of which appears in §3.1's delete list
and both of which go.**

**[R] Phase 0-A sharpening:** `farSetup` cost is essentially *flat* in `nBlk`; the real cost driver
is the offset-list size. So `nBlk` belongs in the key for the correctness reason this section is
about (a set built for 32³ and reused at 128³ exceeds `rHi`), not for a cost reason. The verification suite's own part (b) committed this bug and lost a factor 1.2
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

**[R] Both are one function with two methods, not two selectors.** "Add an entry point" is exactly
where a second copy of the routing logic gets written. The general form is the body and the old
signature becomes a one-line method: `farRte(fs, R::NTuple{3,T})` with `(fs, D::NTuple{3,Int})`
forwarding `D .* s`, and `farRteX(fx, m::NTuple{3,Int})` with the `RQ::NTuple{3,QI}` method going
through `xLat` first. If the diff contains two functions that both decide a route, it is wrong.

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
- **Route 3 costs 6–21 s per offset** (BigFloat, `N = 13–44`). **[R] Measured higher in Phase 3: 25 s on a λ/4 cross-scale offset and 41.6 s on a λ/16 equal-cell near offset.** Use those when costing anything that leans on route 3. It is reached by 67 offsets of the
  slender needle and by nothing at all on a cross-scale block (0 of 1.4e6). It caches to disk. If a
  build ever hits many route-3 offsets it will be slow, not wrong.
- **0.13 % of the realistic cross-scale block is certified only to 1–4 × tol** (1800 offsets at
  ρ = 0.55–0.60 that exhaust `L = 56`); where references exist they are correct to 3e-15. Raising
  `LMAX` or routing that band to the box sum would tighten it.
- **The table writer is unlocked** across processes: two processes extending one file each append a
  duplicate segment. Harmless on read since D12 (`mrgGeo` drops columns already present), but it
  wastes disk. If the disk cache is enabled by default anywhere, add a lock.
- **[R] The a posteriori constant is 40 eps, not 20 (Phase 2b).** §3.11 quotes
  `bound + 20·eps·max|T|`, which no row of the *sixty random audit cases* exceeds. On the 172-tensor
  reference set curated for `test/ref/far.txt` it is exceeded once — λ/8 cube, `f = 1`, `D = (32,32,32)`,
  route 1: error 3.09e-18 against a truncation bound of 2.51e-19 with `max|T| = 5.61e-4`, i.e.
  **24.8 eps·max|T|**. That is consistent with §3.11's own ≈40 eps per entry for the 427-reference
  equal-cell set; the 20 came from a smaller sample. The tests use **`bound + 40·eps·max|T|`**, which
  holds on all 172 with 1.6× margin. The truncation certificate itself is never violated.
- **`est` is a scale, not a bound.** It over-states `max|T|` by 1.1–6.3× on cubes (safe direction) and
  under-states by 2–4× on a ratio-16 rod. The a posteriori certificate discharges the hypothesis at
  no cost and evaluates to ≤ 4.5e-14 over the reference set; use it in the tests.
- **Non-integer cell ratios are refused**, matching `GlaExtInf`'s own rule. A three-line rational-gcd
  change makes them work (`notes/farfield/gcd.md`, verified) if Gila ever allows them.
- `setprecision` is process-global inside route 3; `farBlock!` sets it once around the threaded
  k-series pass. Anything else in the process that cares about BigFloat precision must not run
  concurrently.

### 4.5 [R] `momSer`'s convergence indicator does not detect cancellation (Phase 0-B)

`momentSeries` returns `|last term|/|partial sum|`, and §5 previously described it as though it
gated quality. **It does not, and it must never be wired as a correctness test.** It certifies
term *decay* — truncation — and says nothing about the conditioning of the partial sum.

Measured: at a λ cell, sub-code Ef, `f = 0.5+2i`, `mMax = 100`, the indicator reads `6.04e-17`,
which reads as sixteen good digits, while the value is wrong by **10.2 digits** against a 256-bit
reference. The mechanism is `cof *= z/(n+1)` with `z = 2πif`: the running term peaks near
`n ≈ 2π|f|·rMax` before factorial decay pulls it back, and the terms rotate in phase, so the partial
sum transiently holds a magnitude far above the converged answer and then cancels back down to it.
Rounding tracks the peak, not the result. The two quantities diverge exactly when peak/final is
large, which is why the indicator is trustworthy in the regime `moments.tex` measured it
(λ/32–λ/4, `f ∈ {1, 1+0.1i, 0.37}`) and misleading outside it.

Use it as a diagnostic and as the term-count stopping rule, which is what it is for. Correctness
comes from running the contact block at 128 bits (§3.8), which is what makes the whole question
moot.


### 4.6 [R] Six test files are orphaned and never run (found in Phase 0.5)

`test/runtests.jl` does not include `test/vacuum/runtests.jl`. So `intConTest.jl`, `anaTest.jl`,
`extSlfTest.jl`, **`posDefTest.jl`**, `cpuGpuTest.jl` and `serializationTest.jl` are not exercised by
`Pkg.test()` or by CI (`julia-actions/julia-runtest` runs the same entry point). The 1819/3/1822
baseline of §7 covers none of them.

This lands directly on this plan's acceptance. **`posDefTest.jl` is the vehicle for Phase 1's
"`posDefTest` tightened to the Float64 floor at `f = 1` and positive definite at `f = 1+0.1i`
including a 6×6×12 slender block", for §7's `asym(G₀)` PSD criterion, and for Phase 5's PSD checks** —
all of it in a file nothing executes. Writing tighter assertions into it would change nothing
observable.

**Measured 2026-09-09: the six files are healthy, just unwired — `Vacuum Tests | 42 pass, 42 total,
1m33.5s`**, no failures and nothing broken (`cpuGpuTest.jl` guards itself on this GPU-less machine).
So wiring them in is a two-line change and the re-baseline is **1861 pass, 3 broken, 1864 total**.
Phase 1 does that before it tightens anything there. §3.1's "delete `intConTest.jl`'s
order-convergence test" removes a file that never ran, so it changes no count either way.


### 4.7 [R] Moment scale-homogeneity, and the shipped table we chose not to build

`I_m(λs) = λ^{m+4} I_m(s)`: the face-pair moments of a cell scale as a pure power of its size, so one
table at a given **aspect ratio** determines that ratio at every size — every resolution, and (the
moments being frequency-free) every frequency. Measured 2026-09-09 against direct computation, worst
relative error over five face pairs at the corner contact offset, `mMax = 22`, `Double64`:

| case | worst rel err |
|---|---|
| cube, dyadic scales (λ/4, λ/8, λ/32, λ/128) | **0.000e+00, bitwise exact** |
| cube, non-dyadic (λ/3, λ/7, λ/10, λ/100) | 7.8e-32 … 1.7e-31 |
| aspect 1:1:1/16 and 1:2:4, dyadic / non-dyadic | 0.000e+00 / 1.6e-31, 1.8e-31 |

Dyadic scales are exact because `λ^{m+4}` is then an exact power of two. Off the dyadic grid the cost
is 2.4-3.6 ulp of `Double64`, against 2.3 ulp for direct computation itself — **scaling from a table
is as accurate as computing from scratch.**

**Two consequences.** First, this is a *test*, and a sharp one: Phase 1 asserts it with `===` rather
than `≈`. Second, it means a prebuilt moment table committed to the repository would be **one file per
aspect ratio, not one per resolution** — a single unit-cube table (~293 kB at `mMax = 63`, ~460 kB at
`mMax = 100` to cover λ/2 and λ cells) would serve every cubic cell Gila can build.

**Decided 2026-09-09 (Paul): do not build it.** It buys 3 s at λ/32 and about a minute of threaded
time at λ/4, once per session, on a workflow that builds one `G0` and then solves repeatedly — while
costing a committed artifact that must stay in step with `glaVacIntMom.jl` forever, a CI regeneration
test, and a failure mode of silently wrong Green functions rather than a slow build. Deferring costs
nothing: the homogeneity relation above is the whole enabling fact, it is verified, and the table is
derivable from it at any later date with no design change. **Recorded here so the option stays visible
and nobody re-derives it.**


### 4.8 [R] The two errors cancel today, so intermediate states look worse (Phase 1b)

Gila's contact rule and its far field are both wrong on the slender cell, **in opposite directions**,
and today they partially cancel. From `farfield/scratch/reports/merge.md` §6, the 6×6×12 slender
block's `asym(G₀)` smallest eigenvalue at `f = 1+0.1i`:

| build | `lamMin` |
|---|---|
| Gila as shipped (both wrong) | −9.94e-3 |
| exact far field, Gila's contact rule | **−1.465e-2** (worse) |
| exact contact, Gila's far field (measured Phase 1b) | **−1.65e-2** (worse) |
| both exact | **+3.25e-4**, no negative eigenvalue |

**Fixing either half alone makes the measured indefiniteness worse**, because it removes the
compensating error at separation 2. This is not a regression and must not be treated as one.

Two consequences. **Phase 1's PSD acceptance was unreachable as written** — §6 asked Phase 1 to
assert the slender block positive definite at `f = 1+0.1i` and PSD to the Float64 floor at `f = 1`,
which needs the far field too. Phase 1b correctly wrote those two assertions as `@test_broken` naming
Phase 2 rather than retargeting them silently; **Phase 2 flips them green** and that is where the
acceptance belongs. And more generally: any metric that mixes contact and separated offsets is
untrustworthy between Phase 1 and Phase 2. Judge the halves separately until both have landed.


## 5. Architecture

**[R] The golden rule, above every other instruction here: write as little code as possible.** Less
code is the deliverable. Every item below that says "delete" or "do not port" outranks every item
that says "add". If a phase finishes with fewer lines in `src/` than it started with, that is a
success, not a shortfall.

**[R] Code organisation (Paul, 2026-09-09): the integral libraries live in
`src/vacuum/integrals/`.** The repo prefixes filenames after their directory (`src/vacuum/` →
`glaVac*`), so the new directory extends that to `glaVacInt*` and the now-redundant `OprMem` drops
out of the names:

| file | was going to be | holds |
|---|---|---|
| `src/vacuum/integrals/glaVacIntMom.jl` | `glaVacOprMemMom.jl` | the near-field moment library, `cntBlk!`, `sclEgo`/`sclEgoN` |
| `src/vacuum/integrals/glaVacIntFar.jl` | `glaVacOprMemFar.jl` | the far-field expansion, both block fills |

`module GilaVacuum` is a flat nested-include chain with no sub-modules, so this is a path change and
an `include` edit, nothing more. `facPar`/`srfScl`/`srfSum!` still go to `glaVacOprMemGen.jl` as
below — they are the generation path's assembly step, not part of the integral libraries, and moving
them too would be churn for its own sake.

Both ports carry the numerics over verbatim and are proved bit-identical to the notes implementation
before any restyling:

- **`integrals/glaVacIntMom.jl`** (from `moments.jl`): `parBxs` with exact Rational breakpoints,
  `box2`/`box3`, `parMom`, the panel bookkeeping (`boxFace`, `facePair`, `panExp`, `axsCnv`), `momSer`
  with its `|last term|/|sum|` indicator (a term-count stopping rule, **not** a correctness gate --
  see §4.5), and `cntBlk!(egoToe, vol, cmpInf)` filling the eight contact
  offsets plus the touching shell **at a fixed 128 bits** (§3.8). Moments memoized per `(scl, mMax)`
  — the frequency-free successor of `wekMem`/`wekLck`, which go. Series order from the a priori bound
  `(kD)^{N+1}/(N+1)!·e^{kD}`.
  **[R] The port set is computed, not guessed.** Call-graph reachability from the production entry
  points (`faceMoments`, `pairMoments`, `momentSeries`, `pairBxs`, `facePair`, `boxFace`, `panExp`,
  `axsCnv`) gives **52 functions to port and 35 to leave behind**.

  *Port these 52* (plus `faceMoments` itself, a multi-line one-liner the scan misses): `asnD`,
  `asnMz`, `atnY`, `axsCnv`, `box2`, `box2Tay`, `box2lad`, `box3`, `box3Ser`, `box3Slv`, `box3Thn`,
  `boxFace`, `ctrM1`, `dchs`, `ddAtn`, `djSeg`, `domAxs`, `facePair`, `gauLeg`, `jFal`, `jSeg`,
  `kBox3`, `lad1`, `momentSeries`, `offD`, `pairBxs`, `pairMoments`, `panExp`, `pikCmb`, `polMom`,
  `potBox`, `potCrn`, `powSum`, `psiBox`, `psiWm`, `psiWp`, `scl2`, `sclPan`, `seg1`, `slAng`,
  `slvAxs`, `slvOrd`, `thnAxs`, `wgt2`, `wgtSpl`, `wmBox3`, `wmSed`, `wmSeg`, `wpBox3`, `wpSeg`,
  `xmAt`, `xmLog`. Note `gauLeg` is among them — that is §4.4's thin/slender rule, which is kept
  deliberately; it does **not** fire on contact pairs (measured: zero calls).

  *Do not port these 35*: `axsSig`, `bCof`, `cSum`, `canonPanels`, `classifyFull`, `dJ`, `dPsi`,
  `dblFac`, `geomCode`, `gm00`, `gm10`, `gm11`, `ivrLap`, `jInt`, `jMom`, `jMomExp`, `jOdd`, `kBox`,
  `lMom`, `loadRef`, `mIJ`, `momFlt`, `momGen`, `momVtxCop`, `pBox`, `pBoxCf`, `pMom`, `pairSig`,
  `potRct`, `prpMom`, `psiOdd`, `psiRct`, `psiSed`, `rctMom`, `vcMom` — the explicit cross-check
  forms, the reference loader, and the sub-geometry classifier with its tables (`CODES`, `SUBCODES`,
  `SUBFAM`, `REFSIG`, `FACENAMES`). About 200 lines with no production caller: `parMom` never needs a
  sub-geometry name. Phase 1's symmetry tests do need `canonPanels`, so it goes to `test/`, not
  `src/`.
- **`integrals/glaVacIntFar.jl`** (from `farfield.jl`): the exact table build (in-place GMP), the shape cache
  (memory always, disk behind the keyword), `farSet`/`farSetX` with the Theorem A/A′/B selector and
  the `[rNc, rHi]` guard, `tnsWhl!`/`tnsOct!`/`tnsGcd!`, route 3 delegating directly to
  `glaVacIntMom.jl`, and the two block fills. Both interface changes of §4.3 land here.
  **[R] Do not port**: `jbnd` (dead, zero callers); `needMom`, `MOMJL`, `MOMF`, `MOMLK` (a
  runtime-`include` and world-age shim that exists only because the two files are separate scripts —
  inside one module route 3 just calls `momSer`); `farRouteStat` and the other verification hooks;
  and farfield.jl's local `srfSum`, which duplicates `srfSum!` in `glaVacOprMemGen.jl`. **[R]
  Collapse** `boundL` and `boundLX`: byte-identical five-line bodies over `fs.thr` and `fx.thr`, so
  one function with two methods.
  `farTns`/`farTnsX` (today `farTensor`/`farTensorX`) have **zero** callers in generation — Gila
  fills blocks. Keep them, because the whole reference test matrix calls them, but keep them
  *knowingly* as test entry points; do not build a per-offset path through `src/` alongside the block
  path.
- **[R] `glaVacOprMemInt.jl` is deleted.** After the integrators go it holds four functions;
  `glaVacOprMemGen.jl` already owns `srfSum!` and is the only caller of `facPar` and `srfScl`, so
  those two move there and `sclEgo`/`sclEgoN` move to `integrals/glaVacIntMom.jl` (they are the kernel the
  moment series sums). One fewer file, one fewer include. `glaVacOprMemIntSup.jl` is deleted outright.
- **[R] `facPar`, `srfScl` and `srfSum!` all survive and must become generic in `genPrc`.** ~~`srfScl`~~ **[R] `srfScl` did NOT survive** — after the Phase 3 sweep it has zero consumers (the contact path scales inside `parMom`) and was deleted. §3.11
  settles the "if anything still needs them" question: the contact path still forms the signed
  36-face-pair sum. They are `ComplexF64`/`SVector{36,Float64}` today and will not carry `genPrc`
  otherwise. `facPar()` becomes a module `const`, not a nullary function threaded as a runtime
  argument.
- `glaVacOprMemGen.jl`: `genEgoCrcSlf!` becomes `cntBlk!` + `farBlk!` + identity term + embedding;
  `genEgoCrcExt!` the same on the offset vector, keeping `egoCntOut!`/`genCntVol` for the contact
  class unchanged; `egoToeCrc!` and `sepGrd` glue unchanged.
  **[R] Strip the dead argument train.** `trgFac`, `srcFac`, `facPar` and `srfScales` are threaded
  through `genEgoCrcSlf!`, `genEgoCrcExt!`, `egoFunInn!`, `egoFunOut!`, `egoFunSng!`, `egoSrfAdp!` and
  `egoSrfFxd!`. This plan deletes every consumer of them on the self path; delete the parameters too
  rather than forwarding arguments nobody reads.
- **[R] Do not port `gilacmp.jl`'s `wekFrm`, `wekPnl`, `canonArg`, `MAP`, `LBL`.** That is the
  18-entry `wekS/E/V` layout adapter — a shadow of an interface this plan deletes. `cntBlk!` writes
  tensors directly. It belongs in the Phase-1 bracketing test and nowhere else.

### 5.1 [R] Style, and it is not optional

**Naming.** `src/` uses three-letter syllables — stem length a multiple of three — in camelCase,
plus optional trailing single-capital markers: `sclEgoN`, `wekEDir`, `egoCntOut`, `crcIndClc`,
`intNE`. Full English words are reserved for exported API (`GreenOperator`, `discretize`). **79 of
the 156 function names across the two libraries are off this rule** and must be renamed on the way
in; no phase previously budgeted for it, so the porting phase owns it. Three names in this plan's own
earlier draft were wrong and are corrected above: `nearBlk!` → **`cntBlk!`**, `pairBxs` →
**`parBxs`**, `pairMom` → **`parMom`** (`facPar()` already establishes `Par` = pair). `farSet`,
`farSetX`, `farBlk!`, `farBlkX!`, `tnsWhl!`, `tnsOct!`, `tnsGcd!`, `momSer` are correct as written.
`box2`/`box3` are fine by the `wekS` pattern (three-letter stem plus index marker); `box2lad` needs
`box2Lad`.

**Dependencies.** Do not avoid them. Paul: "do not be afraid of using external dependencies!! If we
can use other people's code to simplify our own code and make Gila more legible and more enjoyable to
read and edit, then we should!" The golden rule is *less code in Gila*, not fewer names in
`Project.toml` — a package that deletes Gila code is a win by that measure. Tell Paul when you add
one, with what its closure pulls in and whether anything in it is a binary artifact.

**Comments. Code must be self-documenting.** Do not over-comment — comments are harder to read than
code.

- Triple-quoted docstrings for **exported** functions only.
- Everything else gets a comment **only where the code is deliberately confusing**, and then a
  small, pithy one.
- The two libraries carry roughly 150 one-line docstrings on internal functions. **Nearly all of
  them go.** This is a net deletion and it is meant to be.
- **[R] Say WHAT the function computes, not how it dodges cancellation.** This corrects the rule
  Phase 1a was given, which asked for the *mechanism* and produced comments like `logMns`'s
  "cancellation-free below 1/2" — true, and useless to someone meeting the function for the first
  time. **The reader has none of this project's history.** Lead with the plain mathematical form, then
  tag the cancellation in a few words. Do not explain which term cancels, by how much, or why this
  grouping was chosen over another.

  | function | wrong | right |
  |---|---|---|
  | `logMns` | `# cancellation-free below 1/2` | `# x - log(1 + x), avoiding catastrophic cancellation` |
  | `asnMns` | `# asinh(z) - z and x - atan(x), cancellation-free below 1/2` | `# asinh(z) - z, avoiding catastrophic cancellation` |
  | `cshMns` | `# all-positive series: d cosh d - sinh d cancels away for small d` | `# d cosh(d) - sinh(d), avoiding catastrophic cancellation` |
  | `asnD` | `# log1p branch below 1/2; the plain asinh difference cancels there` | `# the asinh difference log((t2 + sqrt(t2^2 + dsq))/(t1 + sqrt(t1^2 + dsq))), avoiding catastrophic cancellation` |
  | `pikCmb` | `# dh^2 J^h - dl^2 J^l and dh^2 D + ddsq J^l are both exact; take the smaller term sum` | `# dh^2 Jh - dl^2 Jl, by whichever of its two exact forms cancels less` |

- **Spell the name out when the acronym does not.** The three-syllable convention is mandatory and it
  makes some names opaque; a few real words cost nothing. `pikCmb` is "pick combination", `ladFal` a
  falling ladder rung, `cshMns` "cosh minus". If a reader cannot tell what a function is *for* from
  its name, the comment must supply it.

- **Fine detail is not required.** The high-level headline must be clear; the intricate reasoning
  behind a particular grouping is not. Two exceptions, one short line each: van Oosterom–Strackee for
  the solid angle (a four-corner arctangent difference cancels, so the odd form is deliberate), and
  constants that came out of a theorem rather than a preference (`LMAX`, `NCAP`, `LEXT`, `TABPRC`,
  `BLDPRC = 256`, `KSRPRC = 128`). Code cannot state a theorem; that is what earns the comment.
- No banner comments. No restating the signature in prose. No section dividers.

Commit nothing; leave the tree dirty. Suite:
`JULIA_NUM_THREADS=auto julia --project -e 'using Pkg; Pkg.test()'`. Reports carry files, numbers and
what could not be verified, never status. The Manifest was stale and has been resolved;
`using GilaElectromagnetics` works.

## 6. Phases

**[R] Strictly serial.** Phase 0's two agents are read-only on `src/` and run in parallel; everything
after runs one agent at a time on the live tree, each starting from the previous one's output. Phases
1 and 2 both edit `genEgoCrcSlf!` and the `genPrc` plumbing, so nothing here is parallelised and no
worktrees are used. Order: **0 → 0.5 → 1 → 2 → 3 → 4 → 5 → 6**.

**Phase 0 — measurements (two sonnet agents in parallel, read-only on `src/`).**

A (far field). **[R] Rescoped.** `TOL = 1e-14` already *is* `c ≈ 45·eps(Float64)` and produced every
number in §1, and §3.11 records that the far field is at 2.3 ulp median / 16.4 ulp worst in max-norm —
the Float64 noise floor. So 0-A does not re-derive `c`; it **confirms the floor and explains the
places that are not at it**:
  1. the single unequal-cell entry at 5.3e-14 (~240 ulp), one of 489 references — what offset, what
     shape, what route, and whether it is the `l`-sum cancellation or something structural;
  2. the 0.13 % band at `ρ = 0.55–0.60` (1800 offsets exhausting `L = 56`, certified only to 1–4×
     tol) — whether raising `LMAX` or routing it to the box sum closes it, and at what cost;
  3. whether the `l`-sum accumulation in `boxAcc`/`tnsWhl!` leaves anything on the table (the file
     already carries `twoSum`, `twoPrd`, `ddAdd` for the phase) — report the number, do not implement.
Then, unchanged: table build time, peak RSS and file size vs `L` for the production shapes and two
unequal pairs; **the offsets Gila's *external* equal-cell pairs actually produce** (enumerate `Δorg/s`
over `extSlfTest.jl`, `crsSclTest.jl`, `cmpOprTest.jl`) and how many land in the route-2/route-3 band
— this is §8's one open question and is the highest-value item in 0-A; and `farSetup` cost vs `nBlk`
for the memo key of §4.1.

B (near field). **[R] Rescoped.** No `Double64` (§3.9) and no precision-escalation constants
(§3.8 fixes the contact block at 128 bits). What remains: 288 face pairs per shape at `mMax` 12 and
30, serial and threaded; series digits lost vs 256 bits for `f ∈ {1, 0.37, 3, 1+0.1i, 1+i, 0.5+2i,
2+2i}` at λ/32, λ/8, λ/4, λ/2, λ and the slender cell; **one confirming number** — `srfSum!`
amplification `Λ` on the eight near offsets per shape, to check that `eps₁₂₈·Λ` is far below target
(it will be; record it and move on, do not fit a constant to it); and the baseline full suite (pass
count, time), which Phase 5 compares against.

**Phase 0.5 — `genPrc` plumbing (sonnet). [R] New, and it must come first.** Every generation
signature is hard-typed `ComplexF64` today — `genEgoCrcSlf!`, `genEgoCrcExt!`, `genEgoSlf!`,
`genEgoExt!`, `egoToeCrc!`, `srfSum!`, `genEgoFur(::Array{ComplexF64,6})` — and `srfScl` returns
`SVector{36,Float64}`. If this lands after Phases 1–3 they will all be written in `ComplexF64` and
then rewritten. So: the `intOrd` → `genPrc` rename (§3.3), `intOrd` out of the export list and the
four serializers with no compatibility shim, `egoToe::Complex{genPrc}` with the single narrowing at
embedding, the `Float32` refusal, and `GlaVacOprMem`'s docstring. (The §3.5 scratch-space keyword
moves to **Phase 2**, where the shape cache it configures actually appears.) Suite green on the
existing numerics before Phase 1 starts.

**The interim problem, and its answer.** The integrators being deleted still read `intOrd` — the
`wekTrp` memo key and `gauQud(intOrd(cmpInf))` in `glaVacOprMemGen.jl:134,140` — and they must keep
working until Phase 1 and Phase 4 remove them. Do **not** keep both fields. Give the doomed code a
module `const INTORD = 48` (the current default) and delete it in Phase 4 with its last caller. Tests
that construct a `CPUKerOpt` positionally with an order need the same treatment; `quadTest.jl` and
`intConTest.jl` are deleted wholesale in Phase 4, so only `vacTest.jl` and
`vacuum/serializationTest.jl` need editing now.

**Phase 1 — near field (opus).** **[R] First: wire `test/vacuum/runtests.jl` into
`test/runtests.jl` and re-baseline (§4.6) — the PSD acceptance this phase is supposed to tighten lives
in a file that never runs.** Then port, wire `cntBlk!`, reproduce `wekTrp`/`egoFunSng!` against
DIRECTFN at intOrd 48 and 64 (bracketing) *before* deleting them, then delete. **[R]** The bracketing
harness — `wekFrm`/`wekPnl`/`canonArg`/`MAP` from `gilacmp.jl` — is throwaway test scaffolding under
`test/`; it must not reach `src/`. Tests: even moments vs exact rationals, homogeneity in `f`, **[R] homogeneity in
*scale*, asserted bitwise** — `parMom(λs) === λ^(m+4) · parMom(s)` with `===`, not `≈`, which holds
exactly for dyadic λ (§4.7) and is an unusually sharp check on the whole moment path — Ec
`b↔c` and Vp `a↔c` symmetry, `test/ref/` moments at 1e-13, the slender shell offsets (the 117 % case),
`posDefTest` tightened on the **cubic** blocks (measured: `lamMin/lamMax > -1e-12` at `f = 1`, worst
−7.57e-14); **[R] the two slender assertions are `@test_broken` here and flip green in Phase 2** —
they need the exact far field as well, and fixing one half alone makes the number worse (§4.8).
fp32 build bitwise equal to the narrowed fp64 build.

**Phase 2 — far field, equal cells (opus).** Port, shape cache with the §3.5 scratch-space keyword
(moved here from Phase 0.5), `tol` from `genPrc`, the
offset-vector entry point of §4.3, the `(shape, f, tol, nBlk)` memo key of §4.1, route 3 in `genPrc`;
wire `farBlk!` into the self and external equal-cell builds; delete the fixed rule and the GPU batch.
Tests: `test/ref/` tensors across shapes, directions and the seven frequencies at 1e-13; the near/far
overlap at two cells (k-series vs expansion — a free end-to-end check of bookkeeping and
normalization); `M_ab(R) = M_ba(−R)`; thread-count determinism; the a posteriori certificate never
violated. **[R] Flip Phase 1's two `@test_broken` slender PSD assertions green** (§4.8): with both
halves exact, `merge.md` measured `lamMin = +3.25e-4` at `f = 1+0.1i` and −2.1 eps·lamMax at `f = 1`.

**Phase 3 — far field, cross-scale (opus). [R] Nothing left to port — this phase only wires.**
Phase 2a ported the X path with the equal-cell one (it shared 88 of its 89 dependencies, so splitting
would have meant porting a shared core twice), and the integer-lattice entry point of §4.3, the fine
equal-cell set and the box sum came with it, all inside the 2,540,737-comparison bitwise proof. So:
wire `farBlkX!` into `genEgoCrcExt!`, replacing the `egoFunOut!` loop with `egoCntOut!` left in place,
and **finish the deletions Phase 2b had to leave standing** — the fixed and adaptive rules and the
external half of the GPU batch survive 2b only because the cross-scale path still routes through them
(2b's report says exactly which). `HCubature` and `FastGaussQuadrature` leave `Project.toml` here if
they did not leave in 2b, and `glaVacOprMemInt.jl` and `test/quadTest.jl` finally go. Tests: the cross-scale reference tensors at ratios 2, 4, 8, 16
on one, two and three axes at three frequencies; `crsSclTest.jl` retargeted from 1e-6 to 1e-12; the
swap identity `G(R; sT, sS) = (V_s/V_t) G(−R; sS, sT)`; the reflection identity; a composite operator
build end to end.

**Phase 4 — deletion sweep (sonnet). [R] Rescoped:** the `genPrc` work moved to Phase 0.5, so this
phase is pure deletion. Delete every dead function and constant from §3.1; `Project.toml` loses
`HCubature` and `FastGaussQuadrature`. **The test deletions are much larger than "sweep for
`intOrd`":** `test/quadTest.jl` goes **wholesale** (584 lines, 74 references to
`quadOrd`/`egoSrfFxd!`/`egoSrfAdp!`/`srfKer`/`cubFac`/`gauQud`/`facPar`), `test/vacuum/intConTest.jl`
goes wholesale (57 lines), plus 10 references in `test/vacTest.jl` and 2 in
`test/vacuum/serializationTest.jl`. Confirm no quadrature is reachable from any generation path.

**Phase 5 — adversarial verification (opus).** **[R]** Independent of Phases 1–3 in authorship, not
in schedule: it runs after Phase 4. New 220-bit references at shapes nobody has used (aspect
1e-3…1e3, λ/2 and λ cells), the seven frequencies, fp32/fp64 builds, PSD of `asym(G₀)` on 6³ and
6×6×12 blocks, `egoOpr!` against a dense reference, before/after build times at 32³/64³/128³ (λ/32)
and 32³ (λ/8, λ/4) at one and twelve threads, the full suite. Every number reported.

**Phase 6 — cleanup and final run (sonnet). [R] Rescoped 2026-09-10 by Paul.** This plan was about
getting the integrals *implemented*; the documentation is a separate job Paul will do himself,
alongside other doc work that reaches well beyond the analytic integrals. So Phase 6 does **not**
write `docs/src/concepts.md` or `docs/src/usage.md`, and **nothing moves out of `notes/`** — the
notes stay where they are.

What is left: delete the temporary files the phases created, correct the statements in the code and
docs that the work made false, and run the full suite. The known false statements are
`src/glaOpr.jl:212` ("generation is always done in `Float64` and rounded once", untrue since
`genPrc`), and the stale mentions of the deleted order schedule, DIRECTFN, `benchmark/quadCnv.jl`
and the six-cell rule in `docs/src/concepts.md` (4), `docs/src/usage.md` (3) and `docs/gilaDoc.tex`
(3). The six-cell rule is superseded by the measured lambda/3 distance rule, `notes/prx/RESULTS.md`.

## 7. Acceptance

- Suite green; no quadrature reachable from any generation path except the thin-axis rule inside
  route 3, which is kept deliberately (§4.4) and runs at 128 bits.
- **`genPrc = Float64`, the default and the verified configuration:** every tensor entry within
  1e-13 of the 220-bit reference on the test matrix (equal and unequal cells, cubic, anisotropic,
  slender, coarse, fine; real and complex `f`). This is the fp64 target and it is met by what
  already exists; 1e-14 is the measured level and 1e-13 is the acceptance bar with margin.
  **[R]** In max-norm the real level is the Float64 noise floor itself — 2.3 ulp median, 16.4 ulp
  worst over the sixty audit cases — so a max-norm regression beyond `bound + 20·eps·max|T|` is a
  defect, not a tolerance question.
- **`T = Float32` storage:** the build is bitwise equal to the narrowed `genPrc = Float64` build,
  and `genPrc = Float32` raises.
- **[R] `genPrc = Double64` is out of scope** (§3.11). Both new files stay generic in the
  number type so it remains available later, but nothing in this round tests or claims it.
- `asym(G₀)` PSD to the Float64 floor at real `f` and positive definite at `f = 1+0.1i` on the
  slender block — **[R] in a `posDefTest.jl` that actually runs (§4.6)**, which it does not today.
- 128³ self build at λ/32: far fill ≤ 0.3 s (`genPrc = Float64`, the default — the measured figure is
  0.200 s and `Double64` would break it, which is why decision 9 keeps the far field in Float64).
- **[R] Near field, restated per shape.** The old single "≤ 0.5 s" line came from §1's "0.1-1 ms per
  face pair", which was quoted at 14 moments; λ/4 needs `mMax = 63` and costs ~55 s for the 288 pairs
  *in Float64*, so that line was unmeetable at any precision. The contact block is 8 offsets × 36 face
  pairs = 288 pairs, **independent of grid size** (128³ costs no more than 8³), frequency-independent,
  and memoized per `(scl, mMax)`. The acceptance is therefore: λ/32 and the slender cell ≤ 5 s cold on
  one thread; λ/8 ≤ 60 s; λ/4 ≤ 10 min; and **zero on every later build of the same shape**.
  **[R] Phase 5 measured: slender 4.46 s, λ/8 29.8 s, λ/4 115.5 s, repeats 0.000-0.001 s — all met.
  λ/32 MISSES at 6.48 s.** The 5 s bar came from the "0.35 s x 7-8.6x" estimate above, which lands at
  ~3 s; 6.5 s is the truth. Since every repeat is free and Paul weighs matvec cost over build time,
  **the bar is corrected to the measured 6.5 s** rather than the code being optimised to reach 5. Report
  the `Double64` numbers against the Float64 and BigFloat-128 columns of decision 8. **There is no
  moments disk cache and no shipped moments table** — the in-memory memo on `(scl, mMax)` is the whole
  mechanism; see §4.7 for why, and for the option we deliberately did not take.
- Cross-scale: the realistic 1.4e6-offset block in seconds, and `crsSclTest` at 1e-12 where it is at
  1e-6 today.
- ~~**[R] Fewer lines in `src/` than before.**~~ **[R] This line was wrong when written and is
  withdrawn.** §5's golden rule is about not writing code that earns nothing, not about the total:
  the work deletes ~1620 lines of numerics and adds ~3200 of analytic library, so a net rise was
  always the intended outcome. Final: **7975 -> 9520, +1545**. Report the delta, do not target it.
- **[R] Suite baseline to beat (Phase 0-B, commit `5965fdf`): 1819 pass, 3 broken, 1822 total** —
  and note this covers only `test/runtests.jl`; the six files under `test/vacuum/` are orphaned and
  contribute nothing to it (§4.6). They pass on their own (42/42), so once Phase 1 wires them in the
  bar becomes **1861 pass, 3 broken, 1864 total**.
  The three are pre-existing: `physTest.jl:69` (`@test_broken`, Asym(G₀) PSD for overlapping
  unions), `vacTest.jl:53` (`@test_skip`, overlap guard hangs during integration) and
  `quadTest.jl:333` (`@test_skip`, no CUDA device here). **Phase 4 deletes `quadTest.jl` outright,
  so the post-change count should be 2 broken, not 3** — that drop is expected, not a regression.
  The 200.3 s wall time was taken while both Phase 0 agents were loading the machine and is **not**
  a usable timing baseline; Phase 5 must re-time on a quiet machine.

## 8. [R] Open question — closed by Phase 0-A

All three items that stood here are resolved and nothing blocks execution.

**`Double64`** is out of scope (§3.9). **fp64** was never open: §3.11 records that the far field is
already at the Float64 noise floor in max-norm.

**The misaligned external equal-cell offsets — the one real question — are a non-issue.** Phase 0-A
(`notes/phase0/A.md` §1) constructed every same-scale external separated pair reachable from
`extSlfTest.jl`, `crsSclTest.jl` and `cmpOprTest.jl`: **every one has an integer `Δorg/s`, so the
route-3 set is empty.** That is structural, not luck — `GlaCmpVol`'s constructor
(`src/glaCmpVol.jl:150`) throws on any two regions whose corners are off the common grid, so
misalignment cannot exist inside a valid composite volume, and `genEveExtInf` only equalizes cell
parity without touching `org`.

Two qualifications, neither a blocker:

- Plain non-composite `GlaVol` pairs are *not* validated. A hand-built pair at `Δorg/s = 49/3` was
  accepted by `GlaVacOprMem` today, because `egoFunOut!` works on physical distances and ignores the
  lattice. **Phase 2 still implements the offset-vector entry point of §4.3** — it is what makes that
  pair work once `farBlk!` replaces `egoFunOut!`.
- ~~Misalignment adds no route-3 exposure.~~ **[R] Wrong at sub-cell gaps, corrected in Phase 3.**
  The sweep that supported this perturbed offsets by 0.37 of a cell and never changed the route
  family, but it never combined misalignment with a gap inside one cell. It does not hold there: for
  a half-cell lateral shift at a 1/64 gap, one octant's expansion point sits at `|v| = 0.056` against
  a sub-box half-diagonal of `0.054` — on the box boundary, where no `l ≤ LMAX` converges. Route 2
  cannot absorb that pair and route 3 costs 41.6 s per offset, 15 of them.

  **Such pairs are now refused** (Paul, Phase 3). They are unreachable from `GlaCmpVol`, so only a
  hand-built two-volume `GlaOprVac` can construct one, and what the refusal replaces was the
  1e-6-accurate adaptive quadrature. Off-lattice pairs that are *separated* are unaffected — they go
  through routes 1/2 as this bullet claimed, and that part stands.
