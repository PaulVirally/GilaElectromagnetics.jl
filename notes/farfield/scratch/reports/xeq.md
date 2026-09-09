# D15 --- the equal-cell twin of D13: a `FrqSet` reused at a radius where its n-cut was never certified

Work dir `notes/farfield/scratch/xwork/xeq`.  Every timing below is followed by the 1-minute
load average on this shared machine.

## 1. The defect

`farSetup` builds a `FrqSet` for a *set of offsets*.  From those offsets it derives one radius
interval,

    rNc = min over the built offsets of (|R| on route 1, the octant vertex radii on route 2)
    rHi = 4 * nBlk * r_d                       (r_d the cell diagonal, nBlk the block width)

and computes the per-l k-series cut `nCut` as the elementwise max of `nCutVec` evaluated at the
two endpoints `k*rNc` and `k*rHi`.  Both endpoints are needed: `|h_l(kr)|` falls with r but so
does the budget `tol*est(r)/8`, and either end can bind (that is the D11 lesson).  The cut is
therefore certified on `[rNc, rHi]` and nowhere else.

`farRoute` then routed *any* offset the caller asked for.  It checked the shell count
(`L <= fs.Lw`, `maximum(Lc) <= fs.Lo`) but never the radius.  A set reused at a radius outside
its interval was answered with the wrong number of k-series terms, silently, and still returned
a certificate computed from that same wrong cut.

This is exactly D13 with `FrqSetX` replaced by `FrqSet` and `farRouteX` by `farRoute`.  It was
looked for because D13's mechanism obviously generalises; it was not otherwise suspected.

## 2. Measured, before and after

Reproducer `xeq/repro.jl` (output `repro_pre.txt` / `repro_post.txt`), realistic-misuse scan
`xeq/scan2.jl` (`scan2_pre.txt` / `scan2_post.txt`).  Shapes: `c32` = (1/32)^3, `c8` = (1/8)^3,
`c4` = (1/4)^3, `sl` = (1/32, 1/32, 1/512), all in wavelengths.  "cert" is `certEq` divided by
`max|G|`, i.e. the certificate the call itself returns.

### 2a. Worst case overall: a set that certified nothing

`farSetup(s, f; offs = (), nBlk = 128)` --- an explicit *empty* offset tuple.  Nothing is
certified, so pre-fix `rNc` stayed at `rHi` and every request was served with the far-end cut.
Scanning every route-1 offset of the block:

| case | worst D | rr | L | error vs a fresh set | fresh set vs 220-bit ref | cert | violation |
|---|---|---|---|---|---|---|---|
| sl f=1 | (2,2,6) | 0.0892 | 42 | **3.215e-08** | -- | 6.465e-15 | **4.97e6 x** |
| c32 f=1+0.1i | (1,3,0) | 0.0988 | 42 | 8.287e-09 | -- | 6.464e-15 | 1.28e6 x |
| c32 f=1 | (0,1,3) | 0.0988 | 42 | 8.040e-09 | -- | 6.719e-15 | 1.20e6 x |
| c8 f=1+0.1i | (0,3,0) | 0.3750 | 46 | 5.116e-09 | -- | 1.095e-14 | 4.67e5 x |
| c4 f=1 | (0,0,3) | 0.7500 | 48 | 1.494e-10 | -- | 9.079e-15 | 1.65e4 x |

Post-fix the same scan gives `0.000e+00` on every row: the offsets that stay inside the
interval are bitwise equal to a fresh set, and nothing is left wrong to measure.

### 2b. The realistic misuse: a set built for a far block, reused nearer

`farSetup(s, f; offs = ((100,0,0),), nBlk = 128)`, then every nearer offset the cached `Lw`
still admits.  Checked against a 220-bit face-pair reference computed in the same script:

| case | far offs \|R\| | worst D | rr | L | reused set vs ref | fresh set vs ref | cert | violation |
|---|---|---|---|---|---|---|---|---|
| sl f=1 | 3.125 | (2,2,0) | 0.0884 | 42 | **9.444e-09** | 1.119e-15 | 9.087e-15 | 1.04e6 x |
| c32 f=1+0.1i | 3.125 | (1,3,0) | 0.0988 | 42 | 8.287e-09 | 9.173e-16 | 6.464e-15 | 1.28e6 x |
| c32 f=1 | 3.125 | (0,1,3) | 0.0988 | 42 | 8.041e-09 | 2.836e-16 | 6.719e-15 | 1.20e6 x |
| c8 f=1 | 12.5 | (2,2,0) | 0.3536 | 50 | 6.446e-10 | 1.312e-15 | 1.865e-14 | 3.46e4 x |
| c4 f=1 | 25.0 | (0,2,2) | 0.7071 | 52 | 1.056e-10 | 4.708e-16 | 1.465e-14 | 7.21e3 x |

The reused set is wrong by up to 9.4e-9 where a fresh set is right to 1.1e-15: a loss of about
seven digits, under a certificate claiming fourteen.  Post-fix (`scan2_post.txt`) every one of
these five cases reports "every nearer offset now raises".  253 s at load 25.8 (pre) and 21 s
at load 3.7 (post).

### 2c. Single reuses, one request at a time (`repro_pre.txt` head table)

A set built for `offs = ((26,0,0),)` and asked at `(4,4,4)` or `(8,0,0)` is only *slightly*
wrong (0 to 1.2e-15): one far offset still pins `rNc` low enough that the cut happens to hold.
The damage is concentrated in the cases where the built offsets certify nothing near the
request --- `offs = ()`, or a far block --- which is why the scans above are the honest measure.
Post-fix all seven of those reuses raise instead of answering.

## 3. The fix

`xeq/fix.diff` (pre-edit copy `xeq/farfield_pre_eq.jl`).  Three hunks, 15 added lines, in the
file's own style.

1. `FrqSet` gains `rNc::Float64` and `rHi::Float64`, and `farSetup` passes the two radii it
   already computed into the constructor.
2. One line in `farSetup`: `rNc == rHi && isfinite(thr[end]) && (rNc = min(rNc, thr[end]))`.
   When the built offsets certified nothing (`offs = ()`, or every offset off the route-1/2
   ladder) `rNc` is still `rHi`; falling back to the last whole-box threshold keeps the
   interval honest instead of degenerate.
3. A new `ckRad(fs, rLo, rUp, D)` raising unless `rNc*(1-1e-12) <= rLo && rUp <= rHi*(1+1e-12)`,
   called from `farRoute` on the whole-box radius (route 1) and on the extrema of the eight
   octant vertex radii (route 2).

It only raises.  No branch of it can change a returned value, which is what the bit-identity
proof below checks directly.

## 4. Exactly when a caller can hit it

**`offs = nothing` (the default) is safe, and this was measured, not argued.**  With `offs`
unset `farSetup` derives its offsets through `nearOff`, which enumerates *every* lattice offset
with `max(n) >= nMin = 2` inside `rCr = min(thr[lDef/2+1], rHi)`.  `rNc` therefore lands on the
smallest radius the block contains and `rHi = 4*nBlk*r_d` is the block's own ceiling, so the
interval covers the whole block by construction.  `xeq/dflt.jl` sweeps the complete `nBlk^3`
offset set of such a default-built set --- four shapes x two frequencies x `nBlk` in
{16, 128}, **16 785 152 offsets** --- and counts **0 raises**:

| case | nBlk | rNc | rHi | offsets | routes 1/2/3 | raises |
|---|---|---|---|---|---|---|
| c32 f=1 | 128 | 0.05182 | 27.713 | 2 097 144 | 2097132/12/0 | 0 |
| c8 f=1 | 128 | 0.20729 | 110.85 | 2 097 144 | 2097132/12/0 | 0 |
| c4 f=1 | 128 | 0.41458 | 221.70 | 2 097 144 | 2097132/12/0 | 0 |
| sl f=1 | 128 | 0.04070 | 22.650 | 2 097 144 | 2096966/111/67 | 0 |

(the four `f = 1+0.1i` rows and all eight `nBlk = 16` rows are identical in this respect; full
table in `xeq/dflt.txt`).  36 s at load 3.0.  The same holds for `fineSet`, the cross-scale
route-2 entry point, which also calls `farSetup` with `offs = nothing`.

**An explicit `offs` is the exposure.**  Three ways in:

- `offs = ()` --- the empty tuple.  Nothing is certified; pre-fix every request in the block
  got the far-end cut.  This is section 2a and the worst case, 3.2e-8.
- `offs = (far offsets only)` and then a request nearer than the nearest built offset.  This is
  section 2b, the realistic form, up to 9.4e-9.
- a request *above* `rHi = 4*nBlk*r_d`, i.e. outside the block the set was sized for.  Measured
  once, at `(64,0,0)` on `c32` from an `nBlk = 4` set: error 1.689e-16 against a certificate of
  1.076e-18, a 157x violation but at roundoff magnitude.  Real, harmless in size, still
  uncertified --- and this is the one the suite tripped over (section 5).

The user-facing entry points `farTensor` and `farBlock!`, called without an `fs`, build their
own set for their own offsets and cannot hit any of the three.

## 5. The guard fired once in the verification suite, and the caller was wrong

`verify.jl` part (b) raised on `c32 f=1` at `(64,0,0)`.  Cause: part (b) caches one `FrqSet`
per (shape, frequency) but sized it on the **first** offset of the group,
`farSetup(s, f; nBlk = max(4, maximum(abs, D)))` --- so a set built for `nBlk = 4` was reused
at `(64,0,0)` and `(64,64,64)`.  That is D15's third form verbatim, committed by our own
verifier.  Part (d), five lines away, already did it correctly (`nBlk` from the max over the
whole group).

Fixed in the caller, not by loosening the guard: two lines in `partB` build the group's `nBlk`
first.  The two affected rows of `tables/b_volume.txt` moved, and they moved the right way ---
`(64,0,0)` 1.13e-15 -> 1.01e-15, `(64,64,64)` 2.12e-15 -> 1.75e-15.  So the reuse had been
costing accuracy in the shipped table all along, by a factor of about 1.2.

## 6. Proofs

**Bit-identity** (`xeq/bitid_eq.jl`, output `bitid_eq.txt`, whose header still says "D14 fix in place": it was written before the renumbering, and the log is left as run; 340 s at load 3.5-3.7).  Old =
`farfield_pre_eq.jl`, New = the fixed `farfield.jl`, each in its own module with its own empty
table and k-series cache directories:

- (a) `tabWhl(L=10)` / `tabOct(L=7)` at `BigFloat(256)`, `farMomW(80)`, `nCutVec(56)` over four
  radii/caps and four shapes: **0 mismatches**.
- (b) `farTensor` over 30 offsets x four shapes: **120/120 bitwise identical**, with equal
  `Lw`/`Lo` and equal `nCut` on every shape (routes 108/12/0).
- (c) `farBlock!` on two 12^3 blocks (`c32`, `sl`): bitwise identical, 13894 and 13952 nonzero
  entries, **0 differing**.
- (a') a fresh `farShape` build after (b) and (c), from disk: **byte-identical**.

**The verification suite.**  Parts (a)-(i) plus the cross-scale (k) and (l).  Every part exits
0 and the guard fires nowhere:

| part | s | load | result |
|---|---|---|---|
| a | 4.7 | 3.7 | geometry, rational divergence identity: unchanged |
| b | 12.6 | 3.7 | worst volume-form 1.05e-49, farTensor vs ref 4.40e-15 |
| c | 407.5 | 9-19 | Gila today vs 220-bit reference: unchanged |
| d | 93.1 | 9-19 | 427 rows, worst eMx 8.60e-15, pEn 9.34e-15 |
| e | 73.9 | 9-19 | worst 1.59 digits lost (c4, f=1, (64,0,0)) |
| f | 514.6 | 19-25 | k-series band edges and famG overlap: unchanged |
| g | 18.9 | 3.0 | symmetries 0, permutation <= 1.63e-15, homogeneity <= 6.55e-15 |
| h | 449.8 | 3.0 | anti-Hermitian positivity, Gila vs farBlock!: unchanged |
| i | 21.3 | 2.8 | 681-1413 ns/offset |
| k | 51.3 | 3.2 | 66 offsets, worst eMx 5.12e-15, **0 certificate violations** |
| l (t1/t4/t12) | 14.3/13.5/13.1 | 2.2-2.9 | 126 360 offsets, all three thread counts bitwise identical |

The only "n-cut is certified" string anywhere in the logs is in `verify_k.log`, and it is
`farRouteX` --- D13's guard --- being deliberately exercised by part (k) section (8).  The
archived failing run of part (b) is kept as `xeq/verify_b_raised.log`.

**Table accounting** against `xeq/tables_before/` (37 files, snapshotted before any of this):

- 28 files byte-identical.
- `b_volume.txt`: two rows, both explained in section 5, both improvements.
- `e_route.txt`, `h_pos.txt`, `i_cost.txt`, `k_adv.txt`, `l_route.txt`: timing and load-average
  columns only.  Every error, certificate, route, `L`, count and histogram column is
  byte-identical.  (`k_adv.txt`'s changed column is `t s`, farTensorX seconds.)
- `l_thread_t1.txt`, `l_thread_t4.txt`: timings, plus extra `vs l_blk_r4_t*.bin: bitwise
  identical` lines.  Those appear because the other thread counts' dumps already existed on
  disk this time; in the baseline t1 ran first and had nothing to compare against.  The added
  comparisons pass.
- `tables/l_blk_r4_t{1,4,12}.bin` are not "new": the baseline snapshot copied `.txt` only.

## 7. Residual risk

- **The interval is certified at its endpoints, not throughout.**  `nCut` is the max of
  `nCutVec` at `k*rNc` and `k*rHi`; nothing proves it is adequate strictly inside.  That was
  equally true before the fix, and the evidence that it is adequate is empirical: parts (d),
  (f) and (j) over 427 + 438 references.  The guard makes the claim checkable, it does not make
  it a theorem.
- **The `thr[end]` fallback is a heuristic.**  When the built offsets certify nothing, `rNc`
  falls back to the last whole-box threshold.  That is a defensible floor, not a proof, and it
  is why an `offs = ()` set now *raises* on a request below it (seen once in `repro_post.txt`)
  rather than answering.  Raising is the intended behaviour there: pre-fix that same request
  was wrong by 4.357e-09 under a certificate of 1.766e-14.
- **The octant side is belt and braces.**  `repro_pre.txt` searched for (set, request) pairs
  whose request vertex radius is below the set's.  The cubes have none; the slender shape has
  136, and every one of them already raises on the pre-existing `maximum(Lc) <= fs.Lo` guard.
  So no octant-only escape was measurable, and `ckRad` on the vertex radii is untested against
  a real failure.
- **The 1e-12 relative slack** in `ckRad` absorbs the difference between the radius `farSetup`
  computed and the one `farRoute` recomputes.  It is many orders below any `nCut` step, but it
  is a tolerance and not an exact comparison.
- **Only `farRoute` is guarded.**  A caller reaching `tnsWhl!` / `tnsOct!` / `ksrCached!`
  directly with a hand-made `L` bypasses the check.  No shipped path does.
- Fixed shapes and frequencies: everything here is the four shipped shapes at `f = 1` and
  `1+0.1i`.  The mechanism is shape-independent but the magnitudes are not.

## DONE
