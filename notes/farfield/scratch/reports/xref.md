# xref: 220-bit cross-scale reference tensors (item 6, reference half)

Dir: `SCRATCH/xwork/xref/`.  Library: `refx.jl` (include it: `xRead()`, `xKey(R, sT, sS, f)`, `xPut`, `xParseKey`,
`refA`, `refB`, `xMatrix`).  Cache: `SCRATCH/xwork/xref/reftensors_x.txt`, append-only under the mkdir lock
`reftensors_x.txt.lock`; record

    vol|R=r1,r2,r3|sT=t1,t2,t3|sS=s1,s2,s3|f=re;im|p=220|v1 v2 ... v9      (v = re;im, column major, BigFloat strings)

R = target centre, sT = target edges, sS = source edges (source at the origin), exact rationals `n//d`.
Values are rule A (36 pairKer face pairs / V_t with the srfSum! signs) unless a section below says otherwise.
Workers: `run.jl <shard> <n> <ordN>` (sentinels `done_run<shard>`, logs `log_run<shard>.txt`).

## 1. Sanity of rule A (`refA`) on equal cells

`t1.jl`: equal cells g = (1/32)^3, D = (5,1,0), f = 1, ordN 44: `refA((5g, g, 0), g, g, 1)` reproduces the round-1
record `pairs|D=5,1,0|...|n=44` of `notes/farfield/refcache/reftensors.txt` to 0.0 (identical values: same panels,
same pairKer, same srfSum! signs), 30.6 s at load 9.6 (the root measured 13.7 s unloaded).
Geometry matrix `xMatrix()`: 328 records (84 x-facing lateral (0,0) at f = 1; 56 corner + 28 corner-corner laterals;
80 diagonal; 80 at f = 1+0.1i and 0.37 on {x} and {x,y,z}), plus 8 slender-pair records.

## 2. Cost of the two rules (measured, load 10-11 throughout; other agents' processes running)

Rule A (`refA`, 36 x pairKer at ordN 44, 220 bits): equal cells 30.6 s; ratio 2 {x} kap 1: 152.9 s (load 11.2);
ratio 2 {x,y,z} kap 1: 122.4 s at ordN 32, 339.7 s at ordN 44 (log_ordchk.txt).  Cross-scale face pairs cost 5-11x
the equal-cell pairs: the convolution weight of a g interval with an r g interval has 3 pieces (2 after the split at 0
for the triangle), so a face pair has up to 16 graded 2D boxes instead of 4, and the coarse panel's long side needs
log2(r g / dmn) doubling panels.
Rule B (`refB`, graded tensor GL with the trapezoid weight, 220 bits): 16.5 us per point; ratio 2 {x} kap 1 at
ord 32: 393k points, 6.4-7.4 s; the point counts of the whole matrix at ord 32 range from 0.4 M ({x}, kap >= 2) to
20.2 M (ratio 16 {x,y,z} kap 1, ~330 s).  Rule B is therefore 10-20x cheaper than rule A on these geometries and is
the production rule of the cache (`runB.jl`, ord 32); rule A is the independent check on a subset (section 4).
`rule_x.txt` records which rule wrote each key ("B32", "A44", "A64").  Records without an entry in rule_x.txt were
written by xgeom's `xwork/xgeom/ref.jl`, which includes this refx.jl and calls `refB(...; ord = 32, grd = 2)` then
`xPut` (same code, same lock), so they are rule B ord 32 as well.

Swap identity (rule A, ratio 2 {x} kap 1, f = 1): G(R; g, sS) vs (V_s/V_t) G(-R; sS, g): max-entry-normalized
7.0e-65, worst per entry 8.4e-65 (the 220-bit floor is 1.3e-64).

## 3. Cache contents vs the brief's matrix (inv.jl -> out_inv2.txt; gaps.jl = the brief's full matrix `xBrief()` and the fill list `xGaps()`)

Brief's matrix as enumerated (`xBrief()`, 416 keys): f = 1, 12 coarse shapes (ratios 2, 4, 8, 16 on {x}, {x,y}, {x,y,z}), x-facing
laterals (0,0) / corner / corner-corner (the laterals collapse when the axis is not coarsened: {x} has only (0,0), {x,y} has
(0,0) and corner), kap in {1, 2, 3, 4, 6, 8, 16, 32}; diagonal family kap in {1, 2, 4, 8, 16} with lat_z in {0, (r_z-1)/2 g}
({x,y,z} only has both); f = 1+0.1i and 0.37 on {x} and {x,y,z}, x-facing (0,0), all 8 kap; the two slender pairs
(target (1/32,1/32,1/512) vs source (1/32,1/32,1/64) and vs (1/8,1/8,1/512)), offsets along z (the slender axis, gap
kap fine z-cells 1/512) and along x (gap kap x-cells 1/32), kap in {1, 2, 4, 8}.  The f != 1 laterals and diagonals are
read as NOT part of the brief ("the others on the {x} and {x,y,z} sets" of the x-facing (0,0) family).

Before this resume (cache at 22:16, 359 records, 356 keys): 334 of the 416 present.  Missing then, ALL filled now:
| gap | entries | records | rule-B time (both runG shards, load 3.9-7.6) |
|---|---|---|---|
| (a) f = 1, (0,0) and corner-corner, kap {1,2,4,8,32} | 0 missing (runB + xgeom had them) | 0 | - |
| (b) diagonal kap {1,4,16}, ratios 2, 16, {x}, {x,y,z} | 0 missing | 0 | - |
| (c) slender pairs, kap {1,2,8}, along z and along x | 12 | 12 | 1060 s (z-facing records 108-292 s each: 4.0-23.7 Mpts) |
| (d) f = 1+0.1i, 0.37, ratios 4, 8, {x,y,z}, kap 1, 8 | 0 missing | 0 | - |
| (e) kap 6, lateral (0,0), f = 1 | 6 of 12 (xgeom had the {x} and {x,y,z} ones) | 6 | 145 s |
| (f) slender pairs kap 4 | 4 | 4 | 290 s |
| (g) kap 6, corner / corner-corner laterals, f = 1 | 12 | 12 | 251 s |
| (h) f != 1, kap {3, 6, 16}, {x} and {x,y,z} | 48 | 48 | 1340 s |
Total: 82 records, 3086 s of rule B in 32.3 min wall on two processes (00:00:34 -> 00:32:52).  Cache now: 441 records,
438 distinct keys (3 keys were written twice by runB and xgeom racing, identical values: err 0.0), 416/416 brief keys
present, 22 keys outside the brief (xgeom's further axial offsets, listed in log_sln.txt: "extra").  Matrix entries NOT
computed: none of the 416.  Per (ratio, axis set, f), every family is complete:
| ratio | axes | f = 1: (0,0) kap / corner kap / corner-corner kap / diag kap / diagZ kap | f = 1+0.1i (0,0) kap | f = 0.37 (0,0) kap |
|---|---|---|---|---|
| 2, 4, 8, 16 | {x} | all 8 / - / - / all 5 / - | all 8 | all 8 |
| 2, 4, 8, 16 | {x,y} | all 8 / all 8 / - / all 5 / - | - (not in brief) | - |
| 2, 4, 8, 16 | {x,y,z} | all 8 / all 8 / all 8 / all 5 / all 5 | all 8 | all 8 |
| slender 1: (1/32,1/32,1/512) vs (1/32,1/32,1/64) | z-facing / x-facing | kap 1,2,4,8 / 1,2,4,8 | - | - |
| slender 2: (1/32,1/32,1/512) vs (1/8,1/8,1/512) | z-facing / x-facing | kap 1,2,4,8 / 1,2,4,8 | - | - |
Who wrote what (rule_x.txt; keys absent from it are xgeom's): 373 records by runB/runG (rule B ord 32, "B32"); 68 by xgeom
(66 rule B ord 32 via the same refB, and TWO rule A ordN 44: ratio 2 {x} kap 1 `R=5//64 sS=1//16,1//32,1//32` and ratio 16 {x}
kap 1 `R=19//64 sS=1//2,1//32,1//32`, per xgeom's ref_log.txt "t44=" lines; this corrects Section 2's "xgeom's records
are all rule B").  No rule-A ordN 64 tensor is in the cache (ordchk's four keys were already present when it finished).

## 4. Rule A vs rule B (production: rule B ord 32 for the cache; rule A ordN 44 as the independent check)

Rule A's own ordN convergence (ordchk.jl -> out_ordchk.txt; {x,y,z} cubes, x-facing lateral (0,0), kap 1, f = 1; the
36 face-pair values compared directly, then the assembled tensor; times at the load shown):
| ratio | ordN 32 vs 64: pairs / tensor max / per entry | ordN 44 vs 64: pairs / tensor max / per entry | t32 / t44 / t64 (s) (load) |
|---|---|---|---|
| 2  | 2.9e-61 / 8.6e-61 / 2.2e-60 | 7.6e-64 / 8.7e-64 / 1.6e-63 | 122 / 340 / 1083 (10.4-11.1) |
| 4  | 1.3e-59 / 4.1e-61 / 1.2e-60 | 1.1e-63 / 2.7e-63 / 1.3e-63 | 242 / 552 / 1492 (11.4-3.4) |
| 8  | 2.8e-55 / 3.1e-57 / 7.8e-57 | 3.0e-63 / 6.7e-63 / 1.4e-63 | 177 / 450 / 1373 (3.7-2.5) |
| 16 | 1.5e-53 / 6.8e-53 / 8.0e-53 | 3.3e-63 / 2.2e-63 / 6.5e-64 | 233 / 604 / 1854 (2.6) |
xgeom's ordN check on ratio 2 {x} kap 1: |A44 - A64| = 1.6e-64 / 2.8e-64 (t64 = 446 s at load 10.6).  So rule A at
ordN 44 is converged to 1-7e-63 on every shape (ordN 32 is not: 1e-53 at ratio 16), and ordN 44 is the production
order of rule A.  (The ordchk A64 tensors were not cached: runB had already written B32 records for all four keys.)

Rule A (ordN 44) against the cache's rule B (ord 32), all at f = 1, lateral (0,0):
| shape, kap | who | (max-entry-normalized, worst per entry) | rule-A cost |
|---|---|---|---|
| ratio 2 {x}, kap 1     | xgeom (record IS the A44 value; B32 recomputed) | 1.9e-58, 2.3e-58 | 122 s (load 5.9) |
| ratio 4 {x}, kap 1     | xgeom | 1.6e-58, 2.0e-58 | 200 s (load 11.1) |
| ratio 8 {x,y,z}, kap 1 | NEW (achk.jl 8:3:1:0 44 -> log_achk.txt) | 3.0e-57, 3.9e-57 | 482 s (load 5.1) |
| equal cells D = (2,0,0) (gap 1 g) | gcd.jl, B32 vs round-1 A44 | 2.4e-58, 2.8e-58 | - |
| equal cells D = (3,0,0) | gcd.jl | 3.2e-59, 4.3e-59 | - |
| equal cells D = (4,0,0) | gcd.jl | 1.2e-63, 1.4e-63 | - |
Rule B at higher order against the same rule-A values (chk48.jl -> log_chk48.txt, load 3.5-3.8):
| offset | B32 vs A44 | B40 vs A44 | B48 vs A44 | points B32 / B40 / B48 | time B32 / B40 / B48 |
|---|---|---|---|---|---|
| ratio 2 {x} kap 1 (record = xgeom A44) | 1.9e-58, 2.3e-58 | 1.4e-64, 3.3e-64 | 4.2e-64, 4.2e-64 | 1.31 / 2.56 / 4.42 M | 16.5 / 30.6 / 52.4 s |
| equal cells D = (2,0,0) (round-1 A44) | 2.4e-58, 2.8e-58 | - | 1.4e-64, 3.1e-64 | 1.18 / - / 3.98 M | 13.9 / - / 47.6 s |
So ord 40 already puts rule B at the 220-bit floor on the one-cell-gap offsets (1.4e-64 vs rule A), at 1.95x the ord-32
cost; the 1.9e-58 at ord 32 is pure Gauss-Legendre truncation on the two near panels of width g/2 at distance g.
Rule B's own convergence (runB / runG "self B24 vs B32" on the first record of each shape; xgeom's B32 vs B40):
| shape | offset | B24 vs B32 | B32 vs B40 (xgeom) |
|---|---|---|---|
| ratio 2 {x}        | kap 2 (R = 9/64)  | 5.1e-52 | - |
| ratio 2 {x,y}      | kap 1 | 6.1e-44 | - |
| ratio 2 {x,y,z}    | kap 8 (R = 19/64) | 2.1e-64 | kap 1: 2.9e-61 |
| ratio 4 {x} / {x,y} / {x,y,z} | kap 1 | 1.4e-43 / 4.4e-44 / 8.7e-45 | {x,y,z} kap 1: 2.4e-59 |
| ratio 8 {x} / {x,y} / {x,y,z} | kap 1 | 1.3e-43 / 5.4e-43 / 2.8e-43 | - |
| ratio 16 {x} (kap 8) / {x,y} (kap 1) / {x,y,z} (kap 32) | | 1.0e-46 / 1.3e-39 / 1.2e-50 | {x,y,z} kap 1: 6.8e-53 |
| kap 6 records (runG): ratio 2 {x,y}, 4 {x}, 4 {x,y,z}, 16 {x,y} corner, 16 {x,y,z} corner-corner | | 1.4e-64, 3.1e-47, 1.3e-44, 1.3e-41, 5.5e-42 | |
| slender pair 1 x-facing kap 1 / pair 2 x-facing kap 1 | | 1.4e-43 / 1.5e-44 | |
| slender pair 1 z-facing kap 1 (724 panels) / pair 2 z-facing kap 1 (608 panels) | | 4.4e-38 / 3.1e-37 (chk48.jl, 10.0 / 8.4 Mpts at ord 24, 122 / 104 s) | pair 1: B32 vs B40 = 3.0e-50 (slnB40.jl, 46.3 Mpts, 559 s, load 4.6) |
Slender pair 1 z-facing kap 1 is the one record where rule B ord 32 was checked against BOTH a higher order and rule A
(slnA44.jl: pairKer ordN 44, 2027 s at load 3.9): |A44 - B32| = 3.0043777117e-50 / 3.0743152743e-50 and |B40 - B32| =
3.0043777117e-50 / 3.0743152743e-50, the same to 11 digits, so A44 and B40 agree to ~1e-61 there and the 3.0e-50 is
rule B's ord-32 error on that record (target 1/512 thick, singularity 1/512 from the source face: the graded panels
start at dmn/2 = 1/1024 and need 6 doublings to cross the 1/32 transverse extent).
Agreement floor.  Rule A ordN 44 and rule B ord 32 agree to 1.6-1.9e-58 at ratio 2-4 and 3e-57 at ratio 8 on the
NEAREST offsets (kap 1, face gap one fine cell), and to 1-2e-63 (the 220-bit floor of the two rules is 1-4e-64) from
a gap of two cells on; the discrepancy at kap 1 is rule B's (its B32-vs-B40 differences reproduce it: 2.9e-61 at r = 2,
2.4e-59 at r = 4, 6.8e-53 at r = 16 {x,y,z}, while A44 sits 1e-63 from A64).  The B24-vs-B32 differences (1e-39 to
1e-43 at kap 1) extrapolate, with the observed ~10^-4 per 8 orders at the near panels, to a rule-B ord-32 error of
1e-47 to 1e-52 in the worst cases (ratio 16 {x,y}/{x,y,z} kap 1), which is what the direct comparisons show (6.8e-53).
The identical numbers 6.750e-53 for A32-vs-A64 and B32-vs-B40 at ratio 16 {x,y,z} kap 1 are the same 32-point
Gauss-Legendre error on the same outermost x panel (both rules grade by factor 2 from the singularity, so both end
with one panel [1/4, 1/2] lambda along x carrying the same e^{ikr} oscillation).  Cache accuracy therefore: every
record is good to at least 52 digits on the 12 coarse shapes (worst: ratio 16 {x,y,z} kap 1, 6.8e-53), most to 58-63;
the slender z-facing kap-1 records are the least accurate of the cache (pair 1 measured 3.0e-50; pair 2, whose B24-B32
difference is 7x larger, is estimated 1e-49 to 1e-48 and was not measured directly); the brief's 1e-40 bar between the
two rules is met everywhere with 9-23 digits to spare.  A future upgrade of the 36 kap-1 records to ord 40 or 48
costs (40/32)^3 - (48/32)^3 = 2-3.4x their B32 time (see the chk48 line for what ord 48 buys at ratio 2).

## 5. The gcd identity (gcd.jl -> log_gcd.txt; load 3.9-7.2)

Ratio 2 on x: coarse source sS = (1/16, 1/32, 1/32) at the origin, fine target g at R = (3g/2 + kap g, 0, 0), so the two
source sub-cells sit at c' = -+g/2 and the identity is a SUM (N_t = 1): G(R; g, sS) = G((kap+1) g; g, g) + G((kap+2) g; g, g).
The cross-scale tensor is the cache record; the equal-cell pieces come from (i) the round-1 refcache
(`pairs|D=n,0,0|...|n=44`, rule A ordN 44) and (ii) rule B ord 32.  eps(220 bits) = 1.2e-66; the practical floor
of a 9-entry sum at 220 bits has been 1-4e-64 in every check.  Errors are (max-entry-normalized, worst per entry).

| kap | cross-scale record (rule) | sum of A44 pieces vs record | sum of B32 pieces vs record | B32 piece vs A44 piece: D=kap+1 / D=kap+2 |
|---|---|---|---|---|
| 1 | vol R=5//64 sS=1//16,1//32,1//32 (xgeom, **A44**; xgeom's A44-A64 = 1.6e-64) | 3.2e-64 / 3.8e-64 | 1.9e-58 / 2.3e-58 | D=2: 2.4e-58 / 2.8e-58; D=3: 3.2e-59 / 4.3e-59 |
| 2 | vol R=7//64 sS=1//16,1//32,1//32 (xgeom, B32) | 2.2e-59 / 3.0e-59 | 1.7e-64 / 1.8e-64 | D=3: 3.2e-59 / 4.3e-59; D=4: 1.2e-63 / 1.4e-63 |

Reading: the identity holds to the 220-bit floor whenever the pieces and the whole are the SAME rule (kap 1 with A44
pieces against the A44 record; kap 2 with B32 pieces against the B32 record); the off-floor residuals are exactly the
rule-B ord-32 quadrature error at the nearest offsets (1.9e-58 at kap 1 = xgeom's own |A44-B32| = 1.913e-58 on that
record; 3.2e-59 at D = 3), not a failure of the identity.  In rule B the panel breakpoints of the two equal-cell pieces
(cuts at 0 and -+g) and of the trapezoid (cuts at -+a = -+g/2, -+b = -+3g/2, shifted by g/2) coincide, so the B-sum and the
B-whole share their node sets and agree to rounding by construction; the A-sum against the A-whole telescopes the
interior face pairs.  The cross-rule comparisons are the informative ones: rule B ord 32 carries ~2e-58 at a one-cell
gap and 1e-63 at a two-cell gap (D = 4); rule A ordN 44 is at the floor at both (1.6e-64 vs ordN 64 at kap 1).

Swap identity G(-R; sS, g) = (V_s / V_t) G(R; g, sS), rule B ord 32 for the swapped (coarse-target) tensor:
| offset | (V_s/V_t) G_swap vs record | swap vs (G1+G2)/2 (average form, coarse target) | B32 points / time |
|---|---|---|---|
| {x} ratio 2 kap 1 | 1.9e-58 / 2.3e-58 (record is A44: this is again rule B's ord-32 error) | 7.6e-65 / 1.8e-64 | 1.31 M / 15.9 s |
| {x} ratio 2 kap 2 | 1.9e-65 / 5.2e-65 | 1.7e-64 / 1.7e-64 | 0.39 M / 4.9 s |
| {x,y,z} ratio 2 kap 1, sS = (1/16)^3 | 1.4e-64 / 3.6e-64 | (8-piece SUM identity, B32 pieces: 1.1e-64 / 1.1e-64, 93.5 s) | 3.41 M / 40.6 s |
(Section 2 had the swap identity with rule A on {x} kap 1: 7.0e-65 / 8.4e-65.)

## 6. Cost (rule B ord 32, 220 bits, JULIA_NUM_THREADS=1; 16.5 us per point at load 2-5, ~20 us at load 10-11)

Record count and time per (ratio, axis set, f); "s/record" is wall time of refB per record from the logs (runB 19:34-22:16
at load 10 falling to 1.2; runG 00:00-00:33 at load 3.9-7.6; xgeom's from its ref_log.txt).
| ratio | axes | f | records (runB+runG / xgeom) | s/record min / mean / max | mean Mpts | total s |
|---|---|---|---|---|---|---|
| 2 | {x} | 1 | 9 / 9 | 5.1 / 6.5 / 7.4 | 0.39 | 43+8+71 = 122 (one xgeom record is A44: 122 s) |
| 2 | {x,y} | 1 | 21 / 0 | 7.3 / 11.7 / 33.9 | 0.82 | 244 |
| 2 | {x,y,z} | 1 | 25 / 12 | 11.2 / 16.4 / 41.4 | 1.20 | 410 + 345 |
| 4 | {x} | 1 | 10 / 8 | 5.1 / 8.8 / 25.7 | 0.57 | 88 + 85 |
| 4 | {x,y} | 1 | 21 / 0 | 7.5 / 16.7 / 67.5 | 1.16 | 351 |
| 4 | {x,y,z} | 1 | 30 / 10 | 11.6 / 29.8 / 145.1 | 2.08 | 894 + 550 |
| 8 | {x} | 1 | 12 / 0 | 5.1 / 10.8 / 28.7 | 0.71 | 130 |
| 8 | {x,y} | 1 | 21 / 0 | 7.3 / 22.9 / 74.5 | 1.56 | 480 |
| 8 | {x,y,z} | 1 | 33 / 0 | 11.4 / 51.2 / 183.4 | 3.76 | 1688 |
| 16 | {x} | 1 | 5 / 11 | 5.5 / 7.5 / 9.1 | 0.55 | 38 + 126 (one xgeom record is A44: 243 s) |
| 16 | {x,y} | 1 | 21 / 0 | 7.7 / 31.0 / 84.3 | 2.32 | 650 |
| 16 | {x,y,z} | 1 | 23 / 16 | 11.6 / 66.4 / 194.3 | 5.09 | 1527 + 1817 |
| 2 / 4 / 8 / 16 | {x} | 1+0.1i | 8 each | 5.5-5.9 / 7.4-12.3 / 19.1-26.9 | 0.51-0.84 | 59 / 73 / 78 / 97 |
| 2 / 4 / 8 / 16 | {x,y,z} | 1+0.1i | 8 each | 22.2-23.0 / 26.4-64.3 / 50.1-294.7 | 1.80-9.10 | 211 / 391 / 512 / 1069 |
| 2 / 4 / 8 / 16 | {x} | 0.37 | 8 each | 4.8-5.1 / 6.4-10.5 / 16.0-22.7 | 0.51-0.84 | 51 / 62 / 66 / 84 |
| 2 / 4 / 8 / 16 | {x,y,z} | 0.37 | 8 each | 19.2-20.2 / 22.5-54.7 / 42.0-249.6 | 1.80-9.10 | 180 / 336 / 438 / 915 |
| slender 1 (1/32,1/32,1/64) | z / x | 1 | 4 / 4 | z: 67-292 (kap 8 -> 1), x: 6.4-16.8 | 7.23 | 716 |
| slender 2 (1/8,1/8,1/512) | z / x | 1 | 4 / 4 | z: 108-231, x: 10.3-34.9 | 6.44 | 634 |
(f != 1 rows: kap {1,2,4,8,32} by runB, kap {3,6,16} by runG; the complex-f rows cost 1.15-1.2x the f = 1 rows per point,
exp(ik r) with complex k.)  The per-record cost is set by the number of graded sub-panels (dmn-graded, factor 2) times
32^3 points: 12 panels (0.39 Mpts, 5-7 s) for {x} at kap >= 2 up to 616 panels (20.2 Mpts, 250-330 s) for ratio 16
{x,y,z} kap 1 and 724 panels (23.7 Mpts, 292 s) for slender-1 z-facing kap 1.

Machine time of this agent, all phases (wall x processes):
| phase | scripts | wall | processes | rule-B / rule-A seconds |
|---|---|---|---|---|
| before the resume (19:14-22:16) | t0/t1, run.jl (abandoned), ordchk, runB | 3.0 h | 1-2 | runB 9433 s B32 (291 records); ordchk 8519 s A32/A44/A64; t1 ~400 s A44 |
| resume: inventory | inv.jl x2 | 2 x ~2 min | 1 | - |
| gap fill | runG.jl x2 (00:00-00:33) | 32.3 min | 2 | 3086 s B32 (82 records) + 8 self checks B24 |
| identities / checks | gcd.jl (3.0 min), achk 8:3:1:0 44 (8.3 min), chk48.jl (6.5 min) | 18 min | on the second slot | gcd 179 s B32; achk 482 s A44; chk48 387 s B32/B40/B48/B24 |
| slender convergence | slnB40.jl, slnA44.jl | 33.9 min (00:41:18 -> 01:15:10) | 2 | B40 559 s (46.3 Mpts); A44 2027 s (pairKer on the slender pair, 17x its cost on a cubic ratio-2 pair) |
Load averages: 10-11 during the pre-resume runB (other agents' processes), 1.2-2.3 at its end, 3.9-7.6 during runG, 3.1-3.8
during chk48.  xgeom's 68 records: 2993 s of B32 plus 2 x A44 (122 + 243 s) plus its checks.

Where I stopped and why: the brief's matrix is complete (416/416 keys) and every item (a)-(d) of the resume was already
present or has been filled, plus the kap-6 family, the slender kap 4 and the f != 1 kap {3,6,16} entries; the resume's
machine time is 75 min wall on two processes (00:00 -> 01:15: 32 min gap fill, 18 min identities and order checks, 34 min
slender convergence), i.e. about the 90-minute budget with the slender A44 run (2027 s) as the last job.  Remaining, not
done: (1) direct measurement of rule B's ord-32 error on the slender pair-2 z-facing kap 1 record (B40 would cost ~500 s,
A44 ~35 min; estimated 1e-49 to 1e-48 from its B24-B32 difference of 3.1e-37); (2) an ord-40 upgrade of the 38 kap-1
records if anyone needs more than 50-58 digits there (2x their B32 time, ~1.5 h of one process; ord 40 reaches the
220-bit floor at ratio 2); (3) f != 1 laterals and diagonals, which I read as outside the brief.  Scripts of this resume:
gaps.jl (xBrief, xGaps), inv.jl (-> out_inv2.txt), runG.jl (-> log_runG_97323.txt, log_runG_98222.txt), gcd.jl
(-> log_gcd.txt), achk.jl 8:3:1:0 44 (-> log_achk.txt), chk48.jl (-> log_chk48.txt), slnB40.jl / slnA44.jl (-> log_sln.txt).
