# xverify: adversarial verification of the integrated cross-scale library (farfield.jl, 2009 lines)

Dir: `SCRATCH/xwork/xverify/`.  Deliverable: `notes/farfield/verify.jl` parts (j), (k), (l), (m) (appended; parts
(a)-(i) byte-identical, md5 of their body 163e37cede2587a91c6b4dc25adbeb8c before and after), writing
`notes/farfield/tables/{j_xref,k_adv,l_route,l_thread_t<n>,m_gila,m_block_t<n>}.txt`.  Scripts here: `probe.jl`
(first look, `probe.txt`), `refs_k.jl` (the new 220-bit rule-B references, appended to `xwork/xref/reftensors_x.txt`
under its lock, log `refs_k.txt`), `farfield_gcd.jl` + `gcdexp.jl` (a patched COPY with a rational gcd cell, for
the non-integer-ratio question; farfield.jl itself untouched), `verify_*.log`.  Run: `JULIA_NUM_THREADS=1 julia
--startup-file=no --project=$SCRATCH/env notes/farfield/verify.jl <part>` from `notes/farfield`; the shipped table dir
`notes/farfield/shapetab/` receives the pair tables, `notes/farfield/ksrcache/` the route-3 `ksrx_*` files.
Errors: eMx = max_ab |G - Gr| / max |Gr|, pEn = worst per entry over entries above 1e-8 max|Gr|, digits lost =
log10(pEn / eps).  Every timing states the 1-min load average (the user's `simple_shapes.jl` ran at 6-12 cores
through most of this session).

## 0. What was read and the plan
BRIEF_cross.md, PROMPT_crossscale.md, REGISTRY.md (cross-scale entries), xunify.md (all), xtable.md 0/3b-3e/5b/6,
xgeom.md 0/2b/5, xref.md 3-5, xtheory.md 1, farfield.jl 1537-2009 and the equal-cell farSetup/farRoute/farTensor/
farBlock!, verify.jl, xunify's acc.jl/block.jl, xgeom's acc.jl/offs.jl/enum.jl, refx.jl.  Code review notes (before
running anything) are in Section 7 together with what the runs showed.

## 1. Part (j): every cached reference, independently (`verify.jl j` -> `tables/j_xref.txt`, 12:45-17:37, 17582.7 s single-threaded, 1-min load 5.4 at the start and 3.9 at the end; log `xwork/xverify/verify_j.log`; my own script, not xunify's acc.jl)

439 records (the cross-scale cache as it stood when the run started: xref's 438 plus the r3x record refs_k had already
appended).  Every record is evaluated four ways -- farBlockX! over the group's offsets with cert = true, farTensorX per
offset (bitwise compared with the block, certificate compared too), the swapped orientation (coarse target at -R,
G(R; sT, sS) =? (V_s/V_t) G(-R; sS, sT)) both ways -- plus a check xunify did not make: the reflection identity
G_ab(sigma R) = sigma_a sigma_b G_ab(R) over the three axis flips and -R (column flip), which exercises the sign logic
of tnsGcdX!/boxSumX! and the route selection at mirrored offsets.

Per route over all 439 records:
| route | n | eMx median / worst | pEn median / worst | digits lost | cert/max\|G\| min / median / max | cert / actual error min / median / max | swap identity worst | flip worst | violations |
|---|---|---|---|---|---|---|---|---|---|
| 1 whole trapezoid box | 244 | 4.89e-16 / 2.44e-15 | 6.93e-16 / 4.71e-15 | 1.3 | 4.55e-15 / 1.28e-14 / 3.50e-14 | 2.4 / 23.9 / 135.3 | 0 (exactly) | 0 (exactly) | 0 |
| 2 gcd box sum | 92 | 5.86e-16 / 3.42e-15 | 7.94e-16 / 4.66e-15 | 1.3 | 4.95e-15 / 1.66e-14 / 3.17e-14 | 4.0 / 29.2 / 673.4 | 4.88e-15 | 5.17e-15 | 0 |
| 3 k-series | 103 | 4.19e-17 / 9.67e-17 | 5.97e-17 / 9.67e-17 | -0.4 | 4.62e-15 / 4.76e-15 / 8.05e-13 | 48.7 / 115.2 / 8930.4 | 2.64e-36 | 5.28e-36 | 0 |
Worst overall: eMx 3.42e-15 at 16x16x16, f = 1, R/g = (12.5, 12.5, 7.5) (route 2, 4096 pieces; certificate 4.0x the
error, the tightest of the run); pEn 4.71e-15 at 2x2x2, f = 1, (5.5, 0.5, 0.5) (route 1, L 34) = 1.3 digits lost.
0 certificate violations in 878 evaluations (439 direct + 439 swapped, each compared against its own reference).
farTensorX is bitwise equal to farBlockX! on 439 of 439 and returns the same certificate on 439 of 439.  The loosest
certificate, 8.05e-13 max|G| (8930x the actual error), is the slender 4x4x1 pair's thin-axis record (0, 0, 2) g_z --
xunify's four loose records, reproduced.

Agreement with xunify Section C: identical to every digit I can compare -- route-1/2/3 counts 244/92/103 (xunify
244/91/103: my run has the one extra r3x record, a route-2 offset), route-1 worst 2.44e-15, route-2 worst 3.42e-15
(pEn 4.66e-15 vs xunify's 4.70e-15 for route 2 -- the same record, (9.5, 7.5, 7.5)), route-3 worst 9.67e-17, digits lost
1.3, 0 violations, swap identity exactly 0 on every route-1 record and <= 4.88e-15 on route 2 (xunify 4.9e-15), 1e-36 on
route 3, and the same cert/error ranges (2.4 / 23.9 / 135.3 route 1; 4.0 / 29.2 / 673.4 route 2; 48.7 / 115.2 / 8930.4
route 3).  So xunify's C is INDEPENDENTLY CONFIRMED, from my own script, my own reference reader and my own error
metrics.  The reflection identity, new here, is 0 exactly on every route-1 record (the table is even in each axis and
the offset enters only through the harmonics' signs), <= 5.17e-15 on route 2 (the box sum visits the mirrored pieces in
a different order) and 1e-36 on route 3.

Per (pair, f) -- n, routes, worst eMx / pEn / digits, worst swap identity (sid), swapped-orientation error vs the
reference (seMx), worst reflection deviation (flip), worst certificate over max|G|, violations:

| pair | f | n | r1/2/3 | eMx | pEn | dig | sid | seMx | flip | c/G | viol |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2x1x1;1x1x1 | 0.37 | 8 | 7/1/0 | 2.67e-15 | 2.67e-15 | 1.1 | 0 | 2.67e-15 | 0 | 1.86e-14 | 0 |
| 4x1x1;1x1x1 | 0.37 | 8 | 6/2/0 | 2.27e-15 | 2.27e-15 | 1.0 | 5.02e-19 | 2.27e-15 | 5.02e-19 | 1.85e-14 | 0 |
| 2x2x2;1x1x1 | 0.37 | 8 | 6/0/2 | 9.92e-16 | 1.30e-15 | 0.8 | 0 | 9.92e-16 | 0 | 2.28e-14 | 0 |
| 8x1x1;1x1x1 | 0.37 | 8 | 5/3/0 | 2.13e-15 | 2.13e-15 | 1.0 | 3.15e-16 | 2.01e-15 | 3.15e-16 | 2.92e-14 | 0 |
| 16x1x1;1x1x1 | 0.37 | 8 | 3/5/0 | 2.44e-15 | 2.44e-15 | 1.0 | 2.98e-16 | 2.44e-15 | 2.98e-16 | 1.78e-14 | 0 |
| 4x4x4;1x1x1 | 0.37 | 8 | 4/0/4 | 4.46e-16 | 8.78e-16 | 0.6 | 4.58e-38 | 4.46e-16 | 8.09e-38 | 2.62e-14 | 0 |
| 8x8x8;1x1x1 | 0.37 | 8 | 3/0/5 | 1.49e-15 | 1.65e-15 | 0.9 | 8.04e-38 | 1.49e-15 | 1.47e-37 | 1.71e-14 | 0 |
| 16x16x16;1x1x1 | 0.37 | 8 | 2/0/6 | 4.40e-16 | 6.36e-16 | 0.5 | 8.22e-38 | 4.40e-16 | 1.51e-37 | 2.27e-14 | 0 |
| 1x1x8;1x1x1 | 1.0 | 8 | 3/4/1 | 9.97e-16 | 9.97e-16 | 0.7 | 2.11e-16 | 9.97e-16 | 2.11e-16 | 1.39e-14 | 0 |
| 4x4x1;1x1x1 | 1.0 | 8 | 2/0/6 | 4.09e-16 | 6.29e-16 | 0.5 | 3.98e-37 | 4.09e-16 | 7.96e-37 | 8.05e-13 | 0 |
| 2x1x1;1x1x1 | 1.0 | 17 | 16/1/0 | 3.24e-15 | 3.24e-15 | 1.2 | 4.79e-21 | 3.24e-15 | 4.79e-21 | 3.49e-14 | 0 |
| 3x1x1;1x1x1 | 1.0 | 1 | 0/1/0 | 2.98e-15 | 2.98e-15 | 1.1 | 5.69e-17 | 2.98e-15 | 5.69e-17 | 2.05e-14 | 0 |
| 2x2x1;1x1x1 | 1.0 | 21 | 16/3/2 | 1.53e-15 | 3.22e-15 | 1.2 | 1.58e-16 | 1.69e-15 | 1.58e-16 | 3.10e-14 | 0 |
| 4x1x1;1x1x1 | 1.0 | 16 | 13/3/0 | 2.92e-15 | 2.92e-15 | 1.1 | 2.14e-16 | 2.71e-15 | 2.14e-16 | 3.17e-14 | 0 |
| 2x2x2;1x1x1 | 1.0 | 37 | 29/3/5 | 1.28e-15 | 4.71e-15 | 1.3 | 1.97e-16 | 1.28e-15 | 2.20e-16 | 2.72e-14 | 0 |
| 8x1x1;1x1x1 | 1.0 | 13 | 8/5/0 | 2.55e-15 | 2.55e-15 | 1.1 | 1.95e-16 | 2.55e-15 | 1.95e-16 | 3.07e-14 | 0 |
| 4x4x1;1x1x1 | 1.0 | 21 | 14/4/3 | 1.20e-15 | 1.31e-15 | 0.8 | 2.11e-16 | 1.20e-15 | 2.19e-16 | 2.51e-14 | 0 |
| 16x1x1;1x1x1 | 1.0 | 17 | 9/8/0 | 2.70e-15 | 2.70e-15 | 1.1 | 3.46e-16 | 2.50e-15 | 3.46e-16 | 3.09e-14 | 0 |
| 4x4x4;1x1x1 | 1.0 | 38 | 22/6/10 | 1.48e-15 | 2.11e-15 | 1.0 | 4.11e-16 | 1.48e-15 | 4.44e-16 | 3.39e-14 | 0 |
| 8x8x1;1x1x1 | 1.0 | 21 | 10/6/5 | 1.18e-15 | 1.57e-15 | 0.8 | 5.40e-16 | 1.18e-15 | 5.40e-16 | 2.77e-14 | 0 |
| 16x16x1;1x1x1 | 1.0 | 21 | 6/9/6 | 2.50e-15 | 2.89e-15 | 1.1 | 1.78e-15 | 7.24e-16 | 1.80e-15 | 2.47e-14 | 0 |
| 8x8x8;1x1x1 | 1.0 | 34 | 14/7/13 | 1.21e-15 | 1.37e-15 | 0.8 | 1.47e-15 | 9.47e-16 | 1.72e-15 | 3.50e-14 | 0 |
| 16x16x16;1x1x1 | 1.0 | 38 | 11/10/17 | 3.42e-15 | 4.66e-15 | 1.3 | 4.88e-15 | 5.09e-15 | 5.17e-15 | 2.39e-14 | 0 |
| 2x1x1;1x1x1 | 1.0+0.1i | 8 | 7/1/0 | 6.13e-16 | 1.58e-15 | 0.9 | 1.60e-17 | 6.13e-16 | 1.60e-17 | 2.13e-14 | 0 |
| 4x1x1;1x1x1 | 1.0+0.1i | 8 | 6/2/0 | 1.02e-15 | 1.23e-15 | 0.7 | 1.41e-16 | 1.02e-15 | 1.41e-16 | 2.05e-14 | 0 |
| 2x2x2;1x1x1 | 1.0+0.1i | 8 | 6/0/2 | 7.52e-16 | 1.22e-15 | 0.7 | 1.53e-38 | 7.52e-16 | 2.71e-38 | 2.12e-14 | 0 |
| 8x1x1;1x1x1 | 1.0+0.1i | 8 | 5/3/0 | 1.15e-15 | 1.49e-15 | 0.8 | 2.03e-16 | 1.15e-15 | 2.03e-16 | 1.91e-14 | 0 |
| 16x1x1;1x1x1 | 1.0+0.1i | 8 | 3/5/0 | 1.35e-15 | 1.35e-15 | 0.8 | 2.65e-16 | 1.35e-15 | 2.65e-16 | 2.15e-14 | 0 |
| 4x4x4;1x1x1 | 1.0+0.1i | 8 | 4/0/4 | 6.95e-16 | 6.95e-16 | 0.5 | 2.80e-38 | 6.95e-16 | 6.21e-38 | 2.47e-14 | 0 |
| 8x8x8;1x1x1 | 1.0+0.1i | 8 | 3/0/5 | 9.05e-16 | 9.83e-16 | 0.6 | 7.82e-38 | 9.05e-16 | 1.48e-37 | 3.34e-14 | 0 |
| 16x16x16;1x1x1 | 1.0+0.1i | 8 | 1/0/7 | 1.68e-16 | 6.45e-16 | 0.5 | 2.64e-36 | 1.68e-16 | 5.28e-36 | 8.96e-15 | 0 |

Times in the full table: farSetupX 0.1-2.4 s per pair (the tables were on disk from the earlier runs), farBlockX!
0.7-1481 s per group (the route-3 groups dominate: 16x16x16 f = 1 with 17 k-series offsets is the 1481 s), route-3
offsets 6-90 s each at N 13-44 in BigFloat(128).  The whole part is 4.9 h single-threaded, of which ~4.7 h is route 3;
with the ksrcache/ files it now leaves behind it re-runs in about 5 minutes.

Cost breakdown of the 17582.7 s: farBlockX! 4541.6 s over the 31 groups (max 1480.9 s, the 38 offsets of
16x16x16;1x1x1 at f = 1), farSetupX 31.9 s over the 62 sets (31 direct, 31 swapped), and the remaining 13.0 ks in the
per-record farTensorX calls, six per record (direct, swapped, four reflections).  The direct route-3 column of the
table reads 0.0 s throughout because farBlockX! had already written those `ksrcache/ksrx_*` files; the swapped and
reflected route-3 offsets are fresh keys, and they are what the 13 ks buys.  Largest whole-box L 56, largest n-cut 13,
|R|/g from 2.0 to 128.5.

D13 and this table.  The process loaded `farfield.jl` at 12:45, before the D13 fix was written to it (13:43 and 14:0x,
Section 5), so the numbers above come from the pre-fix file.  They hold for the fixed file as well, by construction of
part (j): each set is built with `offs` = every offset it is then asked for, so rNc, the smallest |R| among that
group's route-1 offsets, is <= every route-1 request, and rHi = 4 x 128 x r_d >= 512 sqrt(3) g = 887 g exceeds the
largest offset in the table (128.5 g) -- the radius check added to farRouteX cannot fire.  The reflection probes reuse
the same set at |sigma R| = |R| and cannot fire it either.  The fix's other hunk (rNc = thr[end] when no offset of
`offs` takes route 1) changes nCut only in a group with no route-1 evaluation, and nCut enters route 1 alone; exactly
one group is of that kind, 3x1x1;1x1x1 at f = 1 (n = 1, routes 0/1/0), whose reported nCut 7 would change while its
tensor values would not.  No tensor value in `j_xref.txt` differs between the two versions of the file, so part (j) was
not re-run.


## 2a. New 220-bit references for the adversarial geometries (`refs_k.jl`, `refs_k2.jl` -> `refs_k.txt`; rule B ord 32 grd 2 of refx.jl, appended to `xwork/xref/reftensors_x.txt` under its lock with rule "B32" in `rule_x.txt`; keys in `refs_k_keys.txt`)

45 records in 38 min (12:43-13:21, JULIA_NUM_THREADS=1, load 3.6-11.7: the user's simple_shapes.jl held 6-12 cores
for the first 20 min) plus 2 far r4x records (kap 100, 300) from refs_k2.jl; the cache grew from 441 to 488 records.
All are fine target g = (1/32)^3 at R, source at 0, f = 1, unless stated.  npt = rule-B points (16.5-19 us each).

| item | tag | R/g (gcd cells) | sT ; sS | npt | s | max|G| |
|---|---|---|---|---|---|---|
| (1) odd ratio 3 on x | r3x_k1, r3x_k2, r3x_k1_lat | (3,0,0) (4,0,0) (3,1,0) | g ; (3,1,1) g | 1.4e6, 5.2e5, 1.4e6 | 20, 9, 20 | 3.1e-2 .. 1.2e-2 |
| (1) ratio 3 xyz | r3xyz_k1, k2, k1_cc, d1 | (3,0,0) (4,0,0) (3,1,1) (3,3,0) | g ; (3g)^3 | 2.1e6-5.8e6 | 31-78 | 0.16 .. 0.048 |
| (1) ratio 6 on x | r6x_k1, r6x_k4 | (4.5,0,0) (7.5,0,0) | g ; (6,1,1) g | 1.7e6, 5.2e5 | 27, 8 | 3.4e-2, 5.0e-3 |
| (1) ratio 6 xyz | r6xyz_k1, k1_cc, d1_cc | (4.5,0,0) (4.5,2.5,2.5) (4.5,4.5,2.5) | g ; (6g)^3 | 1.2e7, 7.7e6, 4.5e6 | 161, 106, 66 | 0.35, 0.18, 0.10 |
| (2) mixed | mix_x_k1/k2/k4, mix_x_k1_off, mix_y_k1/k2, mix_d1, mix_x_k1_z | (3.5,.5,0) (4.5,.5,0) (6.5,.5,0) (3.5,0,0) (.5,2.5,0) (.5,3.5,0) (3.5,2.5,0) (3.5,.5,1) | (4,1,1) g ; (1,2,1) g | 7.9e5-2.6e6 | 11-35 | 1.6e-2 .. 2.0e-3 |
| (3) ratio 3/2 | ni3_64_k1/k2/k8 | R = 5/128 + kap/64 on x, kap 1, 2, 8 | (3/64, g, g) ; g | 4.2e6, 1.3e6, 3.9e5 | 55, 18, 5 | 3.5e-2 .. 1.6e-3 |
| (3) ratio 3/2, 1/48 | ni48_k1_lat, ni48_k4 | (5/192+1/96, 1/192, 1/192), (5/192+4/96, 0, 0) | g ; (1/48)^3 | 8.5e6, 1.6e6 | 112, 21 | 2.5e-2, 5.0e-3 |
| (5) f = 1+i | r4x_f1p1i_k1/k8, r4xyz_f1p1i_cc1/cc8/k1 | (3.5,0,0) (10.5,0,0); (3.5,1.5,1.5) (10.5,1.5,1.5) (3.5,0,0) | g ; (4,1,1) g / (4g)^3 | 3.9e5-8.7e6 | 6-125 | 1.4e-2, 2.0e-4; 4.4e-2, 2.8e-3, 8.3e-2 |
| (5) f = 0.5+2i | r4x_f05p2i_k1/k8, r4xyz_f05p2i_cc1/cc8/k1 | same offsets | same | same | 6-129 | 5.2e-3, 2.9e-5; 1.7e-2, 3.7e-4, 2.7e-2 |
| (6) mixed signs | neg_r4xyz_a/b/y | (-3.5,1.5,-1.5) (-5.5,-1.5,1.5) (1.5,-3.5,-1.5) | g ; (4g)^3 | 1.9e6-5.1e6 | 25-70 | 0.135, 0.062, 0.135 |
| (6) mixed signs, rod | neg_r16x_y, neg_r16x_x | (0.5,2,0) (-9.5,0,0) | g ; (16,1,1) g | 3.4e6, 1.8e6 | 45, 24 | 4.7e-2, 3.6e-2 |
| (4) lambda rod | lamrod_k1/k4/k32 | (17.5,0,0) (20.5,0,0) (48.5,0,0) | g ; (1, g, g) | 2.0e6, 9.2e5, 5.2e5 | 27, 13, 7 | 3.5e-2, 5.2e-3, 2.3e-4 |
| (4) lambda cube | lamcube_cc1, lamcube_k32 | (17.5,15.5,15.5) (48.5,0,0) | g ; (1,1,1) | 1.6e7, 2.1e6 | 220, 29 | 0.43, 0.20 |
| (8) far r4x | r4x_k100, r4x_k300 | (102.5,0,0) (302.5,0,0) | g ; (4,1,1) g | see refs_k.txt | | |

Rule self-checks on the first record of each new geometry type (B24 vs B32, max-entry / per entry): r3x_k1 1.5e-43 /
1.8e-43; mix_x_k1 7.7e-44 / 1.1e-43; ni3_64_k1 1.6e-42 / 1.8e-42; r4x_f1p1i_k1 1.6e-43 / 1.6e-43; neg_r4xyz_a 4.4e-43 /
7.2e-43; lamrod_k1 2.5e-39 / 2.8e-39.  These are the same magnitudes xref saw on the matrix (1e-39 .. 1e-43 at kap 1,
where rule B's ord-32 error was then measured at 1e-47 .. 1e-58 against ord 40 and rule A), so the new records are good to
>= 39 digits at worst and far below anything Float64 can see.  Rule A (pairKer ordN 44, 157 s) vs rule B on r3x_k1:
1.7e-58 / 2.1e-58 -- the rule-B ord-32 quadrature error at a one-cell face gap, exactly xref's 1.6-1.9e-58 at kap 1 for
the ratio-2 and -4 rods; the odd-ratio geometry (kinks at +-a = +-g, +-b = +-2g) changes nothing in the two rules'
agreement.  A lambda-cube lateral-0 offset at kap 1 was NOT referenced (rule B would need ~2e8 points, ~1 h; the
library takes route 3 there, 168 s per offset at kap 32 -- Section 2b (4)).

## 2b. Part (k): the adversarial geometries through the library (`verify.jl k` -> `tables/k_adv.txt`, 1025 s single-threaded, load 3.6-5.2; the pre-fix table is `xwork/xverify/k_adv_prefix.txt`, see Section 5)

66 offsets evaluated, 50 with a 220-bit reference: worst eMx 5.1e-15, worst pEn 5.3e-14 (2.4 digits, one entry of the
lambda-cube box sum, below), swap identity worst 3.5e-15, reflection identity worst 8.4e-16, 0 certificate violations
(direct and swapped).  Per item (R/g in gcd cells; err = eMx, pEn; c/e = certificate over actual error):

(1) Odd ratios (R/g integer, nS - nT even, so m = R/g + (nS - nT)/2 is an integer: the lattice test passes where the
even ratios' lateral-0 offsets fail):
| pair | R/g | route (L) | err | c/e | swap id | flip |
|---|---|---|---|---|---|---|
| 3x1x1 | (3,0,0) (4,0,0) (3,1,0) | 2, 1 (54), 2 | 3.0e-15, 3.0e-15; 7.6e-16, 1.9e-15; 9.4e-16, 9.4e-16 | 6.9 / 21.6 / 18.3 | <= 1.5e-16 | <= 5.7e-17 |
| 3x3x3 | (3,0,0) (4,0,0) (3,1,1) (3,3,0) | 2, 2, 2, 2 (27 pieces) | 9.6e-17 .. 7.2e-16 (pEn <= 7.2e-16) | 33-199 | <= 2.6e-16 | <= 2.6e-16 |
| 6x1x1 | (4.5,0,0) (7.5,0,0) | 2, 1 (46) | 2.8e-15, 2.8e-15; 4.7e-16, 9.9e-16 | 7.1, 24.1 | <= 1.8e-16 | <= 5e-17 |
| 6x6x6 | (4.5,0,0) | 3 (N 23, Lambda 18; 46 s) | 1.9e-17, 6.0e-17 | 250 | 4.0e-17 | 1.5e-39 |
| 6x6x6 | (4.5,2.5,2.5) (4.5,4.5,2.5) | 2, 2 (216 pieces) | 7.1e-16, 1.1e-15; 4.5e-16, 7.0e-16 | 37, 57 | <= 8.4e-16 | <= 8.4e-16 |
Ratio 6 behaves like the even ratios (lateral 0 off-lattice -> route 3); ratio 3 keeps everything on the lattice.

(2) Mixed pair sT = (4g, g, g), sS = (g, 2g, g) (target coarser on x, source coarser on y; nT = (4,1,1), nS = (1,2,1),
b = (2.5, 1.5, 1) g) and the swapped pair (1x2x1 target) as its own group at -R:
| R/g | route (L) | err | c/e | swap identity (direct group) | flip | swapped group: route, sid |
|---|---|---|---|---|---|---|
| (3.5,.5,0) | 2 (8 pieces) | 1.4e-15, 1.4e-15 | 13.0 | 6.6e-17 | 1.3e-16 | 2, 6.6e-17 |
| (4.5,.5,0) | 2 | 2.6e-16, 7.7e-16 | 52 | 7.5e-17 | 1.5e-16 | 2, 7.5e-17 |
| (6.5,.5,0) | 1 (38) | 2.4e-16, 9.2e-16 | 72 | 0 | 0 | 1, 0 |
| (3.5,0,0) | 3 (N 20, Lambda 56; 24 s) | 2.7e-17, 2.7e-17 | 172 | 2.6e-38 | 5.1e-38 | 3, 2.6e-38 |
| (.5,2.5,0) (.5,3.5,0) | 2, 2 | 8.1e-16, 8.1e-16; 2.2e-16, 7.9e-16 | 28, 73 | <= 6.5e-17 | <= 6.5e-17 | 2 |
| (3.5,2.5,0) (3.5,.5,1) | 2, 2 | 2.4e-16, 4.3e-16; 4.7e-16, 7.8e-16 | 117, 39 | <= 2.0e-16 | <= 2.2e-16 | 2 |
The lattice logic handles a pair that is coarser on different axes in the two cells (m = (2,1,0) at (3.5,.5,0): x
half-integer from nS - nT = -3, y half-integer from +1); the swapped group reproduces every value to <= 2e-16 (route 1
exactly).

(3) Non-integer ratios: `farSetupX` refuses both pairs with "the edges must be integer multiples per axis (Gila's
GlaExtInf rule)"; `farTensorX` and `farBlockX!` without `fs` refuse with `InexactError: Int64(3//2)` (C1: the gcd
sub-cell count is computed before the assertion -- refuses, but the message is not the clear one).  Nothing returns a
value.  See 2c for what a rational gcd cell would give.

(4) lambda/2 and lambda coarse cells (f = 1; |k| r_d = 2.9 (16 g cube), 3.2 (32 g rod), 5.6 (32 g cube); nCut max
11 / 13 / 14 / 18, the nCap = 40 refusal never fired):
| pair | R/g | route (L) | err | c/e | t (s) |
|---|---|---|---|---|---|
| 16x1x1 | (9.5,0,0) (10.5,0,0) (12.5,0,0) | 2 (16 pieces) | 2.7e-15, 2.7e-15; 3.6e-16, 7.5e-16; 2.0e-16, 3.0e-16 | 7.3, 43, 67 | 1e-4 |
| 16x1x1 | (16.5,0,0) (40.5,0,0) | 1 (54), 1 (24) | 4.4e-16, 5.1e-16; 3.4e-16, 1.0e-15 | 38, 77 | 1e-5 |
| 16x16x16 | (9.5,7.5,7.5) (16.5,7.5,7.5) | 2 (4096 pieces) | 3.2e-15, 4.7e-15; 2.2e-15, 2.9e-15 | 7.3, 5.6 | 2e-2 (0.96 incl. the fine set) |
| 16x16x16 | (40.5,7.5,7.5) | 1 (32) | 4.7e-16, 1.9e-15 | 46 | 1.5e-5 |
| 16x16x16 | (9.5,0,0) off-lattice | 3 (N 34, Lambda 15) | 4.1e-17, 4.9e-17 | 122 | 77 |
| 16x16x16 | (40.5,0,0) | 1 (34) | 7.8e-16, 1.1e-15 | 11.5 | 1.7e-5 |
| 32x1x1 (lambda rod) | (17.5,0,0) (20.5,0,0) | 2 (32 pieces) | 2.7e-15, 2.7e-15; 2.4e-16, 4.7e-16 | 7.5, 63 | 2e-4 |
| 32x1x1 | (48.5,0,0) | 1 (38) | 1.2e-15, 1.8e-15 | 27 | 1.4e-5 |
| 32x32x32 (lambda cube) | (17.5,15.5,15.5) | 2 (32768 pieces) | 5.1e-15, **5.3e-14** | 7.0 | 0.97 |
| 32x32x32 | (48.5,0,0) off-lattice | 3 (N 65, Lambda 22) | 6.1e-17, 6.1e-17 | 96 | 165 |
Nothing refuses; the lambda cube's near offset is a 32768-term box sum whose worst ENTRY is 5.3e-14 relative (the
entry is 0.1 of the largest; max-entry-normalised 5.1e-15, 2.4 digits lost; the certificate is 7x the actual error and
holds).  The lambda-cube lateral-0 offsets are off the lattice (nS = 32 even) and cost 77-165 s each in the k-series
(N 34-65 at 128 bits); Gila does not produce them (they are the "fine cell aligned with the coarse centre" positions
of xtable S5b).  The whole-box table of the lambda cube needs nCut 18 (file `..._n18_...`), of the lambda rod 14.

(5) Complex f = 1+i and 0.5+2i (r4x, r4xyz; Im k = 6.3 and 12.6): every route works; vs reference (5 records per f):
1+i: 1.8e-15 (r4x kap 1, route 2), 6.6e-16 (kap 8, route 1 L 24), 7.2e-16 / 1.1e-15 (r4xyz cc kap 1 / 8), 6.0e-17
(r4xyz (3.5,0,0) route 3, N 24, Lambda 31, 58 s); 0.5+2i: 1.5e-15 / 2.0e-15 pEn (r4x kap 1), 9.1e-16 (kap 8), 5.3e-16
/ 7.8e-16 (cc 1 / 8), 4.4e-17 (route 3, N 28, Lambda 45, 80 s).  Certificates 6-106x the error, 0 violations; swap
identity <= 4.8e-16, flip <= 4.8e-16.  The route-3 group at 0.5+2i reports nCut max 0 (the table's n-cut is 0 at every
l because the k-series budget tol est is met by the l-th shell alone when e^{-Im k |R|} is small); route 1 at
(10.5,0,0) f = 0.5+2i, L 26: 9.1e-16 -- so the n-cut logic is right there too.

(6) Mixed-sign offsets (3 new r4xyz references, 2 r16x): (-3.5,1.5,-1.5) 6.6e-16; (-5.5,-1.5,1.5) 1.7e-16 / 4.9e-16;
(1.5,-3.5,-1.5) (y-facing) 6.7e-16; r16x (0.5,2,0) (the fine cell beside the rod's long face, m = (8,2,0), route 2)
1.2e-15 / 4.3e-15; (-9.5,0,0) 2.5e-15; (-9.5,-1,0) no reference, flip 2.2e-16; (0,-3,0) off-lattice -> route 3.  All
routes 2; certificates 8-84x the error; flip identity <= 3.1e-16.  The reflection signs of tnsGcdX!/boxSumX! and the
negative lattice coordinates (m = (-2,3,0), (-4,0,3), (3,-2,0), (-2,0,0)) are right.

(7) Float32 (`farTensorX(R, g, sS, ComplexF32(1))`, 10 offsets, r4x and r4xyz): route 1 gives 1.4e-7 and 1.9e-7
relative to Float64 (kap 8, 32 of r4x; = 1.2-1.6 eps(Float32)), route 3 7.0e-9 (r4xyz (3.5,0,0); the 128-bit k-series
rounded once).  **Route 2 returns NaN in every entry** on all 7 route-2 offsets (r4x kap 1, 2; r4xyz cc kap 1, 4 and
(3.5,3.5,1.5)), and farBlockX! in Float32 returns the same NaN (bitwise equal, "y" in the table).  The certificate is
returned as 2.4e-6 max|G| with G = NaN.  Attribution in Section 8 (B2): the equal-cell Bessel workspace overflows
Float32 at the l the near pieces need.

(8) A FrqSetX reused outside the offsets it was built for (r4x pair): a set built with offs = (kap 1,) only and nBlk = 4
(Lw 46 from the disk table, rHi = 46 g, no route-1 offset so rNc = rHi) asked for kap 4 (route 1, L 36): **1.59e-11 vs
the reference and vs a fresh set** (D13, Section 5); kap 32 (L 14) 4.4e-16 and kap 100 (beyond rHi, L 14) 3.5e-16
(new reference).  A set built for kap 32 only, asked for kap 1 (route 2): 2.9e-15 (the fine set is built on demand with
nBlk 128, certified on its own).

(9) Touching / overlapping: R = 0, face touch |R_x| = b_x exactly, edge touch, corner touch (with negative
components) all raise "the cells at R = ... touch or overlap (Gila's contact path)"; (g, 2g, 0) (inside on x, outside on
y) is separated and routes to 3 (off lattice); a gap of g/1000 on x (R = 3g/2 + g/1000) routes to 3, N 16, Lambda 19,
13 s, certificate 4.6e-15 max|G| -- no assertion, no garbage.

## 2c. Non-integer ratios through a rational gcd cell (`farfield_gcd.jl`, a COPY of farfield.jl with `gQ = gcd(sT, sS)` per axis in farSetupX/farTensorX/farBlockX! and the integer-ratio assertion removed; `gcdexp.jl` -> `gcdexp.txt`, load 4.4-5.0)

| pair | gcd cell | nT, nS | R/gcd | route | err vs reference | swap route, identity | t |
|---|---|---|---|---|---|---|---|
| (3/64, g, g) vs g | (1/64, 1/32, 1/32) | (3,1,1), (2,1,1) | (3.5,0,0) (4.5,0,0) | 2 (6 pieces of the 1/64 x 1/32 x 1/32 cell) | 7.6e-16, 9.1e-16; 7.2e-16, 1.4e-15 | 2, 2.0e-16 / 1.0e-16 | 2.7 s (fine set built) / 0.02 |
| same | | | (10.5,0,0) | 1 (L 26) | 3.7e-16, 1.0e-15 | 1, 1.4e-16 | 0.02 |
| g vs (1/48)^3 | (1/96)^3 | (3,3,3), (2,2,2) | (3.5,.5,.5) | 2 (216 pieces) | 2.5e-16, 5.4e-16 | 2, 2.8e-16 | 0.9 |
| same | | | (6.5,0,0) | 3 (off lattice) | 3.2e-17, 7.5e-17 | 3, 2.7e-18 | 44 |
So the three routes need nothing but the rational gcd; the fine set of a non-cubic gcd cell ((1/64, 1/32, 1/32)) is
built by farSetup in 1.9 s (table to L = 26).  The shipped file keeps the refusal (it mirrors Gila's GlaExtInf rule and
I did not want to widen the API in a verification round); if the root wants it, the change is the three `gcd` lines of
`farfield_gcd.jl` (`diff farfield.jl xwork/xverify/farfield_gcd.jl`, 3 hunks) and C1's message becomes moot.

## 3. Part (l): routing and lattice logic (`verify.jl l` -> `tables/l_route.txt`, `l_thread_t<n>.txt`, `l_blk_r4_t<n>.bin`; 15 s single-threaded, load 5.0)

xgeom's blocks re-enumerated from the closed form R/g = (k1, k2, k3) + 1/2, k1 in r/2 .. 7r/2 - 1, k2, k3 in -3r/2 ..
3r/2 - 1 (27 r^3 offsets), touching = |R_d| <= (r+1)/2 g on every axis (independent exact test), lattice and sub-offset
checks re-derived (xLatChk in verify.jl, not the library's xLat/xNear):

| r | offsets | touching | raised by farRouteX | route 1 | route 2 | route 3 | off-lattice | m != xLat | route-2 with a sub-offset of max-norm < 2 | min sub-offset max-norm | rho route 1 max | rho route 2 min | L max | L histogram |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 216 | 16 | 16 | 136 | 64 | 0 | 0 | 0 | 0 | 2 | 0.635 | 0.676 | 54 | 26:12 .. 54:4 |
| 4 | 1728 | 36 | 36 | 1352 | 340 | 0 | 0 | 0 | 0 | 2 | 0.633 | 0.647 | 54 | 24:148 26:220 .. 54:16 (= xunify D.1) |
| 8 | 13824 | 100 | 100 | 11636 | 2088 | 0 | 0 | 0 | 0 | 2 | 0.623 | 0.627 | 56 | 24:992 .. 54:252 |
| 16 | 110592 | 324 | 324 | 93360 | 16908 | 0 | 0 | 0 | 0 | 2 | 0.591 | 0.592 | 56 | 26:3768 28:17708 .. 52:2120 |
Every touching offset raises, every separated one takes route 1 or 2 (route 3 never; nothing is off the lattice, as
xgeom S1/S5.4 said), every route-2 offset's nearest piece has max-norm exactly 2 at the minimum (the one-fine-cell face
gap), xLat and xNear agree with the independent derivation on all 125,256 separated offsets.  The route-1/route-2 rho
boundary is 0.59-0.68 (the Theorem A' thresholds at L = 56).  Route pass 0.06 / 0.01 / 0.14 / 1.6 s.  Off-lattice
control (r4 pair): (3.5,0,0), (4.5,1.5,0), (4,0,0), (4,2,2), (6.5,0,0) are off the lattice and route to 3; (3.5,1.5,0.5)
is on it (m = (5,3,2), nearest piece 2) and routes to 2 -- integer R/g is OFF the lattice for even ratios, as it must be.

farBlockX! (box sum) against farTensorX (per-offset gcd average) bitwise: r = 2 200/200, r = 4 1692/1692, r = 8
13724/13724 offsets (routes 136/64, 1352/340, 11636/2088); warm farBlockX! 0.00 / 0.01 / 0.14 s with the FrqSetX held.
Thread determinism: the r = 4 block's 1692 x 9 entries dumped by the 1-, 4- and 12-thread runs -- see Section 3b.

### 3b. Thread determinism (`tables/l_thread_t{1,4,12}.txt`, `l_blk_r4_t{1,4,12}.bin`)
The r = 4 block (1692 offsets, 1352 whole-box + 340 box sums) filled by farBlockX! with JULIA_NUM_THREADS = 1 (13:43,
load 5.0), 4 (13:45, load 3.7) and 12 (Section 3c): the 4-thread dump is bitwise identical to the 1-thread one (15228
complex entries, 0 differing), and farTensorX per offset is bitwise equal to the block on every offset of the r = 2, 4, 8
blocks in both runs (the :static schedule assigns fixed work; the box sum's order is per offset, independent of the
thread).

## 4. Part (m): Gila today vs the library vs the references, and the realistic block (`verify.jl m` -> `tables/m_gila.txt`, `m_block_t1.txt`, `m_block_t12.txt`; single-thread run 123 s at load 3.8-4.7)

Gila = `egoFunOut!` with genEgoCrcExt!'s arguments (CPUKerOpt(f, 48), adaptive hcubature `egoSrfAdp!` for sep <= 1,
`egoSrfFxd!` with quadOrd(sep, kScl) beyond), timed as the better of two calls; library = farTensorX with the FrqSetX
held (route 2 = the N_t N_s equal-cell pieces per offset) and farBlockX! over the whole block; G-L = Gila vs library
(max-entry), the library being within 5e-15 of every reference in (j)/(k).  r = 4 xyz block: all 1692 separated
offsets; r = 16 xyz block: the 13 referenced offsets plus 120 random offsets per class (seed 20260908), 2172 in all.

| block | class | n | Gila vs library min / median / max | Gila vs ref (n) max | library vs ref max | Gila ms median / max | library us median / max |
|---|---|---|---|---|---|---|---|
| r4 | adaptive kap 0 (R_x = 2.5 g, lateral) | 108 | 1.6e-6 / 4.5e-6 / 6.7e-5 | - | - | 3.1 / 22.0 | 117 / 501 |
| r4 | adaptive kap 1 | 144 | 4.9e-7 / 2.4e-6 / 1.8e-4 | 1.8e-6 (3) | 8.7e-16 | 4.1 / 48.5 | 119 / 1134 |
| r4 | adaptive kap 2 | 144 | 6.1e-7 / 1.1e-5 / 4.7e-4 | 1.3e-5 (2) | 3.0e-16 | 2.2 / 11.4 | 103 / 192 |
| r4 | adaptive kap 3 | 144 | 6.0e-7 / 4.6e-6 / 7.7e-5 | 6.7e-6 (1) | 1.5e-16 | 1.7 / 3.7 | 11 / 138 |
| r4 | fixed ord 9 (sep 2) | 576 | 2.0e-14 / 6.8e-14 / 1.8e-12 | 1.4e-12 (2) | 5.1e-16 | 12.5 / 13.9 | 7.8 / 122 |
| r4 | fixed ord 7 (sep 3-4) | 576 | 1.7e-14 / 4.7e-14 / 4.1e-13 | 4.0e-13 (1) | 3.1e-16 | 4.7 / 5.2 | 5.8 / 12.0 |
| r16 | adaptive kap 0-5 | 120-122 each | 5e-10 / 5e-8-2e-7 / 1.8e-6-3.5e-5 | 3.4e-6 (7) | 3.4e-15 | 4.3-4.8 / 6.8-42.8 | 2900-3100 / 4000-6500 (route 2, 4096 pieces) |
| r16 | adaptive kap 6-15 | 120-122 each | 2e-10 / 3e-9-5e-8 / 6e-8-6.5e-6 | 2.7e-7 (3) | 2.4e-15 | 4.1-4.4 / 4.7-7.9 | 10-16 / 3000-3900 (mixed routes 1/2) |
| r16 | fixed ord 9 | 121 | 8.6e-15 / 2.0e-14 / 5.1e-13 | 5.1e-13 (1) | 2.8e-16 | 12.9 / 14.9 | 8.3 / 19.2 |
| r16 | fixed ord 7 | 121 | 3.6e-14 / 6.5e-14 / 3.3e-13 | 3.2e-13 (1) | 4.7e-16 | 4.7 / 5.7 | 6.6 / 19.5 |
Whole blocks: farBlockX! warm 0.01 s for the r4 block (5.8 us per offset), 0.90 s for the r16 block (110,268 offsets,
8.1 us per offset; routes 93360/16908/0).

Reading: the adaptive rule is 1e-7 .. 5e-4 off (its rtol is 1e-6; the worst, 4.7e-4, is an r4 kap-2 lateral offset),
the fixed rule 2e-14 .. 1.8e-12 (ord 9 at the lambda/8 cube's sep-2 shell reaches 1.4e-12 on the referenced (6.5,1.5,1.5)),
the library <= 3.4e-15 against every reference; per offset the library is 1000-2000x cheaper than the fixed rule (6-8 us
against 5-13 ms) and 5-400x cheaper than the adaptive one on the r4 near band (100-600 us for 64 pieces against 2-48 ms);
at r16 the per-offset route 2 costs 3-4 ms (4096 pieces), the same as Gila's adaptive call, but as a box sum in
farBlockX! it is 8 us per offset (the 16,908 box sums of the realistic block take 0.15 s).  Gila's own numbers agree
with xgeom S2b (adaptive 3.6e-9 .. 1.3e-5 median 9e-7 on its 31 offsets; fixed 2.3e-14 .. 1.8e-4, the 1e-4 rows being
the side-displaced rods, which these cubic blocks do not contain).

Realistic block (coarse 4^3 of 16 g vs fine 64^3 of g sharing a face, 1,404,604 separated offsets, f = 1), JULIA_NUM_THREADS=1,
load 4.2-4.7, the pair table (n13) and the fine tables already on disk in shapetab/ (built by part (j)):
| run | total s | setup | route | fine | fill = whole + box sum | maxrss GB |
|---|---|---|---|---|---|---|
| cold process, no FrqSetX held (tables read from disk) | 8.37 | 1.96 | 1.02 | 1.39 | 4.01 = 3.84 + 0.15 | 4.12 |
| warm | 8.48 | 2.18 | 1.11 | 1.04 | 4.15 = 3.98 + 0.15 | 4.21 |
| warm, cert = true | 16.09 | 2.05 | 1.16 | 1.66 | 11.17 = 11.00 + 0.15 | 4.27 |
| warm, FrqSetX held (fs = fx) | 6.56 | - | | | | |
Routes 1,387,696 / 16,908 / 0 (= xunify D.2 and xtable S4); certificates route 1 min/median/max 4.70e-15 / 7.96e-15 /
4.12e-14 max|G| (516,872 above tol), route 2 8.99e-15 / 1.16e-14 / 2.95e-14 (14,196 above tol) -- the same six numbers
as xunify D.2 to every digit; 13 references in the block, worst 3.42e-15, 0 violations.  xunify's 17.9 s cold included
the 6.5 s table build; its 9.8 s warm against 8.5 s here is load (its run carried a 12-thread acc.jl).  The 12-thread
run is Section 4b.

### 3c. 12 threads (`l_thread_t12.txt`, 13:49, load 4.4 incl. this process)
The 12-thread dump of the r = 4 block is bitwise identical to the 1- and 4-thread dumps (0 of 15228 entries differ);
farTensorX bitwise equal to farBlockX! on 200 / 1692 / 13724 offsets of the r = 2 / 4 / 8 blocks.  farBlockX! is
deterministic in the thread count.

### 4b. Realistic block on 12 threads (`m_block_t12.txt`, 13:49-13:50, 1-min load 4.8 at start, 6.6 at the end, this process included; the user's REPL idle)
| run | total s | setup (serial) | route | fine | fill wall = whole CPU-s + box-sum CPU-s | maxrss GB |
|---|---|---|---|---|---|---|
| cold process, no FrqSetX held (tables read) | 7.24 | 3.60 | 0.72 | 1.65 | 1.26 = 12.68 + 0.27 | 3.17 |
| warm | 5.15 | 2.53 | 0.59 | 1.03 | 1.00 = 8.87 + 1.04 | 3.17 |
| warm, cert = true | 6.50 | 2.21 | 0.79 | 1.21 | 2.25 = 25.13 + 0.33 | 3.29 |
| warm, FrqSetX held | 2.44 | - | | | | |
Routes, certificate statistics and the 13 reference errors identical to the single-thread run (Section 4) and to
xunify D.2 (10.7 cold incl. the table build / 5.7 warm / 7.1 cert).  The fill is 4.0 -> 1.0 s wall (12 threads, 8.9 CPU-s:
the whole-box expansion stays memory-bound as xunify said); the serial farSetupX pass (2.2-3.6 s: 1.4e6 exact-rational
offsets through thresholds and n-cut) is half of a warm call and vanishes when the FrqSetX is held (2.44 s).  The box-sum
CPU time varies 0.27-1.04 s between runs at identical results (thread contention on the shared fine egoToe, not code).

## 5. The one fix made to farfield.jl: D13, a FrqSetX reused at an uncertified radius (`diff xwork/xverify/farfield_prefix.jl notes/farfield/farfield.jl`: 3 hunks, +10 lines; pre-fix copy `farfield_prefix.jl`, md5 98148dae6c5c9f084e3b0b0d2479b773)

Cause.  farSetupX certifies the k-series cut of the whole-box table (nCutVec, D11's cutMax) at two radii: rHi = 4 nBlk
r_d and rNc = the smallest |R| among the offsets of `offs` that take route 1 -- or, when no offset does, rNc stays rHi
(the old line 1614 lowered rNc to thr[end] only for EMPTY offs).  The n-tail bound relative to est decreases with |R| for
l >= 2, so a set is certified on [rNc, rHi] and nowhere else; nothing checked the radius of a later request.  A caller
holding a FrqSetX across blocks (the documented way to skip the 2-3 s selection pass) that asks for a nearer offset gets
the far-end N silently.

Before (k_adv_prefix.txt, section (8)): set built with offs = ((3.5 g, 0, 0),) [route 2 only] and nBlk = 4 for the pair
g / (4g, g, g), f = 1; then farTensorX with fs = that set:
| offset | route, L | vs fresh set | vs 220-bit reference | certificate / max|G| |
|---|---|---|---|---|
| kap 4 = (6.5 g, 0, 0) | 1, 36 | 1.59e-11 | 1.59e-11 | 1.7e-14 (violated 1000x) |
| kap 32 = (34.5 g, 0, 0) | 1, 14 | 0 | 4.4e-16 | ok |
| kap 100 = (102.5 g, 0, 0), beyond rHi = 46 g | 1, 14 | 3.6e-18 | 3.5e-16 | ok in practice, uncertified |

Fix (three places, all in the cross-scale section; the equal-cell code is untouched, Section 6a re-proves bit-identity):
1. `FrqSetX` gains `rNc::Float64, rHi::Float64` (after nBlk), set by farSetupX;
2. farRouteX, route 1: `fx.rNc (1 - 1e-12) <= |R| <= fx.rHi (1 + 1e-12)` or
   error("farRouteX: |R| = .. is outside [rNc, rHi], where this set's n-cut is certified; build the set with this offset in offs");
3. farSetupX: `rNc == rHi && isfinite(thr[end]) && (rNc = min(rNc, thr[end]))` -- a set whose offsets certify nothing
   for route 1 (none, or all in the near band) is certified from thr[end], the nearest radius route 1 can be taken at,
   instead of from rHi only (the first version of the fix, `k_adv_fix1.txt`, refused kap 32 as well: the certified
   interval had collapsed to the point rHi).

After (tables/k_adv.txt (8), run 13:53):
| offset | result |
|---|---|
| kap 4 | route 1 L 36, vs fresh set 0 (bitwise), vs reference 1.21e-15 (the set is now certified from thr[end] = 4.69 g) |
| kap 32 | route 1 L 14, 0 / 4.4e-16 |
| kap 100 | error: "|R| = 3.203125 is outside [0.14669, 1.43614], where this set's n-cut is certified; build the set with this offset in offs" |
| far-only set asked for kap 1 | route 2 (fine set), 2.9e-15, unchanged |
Everything else in k_adv.txt is identical to the pre-fix run except the header line of the route-3-only group r4xyz
f = 0.5+2i, whose nCut max rose 0 -> 10 (it is now certified at thr[end]) with identical tensor values; the (j) run
(pre-fix file, sets built by farBlockX! with every offset in offs) is unaffected by construction: its sets always hold
a route-1 offset or evaluate none.  farBlockX!/farTensorX with fs = nothing pass their offsets, so the check can never
fire for them; it fires only for a hand-built or reused set, which is the case that was wrong.

## 6. Parts (a)-(i) re-run at the end (13:53-14:05, one process per part, JULIA_NUM_THREADS=1, load 3.6-4.8; `verify_ai_<p>.log`) and bit-identity re-proved after the second edit

Run times: a 4.7 s, b 11.4, c 3.1, d 17.8, e 47.3, f 88.4, g 18.6, h 468.1, i 22.5; every part exit 0.  `git diff --stat`
over tables/[a-i]_*.txt:
    a_rho.txt   | 78 ++++++++++++++++++++++++++++++++++++++-
    e_route.txt | 28 +++++++-------
    h_pos.txt   | 12 +++---
    i_cost.txt  | 78 +++++++++++++++++++--------------------
    4 files changed, 136 insertions(+), 60 deletions(-)
22 of the 26 tables are byte-identical to their staged copies (md5 of `git show :path` vs the working file: a_band,
a_div, a_mu, b_volume, the 8 c_gila_*, d_ref, e_digits, e_ncut, e_terms, f_bands, f_ksr, f_overlap, g_sym).  The four
that differ: a_rho.txt -- the staged copy is the known 4-line truncation (ends mid-header), the regenerated file is the
81-line table and its first 4 lines equal the staged ones (xunify F reported the same; my accidental `include` of
verify.jl at 12:50, killed after part (a), had left a 5-line file, now regenerated); e_route.txt -- identical except the
last column "setup s" (0.7-1.1 s timings; checked column-wise); h_pos.txt -- the 12 changed lines are all "Gila build
(wekTrp ...) NN s" timings (30.5 -> 31.1 s .. 176.6 -> 186.5 s); i_cost.txt -- the ns/offset and ns/term timing columns
(the shp/f/L/terms columns pair up line for line).  So no numerical output of parts (a)-(i) changed; the code of (a)-(i)
is byte-identical (Section 0).

`bitid_v.jl` re-run after the second edit (farSetupX's rNc line; 13:56, `bitid/bitid.txt`): (a) 0 mismatches, (b)
120/120 farTensor bit patterns identical (routes 108/12/0, nCut equal, Lw/Lo 56/56), (c) both 12^3 farBlock! blocks 0
differing entries, (a') fresh farShape files byte-identical -- the equal-cell paths are bit-identical to
farfield_pre_cross.jl with both edits in place.

Side effects on the deliverable dirs: `notes/farfield/shapetab/` now holds the pair tables of every pair evaluated in
(j)-(m) (17 files, 0.4-2.5 MB each, `..._a<a>_seg.txt`; the four equal-cell files unchanged, md5 as shipped);
`notes/farfield/ksrcache/` holds 36 `ksrx_*` route-3 files (one per ordered pair and f) written by (j)/(k), so a re-run
of (j) is ~5 min instead of ~1.5 h; `tables/l_blk_r4_t{1,4,12}.bin` (244 kB each) are the determinism dumps.

## 6a. Equal-cell bit-identity with the D13 fix in place (`xwork/xverify/bitid/bitid_v.jl` = xunify's bitid.jl with New = the current notes/farfield/farfield.jl, Old = xunify/farfield_pre_cross.jl, private copies of the table and k-series dirs; 13:50, 1 thread, load 4.9-5.1)
(a) fresh tabWhl(L 10)/tabOct(L 7) at 256 bits, farMomW(80), nCutVec(56) at 4 radii/caps, 4 shapes: 0 mismatches;
(b) farTensor at 30 offsets x 4 shapes, f = 1: 120/120 bit patterns identical, routes 108/12/0, nCut vectors equal,
Lw/Lo 56/56 both; (c) farBlock! 12^3 blocks c32 (13894 nonzero entries) and slender (13952): 0 differing; (a') fresh
farShape files byte-identical.  Identical to xunify B'; the fix touches FrqSetX/farSetupX/farRouteX only.

## 7. Code review of the cross-scale section (read before running; each item's fate is stated after the runs)

| # | where | observation | severity |
|---|---|---|---|
| C1 | farTensorX :1867-1868, farBlockX! :1931-1932 | `nT = Int(sTQ / gQ)` with `gQ = min(sT, sS)` runs BEFORE farSetupX's integer-ratio assertion, so a non-integer pair without `fs` fails with `InexactError: Int64(3//2)` instead of the clear message (probe.txt, confirmed) | cosmetic (still refuses, no garbage) |
| C2 | xLat :1660-1663 | m = R/g + (nS - nT)/2: on an axis where the coarse cell is an EVEN multiple of g, a lateral coordinate 0 (fine cell aligned with the coarse centre) is a half-integer m -> route 3 (k-series, seconds to minutes per offset: 75 s at the lambda/2 cube kap 1, 168 s at the lambda cube kap 32, probe.txt).  Gila never produces such offsets (xtable S5b, xgeom S5.4), but a caller of farTensorX can | performance trap, documented, not a defect |
| C3 | fineSet :1641-1648 | the equal-cell set of g is built on FIRST use with nBlk = max(fx.nBlk, nBlk) and never enlarged: a FrqSetX built for a small block and reused (`fs = fx`) on a larger one keeps the small fine set (its rHi is 4 nBlk r_g); the fine set's route selection/n-cut were certified up to that rHi only | tested in (k)(8) |
| C4 | farRouteX :1680-1683 | an offset needing whole-box L > fx.Lw raises (clear) rather than falling back to route 2/3; farBlockX! sizes the table on `offs` so it never fires there; farTensorX with a foreign `fs` can hit it | tested in (k)(8); behaviour acceptable |
| C5 | farSetupX :1601 | rHi = 4 nBlk r_d with nBlk = 128 by default (farTensorX/farBlockX! pass nBlkX, which covers their offsets); with a hand-built `fs` an offset beyond rHi gets the L of thr[1] (0) -- the thresholds were computed only up to rHi | tested in (k)(8) with a new reference at kap 100 |
| C6 | xMult, xNear, boxSumX! | re-derived: c_j - c'_j' = (m + j - j') g, multiplicity min(nT, t + nS) - max(1, t + 1) + 1, smallest sub-offset max-norm = max_d of the per-axis smallest |coordinate| (product set), reflection signs sign(D_a) sign(D_b) on the off-diagonal entries; all correct | -- |
| C7 | tnsGcdX! vs boxSumX! | the per-offset path (farTensorX) sums the pieces in the same order as the box sum (t3, t2, t1 outer to inner) -- bitwise equality is expected and is checked in (j), (l) | -- |
| C8 | ksrCachedX! :1816-1831 | on every memory miss the whole disk file is re-parsed under fx.lk (O(records) per miss); harmless at the route-3 volumes seen | cosmetic |
| C9 | farSetupX :1590 | the integer-ratio assertion refuses pairs whose ratio is rational but not integer (3/64 vs 1/32; 1/32 vs 1/48) although route 1 needs nothing and route 2 only needs the rational gcd cell (`gcd(sT_i, sS_i)` instead of `min`) | limitation, matches Gila's GlaExtInf; a 3-line extension is tried in `farfield_gcd.jl` (Section 2 (3)) |
| C10 | tnsKsrX :1776-1779 | Lambda uses est with V_s and the prefactor 1/V_t: consistent with Theorem A' (xtheory 1.5/1.6); N from ksrOrd(|k| D_max, tol / max(1, Lambda)) | -- |

## 8. Bugs and suspicious behaviour found (reproducers in the tables named; B1/B3 are cross-scale, B2 equal-cell)

| id | what | reproducer | numbers | status |
|---|---|---|---|---|
| B1 = D13 | a FrqSetX reused at a whole-box offset NEARER than the ones it was built for returns an uncertified tensor: farSetupX certifies the k-series n-cut of the table at [rNc, rHi] with rNc = the nearest route-1 offset of `offs` (or rHi when none takes route 1), and farRouteX/farTensorX/farBlockX! never checked |R| against it | `fx = farSetupX(g, (4g,g,g), 1; offs = ((3.5g,0,0),), nBlk = 4); farTensorX((6.5g,0,0), g, (4g,g,g), 1; fs = fx)` | 1.59e-11 vs the 220-bit reference and vs a fresh set (certificate 1.7e-14 max|G|: a 1000x violation); the fresh set gives 1.2e-15 | FIXED in farfield.jl (Section 5): rNc/rHi stored in FrqSetX, farRouteX raises outside [rNc, rHi]; farSetupX certifies at thr[end] when no given offset takes route 1 |
| B2 | Float32 returns NaN at every offset whose whole-box L or octant L_c is >= 28: the upward h_l(kR) recursion overflows Float32 (|h_27(0.79)| > 3.4e38); cross-scale route 2 inherits it on the whole near band | `farTensor((4,0,0), (1/32)^3, ComplexF32(1))`; `farTensorX((3.5g,0,0), g, (4g,g,g), ComplexF32(1))` | NaN (all 9 entries); |D| >= 5 (L <= 26): 1.4e-7 .. 4.4e-7 vs Float64; tnsWhl! forced to L = 24 fine (9e-8), L = 28 Inf, L >= 32 NaN | NOT fixed (equal-cell path under the bit-identity mandate); Sections 2b (7), 8b |
| B3 = C1 | farTensorX / farBlockX! without `fs` on a non-integer-ratio pair fail with `InexactError: Int64(3//2)` before farSetupX's clear message | `farTensorX((5/128+1/64, 0, 0), (3/64, g, g), g, 1)` | error raised, no value | cosmetic; the rational-gcd copy (2c) removes both the message and the limitation |
| S1 | the lateral-0 offsets of an even-ratio pair are off the gcd lattice and go to the k-series: 46-165 s per offset at the lambda/2 - lambda cubes (N 23-65 at 128 bits), 13-80 s at ratios 2-6 | `farTensorX((48.5g,0,0), g, (1,1,1), 1)` | 165 s, error 6.1e-17 | by design (Gila never produces them); documented |
| S2 | the lambda-cube box sum (32768 pieces) loses 2.4 digits on one entry: pEn 5.3e-14 at (17.5,15.5,15.5) g vs eMx 5.1e-15 | Section 2b (4) | certificate 3.6e-14 max|G| = 7x the actual error, holds | the entry is 0.1 of max|G|; the box sum rounds 32768 terms; not a defect |
| S3 | est_X/max|G| < 1 cases: r4xyz f = 0.5+2i route-3 group reports nCut max 0 | Section 2b (5) | route 1 at (10.5,0,0), L 26: 9.1e-16 | the l-th shell alone meets the budget when e^{-Im k |R|} is small; correct |

## 8b. B2: Float32 returns NaN at near offsets (`f32.jl` -> `f32.txt`; equal-cell paths, pre-existing, NOT fixed here)
farTensor(D, g, ComplexF32(1)) for the (1/32)^3 cell: D = (2,0,0) (octants, L_c 50), (2,1,1), (2,2,2) (whole box L 36),
(3,0,0) (L 44), (3,1,0) (L 42), (4,0,0) (L 32) return NaN in every entry; (5,0,0) (L 26) and beyond give 1.4e-7 ..
4.4e-7 relative to Float64.  At D = (4,0,0) with tnsWhl! forced to L = 4 .. 56: 9e-8 .. 1.2e-7 up to L = 24 (max |h_l|
in the Float32 workspace 5.0e32), Inf at L = 28, NaN from L = 32: the upward h_l(kR) recursion (hFill!) overflows
Float32 (3.4e38) at l ~ 27 for kR = 0.79, and every offset whose selected L is >= 28 (all of |D| <= 4 at lambda/32,
f = 1) is lost.  Part (g) of verify.jl tested Float32 at one far offset only (6.2e-8) and missed it.  Cross-scale route 2
inherits it on every near-band offset (Section 2b (7)); routes 1 and 3 are fine (route 1 near offsets would fail the
same way once L >= 28, e.g. r4x kap 3 (L 34) -- those went to route 2 here).  Fix (not applied: equal-cell path, and
the bit-identity mandate): run hFill!/shrm! in Float64 and round the products, or scale the recursion (h_l / (2l-1)!!
stays O(1)), or at least `isfinite` the workspace and raise.  A user of the Float32 API today must keep L < 28, which
the routing does not know.
## 9. Verdict, and what the document agent should carry

The integrated cross-scale library is correct at the level this round can test.  Against 439 cached references plus 50
newly referenced adversarial offsets: worst max-entry error 3.4e-15, worst per-entry 5.3e-14 (a 32768-piece box sum at a
lambda coarse cell; 4.7e-15 everywhere else), 0 certificate violations in 978 certified evaluations, farTensorX bitwise
equal to farBlockX! on 439 + 15,616 offsets, farBlockX! bitwise deterministic in the thread count (1, 4, 12), routing and
lattice logic exactly right on 125,256 separated offsets of the four test blocks with every touching offset refused, and
the swap and reflection identities exact on route 1 and at rounding level on routes 2/3.  xunify's accuracy section C is
independently reproduced (Section 1).  One real defect was found and fixed (D13, Section 5); one pre-existing equal-cell
defect was found and left (B2, Float32 NaN, Section 8b); the non-integer-ratio refusal is a limitation, not a bug, and a
three-line change makes it work (Section 2c).

For the document: (a) D13 and its fix belong in the defect table next to D11/D12 -- the certified radius interval
[rNc, rHi] of a FrqSetX is now part of the API contract ("a set may only be reused at offsets it was built for; pass them
in `offs`"); (b) B2 belongs in the uncertain/defect list: the Float32 API is unusable below |D| = 5 at lambda/32 and
across the whole cross-scale near band; (c) the "est_X is loose across a slender thin axis" observation (8930x on
(0,0,2) g_z) is confirmed as the loosest certificate in the whole cache; (d) Gila's numbers of Section 4 (adaptive
1e-7..5e-4, fixed 2e-14..1.8e-12, library <= 3.4e-15, 1000-2000x cheaper per offset) are the before/after the prompt
asked for.

## Where I stopped
All deliverables are done: (j) 439 references, (k) 66 adversarial offsets with 47 new 220-bit references, (l) routing and
lattice logic on the four blocks plus thread determinism at 1/4/12 threads, (m) Gila before/after on the r = 4 and r = 16
blocks and the realistic 1.4e6-offset block at 1 and 12 threads, the (a)-(i) re-run with the diff stat, and bitid.
verify.jl is 1513 lines (parts (a)-(i) byte-identical, md5 of their body 163e37cede2587a91c6b4dc25adbeb8c before and
after); farfield.jl carries the D13 fix (3 hunks, +10 lines) with the equal-cell paths re-proved bit-identical.
Not done, and why: no reference was computed for a lambda-cube lateral-0 kap-1 offset (rule B needs ~2e8 points, about
an hour, and the library takes route 3 there, which the cache's other route-3 records already validate to 1e-17); the
Float32 defect B2 was diagnosed but not fixed (equal-cell code, under the bit-identity mandate -- it is the root's call);
the rational-gcd extension exists only as the patched copy `farfield_gcd.jl`.  Julia processes started by me: probe.jl
(killed after 12 min once (k) covered it), refs_k.jl, refs_k2.jl, gcdexp.jl, f32.jl, bitid_v.jl x2, verify.jl j, k x3,
l x3 (1, 4, 12 threads), m x2 (1, 12 threads), and the nine (a)-(i) parts -- never more than two at once, and the
12-thread runs (l, m) were the only multithreaded ones.  Load averages are stated with every timing; the user's own
`simple_shapes.jl` held 6-12 cores from 12:40 to 13:10, which is why the early reference timings are the slow ones.

## DONE
