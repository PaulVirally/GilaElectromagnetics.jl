# xunify: the unequal-cell far field integrated into farfield.jl

Dir: `SCRATCH/xwork/xunify/`.  Deliverable copy: `farfield_cross.jl` (farfield.jl + cross-scale); the original
kept as `farfield_pre_cross.jl` (md5 b4d74cabf9ec5d6c17cb5dd89c3c37d3 = notes/farfield/farfield.jl at the start).
Scripts: `smoke.jl` (first run of the X API), `bitid.jl` -> `bitid.txt` (B), `acc.jl` -> `acc.txt`, `acc_rows.md` (C),
`block.jl <r4|big>` -> `block_<which>_t<threads>.txt` (D), `build.sh`/`build.jl` -> `build.txt` (E).  Table copies:
`tab_old/`, `tab_new/` (the shipped shapetab/, one copy per module of bitid.jl; the New copy also receives the
cross-scale tables), `ksr_old/`, `ksr_new/` (the shipped ksrcache/).  Run: `JULIA_NUM_THREADS=1 julia
--startup-file=no --project=SCRATCH/env <script>`; the accuracy run (C) and one block run (D) with 12 threads.

## A. The integrated file

`farfield_cross.jl`: 2009 lines after the D12 fix of 08:28 (1994 before it; farfield.jl: 1488).  `diff
farfield_pre_cross.jl farfield_cross.jl`: 54 lines of the original removed/replaced, 575 added; the cross-scale
section is lines 1537-2009 (473 lines; the public API from 1853), the changes to the equal-cell part are 27 hunks
between lines 13 and 948.  The line numbers in the table below are those of the 08:01 file; in the shipped file the
definitions sit at: wgtX 42, wgtTrpX 44, tabWhl 239, ShpTab 327, SHPC 355, xKey 368, shpFile 370-371, mrgGeo 405-416
(D12, new), farShape 427, lamX 670, farMomW 671, nCutVec 865, cutMax 888, farSetup's `cut(mm, nc) = cutMax(..)` 948,
trpAB 1551, farSetupX's cutMax 1615.

Equal-cell part, what changed (everything else is byte-identical; proof of bit-identity of the results in B):

| where (new line) | change | why |
|---|---|---|
| 13-17 | header comment: the unequal-cell paragraph | documentation |
| 39-44 | new `wgtX(n, a, b)`, `wgtTrpX(a, b, dM)`: trapezoid moments 2 (b^{n+2} - a^{n+2})/((n+1)(n+2)) | trapezoid = triangle(b) - triangle(a); at a = 0 the subtraction of an exact 0 leaves wgtT's operations |
| 235-244 | `tabWhl(s, ..)` forwards to `tabWhl(ta, tb, ..)`; body: `W = wgtTrpX(ta[d], tb[d], dM)`, `vt = prod(tb)` (was wgtTrpT(s[d]), prod(s)) | one body for both weights; table stored divided by prod(b) |
| 328-329 | `ShpTab` gains `a::NTuple{3,QI}` (0 for equal cells) | the table of a pair is keyed on (a, b) |
| 355 | `SHPC` keyed `(a, b, nMax)` (was `(s, nMax)`) | same |
| 365-372 | `xKey(ta)` = "" for a = 0, else `_a<n>_<d>-<n>_<d>-<n>_<d>`; `shpFile(s, nMax, dir)` forwards to `shpFile(ta, tb, nMax, dir)` = `shpKey(tb, 0, nMax) * xKey(ta) * "_seg.txt"` | equal-cell files keep their names; a pair's file sits next to them in TABDIR |
| 417-484 | `farShape(s; kw...)` forwards to `farShape(ta, tb; ...)`; body: key `(ta, tb, nMax)`, `Lo < 0` (or `Lw < 0`) skips that slot (an unequal pair has no octant table; the skipped slot is the empty segment already used when only one table grew), `tabWhl(aB, bB, ..)`, ShpTab built with `emtGeo` for a skipped slot | the variable `tb` was renamed `tb0` because `tb` is now the b triple |
| 659-680 | `farMomW(lTop, s)` forwards to `farMomW(lTop, ta, tb)`; `lamX(n, a, b)`; body: `tri = wgtX(2i, a, b)`, `flt = lamX(2i, a, b)`, `pnt = b^{2i}`, `zro = a^{2i}` (was the [1, 0, 0, ..] vector) | Theorem A' measures; 0^0 = 1, 0^{2i} = 0 reproduces `zro` |
| 855-862 | `nCutVec(lTop, mom, s, ..)` forwards to `nCutVec(lTop, mom, vt, vs, rd, ..)` with `vt = vs = prod(s)`, `rd = |s|` (the very expressions the old body computed); body: `est(.., vs)` | V_t prefactor and V_s scale separated |
| 880-881 | new `cutMax(a, b)`: elementwise max in which -1 is absorbing | D11 |
| 939-942 | farSetup: `cut(mm, nc) = cutMax(nCutVec(.. rNc ..), nCutVec(.. rHi ..))` (was `max.`) | D11: an uncertified -1 at the near radius was swallowed by the far radius' N whenever the far end was met at nMax; it now propagates and triggers the nCap recomputation |

Function bodies of the equal-cell library that changed at all: `tabWhl` (two lines: W and vt), `farShape` (key,
file name, skip logic), `farMomW` (four moment vectors), `nCutVec` (two lines: vt/rd removed, est(.., vs)),
`farSetup` (the cutMax line).  Not touched: tabOct, farMomO, bndTrm/bndTrmO/bndCum, whlThr, nearOff, boundL,
boundLoct, farRoute, every Bessel/harmonic routine, tnsWhl!/tnsOct!, route (iii), farTensor, farBlock!,
farRouteStat.  The 27-piece split (tabPce, farMomO's general affine piece, tnsPceX!) of the prototype is NOT in the
deliverable (root's decision 1); it stays in xwork/xtable/farx.jl as the cross-check.

Cross-scale section (lines 1528-1994), public names in bold:

| name | what |
|---|---|
| `trpAB(sT, sS)` | (a, b) of the pair |
| `FrqSetX{T,E}` | pair set: sT, sS, exact edges, gcd cell gQ and the sub-cell counts nT, nS, frq, k, prf = ik/f^2 prod(b)/V_t, trapezoid FrqBox `whl`, thresholds, bound factors, nCut, and lazily the equal-cell FrqSet of g (`fine::RefValue`), the route-3 caches `ksr`, `ksrI` and the lock |
| **`farSetupX(sT, sS, f; tol, nBlk, offs, dir, ..)`** | thresholds from Theorem A' (farMomW(a, b), r_d = |b|, est with V_s, prefactor V_t), table sized on the route-1 offsets of `offs`, n-cut certified at the nearest of them (without offsets: at thr[end], the nearest radius route 1 can be taken at), `cutMax` combination, table `farShape(taQ, tbQ; Lw, Lo = -1, nMax)` |
| `fineSet(fx; nBlk)` | the equal-cell set of g, `farSetup(gQ, frq; tol, nBlk = max(fx.nBlk, nBlk), ..)` on first use, under fx.lk |
| `boundLX`, `xLat`, `xMult`, `xNear` | whole-box L; lattice coordinate m = R/g + (nS - nT)/2 (nothing off lattice); multiplicity of t = j - j'; smallest max-norm of the sub-offsets |
| **`farRouteX(fx, R)`** | (kind, L): 1 whole box at l = L (Theorem A' met within the table), else 2 if R is on the gcd lattice and every sub-offset has max-norm >= 2, else 3; touching/overlapping pairs (|R_i| <= b_i on every axis, exact rationals) raise an error |
| `tnsWhlX!`, `bndWhlX` | route 1 and its Theorem A' bound at (rr, L) |
| `certEq(fs, D, kind, L, Lc)` | certified absolute bound of the equal-cell tensor at D: Theorem A cum at L (route i), the sum of the eight Theorem B cums (route ii), KSRTOL est_g (route iii) |
| `tnsGcdX!(G, fx, fs, ws, m, tmp, cert)` | route 2 for one offset as farTensor's routes on every sub-offset, reflected from |D|, weights xMult/N_t; returns (1/N_t) sum of the pieces' certificates |
| `FACESX`, `boxFaceQ`, `facePairX`, `MOMCX`, `tnsKsrX`, `ksrCachedX!` | route 3: the 36 unequal face-pair k-series (BigFloat KSRPRC = 128, N from ksrBnd at tol/Lambda with Lambda = sum I_{-1}/(4 pi |f|^2 V_t)/est_X), memory cache per pair set and a disk cache `KSRDIR/ksrx_<sT>_<sS>_f<re>_<im>.txt` (26 tokens per line: the exact R, N, Lambda, 9 complex BigFloat values) |
| **`farTensorX(R, sT, sS, f; tol, fs, cert, dir)`** | one exact-rational offset; `cert = true` returns (G, c) with c = Theorem A' bound at the selected L (route 1), the summed pieces' certificates (route 2), KSRTOL est_X (route 3), each + 20 eps(T) max|G| |
| `boxSumX!(G, ego, cg, m, nT, nS)` | the box sum over the fine egoToe (reflected entries, weights xMult, /N_t) and its certificate from the per-entry array cg |
| **`farBlockX!(G, Rs, sT, sS, f; tol, fs, cert, dir, tms)`** | routes (threaded), ONE fine egoToe of side nbx = the largest sub-offset index + 1 filled by `farBlock!` (fine set from `fineSet`), route 3 threaded under KSRPRC, routes 1/2 threaded over offsets with per-thread FarWs; returns the route kinds, with `cert = true` also the certificates; `tms` receives the phase times (:setup, :route, :fine, :ksr, :fill, :whole, :boxsum) |

Design points as decided by the root: the whole box is the production far route (Theorem A', table keyed on the
unordered pair via (a, b), stored /prod(b), prf carries prod(b)/V_t); the near band is the gcd average as a box sum
over one fine equal-cell egoToe (weights +1/N_t, N_t x N_s integer fine offsets: m + j - j' with m = R/g + (nS -
nT)/2, so the half-integer R/g of even ratios cancels); the k-series on the 36 unequal face pairs where the pair is
off the gcd lattice or a sub-offset would touch.  est scales with V_s, r_d = |b|.  The equal-cell tolerance of the
fine pieces is the pair's tol (the a posteriori certificate is the sum of the pieces' certificates; it is reported
against the actual error in C and D).  Generic in T (FrqSetX{T,E}, weights T(xMult), ego Complex{T}), constants typed
(KSRTOL = 1e-16 is tnsKsr's default tolerance, now named), no allocation in boxSumX!/tnsWhlX!/tnsGcdX! per offset
beyond the caller's tmp, locks around SHPC (SHPLK), MOMC/MOMCX (MOMLK), fx.ksr/fx.fine (fx.lk); disk format of the
shape tables unchanged (segments of (whl, oct) pairs, the oct slot of a pair is the empty segment).

Observation on the way (pre-existing, unchanged): farShape consults SHPC before the disk, so an in-memory table
built into one dir and a later call with another `dir` appends the extension to the second dir's file without
reading it (bitid.jl's first run did exactly that: its fresh L = 12 build into tab_old_fresh/ made the following
farSetup(c32) append l = 14..48 / 10..51 segments to the copies of the shipped c32 file, which already held l <= 56;
a file with overlapping segments would then be read with duplicated columns).  The library's normal use has one
TABDIR per process, so this cannot happen there; bitid.jl was reordered and the copies restored from the shipped file.

## B. Bit-identity of the equal-cell paths (`bitid.jl` -> `bitid.txt`; Old = farfield_pre_cross.jl, New = farfield_cross.jl, two modules in one process, each with its own copy of the shipped shapetab/ and ksrcache/; JULIA_NUM_THREADS=1, load 2.8-3.1)

| check | result |
|---|---|
| (a) fresh `tabWhl(L = 10)` and `tabOct(L = 7)` at BigFloat(256), nMax 12, every GeoBox field (value, sign, precision of every BigFloat; index lists; cnc), 4 shipped shapes ((1/32)^3, (1/8)^3, (1/4)^3, (1/32,1/32,1/512)) | 0 mismatches |
| (a) `farMomW(80)` Float64 arrays (six fc vectors and W), 4 shapes | 0 mismatches |
| (a) `nCutVec(56, ..)` at kR = 2 pi 0.3 and 2 pi 30, nCap 12 and 40, 4 shapes | 0 mismatches |
| (a') fresh `farShape((1/32)^3; Lw = 12, Lo = 9)` written to a fresh dir by each module, files compared byte for byte | identical (0c4b4419cb8c95ae387954e10e0401df both); the accidental first run's l = 14..48 / 10..51 extensions written by both modules were also byte-identical (34974803 bytes, `cmp` silent) |
| (b) `farSetup` then `farTensor` at 30 offsets x 4 shapes, f = 1 (offsets of xtable S5: (2,0,0) .. (40,7,2)), UInt64 bit patterns of Re and Im of all 9 entries + route kind | 120/120 identical; routes (i)/(ii)/(iii) = 108/12/0; `nCut` vectors equal for all 4 shapes; table Lw/Lo held 56/56 for all four shapes in both modules (the first, polluted run had 48/51 for c32 and was also 120/120) |
| (c) `farBlock!` on a 12^3 egoToe, (1/32)^3 at f = 1 | whole array bitwise identical (13894 nonzero Complex entries, 0 differing); 1.6 s old / 1.5 s new |
| (c) `farBlock!` on a 12^3 egoToe, slender (1/32,1/32,1/512) at f = 1 | bitwise identical (13952 nonzero, 0 differing); 0.8 s / 0.8 s |
| (d) the four shipped shapetab files, read by New | read as before (Lw/Lo 56/56 from disk); md5 of the copies after every run = the shipped files (c32: 01ed86a48e2083529b041b4ddb42d0c4); the cross-scale tables of the pairs sit next to them under their own names (`s3_64-1_32-1_32_L0_n12_p192_a1_64-0_1-0_1_seg.txt` for (g, (2g, g, g)) etc.) |

Why the equal-cell results cannot move: `wgtX(n, 0, s) = 2 (s^{n+2} - 0^{n+2})/(..)`, and `x - 0 == x` in every
type (BigFloat included, sign and precision); `lamX(n, 0, s) = 2 (s^{n+1} - 0)/(n+1)`; `0.0^0 = 1.0`, `0.0^{2i} = 0.0`
is farMomW's `zro`; `nCutVec`'s forwarding method passes the very expressions `prod(s)`, `sqrt(sum(s[i]^2))` the
old body computed; `prf` of the equal-cell FrqSet is untouched (the prod(b)/V_t factor lives in FrqSetX only).
D11 (`cutMax` in farSetup): at lambda/32-lambda/4 and the slender cell both nCutVec evaluations are certified
(no -1 in either), so cutMax == max. there and the nCut vectors are equal (row (b)); the 120 tensors and the two
12^3 blocks being bit-identical is the proof that no equal-cell result changed.

Bonus (smoke.jl): the equal-cell pair through the X API, `farTensorX(D .* g, g, g, f)` vs `farTensor(D, g, f)`,
is bitwise identical at D = (4,1,0) (route 1/1) and (2,0,0) (route 2 = octants / route 2 = gcd with N_t = N_s = 1,
i.e. one farTensor call): prf's factor is T(1//1) = 1 and the (0, g) table is the equal-cell table.

## B'. Bit-identity re-proved after the mrgGeo fix (`bitid.jl` re-run 09:27 -> `bitid.txt`; the 08:01 run kept as `bitid_run2_premrg.txt`, the polluted first run as `bitid_run1.txt`; JULIA_NUM_THREADS=1, 1-min load 11.2 at start, 9.9 at the end, acc.jl with 12 threads alongside)

farfield_cross.jl as of 08:28 (with the `mrgGeo` duplicate-column drop of D.0 = D12) against farfield_pre_cross.jl.
Every row of B holds unchanged: (a) 0 mismatches; (b) 120/120 tensors bitwise identical, routes 108/12/0, Lw/Lo
56/56 in both modules, nCut equal, farSetup 2.5/2.1, 1.0/1.0, 1.1/1.0, 1.0/0.9 s old/new; (c) c32 12^3 block 13894
nonzero entries 0 differing (1.7/1.5 s), slender 13952 nonzero 0 differing (0.8/0.8 s); (a') fresh farShape files
byte-identical (0.2/0.3 s).  `diff bitid_run2_premrg.txt bitid.txt`: the two load lines, the timing columns, and
the (d) file list (tab_new/ now also holds the 16 cross-scale pair tables acc.jl built; the four equal-cell files are
byte-identical to the shipped shapetab/ files: md5 01ed86a4.., 9cfc80b1.., b2255a87.., 105f05fe.. on both sides).
The shipped equal-cell tables are multi-segment files read through mrgGeo, so rows (b)-(d) exercise the fixed
merge on well-formed input; the duplicated-file path is attrib3.jl (D.0: the corrupted 602-column file reads as 406
columns and gives 5.12e-16 / 1.13e-15 at (6.5, 1.5, 1.5) g, equal to the fresh build).

## C. Accuracy against the 438-record 220-bit cache (`acc.jl` -> `acc.txt`, `acc_rows.md`; summary `accsum.py` -> `accsum.txt`)

Run: JULIA_NUM_THREADS=12, 08:01-09:29, table dir `tab_new/` (the shipped equal-cell tables plus the pair tables it
built: 29.7 s of farSetupX in total), k-series disk cache `ksr_new/` (36 `ksrx_*` files written, one per ordered pair
and f).  1-min load 2.4 at the start, 4-9 through the f = 1 groups, 19-54 during the 16x16x16 f = 1 swap run and the
first f = 1+0.1i groups (other agents), 6.8 at the end; maxrss 5.56 GB.  Every record is evaluated four ways: `farBlockX!`
on the group's offsets with `cert = true`, `farTensorX` per offset (compared bitwise with the block), and both again in
the swapped orientation (coarse target, fine source, offset -R) for the identity G(R; sT, sS) = (V_s/V_t) G(-R; sS, sT).
acc.jl loaded farfield_cross.jl at 08:01, before the mrgGeo fix of 08:28; its tables lived in memory (the duplicated
tab_new file of D.0 was never re-read by it), and the D.1 block with the fixed file reproduces its values at the shared
offsets ((6.5, 1.5, 1.5) g: 5.1e-16 in both).  Errors are against the 220-bit reference: max-entry = max |G - ref| /
max |ref|, per-entry = worst relative error over the entries above 1e-8 max|ref|; digits lost = log10(per-entry / eps).

Per (pair, f); the pair as sS/g x sT/g with g the gcd cell; "slender" = sT (1/32, 1/32, 1/512):

| f | pair (sS/g; sT/g) | n | routes 1/2/3 | worst max-entry | worst per-entry | digits lost | swap routes 1/2/3 | swap identity worst | swap vs ref worst | farTensorX == farBlockX! | violations |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.37 | 2x1x1; 1x1x1 | 8 | 7/1/0 | 2.7e-15 | 2.7e-15 | 1.1 | 7/1/0 | 0.0e+00 | 2.7e-15 | 8/8 | 0 |
| 0.37 | 4x1x1; 1x1x1 | 8 | 6/2/0 | 2.3e-15 | 2.3e-15 | 1.0 | 6/2/0 | 5.0e-19 | 2.3e-15 | 8/8 | 0 |
| 0.37 | 2x2x2; 1x1x1 | 8 | 6/0/2 | 9.9e-16 | 1.3e-15 | 0.8 | 6/0/2 | 0.0e+00 | 9.9e-16 | 8/8 | 0 |
| 0.37 | 8x1x1; 1x1x1 | 8 | 5/3/0 | 2.1e-15 | 2.1e-15 | 1.0 | 5/3/0 | 3.1e-16 | 2.0e-15 | 8/8 | 0 |
| 0.37 | 16x1x1; 1x1x1 | 8 | 3/5/0 | 2.4e-15 | 2.4e-15 | 1.0 | 3/5/0 | 3.0e-16 | 2.4e-15 | 8/8 | 0 |
| 0.37 | 4x4x4; 1x1x1 | 8 | 4/0/4 | 4.5e-16 | 8.8e-16 | 0.6 | 4/0/4 | 4.6e-38 | 4.5e-16 | 8/8 | 0 |
| 0.37 | 8x8x8; 1x1x1 | 8 | 3/0/5 | 1.5e-15 | 1.6e-15 | 0.9 | 3/0/5 | 8.0e-38 | 1.5e-15 | 8/8 | 0 |
| 0.37 | 16x16x16; 1x1x1 | 8 | 2/0/6 | 4.4e-16 | 6.4e-16 | 0.5 | 2/0/6 | 8.2e-38 | 4.4e-16 | 8/8 | 0 |
| 1 | slender 1x1x8; 1x1x1 | 8 | 3/4/1 | 1.0e-15 | 1.0e-15 | 0.7 | 3/4/1 | 2.1e-16 | 1.0e-15 | 8/8 | 0 |
| 1 | slender 4x4x1; 1x1x1 | 8 | 2/0/6 | 4.1e-16 | 6.3e-16 | 0.5 | 2/0/6 | 4.0e-37 | 4.1e-16 | 8/8 | 0 |
| 1 | 2x1x1; 1x1x1 | 17 | 16/1/0 | 3.2e-15 | 3.2e-15 | 1.2 | 16/1/0 | 4.8e-21 | 3.2e-15 | 17/17 | 0 |
| 1 | 2x2x1; 1x1x1 | 21 | 16/3/2 | 1.5e-15 | 3.2e-15 | 1.2 | 16/3/2 | 1.6e-16 | 1.7e-15 | 21/21 | 0 |
| 1 | 4x1x1; 1x1x1 | 16 | 13/3/0 | 2.9e-15 | 2.9e-15 | 1.1 | 13/3/0 | 2.1e-16 | 2.7e-15 | 16/16 | 0 |
| 1 | 2x2x2; 1x1x1 | 37 | 29/3/5 | 1.3e-15 | 4.7e-15 | 1.3 | 29/3/5 | 2.0e-16 | 1.3e-15 | 37/37 | 0 |
| 1 | 8x1x1; 1x1x1 | 13 | 8/5/0 | 2.6e-15 | 2.6e-15 | 1.1 | 8/5/0 | 1.9e-16 | 2.6e-15 | 13/13 | 0 |
| 1 | 4x4x1; 1x1x1 | 21 | 14/4/3 | 1.2e-15 | 1.3e-15 | 0.8 | 14/4/3 | 2.1e-16 | 1.2e-15 | 21/21 | 0 |
| 1 | 16x1x1; 1x1x1 | 17 | 9/8/0 | 2.7e-15 | 2.7e-15 | 1.1 | 9/8/0 | 3.5e-16 | 2.5e-15 | 17/17 | 0 |
| 1 | 4x4x4; 1x1x1 | 38 | 22/6/10 | 1.5e-15 | 2.1e-15 | 1.0 | 22/6/10 | 4.1e-16 | 1.5e-15 | 38/38 | 0 |
| 1 | 8x8x1; 1x1x1 | 21 | 10/6/5 | 1.2e-15 | 1.6e-15 | 0.9 | 10/6/5 | 5.4e-16 | 1.2e-15 | 21/21 | 0 |
| 1 | 16x16x1; 1x1x1 | 21 | 6/9/6 | 2.5e-15 | 2.9e-15 | 1.1 | 6/9/6 | 1.8e-15 | 7.2e-16 | 21/21 | 0 |
| 1 | 8x8x8; 1x1x1 | 34 | 14/7/13 | 1.2e-15 | 1.4e-15 | 0.8 | 14/7/13 | 1.5e-15 | 9.5e-16 | 34/34 | 0 |
| 1 | 16x16x16; 1x1x1 | 38 | 11/10/17 | 3.4e-15 | 4.7e-15 | 1.3 | 11/10/17 | 4.9e-15 | 5.1e-15 | 38/38 | 0 |
| 1+0.1i | 2x1x1; 1x1x1 | 8 | 7/1/0 | 6.1e-16 | 1.6e-15 | 0.9 | 7/1/0 | 1.6e-17 | 6.1e-16 | 8/8 | 0 |
| 1+0.1i | 4x1x1; 1x1x1 | 8 | 6/2/0 | 1.0e-15 | 1.2e-15 | 0.7 | 6/2/0 | 1.4e-16 | 1.0e-15 | 8/8 | 0 |
| 1+0.1i | 2x2x2; 1x1x1 | 8 | 6/0/2 | 7.5e-16 | 1.2e-15 | 0.7 | 6/0/2 | 1.5e-38 | 7.5e-16 | 8/8 | 0 |
| 1+0.1i | 8x1x1; 1x1x1 | 8 | 5/3/0 | 1.2e-15 | 1.5e-15 | 0.8 | 5/3/0 | 2.0e-16 | 1.2e-15 | 8/8 | 0 |
| 1+0.1i | 16x1x1; 1x1x1 | 8 | 3/5/0 | 1.4e-15 | 1.4e-15 | 0.8 | 3/5/0 | 2.7e-16 | 1.4e-15 | 8/8 | 0 |
| 1+0.1i | 4x4x4; 1x1x1 | 8 | 4/0/4 | 6.9e-16 | 6.9e-16 | 0.5 | 4/0/4 | 2.8e-38 | 6.9e-16 | 8/8 | 0 |
| 1+0.1i | 8x8x8; 1x1x1 | 8 | 3/0/5 | 9.0e-16 | 9.8e-16 | 0.6 | 3/0/5 | 7.8e-38 | 9.0e-16 | 8/8 | 0 |
| 1+0.1i | 16x16x16; 1x1x1 | 8 | 1/0/7 | 1.7e-16 | 6.5e-16 | 0.5 | 1/0/7 | 2.6e-36 | 1.7e-16 | 8/8 | 0 |

Per f and per route over all 438 records:

| f | records | routes 1/2/3 | worst max-entry | worst per-entry | digits lost | swap identity worst | violations |
|---|---|---|---|---|---|---|---|
| 0.37 | 64 | 36/11/17 | 2.7e-15 | 2.7e-15 | 1.1 | 3.1e-16 | 0 |
| 1 | 310 | 173/69/68 | 3.4e-15 | 4.7e-15 | 1.3 | 4.9e-15 | 0 |
| 1+0.1i | 64 | 35/11/18 | 1.4e-15 | 1.6e-15 | 0.9 | 2.7e-16 | 0 |

| route | records | max-entry: median / worst | per-entry: median / worst | digits lost worst | cert/max\|G\|: min / median / max (above tol) | cert / actual abs error: min / median / max | violations | swap identity worst (exactly 0) |
|---|---|---|---|---|---|---|---|---|
| 1 whole box | 244 | 4.9e-16 / 2.4e-15 | 6.9e-16 / 4.7e-15 | 1.3 | 4.6e-15 / 1.3e-14 / 3.5e-14 (143) | 2.4 / 23.9 / 135.3 | 0 | 0 (244 of 244) |
| 2 gcd box sum | 91 | 5.8e-16 / 3.4e-15 | 7.8e-16 / 4.7e-15 | 1.3 | 4.9e-15 / 1.6e-14 / 3.2e-14 (85) | 4.0 / 29.3 / 673.4 | 0 | 4.9e-15 (2 of 91) |
| 3 k-series | 103 | 4.2e-17 / 9.7e-17 | 6.0e-17 / 9.7e-17 | -0.4 | 4.6e-15 / 4.8e-15 / 8.1e-13 (4) | 48.7 / 115.2 / 8930.4 | 0 | 2.6e-36 (9 of 103) |

Worst records: route 1 max-entry 2.40e-15 at 16x1x1, f = 0.37, R/g = (24.5, 0, 0), L = 34 (cert/max|G| 7.8e-15,
cert 3.2x the error; xtable S3d has the same record, r16x x16c, at 2.4e-15 and L 34); route 1 per-entry 4.70e-15 at
2x2x2, f = 1, (5.5, 0.5, 0.5), L = 34; route 2 max-entry 3.40e-15 (per-entry 4.60e-15) at 16x16x16, f = 1,
(12.5, 12.5, 7.5), a 4096-term box sum (cert 4.0x the error, the tightest certificate of the run); route 2 per-entry
4.70e-15 at (9.5, 7.5, 7.5); route 3 9.70e-17 at 4x4x4, f = 1, (4.5, 0, 0).  Overall worst max-entry 3.4e-15, worst
per-entry 4.7e-15 = 1.3 digits, 0 certificate violations (cs < actual error) among the 438 `cert = true` evaluations
recorded in acc_rows.md (the VIOLATION tag never fired; no separate certificate pass was needed).  `farTensorX` is
bitwise equal to `farBlockX!` on 438 of 438 records and returns the same certificate value on 438 of 438; the route
kind of the swapped orientation equals the direct one on 438 of 438.  Consistency with xtable: S3b's whole-box worst
1.4e-15 (f = 1), S3c's 1.4e-15 (r16x x16c, f = 1+0.1i) and S3d's 2.4e-15 (r16x x16c, f = 0.37) are the route-1 worsts
here per f (2.4e-15 at f = 0.37 is that record; at f = 1 the route-1 worst is 2.9e-15 at 4x1x1 (5.5, 0, 0)-class
offsets that S3b listed under the fixed rule with the same L); S3d's gcd worst 2.7e-15 (r2x x1c) is the route-2
record (2.5, 0, 0) of 2x1x1 at f = 0.37 here, 2.7e-15.  S3b's gcd worst 5.1e-15 at r16xyz x8v (16.5, 7.5, 7.5) is
2.2e-15 here (the box sum adds the 4096 pieces in a different order from farx.jl's per-offset gcd average; xtable's
own farBlockX! run, S4b, gave <= 3.5e-15 on the same 13 references, as D.2 does).

Swap identity.  Route 1 is exactly 0 on all 244 records: the table is keyed on the unordered pair, prf carries
prod(b)/V_t, and (V_s/V_t) G(-R; sS, sT) reproduces every bit of G(R; sT, sS).  Route 2 differs by up to 4.9e-15
(15 of 91 above 1e-15, all at 8x8x8 and 16x16x16 f = 1: the box sum over the fine egoToe visits the N_t N_s pieces in
the reflected order, and (V_s/V_t) is a power of two only up to the /N_t weight; the swapped value is itself within
5.1e-15 of the reference).  Route 3 differs by 1e-38 .. 2.6e-36: the 128-bit k-series is assembled in the other
orientation's face order and divided by the other V_t before the Float64 rounding.

Route 3 (k-series) usage: 103 of 438 records, every one of them because the pair is OFF the gcd lattice, none because
a sub-offset would touch (xNear < 2 never fired: the matrix starts at |R| = (r + 3)/2 g, where the nearest sub-offset
has max-norm 2).  The lattice coordinate is m = R/g + (nS - nT)/2; on an axis where the source cell is an even
multiple of g (nS - nT odd) a lateral coordinate 0 of R/g gives a half-integer m.  So the {x, y, z} and {x, y} sets'
lateral-0 records (the `x?c`/`d?c` offsets: (k + 1/2, 0, 0), (k + 1/2, k' + 1/2, 0), (k + 1/2, 1.5, 0) etc. with a 0 on an
even axis) and the slender pairs' lateral-0 records go to route 3; the lateral (1.5, 1.5), (3.5, 3.5), (7.5, 7.5)
records (`x?v`, `d?e`) are on the lattice and take route 2.  The {x} pairs (nS - nT = (r - 1, 0, 0)) are never off the
lattice.  Which records, their k-series order N (from ksrBnd at tol/Lambda) and Lambda = sum I_{-1}/(4 pi |f|^2 V_t)/est_X:

| f | pair | route-3 records | R/g | N | Lambda | worst max-entry |
|---|---|---|---|---|---|---|
| 0.37 | 2x2x2 | 2 of 8 | 2.5,0,0; 3.5,0,0 | 13-14 | 31-57 | 1.2e-17 |
| 0.37 | 4x4x4 | 4 of 8 | 3.5 .. 6.5 on x | 15-17 | 28-82 | 6.5e-17 |
| 0.37 | 8x8x8 | 5 of 8 | 5.5 .. 10.5 on x | 18-20 | 28-84 | 8.7e-17 |
| 0.37 | 16x16x16 | 6 of 8 | 9.5 .. 16.5 on x | 22-25 | 32-74 | 9.6e-17 |
| 1 | slender 1x1x8 | 1 of 8 | 2,0,0 (the slender x offset; m_z = 3.5) | 17 | 270 | 5.9e-17 |
| 1 | slender 4x4x1 | 6 of 8 | 0,0,2; 0,0,3; 0,0,5; 0,0,9 (thin axis); 3.5,0,0; 4.5,0,0 | 16-23 | 0.61-3900 | 9.0e-17 |
| 1 | 2x2x1 | 2 of 21 | 2.5,0,0; 3.5,0,0 | 18-19 | 32-54 | 4.4e-17 |
| 1 | 2x2x2 | 5 of 37 | 2.5,0,0; 3.5,0,0; 2.5,0.5,0; 3.5,0.5,0; 2.5,2.5,0 | 18-20 | 24-40 | 6.8e-17 |
| 1 | 4x4x1 | 3 of 21 | 3.5, 4.5, 5.5 on x | 21-23 | 42-75 | 6.8e-17 |
| 1 | 4x4x4 | 10 of 38 | 3.5 .. 6.5 on x; (3.5..6.5, 1.5, 0); (3.5,3.5,0); (4.5,4.5,0) | 21-25 | 19-46 | 9.7e-17 |
| 1 | 8x8x1 | 5 of 21 | 5.5 .. 10.5 on x | 25-30 | 66-130 | 5.7e-17 |
| 1 | 16x16x1 | 6 of 21 | 9.5 .. 16.5 on x | 33-38 | 110-160 | 6.4e-17 |
| 1 | 8x8x8 | 13 of 34 | 5.5 .. 10.5 on x; (5.5..10.5, 3.5, 0); (5.5,5.5,0); (6.5,6.5,0); (8.5,8.5,0) | 26-32 | 17-42 | 9.5e-17 |
| 1 | 16x16x16 | 17 of 38 | 9.5 .. 24.5 on x; (9.5..16.5, 7.5, 0); (9.5,9.5,0) .. (16.5,16.5,0) | 34-44 | 15-33 | 8.1e-17 |
| 1+0.1i | 2x2x2 | 2 of 8 | 2.5,0,0; 3.5,0,0 | 18-19 | 25-42 | 7.5e-17 |
| 1+0.1i | 4x4x4 | 4 of 8 | 3.5 .. 6.5 on x | 21-24 | 20-51 | 7.4e-17 |
| 1+0.1i | 8x8x8 | 5 of 8 | 5.5 .. 10.5 on x | 26-30 | 19-45 | 5.3e-17 |
| 1+0.1i | 16x16x16 | 7 of 8 | 9.5 .. 24.5 on x | 34-44 | 18-53 | 7.9e-17 |

Route 3 delivers the reference to 1e-17 (Float64 rounding of a 128-bit value; -0.4 digits lost).  Its certificate is
KSRTOL est_X + 20 eps max|G| = 4.6e-15 .. 5.0e-15 of max|G| on 99 records; the four exceptions are the slender
4x4x1 pair's thin-axis records (0, 0, n) g_z at f = 1: cert/max|G| 1.5e-14 (n = 9), 5.9e-14 (5), 2.5e-13 (3),
8.1e-13 (2), i.e. est_X over-states max|G| by 100x, 550x, 2500x, 8100x there (the tensor across the thin axis of a
1/32 x 1/32 x 1/512 cell at 2-9 g_z separation is 1e2-1e4 below the pointwise scale; Lambda 0.61-43); the actual
error is 2.3e-17 .. 9.0e-17, the certificate 2600-8900x the error: loose, valid.

Route 3 cost (BigFloat 128, 36 face pairs, threaded over the route-3 offsets of a call):

| f | pair | route-3 records | k-series s (12 threads) | s per record | swap-run farBlockX! s | 1-min load at group end |
|---|---|---|---|---|---|---|
| 0.37 | 2x2x2 | 2 | 29.1 | 14.5 | 27.9 | 4.5 |
| 0.37 | 4x4x4 | 4 | 30.3 | 7.6 | 30.6 | 5.2 |
| 0.37 | 8x8x8 | 5 | 30.9 | 6.2 | 31.1 | 6.2 |
| 0.37 | 16x16x16 | 6 | 39.0 | 6.5 | 39.0 | 6.4 |
| 1 | slender 1x1x8 | 1 | 14.2 | 14.2 | 15.0 | 5.1 |
| 1 | slender 4x4x1 | 6 | 38.9 | 6.5 | 38.5 | 6.9 |
| 1 | 2x2x1 | 2 | 27.8 | 13.9 | 28.5 | 5.1 |
| 1 | 2x2x2 | 5 | 65.2 | 13.0 | 67.1 | 5.1 |
| 1 | 4x4x1 | 3 | 35.3 | 11.8 | 36.9 | 5.0 |
| 1 | 4x4x4 | 10 | 84.5 | 8.5 | 87.0 | 8.8 |
| 1 | 8x8x1 | 5 | 56.7 | 11.3 | 57.5 | 6.4 |
| 1 | 16x16x1 | 6 | 103.7 | 17.3 | 106.1 | 6.1 |
| 1 | 8x8x8 | 13 | 180.0 | 13.8 | 182.2 | 5.9 |
| 1 | 16x16x16 | 17 | 361.2 | 21.2 | 3007.4 | 35.5 |
| 1+0.1i | 2x2x2 | 2 | 4.8 | 2.4 | 55.6 | 31.6 |
| 1+0.1i | 4x4x4 | 4 | 9.8 | 2.5 | 14.3 | 54.5 |
| 1+0.1i | 8x8x8 | 5 | 59.3 | 11.9 | 52.1 | 19.3 |
| 1+0.1i | 16x16x16 | 7 | 96.4 | 13.8 | 88.3 | 6.8 |

6-21 s per route-3 offset (N = 13-44; the per-record time also carries the thread imbalance of a 1-17-offset batch on
12 threads, hence 14.5 s for the 2-record groups), against 3 us for a whole box and 10-4000 us for a box sum: route 3
is the fallback for the off-lattice positions only, which Gila's block geometry does not produce (xgeom: every
offset of a Gila pair is on the half-integer gcd lattice; the D.2 block has 0 route-3 offsets).  The 16x16x16 f = 1
swap run took 3007 s against 361 s for the direct run with the same 17 offsets and the same N (the 1-min load was 35
then, 6 before; other agents' processes), and the 2x2x2 f = 1+0.1i swap took 55.6 s against 4.8 s at load 32: both
are load, not code (the direct/swap pairs at load 5-7 agree to within 5%).  The route-3 results were also written to
`ksr_new/ksrx_<sT>_<sS>_f<re>_<im>.txt` (0.4-9.9 kB per pair and orientation), so a repeated call reads them.

## E. Table build cost per pair (`build.sh` -> `build.jl` per pair -> `build.txt`; one fresh Julia process per pair, fresh table dir `tab_e/<pair>/`, JULIA_NUM_THREADS=1, f = 1, load 3.5-5.5 (acc.jl was running with 12 threads alongside))

Cold = `farSetupX(g, sS, f; offs = the pair's matrix offsets)` in a process that has only loaded the file: Theorem A'
thresholds + n-cut (the "warm" column, 0.07-0.34 s, is this selection arithmetic plus the frqBox contraction) + the
BLDPRC = 256 build of the whole trapezoid box to the L the offsets ask for + the disk write.  maxrss at the end
(start = after loading the file).  The equal-cell g reference in the same conditions: whole box alone to L = 56,
nMax 12: 4.0 s, maxrss 0.49 GB (start 0.27 GB).

| pair (sS/g; sT = g unless stated) | table Lw | nCut max (table nMax) | cold s | warm s | maxrss GB | file MB |
|---|---|---|---|---|---|---|
| 2x1x1 | 56 | 7 (12) | 5.1 | 0.13 | 0.56 | 2.5 |
| 2x2x1 | 40 | 7 (12) | 3.3 | 0.09 | 0.52 | 1.1 |
| 2x2x2 | 46 | 7 (12) | 3.9 | 0.10 | 0.53 | 1.4 |
| 4x1x1 | 46 | 7 (12) | 3.9 | 0.10 | 0.53 | 1.7 |
| 4x4x1 | 50 | 8 (12) | 4.2 | 0.10 | 0.54 | 1.6 |
| 4x4x4 | 54 | 8 (12) | 4.6 | 0.11 | 0.56 | 1.9 |
| 8x1x1 | 54 | 9 (12) | 4.9 | 0.13 | 0.55 | 2.4 |
| 8x8x1 | 54 | 9 (12) | 4.9 | 0.11 | 0.56 | 1.9 |
| 8x8x8 | 56 | 10 (12) | 5.1 | 0.12 | 0.57 | 2.0 |
| 16x1x1 | 54 | 11 (12) | 5.0 | 0.12 | 0.56 | 2.4 |
| 16x16x1 | 50 | 12 (12) | 4.3 | 0.11 | 0.54 | 1.6 |
| 16x16x16 | 54 | 13 (13) | 5.7 | 0.34 | 0.63 | 2.0 |
| slender z: sT (1/32,1/32,1/512), sS (1/32,1/32,1/64) | 24 (= LDEF; no matrix offset takes route 1) | 6 (12) | 2.9 | 0.07 | 0.50 | 0.4 |
| slender x: sT (1/32,1/32,1/512), sS (1/8,1/8,1/512) | 50 | 8 (12) | 4.6 | 0.11 | 0.54 | 1.6 |

The trapezoid whole-box table costs the same as the equal-cell whole box at the same L (3.3-5.7 s against 4.0 s to
L = 56), +0.2-0.3 GB over the loaded process; the only pair whose n-cut exceeds the NMAX = 12 floor is the lambda/2
cube (nCut max 13 -> table nMax 13, file `..._n13_...`; the nCap recomputation path of D11 ran there).  The files are
0.4-2.5 MB (whole box only: even l, four (l,m) classes) against 19.5 MB for an equal-cell file (whole box + octant
table to 56).  The equal-cell set of g that the near band needs is the shipped table (read, not built).

## D. farBlockX! on blocks (`block.jl r4|big` -> `block_<which>_t<threads>.txt`; f = 1; table dir `tab_d/` = a copy of the shipped equal-cell tables, private to these runs)

### D.1 xgeom's r = 4 xyz test block: fine g target vs coarse (4g)^3 source, R = ((2k+1)/64)_{k}, k_x in 2..13, k_y, k_z in -6..5, the 36 touching offsets (|R_i| <= 5/64 on every axis) excluded: 1692 offsets; JULIA_NUM_THREADS=1, load 8.4-9.4 (acc.jl with 12 threads and other agents alongside)

| run | total s | setup | route | fine egoToe (farBlock! of g) | k-series | fill (whole + box sum) | maxrss GB |
|---|---|---|---|---|---|---|---|
| cold (process had loaded the file; table of the pair built into tab_d, fine set built) | 8.91 | 3.28 | 0.21 | 2.33 | 0 | 0.16 | 0.72 |
| warm (tables and fine set in memory; the fine egoToe is refilled per call) | 1.21 | 0.13 | 0.00 | 1.07 | 0 | 0.01 | 0.77 |
| warm, cert = true | 1.29 | 0.13 | 0.00 | 1.14 (incl. the per-entry certificates certEq) | 0 | 0.02 | 0.79 |

The same block with JULIA_NUM_THREADS=12 (`block_r4_t12.txt`, run 09:31 after acc.jl had exited; 1-min load
4.9-5.3 with this 12-thread process included, 5-min 12.3-12.6 from the earlier agents; the pair's table was on disk in
tab_d from the single-thread run, so "cold" here is a cold process reading it, not building it):

| run | total s | setup | route | fine egoToe | k-series | fill (whole + box sum, wall) | maxrss GB |
|---|---|---|---|---|---|---|---|
| cold (process loaded the file; table read from tab_d; fine set built) | 4.86 | 0.33 | 0.10 | 1.83 | 0 | 0.11 | 0.67 |
| warm | 0.99 | 0.13 | 0.00 | 0.86 | 0 | 0.00 | 0.71 |
| warm, cert = true | 1.09 | 0.13 | 0.00 | 0.96 | 0 | 0.00 | 0.77 |

Routes, L histogram, certificates (route 1 min/median/max 6.45e-15/1.93e-14/5.15e-14, 1184 above tol; route 2
1.23e-14/2.12e-14/3.22e-14, 340 above tol), the 9 reference errors (worst 8.7e-16 at (3.5, 1.5, 1.5)), 0 violations
and farTensorX == farBlockX! on 1692 of 1692: identical to the single-thread run.  The block is dominated by the
fine phase (0.86 of 0.99 s), which the threads do not shorten (1.07 s single): it is fineSet's serial farSetup(g)
of the equal-cell set, rebuilt per call because block.jl passes no `fs` (the 12^3 fill itself is ~1 us per offset,
`fine_t.jl` for the 33^3 case).

Routes 1/2/3 = 1352/340/0 (route 2 = every non-touching offset whose whole box does not converge; all on the gcd
lattice as xgeom Section 1 says).  Whole-box L histogram: 24:148 26:220 28:176 30:132 32:84 34:116 36:72 38:68
40:64 42:32 44:72 46:44 48:24 50:28 52:56 54:16.  Certificates cert/max|G|: route 1 min 6.5e-15, median 1.9e-14,
max 5.2e-14 (1184 of 1352 above tol = 1e-14, i.e. est/max|G| = 1.5-5 as for equal cells: the rule certifies
tol est, not tol max|G|); route 2 min 1.2e-14, median 2.1e-14, max 3.2e-14 (340 of 340 above tol).
`farTensorX` (per offset, gcd route as farTensor calls) is bitwise equal to `farBlockX!` (box sum) on 1692 of 1692
offsets: same fine tensors, same weights, same summation order.

Against the 9 references of the cache that lie in this block (r4xyz corner-corner lateral (1.5, 1.5) g, x-facing
kap 1..8 and diagonal kap 1, 2, and the mirror (1.5, -1.5)):

| R/g | route | err max-entry, per-entry | digits lost | cert/max|G| | cert / actual abs error |
|---|---|---|---|---|---|
| 3.5,1.5,-1.5 | 2 | 4.6e-16, 5.8e-16 | 0.3 | 2.6e-14 | 55.6 |
| 3.5,1.5,1.5 | 2 | 8.7e-16, 8.7e-16 | 0.6 | 2.6e-14 | 29.3 |
| 4.5,1.5,1.5 | 2 | 2.4e-16, 5.8e-16 | 0.0 | 1.8e-14 | 76.7 |
| 5.5,1.5,1.5 | 2 | 1.5e-16, 3.7e-16 | -0.2 | 1.4e-14 | 98.6 |
| 6.5,1.5,1.5 | 1 (L 54) | 5.1e-16, 1.1e-15 | 0.4 | 3.4e-14 | 66.1 |
| 8.5,1.5,1.5 | 1 (L 38) | 4.2e-16, 1.3e-15 | 0.3 | 1.7e-14 | 41.8 |
| 10.5,1.5,1.5 | 1 (L 30) | 3.1e-16, 1.2e-15 | 0.1 | 2.8e-14 | 90.7 |
| 3.5,3.5,1.5 | 2 | 2.6e-16, 4.5e-16 | 0.1 | 2.6e-14 | 97.7 |
| 4.5,4.5,1.5 | 2 | 3.0e-16, 4.3e-16 | 0.1 | 2.1e-14 | 68.9 |

Worst 8.7e-16 max-entry, certificate violations 0 (the certificate is 29-99x the actual error).

### D.0 A defect found on the way, fixed: the table cache was not multi-process safe (`attrib.jl`, `attrib2.jl`, `attrib3.jl` -> `attrib*.out`; evidence file `tab_corrupt/s5_64-5_64-5_64_L0_n12_p192_a3_64-3_64-3_64_seg.txt`)

The first two r4 runs shared `tab_new/` with the running acc.jl and reported 4.68e-13 (per entry 7.25e-12) at
(6.5, 1.5, 1.5) g, route 1, L = 54, certificate 3.4e-14 max|G| -- a certificate violation by 14x, while acc.jl's
own value at the same offset with the same L and the same nCut was 5.1e-16.  Attribution (attrib.jl): the n-cut is
not it (uniform N = 6, 8, 10, 12 on that table: 4.68e-13 each; L = 50: 1.0e-15, L = 52: 6.6e-16, L = 54: 4.68e-13);
a freshly built nMax-16 table with the same nCut: 5.12e-16.  attrib2.jl: the n12 file in tab_new had 602 columns of
which 406 distinct, l = 40..54 present twice: acc.jl had built the pair's table to L = 38 (its f = 0.37 group), the
r4 run read that file and extended it to 54 (appending), and acc.jl's f = 1 group, holding L = 38 in memory,
extended it to 54 again and appended a second copy; a later reader concatenated both (mrgGeo) and summed the
duplicated shells (boxAcc breaks at the first l > L, so only L = 54 reaches the second copy: 4.7e-13 = the size of
shells 40..54).  Fresh n12 builds (0..54; 0..38 then 40..54; the two-segment file re-read) are bitwise identical to
each other and give 5.12e-16, and their columns equal the n16 table's n <= 12 rows bitwise.  This is the shipped
farShape's lazy-append design (append under SHPLK, which is per process; no re-read before appending): any two
Julia processes extending the same table file produce it.  Fix in farfield_cross.jl (`mrgGeo`, lines 405-416): a
segment's columns whose (l, m) index is already present are dropped when merging, so a duplicated file is read
correctly (attrib3.jl: the corrupted file now reads as 406 columns and gives 5.12e-16 / 1.13e-15 at that offset); a
well-formed file has no duplicates and is read exactly as before (bit-identity re-proved in B' below).  The writer
still appends without a lock (a duplicated segment is now harmless, only disk space).

### D.2 The realistic block of xtable Section 4: coarse 4^3 cells of 16 g (lambda/2) against fine 64^3 cells of g, sharing a face; R = ((2k+1)/64), k_x in 8..119, k_y, k_z in -56..55: 1,404,928 offsets, 324 touching excluded -> 1,404,604; f = 1

JULIA_NUM_THREADS=1, load 6.2-8.1 (acc.jl with 12 threads and other agents alongside; the numbers are upper bounds):

| run | total s | setup (farSetupX: thresholds, n-cut, table) | route (1.4e6 rational -> Float64, thresholds, lattice test on the band) | fine egoToe 33^3 (farBlock! of g) | fill = whole (1,387,696 expansions) + box sum (16,908 x 4096 adds) | maxrss GB |
|---|---|---|---|---|---|---|
| cold (file loaded; the pair's table built, the fine set built) | 17.92 | 6.53 | 1.60 | 2.81 | 4.50 = 4.18 + 0.15 | 2.95 |
| warm | 9.81 | 2.71 | 1.45 | 1.04 | 4.60 = 4.41 + 0.16 | 2.95 |
| warm, cert = true | 17.63 | 3.01 | 1.29 | 1.70 (with certEq per fine entry) | 11.60 = 11.42 + 0.15 (bndWhlX per whole-box offset: hbndS is O(L^2), 5 us each) | 3.08 |

JULIA_NUM_THREADS=12 (`block_big_t12.txt`, run 09:30-09:31 after acc.jl had exited, the only Julia process on the
machine; 1-min load 5.5-5.8 with this 12-thread process included, 5-min 14.4-15.3 left by the earlier agents; the
n13 pair table and the fine tables on disk in tab_d from the single-thread run).  `whole` and `box sum` are the sums
of the per-thread times (CPU-seconds), `fill` is the wall time of that phase:

| run | total s | setup (farSetupX, serial) | route (threaded) | fine egoToe 33^3 (farBlock! of g) | fill wall = whole CPU-s + box sum CPU-s | maxrss GB |
|---|---|---|---|---|---|---|
| cold (file loaded; the pair's table read from disk; fine set built) | 10.71 | 3.53 | 1.05 | 2.18 | 1.32 = 13.51 + 0.26 | 2.53 |
| warm | 5.69 | 2.94 | 0.67 | 1.08 | 1.00 = 10.99 + 0.26 | 2.72 |
| warm, cert = true | 7.13 | 2.73 | 0.84 | 1.28 | 2.24 = 25.82 + 0.30 | 2.87 |

Routes 1/2/3 = 1,387,696 / 16,908 / 0, the L histogram, the certificate statistics (route 1 min/median/max
4.70e-15/7.96e-15/4.12e-14, 516,872 above tol; route 2 8.99e-15/1.16e-14/2.95e-14, 14,196 above tol) and the 13
reference errors (worst 3.4e-15 at (12.5, 12.5, 7.5), 0 violations) are identical to the single-thread run.  The
fill goes from 4.60 s to 1.00 s wall (4.6x; its CPU-seconds rise from 4.41 to 10.99, 2.5x: the whole-box expansion
is memory-bound on the table's four (l, m) classes), the certified fill from 11.60 to 2.24 s (5.2x); the route pass
from 1.45 to 0.67 s.  What does not shrink is serial: the farSetupX selection pass (2.71 -> 2.94 s; the 1.4e6
Rational{BigInt} -> Float64 conversions, thresholds and the n-cut certification, 2.1 us each) and the fine egoToe
(1.04 -> 1.08 s: `fine_t.jl` splits it into fineSet's serial farSetup(g; nBlk = 33), 0.81-0.83 s on 12 threads /
0.90-0.95 s on 1, and farBlock!'s threaded fill, 0.01 s on 12 threads / 0.03-0.04 s on 1), so the warm 12-thread
total is 5.69 s = 4.0 s serial + 1.7 s threaded.  xtable S4b's 2.19 s on 12 threads for farx.jl's farBlockX! on this block had a 0.52 s setup pass (its
selection did 1.4e6 bndLw + 13 whlThr levels without the per-offset n-cut certification) and the same 1.0 s fine
block; the 3 us whole box and the 1.0 s fill are the same code.  A caller holding a FrqSetX (`fs = fx`) skips the
setup pass and gets 2.7 s warm.

Routes 1/2/3 = 1,387,696 / 16,908 / 0 (xtable's census: 1,387,696 whole-box, 16,908 near band; identical).
Whole-box L histogram: 20:13636 22:539332 24:319500 26:205048 28:118496 30:64400 32:38236 34:24708 36:16548 38:11988
40:8652 42:6520 44:5112 46:4020 48:3372 50:2500 52:2216 54:1868 56:1544 (xtable's, identical).  Per offset: whole box
3.2 us (4.41 s / 1.39e6), box sum 9.5 us (0.16 s / 16908; 4096 weighted adds each), fine egoToe 1.0 s per call (measured split, `fine_t.jl`, 1 thread, load 3.0-3.2: fineSet's `farSetup(g;
nBlk = 33)` 0.90-0.95 s -- thresholds, n-cut, route selection of the equal-cell set, serial, redone per call because
block.jl passes no `fs` -- and farBlock!'s fill of the 35,937 fine offsets 0.03-0.04 s, 0.7-1.0 us each), the two
selection passes (farSetupX's loop
over the offsets, 1.9 us each, and the route phase, 1.0 us each) 4.2 s together: the exact-rational offsets cost
more than the far field itself; a caller holding a FrqSetX (`fs = fx`) skips the setup pass.  Memory 2.95 GB: G is
3 x 3 x 1.4e6 ComplexF64 (202 MB), the Rs vector 1.4e6 x 3 Rational{BigInt} (~1 GB of BigInt), the tables.

Certificates (cert = true): route 1 cert/max|G| min 4.7e-15, median 8.0e-15, max 4.1e-14, 516,872 of 1,387,696
above tol; route 2 min 9.0e-15, median 1.2e-14, max 3.0e-14, 14,196 of 16,908 above tol (again est/max|G| = 1-4).

Against the 13 cache references inside this block (r16xyz corner-corner lateral (7.5, 7.5) g: x-facing kap 1..8, 16,
32 and the diagonal family kap 1, 2, 4, 8, 16):

| R/g | route | err max-entry, per-entry | digits lost | cert/max|G| | cert / actual error |
|---|---|---|---|---|---|
| 9.5,7.5,7.5 | 2 | 3.2e-15, 4.7e-15 | 1.2 | 2.4e-14 | 7.3 |
| 10.5,7.5,7.5 | 2 | 2.5e-15, 2.5e-15 | 1.1 | 1.9e-14 | 7.6 |
| 11.5,7.5,7.5 | 2 | 1.9e-15, 3.7e-15 | 0.9 | 1.7e-14 | 8.7 |
| 12.5,7.5,7.5 | 2 | 2.4e-15, 3.2e-15 | 1.0 | 1.6e-14 | 6.7 |
| 14.5,7.5,7.5 | 2 | 2.3e-15, 3.2e-15 | 1.0 | 1.4e-14 | 6.1 |
| 16.5,7.5,7.5 | 2 | 2.2e-15, 2.9e-15 | 1.0 | 1.3e-14 | 5.6 |
| 24.5,7.5,7.5 | 1 | 2.8e-16, 9.0e-16 | 0.1 | 1.4e-14 | 49.9 |
| 40.5,7.5,7.5 | 1 | 4.7e-16, 1.9e-15 | 0.3 | 2.1e-14 | 45.6 |
| 9.5,9.5,7.5 | 2 | 3.1e-15, 3.1e-15 | 1.1 | 1.9e-14 | 6.1 |
| 10.5,10.5,7.5 | 2 | 2.2e-15, 3.4e-15 | 1.0 | 1.6e-14 | 7.3 |
| 12.5,12.5,7.5 | 2 | 3.4e-15, 4.6e-15 | 1.2 | 1.4e-14 | 4.0 |
| 16.5,16.5,7.5 | 2 | 2.4e-15, 2.4e-15 | 1.0 | 9.9e-15 | 4.1 |
| 24.5,24.5,7.5 | 1 | 6.5e-16, 6.5e-16 | 0.5 | 9.7e-15 | 14.9 |

Worst 3.4e-15 max-entry (a 4096-piece box sum; 1.2 digits), 0 certificate violations; the box-sum certificate is
4-9x the actual error, the whole-box certificate 15-50x.

## F. The deliverable in place (`sectionF.txt`)

`farfield_cross.jl` copied over `notes/farfield/farfield.jl` at 09:29 (md5 of both now 
identical; 2009 lines; the previous file, md5 b4d74cabf9ec5d6c17cb5dd89c3c37d3, is `farfield_pre_cross.jl`).  From the
repo root, `JULIA_NUM_THREADS=1 julia --startup-file=no --project=notes/farfield/scratch/env notes/farfield/verify.jl a`
ran unmodified (verify.jl part (a) calls wgtT, facePair, momPol, srfSum, none of which changed): 4.9 s of part (a),
7.6 s wall, exit 0, 1-min load 6.9 -> 6.8, acc.jl with 12 threads alongside.  Its output: the rho / rho_oct table, the
slow-band counts (cube 30/134/429/3089 at every N; slender 242..288/682..1192/1417..3618/5458..19736), mu_n exact for 5
lengths x n = 0..24 (0 disagreements), the divergence identity 0//1 on 18 (shape, D, j) cases.  `git -C <repo> diff --stat
notes/farfield/tables`:

    notes/farfield/tables/a_rho.txt | 78 ++++++++++++++++++++++++++++++++++++++++-
    1 file changed, 77 insertions(+), 1 deletion(-)

a_band.txt, a_div.txt, a_mu.txt: byte-identical to the staged files (md5 6188c281.., 1ed5329d.., 62efc5fe.. before and
after, `a_tables.md5.before`).  a_rho.txt: the STAGED file is 4 lines long and ends in the middle of the header row
(`class ` with no newline; `git show :notes/farfield/tables/a_rho.txt | wc -l` = 4) -- it was staged truncated; the
regenerated file has the 81 lines the same header announces, and its first 4 lines equal the staged ones.  The rho
values come from verify.jl's own rhoWhl/rhoOct (not from farfield.jl), so this is not a change of the library's
output; the working-tree file is the complete one and is left in place, unstaged (no git operation was made).
Smoke test `scratch/work/merge/m0_smoke.jl` (JULIA_NUM_THREADS=1, 5.8 s wall): T[1,1] = 0.0015707382212663695 +
0.0003601550792071786im (4.04 s for the first farTensor incl. table read; the root's earlier run in
work/root/mem/smoke_after.txt: the same digits, 3.55 s), Lw 56 Lo 56 nNd 6 nMx 12 nCut [6, 5, 5, 5, 5, 5, 5, 4, 4],
route (5,1,0) = (1, 26, Int64[], 0), cost 1275, isFarInd true/false, stat [492, 12, 0] oct 12 ksr 0, precision
task-local 111/111: every line except the timing identical to the earlier output.

## Defects found and fixed in this round

| id | where | what | fix | evidence |
|---|---|---|---|---|
| D11 | farSetup (equal-cell, farfield.jl :939-942 new) and farSetupX | the two nCutVec evaluations (near radius rNc, far radius rHi) were combined with `max.`, which turns an uncertified -1 at one radius into the other radius' N; and the near radius was taken from octant/piece centres, where the whole-box n-tail bound is vacuous | `cutMax(a, b)`: elementwise max with -1 absorbing, so an uncertified slot propagates and triggers the nCap recomputation; the whole-box n-cut is certified at whole-box radii only (farSetupX: at the nearest route-1 offset of `offs`, else at thr[end]) | xtable S2d/S2e: the 5.9e-14 .. 6.8e-10 whole-box errors at the lambda/4 and lambda/2 coarse cells drop to 3e-16 .. 1.2e-15; here C: route 1 worst 2.4e-15 over 244 records; latent for the shipped shapes (B: nCut vectors unchanged, 120/120 bitwise) |
| D12 | farShape's table cache (`mrgGeo`, farfield.jl :405-416 new) | two processes extending the same table file each append the missing segment (append under the per-process SHPLK, no re-read), and a reader concatenated the duplicated (l, m) columns and summed the shells twice: 4.68e-13 at (6.5, 1.5, 1.5) g, L = 54, a 14x certificate violation | columns already present are dropped when merging segments; a well-formed file is read exactly as before | D.0 / attrib*.jl: the corrupted 602-column file reads as 406 columns, 5.12e-16 = the fresh build; B': bit-identity re-proved with the fix in place |
| -- | farSetupX / farRouteX (design, not a code defect) | est_X under-states max|G| by 2-4x on the r16x pair at f = 1 and 0.37 (xtable S3b/S3d) and over-states it by 100-8100x across the thin axis of the slender 4x4x1 pair (C, route 3) | none; the certificate is tol est_X, valid in both cases (0 violations over 438 + 13 + 9 references) | C route-3 table; D.1/D.2 |

## Open items

1. The table writer still appends without a cross-process lock: a duplicated segment is now harmless on read (D12) but
   costs disk (tab_new's 4x4x4 n12 file is 2.80 MB against 1.88 MB for the clean copy in tab_d).  A lock file or a
   re-read before appending would remove the duplication itself.
2. The exact-rational offset handling costs as much as the far field: on the realistic block the farSetupX selection
   pass (2.1 us per offset) plus the route pass (1.0 us) are 4.2 s single-threaded against 4.4 s of whole-box
   expansions (3.2 us each); on 12 threads the selection pass (serial, 2.9 s) is half the total.  Rs as
   Vector{NTuple{3,Rational{BigInt}}} is ~1 GB for 1.4e6 offsets.  A caller with a FrqSetX in hand (`fs = fx`) skips
   the selection pass; an Int-lattice interface (m and the gcd cell instead of R) would remove both passes.
3. Certificates are tol x est, not tol x max|G|: 143/244 route-1 and 85/91 route-2 records of C, 516,872 of the
   1,387,696 whole-box offsets and 14,196 of the 16,908 box sums of D.2 carry a certificate above 1e-14 max|G|
   (est/max|G| = 1-5 as for equal cells; the actual errors are <= 4.7e-15 per entry everywhere).
4. Route 3 costs 6-21 s per offset at N = 13-44 (BigFloat 128, 36 face pairs); it is reached only off the gcd lattice,
   which Gila's block geometry never produces (D.2: 0 route-3 offsets), so it is a correctness fallback, not a cost.
5. The near band's fine set costs 0.8-0.95 s of serial farSetup(g; nBlk = 33) per farBlockX! call that starts
   without a FrqSetX (the fill of the 33^3 fine egoToe is 0.01-0.04 s; farBlock! threads), as does the farSetupX
   pass; the parallel fraction of a warm 12-thread call is 1.7 of 5.69 s.  Both vanish for a caller that keeps the
   FrqSetX (`fs = fx`): fineSet caches the equal-cell set in fx.fine.
6. 0.13% of the realistic block (rho_w 0.55-0.60, L = 56) is certified only to 1-4 tol (xtable S4b's observation; here
   the D.2 route-1 certificate max is 4.1e-14 max|G|, i.e. within 4.1 tol): LMAX > 56 or routing that band to the box
   sum (fine box 33^3 -> ~41^3) would tighten it.
7. The `tab_new/` cache holds obsolete nMax variants (s5_64 n16, s17_64 n12 next to n13) from the attribution runs and
   the D12 duplicate; tab_e/, tab_attrib*/, tab_corrupt/ are evidence only.  Nothing under xwork/ is needed by the
   deliverable: the pair tables are rebuilt in 3-6 s each (E) into the caller's TABDIR.
8. acc.jl ran the pre-D12 code (loaded 08:01, fix 08:28) with its tables in memory; the D12 read path is covered by
   attrib3.jl, B' and the D.1/D.2 runs, not by acc.jl itself.

## Where I stopped

All items of the resume brief are done: C (438 records, tables above, from the acc.jl run that was still running at
the resume and exited 09:29), B' (bitid.jl re-run 09:27, identical), D.1 and D.2 on 12 threads (09:30-09:32, the only
Julia process on the machine, 1-min load 4.9-5.8 including itself), F (deliverable copied; verify.jl a unmodified,
a_band/a_div/a_mu byte-identical, a_rho.txt's staged copy was truncated at 4 lines and is now the full 81-line table;
m0_smoke.jl identical), the defects and open-items lists.  Not done, outside the brief: no REGISTRY.md entry (the
root's file), no git operation (farfield.jl and a_rho.txt are modified in the working tree, unstaged), no cleanup of
the tab_* caches (open item 7).  Julia processes started by this incarnation: bitid.jl (1 thread), verify.jl a and
m0_smoke.jl (1 thread, sequential), block.jl big and r4 (12 threads, sequential); never more than two at once
including acc.jl.  Files of this round: `accsum.py` -> `accsum.txt`, `sectionF.txt`, `a_tables.md5.before`,
`bitid_run2_premrg.txt`, `block_big_t12.txt`, `block_r4_t12.txt`, `block_big12.log`, `block_r412.log`, `fine_t.jl`
(its output is quoted in D.2; two more single processes, 1 and 12 threads, 09:36-09:37, load 3.0-3.3).
