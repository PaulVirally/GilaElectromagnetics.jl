# xtable -- cross-scale far field: trapezoid whole-box table, near-band routes, cost model

Dir: notes/farfield/scratch/xwork/xtable/ (farx.jl = farfield.jl generalized; scripts listed per section).
Run: JULIA_NUM_THREADS=1 julia --startup-file=no --project=scratch/env <script>.

## 0. Formulas used

Pair (sT, sS): a_i = |sT_i - sS_i|/2, b_i = (sT_i + sS_i)/2.  Trapezoid w_i(t) = (b_i - |t|) - (a_i - |t|)_+,
i.e. the triangle of half-width b minus the triangle of half-width a, so the brief's
  mu_n(a,b) = 2(b-a)a^{n+1}/(n+1) + 2[b(b^{n+1}-a^{n+1})/(n+1) - (b^{n+2}-a^{n+2})/(n+2)]
collapses to
  mu_n(a,b) = 2 (b^{n+2} - a^{n+2}) / ((n+1)(n+2))   (even n; 0 odd)  = wgtT(n,b) - wgtT(n,a).
farx.jl: wgtX(n,a,b) = 2*(b^(n+2) - a^(n+2))/S((n+1)(n+2)); at a = 0 the subtraction of an exact zero
leaves the same BigFloat operations as wgtT, hence bit-identical tables (proved in S5).
Theorem A measures: |w'| flat on the ramps only, lam_n = 2(b^{n+1}-a^{n+1})/(n+1); |w''| point masses
2 b^{2i} + 2 a^{2i} (a = 0: 2 s^{2i} + 2 0^{2i}, the triangle's).  r_d = |b|, rho = |b|/|R|.
Tables stored divided by prod(b); the caller multiplies by prod(b)/V_t (folded into prf).  est uses V_s.
Piece tables (27-split): ramp axis = wgtTrpA(hf, -1, hf) with hf = (b-a)/2 (tabOct's W), centre (a+b)/2;
plateau axis = wgtTrpA(b-a, 0, a), centre 0.  Type = plateau bits per axis (8 tables); a_i = 0 axes
have two ramp pieces only.  Theorem B per axis: aff = 2 al h^{2i+1}/(2i+1), fac = 2 al h^{2i},
ibp = fac + 2|bt| h^{2i+1}/(2i+1), bet = 2|bt| h^{2i} (octant: al = h, bt = -1).
gcd average: G(R; sT, sS) = (1/N_t) sum_j sum_j' G(R + c_j - c'_j'; g, g), evaluated at |D| and reflected.

## 5. Equal-cell paths of farx.jl are bit-identical to farfield.jl (xwork/xtable/bitid.jl, bitid.out)

farx.jl changes to the equal-cell code: tabWhl(s,..) -> tabWhl(0, s, ..) with W = wgtTrpX(0, s); tabOct(s,..) ->
tabPce(0, s, RRR, ..) with hf = (s - 0)/2; farMomW(lTop, s) -> farMomW(lTop, 0, s); farMomO(lTop, hf) ->
farMomO(lTop, hf, (-1,-1,-1), hf); nCutVec(.., s, ..) -> nCutVec(.., prod(s), prod(s), |s|, ..); farShape(s) ->
farShape(0, s, RRR) (cache key (a, b, typ, nMax); file name unchanged when a = 0 and typ = RRR).
Old = notes/farfield/farfield.jl, New = farx.jl loaded as two modules in one process; New reads a copy of the
shipped shapetab/ksrcache files.

| check | result |
|---|---|
| exact-rational wgtX vs brief formula vs direct integral, 5 pairs x n = 0..40 even, odd n = 0, mass sT sS | 0 mismatches (synchk.jl) |
| wgtX(n,0,s) == wgtT(n,s) and lamX(n,0,s) == lamW(n,s) in BigFloat(256), 3 s x n = 0..80 (value, sign, precision) | 0 mismatches |
| tabWhl(L=10) and tabOct(L=7) at BigFloat(256), nMax 12, 4 shipped shapes: every GeoBox field (Ad, Bd, Ao, index lists, cnc) | 0 mismatches |
| farMomW(80), farMomO(72), nCutVec(56) Float64 arrays, 4 shapes | 0 mismatches |
| farTensor at 30 offsets x 4 shapes ((1/32)^3, (1/8)^3, (1/4)^3, (1/32,1/32,1/512)), f = 1, UInt64 bit patterns of re/im of all 9 entries + route kind | 120/120 identical; routes (i)/(ii)/(iii) = 108/12/0 |

Offsets: (2,0,0) (2,1,0) (2,1,1) (2,2,0) (2,2,2) (3,0,0) (3,1,0) (3,2,1) (3,3,3) (4,0,0) (4,1,0) (4,2,2) (4,4,0) (5,1,0) (5,3,2)
(6,0,0) (6,6,6) (7,2,1) (8,0,0) (8,4,2) (8,8,8) (10,1,0) (12,5,3) (16,0,0) (16,16,16) (20,3,1) (24,8,0) (32,0,0) (32,32,32) (40,7,2).

## 5b. A matrix flaw found on the way
For an even ratio r on an axis in the set, the coarse cell's sub-cell centres sit at (j - (r+1)/2) g, i.e. half-integer
multiples of g, so the brief's lateral position 0 ("fine cell aligned with the coarse centre") on such an axis is NOT a
gcd-lattice position (the fine cell straddles a sub-cell boundary): Gila never produces it and the gcd average is
undefined there (tnsGcd! refuses, onLat).  The whole-box and split routes and the references are unaffected.  Affected:
lateral (0,0) for the xy and xyz sets, lateral ((r-1)/2 g, 0) for xyz, diagonal family lat_z = 0 for xyz.  On the x set
every matrix position is on the lattice.  Tables below mark these "offlat" in the gcd columns.

## 1. Whole trapezoid box table: what changed in farx.jl (line map of the copy in xwork/xtable/farx.jl)

- `wgtX(n, a, b)`, `wgtTrpX(a, b, dM)` next to wgtT/wgtTrpA (exact rationals or BigFloat; 0 mismatches vs the
  brief's formula, S5).  `lamX(n, a, b)` is the ramp flat measure.
- `tabWhl(ta, tb, lLo, L, nMax, T, ztl)`: the only change to the body is `W = wgtTrpX(ta[d], tb[d], dM)` and
  `vt = prod(tb)` (stored divided by prod(b)); `tabWhl(s, ...)` forwards with ta = 0.
- `tabPce(ta, tb, typ, ...)`: tabOct's body with per-axis W = ramp `wgtTrpA(hf, -1, hf)`, hf = (b-a)/2, or plateau
  `wgtTrpA(b-a, 0, a)`; vt = prod(b); `tabOct(s, ...)` forwards with ta = 0, typ = RRR.
- `farMomW(lTop, ta, tb)`, `farMomO(lTop, al, bt, hf)`, `nCutVec(lTop, mom, vt, vs, rd, ...)`: Theorem A/B moments
  and the n-cut with the trapezoid measures and the two volumes separated (V_t prefactor, V_s scale, r_d = |b|);
  the old signatures forward with the equal-cell arguments (bit-identical, S5).
- `ShpTab` gains `a` and `typ`; `SHPC` is keyed (a, b, typ, nMax); `farShape(ta, tb, typ; Lw, Lo, ...)` builds the
  whole box in the whl slot (Lw < 0 skips it) and the piece type typ in the oct slot (Lo < 0 skips); `farShape(s; ..)`
  forwards.  Disk format unchanged (segments of (whl, oct) pairs; a skipped slot is the empty segment already
  used when only one table grew).  File name: `shpKey(b, 0, nMax) * xKey(a, typ) * "_seg.txt"` with
  xKey = "" for a = 0 and typ = RRR (existing files keep their names and are read as before), else
  "_a<n>_<d>-<n>_<d>-<n>_<d>_t<rrr|prr|...>", e.g. `s3_64-1_32-1_32_L0_n12_p192_a1_64-0_1-0_1_trrr_seg.txt` for the
  pair (g, (2g, g, g)): whole box + all-ramp piece type in one file, the other piece types in their own files.
  The key is on (a, b), i.e. the UNORDERED pair; the caller supplies prod(b)/V_t (in FrqSetX.prf) and the sign of R.
- New: `BndX`/`bndX` (Float64 selection data), `FrqSetX`/`farSetupX(sTQ, sSQ, f; offs, split, gcd)`, `boundLX`,
  `boundLpce`, `tnsWhlX!`, `tnsPceX!`, `tnsGcd!` (route cache `rtc` on the sub-offsets, `onLat` lattice check),
  `farRouteX`, `farTensorX(RQ, fx)` on an exact rational offset.
- The cross tables: whole-box cnc (worst sum|term|/|acc| of the contraction) 1.1e11 at L = 56, 1.1e8 at L = 40;
  ramp piece types 1.7e11 at L = 50-55; plateau-containing types 2.5e6-2.5e7 at L = 30: the same cancellation
  class as the equal-cell tables (7.3e11 at L = 56), so BLDPRC = 256 / TABPRC = 192 carry over unchanged.

Equal-cell reference build at today's load (eqbuild.jl, fresh dir, fresh process, load 10.1-11.7):
g = (1/32)^3 cold (Lw, Lo) = (24, 24): 4.4 s, maxrss 0.44 GB; whole box alone to L = 56: 3.3 s, 0.51 GB;
octant 0..51 increment: 14.8 s, 0.60 GB.  (mem.md's 14-18 s / 0.7 GB was build + farSetup at a quieter machine;
the octant table is the expensive one, the whole box is cheap.)  The trapezoid whole-box builds are in S1b below.

## 4. Cost model: coarse 4^3 cells of edge 16 g against fine 64^3 cells of edge g = 1/32, sharing a face
(scan.jl -> census.txt, census2.jl -> census2.txt, costsum.jl/costsum2.jl -> costsum*.out; f = 1)

How the pair arises (src read): glaCmpOpr.jl:224-236 `_cmpBlk` sends a TOUCHING pair of regions with different
`scl` to `_sndBlk` (:212-221), which remeshes both to the fine scale and builds an ordinary same-scale external
block (64^3 vs 64^3 at g: (64+64-1)^3 = 2.05e6 equal-cell offsets -- the equal-cell library's job); the unequal
external block `GlaOprVac(trg, src)` with different scl arises for non-touching region pairs and for the direct
constructor (test/crsSclTest.jl:15-17, 41-42, 151-152 are separated cross-scale pairs; :136-137 is touching and goes
through egoFunExtCnt!).  Inside that block (glaVacOprMemGen.jl:198-199 sepGrd, :935-946 grdSel, glaVol.jl:307-331)
the circulant holds every distinct offset, prod_i(nA_i + nB_i - 1) on the common lattice, no reflection reuse
(the self path's egoToeCrc! has no external analogue); with lcm partitions the count is prod_i(dS_i nA_i + dT_i nB_i -
dT_i dS_i) = 112^3 = 1,404,928 for this block (dT = 1, dS = 16, nA = 4, nB = 64), which equals the distinct
offsets R = (k + 1/2) g, k_x in 8..119, k_y, k_z in -56..55.  Per cell pair: |R_i| < 8.5 g + 1e-8 on all axes
-> contact path (324 offsets here); else egoFunOut! with sep = max_i round(|R_i|/16 g) <= 1 -> egoSrfAdp!, else
egoSrfFxd! (glaVacOprMemGen.jl:440-451).

Census of the 1,404,928 offsets (b = 8.5 g on every axis, |b| = 14.72 g; rho_whole = |b|/|R|):

| class | count |
|---|---|
| touching (contact path) | 324 |
| whole-box route converges (Theorem A, tol 1e-14, L <= 56) | 1,387,696 |
| whole-box L histogram | 20: 13636, 22: 539332, 24: 319500, 26: 205048, 28: 118496, 30: 64400, 32: 38236, 34: 24708, 36: 16548, 38: 11988, 40: 8652, 42: 6520, 44: 5112, 46: 4020, 48: 3372, 50: 2500, 52: 2216, 54: 1868, 56: 1544 |
| near band, whole box fails | 16,908 (rho_whole > 0.6: 15,936; the band edge sits at rho 0.58-0.60) |
| 27-piece split converges on the near band | 0 of 2,213 in the sector k_y >= 0, k_z >= k_y (all-plateau piece radius 7.5 sqrt3 g = 13.0 g, ramp-plateau pieces 10.6 g against |R| >= 12.7 g) |
| distinct fine-lattice sub-offsets |D| the gcd average of the whole band needs | 22,949 (max-norm 2..32; routes (i)/(ii)/(iii) = 22941/8/0) |

Measured single-thread costs (load 10.9-11.0, JULIA_NUM_THREADS=1):

| piece of the fill | measured | total for the block |
|---|---|---|
| whole-box route, t_w(L) us: 12:0.88 14:1.19 16:1.59 18:1.38 20:2.13 22:2.44 24:2.35 26:2.88 28:3.72 30:3.77 32:4.33 34:5.04 36:5.28 38:6.34 40:7.64 42:7.61 44:8.25 46:8.99 48:9.91 50:10.83 52:11.19 54:11.77 56:12.53 | sum over the L histogram | 4.15 s |
| near band as a naive gcd average: tnsGcd! 4096 equal-cell evaluations per offset (route cache warm) | 38.9 ms/offset | 16908 x 38.9 ms = 658 s |
| near band as a box sum: farBlock! of shape g on the 33^3 egoToe the band needs (35,929 offsets) | 0.95 s (26 us/offset incl. route decisions) | 0.95 s |
|   + per offset the sum of the 16^3 reflected entries of that block (boxSum!, costsum.jl) | 28.6 us/offset | 0.48 s |
|   box sum vs tnsGcd! at k = (8,-23,-6) | 0.00e+00 (identical) | |
| trapezoid whole-box table for this pair, L = 54 (r16xyz run, S1b) | see S1b | once per (a, b) |

So the unequal block costs ~4.2 s (far, 98.8% of the offsets) + ~1.4 s (near band, 1.2% of the offsets, as a
box sum over the fine equal-cell block that the same library already produces) single-threaded, against
1.4e6 egoSrfFxd! calls today; the naive per-offset gcd average would cost 160x more than the rest of the fill and
is the wrong implementation of the same formula -- the gcd average is a convolution of the fine-lattice equal-cell
egoToe with the coarse-cell indicator (all weights 1/N_t), so it is 16^3 adds per near-band offset, or one FFT.

## 1b. Build table, f = 1 (xwork/xtable/near.jl per shape -> <shape>_f1.txt; times single-threaded, load 8.5-12.6)

Whole-box table at the L the selector asked for over the shape's matrix offsets (kap 1..32, all lateral positions); each piece-type table at the largest L any near-band piece asked for (-1 = no offset of the matrix converges through that type). Time = tabWhl/tabPce at BLDPRC 256 + write; maxrss after the whole-box build alone (the process starts at 0.47-0.49 GB with farx.jl loaded). Equal-cell g reference at the same load: whole box to 56 in 3.3 s / 0.51 GB, octant 0..51 in 14.8 s / 0.60 GB, (24,24) default in 4.4 s / 0.44 GB.

| shape (coarse sS/g) | selector Lw / Lp by type rrr,prr,rpr,ppr,rrp,prp,rpp,ppp | whl build s | maxrss GB after whl | pieces built (type L: s) | pieces total s | maxrss GB end | load |
|---|---|---|---|---|---|---|---|
| r2x | 56 / 51, 30, -1, -1, -1, -1, -1, -1 (12 pieces) | 5.4 (L=56, cnc 1.12e+11) | 0.60 | rrr 51: 13.6; prr 30: 1.4 | 15.0 | 1.14737152 | 8.53857421875 |
| r2xy | 40 / 50, 30, 55, 31, -1, -1, -1, -1 (18 pieces) | 3.5 (L=40, cnc 1.14e+08) | 0.57 | rrr 50: 13.6; prr 30: 1.6; rpr 55: 9.3; ppr 31: 0.9 | 25.4 | 2.924838912 | 10.5712890625 |
| r2xyz | 46 / 50, 30, 51, 31, 55, 31, 56, 32 (27 pieces) | 4.2 (L=46, cnc 3.56e+10) | 0.56 | rrr 50: 13.2; prr 30: 1.4; rpr 51: 7.8; ppr 31: 0.9; rrp 55: 10.4; prp 31: 1.0; rpp 56: 7.1; ppp 32: 0.8 | 42.6 | 1.457979392 | 10.5126953125 |
| r4x | 46 / 51, 46, -1, -1, -1, -1, -1, -1 (12 pieces) | 4.0 (L=46, cnc 1.78e+09) | 0.56 | rrr 51: 14.2; prr 46: 5.6 | 19.8 | 1.279950848 | 10.9599609375 |
| r4xy | 50 / 50, 45, 47, 48, -1, -1, -1, -1 (18 pieces) | 4.5 (L=50, cnc 1.56e+09) | 0.57 | rrr 50: 12.4; prr 45: 4.9; rpr 47: 5.4; ppr 48: 3.7 | 26.4 | 1.133510656 | 10.869140625 |
| r4xyz | 54 / 48, 44, 52, 55, 52, 55, 56, 54 (27 pieces) | 5.2 (L=54, cnc 4.77e+10) | 0.57 | rrr 48: 11.5; prr 44: 4.7; rpr 52: 8.1; ppr 55: 6.0; rrp 52: 9.1; prp 55: 6.3; rpp 56: 6.5; ppp 54: 3.7 | 55.9 | 2.328641536 | 11.11181640625 |
| r8x | 54 / 51, 56, -1, -1, -1, -1, -1, -1 (12 pieces) | 5.3 (L=54, cnc 1.07e+09) | 0.59 | rrr 51: 14.2; prr 56: 10.8 | 25.0 | 3.199926272 | 11.7236328125 |
| r8xy | 54 / 49, 55, 55, 49, -1, -1, -1, -1 (18 pieces) | 5.5 (L=54, cnc 2.51e+09) | 0.61 | rrr 49: 12.8; prr 55: 10.1; rpr 55: 10.3; ppr 49: 4.1 | 37.3 | 2.830286848 | 11.4951171875 |
| r8xyz | 56 / 47, 53, 52, 56, 49, 54, 53, 55 (27 pieces) | 5.6 (L=56, cnc 5.61e+10) | 0.57 | rrr 47: 10.6; prr 53: 8.8; rpr 52: 8.4; ppr 56: 6.1; rrp 49: 6.7; prp 54: 5.5; rpp 53: 5.2; ppp 55: 3.7 | 55.0 | 2.23019008 | 10.71533203125 |
| r16x | 54 / 52, -1, -1, -1, -1, -1, -1, -1 (12 pieces) | 5.3 (L=54, cnc 6.65e+06) | 0.57 | rrr 52: 14.6 | 14.6 | - | 12.6259765625 |
| r16xy | 50 / 49, 56, 42, -1, -1, -1, -1, -1 (18 pieces) | 4.4 (L=50, cnc 2.47e+08) | 0.56 | rrr 49: 12.0; prr 56: 10.9; rpr 42: 4.4 | 27.3 | - | 11.68701171875 |
| r16xyz | 54 / 46, 55, 48, 55, 56, 56, 56, -1 (27 pieces) | 5.5 (L=54, cnc 5.36e+10) | 0.56 | rrr 46: 10.0; prr 55: 10.3; rpr 48: 6.3; ppr 55: 5.9; rrp 56: 9.9; prp 56: 6.1; rpp 56: 6.8 | 55.3 | - | 10.97314453125 |

The trapezoid whole-box table costs the same as the equal-cell one at equal L (3.5-5.6 s to L = 40-56, +0.08-0.12 GB); the whole cross-scale table set of a pair (whole box + 2-8 piece types) is 20-60 s and stays under 0.85 GB. The 'maxrss GB end' column is the whole run incl. the BigFloat read of up to 9 tables and the gcd shape (2-3 GB), not the build.

## 2. Near band, f = 1: kap = 1, 2, 3 (x family) and the diagonal kap = 1, 2 (near.jl, all 12 shapes)

Columns: rho_w = |b|/|R|; L_w Theorem A (-1: L > 56 needed); rho_p worst piece radius/distance over the 12/18/27 pieces; L_p max piece L (Theorem B, budget tol est/n_pieces; -1: some piece needs L > 56); rho_gcd = sqrt3 g/|D_min| of the nearest sub-offset; sub-routes = how many of the N_s equal-cell sub-tensors go through (i)/(ii)/(iii); t in us (tim = repeated calls, >= 0.2 s), load 8.5-12.6; err = (max-entry-normalised, worst per-entry) against the 220-bit reference where one existed at run time; offlat = not a gcd-lattice position (S5b).

| shape | off | R/g | rho_w | L_w | t_w | err_w (max,ent) | rho_p | L_p | t_p | err_p | rho_gcd | sub-routes 1/2/3 | N_s | t_gcd | err_gcd | cross-route agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r2x | x1c | 2.5,0.0,0.0 | 0.825 | -1 | NaN | - | 0.522 | 51 | 122.84 | 3.9e-16,9.2e-16 | 0.866 | 1/1/0 | 2 | 141.8 | 3.2e-15,3.2e-15 |  |
| r2x | x2c | 3.5,0.0,0.0 | 0.589 | 48 | 8.96 | - | 0.333 | 30 | 56.53 | - | 0.577 | 2/0/0 | 2 | 33.0 | - | whole vs gcd: 5.24e-16 (max-entry); split vs gcd: 8.73e-16 (max-entry); whole vs split: 3.51e-16 (max-entry) |
| r2x | x3c | 4.5,0.0,0.0 | 0.458 | 34 | 4.55 | - | 0.243 | 24 | 40.28 | - | 0.433 | 2/0/0 | 2 | 30.7 | - | whole vs gcd: 1.06e-15 (max-entry); split vs gcd: 4.85e-16 (max-entry); whole vs split: 8.82e-16 (max-entry) |
| r2x | d1c | 2.5,2.0,0.0 | 0.644 | 56 | 12.46 | - | 0.397 | 36 | 67.54 | - | 0.612 | 2/0/0 | 2 | 35.1 | - | whole vs gcd: 1.81e-15 (max-entry); split vs gcd: 4.28e-16 (max-entry); whole vs split: 1.60e-15 (max-entry) |
| r2x | d2c | 3.5,3.0,0.0 | 0.447 | 34 | 4.54 | - | 0.243 | 24 | 38.97 | - | 0.408 | 2/0/0 | 2 | 31.9 | - | whole vs gcd: 2.40e-16 (max-entry); split vs gcd: 7.59e-16 (max-entry); whole vs split: 9.11e-16 (max-entry) |
| r2xy | x1c | 2.5,0.0,0.0 | 0.938 | -1 | NaN | - | 0.548 | 55 | 136.14 | - | NaN | offlat | 4 | NaN | - |  |
| r2xy | x1e | 2.5,0.5,0.0 | 0.920 | -1 | NaN | - | 0.522 | 51 | 127.00 | - | 0.866 | 2/2/0 | 4 | 249.3 | - | split vs gcd: 6.33e-16 (max-entry) |
| r2xy | x2c | 3.5,0.0,0.0 | 0.670 | -1 | NaN | - | 0.340 | 31 | 72.71 | - | NaN | offlat | 4 | NaN | - |  |
| r2xy | x2e | 3.5,0.5,0.0 | 0.663 | -1 | NaN | - | 0.333 | 31 | 72.65 | - | 0.577 | 4/0/0 | 4 | 70.8 | - | split vs gcd: 4.90e-16 (max-entry) |
| r2xy | x3c | 4.5,0.0,0.0 | 0.521 | 40 | 5.97 | - | 0.245 | 24 | 51.88 | - | NaN | offlat | 4 | NaN | - | whole vs split: 3.66e-16 (max-entry) |
| r2xy | x3e | 4.5,0.5,0.0 | 0.518 | 40 | 6.55 | - | 0.243 | 24 | 49.63 | - | 0.433 | 4/0/0 | 4 | 55.5 | - | whole vs gcd: 1.13e-15 (max-entry); split vs gcd: 4.24e-16 (max-entry); whole vs split: 9.41e-16 (max-entry) |
| r2xy | d1c | 2.5,2.5,0.0 | 0.663 | -1 | NaN | - | 0.397 | 36 | 67.12 | - | 0.612 | 4/0/0 | 4 | 55.2 | - | split vs gcd: 9.75e-16 (max-entry) |
| r2xy | d2c | 3.5,3.5,0.0 | 0.474 | 36 | 4.63 | - | 0.243 | 24 | 43.45 | - | 0.408 | 4/0/0 | 4 | 40.8 | - | whole vs gcd: 2.76e-16 (max-entry); split vs gcd: 7.27e-16 (max-entry); whole vs split: 9.09e-16 (max-entry) |
| r2xyz | x1c | 2.5,0.0,0.0 | 1.039 | -1 | NaN | - | 0.577 | -1 | NaN | - | NaN | offlat | 8 | NaN | - |  |
| r2xyz | x1e | 2.5,0.5,0.0 | 1.019 | -1 | NaN | - | 0.548 | 56 | 178.54 | - | NaN | offlat | 8 | NaN | - |  |
| r2xyz | x1v | 2.5,0.5,0.5 | 1.000 | -1 | NaN | - | 0.522 | 52 | 185.49 | - | 0.866 | 4/4/0 | 8 | 494.5 | - | split vs gcd: 5.91e-16 (max-entry) |
| r2xyz | x2c | 3.5,0.0,0.0 | 0.742 | -1 | NaN | - | 0.346 | 32 | 101.33 | - | NaN | offlat | 8 | NaN | - |  |
| r2xyz | x2e | 3.5,0.5,0.0 | 0.735 | -1 | NaN | - | 0.340 | 32 | 99.73 | - | NaN | offlat | 8 | NaN | - |  |
| r2xyz | x2v | 3.5,0.5,0.5 | 0.728 | -1 | NaN | - | 0.333 | 31 | 101.14 | - | 0.577 | 8/0/0 | 8 | 144.3 | - | split vs gcd: 3.83e-16 (max-entry) |
| r2xyz | x3c | 4.5,0.0,0.0 | 0.577 | 46 | 8.51 | - | 0.247 | 25 | 76.76 | - | NaN | offlat | 8 | NaN | - | whole vs split: 3.81e-16 (max-entry) |
| r2xyz | x3e | 4.5,0.5,0.0 | 0.574 | 44 | 7.80 | - | 0.245 | 25 | 78.81 | - | NaN | offlat | 8 | NaN | - | whole vs split: 3.44e-16 (max-entry) |
| r2xyz | x3v | 4.5,0.5,0.5 | 0.570 | 44 | 8.11 | - | 0.243 | 25 | 78.90 | - | 0.433 | 8/0/0 | 8 | 101.7 | - | whole vs gcd: 8.04e-16 (max-entry); split vs gcd: 2.62e-16 (max-entry); whole vs split: 8.03e-16 (max-entry) |
| r2xyz | d1c | 2.5,2.5,0.0 | 0.735 | -1 | NaN | - | 0.408 | 38 | 105.83 | - | NaN | offlat | 8 | NaN | - |  |
| r2xyz | d1e | 2.5,2.5,0.5 | 0.728 | -1 | NaN | - | 0.397 | 37 | 108.78 | - | 0.612 | 8/0/0 | 8 | 132.7 | - | split vs gcd: 3.09e-16 (max-entry) |
| r2xyz | d2c | 3.5,3.5,0.0 | 0.525 | 40 | 6.35 | - | 0.245 | 24 | 70.93 | - | NaN | offlat | 8 | NaN | - | whole vs split: 2.83e-16 (max-entry) |
| r2xyz | d2e | 3.5,3.5,0.5 | 0.522 | 40 | 6.27 | - | 0.243 | 24 | 71.10 | - | 0.408 | 8/0/0 | 8 | 102.7 | - | whole vs gcd: 4.98e-16 (max-entry); split vs gcd: 1.91e-16 (max-entry); whole vs split: 3.82e-16 (max-entry) |
| r4x | x1c | 3.5,0.0,0.0 | 0.821 | -1 | NaN | - | 0.522 | 51 | 157.73 | - | 0.866 | 3/1/0 | 4 | 173.0 | - | split vs gcd: 1.49e-15 (max-entry) |
| r4x | x2c | 4.5,0.0,0.0 | 0.638 | -1 | NaN | - | 0.364 | 36 | 68.34 | - | 0.577 | 4/0/0 | 4 | 58.6 | - | split vs gcd: 1.49e-15 (max-entry) |
| r4x | x3c | 5.5,0.0,0.0 | 0.522 | 44 | 7.31 | - | 0.299 | 30 | 46.32 | - | 0.433 | 4/0/0 | 4 | 58.2 | - | whole vs gcd: 1.27e-15 (max-entry); split vs gcd: 7.65e-16 (max-entry); whole vs split: 2.04e-15 (max-entry) |
| r4x | d1c | 3.5,2.0,0.0 | 0.713 | -1 | NaN | - | 0.432 | 43 | 75.47 | - | 0.612 | 4/0/0 | 4 | 59.9 | - | split vs gcd: 4.34e-16 (max-entry) |
| r4x | d2c | 4.5,3.0,0.0 | 0.531 | 46 | 8.32 | - | 0.321 | 32 | 50.86 | - | 0.408 | 4/0/0 | 4 | 48.5 | - | whole vs gcd: 7.65e-16 (max-entry); split vs gcd: 3.32e-16 (max-entry); whole vs split: 1.09e-15 (max-entry) |
| r4xy | x1c | 3.5,0.0,0.0 | 1.050 | -1 | NaN | - | 1.049 | -1 | NaN | - | NaN | offlat | 16 | NaN | - |  |
| r4xy | x1e | 3.5,1.5,0.0 | 0.965 | -1 | NaN | - | 0.761 | -1 | NaN | - | 0.866 | 14/2/0 | 16 | 339.0 | - |  |
| r4xy | x2c | 4.5,0.0,0.0 | 0.816 | -1 | NaN | - | 0.650 | -1 | NaN | - | NaN | offlat | 16 | NaN | - |  |
| r4xy | x2e | 4.5,1.5,0.0 | 0.775 | -1 | NaN | - | 0.561 | -1 | NaN | - | 0.577 | 16/0/0 | 16 | 228.6 | - |  |
| r4xy | x3c | 5.5,0.0,0.0 | 0.668 | -1 | NaN | - | 0.469 | 47 | 81.22 | - | NaN | offlat | 16 | NaN | - |  |
| r4xy | x3e | 5.5,1.5,0.0 | 0.645 | -1 | NaN | - | 0.432 | 43 | 80.59 | - | 0.433 | 16/0/0 | 16 | 184.6 | - | split vs gcd: 6.84e-16 (max-entry) |
| r4xy | d1c | 3.5,3.5,0.0 | 0.742 | -1 | NaN | - | 0.438 | 43 | 109.46 | - | 0.612 | 16/0/0 | 16 | 183.3 | - | split vs gcd: 6.25e-16 (max-entry) |
| r4xy | d2c | 4.5,4.5,0.0 | 0.577 | 50 | 9.80 | - | 0.341 | 33 | 70.91 | - | 0.408 | 16/0/0 | 16 | 193.0 | - | whole vs gcd: 9.82e-16 (max-entry); split vs gcd: 6.55e-16 (max-entry); whole vs split: 8.18e-16 (max-entry) |
| r4xyz | x1c | 3.5,0.0,0.0 | 1.237 | -1 | NaN | - | 1.453 | -1 | NaN | - | NaN | offlat | 64 | NaN | - |  |
| r4xyz | x1e | 3.5,1.5,0.0 | 1.137 | -1 | NaN | - | 1.049 | -1 | NaN | - | NaN | offlat | 64 | NaN | - |  |
| r4xyz | x1v | 3.5,1.5,1.5 | 1.058 | -1 | NaN | - | 0.839 | -1 | NaN | - | 0.866 | 60/4/0 | 64 | 1030.9 | - |  |
| r4xyz | x2c | 4.5,0.0,0.0 | 0.962 | -1 | NaN | - | 0.872 | -1 | NaN | - | NaN | offlat | 64 | NaN | - |  |
| r4xyz | x2e | 4.5,1.5,0.0 | 0.913 | -1 | NaN | - | 0.748 | -1 | NaN | - | NaN | offlat | 64 | NaN | - |  |
| r4xyz | x2v | 4.5,1.5,1.5 | 0.870 | -1 | NaN | - | 0.665 | -1 | NaN | - | 0.577 | 64/0/0 | 64 | 694.9 | - |  |
| r4xyz | x3c | 5.5,0.0,0.0 | 0.787 | -1 | NaN | - | 0.623 | -1 | NaN | - | NaN | offlat | 64 | NaN | - |  |
| r4xyz | x3e | 5.5,1.5,0.0 | 0.760 | -1 | NaN | - | 0.572 | -1 | NaN | - | NaN | offlat | 64 | NaN | - |  |
| r4xyz | x3v | 5.5,1.5,1.5 | 0.735 | -1 | NaN | - | 0.533 | 56 | 131.72 | - | 0.433 | 64/0/0 | 64 | 851.8 | - | split vs gcd: 6.69e-16 (max-entry) |
| r4xyz | d1c | 3.5,3.5,0.0 | 0.875 | -1 | NaN | - | 0.782 | -1 | NaN | - | NaN | offlat | 64 | NaN | - |  |
| r4xyz | d1e | 3.5,3.5,1.5 | 0.837 | -1 | NaN | - | 0.638 | -1 | NaN | - | 0.612 | 64/0/0 | 64 | 747.7 | - |  |
| r4xyz | d2c | 4.5,4.5,0.0 | 0.680 | -1 | NaN | - | 0.469 | 46 | 104.70 | - | NaN | offlat | 64 | NaN | - |  |
| r4xyz | d2e | 4.5,4.5,1.5 | 0.662 | -1 | NaN | - | 0.432 | 42 | 116.50 | - | 0.408 | 64/0/0 | 64 | 545.5 | - | split vs gcd: 2.99e-16 (max-entry) |
| r8x | x1c | 5.5,0.0,0.0 | 0.858 | -1 | NaN | - | 0.644 | -1 | NaN | - | 0.866 | 7/1/0 | 8 | 227.1 | - |  |
| r8x | x2c | 6.5,0.0,0.0 | 0.726 | -1 | NaN | - | 0.546 | -1 | NaN | - | 0.577 | 8/0/0 | 8 | 90.8 | - |  |
| r8x | x3c | 7.5,0.0,0.0 | 0.629 | -1 | NaN | - | 0.474 | 51 | 85.85 | - | 0.433 | 8/0/0 | 8 | 90.1 | - | split vs gcd: 6.65e-16 (max-entry) |
| r8x | d1c | 5.5,2.0,0.0 | 0.806 | -1 | NaN | - | 0.624 | -1 | NaN | - | 0.612 | 8/0/0 | 8 | 114.9 | - |  |
| r8x | d2c | 6.5,3.0,0.0 | 0.659 | -1 | NaN | - | 0.511 | 56 | 99.37 | - | 0.408 | 8/0/0 | 8 | 124.9 | - | split vs gcd: 1.22e-15 (max-entry) |
| r8xy | x1c | 5.5,0.0,0.0 | 1.171 | -1 | NaN | - | 2.258 | -1 | NaN | - | NaN | offlat | 64 | NaN | - |  |
| r8xy | x1e | 5.5,3.5,0.0 | 0.988 | -1 | NaN | - | 0.930 | -1 | NaN | - | 0.866 | 62/2/0 | 64 | 831.6 | - |  |
| r8xy | x2c | 6.5,0.0,0.0 | 0.991 | -1 | NaN | - | 1.401 | -1 | NaN | - | NaN | offlat | 64 | NaN | - |  |
| r8xy | x2e | 6.5,3.5,0.0 | 0.873 | -1 | NaN | - | 0.825 | -1 | NaN | - | 0.577 | 64/0/0 | 64 | 727.4 | - |  |
| r8xy | x3c | 7.5,0.0,0.0 | 0.859 | -1 | NaN | - | 1.010 | -1 | NaN | - | NaN | offlat | 64 | NaN | - |  |
| r8xy | x3e | 7.5,3.5,0.0 | 0.778 | -1 | NaN | - | 0.718 | -1 | NaN | - | 0.433 | 64/0/0 | 64 | 609.3 | - |  |
| r8xy | d1c | 5.5,5.5,0.0 | 0.828 | -1 | NaN | - | 0.638 | -1 | NaN | - | 0.612 | 64/0/0 | 64 | 524.7 | - |  |
| r8xy | d2c | 6.5,6.5,0.0 | 0.701 | -1 | NaN | - | 0.540 | -1 | NaN | - | 0.408 | 64/0/0 | 64 | 733.5 | - |  |
| r8xyz | x1c | 5.5,0.0,0.0 | 1.417 | -1 | NaN | - | 3.317 | -1 | NaN | - | NaN | offlat | 512 | NaN | - |  |
| r8xyz | x1e | 5.5,3.5,0.0 | 1.196 | -1 | NaN | - | 2.258 | -1 | NaN | - | NaN | offlat | 512 | NaN | - |  |
| r8xyz | x1v | 5.5,3.5,3.5 | 1.053 | -1 | NaN | - | 0.962 | -1 | NaN | - | 0.866 | 508/4/0 | 512 | 5782.7 | - |  |
| r8xyz | x2c | 6.5,0.0,0.0 | 1.199 | -1 | NaN | - | 1.990 | -1 | NaN | - | NaN | offlat | 512 | NaN | - |  |
| r8xyz | x2e | 6.5,3.5,0.0 | 1.056 | -1 | NaN | - | 1.401 | -1 | NaN | - | NaN | offlat | 512 | NaN | - |  |
| r8xyz | x2v | 6.5,3.5,3.5 | 0.954 | -1 | NaN | - | 0.897 | -1 | NaN | - | 0.577 | 512/0/0 | 512 | 4877.3 | - |  |
| r8xyz | x3c | 7.5,0.0,0.0 | 1.039 | -1 | NaN | - | 1.421 | -1 | NaN | - | NaN | offlat | 512 | NaN | - |  |
| r8xyz | x3e | 7.5,3.5,0.0 | 0.942 | -1 | NaN | - | 1.010 | -1 | NaN | - | NaN | offlat | 512 | NaN | - |  |
| r8xyz | x3v | 7.5,3.5,3.5 | 0.867 | -1 | NaN | - | 0.821 | -1 | NaN | - | 0.433 | 512/0/0 | 512 | 4959.3 | - |  |
| r8xyz | d1c | 5.5,5.5,0.0 | 1.002 | -1 | NaN | - | 1.683 | -1 | NaN | - | NaN | offlat | 512 | NaN | - |  |
| r8xyz | d1e | 5.5,5.5,3.5 | 0.914 | -1 | NaN | - | 0.872 | -1 | NaN | - | 0.612 | 512/0/0 | 512 | 5471.8 | - |  |
| r8xyz | d2c | 6.5,6.5,0.0 | 0.848 | -1 | NaN | - | 1.010 | -1 | NaN | - | NaN | offlat | 512 | NaN | - |  |
| r8xyz | d2e | 6.5,6.5,3.5 | 0.792 | -1 | NaN | - | 0.718 | -1 | NaN | - | 0.408 | 512/0/0 | 512 | 5208.6 | - |  |
| r16x | x1c | 9.5,0.0,0.0 | 0.907 | -1 | NaN | - | 0.791 | -1 | NaN | - | 0.866 | 15/1/0 | 16 | 300.0 | 2.5e-15,2.5e-15 |  |
| r16x | x2c | 10.5,0.0,0.0 | 0.821 | -1 | NaN | - | 0.716 | -1 | NaN | - | 0.577 | 16/0/0 | 16 | 169.3 | - |  |
| r16x | x3c | 11.5,0.0,0.0 | 0.749 | -1 | NaN | - | 0.654 | -1 | NaN | - | 0.433 | 16/0/0 | 16 | 171.9 | - |  |
| r16xy | x1c | 9.5,0.0,0.0 | 1.270 | -1 | NaN | - | 4.764 | -1 | NaN | - | NaN | offlat | 256 | NaN | - |  |
| r16xy | x1e | 9.5,7.5,0.0 | 0.997 | -1 | NaN | - | 0.983 | -1 | NaN | - | 0.866 | 254/2/0 | 256 | 2654.0 | - |  |
| r16xy | x2c | 10.5,0.0,0.0 | 1.149 | -1 | NaN | - | 2.955 | -1 | NaN | - | NaN | offlat | 256 | NaN | - |  |
| r16xy | x2e | 10.5,7.5,0.0 | 0.935 | -1 | NaN | - | 0.951 | -1 | NaN | - | 0.577 | 256/0/0 | 256 | 2702.9 | - |  |
| r16xy | x3c | 11.5,0.0,0.0 | 1.049 | -1 | NaN | - | 2.131 | -1 | NaN | - | NaN | offlat | 256 | NaN | - |  |
| r16xy | x3e | 11.5,7.5,0.0 | 0.879 | -1 | NaN | - | 0.909 | -1 | NaN | - | 0.433 | 256/0/0 | 256 | 2522.1 | - |  |
| r16xyz | x1c | 9.5,0.0,0.0 | 1.550 | -1 | NaN | - | 7.079 | -1 | NaN | - | NaN | offlat | 4096 | NaN | - |  |
| r16xyz | x1e | 9.5,7.5,0.0 | 1.216 | -1 | NaN | - | 4.764 | -1 | NaN | - | NaN | offlat | 4096 | NaN | - |  |
| r16xyz | x1v | 9.5,7.5,7.5 | 1.034 | -1 | NaN | - | 0.991 | -1 | NaN | - | 0.866 | 4092/4/0 | 4096 | 42957.3 | - |  |
| r16xyz | x2c | 10.5,0.0,0.0 | 1.402 | -1 | NaN | - | 4.247 | -1 | NaN | - | NaN | offlat | 4096 | NaN | - |  |
| r16xyz | x2e | 10.5,7.5,0.0 | 1.141 | -1 | NaN | - | 2.955 | -1 | NaN | - | NaN | offlat | 4096 | NaN | - |  |
| r16xyz | x2v | 10.5,7.5,7.5 | 0.986 | -1 | NaN | - | 0.974 | -1 | NaN | - | 0.577 | 4096/0/0 | 4096 | 39345.6 | - |  |
| r16xyz | x3c | 11.5,0.0,0.0 | 1.280 | -1 | NaN | - | 3.034 | -1 | NaN | - | NaN | offlat | 4096 | NaN | - |  |
| r16xyz | x3e | 11.5,7.5,0.0 | 1.072 | -1 | NaN | - | 2.131 | -1 | NaN | - | NaN | offlat | 4096 | NaN | - |  |
| r16xyz | x3v | 11.5,7.5,7.5 | 0.941 | -1 | NaN | - | 0.951 | -1 | NaN | - | 0.433 | 4096/0/0 | 4096 | 40316.9 | - |  |

| shape | off | R/g | rho_w | L_w | t_w | L_p | t_p | t_gcd | agreement |
|---|---|---|---|---|---|---|---|---|---|
| r2x | x4c | 5.5,0.0,0.0 | 0.375 | 28 | 2.97 | 21 | 32.42 | 28.0 | whole vs gcd: 1.04e-15 (max-entry); split vs gcd: 7.46e-16 (max-entry); whole vs split: 4.01e-16 (max-entry) |
| r2x | x6c | 7.5,0.0,0.0 | 0.275 | 22 | 2.13 | 17 | 25.13 | 33.4 | whole vs gcd: 9.57e-16 (max-entry); split vs gcd: 4.85e-16 (max-entry); whole vs split: 4.85e-16 (max-entry) |
| r2x | x8c | 9.5,0.0,0.0 | 0.217 | 20 | 1.66 | 16 | 21.83 | 31.2 | whole vs gcd: 4.34e-16 (max-entry); split vs gcd: 2.75e-16 (max-entry); whole vs split: 4.86e-16 (max-entry) |
| r2x | x16c | 17.5,0.0,0.0 | 0.118 | 16 | 1.26 | 13 | 16.34 | 31.1 | whole vs gcd: 5.43e-16 (max-entry); split vs gcd: 3.62e-16 (max-entry); whole vs split: 5.12e-16 (max-entry) |
| r2x | x32c | 33.5,0.0,0.0 | 0.062 | 12 | 0.88 | 11 | 15.07 | 32.9 | whole vs gcd: 3.02e-16 (max-entry); split vs gcd: 4.83e-16 (max-entry); whole vs split: 3.77e-16 (max-entry) |
| r2x | d4c | 5.5,5.0,0.0 | 0.277 | 22 | 2.17 | 18 | 27.81 | 30.1 | whole vs gcd: 3.97e-16 (max-entry); split vs gcd: 2.81e-16 (max-entry); whole vs split: 3.97e-16 (max-entry) |
| r2x | d8c | 9.5,9.0,0.0 | 0.158 | 16 | 1.24 | 14 | 18.39 | 24.5 | whole vs gcd: 4.17e-16 (max-entry); split vs gcd: 5.13e-16 (max-entry); whole vs split: 2.56e-16 (max-entry) |
| r2x | d16c | 17.5,17.0,0.0 | 0.084 | 14 | 1.02 | 12 | 16.48 | 26.8 | whole vs gcd: 3.32e-16 (max-entry); split vs gcd: 3.71e-16 (max-entry); whole vs split: 4.56e-16 (max-entry) |
| r2xy | x4c | 5.5,0.0,0.0 | 0.426 | 32 | 3.88 | 21 | 45.20 | NaN | whole vs split: 5.01e-16 (max-entry) |
| r2xy | x4e | 5.5,0.5,0.0 | 0.425 | 32 | 6.67 | 21 | 45.58 | 50.7 | whole vs gcd: 3.13e-16 (max-entry); split vs gcd: 4.73e-16 (max-entry); whole vs split: 3.13e-16 (max-entry) |
| r2xy | x6c | 7.5,0.0,0.0 | 0.313 | 24 | 2.35 | 18 | 34.47 | NaN | whole vs split: 8.68e-16 (max-entry) |
| r2xy | x6e | 7.5,0.5,0.0 | 0.312 | 24 | 2.41 | 18 | 36.43 | 43.5 | whole vs gcd: 5.15e-16 (max-entry); split vs gcd: 4.88e-16 (max-entry); whole vs split: 4.15e-16 (max-entry) |
| r2xy | x8c | 9.5,0.0,0.0 | 0.247 | 20 | 1.78 | 16 | 29.91 | NaN | whole vs split: 9.78e-16 (max-entry) |
| r2xy | x8e | 9.5,0.5,0.0 | 0.247 | 20 | 1.75 | 16 | 30.56 | 45.3 | whole vs gcd: 3.55e-16 (max-entry); split vs gcd: 2.87e-16 (max-entry); whole vs split: 3.75e-16 (max-entry) |
| r2xy | x16c | 17.5,0.0,0.0 | 0.134 | 16 | 1.07 | 13 | 21.10 | NaN | whole vs split: 7.47e-16 (max-entry) |
| r2xy | x16e | 17.5,0.5,0.0 | 0.134 | 16 | 1.08 | 13 | 20.88 | 41.3 | whole vs gcd: 5.85e-16 (max-entry); split vs gcd: 5.44e-16 (max-entry); whole vs split: 5.67e-16 (max-entry) |
| r2xy | x32c | 33.5,0.0,0.0 | 0.070 | 12 | 0.80 | 11 | 17.78 | NaN | whole vs split: 4.53e-16 (max-entry) |
| r2xy | x32e | 33.5,0.5,0.0 | 0.070 | 12 | 0.80 | 11 | 17.75 | 32.7 | whole vs gcd: 3.77e-16 (max-entry); split vs gcd: 4.06e-16 (max-entry); whole vs split: 2.72e-16 (max-entry) |
| r2xy | d4c | 5.5,5.5,0.0 | 0.302 | 24 | 2.18 | 17 | 35.10 | 44.3 | whole vs gcd: 2.79e-16 (max-entry); split vs gcd: 1.96e-16 (max-entry); whole vs split: 3.46e-16 (max-entry) |
| r2xy | d8c | 9.5,9.5,0.0 | 0.175 | 18 | 1.48 | 14 | 25.91 | 42.4 | whole vs gcd: 7.46e-16 (max-entry); split vs gcd: 2.70e-16 (max-entry); whole vs split: 5.52e-16 (max-entry) |
| r2xy | d16c | 17.5,17.5,0.0 | 0.095 | 14 | 1.07 | 12 | 22.38 | 47.0 | whole vs gcd: 5.03e-16 (max-entry); split vs gcd: 2.25e-16 (max-entry); whole vs split: 4.09e-16 (max-entry) |
| r2xyz | x4c | 5.5,0.0,0.0 | 0.472 | 34 | 4.63 | 21 | 58.85 | NaN | whole vs split: 4.69e-16 (max-entry) |
| r2xyz | x4e | 5.5,0.5,0.0 | 0.470 | 34 | 4.68 | 21 | 60.15 | NaN | whole vs split: 1.44e-15 (max-entry) |
| r2xyz | x4v | 5.5,0.5,0.5 | 0.469 | 34 | 4.74 | 21 | 61.13 | 97.2 | whole vs gcd: 4.86e-16 (max-entry); split vs gcd: 3.27e-16 (max-entry); whole vs split: 4.88e-16 (max-entry) |
| r2xyz | x6c | 7.5,0.0,0.0 | 0.346 | 26 | 2.67 | 18 | 51.48 | NaN | whole vs split: 1.63e-16 (max-entry) |
| r2xyz | x6e | 7.5,0.5,0.0 | 0.346 | 26 | 2.64 | 18 | 52.48 | NaN | whole vs split: 1.65e-16 (max-entry) |
| r2xyz | x6v | 7.5,0.5,0.5 | 0.345 | 26 | 2.56 | 18 | 51.60 | 95.0 | whole vs gcd: 3.35e-16 (max-entry); split vs gcd: 1.66e-16 (max-entry); whole vs split: 4.16e-16 (max-entry) |
| r2xyz | x8c | 9.5,0.0,0.0 | 0.273 | 22 | 2.03 | 16 | 45.25 | NaN | whole vs split: 5.74e-16 (max-entry) |
| r2xyz | x8e | 9.5,0.5,0.0 | 0.273 | 22 | 1.99 | 16 | 41.89 | NaN | whole vs split: 3.13e-16 (max-entry) |
| r2xyz | x8v | 9.5,0.5,0.5 | 0.273 | 22 | 2.11 | 16 | 41.65 | 89.4 | whole vs gcd: 3.59e-16 (max-entry); split vs gcd: 4.23e-16 (max-entry); whole vs split: 4.28e-16 (max-entry) |
| r2xyz | x16c | 17.5,0.0,0.0 | 0.148 | 16 | 1.30 | 13 | 32.34 | NaN | whole vs split: 4.59e-16 (max-entry) |
| r2xyz | x16e | 17.5,0.5,0.0 | 0.148 | 16 | 1.27 | 13 | 34.24 | NaN | whole vs split: 4.08e-16 (max-entry) |
| r2xyz | x16v | 17.5,0.5,0.5 | 0.148 | 16 | 1.21 | 13 | 32.06 | 82.2 | whole vs gcd: 3.63e-16 (max-entry); split vs gcd: 3.35e-16 (max-entry); whole vs split: 5.20e-16 (max-entry) |
| r2xyz | x32c | 33.5,0.0,0.0 | 0.078 | 14 | 1.38 | 11 | 28.96 | NaN | whole vs split: 6.40e-16 (max-entry) |
| r2xyz | x32e | 33.5,0.5,0.0 | 0.078 | 14 | 1.10 | 11 | 31.60 | NaN | whole vs split: 2.72e-16 (max-entry) |
| r2xyz | x32v | 33.5,0.5,0.5 | 0.078 | 14 | 1.06 | 11 | 29.19 | 78.2 | whole vs gcd: 4.83e-16 (max-entry); split vs gcd: 2.13e-16 (max-entry); whole vs split: 5.06e-16 (max-entry) |
| r2xyz | d4c | 5.5,5.5,0.0 | 0.334 | 26 | 2.67 | 18 | 48.68 | NaN | whole vs split: 2.52e-16 (max-entry) |
| r2xyz | d4e | 5.5,5.5,0.5 | 0.333 | 26 | 2.55 | 18 | 49.29 | 90.8 | whole vs gcd: 2.82e-16 (max-entry); split vs gcd: 1.99e-16 (max-entry); whole vs split: 4.23e-16 (max-entry) |
| r2xyz | d8c | 9.5,9.5,0.0 | 0.193 | 18 | 1.47 | 14 | 37.43 | NaN | whole vs split: 8.37e-16 (max-entry) |
| r2xyz | d8e | 9.5,9.5,0.5 | 0.193 | 18 | 1.55 | 14 | 34.74 | 85.0 | whole vs gcd: 7.23e-16 (max-entry); split vs gcd: 5.24e-16 (max-entry); whole vs split: 2.58e-16 (max-entry) |
| r2xyz | d16c | 17.5,17.5,0.0 | 0.105 | 14 | 1.08 | 12 | 33.08 | NaN | whole vs split: 3.42e-16 (max-entry) |
| r2xyz | d16e | 17.5,17.5,0.5 | 0.105 | 14 | 1.09 | 12 | 30.10 | 88.0 | whole vs gcd: 3.42e-16 (max-entry); split vs gcd: 3.03e-16 (max-entry); whole vs split: 3.56e-16 (max-entry) |
| r4x | x4c | 6.5,0.0,0.0 | 0.442 | 36 | 4.89 | 27 | 38.69 | 46.9 | whole vs gcd: 8.20e-16 (max-entry); split vs gcd: 4.37e-16 (max-entry); whole vs split: 4.22e-16 (max-entry) |
| r4x | x6c | 8.5,0.0,0.0 | 0.338 | 28 | 3.06 | 23 | 27.44 | 42.5 | whole vs gcd: 6.20e-16 (max-entry); split vs gcd: 6.11e-16 (max-entry); whole vs split: 7.13e-16 (max-entry) |
| r4x | x8c | 10.5,0.0,0.0 | 0.274 | 24 | 2.32 | 20 | 23.76 | 42.2 | whole vs gcd: 3.03e-16 (max-entry); split vs gcd: 6.06e-16 (max-entry); whole vs split: 4.20e-16 (max-entry) |
| r4x | x16c | 18.5,0.0,0.0 | 0.155 | 18 | 1.37 | 16 | 18.01 | 42.4 | whole vs gcd: 2.73e-16 (max-entry); split vs gcd: 4.41e-16 (max-entry); whole vs split: 3.12e-16 (max-entry) |
| r4x | x32c | 34.5,0.0,0.0 | 0.083 | 14 | 1.02 | 14 | 14.85 | 35.9 | whole vs gcd: 4.47e-16 (max-entry); split vs gcd: 4.47e-16 (max-entry); whole vs split: 5.06e-16 (max-entry) |
| r4x | d4c | 6.5,5.0,0.0 | 0.350 | 30 | 3.38 | 24 | 29.12 | 34.7 | whole vs gcd: 7.78e-16 (max-entry); split vs gcd: 7.77e-16 (max-entry); whole vs split: 1.56e-15 (max-entry) |
| r4x | d8c | 10.5,9.0,0.0 | 0.208 | 20 | 2.15 | 18 | 20.68 | 44.6 | whole vs gcd: 4.65e-16 (max-entry); split vs gcd: 1.47e-16 (max-entry); whole vs split: 5.01e-16 (max-entry) |
| r4x | d16c | 18.5,17.0,0.0 | 0.114 | 16 | 1.23 | 15 | 16.61 | 44.3 | whole vs gcd: 2.57e-16 (max-entry); split vs gcd: 3.49e-16 (max-entry); whole vs split: 1.28e-16 (max-entry) |
| r4xy | x4c | 6.5,0.0,0.0 | 0.565 | 48 | 9.60 | 36 | 62.36 | NaN | whole vs split: 3.52e-16 (max-entry) |
| r4xy | x4e | 6.5,1.5,0.0 | 0.551 | 46 | 9.01 | 34 | 59.20 | 150.7 | whole vs gcd: 1.14e-15 (max-entry); split vs gcd: 2.26e-16 (max-entry); whole vs split: 1.28e-15 (max-entry) |
| r4xy | x6c | 8.5,0.0,0.0 | 0.432 | 34 | 4.23 | 27 | 46.70 | NaN | whole vs split: 4.29e-16 (max-entry) |
| r4xy | x6e | 8.5,1.5,0.0 | 0.426 | 34 | 4.56 | 27 | 46.56 | 161.8 | whole vs gcd: 7.50e-16 (max-entry); split vs gcd: 2.85e-16 (max-entry); whole vs split: 5.95e-16 (max-entry) |
| r4xy | x8c | 10.5,0.0,0.0 | 0.350 | 28 | 2.91 | 23 | 36.46 | NaN | whole vs split: 1.24e-15 (max-entry) |
| r4xy | x8e | 10.5,1.5,0.0 | 0.346 | 28 | 2.97 | 23 | 35.13 | 138.4 | whole vs gcd: 6.38e-16 (max-entry); split vs gcd: 7.27e-16 (max-entry); whole vs split: 4.03e-16 (max-entry) |
| r4xy | x16c | 18.5,0.0,0.0 | 0.199 | 20 | 1.78 | 18 | 27.18 | NaN | whole vs split: 3.87e-16 (max-entry) |
| r4xy | x16e | 18.5,1.5,0.0 | 0.198 | 20 | 1.68 | 18 | 26.01 | 155.0 | whole vs gcd: 7.01e-16 (max-entry); split vs gcd: 8.01e-16 (max-entry); whole vs split: 1.50e-15 (max-entry) |
| r4xy | x32c | 34.5,0.0,0.0 | 0.106 | 16 | 1.17 | 15 | 21.93 | NaN | whole vs split: 1.20e-15 (max-entry) |
| r4xy | x32e | 34.5,1.5,0.0 | 0.106 | 16 | 1.18 | 15 | 22.22 | 165.0 | whole vs gcd: 9.26e-16 (max-entry); split vs gcd: 6.38e-16 (max-entry); whole vs split: 3.54e-16 (max-entry) |
| r4xy | d4c | 6.5,6.5,0.0 | 0.400 | 32 | 4.13 | 25 | 46.85 | 129.3 | whole vs gcd: 3.67e-16 (max-entry); split vs gcd: 4.32e-16 (max-entry); whole vs split: 4.05e-16 (max-entry) |
| r4xy | d8c | 10.5,10.5,0.0 | 0.247 | 22 | 2.12 | 20 | 32.07 | 148.5 | whole vs gcd: 2.82e-16 (max-entry); split vs gcd: 3.18e-16 (max-entry); whole vs split: 2.00e-16 (max-entry) |
| r4xy | d16c | 18.5,18.5,0.0 | 0.140 | 16 | 1.45 | 16 | 25.45 | 154.4 | whole vs gcd: 2.41e-16 (max-entry); split vs gcd: 4.35e-16 (max-entry); whole vs split: 5.12e-16 (max-entry) |
| r4xyz | x4c | 6.5,0.0,0.0 | 0.666 | -1 | NaN | 48 | 101.36 | NaN |  |
| r4xyz | x4e | 6.5,1.5,0.0 | 0.649 | -1 | NaN | 45 | 101.23 | NaN |  |
| r4xyz | x4v | 6.5,1.5,1.5 | 0.633 | 54 | 11.48 | 43 | 98.47 | 620.7 | whole vs gcd: 5.29e-16 (max-entry); split vs gcd: 1.51e-16 (max-entry); whole vs split: 6.80e-16 (max-entry) |
| r4xyz | x6c | 8.5,0.0,0.0 | 0.509 | 40 | 6.54 | 33 | 76.33 | NaN | whole vs split: 5.66e-16 (max-entry) |
| r4xyz | x6e | 8.5,1.5,0.0 | 0.502 | 38 | 5.65 | 32 | 72.16 | NaN | whole vs split: 7.75e-16 (max-entry) |
| r4xyz | x6v | 8.5,1.5,1.5 | 0.494 | 38 | 5.77 | 31 | 70.78 | 577.8 | whole vs gcd: 2.61e-16 (max-entry); split vs gcd: 3.17e-16 (max-entry); whole vs split: 5.67e-16 (max-entry) |
| r4xyz | x8c | 10.5,0.0,0.0 | 0.412 | 32 | 4.05 | 26 | 59.08 | NaN | whole vs split: 9.03e-16 (max-entry) |
| r4xyz | x8e | 10.5,1.5,0.0 | 0.408 | 32 | 3.81 | 26 | 58.99 | NaN | whole vs split: 5.85e-16 (max-entry) |
| r4xyz | x8v | 10.5,1.5,1.5 | 0.404 | 30 | 3.42 | 26 | 59.15 | 605.3 | whole vs gcd: 6.76e-16 (max-entry); split vs gcd: 2.94e-16 (max-entry); whole vs split: 4.83e-16 (max-entry) |
| r4xyz | x16c | 18.5,0.0,0.0 | 0.234 | 20 | 1.71 | 19 | 49.60 | NaN | whole vs split: 2.46e-16 (max-entry) |
| r4xyz | x16e | 18.5,1.5,0.0 | 0.233 | 20 | 1.83 | 19 | 44.01 | NaN | whole vs split: 6.18e-16 (max-entry) |
| r4xyz | x16v | 18.5,1.5,1.5 | 0.233 | 20 | 1.80 | 19 | 39.52 | 580.5 | whole vs gcd: 6.36e-16 (max-entry); split vs gcd: 2.79e-16 (max-entry); whole vs split: 3.94e-16 (max-entry) |
| r4xyz | x32c | 34.5,0.0,0.0 | 0.126 | 16 | 1.99 | 16 | 32.59 | NaN | whole vs split: 5.01e-16 (max-entry) |
| r4xyz | x32e | 34.5,1.5,0.0 | 0.125 | 16 | 1.19 | 16 | 32.56 | NaN | whole vs split: 7.09e-16 (max-entry) |
| r4xyz | x32v | 34.5,1.5,1.5 | 0.125 | 16 | 1.33 | 16 | 34.25 | 623.4 | whole vs gcd: 3.97e-16 (max-entry); split vs gcd: 3.55e-16 (max-entry); whole vs split: 4.83e-16 (max-entry) |
| r4xyz | d4c | 6.5,6.5,0.0 | 0.471 | 36 | 5.37 | 29 | 69.65 | NaN | whole vs split: 8.68e-16 (max-entry) |
| r4xyz | d4e | 6.5,6.5,1.5 | 0.465 | 36 | 5.34 | 28 | 64.47 | 578.6 | whole vs gcd: 3.19e-16 (max-entry); split vs gcd: 9.90e-17 (max-entry); whole vs split: 3.20e-16 (max-entry) |
| r4xyz | d8c | 10.5,10.5,0.0 | 0.292 | 24 | 2.37 | 21 | 48.49 | NaN | whole vs split: 5.80e-16 (max-entry) |
| r4xyz | d8e | 10.5,10.5,1.5 | 0.290 | 24 | 2.29 | 21 | 44.46 | 633.3 | whole vs gcd: 1.01e-15 (max-entry); split vs gcd: 8.94e-16 (max-entry); whole vs split: 3.07e-16 (max-entry) |
| r4xyz | d16c | 18.5,18.5,0.0 | 0.166 | 18 | 1.55 | 17 | 38.81 | NaN | whole vs split: 1.91e-16 (max-entry) |
| r4xyz | d16e | 18.5,18.5,1.5 | 0.165 | 18 | 1.57 | 17 | 36.69 | 580.4 | whole vs gcd: 1.82e-16 (max-entry); split vs gcd: 3.85e-16 (max-entry); whole vs split: 3.44e-16 (max-entry) |
| r8x | x4c | 8.5,0.0,0.0 | 0.555 | 54 | 11.71 | 44 | 66.89 | 76.4 | whole vs gcd: 1.29e-15 (max-entry); split vs gcd: 4.35e-16 (max-entry); whole vs split: 1.62e-15 (max-entry) |
| r8x | x6c | 10.5,0.0,0.0 | 0.449 | 42 | 7.68 | 35 | 48.41 | 82.3 | whole vs gcd: 8.30e-16 (max-entry); split vs gcd: 7.70e-16 (max-entry); whole vs split: 1.13e-15 (max-entry) |
| r8x | x8c | 12.5,0.0,0.0 | 0.377 | 34 | 4.88 | 31 | 35.94 | 81.2 | whole vs gcd: 2.66e-16 (max-entry); split vs gcd: 1.10e-15 (max-entry); whole vs split: 1.23e-15 (max-entry) |
| r8x | x16c | 20.5,0.0,0.0 | 0.230 | 24 | 3.44 | 23 | 24.16 | 78.3 | whole vs gcd: 1.17e-15 (max-entry); split vs gcd: 7.08e-16 (max-entry); whole vs split: 6.07e-16 (max-entry) |
| r8x | x32c | 36.5,0.0,0.0 | 0.129 | 18 | 1.83 | 18 | 18.80 | 61.2 | whole vs gcd: 4.84e-16 (max-entry); split vs gcd: 1.11e-15 (max-entry); whole vs split: 9.16e-16 (max-entry) |
| r8x | d4c | 8.5,5.0,0.0 | 0.478 | 44 | 8.82 | 38 | 48.87 | 72.8 | whole vs gcd: 8.80e-16 (max-entry); split vs gcd: 9.87e-16 (max-entry); whole vs split: 8.98e-16 (max-entry) |
| r8x | d8c | 12.5,9.0,0.0 | 0.306 | 28 | 3.31 | 27 | 29.13 | 69.6 | whole vs gcd: 6.54e-16 (max-entry); split vs gcd: 7.49e-16 (max-entry); whole vs split: 8.24e-16 (max-entry) |
| r8x | d16c | 20.5,17.0,0.0 | 0.177 | 20 | 1.85 | 20 | 20.82 | 65.9 | whole vs gcd: 3.81e-16 (max-entry); split vs gcd: 2.83e-16 (max-entry); whole vs split: 4.01e-16 (max-entry) |
| r8xy | x4c | 8.5,0.0,0.0 | 0.758 | -1 | NaN | -1 | NaN | NaN |  |
| r8xy | x4e | 8.5,3.5,0.0 | 0.701 | -1 | NaN | -1 | NaN | 500.4 |  |
| r8xy | x6c | 10.5,0.0,0.0 | 0.614 | -1 | NaN | -1 | NaN | NaN |  |
| r8xy | x6e | 10.5,3.5,0.0 | 0.582 | 54 | 11.19 | 51 | 99.64 | 706.1 | whole vs gcd: 5.92e-14 (max-entry); split vs gcd: 1.73e-15 (max-entry); whole vs split: 5.74e-14 (max-entry) |
| r8xy | x8c | 12.5,0.0,0.0 | 0.515 | 46 | 8.29 | 43 | 73.94 | NaN | whole vs split: 7.61e-16 (max-entry) |
| r8xy | x8e | 12.5,3.5,0.0 | 0.496 | 44 | 8.07 | 39 | 68.83 | 647.1 | whole vs gcd: 8.11e-16 (max-entry); split vs gcd: 5.35e-16 (max-entry); whole vs split: 1.30e-15 (max-entry) |
| r8xy | x16c | 20.5,0.0,0.0 | 0.314 | 28 | 3.31 | 27 | 42.90 | NaN | whole vs split: 6.51e-16 (max-entry) |
| r8xy | x16e | 20.5,3.5,0.0 | 0.310 | 28 | 3.09 | 27 | 39.09 | 712.7 | whole vs gcd: 9.32e-16 (max-entry); split vs gcd: 4.76e-16 (max-entry); whole vs split: 1.17e-15 (max-entry) |
| r8xy | x32c | 36.5,0.0,0.0 | 0.176 | 20 | 1.76 | 21 | 28.87 | NaN | whole vs split: 4.02e-16 (max-entry) |
| r8xy | x32e | 36.5,3.5,0.0 | 0.176 | 20 | 1.79 | 21 | 28.33 | 661.0 | whole vs gcd: 5.72e-16 (max-entry); split vs gcd: 3.73e-16 (max-entry); whole vs split: 3.73e-16 (max-entry) |
| r8xy | d4c | 8.5,8.5,0.0 | 0.536 | 48 | 9.20 | 42 | 80.46 | 710.0 | whole vs gcd: 2.52e-15 (max-entry); split vs gcd: 8.41e-16 (max-entry); whole vs split: 3.36e-15 (max-entry) |
| r8xy | d8c | 12.5,12.5,0.0 | 0.364 | 32 | 3.90 | 30 | 44.22 | 686.2 | whole vs gcd: 5.57e-16 (max-entry); split vs gcd: 4.40e-16 (max-entry); whole vs split: 4.05e-16 (max-entry) |
| r8xy | d16c | 20.5,20.5,0.0 | 0.222 | 22 | 2.05 | 23 | 33.82 | 574.2 | whole vs gcd: 5.16e-16 (max-entry); split vs gcd: 5.21e-16 (max-entry); whole vs split: 2.58e-16 (max-entry) |
| r8xyz | x4c | 8.5,0.0,0.0 | 0.917 | -1 | NaN | -1 | NaN | NaN |  |
| r8xyz | x4e | 8.5,3.5,0.0 | 0.848 | -1 | NaN | -1 | NaN | NaN |  |
| r8xyz | x4v | 8.5,3.5,3.5 | 0.792 | -1 | NaN | -1 | NaN | 5215.5 |  |
| r8xyz | x6c | 10.5,0.0,0.0 | 0.742 | -1 | NaN | -1 | NaN | NaN |  |
| r8xyz | x6e | 10.5,3.5,0.0 | 0.704 | -1 | NaN | -1 | NaN | NaN |  |
| r8xyz | x6v | 10.5,3.5,3.5 | 0.671 | -1 | NaN | -1 | NaN | 4263.4 |  |
| r8xyz | x8c | 12.5,0.0,0.0 | 0.624 | 56 | 12.23 | -1 | NaN | NaN |  |
| r8xyz | x8e | 12.5,3.5,0.0 | 0.600 | 54 | 11.62 | -1 | NaN | NaN |  |
| r8xyz | x8v | 12.5,3.5,3.5 | 0.580 | 50 | 9.88 | 52 | 112.55 | 5641.4 | whole vs gcd: 7.71e-13 (max-entry); split vs gcd: 1.76e-13 (max-entry); whole vs split: 7.26e-13 (max-entry) |
| r8xyz | x16c | 20.5,0.0,0.0 | 0.380 | 32 | 4.18 | 31 | 68.51 | NaN | whole vs split: 4.15e-16 (max-entry) |
| r8xyz | x16e | 20.5,3.5,0.0 | 0.375 | 30 | 3.74 | 30 | 68.67 | NaN | whole vs split: 4.24e-16 (max-entry) |
| r8xyz | x16v | 20.5,3.5,3.5 | 0.370 | 30 | 3.81 | 30 | 66.95 | 5198.1 | whole vs gcd: 9.43e-16 (max-entry); split vs gcd: 8.91e-16 (max-entry); whole vs split: 3.29e-16 (max-entry) |
| r8xyz | x32c | 36.5,0.0,0.0 | 0.214 | 22 | 2.23 | 23 | 46.73 | NaN | whole vs split: 2.71e-16 (max-entry) |
| r8xyz | x32e | 36.5,3.5,0.0 | 0.213 | 22 | 2.29 | 22 | 49.64 | NaN | whole vs split: 1.82e-16 (max-entry) |
| r8xyz | x32v | 36.5,3.5,3.5 | 0.212 | 22 | 2.25 | 22 | 46.21 | 5059.3 | whole vs gcd: 9.21e-16 (max-entry); split vs gcd: 9.26e-16 (max-entry); whole vs split: 1.86e-16 (max-entry) |
| r8xyz | d4c | 8.5,8.5,0.0 | 0.648 | -1 | NaN | -1 | NaN | NaN |  |
| r8xyz | d4e | 8.5,8.5,3.5 | 0.623 | 56 | 12.39 | 50 | 123.62 | 5244.0 | whole vs gcd: 3.23e-12 (max-entry); split vs gcd: 1.54e-13 (max-entry); whole vs split: 3.22e-12 (max-entry) |
| r8xyz | d8c | 12.5,12.5,0.0 | 0.441 | 36 | 5.19 | 35 | 75.72 | NaN | whole vs split: 5.72e-16 (max-entry) |
| r8xyz | d8e | 12.5,12.5,3.5 | 0.433 | 36 | 5.12 | 34 | 77.10 | 4110.7 | whole vs gcd: 4.78e-16 (max-entry); split vs gcd: 2.04e-16 (max-entry); whole vs split: 5.41e-16 (max-entry) |
| r8xyz | d16c | 20.5,20.5,0.0 | 0.269 | 24 | 2.33 | 25 | 51.58 | NaN | whole vs split: 3.89e-16 (max-entry) |
| r8xyz | d16e | 20.5,20.5,3.5 | 0.267 | 24 | 2.28 | 25 | 51.48 | 5717.4 | whole vs gcd: 5.32e-16 (max-entry); split vs gcd: 3.77e-16 (max-entry); whole vs split: 3.04e-16 (max-entry) |
| r16x | x4c | 12.5,0.0,0.0 | 0.689 | -1 | NaN | -1 | NaN | 175.7 |  |
| r16x | x6c | 14.5,0.0,0.0 | 0.594 | -1 | NaN | -1 | NaN | 145.6 |  |
| r16xy | x4c | 12.5,0.0,0.0 | 0.965 | -1 | NaN | -1 | NaN | NaN |  |
| r16xy | x4e | 12.5,7.5,0.0 | 0.827 | -1 | NaN | -1 | NaN | 2003.5 |  |
| r16xy | x6c | 14.5,0.0,0.0 | 0.832 | -1 | NaN | -1 | NaN | NaN |  |
| r16xy | x6e | 14.5,7.5,0.0 | 0.739 | -1 | NaN | -1 | NaN | 2572.8 |  |
| r16xy | x8c | 16.5,0.0,0.0 | 0.731 | -1 | NaN | -1 | NaN | NaN |  |
| r16xy | x8e | 16.5,7.5,0.0 | 0.666 | -1 | NaN | -1 | NaN | 2375.0 |  |
| r16xyz | x4c | 12.5,0.0,0.0 | 1.178 | -1 | NaN | -1 | NaN | NaN |  |
| r16xyz | x4e | 12.5,7.5,0.0 | 1.010 | -1 | NaN | -1 | NaN | NaN |  |
| r16xyz | x4v | 12.5,7.5,7.5 | 0.898 | -1 | NaN | -1 | NaN | 34259.7 |  |
| r16xyz | x6c | 14.5,0.0,0.0 | 1.015 | -1 | NaN | -1 | NaN | NaN |  |
| r16xyz | x6e | 14.5,7.5,0.0 | 0.902 | -1 | NaN | -1 | NaN | NaN |  |
| r16xyz | x6v | 14.5,7.5,7.5 | 0.819 | -1 | NaN | -1 | NaN | 43506.1 |  |
| r16xyz | x8c | 16.5,0.0,0.0 | 0.892 | -1 | NaN | -1 | NaN | NaN |  |
| r16xyz | x8e | 16.5,7.5,0.0 | 0.812 | -1 | NaN | -1 | NaN | NaN |  |
| r16xyz | x8v | 16.5,7.5,7.5 | 0.751 | -1 | NaN | -1 | NaN | 33130.4 |  |
| r16xyz | x16c | 24.5,0.0,0.0 | 0.601 | -1 | NaN | -1 | NaN | NaN |  |
| r16xyz | x16e | 24.5,7.5,0.0 | 0.575 | 54 | 11.70 | -1 | NaN | NaN |  |
| r16xyz | x16v | 24.5,7.5,7.5 | 0.551 | 52 | 10.84 | -1 | NaN | 34572.4 | whole vs gcd: 6.82e-10 (max-entry) |
## 3. Accuracy against the references present at run time, f = 1 (the full recheck against the final file is S3b)

| shape | off | R/g | whole (L) | split (L) | gcd |
|---|---|---|---|---|---|
| r2x | x1c | 2.5,0.0,0.0 | - (-1) | 3.9e-16,9.2e-16 (51) | 3.2e-15,3.2e-15 |
| r2xyz | x1c | 2.5,0.0,0.0 | - (-1) | - (-1) | - |
| r16x | x1c | 9.5,0.0,0.0 | - (-1) | - (-1) | 2.5e-15,2.5e-15 |
| r16xyz | x1c | 9.5,0.0,0.0 | - (-1) | - (-1) | - |

4 offsets with a reference.

## 2c. Far offsets, f = 1 (kap >= 4): whole-box route, split and gcd for comparison

| shape | off | R/g | rho_w | L_w | t_w | L_p | t_p | t_gcd | agreement |
|---|---|---|---|---|---|---|---|---|---|
| r2x | x4c | 5.5,0.0,0.0 | 0.375 | 28 | 2.97 | 21 | 32.42 | 28.0 | whole vs gcd: 1.04e-15 (max-entry); split vs gcd: 7.46e-16 (max-entry); whole vs split: 4.01e-16 (max-entry) |
| r2x | x6c | 7.5,0.0,0.0 | 0.275 | 22 | 2.13 | 17 | 25.13 | 33.4 | whole vs gcd: 9.57e-16 (max-entry); split vs gcd: 4.85e-16 (max-entry); whole vs split: 4.85e-16 (max-entry) |
| r2x | x8c | 9.5,0.0,0.0 | 0.217 | 20 | 1.66 | 16 | 21.83 | 31.2 | whole vs gcd: 4.34e-16 (max-entry); split vs gcd: 2.75e-16 (max-entry); whole vs split: 4.86e-16 (max-entry) |
| r2x | x16c | 17.5,0.0,0.0 | 0.118 | 16 | 1.26 | 13 | 16.34 | 31.1 | whole vs gcd: 5.43e-16 (max-entry); split vs gcd: 3.62e-16 (max-entry); whole vs split: 5.12e-16 (max-entry) |
| r2x | x32c | 33.5,0.0,0.0 | 0.062 | 12 | 0.88 | 11 | 15.07 | 32.9 | whole vs gcd: 3.02e-16 (max-entry); split vs gcd: 4.83e-16 (max-entry); whole vs split: 3.77e-16 (max-entry) |
| r2x | d4c | 5.5,5.0,0.0 | 0.277 | 22 | 2.17 | 18 | 27.81 | 30.1 | whole vs gcd: 3.97e-16 (max-entry); split vs gcd: 2.81e-16 (max-entry); whole vs split: 3.97e-16 (max-entry) |
| r2x | d8c | 9.5,9.0,0.0 | 0.158 | 16 | 1.24 | 14 | 18.39 | 24.5 | whole vs gcd: 4.17e-16 (max-entry); split vs gcd: 5.13e-16 (max-entry); whole vs split: 2.56e-16 (max-entry) |
| r2x | d16c | 17.5,17.0,0.0 | 0.084 | 14 | 1.02 | 12 | 16.48 | 26.8 | whole vs gcd: 3.32e-16 (max-entry); split vs gcd: 3.71e-16 (max-entry); whole vs split: 4.56e-16 (max-entry) |
| r2xy | x4c | 5.5,0.0,0.0 | 0.426 | 32 | 3.88 | 21 | 45.20 | NaN | whole vs split: 5.01e-16 (max-entry) |
| r2xy | x4e | 5.5,0.5,0.0 | 0.425 | 32 | 6.67 | 21 | 45.58 | 50.7 | whole vs gcd: 3.13e-16 (max-entry); split vs gcd: 4.73e-16 (max-entry); whole vs split: 3.13e-16 (max-entry) |
| r2xy | x6c | 7.5,0.0,0.0 | 0.313 | 24 | 2.35 | 18 | 34.47 | NaN | whole vs split: 8.68e-16 (max-entry) |
| r2xy | x6e | 7.5,0.5,0.0 | 0.312 | 24 | 2.41 | 18 | 36.43 | 43.5 | whole vs gcd: 5.15e-16 (max-entry); split vs gcd: 4.88e-16 (max-entry); whole vs split: 4.15e-16 (max-entry) |
| r2xy | x8c | 9.5,0.0,0.0 | 0.247 | 20 | 1.78 | 16 | 29.91 | NaN | whole vs split: 9.78e-16 (max-entry) |
| r2xy | x8e | 9.5,0.5,0.0 | 0.247 | 20 | 1.75 | 16 | 30.56 | 45.3 | whole vs gcd: 3.55e-16 (max-entry); split vs gcd: 2.87e-16 (max-entry); whole vs split: 3.75e-16 (max-entry) |
| r2xy | x16c | 17.5,0.0,0.0 | 0.134 | 16 | 1.07 | 13 | 21.10 | NaN | whole vs split: 7.47e-16 (max-entry) |
| r2xy | x16e | 17.5,0.5,0.0 | 0.134 | 16 | 1.08 | 13 | 20.88 | 41.3 | whole vs gcd: 5.85e-16 (max-entry); split vs gcd: 5.44e-16 (max-entry); whole vs split: 5.67e-16 (max-entry) |
| r2xy | x32c | 33.5,0.0,0.0 | 0.070 | 12 | 0.80 | 11 | 17.78 | NaN | whole vs split: 4.53e-16 (max-entry) |
| r2xy | x32e | 33.5,0.5,0.0 | 0.070 | 12 | 0.80 | 11 | 17.75 | 32.7 | whole vs gcd: 3.77e-16 (max-entry); split vs gcd: 4.06e-16 (max-entry); whole vs split: 2.72e-16 (max-entry) |
| r2xy | d4c | 5.5,5.5,0.0 | 0.302 | 24 | 2.18 | 17 | 35.10 | 44.3 | whole vs gcd: 2.79e-16 (max-entry); split vs gcd: 1.96e-16 (max-entry); whole vs split: 3.46e-16 (max-entry) |
| r2xy | d8c | 9.5,9.5,0.0 | 0.175 | 18 | 1.48 | 14 | 25.91 | 42.4 | whole vs gcd: 7.46e-16 (max-entry); split vs gcd: 2.70e-16 (max-entry); whole vs split: 5.52e-16 (max-entry) |
| r2xy | d16c | 17.5,17.5,0.0 | 0.095 | 14 | 1.07 | 12 | 22.38 | 47.0 | whole vs gcd: 5.03e-16 (max-entry); split vs gcd: 2.25e-16 (max-entry); whole vs split: 4.09e-16 (max-entry) |
| r2xyz | x4c | 5.5,0.0,0.0 | 0.472 | 34 | 4.63 | 21 | 58.85 | NaN | whole vs split: 4.69e-16 (max-entry) |
| r2xyz | x4e | 5.5,0.5,0.0 | 0.470 | 34 | 4.68 | 21 | 60.15 | NaN | whole vs split: 1.44e-15 (max-entry) |
| r2xyz | x4v | 5.5,0.5,0.5 | 0.469 | 34 | 4.74 | 21 | 61.13 | 97.2 | whole vs gcd: 4.86e-16 (max-entry); split vs gcd: 3.27e-16 (max-entry); whole vs split: 4.88e-16 (max-entry) |
| r2xyz | x6c | 7.5,0.0,0.0 | 0.346 | 26 | 2.67 | 18 | 51.48 | NaN | whole vs split: 1.63e-16 (max-entry) |
| r2xyz | x6e | 7.5,0.5,0.0 | 0.346 | 26 | 2.64 | 18 | 52.48 | NaN | whole vs split: 1.65e-16 (max-entry) |
| r2xyz | x6v | 7.5,0.5,0.5 | 0.345 | 26 | 2.56 | 18 | 51.60 | 95.0 | whole vs gcd: 3.35e-16 (max-entry); split vs gcd: 1.66e-16 (max-entry); whole vs split: 4.16e-16 (max-entry) |
| r2xyz | x8c | 9.5,0.0,0.0 | 0.273 | 22 | 2.03 | 16 | 45.25 | NaN | whole vs split: 5.74e-16 (max-entry) |
| r2xyz | x8e | 9.5,0.5,0.0 | 0.273 | 22 | 1.99 | 16 | 41.89 | NaN | whole vs split: 3.13e-16 (max-entry) |
| r2xyz | x8v | 9.5,0.5,0.5 | 0.273 | 22 | 2.11 | 16 | 41.65 | 89.4 | whole vs gcd: 3.59e-16 (max-entry); split vs gcd: 4.23e-16 (max-entry); whole vs split: 4.28e-16 (max-entry) |
| r2xyz | x16c | 17.5,0.0,0.0 | 0.148 | 16 | 1.30 | 13 | 32.34 | NaN | whole vs split: 4.59e-16 (max-entry) |
| r2xyz | x16e | 17.5,0.5,0.0 | 0.148 | 16 | 1.27 | 13 | 34.24 | NaN | whole vs split: 4.08e-16 (max-entry) |
| r2xyz | x16v | 17.5,0.5,0.5 | 0.148 | 16 | 1.21 | 13 | 32.06 | 82.2 | whole vs gcd: 3.63e-16 (max-entry); split vs gcd: 3.35e-16 (max-entry); whole vs split: 5.20e-16 (max-entry) |
| r2xyz | x32c | 33.5,0.0,0.0 | 0.078 | 14 | 1.38 | 11 | 28.96 | NaN | whole vs split: 6.40e-16 (max-entry) |
| r2xyz | x32e | 33.5,0.5,0.0 | 0.078 | 14 | 1.10 | 11 | 31.60 | NaN | whole vs split: 2.72e-16 (max-entry) |
| r2xyz | x32v | 33.5,0.5,0.5 | 0.078 | 14 | 1.06 | 11 | 29.19 | 78.2 | whole vs gcd: 4.83e-16 (max-entry); split vs gcd: 2.13e-16 (max-entry); whole vs split: 5.06e-16 (max-entry) |
| r2xyz | d4c | 5.5,5.5,0.0 | 0.334 | 26 | 2.67 | 18 | 48.68 | NaN | whole vs split: 2.52e-16 (max-entry) |
| r2xyz | d4e | 5.5,5.5,0.5 | 0.333 | 26 | 2.55 | 18 | 49.29 | 90.8 | whole vs gcd: 2.82e-16 (max-entry); split vs gcd: 1.99e-16 (max-entry); whole vs split: 4.23e-16 (max-entry) |
| r2xyz | d8c | 9.5,9.5,0.0 | 0.193 | 18 | 1.47 | 14 | 37.43 | NaN | whole vs split: 8.37e-16 (max-entry) |
| r2xyz | d8e | 9.5,9.5,0.5 | 0.193 | 18 | 1.55 | 14 | 34.74 | 85.0 | whole vs gcd: 7.23e-16 (max-entry); split vs gcd: 5.24e-16 (max-entry); whole vs split: 2.58e-16 (max-entry) |
| r2xyz | d16c | 17.5,17.5,0.0 | 0.105 | 14 | 1.08 | 12 | 33.08 | NaN | whole vs split: 3.42e-16 (max-entry) |
| r2xyz | d16e | 17.5,17.5,0.5 | 0.105 | 14 | 1.09 | 12 | 30.10 | 88.0 | whole vs gcd: 3.42e-16 (max-entry); split vs gcd: 3.03e-16 (max-entry); whole vs split: 3.56e-16 (max-entry) |
| r4x | x4c | 6.5,0.0,0.0 | 0.442 | 36 | 4.89 | 27 | 38.69 | 46.9 | whole vs gcd: 8.20e-16 (max-entry); split vs gcd: 4.37e-16 (max-entry); whole vs split: 4.22e-16 (max-entry) |
| r4x | x6c | 8.5,0.0,0.0 | 0.338 | 28 | 3.06 | 23 | 27.44 | 42.5 | whole vs gcd: 6.20e-16 (max-entry); split vs gcd: 6.11e-16 (max-entry); whole vs split: 7.13e-16 (max-entry) |
| r4x | x8c | 10.5,0.0,0.0 | 0.274 | 24 | 2.32 | 20 | 23.76 | 42.2 | whole vs gcd: 3.03e-16 (max-entry); split vs gcd: 6.06e-16 (max-entry); whole vs split: 4.20e-16 (max-entry) |
| r4x | x16c | 18.5,0.0,0.0 | 0.155 | 18 | 1.37 | 16 | 18.01 | 42.4 | whole vs gcd: 2.73e-16 (max-entry); split vs gcd: 4.41e-16 (max-entry); whole vs split: 3.12e-16 (max-entry) |
| r4x | x32c | 34.5,0.0,0.0 | 0.083 | 14 | 1.02 | 14 | 14.85 | 35.9 | whole vs gcd: 4.47e-16 (max-entry); split vs gcd: 4.47e-16 (max-entry); whole vs split: 5.06e-16 (max-entry) |
| r4x | d4c | 6.5,5.0,0.0 | 0.350 | 30 | 3.38 | 24 | 29.12 | 34.7 | whole vs gcd: 7.78e-16 (max-entry); split vs gcd: 7.77e-16 (max-entry); whole vs split: 1.56e-15 (max-entry) |
| r4x | d8c | 10.5,9.0,0.0 | 0.208 | 20 | 2.15 | 18 | 20.68 | 44.6 | whole vs gcd: 4.65e-16 (max-entry); split vs gcd: 1.47e-16 (max-entry); whole vs split: 5.01e-16 (max-entry) |
| r4x | d16c | 18.5,17.0,0.0 | 0.114 | 16 | 1.23 | 15 | 16.61 | 44.3 | whole vs gcd: 2.57e-16 (max-entry); split vs gcd: 3.49e-16 (max-entry); whole vs split: 1.28e-16 (max-entry) |
| r4xy | x4c | 6.5,0.0,0.0 | 0.565 | 48 | 9.60 | 36 | 62.36 | NaN | whole vs split: 3.52e-16 (max-entry) |
| r4xy | x4e | 6.5,1.5,0.0 | 0.551 | 46 | 9.01 | 34 | 59.20 | 150.7 | whole vs gcd: 1.14e-15 (max-entry); split vs gcd: 2.26e-16 (max-entry); whole vs split: 1.28e-15 (max-entry) |
| r4xy | x6c | 8.5,0.0,0.0 | 0.432 | 34 | 4.23 | 27 | 46.70 | NaN | whole vs split: 4.29e-16 (max-entry) |
| r4xy | x6e | 8.5,1.5,0.0 | 0.426 | 34 | 4.56 | 27 | 46.56 | 161.8 | whole vs gcd: 7.50e-16 (max-entry); split vs gcd: 2.85e-16 (max-entry); whole vs split: 5.95e-16 (max-entry) |
| r4xy | x8c | 10.5,0.0,0.0 | 0.350 | 28 | 2.91 | 23 | 36.46 | NaN | whole vs split: 1.24e-15 (max-entry) |
| r4xy | x8e | 10.5,1.5,0.0 | 0.346 | 28 | 2.97 | 23 | 35.13 | 138.4 | whole vs gcd: 6.38e-16 (max-entry); split vs gcd: 7.27e-16 (max-entry); whole vs split: 4.03e-16 (max-entry) |
| r4xy | x16c | 18.5,0.0,0.0 | 0.199 | 20 | 1.78 | 18 | 27.18 | NaN | whole vs split: 3.87e-16 (max-entry) |
| r4xy | x16e | 18.5,1.5,0.0 | 0.198 | 20 | 1.68 | 18 | 26.01 | 155.0 | whole vs gcd: 7.01e-16 (max-entry); split vs gcd: 8.01e-16 (max-entry); whole vs split: 1.50e-15 (max-entry) |
| r4xy | x32c | 34.5,0.0,0.0 | 0.106 | 16 | 1.17 | 15 | 21.93 | NaN | whole vs split: 1.20e-15 (max-entry) |
| r4xy | x32e | 34.5,1.5,0.0 | 0.106 | 16 | 1.18 | 15 | 22.22 | 165.0 | whole vs gcd: 9.26e-16 (max-entry); split vs gcd: 6.38e-16 (max-entry); whole vs split: 3.54e-16 (max-entry) |
| r4xy | d4c | 6.5,6.5,0.0 | 0.400 | 32 | 4.13 | 25 | 46.85 | 129.3 | whole vs gcd: 3.67e-16 (max-entry); split vs gcd: 4.32e-16 (max-entry); whole vs split: 4.05e-16 (max-entry) |
| r4xy | d8c | 10.5,10.5,0.0 | 0.247 | 22 | 2.12 | 20 | 32.07 | 148.5 | whole vs gcd: 2.82e-16 (max-entry); split vs gcd: 3.18e-16 (max-entry); whole vs split: 2.00e-16 (max-entry) |
| r4xy | d16c | 18.5,18.5,0.0 | 0.140 | 16 | 1.45 | 16 | 25.45 | 154.4 | whole vs gcd: 2.41e-16 (max-entry); split vs gcd: 4.35e-16 (max-entry); whole vs split: 5.12e-16 (max-entry) |
| r4xyz | x4c | 6.5,0.0,0.0 | 0.666 | -1 | NaN | 48 | 101.36 | NaN |  |
| r4xyz | x4e | 6.5,1.5,0.0 | 0.649 | -1 | NaN | 45 | 101.23 | NaN |  |
| r4xyz | x4v | 6.5,1.5,1.5 | 0.633 | 54 | 11.48 | 43 | 98.47 | 620.7 | whole vs gcd: 5.29e-16 (max-entry); split vs gcd: 1.51e-16 (max-entry); whole vs split: 6.80e-16 (max-entry) |
| r4xyz | x6c | 8.5,0.0,0.0 | 0.509 | 40 | 6.54 | 33 | 76.33 | NaN | whole vs split: 5.66e-16 (max-entry) |
| r4xyz | x6e | 8.5,1.5,0.0 | 0.502 | 38 | 5.65 | 32 | 72.16 | NaN | whole vs split: 7.75e-16 (max-entry) |
| r4xyz | x6v | 8.5,1.5,1.5 | 0.494 | 38 | 5.77 | 31 | 70.78 | 577.8 | whole vs gcd: 2.61e-16 (max-entry); split vs gcd: 3.17e-16 (max-entry); whole vs split: 5.67e-16 (max-entry) |
| r4xyz | x8c | 10.5,0.0,0.0 | 0.412 | 32 | 4.05 | 26 | 59.08 | NaN | whole vs split: 9.03e-16 (max-entry) |
| r4xyz | x8e | 10.5,1.5,0.0 | 0.408 | 32 | 3.81 | 26 | 58.99 | NaN | whole vs split: 5.85e-16 (max-entry) |
| r4xyz | x8v | 10.5,1.5,1.5 | 0.404 | 30 | 3.42 | 26 | 59.15 | 605.3 | whole vs gcd: 6.76e-16 (max-entry); split vs gcd: 2.94e-16 (max-entry); whole vs split: 4.83e-16 (max-entry) |
| r4xyz | x16c | 18.5,0.0,0.0 | 0.234 | 20 | 1.71 | 19 | 49.60 | NaN | whole vs split: 2.46e-16 (max-entry) |
| r4xyz | x16e | 18.5,1.5,0.0 | 0.233 | 20 | 1.83 | 19 | 44.01 | NaN | whole vs split: 6.18e-16 (max-entry) |
| r4xyz | x16v | 18.5,1.5,1.5 | 0.233 | 20 | 1.80 | 19 | 39.52 | 580.5 | whole vs gcd: 6.36e-16 (max-entry); split vs gcd: 2.79e-16 (max-entry); whole vs split: 3.94e-16 (max-entry) |
| r4xyz | x32c | 34.5,0.0,0.0 | 0.126 | 16 | 1.99 | 16 | 32.59 | NaN | whole vs split: 5.01e-16 (max-entry) |
| r4xyz | x32e | 34.5,1.5,0.0 | 0.125 | 16 | 1.19 | 16 | 32.56 | NaN | whole vs split: 7.09e-16 (max-entry) |
| r4xyz | x32v | 34.5,1.5,1.5 | 0.125 | 16 | 1.33 | 16 | 34.25 | 623.4 | whole vs gcd: 3.97e-16 (max-entry); split vs gcd: 3.55e-16 (max-entry); whole vs split: 4.83e-16 (max-entry) |
| r4xyz | d4c | 6.5,6.5,0.0 | 0.471 | 36 | 5.37 | 29 | 69.65 | NaN | whole vs split: 8.68e-16 (max-entry) |
| r4xyz | d4e | 6.5,6.5,1.5 | 0.465 | 36 | 5.34 | 28 | 64.47 | 578.6 | whole vs gcd: 3.19e-16 (max-entry); split vs gcd: 9.90e-17 (max-entry); whole vs split: 3.20e-16 (max-entry) |
| r4xyz | d8c | 10.5,10.5,0.0 | 0.292 | 24 | 2.37 | 21 | 48.49 | NaN | whole vs split: 5.80e-16 (max-entry) |
| r4xyz | d8e | 10.5,10.5,1.5 | 0.290 | 24 | 2.29 | 21 | 44.46 | 633.3 | whole vs gcd: 1.01e-15 (max-entry); split vs gcd: 8.94e-16 (max-entry); whole vs split: 3.07e-16 (max-entry) |
| r4xyz | d16c | 18.5,18.5,0.0 | 0.166 | 18 | 1.55 | 17 | 38.81 | NaN | whole vs split: 1.91e-16 (max-entry) |
| r4xyz | d16e | 18.5,18.5,1.5 | 0.165 | 18 | 1.57 | 17 | 36.69 | 580.4 | whole vs gcd: 1.82e-16 (max-entry); split vs gcd: 3.85e-16 (max-entry); whole vs split: 3.44e-16 (max-entry) |
| r8x | x4c | 8.5,0.0,0.0 | 0.555 | 54 | 11.71 | 44 | 66.89 | 76.4 | whole vs gcd: 1.29e-15 (max-entry); split vs gcd: 4.35e-16 (max-entry); whole vs split: 1.62e-15 (max-entry) |
| r8x | x6c | 10.5,0.0,0.0 | 0.449 | 42 | 7.68 | 35 | 48.41 | 82.3 | whole vs gcd: 8.30e-16 (max-entry); split vs gcd: 7.70e-16 (max-entry); whole vs split: 1.13e-15 (max-entry) |
| r8x | x8c | 12.5,0.0,0.0 | 0.377 | 34 | 4.88 | 31 | 35.94 | 81.2 | whole vs gcd: 2.66e-16 (max-entry); split vs gcd: 1.10e-15 (max-entry); whole vs split: 1.23e-15 (max-entry) |
| r8x | x16c | 20.5,0.0,0.0 | 0.230 | 24 | 3.44 | 23 | 24.16 | 78.3 | whole vs gcd: 1.17e-15 (max-entry); split vs gcd: 7.08e-16 (max-entry); whole vs split: 6.07e-16 (max-entry) |
| r8x | x32c | 36.5,0.0,0.0 | 0.129 | 18 | 1.83 | 18 | 18.80 | 61.2 | whole vs gcd: 4.84e-16 (max-entry); split vs gcd: 1.11e-15 (max-entry); whole vs split: 9.16e-16 (max-entry) |
| r8x | d4c | 8.5,5.0,0.0 | 0.478 | 44 | 8.82 | 38 | 48.87 | 72.8 | whole vs gcd: 8.80e-16 (max-entry); split vs gcd: 9.87e-16 (max-entry); whole vs split: 8.98e-16 (max-entry) |
| r8x | d8c | 12.5,9.0,0.0 | 0.306 | 28 | 3.31 | 27 | 29.13 | 69.6 | whole vs gcd: 6.54e-16 (max-entry); split vs gcd: 7.49e-16 (max-entry); whole vs split: 8.24e-16 (max-entry) |
| r8x | d16c | 20.5,17.0,0.0 | 0.177 | 20 | 1.85 | 20 | 20.82 | 65.9 | whole vs gcd: 3.81e-16 (max-entry); split vs gcd: 2.83e-16 (max-entry); whole vs split: 4.01e-16 (max-entry) |
| r8xy | x4c | 8.5,0.0,0.0 | 0.758 | -1 | NaN | -1 | NaN | NaN |  |
| r8xy | x4e | 8.5,3.5,0.0 | 0.701 | -1 | NaN | -1 | NaN | 500.4 |  |
| r8xy | x6c | 10.5,0.0,0.0 | 0.614 | -1 | NaN | -1 | NaN | NaN |  |
| r8xy | x6e | 10.5,3.5,0.0 | 0.582 | 54 | 11.19 | 51 | 99.64 | 706.1 | whole vs gcd: 5.92e-14 (max-entry); split vs gcd: 1.73e-15 (max-entry); whole vs split: 5.74e-14 (max-entry) |
| r8xy | x8c | 12.5,0.0,0.0 | 0.515 | 46 | 8.29 | 43 | 73.94 | NaN | whole vs split: 7.61e-16 (max-entry) |
| r8xy | x8e | 12.5,3.5,0.0 | 0.496 | 44 | 8.07 | 39 | 68.83 | 647.1 | whole vs gcd: 8.11e-16 (max-entry); split vs gcd: 5.35e-16 (max-entry); whole vs split: 1.30e-15 (max-entry) |
| r8xy | x16c | 20.5,0.0,0.0 | 0.314 | 28 | 3.31 | 27 | 42.90 | NaN | whole vs split: 6.51e-16 (max-entry) |
| r8xy | x16e | 20.5,3.5,0.0 | 0.310 | 28 | 3.09 | 27 | 39.09 | 712.7 | whole vs gcd: 9.32e-16 (max-entry); split vs gcd: 4.76e-16 (max-entry); whole vs split: 1.17e-15 (max-entry) |
| r8xy | x32c | 36.5,0.0,0.0 | 0.176 | 20 | 1.76 | 21 | 28.87 | NaN | whole vs split: 4.02e-16 (max-entry) |
| r8xy | x32e | 36.5,3.5,0.0 | 0.176 | 20 | 1.79 | 21 | 28.33 | 661.0 | whole vs gcd: 5.72e-16 (max-entry); split vs gcd: 3.73e-16 (max-entry); whole vs split: 3.73e-16 (max-entry) |
| r8xy | d4c | 8.5,8.5,0.0 | 0.536 | 48 | 9.20 | 42 | 80.46 | 710.0 | whole vs gcd: 2.52e-15 (max-entry); split vs gcd: 8.41e-16 (max-entry); whole vs split: 3.36e-15 (max-entry) |
| r8xy | d8c | 12.5,12.5,0.0 | 0.364 | 32 | 3.90 | 30 | 44.22 | 686.2 | whole vs gcd: 5.57e-16 (max-entry); split vs gcd: 4.40e-16 (max-entry); whole vs split: 4.05e-16 (max-entry) |
| r8xy | d16c | 20.5,20.5,0.0 | 0.222 | 22 | 2.05 | 23 | 33.82 | 574.2 | whole vs gcd: 5.16e-16 (max-entry); split vs gcd: 5.21e-16 (max-entry); whole vs split: 2.58e-16 (max-entry) |
| r8xyz | x4c | 8.5,0.0,0.0 | 0.917 | -1 | NaN | -1 | NaN | NaN |  |
| r8xyz | x4e | 8.5,3.5,0.0 | 0.848 | -1 | NaN | -1 | NaN | NaN |  |
| r8xyz | x4v | 8.5,3.5,3.5 | 0.792 | -1 | NaN | -1 | NaN | 5215.5 |  |
| r8xyz | x6c | 10.5,0.0,0.0 | 0.742 | -1 | NaN | -1 | NaN | NaN |  |
| r8xyz | x6e | 10.5,3.5,0.0 | 0.704 | -1 | NaN | -1 | NaN | NaN |  |
| r8xyz | x6v | 10.5,3.5,3.5 | 0.671 | -1 | NaN | -1 | NaN | 4263.4 |  |
| r8xyz | x8c | 12.5,0.0,0.0 | 0.624 | 56 | 12.23 | -1 | NaN | NaN |  |
| r8xyz | x8e | 12.5,3.5,0.0 | 0.600 | 54 | 11.62 | -1 | NaN | NaN |  |
| r8xyz | x8v | 12.5,3.5,3.5 | 0.580 | 50 | 9.88 | 52 | 112.55 | 5641.4 | whole vs gcd: 7.71e-13 (max-entry); split vs gcd: 1.76e-13 (max-entry); whole vs split: 7.26e-13 (max-entry) |
| r8xyz | x16c | 20.5,0.0,0.0 | 0.380 | 32 | 4.18 | 31 | 68.51 | NaN | whole vs split: 4.15e-16 (max-entry) |
| r8xyz | x16e | 20.5,3.5,0.0 | 0.375 | 30 | 3.74 | 30 | 68.67 | NaN | whole vs split: 4.24e-16 (max-entry) |
| r8xyz | x16v | 20.5,3.5,3.5 | 0.370 | 30 | 3.81 | 30 | 66.95 | 5198.1 | whole vs gcd: 9.43e-16 (max-entry); split vs gcd: 8.91e-16 (max-entry); whole vs split: 3.29e-16 (max-entry) |
| r8xyz | x32c | 36.5,0.0,0.0 | 0.214 | 22 | 2.23 | 23 | 46.73 | NaN | whole vs split: 2.71e-16 (max-entry) |
| r8xyz | x32e | 36.5,3.5,0.0 | 0.213 | 22 | 2.29 | 22 | 49.64 | NaN | whole vs split: 1.82e-16 (max-entry) |
| r8xyz | x32v | 36.5,3.5,3.5 | 0.212 | 22 | 2.25 | 22 | 46.21 | 5059.3 | whole vs gcd: 9.21e-16 (max-entry); split vs gcd: 9.26e-16 (max-entry); whole vs split: 1.86e-16 (max-entry) |
| r8xyz | d4c | 8.5,8.5,0.0 | 0.648 | -1 | NaN | -1 | NaN | NaN |  |
| r8xyz | d4e | 8.5,8.5,3.5 | 0.623 | 56 | 12.39 | 50 | 123.62 | 5244.0 | whole vs gcd: 3.23e-12 (max-entry); split vs gcd: 1.54e-13 (max-entry); whole vs split: 3.22e-12 (max-entry) |
| r8xyz | d8c | 12.5,12.5,0.0 | 0.441 | 36 | 5.19 | 35 | 75.72 | NaN | whole vs split: 5.72e-16 (max-entry) |
| r8xyz | d8e | 12.5,12.5,3.5 | 0.433 | 36 | 5.12 | 34 | 77.10 | 4110.7 | whole vs gcd: 4.78e-16 (max-entry); split vs gcd: 2.04e-16 (max-entry); whole vs split: 5.41e-16 (max-entry) |
| r8xyz | d16c | 20.5,20.5,0.0 | 0.269 | 24 | 2.33 | 25 | 51.58 | NaN | whole vs split: 3.89e-16 (max-entry) |
| r8xyz | d16e | 20.5,20.5,3.5 | 0.267 | 24 | 2.28 | 25 | 51.48 | 5717.4 | whole vs gcd: 5.32e-16 (max-entry); split vs gcd: 3.77e-16 (max-entry); whole vs split: 3.04e-16 (max-entry) |
| r16x | x4c | 12.5,0.0,0.0 | 0.689 | -1 | NaN | -1 | NaN | 175.7 |  |
| r16x | x6c | 14.5,0.0,0.0 | 0.594 | -1 | NaN | -1 | NaN | 145.6 |  |
| r16xy | x4c | 12.5,0.0,0.0 | 0.965 | -1 | NaN | -1 | NaN | NaN |  |
| r16xy | x4e | 12.5,7.5,0.0 | 0.827 | -1 | NaN | -1 | NaN | 2003.5 |  |
| r16xy | x6c | 14.5,0.0,0.0 | 0.832 | -1 | NaN | -1 | NaN | NaN |  |
| r16xy | x6e | 14.5,7.5,0.0 | 0.739 | -1 | NaN | -1 | NaN | 2572.8 |  |
| r16xy | x8c | 16.5,0.0,0.0 | 0.731 | -1 | NaN | -1 | NaN | NaN |  |
| r16xy | x8e | 16.5,7.5,0.0 | 0.666 | -1 | NaN | -1 | NaN | 2375.0 |  |
| r16xyz | x4c | 12.5,0.0,0.0 | 1.178 | -1 | NaN | -1 | NaN | NaN |  |
| r16xyz | x4e | 12.5,7.5,0.0 | 1.010 | -1 | NaN | -1 | NaN | NaN |  |
| r16xyz | x4v | 12.5,7.5,7.5 | 0.898 | -1 | NaN | -1 | NaN | 34259.7 |  |
| r16xyz | x6c | 14.5,0.0,0.0 | 1.015 | -1 | NaN | -1 | NaN | NaN |  |
| r16xyz | x6e | 14.5,7.5,0.0 | 0.902 | -1 | NaN | -1 | NaN | NaN |  |
| r16xyz | x6v | 14.5,7.5,7.5 | 0.819 | -1 | NaN | -1 | NaN | 43506.1 |  |
| r16xyz | x8c | 16.5,0.0,0.0 | 0.892 | -1 | NaN | -1 | NaN | NaN |  |
| r16xyz | x8e | 16.5,7.5,0.0 | 0.812 | -1 | NaN | -1 | NaN | NaN |  |
| r16xyz | x8v | 16.5,7.5,7.5 | 0.751 | -1 | NaN | -1 | NaN | 33130.4 |  |
| r16xyz | x16c | 24.5,0.0,0.0 | 0.601 | -1 | NaN | -1 | NaN | NaN |  |
| r16xyz | x16e | 24.5,7.5,0.0 | 0.575 | 54 | 11.70 | -1 | NaN | NaN |  |
| r16xyz | x16v | 24.5,7.5,7.5 | 0.551 | 52 | 10.84 | -1 | NaN | 34572.4 | whole vs gcd: 6.82e-10 (max-entry) |

## 3b. Recheck of every matrix offset against the 438-record reference cache, f = 1 (xwork/xtable/recheck2.jl -> recheck2_f1.md)

Resumed run (the first incarnation was killed by an API limit; its Julia jobs finished).  The cache held 359 records when
this round was assigned and 438 when recheck2 ran (other agents appending); every offset of Sections 2 and 2c with a
record is below.  farx.jl carries the n-cut fix of S2d (cutMax, whole-box radii), so the whole-box column is the FIXED
route; the S2/S2c whole-box numbers for r8xy x6e, r8xyz x8v/d4e, r16xyz x16v were the broken ones (S2d).
Columns: err = (max-entry-normalised, worst per-entry) against the 220-bit reference; dig = digits lost =
log10(|x - ref|/(eps |ref|)) worst entry; bound_A'/max|T| = the Theorem A' tail bound at the selected L over the
reference's max entry (the a priori certificate actually delivered); est_X/max|T| = the over-statement of the scale;
Lambda_gcd = sum_j max|G_j| / (N_t max|sum_j G_j|) of the gcd average; offlat = not a gcd-lattice position (S5b);
L_p = -1: some piece of the 27-split needs L > 56 or its type has no table (the piece types are only built for
offsets where the whole box fails).

Summary: 272 offsets with a reference.  Whole box evaluated at 147 of them: worst 1.4e-15 (r2x d1c), every dig_w <= 2.2
(i.e. <= 160 eps on the worst entry); split evaluated at 152: worst 1.5e-15 (r4x x2c); gcd average at 156: worst 5.1e-15
(r16xyz x8v, N_s = 4096 terms, Lambda 1.66).  bound_A'/max|T| at the selected L: 3.1e-14 at most (tol 1e-14 against
est_X, which over-states max|T| by 0.50-5.80; the 0.50 is r16x x1c, where est UNDER-states by 2x).  No route is off
the reference by more than 5.1e-15 anywhere; the 5.9e-14 .. 6.8e-10 whole-box gaps of S2/S2c are gone.

| shape | off | R/g | rho_w | L_w | err_w (max,ent) | dig_w | bound_A'/max\|T\| | est_X/max\|T\| | L_p | err_p (max,ent) | dig_p | Lambda_gcd | err_gcd (max,ent) | dig_g |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r2x | x1c | 2.5,0.0,0.0 | 0.825 | -1 | - | - | - | 1.74 | 51 | 3.9e-16,9.2e-16 | 0.6 | 1 | 3.2e-15,3.2e-15 | 1.2 |
| r2x | x2c | 3.5,0.0,0.0 | 0.589 | 48 | 2.4e-16,6.4e-16 | 0.5 | 1.1e-14 | 2.07 | 30 | 4.5e-16,1.2e-15 | 0.7 | 1 | 4.2e-16,1.1e-15 | 0.7 |
| r2x | x3c | 4.5,0.0,0.0 | 0.458 | 34 | 4.9e-16,4.9e-16 | 0.3 | 1.5e-14 | 2.28 | 24 | 3.9e-16,1.1e-15 | 0.7 | 1 | 5.7e-16,8.4e-16 | 0.6 |
| r2x | x4c | 5.5,0.0,0.0 | 0.375 | 28 | 7.3e-16,7.7e-16 | 0.5 | 1.4e-14 | 2.44 | 21 | 4.3e-16,4.3e-16 | 0.3 | 1 | 3.2e-16,7.3e-16 | 0.5 |
| r2x | x6c | 7.5,0.0,0.0 | 0.275 | 22 | 8.9e-16,8.9e-16 | 0.6 | 1.6e-14 | 2.66 | 17 | 4.2e-16,4.2e-16 | 0.3 | 1 | 8.7e-17,1.7e-16 | -0.1 |
| r2x | x8c | 9.5,0.0,0.0 | 0.217 | 20 | 3.7e-16,5.0e-16 | 0.4 | 3.1e-15 | 2.84 | 16 | 2.5e-16,2.5e-16 | 0.1 | 1 | 2.0e-16,2.0e-16 | -0.0 |
| r2x | x16c | 17.5,0.0,0.0 | 0.118 | 16 | 6.2e-16,6.2e-16 | 0.4 | 2.5e-16 | 2.23 | 13 | 3.7e-16,4.9e-16 | 0.3 | 1 | 4.0e-16,4.0e-16 | 0.3 |
| r2x | x32c | 33.5,0.0,0.0 | 0.062 | 12 | 2.6e-16,2.6e-16 | 0.1 | 4.0e-15 | 1.55 | 11 | 3.6e-16,4.1e-16 | 0.3 | 1 | 3.4e-16,3.4e-16 | 0.2 |
| r2x | d1c | 2.5,2.0,0.0 | 0.644 | 56 | 1.4e-15,2.3e-15 | 1.0 | 2.1e-14 | 3.15 | 36 | 2.2e-16,4.7e-16 | 0.3 | 1 | 4.4e-16,6.8e-16 | 0.5 |
| r2x | d2c | 3.5,3.0,0.0 | 0.447 | 34 | 9.7e-16,9.7e-16 | 0.6 | 1.1e-14 | 3.72 | 24 | 2.2e-16,4.1e-16 | 0.3 | 1 | 8.2e-16,8.2e-16 | 0.6 |
| r2x | d4c | 5.5,5.0,0.0 | 0.277 | 22 | 3.0e-16,3.0e-16 | 0.1 | 3.0e-14 | 4.27 | 18 | 2.6e-16,3.6e-16 | 0.2 | 1 | 1.9e-16,2.3e-16 | 0.0 |
| r2x | d8c | 9.5,9.0,0.0 | 0.158 | 16 | 4.1e-16,5.2e-16 | 0.4 | 2.2e-14 | 2.82 | 14 | 2.6e-16,3.7e-16 | 0.2 | 1 | 4.3e-16,6.3e-16 | 0.5 |
| r2x | d16c | 17.5,17.0,0.0 | 0.084 | 14 | 2.9e-16,5.2e-16 | 0.4 | 4.1e-16 | 1.80 | 12 | 3.1e-16,4.5e-16 | 0.3 | 1 | 1.4e-16,2.6e-16 | 0.1 |
| r2xy | x1c | 2.5,0.0,0.0 | 0.938 | -1 | - | - | - | 2.00 | 55 | 8.2e-16,8.2e-16 | 0.6 | offlat | offlat | - |
| r2xy | x1e | 2.5,0.5,0.0 | 0.920 | -1 | - | - | - | 2.08 | 51 | 1.1e-15,1.3e-15 | 0.8 | 1 | 1.7e-15,1.7e-15 | 0.9 |
| r2xy | x2c | 3.5,0.0,0.0 | 0.670 | -1 | - | - | - | 2.20 | 31 | 5.7e-16,1.2e-15 | 0.7 | offlat | offlat | - |
| r2xy | x2e | 3.5,0.5,0.0 | 0.663 | -1 | - | - | - | 2.26 | 31 | 4.1e-16,6.5e-16 | 0.5 | 1 | 3.0e-16,9.0e-16 | 0.6 |
| r2xy | x3c | 4.5,0.0,0.0 | 0.521 | 40 | 2.5e-16,2.5e-16 | 0.0 | 1.0e-14 | 2.36 | 24 | 6.1e-16,6.1e-16 | 0.4 | offlat | offlat | - |
| r2xy | x3e | 4.5,0.5,0.0 | 0.518 | 40 | 1.3e-15,1.3e-15 | 0.8 | 8.1e-15 | 2.40 | 24 | 3.6e-16,6.0e-16 | 0.4 | 1 | 2.7e-16,9.6e-16 | 0.6 |
| r2xy | x4c | 5.5,0.0,0.0 | 0.426 | 32 | 5.3e-16,1.1e-15 | 0.7 | 7.3e-15 | 2.49 | 21 | 2.1e-16,5.6e-16 | 0.4 | offlat | offlat | - |
| r2xy | x4e | 5.5,0.5,0.0 | 0.425 | 32 | 5.7e-16,7.3e-16 | 0.5 | 6.4e-15 | 2.52 | 21 | 2.5e-16,5.5e-16 | 0.4 | 1 | 2.8e-16,8.0e-16 | 0.6 |
| r2xy | x6c | 7.5,0.0,0.0 | 0.313 | 24 | 4.7e-16,8.6e-16 | 0.6 | 1.3e-14 | 2.69 | 18 | 4.5e-16,4.5e-16 | 0.3 | offlat | offlat | - |
| r2xy | x6e | 7.5,0.5,0.0 | 0.312 | 24 | 4.2e-16,1.9e-15 | 0.9 | 1.2e-14 | 2.70 | 18 | 4.2e-16,6.1e-16 | 0.4 | 1 | 1.1e-16,4.8e-16 | 0.3 |
| r2xy | x8c | 9.5,0.0,0.0 | 0.247 | 20 | 9.5e-16,9.5e-16 | 0.6 | 2.6e-14 | 2.86 | 16 | 2.4e-16,3.3e-16 | 0.2 | offlat | offlat | - |
| r2xy | x8e | 9.5,0.5,0.0 | 0.247 | 20 | 3.2e-16,3.2e-15 | 1.2 | 2.5e-14 | 2.87 | 16 | 1.8e-16,2.5e-16 | 0.1 | 1 | 1.4e-16,1.9e-16 | -0.1 |
| r2xy | x16c | 17.5,0.0,0.0 | 0.134 | 16 | 6.6e-16,6.6e-16 | 0.5 | 1.2e-15 | 2.23 | 13 | 4.6e-16,4.6e-16 | 0.3 | offlat | offlat | - |
| r2xy | x16e | 17.5,0.5,0.0 | 0.134 | 16 | 4.3e-16,4.3e-16 | 0.3 | 1.2e-15 | 2.23 | 13 | 6.0e-16,6.0e-16 | 0.4 | 1 | 3.5e-16,4.0e-16 | 0.3 |
| r2xy | x32c | 33.5,0.0,0.0 | 0.070 | 12 | 5.0e-16,5.0e-16 | 0.4 | 1.2e-14 | 1.55 | 11 | 8.7e-17,8.7e-17 | -0.4 | offlat | offlat | - |
| r2xy | x32e | 33.5,0.5,0.0 | 0.070 | 12 | 2.5e-16,2.5e-16 | 0.1 | 1.2e-14 | 1.55 | 11 | 2.9e-16,3.5e-16 | 0.2 | 1 | 1.6e-16,2.5e-16 | 0.1 |
| r2xy | d1c | 2.5,2.5,0.0 | 0.663 | -1 | - | - | - | 3.22 | 36 | 9.6e-16,9.6e-16 | 0.6 | 1 | 3.0e-16,6.0e-16 | 0.4 |
| r2xy | d2c | 3.5,3.5,0.0 | 0.474 | 36 | 4.4e-16,7.0e-16 | 0.5 | 1.0e-14 | 3.77 | 24 | 5.2e-16,9.4e-16 | 0.6 | 1 | 4.4e-16,4.4e-16 | 0.3 |
| r2xy | d4c | 5.5,5.5,0.0 | 0.302 | 24 | 1.9e-16,2.5e-16 | 0.0 | 8.4e-15 | 4.28 | 17 | 2.8e-16,2.8e-16 | 0.1 | 1 | 2.8e-16,2.8e-16 | 0.1 |
| r2xy | d8c | 9.5,9.5,0.0 | 0.175 | 18 | 4.7e-16,6.9e-16 | 0.5 | 1.5e-15 | 2.76 | 14 | 2.3e-16,3.6e-16 | 0.2 | 1 | 2.9e-16,4.2e-16 | 0.3 |
| r2xy | d16c | 17.5,17.5,0.0 | 0.095 | 14 | 3.0e-16,5.5e-16 | 0.4 | 1.3e-15 | 1.79 | 12 | 1.9e-16,3.5e-16 | 0.2 | 1 | 2.3e-16,4.2e-16 | 0.3 |
| r2xyz | x1c | 2.5,0.0,0.0 | 1.039 | -1 | - | - | - | 2.29 | -1 | - | - | offlat | offlat | - |
| r2xyz | x1e | 2.5,0.5,0.0 | 1.019 | -1 | - | - | - | 2.38 | 56 | 8.5e-16,9.9e-16 | 0.6 | offlat | offlat | - |
| r2xyz | x1v | 2.5,0.5,0.5 | 1.000 | -1 | - | - | - | 2.47 | 52 | 4.2e-16,4.2e-16 | 0.3 | 1 | 1.0e-15,1.0e-15 | 0.7 |
| r2xyz | x2c | 3.5,0.0,0.0 | 0.742 | -1 | - | - | - | 2.34 | 32 | 3.3e-16,4.9e-16 | 0.3 | offlat | offlat | - |
| r2xyz | x2e | 3.5,0.5,0.0 | 0.735 | -1 | - | - | - | 2.40 | 32 | 2.2e-16,6.4e-16 | 0.5 | offlat | offlat | - |
| r2xyz | x2v | 3.5,0.5,0.5 | 0.728 | -1 | - | - | - | 2.47 | 31 | 1.8e-16,1.0e-15 | 0.7 | 1 | 2.1e-16,6.0e-16 | 0.4 |
| r2xyz | x3c | 4.5,0.0,0.0 | 0.577 | 46 | 4.9e-16,9.0e-16 | 0.6 | 7.1e-15 | 2.45 | 25 | 1.7e-16,4.9e-16 | 0.3 | offlat | offlat | - |
| r2xyz | x3e | 4.5,0.5,0.0 | 0.574 | 44 | 4.9e-16,1.5e-15 | 0.8 | 2.0e-14 | 2.49 | 25 | 1.7e-16,4.7e-16 | 0.3 | offlat | offlat | - |
| r2xyz | x3v | 4.5,0.5,0.5 | 0.570 | 44 | 6.6e-16,9.1e-16 | 0.6 | 1.5e-14 | 2.53 | 25 | 1.5e-16,1.3e-15 | 0.8 | 1 | 1.9e-16,7.1e-16 | 0.5 |
| r2xyz | x4c | 5.5,0.0,0.0 | 0.472 | 34 | 9.3e-16,1.1e-15 | 0.7 | 1.8e-14 | 2.54 | 21 | 4.6e-16,4.6e-16 | 0.3 | offlat | offlat | - |
| r2xyz | x4e | 5.5,0.5,0.0 | 0.470 | 34 | 1.3e-15,1.3e-15 | 0.8 | 1.5e-14 | 2.57 | 21 | 1.7e-16,1.7e-16 | -0.1 | offlat | offlat | - |
| r2xyz | x4v | 5.5,0.5,0.5 | 0.469 | 34 | 5.0e-16,4.7e-15 | 1.3 | 1.4e-14 | 2.60 | 21 | 2.0e-16,5.6e-16 | 0.4 | 1 | 1.3e-16,5.8e-16 | 0.4 |
| r2xyz | x6c | 7.5,0.0,0.0 | 0.346 | 26 | 1.3e-16,2.5e-16 | 0.0 | 6.9e-15 | 2.72 | 18 | 1.9e-16,1.9e-16 | -0.1 | offlat | offlat | - |
| r2xyz | x6e | 7.5,0.5,0.0 | 0.346 | 26 | 3.6e-16,3.6e-16 | 0.2 | 6.6e-15 | 2.73 | 18 | 2.0e-16,2.2e-16 | 0.0 | offlat | offlat | - |
| r2xyz | x6v | 7.5,0.5,0.5 | 0.345 | 26 | 3.1e-16,2.4e-15 | 1.0 | 6.2e-15 | 2.75 | 18 | 1.1e-16,9.3e-16 | 0.6 | 1 | 7.4e-17,5.8e-16 | 0.4 |
| r2xyz | x8c | 9.5,0.0,0.0 | 0.273 | 22 | 2.7e-16,3.7e-16 | 0.2 | 5.9e-15 | 2.88 | 16 | 6.0e-16,6.0e-16 | 0.4 | offlat | offlat | - |
| r2xyz | x8e | 9.5,0.5,0.0 | 0.273 | 22 | 2.6e-16,3.5e-16 | 0.2 | 5.7e-15 | 2.89 | 16 | 2.5e-16,2.5e-16 | 0.1 | offlat | offlat | - |
| r2xyz | x8v | 9.5,0.5,0.5 | 0.273 | 22 | 4.7e-16,1.5e-15 | 0.8 | 5.6e-15 | 2.90 | 16 | 2.9e-16,7.9e-16 | 0.6 | 1 | 1.4e-16,9.4e-16 | 0.6 |
| r2xyz | x16c | 17.5,0.0,0.0 | 0.148 | 16 | 5.3e-16,5.3e-16 | 0.4 | 3.8e-15 | 2.23 | 13 | 3.2e-16,3.2e-16 | 0.2 | offlat | offlat | - |
| r2xyz | x16e | 17.5,0.5,0.0 | 0.148 | 16 | 3.5e-16,4.0e-16 | 0.3 | 3.8e-15 | 2.23 | 13 | 3.3e-16,3.3e-16 | 0.2 | offlat | offlat | - |
| r2xyz | x16v | 17.5,0.5,0.5 | 0.148 | 16 | 6.3e-16,9.2e-16 | 0.6 | 3.8e-15 | 2.23 | 13 | 3.5e-16,5.4e-16 | 0.4 | 1 | 3.0e-16,3.4e-16 | 0.2 |
| r2xyz | x32c | 33.5,0.0,0.0 | 0.078 | 14 | 4.8e-16,7.8e-16 | 0.5 | 1.1e-16 | 1.56 | 11 | 6.0e-16,6.0e-16 | 0.4 | offlat | offlat | - |
| r2xyz | x32e | 33.5,0.5,0.0 | 0.078 | 14 | 2.5e-16,5.0e-16 | 0.4 | 1.1e-16 | 1.56 | 11 | 2.6e-16,4.2e-16 | 0.3 | offlat | offlat | - |
| r2xyz | x32v | 33.5,0.5,0.5 | 0.078 | 14 | 5.7e-16,5.9e-16 | 0.4 | 1.1e-16 | 1.56 | 11 | 2.4e-16,3.2e-16 | 0.2 | 1 | 2.0e-16,5.9e-16 | 0.4 |
| r2xyz | d1c | 2.5,2.5,0.0 | 0.735 | -1 | - | - | - | 3.41 | 38 | 4.5e-16,4.5e-16 | 0.3 | offlat | offlat | - |
| r2xyz | d1e | 2.5,2.5,0.5 | 0.728 | -1 | - | - | - | 3.48 | 37 | 3.7e-16,7.4e-16 | 0.5 | 1 | 2.1e-16,4.3e-16 | 0.3 |
| r2xyz | d2c | 3.5,3.5,0.0 | 0.525 | 40 | 3.4e-16,6.2e-16 | 0.4 | 1.1e-14 | 3.87 | 24 | 2.4e-16,4.5e-16 | 0.3 | offlat | offlat | - |
| r2xyz | d2e | 3.5,3.5,0.5 | 0.522 | 40 | 6.3e-16,7.5e-16 | 0.5 | 9.1e-15 | 3.91 | 24 | 2.5e-16,4.3e-16 | 0.3 | 1 | 2.5e-16,2.5e-16 | 0.0 |
| r2xyz | d4c | 5.5,5.5,0.0 | 0.334 | 26 | 1.6e-16,1.8e-16 | -0.1 | 4.3e-15 | 4.32 | 18 | 1.8e-16,2.3e-16 | 0.0 | offlat | offlat | - |
| r2xyz | d4e | 5.5,5.5,0.5 | 0.333 | 26 | 2.0e-16,2.3e-16 | 0.0 | 4.1e-15 | 4.33 | 18 | 3.7e-16,3.9e-16 | 0.2 | 1 | 2.4e-16,2.8e-16 | 0.1 |
| r2xyz | d8c | 9.5,9.5,0.0 | 0.193 | 18 | 6.1e-16,6.1e-16 | 0.4 | 5.5e-15 | 2.76 | 14 | 2.4e-16,2.4e-16 | 0.0 | offlat | offlat | - |
| r2xyz | d8e | 9.5,9.5,0.5 | 0.193 | 18 | 6.4e-16,6.4e-16 | 0.5 | 5.5e-15 | 2.77 | 14 | 4.1e-16,4.1e-16 | 0.3 | 1 | 2.5e-16,4.8e-16 | 0.3 |
| r2xyz | d16c | 17.5,17.5,0.0 | 0.105 | 14 | 3.0e-16,5.4e-16 | 0.4 | 3.4e-15 | 1.80 | 12 | 2.7e-16,2.7e-16 | 0.1 | offlat | offlat | - |
| r2xyz | d16e | 17.5,17.5,0.5 | 0.105 | 14 | 1.9e-16,1.9e-16 | -0.1 | 3.4e-15 | 1.80 | 12 | 2.4e-16,4.4e-16 | 0.3 | 1 | 3.0e-16,6.0e-16 | 0.4 |
| r4x | x1c | 3.5,0.0,0.0 | 0.821 | -1 | - | - | - | 1.26 | 51 | 1.2e-15,1.2e-15 | 0.7 | 1 | 2.7e-15,2.7e-15 | 1.1 |
| r4x | x2c | 4.5,0.0,0.0 | 0.638 | -1 | - | - | - | 1.75 | 36 | 1.5e-15,2.0e-15 | 0.9 | 1 | 3.8e-16,1.0e-15 | 0.7 |
| r4x | x3c | 5.5,0.0,0.0 | 0.522 | 44 | 9.3e-16,9.3e-16 | 0.6 | 1.7e-14 | 2.08 | 30 | 1.1e-15,1.1e-15 | 0.7 | 1.01 | 3.5e-16,5.7e-16 | 0.4 |
| r4x | x4c | 6.5,0.0,0.0 | 0.442 | 36 | 1.2e-15,1.2e-15 | 0.7 | 1.6e-14 | 2.31 | 27 | 8.2e-16,8.2e-16 | 0.6 | 1.01 | 3.9e-16,4.1e-16 | 0.3 |
| r4x | x6c | 8.5,0.0,0.0 | 0.338 | 28 | 6.4e-16,7.5e-16 | 0.5 | 1.6e-14 | 2.63 | 23 | 6.4e-16,6.4e-16 | 0.5 | 1.01 | 8.2e-17,8.2e-17 | -0.4 |
| r4x | x8c | 10.5,0.0,0.0 | 0.274 | 24 | 4.2e-16,4.2e-16 | 0.3 | 1.4e-14 | 2.86 | 20 | 7.0e-16,7.0e-16 | 0.5 | 1.02 | 2.1e-16,2.6e-16 | 0.1 |
| r4x | x16c | 18.5,0.0,0.0 | 0.155 | 18 | 2.5e-16,4.1e-16 | 0.3 | 3.4e-15 | 2.17 | 16 | 2.5e-16,2.5e-16 | 0.0 | 1.02 | 4.5e-16,4.5e-16 | 0.3 |
| r4x | x32c | 34.5,0.0,0.0 | 0.083 | 14 | 4.4e-16,6.2e-16 | 0.4 | 5.3e-15 | 1.56 | 14 | 2.8e-16,3.6e-16 | 0.2 | 1.02 | 3.7e-16,5.6e-16 | 0.4 |
| r4x | d1c | 3.5,2.0,0.0 | 0.713 | -1 | - | - | - | 2.92 | 43 | 6.1e-16,1.3e-15 | 0.8 | 1.11 | 4.3e-16,5.5e-16 | 0.4 |
| r4x | d2c | 4.5,3.0,0.0 | 0.531 | 46 | 6.4e-16,1.1e-15 | 0.7 | 1.5e-14 | 3.71 | 32 | 4.9e-16,7.5e-16 | 0.5 | 1.1 | 4.9e-16,4.9e-16 | 0.3 |
| r4x | d4c | 6.5,5.0,0.0 | 0.350 | 30 | 6.2e-16,6.2e-16 | 0.4 | 7.0e-15 | 4.31 | 24 | 9.4e-16,9.4e-16 | 0.6 | 1.05 | 1.6e-16,1.6e-16 | -0.1 |
| r4x | d8c | 10.5,9.0,0.0 | 0.208 | 20 | 2.7e-16,3.7e-16 | 0.2 | 1.8e-14 | 2.70 | 18 | 3.0e-16,4.5e-16 | 0.3 | 1.01 | 3.3e-16,3.4e-16 | 0.2 |
| r4x | d16c | 18.5,17.0,0.0 | 0.114 | 16 | 3.2e-16,5.1e-16 | 0.4 | 2.1e-15 | 1.79 | 15 | 4.3e-16,4.3e-16 | 0.3 | 1.01 | 1.5e-16,2.8e-16 | 0.1 |
| r4xy | x1c | 3.5,0.0,0.0 | 1.050 | -1 | - | - | - | 1.97 | -1 | - | - | offlat | offlat | - |
| r4xy | x1e | 3.5,1.5,0.0 | 0.965 | -1 | - | - | - | 2.10 | -1 | - | - | 1.1 | 1.2e-15,1.2e-15 | 0.7 |
| r4xy | x2c | 4.5,0.0,0.0 | 0.816 | -1 | - | - | - | 2.20 | -1 | - | - | offlat | offlat | - |
| r4xy | x2e | 4.5,1.5,0.0 | 0.775 | -1 | - | - | - | 2.42 | -1 | - | - | 1.04 | 2.1e-16,7.1e-16 | 0.5 |
| r4xy | x3c | 5.5,0.0,0.0 | 0.668 | -1 | - | - | - | 2.37 | 47 | 3.9e-16,8.7e-16 | 0.6 | offlat | offlat | - |
| r4xy | x3e | 5.5,1.5,0.0 | 0.645 | -1 | - | - | - | 2.57 | 43 | 8.1e-16,8.1e-16 | 0.6 | 1.01 | 1.4e-16,4.3e-16 | 0.3 |
| r4xy | x4c | 6.5,0.0,0.0 | 0.565 | 48 | 2.4e-16,5.4e-16 | 0.4 | 1.5e-14 | 2.52 | 36 | 3.7e-16,3.7e-16 | 0.2 | offlat | offlat | - |
| r4xy | x4e | 6.5,1.5,0.0 | 0.551 | 46 | 1.0e-15,1.0e-15 | 0.7 | 1.7e-14 | 2.68 | 34 | 2.8e-16,2.9e-16 | 0.1 | 1.01 | 1.8e-16,4.4e-16 | 0.3 |
| r4xy | x6c | 8.5,0.0,0.0 | 0.432 | 34 | 3.2e-16,3.2e-16 | 0.2 | 1.5e-14 | 2.75 | 27 | 3.4e-16,5.6e-16 | 0.4 | offlat | offlat | - |
| r4xy | x6e | 8.5,1.5,0.0 | 0.426 | 34 | 8.1e-16,1.3e-15 | 0.8 | 9.2e-15 | 2.85 | 27 | 2.5e-16,5.2e-16 | 0.4 | 1.01 | 1.1e-16,1.1e-16 | -0.3 |
| r4xy | x8c | 10.5,0.0,0.0 | 0.350 | 28 | 9.5e-16,1.2e-15 | 0.7 | 1.2e-14 | 2.94 | 23 | 3.7e-16,4.5e-16 | 0.3 | offlat | offlat | - |
| r4xy | x8e | 10.5,1.5,0.0 | 0.346 | 28 | 5.5e-16,6.1e-16 | 0.4 | 9.7e-15 | 3.02 | 23 | 5.9e-16,7.1e-16 | 0.5 | 1.02 | 1.3e-16,1.6e-16 | -0.1 |
| r4xy | x16c | 18.5,0.0,0.0 | 0.199 | 20 | 4.7e-16,4.7e-16 | 0.3 | 2.3e-15 | 2.18 | 18 | 3.3e-16,3.3e-16 | 0.2 | offlat | offlat | - |
| r4xy | x16e | 18.5,1.5,0.0 | 0.198 | 20 | 9.5e-16,9.5e-16 | 0.6 | 2.2e-15 | 2.17 | 18 | 5.6e-16,5.6e-16 | 0.4 | 1.02 | 2.5e-16,2.5e-16 | 0.1 |
| r4xy | x32c | 34.5,0.0,0.0 | 0.106 | 16 | 5.6e-16,5.6e-16 | 0.4 | 4.7e-16 | 1.56 | 15 | 7.1e-16,7.1e-16 | 0.5 | offlat | offlat | - |
| r4xy | x32e | 34.5,1.5,0.0 | 0.106 | 16 | 7.2e-16,9.1e-16 | 0.6 | 4.6e-16 | 1.56 | 15 | 4.2e-16,4.2e-16 | 0.3 | 1.02 | 2.3e-16,3.4e-16 | 0.2 |
| r4xy | d1c | 3.5,3.5,0.0 | 0.742 | -1 | - | - | - | 3.24 | 43 | 4.6e-16,6.0e-16 | 0.4 | 1.11 | 2.5e-16,4.2e-16 | 0.3 |
| r4xy | d2c | 4.5,4.5,0.0 | 0.577 | 50 | 1.2e-15,1.2e-15 | 0.7 | 1.9e-14 | 3.87 | 33 | 6.5e-16,9.5e-16 | 0.6 | 1.08 | 2.1e-16,2.1e-16 | -0.0 |
| r4xy | d4c | 6.5,6.5,0.0 | 0.400 | 32 | 3.8e-16,4.0e-16 | 0.3 | 1.2e-14 | 4.04 | 25 | 4.2e-16,4.2e-16 | 0.3 | 1.05 | 1.8e-16,2.2e-16 | 0.0 |
| r4xy | d8c | 10.5,10.5,0.0 | 0.247 | 22 | 2.3e-16,3.5e-16 | 0.2 | 7.7e-15 | 2.56 | 20 | 3.3e-16,5.1e-16 | 0.4 | 1.02 | 1.9e-16,2.6e-16 | 0.1 |
| r4xy | d16c | 18.5,18.5,0.0 | 0.140 | 16 | 2.5e-16,2.5e-16 | 0.1 | 1.8e-14 | 1.77 | 16 | 4.1e-16,4.1e-16 | 0.3 | 1.02 | 1.3e-16,2.0e-16 | -0.0 |
| r4xyz | x1c | 3.5,0.0,0.0 | 1.237 | -1 | - | - | - | 2.91 | -1 | - | - | offlat | offlat | - |
| r4xyz | x1e | 3.5,1.5,0.0 | 1.137 | -1 | - | - | - | 3.07 | -1 | - | - | offlat | offlat | - |
| r4xyz | x1v | 3.5,1.5,1.5 | 1.058 | -1 | - | - | - | 3.33 | -1 | - | - | 1.24 | 4.6e-16,5.8e-16 | 0.4 |
| r4xyz | x2c | 4.5,0.0,0.0 | 0.962 | -1 | - | - | - | 2.70 | -1 | - | - | offlat | offlat | - |
| r4xyz | x2e | 4.5,1.5,0.0 | 0.913 | -1 | - | - | - | 2.94 | -1 | - | - | offlat | offlat | - |
| r4xyz | x2v | 4.5,1.5,1.5 | 0.870 | -1 | - | - | - | 3.21 | -1 | - | - | 1.08 | 9.8e-17,5.4e-16 | 0.4 |
| r4xyz | x3c | 5.5,0.0,0.0 | 0.787 | -1 | - | - | - | 2.69 | -1 | - | - | offlat | offlat | - |
| r4xyz | x3e | 5.5,1.5,0.0 | 0.760 | -1 | - | - | - | 2.89 | -1 | - | - | offlat | offlat | - |
| r4xyz | x3v | 5.5,1.5,1.5 | 0.735 | -1 | - | - | - | 3.11 | 56 | 5.4e-16,7.8e-16 | 0.5 | 1.02 | 1.6e-16,4.1e-16 | 0.3 |
| r4xyz | x4c | 6.5,0.0,0.0 | 0.666 | -1 | - | - | - | 2.73 | 48 | 2.4e-16,5.5e-16 | 0.4 | offlat | offlat | - |
| r4xyz | x4e | 6.5,1.5,0.0 | 0.649 | -1 | - | - | - | 2.90 | 45 | 4.5e-16,4.5e-16 | 0.3 | offlat | offlat | - |
| r4xyz | x4v | 6.5,1.5,1.5 | 0.633 | 54 | 5.1e-16,1.1e-15 | 0.7 | 2.9e-14 | 3.06 | 43 | 2.8e-16,3.5e-16 | 0.2 | 1.01 | 1.3e-16,4.4e-16 | 0.3 |
| r4xyz | x6c | 8.5,0.0,0.0 | 0.509 | 40 | 1.1e-15,1.1e-15 | 0.7 | 8.1e-15 | 2.87 | 33 | 5.4e-16,5.4e-16 | 0.4 | offlat | offlat | - |
| r4xyz | x6e | 8.5,1.5,0.0 | 0.502 | 38 | 5.5e-16,1.3e-15 | 0.8 | 2.2e-14 | 2.97 | 32 | 2.3e-16,3.4e-16 | 0.2 | offlat | offlat | - |
| r4xyz | x6v | 8.5,1.5,1.5 | 0.494 | 38 | 4.2e-16,1.3e-15 | 0.8 | 1.3e-14 | 3.08 | 31 | 1.9e-16,4.6e-16 | 0.3 | 1.02 | 1.6e-16,5.3e-16 | 0.4 |
| r4xyz | x8c | 10.5,0.0,0.0 | 0.412 | 32 | 8.2e-16,9.7e-16 | 0.6 | 5.4e-15 | 3.02 | 26 | 3.0e-16,3.0e-16 | 0.1 | offlat | offlat | - |
| r4xyz | x8e | 10.5,1.5,0.0 | 0.408 | 32 | 5.7e-16,6.5e-16 | 0.5 | 4.1e-15 | 3.09 | 26 | 1.7e-16,1.7e-16 | -0.1 | offlat | offlat | - |
| r4xyz | x8v | 10.5,1.5,1.5 | 0.404 | 30 | 3.1e-16,1.2e-15 | 0.7 | 2.4e-14 | 3.17 | 26 | 2.2e-16,2.6e-16 | 0.1 | 1.03 | 3.7e-16,4.2e-16 | 0.3 |
| r4xyz | x16c | 18.5,0.0,0.0 | 0.234 | 20 | 2.7e-16,4.6e-16 | 0.3 | 2.1e-14 | 2.19 | 19 | 1.6e-16,2.6e-16 | 0.1 | offlat | offlat | - |
| r4xyz | x16e | 18.5,1.5,0.0 | 0.233 | 20 | 7.9e-16,8.0e-16 | 0.6 | 2.0e-14 | 2.18 | 19 | 5.5e-16,5.6e-16 | 0.4 | offlat | offlat | - |
| r4xyz | x16v | 18.5,1.5,1.5 | 0.233 | 20 | 7.4e-16,2.1e-15 | 1.0 | 1.9e-14 | 2.19 | 19 | 4.1e-16,6.6e-16 | 0.5 | 1.03 | 1.5e-16,1.7e-16 | -0.1 |
| r4xyz | x32c | 34.5,0.0,0.0 | 0.126 | 16 | 3.1e-16,5.9e-16 | 0.4 | 2.6e-15 | 1.57 | 16 | 2.2e-16,2.2e-16 | -0.0 | offlat | offlat | - |
| r4xyz | x32e | 34.5,1.5,0.0 | 0.125 | 16 | 1.3e-15,1.3e-15 | 0.8 | 2.6e-15 | 1.57 | 16 | 6.2e-16,6.2e-16 | 0.4 | offlat | offlat | - |
| r4xyz | x32v | 34.5,1.5,1.5 | 0.125 | 16 | 4.1e-16,6.8e-16 | 0.5 | 2.6e-15 | 1.57 | 16 | 7.0e-16,7.0e-16 | 0.5 | 1.02 | 3.8e-16,4.2e-16 | 0.3 |
| r4xyz | d1c | 3.5,3.5,0.0 | 0.875 | -1 | - | - | - | 3.86 | -1 | - | - | offlat | offlat | - |
| r4xyz | d1e | 3.5,3.5,1.5 | 0.837 | -1 | - | - | - | 4.17 | -1 | - | - | 1.14 | 9.4e-17,2.1e-16 | -0.0 |
| r4xyz | d2c | 4.5,4.5,0.0 | 0.680 | -1 | - | - | - | 4.21 | 46 | 5.5e-16,8.2e-16 | 0.6 | offlat | offlat | - |
| r4xyz | d2e | 4.5,4.5,1.5 | 0.662 | -1 | - | - | - | 4.45 | 42 | 3.8e-16,5.2e-16 | 0.4 | 1.09 | 2.2e-16,3.1e-16 | 0.1 |
| r4xyz | d4c | 6.5,6.5,0.0 | 0.471 | 36 | 4.0e-16,4.0e-16 | 0.3 | 1.6e-14 | 4.15 | 29 | 4.7e-16,4.7e-16 | 0.3 | offlat | offlat | - |
| r4xyz | d4e | 6.5,6.5,1.5 | 0.465 | 36 | 2.4e-16,1.1e-15 | 0.7 | 1.0e-14 | 4.22 | 28 | 2.7e-16,4.0e-16 | 0.3 | 1.05 | 2.1e-16,3.8e-16 | 0.2 |
| r4xyz | d8c | 10.5,10.5,0.0 | 0.292 | 24 | 7.1e-16,7.1e-16 | 0.5 | 5.4e-15 | 2.59 | 21 | 1.8e-16,2.2e-16 | -0.0 | offlat | offlat | - |
| r4xyz | d8e | 10.5,10.5,1.5 | 0.290 | 24 | 4.2e-16,5.3e-16 | 0.4 | 4.9e-15 | 2.60 | 21 | 3.3e-16,5.2e-16 | 0.4 | 1.02 | 7.0e-16,7.0e-16 | 0.5 |
| r4xyz | d16c | 18.5,18.5,0.0 | 0.166 | 18 | 2.2e-16,4.0e-16 | 0.3 | 1.9e-15 | 1.78 | 17 | 1.8e-16,3.4e-16 | 0.2 | offlat | offlat | - |
| r4xyz | d16e | 18.5,18.5,1.5 | 0.165 | 18 | 1.8e-16,1.8e-16 | -0.1 | 1.9e-15 | 1.78 | 17 | 2.9e-16,2.9e-16 | 0.1 | 1.02 | 1.7e-16,7.7e-16 | 0.5 |
| r8x | x1c | 5.5,0.0,0.0 | 0.858 | -1 | - | - | - | 0.81 | -1 | - | - | 1.01 | 2.6e-15,2.6e-15 | 1.1 |
| r8x | x2c | 6.5,0.0,0.0 | 0.726 | -1 | - | - | - | 1.33 | -1 | - | - | 1.02 | 3.1e-16,7.8e-16 | 0.5 |
| r8x | x3c | 7.5,0.0,0.0 | 0.629 | -1 | - | - | - | 1.75 | 51 | 7.1e-16,1.6e-15 | 0.9 | 1.03 | 2.9e-16,5.2e-16 | 0.4 |
| r8x | x4c | 8.5,0.0,0.0 | 0.555 | 54 | 1.2e-15,1.2e-15 | 0.7 | 1.5e-14 | 2.08 | 44 | 4.4e-16,5.4e-16 | 0.4 | 1.04 | 1.6e-16,3.1e-16 | 0.1 |
| r8x | x6c | 10.5,0.0,0.0 | 0.449 | 42 | 8.8e-16,1.2e-15 | 0.7 | 4.8e-15 | 2.57 | 35 | 8.5e-16,8.5e-16 | 0.6 | 1.08 | 8.2e-17,9.8e-17 | -0.4 |
| r8x | x8c | 12.5,0.0,0.0 | 0.377 | 34 | 3.1e-16,3.1e-16 | 0.1 | 1.3e-14 | 2.93 | 31 | 9.9e-16,1.0e-15 | 0.7 | 1.16 | 1.2e-16,1.2e-16 | -0.3 |
| r8x | x16c | 20.5,0.0,0.0 | 0.230 | 24 | 9.5e-16,9.5e-16 | 0.6 | 3.5e-15 | 2.16 | 23 | 6.3e-16,1.2e-15 | 0.7 | 1.09 | 2.2e-16,2.2e-16 | -0.0 |
| r8x | x32c | 36.5,0.0,0.0 | 0.129 | 18 | 4.6e-16,1.6e-15 | 0.9 | 4.0e-15 | 1.65 | 18 | 8.7e-16,1.2e-15 | 0.7 | 1.1 | 2.5e-16,5.1e-16 | 0.4 |
| r8x | d1c | 5.5,2.0,0.0 | 0.806 | -1 | - | - | - | 2.27 | -1 | - | - | 1.24 | 3.4e-16,3.4e-16 | 0.2 |
| r8x | d2c | 6.5,3.0,0.0 | 0.659 | -1 | - | - | - | 3.21 | 56 | 1.4e-15,1.4e-15 | 0.8 | 1.2 | 3.7e-16,3.9e-16 | 0.2 |
| r8x | d4c | 8.5,5.0,0.0 | 0.478 | 44 | 8.5e-16,9.7e-16 | 0.6 | 2.0e-14 | 3.72 | 38 | 8.6e-16,8.6e-16 | 0.6 | 1.09 | 1.9e-16,2.1e-16 | -0.0 |
| r8x | d8c | 12.5,9.0,0.0 | 0.306 | 28 | 4.7e-16,7.3e-16 | 0.5 | 1.8e-14 | 2.55 | 27 | 8.5e-16,8.5e-16 | 0.6 | 1.05 | 3.6e-16,3.6e-16 | 0.2 |
| r8x | d16c | 20.5,17.0,0.0 | 0.177 | 20 | 3.2e-16,4.1e-16 | 0.3 | 1.2e-14 | 1.81 | 20 | 2.7e-16,5.0e-16 | 0.4 | 1.06 | 2.2e-16,3.7e-16 | 0.2 |
| r8xy | x1c | 5.5,0.0,0.0 | 1.171 | -1 | - | - | - | 1.99 | -1 | - | - | offlat | offlat | - |
| r8xy | x1e | 5.5,3.5,0.0 | 0.988 | -1 | - | - | - | 2.01 | -1 | - | - | 1.26 | 7.6e-16,7.6e-16 | 0.5 |
| r8xy | x2c | 6.5,0.0,0.0 | 0.991 | -1 | - | - | - | 2.30 | -1 | - | - | offlat | offlat | - |
| r8xy | x2e | 6.5,3.5,0.0 | 0.873 | -1 | - | - | - | 2.61 | -1 | - | - | 1.2 | 2.9e-16,4.0e-16 | 0.3 |
| r8xy | x3c | 7.5,0.0,0.0 | 0.859 | -1 | - | - | - | 2.51 | -1 | - | - | offlat | offlat | - |
| r8xy | x3e | 7.5,3.5,0.0 | 0.778 | -1 | - | - | - | 2.93 | -1 | - | - | 1.17 | 1.7e-16,3.3e-16 | 0.2 |
| r8xy | x4c | 8.5,0.0,0.0 | 0.758 | -1 | - | - | - | 2.67 | -1 | - | - | offlat | offlat | - |
| r8xy | x4e | 8.5,3.5,0.0 | 0.701 | -1 | - | - | - | 3.11 | -1 | - | - | 1.16 | 3.0e-16,6.1e-16 | 0.4 |
| r8xy | x6c | 10.5,0.0,0.0 | 0.614 | -1 | - | - | - | 2.95 | -1 | - | - | offlat | offlat | - |
| r8xy | x6e | 10.5,3.5,0.0 | 0.582 | 54 | 1.1e-15,1.1e-15 | 0.7 | 2.3e-14 | 3.34 | 51 | 9.6e-16,9.6e-16 | 0.6 | 1.2 | 4.6e-16,4.7e-16 | 0.3 |
| r8xy | x8c | 12.5,0.0,0.0 | 0.515 | 46 | 6.2e-16,6.4e-16 | 0.5 | 9.2e-15 | 3.08 | 43 | 1.1e-15,1.2e-15 | 0.7 | offlat | offlat | - |
| r8xy | x8e | 12.5,3.5,0.0 | 0.496 | 44 | 1.2e-15,1.3e-15 | 0.8 | 7.3e-15 | 2.99 | 39 | 2.9e-16,3.0e-16 | 0.1 | 1.12 | 4.3e-16,4.3e-16 | 0.3 |
| r8xy | x16c | 20.5,0.0,0.0 | 0.314 | 28 | 5.4e-16,1.0e-15 | 0.7 | 5.5e-15 | 2.17 | 27 | 8.1e-16,1.5e-15 | 0.8 | offlat | offlat | - |
| r8xy | x16e | 20.5,3.5,0.0 | 0.310 | 28 | 8.0e-16,1.6e-15 | 0.8 | 3.8e-15 | 2.15 | 27 | 4.0e-16,7.7e-16 | 0.5 | 1.09 | 1.9e-16,3.0e-16 | 0.1 |
| r8xy | x32c | 36.5,0.0,0.0 | 0.176 | 20 | 4.4e-16,1.2e-15 | 0.7 | 4.9e-15 | 1.65 | 21 | 4.1e-16,4.2e-16 | 0.3 | offlat | offlat | - |
| r8xy | x32e | 36.5,3.5,0.0 | 0.176 | 20 | 3.3e-16,3.8e-16 | 0.2 | 4.6e-15 | 1.65 | 21 | 3.9e-16,1.4e-15 | 0.8 | 1.1 | 4.2e-16,4.2e-16 | 0.3 |
| r8xy | d1c | 5.5,5.5,0.0 | 0.828 | -1 | - | - | - | 3.16 | -1 | - | - | 1.28 | 1.4e-16,2.2e-16 | -0.0 |
| r8xy | d2c | 6.5,6.5,0.0 | 0.701 | -1 | - | - | - | 3.84 | -1 | - | - | 1.28 | 2.3e-16,2.3e-16 | 0.0 |
| r8xy | d4c | 8.5,8.5,0.0 | 0.536 | 48 | 9.9e-16,1.3e-15 | 0.8 | 1.5e-14 | 3.17 | 42 | 7.5e-16,1.0e-15 | 0.7 | 1.07 | 1.5e-16,2.3e-16 | 0.0 |
| r8xy | d8c | 12.5,12.5,0.0 | 0.364 | 32 | 8.2e-16,8.2e-16 | 0.6 | 3.9e-15 | 2.38 | 30 | 5.2e-16,6.8e-16 | 0.5 | 1.09 | 3.2e-16,3.2e-16 | 0.2 |
| r8xy | d16c | 20.5,20.5,0.0 | 0.222 | 22 | 3.4e-16,6.5e-16 | 0.5 | 9.5e-15 | 1.81 | 23 | 2.3e-16,2.8e-16 | 0.1 | 1.1 | 3.0e-16,3.4e-16 | 0.2 |
| r8xyz | x1c | 5.5,0.0,0.0 | 1.417 | -1 | - | - | - | 4.01 | -1 | - | - | offlat | offlat | - |
| r8xyz | x1e | 5.5,3.5,0.0 | 1.196 | -1 | - | - | - | 4.05 | -1 | - | - | offlat | offlat | - |
| r8xyz | x1v | 5.5,3.5,3.5 | 1.053 | -1 | - | - | - | 4.49 | -1 | - | - | 1.65 | 9.5e-16,9.5e-16 | 0.6 |
| r8xyz | x2c | 6.5,0.0,0.0 | 1.199 | -1 | - | - | - | 3.56 | -1 | - | - | offlat | offlat | - |
| r8xyz | x2e | 6.5,3.5,0.0 | 1.056 | -1 | - | - | - | 3.92 | -1 | - | - | offlat | offlat | - |
| r8xyz | x2v | 6.5,3.5,3.5 | 0.954 | -1 | - | - | - | 4.48 | -1 | - | - | 1.43 | 4.1e-16,4.6e-16 | 0.3 |
| r8xyz | x3c | 7.5,0.0,0.0 | 1.039 | -1 | - | - | - | 3.37 | -1 | - | - | offlat | offlat | - |
| r8xyz | x3e | 7.5,3.5,0.0 | 0.942 | -1 | - | - | - | 3.81 | -1 | - | - | offlat | offlat | - |
| r8xyz | x3v | 7.5,3.5,3.5 | 0.867 | -1 | - | - | - | 4.35 | -1 | - | - | 1.32 | 9.2e-16,1.6e-15 | 0.9 |
| r8xyz | x4c | 8.5,0.0,0.0 | 0.917 | -1 | - | - | - | 3.30 | -1 | - | - | offlat | offlat | - |
| r8xyz | x4e | 8.5,3.5,0.0 | 0.848 | -1 | - | - | - | 3.75 | -1 | - | - | offlat | offlat | - |
| r8xyz | x4v | 8.5,3.5,3.5 | 0.792 | -1 | - | - | - | 4.22 | -1 | - | - | 1.28 | 5.5e-16,1.1e-15 | 0.7 |
| r8xyz | x6c | 10.5,0.0,0.0 | 0.742 | -1 | - | - | - | 3.33 | -1 | - | - | offlat | offlat | - |
| r8xyz | x6e | 10.5,3.5,0.0 | 0.704 | -1 | - | - | - | 3.71 | -1 | - | - | offlat | offlat | - |
| r8xyz | x6v | 10.5,3.5,3.5 | 0.671 | -1 | - | - | - | 3.85 | -1 | - | - | 1.25 | 8.5e-16,8.5e-16 | 0.6 |
| r8xyz | x8c | 12.5,0.0,0.0 | 0.624 | 56 | 2.6e-16,2.6e-16 | 0.1 | 3.1e-14 | 3.27 | -1 | - | - | offlat | offlat | - |
| r8xyz | x8e | 12.5,3.5,0.0 | 0.600 | 54 | 5.8e-16,6.3e-16 | 0.5 | 1.1e-14 | 3.16 | -1 | - | - | offlat | offlat | - |
| r8xyz | x8v | 12.5,3.5,3.5 | 0.580 | 50 | 8.2e-16,8.9e-16 | 0.6 | 2.1e-14 | 3.27 | 52 | 3.9e-16,1.1e-15 | 0.7 | 1.16 | 1.0e-15,1.0e-15 | 0.7 |
| r8xyz | x16c | 20.5,0.0,0.0 | 0.380 | 32 | 5.5e-16,6.9e-16 | 0.5 | 3.1e-15 | 2.21 | 31 | 3.9e-16,3.9e-16 | 0.2 | offlat | offlat | - |
| r8xyz | x16e | 20.5,3.5,0.0 | 0.375 | 30 | 5.2e-16,1.0e-15 | 0.7 | 1.8e-14 | 2.19 | 30 | 7.0e-16,7.0e-16 | 0.5 | offlat | offlat | - |
| r8xyz | x16v | 20.5,3.5,3.5 | 0.370 | 30 | 5.2e-16,8.2e-16 | 0.6 | 1.2e-14 | 2.23 | 30 | 3.7e-16,4.7e-16 | 0.3 | 1.12 | 1.1e-15,1.1e-15 | 0.7 |
| r8xyz | x32c | 36.5,0.0,0.0 | 0.214 | 22 | 3.7e-16,5.1e-16 | 0.4 | 2.3e-15 | 1.66 | 23 | 2.3e-16,2.5e-16 | 0.0 | offlat | offlat | - |
| r8xyz | x32e | 36.5,3.5,0.0 | 0.213 | 22 | 3.2e-16,5.1e-16 | 0.4 | 2.2e-15 | 1.65 | 22 | 1.9e-16,4.3e-16 | 0.3 | offlat | offlat | - |
| r8xyz | x32v | 36.5,3.5,3.5 | 0.212 | 22 | 9.1e-17,4.8e-16 | 0.3 | 2.0e-15 | 1.67 | 22 | 9.9e-17,3.5e-16 | 0.2 | 1.11 | 9.3e-16,9.3e-16 | 0.6 |
| r8xyz | d1c | 5.5,5.5,0.0 | 1.002 | -1 | - | - | - | 4.50 | -1 | - | - | offlat | offlat | - |
| r8xyz | d1e | 5.5,5.5,3.5 | 0.914 | -1 | - | - | - | 5.03 | -1 | - | - | 1.49 | 6.1e-16,7.0e-16 | 0.5 |
| r8xyz | d2c | 6.5,6.5,0.0 | 0.848 | -1 | - | - | - | 4.59 | -1 | - | - | offlat | offlat | - |
| r8xyz | d2e | 6.5,6.5,3.5 | 0.792 | -1 | - | - | - | 4.74 | -1 | - | - | 1.31 | 4.4e-16,4.8e-16 | 0.3 |
| r8xyz | d4c | 8.5,8.5,0.0 | 0.648 | -1 | - | - | - | 3.40 | -1 | - | - | offlat | offlat | - |
| r8xyz | d4e | 8.5,8.5,3.5 | 0.623 | 56 | 3.4e-16,7.3e-16 | 0.5 | 3.0e-14 | 3.53 | -1 | - | - | 1.1 | 8.7e-16,8.7e-16 | 0.6 |
| r8xyz | d8c | 12.5,12.5,0.0 | 0.441 | 36 | 9.0e-16,9.0e-16 | 0.6 | 7.5e-15 | 2.45 | 35 | 3.9e-16,5.0e-16 | 0.4 | offlat | offlat | - |
| r8xyz | d8e | 12.5,12.5,3.5 | 0.433 | 36 | 7.4e-16,7.4e-16 | 0.5 | 4.0e-15 | 2.51 | 34 | 3.3e-16,5.0e-16 | 0.4 | 1.09 | 5.0e-16,8.4e-16 | 0.6 |
| r8xyz | d16c | 20.5,20.5,0.0 | 0.269 | 24 | 4.2e-16,4.2e-16 | 0.3 | 9.4e-15 | 1.83 | 25 | 4.9e-16,6.1e-16 | 0.4 | offlat | offlat | - |
| r8xyz | d16e | 20.5,20.5,3.5 | 0.267 | 24 | 5.8e-16,5.8e-16 | 0.4 | 8.2e-15 | 1.85 | 25 | 4.2e-16,4.3e-16 | 0.3 | 1.1 | 2.9e-16,5.4e-16 | 0.4 |
| r16x | x1c | 9.5,0.0,0.0 | 0.907 | -1 | - | - | - | 0.50 | -1 | - | - | 1.04 | 2.5e-15,2.5e-15 | 1.1 |
| r16x | x2c | 10.5,0.0,0.0 | 0.821 | -1 | - | - | - | 0.98 | -1 | - | - | 1.1 | 3.6e-16,7.4e-16 | 0.5 |
| r16x | x3c | 11.5,0.0,0.0 | 0.749 | -1 | - | - | - | 1.46 | -1 | - | - | 1.17 | 2.9e-16,2.9e-16 | 0.1 |
| r16x | x4c | 12.5,0.0,0.0 | 0.689 | -1 | - | - | - | 1.90 | -1 | - | - | 1.25 | 2.0e-16,3.0e-16 | 0.1 |
| r16x | x6c | 14.5,0.0,0.0 | 0.594 | -1 | - | - | - | 2.69 | -1 | - | - | 1.44 | 1.9e-16,2.1e-16 | -0.0 |
| r16x | x8c | 16.5,0.0,0.0 | 0.522 | 54 | 4.4e-16,5.1e-16 | 0.4 | 1.2e-14 | 2.93 | -1 | - | - | 1.46 | 8.9e-17,1.0e-16 | -0.3 |
| r16x | x16c | 24.5,0.0,0.0 | 0.352 | 34 | 7.1e-16,1.5e-15 | 0.8 | 1.8e-14 | 2.58 | -1 | - | - | 1.48 | 1.9e-16,1.9e-16 | -0.1 |
| r16x | x32c | 40.5,0.0,0.0 | 0.213 | 24 | 3.4e-16,1.0e-15 | 0.7 | 2.1e-14 | 2.19 | -1 | - | - | 1.54 | 2.3e-16,6.3e-16 | 0.5 |
| r16x | d1c | 9.5,2.0,0.0 | 0.888 | -1 | - | - | - | 1.53 | -1 | - | - | 1.37 | 3.7e-16,3.8e-16 | 0.2 |
| r16x | d2c | 10.5,3.0,0.0 | 0.789 | -1 | - | - | - | 2.45 | -1 | - | - | 1.35 | 3.4e-16,3.7e-16 | 0.2 |
| r16x | d4c | 12.5,5.0,0.0 | 0.640 | -1 | - | - | - | 3.07 | -1 | - | - | 1.25 | 8.5e-17,1.1e-16 | -0.3 |
| r16x | d8c | 16.5,9.0,0.0 | 0.458 | 46 | 6.3e-16,1.0e-15 | 0.7 | 6.5e-15 | 2.65 | -1 | - | - | 1.29 | 2.6e-16,3.6e-16 | 0.2 |
| r16x | d16c | 24.5,17.0,0.0 | 0.289 | 30 | 4.3e-16,8.3e-16 | 0.6 | 5.0e-15 | 2.10 | -1 | - | - | 1.3 | 3.4e-16,5.6e-16 | 0.4 |
| r16xy | x1c | 9.5,0.0,0.0 | 1.270 | -1 | - | - | - | 2.25 | -1 | - | - | offlat | offlat | - |
| r16xy | x1e | 9.5,7.5,0.0 | 0.997 | -1 | - | - | - | 2.26 | -1 | - | - | 1.7 | 7.2e-16,8.3e-16 | 0.6 |
| r16xy | x2c | 10.5,0.0,0.0 | 1.149 | -1 | - | - | - | 2.82 | -1 | - | - | offlat | offlat | - |
| r16xy | x2e | 10.5,7.5,0.0 | 0.935 | -1 | - | - | - | 2.81 | -1 | - | - | 1.6 | 4.4e-16,6.6e-16 | 0.5 |
| r16xy | x3c | 11.5,0.0,0.0 | 1.049 | -1 | - | - | - | 3.17 | -1 | - | - | offlat | offlat | - |
| r16xy | x3e | 11.5,7.5,0.0 | 0.879 | -1 | - | - | - | 3.03 | -1 | - | - | 1.53 | 3.3e-16,3.3e-16 | 0.2 |
| r16xy | x4c | 12.5,0.0,0.0 | 0.965 | -1 | - | - | - | 3.34 | -1 | - | - | offlat | offlat | - |
| r16xy | x4e | 12.5,7.5,0.0 | 0.827 | -1 | - | - | - | 3.10 | -1 | - | - | 1.48 | 3.6e-16,3.6e-16 | 0.2 |
| r16xy | x6c | 14.5,0.0,0.0 | 0.832 | -1 | - | - | - | 3.21 | -1 | - | - | offlat | offlat | - |
| r16xy | x6e | 14.5,7.5,0.0 | 0.739 | -1 | - | - | - | 3.07 | -1 | - | - | 1.44 | 2.4e-16,4.8e-16 | 0.3 |
| r16xy | x8c | 16.5,0.0,0.0 | 0.731 | -1 | - | - | - | 3.06 | -1 | - | - | offlat | offlat | - |
| r16xy | x8e | 16.5,7.5,0.0 | 0.666 | -1 | - | - | - | 2.96 | -1 | - | - | 1.43 | 3.6e-16,4.5e-16 | 0.3 |
| r16xy | x16c | 24.5,0.0,0.0 | 0.492 | 46 | 4.1e-16,9.1e-16 | 0.6 | 1.6e-14 | 2.60 | -1 | - | - | offlat | offlat | - |
| r16xy | x16e | 24.5,7.5,0.0 | 0.471 | 44 | 3.7e-16,5.0e-16 | 0.4 | 1.1e-14 | 2.57 | -1 | - | - | 1.48 | 1.4e-16,3.7e-16 | 0.2 |
| r16xy | x32c | 40.5,0.0,0.0 | 0.298 | 30 | 4.9e-16,1.1e-15 | 0.7 | 4.5e-15 | 2.19 | -1 | - | - | offlat | offlat | - |
| r16xy | x32e | 40.5,7.5,0.0 | 0.293 | 30 | 6.5e-16,1.6e-15 | 0.8 | 2.9e-15 | 2.18 | -1 | - | - | 1.53 | 5.9e-16,5.9e-16 | 0.4 |
| r16xy | d1c | 9.5,9.5,0.0 | 0.898 | -1 | - | - | - | 2.80 | -1 | - | - | 1.48 | 5.4e-16,1.2e-15 | 0.7 |
| r16xy | d2c | 10.5,10.5,0.0 | 0.812 | -1 | - | - | - | 3.04 | -1 | - | - | 1.41 | 5.0e-16,1.1e-15 | 0.7 |
| r16xy | d4c | 12.5,12.5,0.0 | 0.682 | -1 | - | - | - | 3.00 | -1 | - | - | 1.41 | 6.0e-16,9.2e-16 | 0.6 |
| r16xy | d8c | 16.5,16.5,0.0 | 0.517 | 50 | 3.1e-16,3.3e-16 | 0.2 | 8.4e-15 | 2.69 | -1 | - | - | 1.48 | 3.7e-16,7.4e-16 | 0.5 |
| r16xy | d16c | 24.5,24.5,0.0 | 0.348 | 34 | 6.8e-16,6.8e-16 | 0.5 | 3.0e-15 | 2.30 | -1 | - | - | 1.52 | 1.0e-15,1.0e-15 | 0.7 |
| r16xyz | x1c | 9.5,0.0,0.0 | 1.550 | -1 | - | - | - | 5.80 | -1 | - | - | offlat | offlat | - |
| r16xyz | x1e | 9.5,7.5,0.0 | 1.216 | -1 | - | - | - | 5.24 | -1 | - | - | offlat | offlat | - |
| r16xyz | x1v | 9.5,7.5,7.5 | 1.034 | -1 | - | - | - | 5.00 | -1 | - | - | 2.2 | 3.6e-15,3.6e-15 | 1.2 |
| r16xyz | x2c | 10.5,0.0,0.0 | 1.402 | -1 | - | - | - | 5.39 | -1 | - | - | offlat | offlat | - |
| r16xyz | x2e | 10.5,7.5,0.0 | 1.141 | -1 | - | - | - | 4.79 | -1 | - | - | offlat | offlat | - |
| r16xyz | x2v | 10.5,7.5,7.5 | 0.986 | -1 | - | - | - | 4.84 | -1 | - | - | 1.99 | 1.6e-15,2.4e-15 | 1.0 |
| r16xyz | x3c | 11.5,0.0,0.0 | 1.280 | -1 | - | - | - | 5.14 | -1 | - | - | offlat | offlat | - |
| r16xyz | x3e | 11.5,7.5,0.0 | 1.072 | -1 | - | - | - | 4.43 | -1 | - | - | offlat | offlat | - |
| r16xyz | x3v | 11.5,7.5,7.5 | 0.941 | -1 | - | - | - | 4.62 | -1 | - | - | 1.88 | 2.3e-15,3.7e-15 | 1.2 |
| r16xyz | x4c | 12.5,0.0,0.0 | 1.178 | -1 | - | - | - | 4.74 | -1 | - | - | offlat | offlat | - |
| r16xyz | x4e | 12.5,7.5,0.0 | 1.010 | -1 | - | - | - | 4.14 | -1 | - | - | offlat | offlat | - |
| r16xyz | x4v | 12.5,7.5,7.5 | 0.898 | -1 | - | - | - | 4.40 | -1 | - | - | 1.8 | 2.3e-15,3.8e-15 | 1.2 |
| r16xyz | x6c | 14.5,0.0,0.0 | 1.015 | -1 | - | - | - | 4.04 | -1 | - | - | offlat | offlat | - |
| r16xyz | x6e | 14.5,7.5,0.0 | 0.902 | -1 | - | - | - | 3.69 | -1 | - | - | offlat | offlat | - |
| r16xyz | x6v | 14.5,7.5,7.5 | 0.819 | -1 | - | - | - | 4.01 | -1 | - | - | 1.71 | 3.7e-15,3.7e-15 | 1.2 |
| r16xyz | x8c | 16.5,0.0,0.0 | 0.892 | -1 | - | - | - | 3.59 | -1 | - | - | offlat | offlat | - |
| r16xyz | x8e | 16.5,7.5,0.0 | 0.812 | -1 | - | - | - | 3.37 | -1 | - | - | offlat | offlat | - |
| r16xyz | x8v | 16.5,7.5,7.5 | 0.751 | -1 | - | - | - | 3.68 | -1 | - | - | 1.66 | 5.1e-15,5.1e-15 | 1.4 |
| r16xyz | x16c | 24.5,0.0,0.0 | 0.601 | -1 | - | - | - | 2.75 | -1 | - | - | offlat | offlat | - |
| r16xyz | x16e | 24.5,7.5,0.0 | 0.575 | 54 | 5.0e-16,7.0e-16 | 0.5 | 1.9e-14 | 2.70 | -1 | - | - | offlat | offlat | - |
| r16xyz | x16v | 24.5,7.5,7.5 | 0.551 | 52 | 2.8e-16,9.0e-16 | 0.6 | 9.4e-15 | 2.87 | -1 | - | - | 1.59 | 1.8e-15,2.6e-15 | 1.1 |
| r16xyz | x32c | 40.5,0.0,0.0 | 0.364 | 34 | 7.8e-16,1.1e-15 | 0.7 | 4.5e-15 | 2.23 | -1 | - | - | offlat | offlat | - |
| r16xyz | x32e | 40.5,7.5,0.0 | 0.357 | 34 | 6.5e-16,1.3e-15 | 0.8 | 2.8e-15 | 2.22 | -1 | - | - | offlat | offlat | - |
| r16xyz | x32v | 40.5,7.5,7.5 | 0.352 | 32 | 4.7e-16,1.9e-15 | 0.9 | 1.7e-14 | 2.28 | -1 | - | - | 1.57 | 1.4e-15,2.7e-15 | 1.1 |
| r16xyz | d1c | 9.5,9.5,0.0 | 1.096 | -1 | - | - | - | 4.60 | -1 | - | - | offlat | offlat | - |
| r16xyz | d1e | 9.5,9.5,7.5 | 0.957 | -1 | - | - | - | 4.72 | -1 | - | - | 1.92 | 3.0e-15,3.0e-15 | 1.1 |
| r16xyz | d2c | 10.5,10.5,0.0 | 0.991 | -1 | - | - | - | 4.10 | -1 | - | - | offlat | offlat | - |
| r16xyz | d2e | 10.5,10.5,7.5 | 0.885 | -1 | - | - | - | 4.38 | -1 | - | - | 1.76 | 1.7e-15,1.7e-15 | 0.9 |
| r16xyz | d4c | 12.5,12.5,0.0 | 0.833 | -1 | - | - | - | 3.48 | -1 | - | - | offlat | offlat | - |
| r16xyz | d4e | 12.5,12.5,7.5 | 0.767 | -1 | - | - | - | 3.80 | -1 | - | - | 1.63 | 2.2e-15,2.4e-15 | 1.0 |
| r16xyz | d8c | 16.5,16.5,0.0 | 0.631 | -1 | - | - | - | 2.86 | -1 | - | - | offlat | offlat | - |
| r16xyz | d8e | 16.5,16.5,7.5 | 0.601 | -1 | - | - | - | 3.07 | -1 | - | - | 1.54 | 1.3e-15,1.3e-15 | 0.8 |
| r16xyz | d16c | 24.5,24.5,0.0 | 0.425 | 38 | 6.4e-16,6.4e-16 | 0.5 | 1.1e-14 | 2.35 | -1 | - | - | offlat | offlat | - |
| r16xyz | d16e | 24.5,24.5,7.5 | 0.415 | 38 | 6.5e-16,6.5e-16 | 0.5 | 5.2e-15 | 2.44 | -1 | - | - | 1.52 | 1.6e-15,2.9e-15 | 1.1 |

offsets with a reference: 272; whole evaluated 147 worst (1.4001895056910193e-15, "r2x d1c"); split evaluated 152 worst (1.494412632877706e-15, "r4x x2c"); gcd evaluated 156 worst (5.091613149847852e-15, "r16xyz x8v")
done load [3.86181640625, 3.15234375, 2.86376953125]

## 2d. The S2c disagreements attributed (xwork/xtable/attrib.jl -> attrib.txt; repro.jl -> repro.out)

Root's hypothesis was that the gcd average cancels (Lambda = sum_j |G_j| / |sum_j G_j| large at lambda/4-lambda/2 coarse cells).
Measured Lambda of the gcd sum: 1.20 (r8xy x6e), 1.16 (r8xyz x8v), 1.10 (d4e), 1.59 (r16xyz x16v), 1.57 (x32v) -- and 1.0-2.2
over the whole matrix (estratio_f1.txt).  The gcd average is the ACCURATE side of every disagreement (4.3e-16 .. 1.8e-15
from the reference).  The whole box was wrong, for a reason unrelated to the geometry table: its k-series was truncated
too early at high l.

Mechanism (farSetupX, and the same two lines in farfield.jl's farSetup :901-902):
`cut(mm, nc) = max.(nCutVec(.., k rNc, ..), nCutVec(.., k rHi, ..))` with nCutVec returning -1 for "no N <= nMax meets the
budget".  `max.` swallows the -1: whenever the near end is satisfiable at nMax and the far end is not (or vice versa), the
sentinel is lost and the OTHER radius' (smaller) N is used, uncertified.  With split = true, rNc was also the smallest
PIECE-centre radius (1.66 g for a ramp piece of half-width g/2 at the kap = 1 offsets), where the whole-box n-tail with
r_d = |b| is vacuous (rho = 8) and asks for N >> 12 at every l >= ~10: -1 there, swallowed, and the far-end value
(N = 5-8 at l >= 12, computed at |R| = 3300-7500 g where hbnd(l, kR) is flat) was used for the near offsets at 11-27 g,
whose high-l columns need N = 9-13.  frqBox then contracted the columns with too few k-series terms.

| shape | rNc old (incl. pieces) | rNc whole-box | nCut OLD l = 0..11 (S2/S2c runs) | nCut FIXED (cutMax, whole-box radii) | fixed incl. piece radii (rejected) |
|---|---|---|---|---|---|
| r8xy | 1.66 g | 11.07 g | 9 9 9 9 9 9 9 9 9 9 9 10 (max 12 at high l) | 9 9 9 9 8 8 8 8 7 7 7 7 (max 9) | max 15 |
| r8xyz | 1.66 g | 12.50 g | 10 10 9 9 9 10 9 10 10 10 10 10 (max 12) | 10 10 9 9 9 9 8 8 8 8 7 7 (max 10) | max 17 |
| r16xyz | 1.66 g | 25.62 g | 12 12 12 12 12 11 10 10 9 9 8 8 (max 12) | 13 13 12 12 11 11 11 11 10 10 10 10 (max 13) | max 25 |

(Full l = 0..56 vectors in ncutfull.txt, S2e.)  r16xyz also lost the far-end l = 0, 1 requirement N = 13 (the far end at
nMax = 12 returns -1, the near end 12: max = 12), so its table was served at nMax 12 instead of 13.

Fix in farx.jl: `cutMax(a, b)` (-1 absorbing) replaces `max.` in BOTH farSetup and farSetupX, and farSetupX evaluates
the whole-box n-cut at the smallest WHOLE-BOX radius only; the piece tables share that vector (they carry no
remainder certificate of their own -- a Theorem B n-tail is not derived; the split is not a production route, S4b).
With cutMax alone but rNc still including piece radii, r16xyz asked for nMax 25 and rebuilt every table (150 s,
attrib_killed.txt): the artefact of certifying the whole-box series at a radius where the whole box is not used.
farfield.jl (equal cells) has the same `max.`: at g = (1/32)^3 nCut = 6 << 12 so it never fires, but a cell with
|k| r_d >~ 2.9 (edge >= lambda/2 at f = 1) would hit it; the equal-cell path of farx.jl with cutMax is bit-identical on
the four shipped shapes (S5c) and only differs where farfield.jl was silently under-truncating.

Per-offset measurements at f = 1 (reference = 220-bit record; "vs gcd" = no record, compared with the gcd average):
est_X/max|T|, the Theorem A' bound at the selected L over max|T| and over est_X, Lambda, and the error of the whole
box with the OLD n-cut (reproduces S2c), with the FIXED n-cut on the n12 table, on the n13 table (r16xyz), with a
uniform N for every l, and of the split and gcd routes.

| shape | off | R/g | \|R\|/g | rho_w | L_w | est_X/max\|T\| | bound_A'(L)/max\|T\| | bound/est | Lambda_gcd | whole OLD n-cut | whole FIXED (n12) | FIXED (n13) | uniform N=8 | N=10 | N=11 | N=12 | split | gcd | cmp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r8xy | x6e | 10.5,3.5,0 | 11.07 | 0.582 | 54 | 3.34 | 2.3e-14 | 7.0e-15 | 1.20 | 5.9e-14,6.8e-14 | 1.1e-15,1.1e-15 | = | 1.1e-15 | 1.1e-15 | 1.1e-15 | 1.1e-15 | 9.6e-16,9.6e-16 | 4.6e-16,4.7e-16 | ref |
| r8xy | x8e | 12.5,3.5,0 | 12.98 | 0.496 | 44 | 2.99 | 7.3e-15 | 2.4e-15 | 1.12 | 1.1e-15,1.2e-15 | 1.2e-15,1.3e-15 | = | 9.1e-16 | 1.2e-15 | 1.2e-15 | 1.2e-15 | 2.9e-16,3.0e-16 | 4.3e-16,4.3e-16 | ref |
| r8xyz | x8v | 12.5,3.5,3.5 | 13.44 | 0.580 | 50 | 3.27 | 2.1e-14 | 6.5e-15 | 1.16 | 7.7e-13,3.4e-12 | 8.2e-16,8.9e-16 | = | 2.4e-15 | 8.2e-16 | 8.2e-16 | 8.2e-16 | 3.9e-16,1.1e-15 | 1.0e-15,1.0e-15 | ref |
| r8xyz | d4e | 8.5,8.5,3.5 | 12.52 | 0.623 | 56 | 3.53 | 3.0e-14 | 8.5e-15 | 1.10 | 3.2e-12,1.1e-11 | 3.4e-16,7.3e-16 | = | 1.5e-15 | 3.4e-16 | 3.4e-16 | 3.4e-16 | - (type table absent) | 8.7e-16,8.7e-16 | ref |
| r8xyz | x16v | 20.5,3.5,3.5 | 21.09 | 0.370 | 30 | 2.23 | 1.2e-14 | 5.5e-15 | 1.12 | 5.2e-16,8.2e-16 | 5.2e-16,8.2e-16 | = | 1.3e-15 | 5.2e-16 | 5.2e-16 | 5.2e-16 | 3.7e-16,4.7e-16 | 1.1e-15,1.1e-15 | ref |
| r8xyz | x6v | 10.5,3.5,3.5 | 11.61 | 0.671 | -1 | 3.85 | - | - | 1.25 | - | - | - | - | - | - | - | - | 8.5e-16,8.5e-16 | ref |
| r16xyz | x16v | 24.5,7.5,7.5 | 26.70 | 0.551 | 52 | 2.87 | 9.4e-15 | 3.3e-15 | 1.59 | 6.8e-10,4.2e-09 | 2.8e-16,9.0e-16 | 2.8e-16,9.0e-16 | 9.8e-11 | 2.0e-14 | 3.4e-16 | 2.8e-16 | - | 1.8e-15,2.6e-15 | ref |
| r16xyz | x16e | 24.5,7.5,0 | 25.62 | 0.575 | 54 | 2.70 | 2.0e-14 | 7.2e-15 | offlat | 1.5e-09,5.0e-09 | 5.0e-16,7.0e-16 | 5.0e-16,7.0e-16 | 9.1e-11 | 1.9e-14 | 4.0e-16 | 5.0e-16 | - | offlat | ref |
| r16xyz | x32v | 40.5,7.5,7.5 | 41.87 | 0.352 | 32 | 2.28 | 1.7e-14 | 7.5e-15 | 1.57 | 7.6e-14,3.0e-13 | 4.7e-16,1.9e-15 | 4.7e-16,1.9e-15 | 9.3e-11 | 2.0e-14 | 6.3e-16 | 4.7e-16 | - | 1.4e-15,2.7e-15 | ref |

L-sensitivity of the fixed whole box (attrib.txt): r8xyz x8v at L = 46/48/50/52/54: 6.1e-16/8.2e-16/8.2e-16/8.6e-16/7.8e-16;
r16xyz x16v at L = 48..54: 2.9e-16/2.9e-16/2.8e-16/2.9e-16 -- flat, i.e. the selected L is not the limiting factor.
The uniform-N columns show the sensitivity: r16xyz needs N >= 11 (N = 10 gives 2e-14, N = 8 gives 1e-10), r8 needs
N >= 10 (N = 8: 1.5e-15 .. 2.4e-15); the OLD vector had N = 8-9 at l >= 12 (r8) and N = 8-9 at l = 10-15, lower
above (r16xyz) -- exactly where the errors came from.  The frequency contraction has no cancellation to speak of
(sum|term|/|sum| worst column 3.0-4.6 at these shapes).

The rule that follows: whole box wherever Theorem A' converges within the table (the certificate at the selected L is
bound_A'/max|T| = 0.7-3 tol here, bound/est_X < tol by construction); the gcd average only where the whole box does
not converge -- the box-sum form of S4 -- where its Lambda is 1.1-2.2 (no cancellation issue at these ratios; the
cancellation root feared would appear only if the coarse cell were several wavelengths across).  est_X over-states
max|T| by 2.2-3.9 on these offsets (0.50-5.80 over the matrix, S3b), so the a priori certificate is tol est_X/max|T|;
the a posteriori check `bound <= tol max|T_computed|` with L raised until it holds is implemented in farBlockX!
(S4b) and its escalation counts are measured there.

## 3c. Recheck at f = 1 + 0.1i (recheck2.jl 2 -> recheck2_f2.md); the {x} and {x,y,z} sets

64 offsets with a reference (5 per shape: the lateral-0 x family at kap 1, 2, 4, 8, 32, plus the lateral-0 positions of
the xyz shapes are offlat for the gcd route).  Whole box evaluated at 35: worst 1.4e-15 (r16x x16c); split at 36: worst
2.6e-15 (r8x x3c); gcd at 32: worst 9.8e-16 (r2x x3c).  bound_A'/max|T| 1.1e-16 .. 2.9e-14; est_X/max|T| 0.43 .. 5.85;
Lambda_gcd 1.0 .. 1.52.  Same columns as S3b.

| shape | off | R/g | rho_w | L_w | err_w (max,ent) | dig_w | bound_A'/max\|T\| | est_X/max\|T\| | L_p | err_p (max,ent) | dig_p | Lambda_gcd | err_gcd (max,ent) | dig_g |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r2x | x1c | 2.5,0.0,0.0 | 0.825 | -1 | - | - | - | 1.67 | 51 | 8.6e-16,8.6e-16 | 0.6 | 1 | 5.8e-16,1.3e-15 | 0.8 |
| r2x | x2c | 3.5,0.0,0.0 | 0.589 | 48 | 3.4e-16,5.1e-16 | 0.4 | 1.1e-14 | 1.98 | 30 | 6.1e-16,1.5e-15 | 0.8 | 1 | 7.7e-16,1.1e-15 | 0.7 |
| r2x | x3c | 4.5,0.0,0.0 | 0.458 | 34 | 6.1e-16,1.6e-15 | 0.9 | 1.4e-14 | 2.18 | 24 | 1.2e-15,1.2e-15 | 0.7 | 1 | 9.8e-16,9.8e-16 | 0.6 |
| r2x | x4c | 5.5,0.0,0.0 | 0.375 | 28 | 2.6e-16,6.2e-16 | 0.4 | 1.3e-14 | 2.32 | 21 | 1.4e-15,1.4e-15 | 0.8 | 1 | 3.0e-16,6.6e-16 | 0.5 |
| r2x | x6c | 7.5,0.0,0.0 | 0.275 | 22 | 5.0e-16,5.5e-16 | 0.4 | 1.5e-14 | 2.54 | 17 | 1.9e-16,3.3e-16 | 0.2 | 1 | 3.8e-16,6.7e-16 | 0.5 |
| r2x | x8c | 9.5,0.0,0.0 | 0.217 | 20 | 3.2e-16,4.2e-16 | 0.3 | 3.0e-15 | 2.73 | 16 | 2.2e-16,2.2e-16 | -0.0 | 1 | 4.4e-16,4.4e-16 | 0.3 |
| r2x | x16c | 17.5,0.0,0.0 | 0.118 | 16 | 5.9e-16,5.9e-16 | 0.4 | 2.5e-16 | 2.14 | 13 | 5.8e-16,9.4e-16 | 0.6 | 1 | 2.9e-16,2.9e-16 | 0.1 |
| r2x | x32c | 33.5,0.0,0.0 | 0.062 | 12 | 3.8e-16,3.8e-16 | 0.2 | 4.0e-15 | 1.53 | 11 | 2.2e-16,6.2e-16 | 0.4 | 1 | 2.3e-16,4.5e-16 | 0.3 |
| r2xyz | x1c | 2.5,0.0,0.0 | 1.039 | -1 | - | - | - | 2.21 | -1 | - | - | offlat | offlat | - |
| r2xyz | x2c | 3.5,0.0,0.0 | 0.742 | -1 | - | - | - | 2.24 | 32 | 3.9e-16,1.0e-15 | 0.7 | offlat | offlat | - |
| r2xyz | x3c | 4.5,0.0,0.0 | 0.577 | 46 | 4.7e-16,1.2e-15 | 0.7 | 6.8e-15 | 2.34 | 25 | 5.1e-16,1.3e-15 | 0.8 | offlat | offlat | - |
| r2xyz | x4c | 5.5,0.0,0.0 | 0.472 | 34 | 5.1e-16,6.2e-16 | 0.4 | 1.7e-14 | 2.43 | 21 | 2.5e-16,5.9e-16 | 0.4 | offlat | offlat | - |
| r2xyz | x6c | 7.5,0.0,0.0 | 0.346 | 26 | 4.1e-16,7.1e-16 | 0.5 | 6.6e-15 | 2.60 | 18 | 6.2e-16,6.2e-16 | 0.4 | offlat | offlat | - |
| r2xyz | x8c | 9.5,0.0,0.0 | 0.273 | 22 | 2.2e-16,2.8e-16 | 0.1 | 5.7e-15 | 2.77 | 16 | 3.0e-16,3.0e-16 | 0.1 | offlat | offlat | - |
| r2xyz | x16c | 17.5,0.0,0.0 | 0.148 | 16 | 3.5e-16,5.6e-16 | 0.4 | 3.7e-15 | 2.15 | 13 | 4.9e-16,4.9e-16 | 0.3 | offlat | offlat | - |
| r2xyz | x32c | 33.5,0.0,0.0 | 0.078 | 14 | 7.5e-16,7.8e-16 | 0.5 | 1.1e-16 | 1.53 | 11 | 5.2e-16,5.2e-16 | 0.4 | offlat | offlat | - |
| r4x | x1c | 3.5,0.0,0.0 | 0.821 | -1 | - | - | - | 1.20 | 51 | 5.9e-16,1.4e-15 | 0.8 | 1 | 4.9e-16,1.0e-15 | 0.7 |
| r4x | x2c | 4.5,0.0,0.0 | 0.638 | -1 | - | - | - | 1.66 | 36 | 4.3e-16,9.5e-16 | 0.6 | 1 | 5.6e-16,1.0e-15 | 0.7 |
| r4x | x3c | 5.5,0.0,0.0 | 0.522 | 44 | 1.0e-15,1.0e-15 | 0.7 | 1.6e-14 | 1.97 | 30 | 1.2e-15,1.2e-15 | 0.7 | 1.01 | 7.6e-16,9.2e-16 | 0.6 |
| r4x | x4c | 6.5,0.0,0.0 | 0.442 | 36 | 5.8e-16,1.2e-15 | 0.7 | 1.6e-14 | 2.19 | 27 | 9.1e-16,1.2e-15 | 0.7 | 1.01 | 2.9e-16,6.1e-16 | 0.4 |
| r4x | x6c | 8.5,0.0,0.0 | 0.338 | 28 | 4.3e-16,4.3e-16 | 0.3 | 1.5e-14 | 2.51 | 23 | 9.0e-16,9.0e-16 | 0.6 | 1.01 | 3.5e-16,5.0e-16 | 0.4 |
| r4x | x8c | 10.5,0.0,0.0 | 0.274 | 24 | 4.8e-16,4.8e-16 | 0.3 | 1.3e-14 | 2.75 | 20 | 6.9e-16,8.1e-16 | 0.6 | 1.02 | 5.3e-16,5.3e-16 | 0.4 |
| r4x | x16c | 18.5,0.0,0.0 | 0.155 | 18 | 5.4e-16,5.4e-16 | 0.4 | 3.3e-15 | 2.09 | 16 | 1.1e-15,1.1e-15 | 0.7 | 1.02 | 1.7e-16,1.7e-16 | -0.1 |
| r4x | x32c | 34.5,0.0,0.0 | 0.083 | 14 | 3.8e-16,3.8e-16 | 0.2 | 5.3e-15 | 1.54 | 14 | 3.1e-16,5.1e-16 | 0.4 | 1.02 | 1.5e-16,5.1e-16 | 0.4 |
| r4xyz | x1c | 3.5,0.0,0.0 | 1.237 | -1 | - | - | - | 2.80 | -1 | - | - | offlat | offlat | - |
| r4xyz | x2c | 4.5,0.0,0.0 | 0.962 | -1 | - | - | - | 2.59 | -1 | - | - | offlat | offlat | - |
| r4xyz | x3c | 5.5,0.0,0.0 | 0.787 | -1 | - | - | - | 2.57 | -1 | - | - | offlat | offlat | - |
| r4xyz | x4c | 6.5,0.0,0.0 | 0.666 | -1 | - | - | - | 2.61 | 49 | 3.2e-16,6.5e-16 | 0.5 | offlat | offlat | - |
| r4xyz | x6c | 8.5,0.0,0.0 | 0.509 | 40 | 6.9e-16,6.9e-16 | 0.5 | 7.8e-15 | 2.75 | 33 | 4.0e-16,4.0e-16 | 0.3 | offlat | offlat | - |
| r4xyz | x8c | 10.5,0.0,0.0 | 0.412 | 32 | 3.6e-16,3.6e-16 | 0.2 | 5.3e-15 | 2.91 | 26 | 2.8e-16,3.2e-16 | 0.2 | offlat | offlat | - |
| r4xyz | x16c | 18.5,0.0,0.0 | 0.234 | 20 | 2.4e-16,2.4e-16 | 0.0 | 2.0e-14 | 2.11 | 19 | 5.1e-16,5.1e-16 | 0.4 | offlat | offlat | - |
| r4xyz | x32c | 34.5,0.0,0.0 | 0.126 | 16 | 2.2e-16,4.4e-16 | 0.3 | 2.6e-15 | 1.54 | 16 | 3.7e-16,3.7e-16 | 0.2 | offlat | offlat | - |
| r8x | x1c | 5.5,0.0,0.0 | 0.858 | -1 | - | - | - | 0.74 | -1 | - | - | 1.01 | 3.9e-16,8.4e-16 | 0.6 |
| r8x | x2c | 6.5,0.0,0.0 | 0.726 | -1 | - | - | - | 1.23 | -1 | - | - | 1.02 | 6.1e-16,8.9e-16 | 0.6 |
| r8x | x3c | 7.5,0.0,0.0 | 0.629 | -1 | - | - | - | 1.62 | 51 | 2.6e-15,2.6e-15 | 1.1 | 1.03 | 6.5e-16,7.2e-16 | 0.5 |
| r8x | x4c | 8.5,0.0,0.0 | 0.555 | 54 | 5.9e-16,1.0e-15 | 0.7 | 1.4e-14 | 1.94 | 44 | 6.8e-16,1.2e-15 | 0.7 | 1.04 | 3.9e-16,5.4e-16 | 0.4 |
| r8x | x6c | 10.5,0.0,0.0 | 0.449 | 42 | 1.2e-15,1.5e-15 | 0.8 | 4.6e-15 | 2.42 | 35 | 9.0e-16,1.2e-15 | 0.7 | 1.08 | 4.4e-16,4.7e-16 | 0.3 |
| r8x | x8c | 12.5,0.0,0.0 | 0.377 | 34 | 7.3e-16,7.4e-16 | 0.5 | 1.3e-14 | 2.78 | 31 | 1.4e-15,1.4e-15 | 0.8 | 1.16 | 4.4e-16,4.4e-16 | 0.3 |
| r8x | x16c | 20.5,0.0,0.0 | 0.230 | 24 | 3.7e-16,3.7e-16 | 0.2 | 3.4e-15 | 2.07 | 23 | 1.7e-15,1.7e-15 | 0.9 | 1.09 | 4.2e-16,4.2e-16 | 0.3 |
| r8x | x32c | 36.5,0.0,0.0 | 0.129 | 18 | 4.5e-16,4.5e-16 | 0.3 | 4.0e-15 | 1.61 | 18 | 7.9e-16,7.9e-16 | 0.6 | 1.1 | 1.3e-16,4.7e-16 | 0.3 |
| r8xyz | x1c | 5.5,0.0,0.0 | 1.417 | -1 | - | - | - | 3.91 | -1 | - | - | offlat | offlat | - |
| r8xyz | x2c | 6.5,0.0,0.0 | 1.199 | -1 | - | - | - | 3.45 | -1 | - | - | offlat | offlat | - |
| r8xyz | x3c | 7.5,0.0,0.0 | 1.039 | -1 | - | - | - | 3.26 | -1 | - | - | offlat | offlat | - |
| r8xyz | x4c | 8.5,0.0,0.0 | 0.917 | -1 | - | - | - | 3.19 | -1 | - | - | offlat | offlat | - |
| r8xyz | x6c | 10.5,0.0,0.0 | 0.742 | -1 | - | - | - | 3.22 | -1 | - | - | offlat | offlat | - |
| r8xyz | x8c | 12.5,0.0,0.0 | 0.624 | 56 | 9.0e-16,9.8e-16 | 0.6 | 2.9e-14 | 3.08 | -1 | - | - | offlat | offlat | - |
| r8xyz | x16c | 20.5,0.0,0.0 | 0.380 | 32 | 4.5e-16,4.5e-16 | 0.3 | 3.0e-15 | 2.14 | 31 | 2.9e-16,4.2e-16 | 0.3 | offlat | offlat | - |
| r8xyz | x32c | 36.5,0.0,0.0 | 0.214 | 22 | 3.7e-16,8.6e-16 | 0.6 | 2.4e-15 | 1.63 | 23 | 2.0e-16,2.5e-16 | 0.1 | offlat | offlat | - |
| r16x | x1c | 9.5,0.0,0.0 | 0.907 | -1 | - | - | - | 0.43 | -1 | - | - | 1.04 | 4.2e-16,9.0e-16 | 0.6 |
| r16x | x2c | 10.5,0.0,0.0 | 0.821 | -1 | - | - | - | 0.84 | -1 | - | - | 1.09 | 6.3e-16,7.8e-16 | 0.5 |
| r16x | x3c | 11.5,0.0,0.0 | 0.749 | -1 | - | - | - | 1.26 | -1 | - | - | 1.16 | 7.0e-16,7.7e-16 | 0.5 |
| r16x | x4c | 12.5,0.0,0.0 | 0.689 | -1 | - | - | - | 1.66 | -1 | - | - | 1.23 | 3.6e-16,5.3e-16 | 0.4 |
| r16x | x6c | 14.5,0.0,0.0 | 0.594 | -1 | - | - | - | 2.38 | -1 | - | - | 1.4 | 3.1e-16,3.4e-16 | 0.2 |
| r16x | x8c | 16.5,0.0,0.0 | 0.522 | 54 | 8.7e-16,8.7e-16 | 0.6 | 1.1e-14 | 2.65 | -1 | - | - | 1.43 | 3.4e-16,3.8e-16 | 0.2 |
| r16x | x16c | 24.5,0.0,0.0 | 0.352 | 34 | 1.4e-15,1.4e-15 | 0.8 | 1.7e-14 | 2.42 | -1 | - | - | 1.46 | 3.5e-16,3.5e-16 | 0.2 |
| r16x | x32c | 40.5,0.0,0.0 | 0.213 | 26 | 8.1e-16,9.8e-16 | 0.6 | 9.7e-16 | 2.10 | -1 | - | - | 1.52 | 2.5e-16,4.7e-16 | 0.3 |
| r16xyz | x1c | 9.5,0.0,0.0 | 1.550 | -1 | - | - | - | 5.85 | -1 | - | - | offlat | offlat | - |
| r16xyz | x2c | 10.5,0.0,0.0 | 1.402 | -1 | - | - | - | 5.40 | -1 | - | - | offlat | offlat | - |
| r16xyz | x3c | 11.5,0.0,0.0 | 1.280 | -1 | - | - | - | 4.99 | -1 | - | - | offlat | offlat | - |
| r16xyz | x4c | 12.5,0.0,0.0 | 1.178 | -1 | - | - | - | 4.50 | -1 | - | - | offlat | offlat | - |
| r16xyz | x6c | 14.5,0.0,0.0 | 1.015 | -1 | - | - | - | 3.84 | -1 | - | - | offlat | offlat | - |
| r16xyz | x8c | 16.5,0.0,0.0 | 0.892 | -1 | - | - | - | 3.42 | -1 | - | - | offlat | offlat | - |
| r16xyz | x16c | 24.5,0.0,0.0 | 0.601 | -1 | - | - | - | 2.65 | -1 | - | - | offlat | offlat | - |
| r16xyz | x32c | 40.5,0.0,0.0 | 0.364 | 34 | 1.7e-16,6.5e-16 | 0.5 | 4.5e-15 | 2.17 | -1 | - | - | offlat | offlat | - |

offsets with a reference: 64; whole evaluated 35 worst (1.3540522813976775e-15, "r16x x16c"); split evaluated 36 worst (2.568542488925113e-15, "r8x x3c"); gcd evaluated 32 worst (9.768702336863342e-16, "r2x x3c")
done load [3.42578125, 3.14599609375, 2.88525390625]

## 3d. Recheck at f = 0.37 (recheck2.jl 3 -> recheck2_f3.md); the {x} and {x,y,z} sets

64 offsets with a reference.  Whole box evaluated at 36: worst 2.4e-15 (r16x x16c); split at 36: worst 2.5e-15 (r4x x2c);
gcd at 32: worst 2.7e-15 (r2x x1c).  bound_A'/max|T| 4.8e-16 .. 2.5e-14; est_X/max|T| 0.25 .. 4.47 (the 0.25 is r16x x1c:
at f = 0.37 the pointwise scale under-states the near-field tensor of the slender pair by 4x, so the a priori
certificate there is 4x tighter than tol); Lambda_gcd 1.0 .. 1.04.

| shape | off | R/g | rho_w | L_w | err_w (max,ent) | dig_w | bound_A'/max\|T\| | est_X/max\|T\| | L_p | err_p (max,ent) | dig_p | Lambda_gcd | err_gcd (max,ent) | dig_g |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r2x | x1c | 2.5,0.0,0.0 | 0.825 | -1 | - | - | - | 1.42 | 51 | 1.5e-15,3.1e-15 | 1.1 | 1 | 2.7e-15,2.7e-15 | 1.1 |
| r2x | x2c | 3.5,0.0,0.0 | 0.589 | 48 | 5.2e-16,5.2e-16 | 0.4 | 8.3e-15 | 1.66 | 31 | 1.5e-16,2.2e-16 | 0.0 | 1 | 7.4e-16,7.4e-16 | 0.5 |
| r2x | x3c | 4.5,0.0,0.0 | 0.458 | 34 | 1.1e-15,1.1e-15 | 0.7 | 1.1e-14 | 1.81 | 24 | 3.6e-16,6.8e-16 | 0.5 | 1 | 6.0e-16,6.0e-16 | 0.4 |
| r2x | x4c | 5.5,0.0,0.0 | 0.375 | 28 | 5.3e-16,1.2e-15 | 0.7 | 9.4e-15 | 1.93 | 21 | 6.8e-16,6.8e-16 | 0.5 | 1 | 2.4e-16,5.4e-16 | 0.4 |
| r2x | x6c | 7.5,0.0,0.0 | 0.275 | 22 | 3.8e-16,8.1e-16 | 0.6 | 9.7e-15 | 2.12 | 17 | 4.5e-16,1.2e-15 | 0.7 | 1 | 2.4e-16,2.4e-16 | 0.0 |
| r2x | x8c | 9.5,0.0,0.0 | 0.217 | 20 | 2.0e-16,5.6e-16 | 0.4 | 1.7e-15 | 2.25 | 15 | 5.7e-16,5.7e-16 | 0.4 | 1 | 3.1e-16,5.3e-16 | 0.4 |
| r2x | x16c | 17.5,0.0,0.0 | 0.118 | 14 | 3.1e-16,3.1e-16 | 0.1 | 1.2e-14 | 2.60 | 12 | 3.1e-16,3.1e-16 | 0.1 | 1 | 4.7e-16,4.7e-16 | 0.3 |
| r2x | x32c | 33.5,0.0,0.0 | 0.062 | 12 | 4.5e-16,4.5e-16 | 0.3 | 9.4e-16 | 2.96 | 11 | 3.0e-16,3.0e-16 | 0.1 | 1 | 3.6e-16,3.6e-16 | 0.2 |
| r2xyz | x1c | 2.5,0.0,0.0 | 1.039 | -1 | - | - | - | 1.90 | -1 | - | - | offlat | offlat | - |
| r2xyz | x2c | 3.5,0.0,0.0 | 0.742 | -1 | - | - | - | 1.90 | 32 | 2.6e-16,5.5e-16 | 0.4 | offlat | offlat | - |
| r2xyz | x3c | 4.5,0.0,0.0 | 0.577 | 44 | 5.9e-16,1.3e-15 | 0.8 | 1.8e-14 | 1.96 | 25 | 3.7e-16,3.7e-16 | 0.2 | offlat | offlat | - |
| r2xyz | x4c | 5.5,0.0,0.0 | 0.472 | 34 | 9.9e-16,9.9e-16 | 0.6 | 1.2e-14 | 2.03 | 21 | 2.8e-16,4.2e-16 | 0.3 | offlat | offlat | - |
| r2xyz | x6c | 7.5,0.0,0.0 | 0.346 | 26 | 8.7e-16,8.7e-16 | 0.6 | 4.3e-15 | 2.17 | 18 | 2.3e-16,6.0e-16 | 0.4 | offlat | offlat | - |
| r2xyz | x8c | 9.5,0.0,0.0 | 0.273 | 22 | 1.2e-16,1.2e-16 | -0.3 | 3.2e-15 | 2.29 | 16 | 1.5e-16,4.2e-16 | 0.3 | offlat | offlat | - |
| r2xyz | x16c | 17.5,0.0,0.0 | 0.148 | 16 | 2.6e-16,6.1e-16 | 0.4 | 1.7e-15 | 2.61 | 13 | 2.4e-16,5.4e-16 | 0.4 | offlat | offlat | - |
| r2xyz | x32c | 33.5,0.0,0.0 | 0.078 | 12 | 7.1e-16,7.4e-16 | 0.5 | 6.4e-15 | 2.96 | 11 | 3.7e-16,3.7e-16 | 0.2 | offlat | offlat | - |
| r4x | x1c | 3.5,0.0,0.0 | 0.821 | -1 | - | - | - | 0.96 | 51 | 1.2e-15,2.4e-15 | 1.0 | 1 | 2.3e-15,2.3e-15 | 1.0 |
| r4x | x2c | 4.5,0.0,0.0 | 0.638 | -1 | - | - | - | 1.33 | 36 | 2.5e-15,2.5e-15 | 1.1 | 1 | 7.3e-16,7.3e-16 | 0.5 |
| r4x | x3c | 5.5,0.0,0.0 | 0.522 | 44 | 3.0e-16,6.8e-16 | 0.5 | 1.1e-14 | 1.59 | 30 | 8.2e-16,8.9e-16 | 0.6 | 1 | 4.6e-16,4.6e-16 | 0.3 |
| r4x | x4c | 6.5,0.0,0.0 | 0.442 | 36 | 3.9e-16,8.9e-16 | 0.6 | 1.0e-14 | 1.77 | 27 | 4.6e-16,1.1e-15 | 0.7 | 1 | 4.0e-16,4.0e-16 | 0.3 |
| r4x | x6c | 8.5,0.0,0.0 | 0.338 | 28 | 1.6e-16,4.3e-16 | 0.3 | 8.8e-15 | 2.03 | 23 | 6.3e-16,6.3e-16 | 0.4 | 1 | 2.6e-16,2.6e-16 | 0.1 |
| r4x | x8c | 10.5,0.0,0.0 | 0.274 | 24 | 7.5e-16,7.5e-16 | 0.5 | 6.8e-15 | 2.20 | 20 | 3.8e-16,5.1e-16 | 0.4 | 1 | 2.2e-16,4.2e-16 | 0.3 |
| r4x | x16c | 18.5,0.0,0.0 | 0.155 | 18 | 1.5e-16,3.2e-16 | 0.2 | 1.5e-15 | 2.60 | 16 | 4.1e-16,5.3e-16 | 0.4 | 1 | 2.5e-16,4.2e-16 | 0.3 |
| r4x | x32c | 34.5,0.0,0.0 | 0.083 | 14 | 5.1e-16,5.1e-16 | 0.4 | 1.1e-15 | 2.88 | 13 | 8.3e-16,8.3e-16 | 0.6 | 1 | 3.9e-16,3.9e-16 | 0.2 |
| r4xyz | x1c | 3.5,0.0,0.0 | 1.237 | -1 | - | - | - | 2.39 | -1 | - | - | offlat | offlat | - |
| r4xyz | x2c | 4.5,0.0,0.0 | 0.962 | -1 | - | - | - | 2.17 | -1 | - | - | offlat | offlat | - |
| r4xyz | x3c | 5.5,0.0,0.0 | 0.787 | -1 | - | - | - | 2.14 | -1 | - | - | offlat | offlat | - |
| r4xyz | x4c | 6.5,0.0,0.0 | 0.666 | -1 | - | - | - | 2.16 | 49 | 2.0e-16,5.0e-16 | 0.3 | offlat | offlat | - |
| r4xyz | x6c | 8.5,0.0,0.0 | 0.509 | 38 | 4.5e-16,7.0e-16 | 0.5 | 2.2e-14 | 2.26 | 32 | 3.1e-16,8.3e-16 | 0.6 | offlat | offlat | - |
| r4xyz | x8c | 10.5,0.0,0.0 | 0.412 | 30 | 3.0e-16,8.8e-16 | 0.6 | 2.0e-14 | 2.35 | 26 | 3.6e-16,3.6e-16 | 0.2 | offlat | offlat | - |
| r4xyz | x16c | 18.5,0.0,0.0 | 0.234 | 20 | 2.0e-16,4.2e-16 | 0.3 | 8.8e-15 | 2.65 | 19 | 3.8e-16,4.2e-16 | 0.3 | offlat | offlat | - |
| r4xyz | x32c | 34.5,0.0,0.0 | 0.126 | 16 | 3.2e-16,3.2e-16 | 0.2 | 4.8e-16 | 2.88 | 15 | 2.7e-16,2.9e-16 | 0.1 | offlat | offlat | - |
| r8x | x1c | 5.5,0.0,0.0 | 0.858 | -1 | - | - | - | 0.53 | -1 | - | - | 1 | 2.0e-15,2.0e-15 | 1.0 |
| r8x | x2c | 6.5,0.0,0.0 | 0.726 | -1 | - | - | - | 0.89 | -1 | - | - | 1 | 4.4e-16,4.4e-16 | 0.3 |
| r8x | x3c | 7.5,0.0,0.0 | 0.629 | -1 | - | - | - | 1.19 | 50 | 1.3e-15,1.3e-15 | 0.8 | 1 | 5.7e-16,5.7e-16 | 0.4 |
| r8x | x4c | 8.5,0.0,0.0 | 0.555 | 54 | 1.5e-15,1.5e-15 | 0.8 | 7.5e-15 | 1.44 | 43 | 1.6e-15,1.6e-15 | 0.9 | 1 | 1.4e-16,3.3e-16 | 0.2 |
| r8x | x6c | 10.5,0.0,0.0 | 0.449 | 40 | 5.5e-16,5.5e-16 | 0.4 | 1.1e-14 | 1.80 | 35 | 9.4e-16,1.3e-15 | 0.8 | 1 | 2.5e-16,2.5e-16 | 0.1 |
| r8x | x8c | 12.5,0.0,0.0 | 0.377 | 34 | 9.5e-16,9.5e-16 | 0.6 | 5.1e-15 | 2.05 | 30 | 4.9e-16,6.8e-16 | 0.5 | 1 | 2.6e-16,3.6e-16 | 0.2 |
| r8x | x16c | 20.5,0.0,0.0 | 0.230 | 22 | 6.8e-16,6.8e-16 | 0.5 | 2.5e-14 | 2.58 | 22 | 1.1e-15,1.1e-15 | 0.7 | 1.01 | 1.3e-16,2.5e-16 | 0.1 |
| r8x | x32c | 36.5,0.0,0.0 | 0.129 | 18 | 1.3e-15,1.3e-15 | 0.8 | 5.3e-16 | 2.75 | 17 | 2.7e-16,2.7e-16 | 0.1 | 1.01 | 3.0e-16,3.0e-16 | 0.1 |
| r8xyz | x1c | 5.5,0.0,0.0 | 1.417 | -1 | - | - | - | 3.32 | -1 | - | - | offlat | offlat | - |
| r8xyz | x2c | 6.5,0.0,0.0 | 1.199 | -1 | - | - | - | 2.86 | -1 | - | - | offlat | offlat | - |
| r8xyz | x3c | 7.5,0.0,0.0 | 1.039 | -1 | - | - | - | 2.64 | -1 | - | - | offlat | offlat | - |
| r8xyz | x4c | 8.5,0.0,0.0 | 0.917 | -1 | - | - | - | 2.54 | -1 | - | - | offlat | offlat | - |
| r8xyz | x6c | 10.5,0.0,0.0 | 0.742 | -1 | - | - | - | 2.49 | -1 | - | - | offlat | offlat | - |
| r8xyz | x8c | 12.5,0.0,0.0 | 0.624 | 56 | 1.5e-15,1.6e-15 | 0.9 | 1.3e-14 | 2.52 | -1 | - | - | offlat | offlat | - |
| r8xyz | x16c | 20.5,0.0,0.0 | 0.380 | 30 | 4.0e-16,4.0e-16 | 0.3 | 9.3e-15 | 2.75 | 30 | 2.6e-16,4.8e-16 | 0.3 | offlat | offlat | - |
| r8xyz | x32c | 36.5,0.0,0.0 | 0.214 | 20 | 3.4e-16,3.4e-16 | 0.2 | 8.3e-15 | 2.77 | 21 | 6.5e-16,7.5e-16 | 0.5 | offlat | offlat | - |
| r16x | x1c | 9.5,0.0,0.0 | 0.907 | -1 | - | - | - | 0.25 | -1 | - | - | 1 | 2.0e-15,2.0e-15 | 0.9 |
| r16x | x2c | 10.5,0.0,0.0 | 0.821 | -1 | - | - | - | 0.51 | -1 | - | - | 1 | 4.7e-16,4.7e-16 | 0.3 |
| r16x | x3c | 11.5,0.0,0.0 | 0.749 | -1 | - | - | - | 0.77 | -1 | - | - | 1 | 4.9e-16,4.9e-16 | 0.3 |
| r16x | x4c | 12.5,0.0,0.0 | 0.689 | -1 | - | - | - | 1.01 | -1 | - | - | 1 | 2.5e-16,3.4e-16 | 0.2 |
| r16x | x6c | 14.5,0.0,0.0 | 0.594 | -1 | - | - | - | 1.42 | -1 | - | - | 1.01 | 2.6e-16,2.6e-16 | 0.1 |
| r16x | x8c | 16.5,0.0,0.0 | 0.522 | 52 | 9.6e-16,9.6e-16 | 0.6 | 1.1e-14 | 1.74 | -1 | - | - | 1.01 | 2.3e-16,4.1e-16 | 0.3 |
| r16x | x16c | 24.5,0.0,0.0 | 0.352 | 34 | 2.4e-15,2.4e-15 | 1.0 | 3.4e-15 | 2.51 | -1 | - | - | 1.03 | 1.6e-16,2.5e-16 | 0.0 |
| r16x | x32c | 40.5,0.0,0.0 | 0.213 | 24 | 1.5e-15,1.5e-15 | 0.8 | 1.2e-15 | 2.58 | -1 | - | - | 1.04 | 3.6e-16,3.6e-16 | 0.2 |
| r16xyz | x1c | 9.5,0.0,0.0 | 1.550 | -1 | - | - | - | 4.47 | -1 | - | - | offlat | offlat | - |
| r16xyz | x2c | 10.5,0.0,0.0 | 1.402 | -1 | - | - | - | 3.95 | -1 | - | - | offlat | offlat | - |
| r16xyz | x3c | 11.5,0.0,0.0 | 1.280 | -1 | - | - | - | 3.61 | -1 | - | - | offlat | offlat | - |
| r16xyz | x4c | 12.5,0.0,0.0 | 1.178 | -1 | - | - | - | 3.38 | -1 | - | - | offlat | offlat | - |
| r16xyz | x6c | 14.5,0.0,0.0 | 1.015 | -1 | - | - | - | 3.12 | -1 | - | - | offlat | offlat | - |
| r16xyz | x8c | 16.5,0.0,0.0 | 0.892 | -1 | - | - | - | 3.01 | -1 | - | - | offlat | offlat | - |
| r16xyz | x16c | 24.5,0.0,0.0 | 0.601 | 56 | 4.4e-16,6.4e-16 | 0.5 | 1.3e-14 | 3.03 | -1 | - | - | offlat | offlat | - |
| r16xyz | x32c | 40.5,0.0,0.0 | 0.364 | 30 | 3.9e-16,3.9e-16 | 0.2 | 1.8e-14 | 2.64 | -1 | - | - | offlat | offlat | - |

offsets with a reference: 64; whole evaluated 36 worst (2.4409703541559522e-15, "r16x x16c"); split evaluated 36 worst (2.497423559459843e-15, "r4x x2c"); gcd evaluated 32 worst (2.673324552344131e-15, "r2x x1c")
done load [3.3564453125, 3.22265625, 2.95556640625]

## 4b. Production form of the near band: farBlockX! (farx.jl :1858-2079; driver xwork/xtable/blockx.jl -> blockx_{r4,big}_t{1,12}.txt)

`farBlockX!(G, sTQ, sSQ, f, RQs; tol = 1e-14, tolG = tol/8, post = true, nQ = 12)` fills G[:, :, i] for a list of exact
rational target-centre offsets (source at the origin) and returns (route, L, cert, rel, stat).  Per offset:
  route 1  whole trapezoid box (Theorem A') wherever `boundLX` converges within the table (farSetupX with split = false,
           gcd = false, sized on the offsets; with post the table is built to LMAX = 56 up front so the escalation has room);
  route 2  the gcd average as a BOX SUM: the fine equal-cell egoToe of shape g = min(sT, sS) is filled ONCE by farBlock!
           (the equal-cell library, tol tolG) over the (max |D| + 1)^3 box the near band needs, then
           G(R) = (1/N_t) sum_{t} mult_1(t_1) mult_2(t_2) mult_3(t_3) sgn-reflect(ego[|m + t|]),  t_i = j_i - j'_i,
           mult_i(t) = #{(j, j') : j - j' = t} = min(nT_i, t + nS_i) - max(1, t + 1) + 1, m = R/g + (nS - nT)/2 the
           integer lattice coordinate (`xLat`; nothing off lattice): prod_i (nT_i + nS_i - 1) weighted adds, no
           per-offset expansion (boxSumX!).  Sub-offsets of max-norm 1 (if a pair needs them; none does in either
           block) are taken from the equal-cell k-series ksrCached!; a sub-offset 0 (self term) sends the offset to route 3.
  route 3  the k-series on the 36 unequal face pairs (`tnsKsrX`: facePairX from the brief, pairMoments on the two panels,
           N from ksrOrd(|k| D_max, tol/max(1, Lambda)), MOMCX cache; in-memory only, no disk cache yet) for off-lattice
           pairs and self-term pairs.  Not exercised by either block (0 offsets), so it is untested against a reference.
Certificates: route 1 cert = tol 2^-q est_X(|R|) with q the finest level (0..nQ) whose tabulated threshold the offset
clears at the selected L (`thrQ[q]` = whlThr at tol 2^-q, monotone in R because every tail term has l >= 2); route 2
cert = (1/N_t) sum of the pieces' tolG est_g(|D|) (a priori, from the equal-cell selection; 1e-16 est_g for k-series
pieces); route 3 tol est_X.  rel = cert/max|G_computed| is the a posteriori certificate; with post = true route 1 raises L
by 2 while cert > tol max|G| and L + 2 <= Lw; `stat.fail` counts the offsets still above tol max|G| at the table's top.

### r = 4 xyz test block of xgeom (coarse 2^3 of (1/8)^3 at 0, fine 8^3 of g sharing the +x face; R = (k + 1/2) g,
k_x = 2..13, k_y, k_z = -6..5: 1728 offsets, 36 touching removed -> 1692), f = 1, JULIA_NUM_THREADS = 1, load 3.7:

| item | value |
|---|---|
| total (after a warm-up call on 8 offsets; tables in memory) | 1.56 s: setupX 0.33, route 0.00, fineBlock 1.21 (farSetup g at tol/8 + farBlock! on 9^3), fill 0.01, ksr 0 |
| routes 1/2/3 | 1352 / 340 / 0; off-lattice 0; max-norm-1 pieces needed: no; fine tables Lw/Lo 56/56; whole table Lw 56, nCut max 8 |
| a posteriori | 1216 escalations of L (of 1352 route-1 offsets), 16 left with cert > tol max\|G\| at L = 56 (rel up to 2.5e-14) |
| whole-box L histogram after escalation | 24:32 26:224 28:180 30:132 32:132 34:80 36:88 38:64 40:72 42:68 44:32 46:60 48:40 50:36 52:40 54:24 56:48 |
| route 1 rel certificate | min 8.2e-16, median 3.9e-15, max 2.5e-14 (16 above tol) |
| route 2 rel certificate | min 3.6e-15, median 5.5e-15, max 8.2e-15 (0 above tol) |
| box sum vs tnsGcd! (the per-offset gcd average) on the 340 route-2 offsets | worst 9.3e-16 (summation order) |
| whole box vs gcd average on the 1352 on-lattice route-1 offsets | worst 2.0e-15 |
| maxrss | 0.88 GB |

Against the references in the block (9 of the block's offsets have a record: the lateral-0 family is off the block's
half-integer lattice):

| R/g | route | L | cert/max\|G\| | err (max-entry, worst entry) | digits lost |
|---|---|---|---|---|---|
| 3.5,1.5,-1.5 | 2 | - | 5.0e-15 | 4.6e-16, 5.8e-16 | 0.4 |
| 3.5,1.5,1.5 | 2 | - | 5.0e-15 | 8.7e-16, 1.1e-15 | 0.7 |
| 4.5,1.5,1.5 | 2 | - | 4.4e-15 | 2.4e-16, 5.8e-16 | 0.4 |
| 5.5,1.5,1.5 | 2 | - | 4.1e-15 | 1.5e-16, 3.7e-16 | 0.2 |
| 6.5,1.5,1.5 | 1 | 56 | 1.5e-14 | 5.5e-16, 1.5e-15 | 0.8 |
| 8.5,1.5,1.5 | 1 | 40 | 3.8e-15 | 4.2e-16, 1.1e-15 | 0.7 |
| 10.5,1.5,1.5 | 1 | 32 | 4.0e-15 | 3.5e-16, 1.1e-15 | 0.7 |
| 3.5,3.5,1.5 | 2 | - | 5.8e-15 | 3.2e-16, 5.4e-16 | 0.4 |
| 4.5,4.5,1.5 | 2 | - | 5.9e-15 | 3.0e-16, 4.3e-16 | 0.3 |

Worst 8.7e-16 (max-entry); every certificate holds (err < cert/max|G| at every row).  Gila today on the same block
(xgeom S2a): 3.5 s adaptive + 12.6 s fixed + the contact self build, single thread.

### The realistic block: coarse 4^3 of 16 g = (1/2)^3 vs fine 64^3 of g sharing the x face (S4): 1,404,928 offsets,
324 touching removed -> 1,404,604; f = 1.  Two processes, one JULIA_NUM_THREADS = 1 (load 3.2-3.3) and one
JULIA_NUM_THREADS = 12 (the only multi-threaded run of the round, alone, load 2.8-2.9); each after a warm-up call
on 8 offsets (compile time excluded; the trapezoid n13 table L = 56 and the fine g tables were on disk).

| phase | 1 thread | 12 threads |
|---|---|---|
| setupX (farSetupX, whole box, 1.4e6 bndLw + 13 whlThr levels) | 0.53 s | 0.52 s |
| route decisions | 0.06 s | 0.03 s |
| fineBlock (farSetup g at tol/8 + farBlock! on the 33^3 box, 35,929 offsets, + est table) | 1.26 s | 1.03 s |
| ksr (route 3) | 0 (no offsets) | 0 |
| fill (1,387,696 whole-box evaluations incl. 434,324 escalations + 16,908 box sums of 4096 terms) | 4.49 s | 0.57 s |
| total | 6.39 s | 2.19 s |
| maxrss | 1.34 GB | 1.24 GB |

routes 1/2/3 = 1,387,696 / 16,908 / 0 (S4's census exactly); off-lattice 0; whole table Lw 56, nCut max 13; fine tables
Lw/Lo 56/56.  Whole-box L histogram after escalation: 22:454848 24:330796 26:219272 28:141272 30:83684 32:48504 34:30540
36:20472 38:14124 40:10532 42:7860 44:6056 46:4704 48:3800 50:3104 52:2500 54:2140 56:3488 (S4's a priori histogram
started at 20:13636, 22:539332; the escalation moves ~31% of the offsets up one shell).  Route 1 rel certificate: min
1.5e-16, median 2.2e-15, max 4.1e-14; 1800 offsets (0.13%) still above tol at L = 56 -- all at rho_w 0.55-0.60, where
the a priori level is 1-4 tol and the table has no more shells; they are correct to <= 3e-15 where a reference exists
(24.5,7.5,7.5 at L = 54: 2.9e-16) but their proven bound is 1-4 tol.  Route 2 rel certificate: min 3.5e-15, median
4.4e-15, max 9.6e-15, none above tol.  The fill parallelises 7.9x on 12 threads (0.57 s); fineBlock is 1.0-1.3 s
because farSetup(g) + the est table are serial (farBlock! itself threads).  Against the 13 references inside the block:

| R/g | route | L | cert/max\|G\| | err (max-entry, worst entry) | digits lost |
|---|---|---|---|---|---|
| 9.5,7.5,7.5 | 2 | - | 8.3e-15 | 3.2e-15, 4.7e-15 | 1.3 |
| 10.5,7.5,7.5 | 2 | - | 7.3e-15 | 2.5e-15, 2.5e-15 | 1.1 |
| 11.5,7.5,7.5 | 2 | - | 6.7e-15 | 1.9e-15, 3.7e-15 | 1.2 |
| 12.5,7.5,7.5 | 2 | - | 6.2e-15 | 2.4e-15, 3.2e-15 | 1.2 |
| 14.5,7.5,7.5 | 2 | - | 5.4e-15 | 2.3e-15, 3.2e-15 | 1.2 |
| 16.5,7.5,7.5 | 2 | - | 4.8e-15 | 2.2e-15, 2.9e-15 | 1.1 |
| 24.5,7.5,7.5 | 1 | 54 | 3.6e-15 | 2.9e-16, 9.4e-16 | 0.6 |
| 40.5,7.5,7.5 | 1 | 34 | 2.8e-15 | 4.0e-16, 1.6e-15 | 0.9 |
| 9.5,9.5,7.5 | 2 | - | 7.1e-15 | 3.1e-15, 3.1e-15 | 1.1 |
| 10.5,10.5,7.5 | 2 | - | 6.2e-15 | 2.2e-15, 3.4e-15 | 1.2 |
| 12.5,12.5,7.5 | 2 | - | 5.1e-15 | 3.5e-15, 4.6e-15 | 1.3 |
| 16.5,16.5,7.5 | 2 | - | 3.9e-15 | 2.4e-15, 2.4e-15 | 1.0 |
| 24.5,24.5,7.5 | 1 | 38 | 6.1e-15 | 6.5e-16, 6.5e-16 | 0.5 |

Worst 3.5e-15 (route 2, 4096-term sums: the 1e-15 floor of the equal-cell pieces adds up to 2-5e-15 with Lambda 1.6-2.2);
every row satisfies err <= cert/max|G|.  Identical numbers from the 1- and 12-thread runs (same references matched,
same worst error 3.49e-15, same histograms and certificate statistics).  Cost against Gila today for this block
(xgeom S3): ~1.8 h single-thread (1.37e6 egoSrfFxd! + 36,540 egoSrfAdp!) -> 6.4 s single-thread / 2.2 s on 12 threads,
with a certificate per offset.

## 2f. Near band and far offsets at f = 1 + 0.1i and f = 0.37 (near.jl, driveall.sh; {x} and {x,y,z} sets; agg_f2.md, agg_f3.md)

These eight-shape runs were produced by the first incarnation with the OLD n-cut (S2d), so their whole-box values at
r8xyz x8v/d4e and r16xyz x16v carry the same artefact (whole vs gcd 9.6e-13, 3.9e-12, 1.1e-9 at f = 1+0.1i; 4.3e-11
at f = 0.37 for x16v) -- superseded by S3c/S3d and S3e, which use the fixed code; the routes, L, timings and the
split/gcd columns stand.  Columns as in S2; times in us, single thread, load 10.4-13.1 (the machine was shared).

### Near band f index 2: kap = 1, 2, 3 (x family) and the diagonal kap = 1, 2; times in us at the stated load

| shape | off | R/g | rho_w | L_w | t_w | err_w (max,ent) | rho_p | L_p | t_p | err_p | rho_gcd | sub-routes 1/2/3 | N_s | t_gcd | err_gcd | cross-route agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r2x | x1c | 2.5,0.0,0.0 | 0.825 | -1 | NaN | - | 0.522 | 51 | 146.52 | - | 0.866 | 1/1/0 | 2 | 158.8 | - | split vs gcd: 1.41e-15 (max-entry) |
| r2x | x2c | 3.5,0.0,0.0 | 0.589 | 48 | 9.89 | - | 0.333 | 30 | 68.37 | - | 0.577 | 2/0/0 | 2 | 37.2 | - | whole vs gcd: 1.11e-15 (max-entry); split vs gcd: 5.69e-16 (max-entry); whole vs split: 5.44e-16 (max-entry) |
| r2x | x3c | 4.5,0.0,0.0 | 0.458 | 34 | 4.81 | - | 0.243 | 24 | 46.63 | - | 0.433 | 2/0/0 | 2 | 33.9 | - | whole vs gcd: 7.54e-16 (max-entry); split vs gcd: 6.61e-16 (max-entry); whole vs split: 9.97e-16 (max-entry) |
| r2x | d1c | 2.5,2.0,0.0 | 0.644 | 56 | 13.24 | - | 0.397 | 36 | 73.11 | - | 0.612 | 2/0/0 | 2 | 35.1 | - | whole vs gcd: 6.50e-16 (max-entry); split vs gcd: 1.08e-15 (max-entry); whole vs split: 1.73e-15 (max-entry) |
| r2x | d2c | 3.5,3.0,0.0 | 0.447 | 34 | 4.76 | - | 0.243 | 24 | 45.01 | - | 0.408 | 2/0/0 | 2 | 33.0 | - | whole vs gcd: 3.45e-16 (max-entry); split vs gcd: 9.29e-16 (max-entry); whole vs split: 9.26e-16 (max-entry) |
| r2xyz | x1c | 2.5,0.0,0.0 | 1.039 | -1 | NaN | - | 0.577 | -1 | NaN | - | NaN | offlat | 8 | NaN | - |  |
| r2xyz | x1e | 2.5,0.5,0.0 | 1.019 | -1 | NaN | - | 0.548 | 56 | 195.44 | - | NaN | offlat | 8 | NaN | - |  |
| r2xyz | x1v | 2.5,0.5,0.5 | 1.000 | -1 | NaN | - | 0.522 | 52 | 206.82 | - | 0.866 | 4/4/0 | 8 | 527.7 | - | split vs gcd: 4.04e-16 (max-entry) |
| r2xyz | x2c | 3.5,0.0,0.0 | 0.742 | -1 | NaN | - | 0.346 | 32 | 115.39 | - | NaN | offlat | 8 | NaN | - |  |
| r2xyz | x2e | 3.5,0.5,0.0 | 0.735 | -1 | NaN | - | 0.340 | 32 | 107.70 | - | NaN | offlat | 8 | NaN | - |  |
| r2xyz | x2v | 3.5,0.5,0.5 | 0.728 | -1 | NaN | - | 0.333 | 31 | 108.23 | - | 0.577 | 8/0/0 | 8 | 130.6 | - | split vs gcd: 2.96e-16 (max-entry) |
| r2xyz | x3c | 4.5,0.0,0.0 | 0.577 | 46 | 11.05 | - | 0.247 | 25 | 84.12 | - | NaN | offlat | 8 | NaN | - | whole vs split: 3.58e-16 (max-entry) |
| r2xyz | x3e | 4.5,0.5,0.0 | 0.574 | 44 | 7.84 | - | 0.245 | 25 | 78.39 | - | NaN | offlat | 8 | NaN | - | whole vs split: 1.25e-15 (max-entry) |
| r2xyz | x3v | 4.5,0.5,0.5 | 0.570 | 44 | 8.11 | - | 0.243 | 25 | 77.75 | - | 0.433 | 8/0/0 | 8 | 113.9 | - | whole vs gcd: 8.67e-16 (max-entry); split vs gcd: 6.76e-16 (max-entry); whole vs split: 3.67e-16 (max-entry) |
| r2xyz | d1c | 2.5,2.5,0.0 | 0.735 | -1 | NaN | - | 0.408 | 38 | 120.87 | - | NaN | offlat | 8 | NaN | - |  |
| r2xyz | d1e | 2.5,2.5,0.5 | 0.728 | -1 | NaN | - | 0.397 | 37 | 119.51 | - | 0.612 | 8/0/0 | 8 | 133.4 | - | split vs gcd: 6.36e-16 (max-entry) |
| r2xyz | d2c | 3.5,3.5,0.0 | 0.525 | 40 | 7.52 | - | 0.245 | 24 | 77.20 | - | NaN | offlat | 8 | NaN | - | whole vs split: 4.77e-16 (max-entry) |
| r2xyz | d2e | 3.5,3.5,0.5 | 0.522 | 40 | 6.72 | - | 0.243 | 24 | 81.46 | - | 0.408 | 8/0/0 | 8 | 115.5 | - | whole vs gcd: 2.08e-16 (max-entry); split vs gcd: 1.17e-15 (max-entry); whole vs split: 9.72e-16 (max-entry) |
| r4x | x1c | 3.5,0.0,0.0 | 0.821 | -1 | NaN | - | 0.522 | 51 | 163.54 | - | 0.866 | 3/1/0 | 4 | 175.1 | - | split vs gcd: 8.89e-16 (max-entry) |
| r4x | x2c | 4.5,0.0,0.0 | 0.638 | -1 | NaN | - | 0.364 | 36 | 75.80 | - | 0.577 | 4/0/0 | 4 | 62.6 | - | split vs gcd: 5.51e-16 (max-entry) |
| r4x | x3c | 5.5,0.0,0.0 | 0.522 | 44 | 8.25 | - | 0.299 | 30 | 55.02 | - | 0.433 | 4/0/0 | 4 | 45.8 | - | whole vs gcd: 4.84e-16 (max-entry); split vs gcd: 1.98e-15 (max-entry); whole vs split: 2.22e-15 (max-entry) |
| r4x | d1c | 3.5,2.0,0.0 | 0.713 | -1 | NaN | - | 0.432 | 43 | 91.72 | - | 0.612 | 4/0/0 | 4 | 62.4 | - | split vs gcd: 7.05e-16 (max-entry) |
| r4x | d2c | 4.5,3.0,0.0 | 0.531 | 46 | 9.03 | - | 0.321 | 32 | 50.61 | - | 0.408 | 4/0/0 | 4 | 51.8 | - | whole vs gcd: 3.39e-16 (max-entry); split vs gcd: 2.49e-16 (max-entry); whole vs split: 4.02e-16 (max-entry) |
| r4xyz | x1c | 3.5,0.0,0.0 | 1.237 | -1 | NaN | - | 1.453 | -1 | NaN | - | NaN | offlat | 64 | NaN | - |  |
| r4xyz | x1e | 3.5,1.5,0.0 | 1.137 | -1 | NaN | - | 1.049 | -1 | NaN | - | NaN | offlat | 64 | NaN | - |  |
| r4xyz | x1v | 3.5,1.5,1.5 | 1.058 | -1 | NaN | - | 0.839 | -1 | NaN | - | 0.866 | 60/4/0 | 64 | 1158.8 | - |  |
| r4xyz | x2c | 4.5,0.0,0.0 | 0.962 | -1 | NaN | - | 0.872 | -1 | NaN | - | NaN | offlat | 64 | NaN | - |  |
| r4xyz | x2e | 4.5,1.5,0.0 | 0.913 | -1 | NaN | - | 0.748 | -1 | NaN | - | NaN | offlat | 64 | NaN | - |  |
| r4xyz | x2v | 4.5,1.5,1.5 | 0.870 | -1 | NaN | - | 0.665 | -1 | NaN | - | 0.577 | 64/0/0 | 64 | 787.5 | - |  |
| r4xyz | x3c | 5.5,0.0,0.0 | 0.787 | -1 | NaN | - | 0.623 | -1 | NaN | - | NaN | offlat | 64 | NaN | - |  |
| r4xyz | x3e | 5.5,1.5,0.0 | 0.760 | -1 | NaN | - | 0.572 | -1 | NaN | - | NaN | offlat | 64 | NaN | - |  |
| r4xyz | x3v | 5.5,1.5,1.5 | 0.735 | -1 | NaN | - | 0.533 | 56 | 163.43 | - | 0.433 | 64/0/0 | 64 | 738.1 | - | split vs gcd: 3.68e-16 (max-entry) |
| r4xyz | d1c | 3.5,3.5,0.0 | 0.875 | -1 | NaN | - | 0.782 | -1 | NaN | - | NaN | offlat | 64 | NaN | - |  |
| r4xyz | d1e | 3.5,3.5,1.5 | 0.837 | -1 | NaN | - | 0.638 | -1 | NaN | - | 0.612 | 64/0/0 | 64 | 748.4 | - |  |
| r4xyz | d2c | 4.5,4.5,0.0 | 0.680 | -1 | NaN | - | 0.469 | 46 | 125.58 | - | NaN | offlat | 64 | NaN | - |  |
| r4xyz | d2e | 4.5,4.5,1.5 | 0.662 | -1 | NaN | - | 0.432 | 42 | 113.64 | - | 0.408 | 64/0/0 | 64 | 723.8 | - | split vs gcd: 4.15e-16 (max-entry) |
| r8x | x1c | 5.5,0.0,0.0 | 0.858 | -1 | NaN | - | 0.644 | -1 | NaN | - | 0.866 | 7/1/0 | 8 | 213.3 | - |  |
| r8x | x2c | 6.5,0.0,0.0 | 0.726 | -1 | NaN | - | 0.546 | -1 | NaN | - | 0.577 | 8/0/0 | 8 | 96.3 | - |  |
| r8x | x3c | 7.5,0.0,0.0 | 0.629 | -1 | NaN | - | 0.474 | 51 | 104.90 | - | 0.433 | 8/0/0 | 8 | 91.9 | - | split vs gcd: 1.94e-15 (max-entry) |
| r8x | d1c | 5.5,2.0,0.0 | 0.806 | -1 | NaN | - | 0.624 | -1 | NaN | - | 0.612 | 8/0/0 | 8 | 104.9 | - |  |
| r8x | d2c | 6.5,3.0,0.0 | 0.659 | -1 | NaN | - | 0.511 | 56 | 99.00 | - | 0.408 | 8/0/0 | 8 | 81.5 | - | split vs gcd: 1.16e-15 (max-entry) |
| r8xyz | x1c | 5.5,0.0,0.0 | 1.417 | -1 | NaN | - | 3.317 | -1 | NaN | - | NaN | offlat | 512 | NaN | - |  |
| r8xyz | x1e | 5.5,3.5,0.0 | 1.196 | -1 | NaN | - | 2.258 | -1 | NaN | - | NaN | offlat | 512 | NaN | - |  |
| r8xyz | x1v | 5.5,3.5,3.5 | 1.053 | -1 | NaN | - | 0.962 | -1 | NaN | - | 0.866 | 508/4/0 | 512 | 5759.3 | - |  |
| r8xyz | x2c | 6.5,0.0,0.0 | 1.199 | -1 | NaN | - | 1.990 | -1 | NaN | - | NaN | offlat | 512 | NaN | - |  |
| r8xyz | x2e | 6.5,3.5,0.0 | 1.056 | -1 | NaN | - | 1.401 | -1 | NaN | - | NaN | offlat | 512 | NaN | - |  |
| r8xyz | x2v | 6.5,3.5,3.5 | 0.954 | -1 | NaN | - | 0.897 | -1 | NaN | - | 0.577 | 512/0/0 | 512 | 5760.0 | - |  |
| r8xyz | x3c | 7.5,0.0,0.0 | 1.039 | -1 | NaN | - | 1.421 | -1 | NaN | - | NaN | offlat | 512 | NaN | - |  |
| r8xyz | x3e | 7.5,3.5,0.0 | 0.942 | -1 | NaN | - | 1.010 | -1 | NaN | - | NaN | offlat | 512 | NaN | - |  |
| r8xyz | x3v | 7.5,3.5,3.5 | 0.867 | -1 | NaN | - | 0.821 | -1 | NaN | - | 0.433 | 512/0/0 | 512 | 4607.2 | - |  |
| r8xyz | d1c | 5.5,5.5,0.0 | 1.002 | -1 | NaN | - | 1.683 | -1 | NaN | - | NaN | offlat | 512 | NaN | - |  |
| r8xyz | d1e | 5.5,5.5,3.5 | 0.914 | -1 | NaN | - | 0.872 | -1 | NaN | - | 0.612 | 512/0/0 | 512 | 6246.4 | - |  |
| r8xyz | d2c | 6.5,6.5,0.0 | 0.848 | -1 | NaN | - | 1.010 | -1 | NaN | - | NaN | offlat | 512 | NaN | - |  |
| r8xyz | d2e | 6.5,6.5,3.5 | 0.792 | -1 | NaN | - | 0.718 | -1 | NaN | - | 0.408 | 512/0/0 | 512 | 5012.5 | - |  |
| r16x | x1c | 9.5,0.0,0.0 | 0.907 | -1 | NaN | - | 0.791 | -1 | NaN | - | 0.866 | 15/1/0 | 16 | 298.0 | - |  |
| r16x | x2c | 10.5,0.0,0.0 | 0.821 | -1 | NaN | - | 0.716 | -1 | NaN | - | 0.577 | 16/0/0 | 16 | 166.4 | - |  |
| r16x | x3c | 11.5,0.0,0.0 | 0.749 | -1 | NaN | - | 0.654 | -1 | NaN | - | 0.433 | 16/0/0 | 16 | 164.1 | - |  |
| r16xyz | x1c | 9.5,0.0,0.0 | 1.550 | -1 | NaN | - | 7.079 | -1 | NaN | - | NaN | offlat | 4096 | NaN | - |  |
| r16xyz | x1e | 9.5,7.5,0.0 | 1.216 | -1 | NaN | - | 4.764 | -1 | NaN | - | NaN | offlat | 4096 | NaN | - |  |
| r16xyz | x1v | 9.5,7.5,7.5 | 1.034 | -1 | NaN | - | 0.991 | -1 | NaN | - | 0.866 | 4092/4/0 | 4096 | 40161.2 | - |  |
| r16xyz | x2c | 10.5,0.0,0.0 | 1.402 | -1 | NaN | - | 4.247 | -1 | NaN | - | NaN | offlat | 4096 | NaN | - |  |
| r16xyz | x2e | 10.5,7.5,0.0 | 1.141 | -1 | NaN | - | 2.955 | -1 | NaN | - | NaN | offlat | 4096 | NaN | - |  |
| r16xyz | x2v | 10.5,7.5,7.5 | 0.986 | -1 | NaN | - | 0.974 | -1 | NaN | - | 0.577 | 4096/0/0 | 4096 | 43057.7 | - |  |
| r16xyz | x3c | 11.5,0.0,0.0 | 1.280 | -1 | NaN | - | 3.034 | -1 | NaN | - | NaN | offlat | 4096 | NaN | - |  |
| r16xyz | x3e | 11.5,7.5,0.0 | 1.072 | -1 | NaN | - | 2.131 | -1 | NaN | - | NaN | offlat | 4096 | NaN | - |  |
| r16xyz | x3v | 11.5,7.5,7.5 | 0.941 | -1 | NaN | - | 0.951 | -1 | NaN | - | 0.433 | 4096/0/0 | 4096 | 39086.3 | - |  |

### Far offsets f index 2: kap >= 4 (whole-box route), times and agreement

| shape | off | R/g | rho_w | L_w | t_w | L_p | t_p | t_gcd | agreement |
|---|---|---|---|---|---|---|---|---|---|
| r2x | x4c | 5.5,0.0,0.0 | 0.375 | 28 | 3.06 | 21 | 37.90 | 31.5 | whole vs gcd: 4.11e-16 (max-entry); split vs gcd: 1.12e-15 (max-entry); whole vs split: 1.45e-15 (max-entry) |
| r2x | x6c | 7.5,0.0,0.0 | 0.275 | 22 | 2.11 | 17 | 28.55 | 24.2 | whole vs gcd: 1.99e-16 (max-entry); split vs gcd: 3.97e-16 (max-entry); whole vs split: 3.55e-16 (max-entry) |
| r2x | x8c | 9.5,0.0,0.0 | 0.217 | 20 | 1.66 | 16 | 26.75 | 27.2 | whole vs gcd: 2.88e-16 (max-entry); split vs gcd: 3.30e-16 (max-entry); whole vs split: 4.00e-16 (max-entry) |
| r2x | x16c | 17.5,0.0,0.0 | 0.118 | 16 | 1.19 | 13 | 19.21 | 33.1 | whole vs gcd: 6.97e-16 (max-entry); split vs gcd: 5.62e-16 (max-entry); whole vs split: 6.16e-16 (max-entry) |
| r2x | x32c | 33.5,0.0,0.0 | 0.062 | 12 | 0.77 | 11 | 14.95 | 27.5 | whole vs gcd: 5.91e-16 (max-entry); split vs gcd: 4.36e-16 (max-entry); whole vs split: 4.36e-16 (max-entry) |
| r2x | d4c | 5.5,5.0,0.0 | 0.277 | 22 | 2.19 | 18 | 29.18 | 24.1 | whole vs gcd: 6.20e-16 (max-entry); split vs gcd: 8.01e-16 (max-entry); whole vs split: 1.18e-15 (max-entry) |
| r2x | d8c | 9.5,9.0,0.0 | 0.158 | 16 | 1.20 | 14 | 20.07 | 31.4 | whole vs gcd: 4.45e-16 (max-entry); split vs gcd: 3.33e-16 (max-entry); whole vs split: 5.34e-16 (max-entry) |
| r2x | d16c | 17.5,17.0,0.0 | 0.084 | 14 | 1.01 | 12 | 16.50 | 24.8 | whole vs gcd: 3.90e-16 (max-entry); split vs gcd: 3.73e-16 (max-entry); whole vs split: 2.64e-16 (max-entry) |
| r2xyz | x4c | 5.5,0.0,0.0 | 0.472 | 34 | 5.02 | 21 | 67.50 | NaN | whole vs split: 3.44e-16 (max-entry) |
| r2xyz | x4e | 5.5,0.5,0.0 | 0.470 | 34 | 4.67 | 21 | 65.58 | NaN | whole vs split: 3.32e-16 (max-entry) |
| r2xyz | x4v | 5.5,0.5,0.5 | 0.469 | 34 | 4.82 | 21 | 79.26 | 96.8 | whole vs gcd: 7.11e-16 (max-entry); split vs gcd: 2.62e-16 (max-entry); whole vs split: 6.08e-16 (max-entry) |
| r2xyz | x6c | 7.5,0.0,0.0 | 0.346 | 26 | 2.50 | 18 | 50.58 | NaN | whole vs split: 6.10e-16 (max-entry) |
| r2xyz | x6e | 7.5,0.5,0.0 | 0.346 | 26 | 2.79 | 18 | 59.12 | NaN | whole vs split: 6.62e-16 (max-entry) |
| r2xyz | x6v | 7.5,0.5,0.5 | 0.345 | 26 | 2.86 | 18 | 52.61 | 100.5 | whole vs gcd: 1.39e-16 (max-entry); split vs gcd: 2.08e-16 (max-entry); whole vs split: 2.97e-16 (max-entry) |
| r2xyz | x8c | 9.5,0.0,0.0 | 0.273 | 22 | 2.16 | 16 | 54.43 | NaN | whole vs split: 4.58e-16 (max-entry) |
| r2xyz | x8e | 9.5,0.5,0.0 | 0.273 | 22 | 2.17 | 16 | 47.65 | NaN | whole vs split: 4.76e-16 (max-entry) |
| r2xyz | x8v | 9.5,0.5,0.5 | 0.273 | 22 | 2.19 | 16 | 49.37 | 95.4 | whole vs gcd: 8.26e-16 (max-entry); split vs gcd: 5.51e-16 (max-entry); whole vs split: 8.26e-16 (max-entry) |
| r2xyz | x16c | 17.5,0.0,0.0 | 0.148 | 16 | 1.21 | 13 | 49.05 | NaN | whole vs split: 5.95e-16 (max-entry) |
| r2xyz | x16e | 17.5,0.5,0.0 | 0.148 | 16 | 1.29 | 13 | 36.26 | NaN | whole vs split: 4.94e-16 (max-entry) |
| r2xyz | x16v | 17.5,0.5,0.5 | 0.148 | 16 | 1.19 | 13 | 34.85 | 88.5 | whole vs gcd: 6.73e-16 (max-entry); split vs gcd: 2.48e-16 (max-entry); whole vs split: 4.43e-16 (max-entry) |
| r2xyz | x32c | 33.5,0.0,0.0 | 0.078 | 14 | 1.12 | 11 | 28.67 | NaN | whole vs split: 4.05e-16 (max-entry) |
| r2xyz | x32e | 33.5,0.5,0.0 | 0.078 | 14 | 0.99 | 11 | 27.41 | NaN | whole vs split: 2.58e-16 (max-entry) |
| r2xyz | x32v | 33.5,0.5,0.5 | 0.078 | 14 | 1.01 | 11 | 27.22 | 83.4 | whole vs gcd: 2.59e-16 (max-entry); split vs gcd: 2.41e-16 (max-entry); whole vs split: 3.27e-16 (max-entry) |
| r2xyz | d4c | 5.5,5.5,0.0 | 0.334 | 26 | 2.83 | 18 | 56.06 | NaN | whole vs split: 2.20e-16 (max-entry) |
| r2xyz | d4e | 5.5,5.5,0.5 | 0.333 | 26 | 2.79 | 18 | 56.77 | 81.8 | whole vs gcd: 4.68e-16 (max-entry); split vs gcd: 4.44e-16 (max-entry); whole vs split: 2.96e-16 (max-entry) |
| r2xyz | d8c | 9.5,9.5,0.0 | 0.193 | 18 | 1.57 | 14 | 39.03 | NaN | whole vs split: 8.07e-16 (max-entry) |
| r2xyz | d8e | 9.5,9.5,0.5 | 0.193 | 18 | 1.57 | 14 | 41.21 | 94.8 | whole vs gcd: 3.27e-16 (max-entry); split vs gcd: 7.93e-17 (max-entry); whole vs split: 3.39e-16 (max-entry) |
| r2xyz | d16c | 17.5,17.5,0.0 | 0.105 | 14 | 1.07 | 12 | 30.86 | NaN | whole vs split: 3.60e-16 (max-entry) |
| r2xyz | d16e | 17.5,17.5,0.5 | 0.105 | 14 | 0.96 | 12 | 33.42 | 72.1 | whole vs gcd: 2.69e-16 (max-entry); split vs gcd: 2.69e-16 (max-entry); whole vs split: 2.00e-16 (max-entry) |
| r4x | x4c | 6.5,0.0,0.0 | 0.442 | 36 | 8.07 | 27 | 47.97 | 49.7 | whole vs gcd: 6.65e-16 (max-entry); split vs gcd: 1.15e-15 (max-entry); whole vs split: 6.19e-16 (max-entry) |
| r4x | x6c | 8.5,0.0,0.0 | 0.338 | 28 | 3.22 | 23 | 29.80 | 49.5 | whole vs gcd: 5.88e-16 (max-entry); split vs gcd: 7.04e-16 (max-entry); whole vs split: 7.14e-16 (max-entry) |
| r4x | x8c | 10.5,0.0,0.0 | 0.274 | 24 | 2.52 | 20 | 27.81 | 44.8 | whole vs gcd: 5.99e-16 (max-entry); split vs gcd: 1.01e-15 (max-entry); whole vs split: 4.45e-16 (max-entry) |
| r4x | x16c | 18.5,0.0,0.0 | 0.155 | 18 | 1.39 | 16 | 20.74 | 45.0 | whole vs gcd: 7.01e-16 (max-entry); split vs gcd: 9.69e-16 (max-entry); whole vs split: 1.58e-15 (max-entry) |
| r4x | x32c | 34.5,0.0,0.0 | 0.083 | 14 | 0.95 | 14 | 16.43 | 38.2 | whole vs gcd: 3.42e-16 (max-entry); split vs gcd: 3.42e-16 (max-entry); whole vs split: 6.17e-16 (max-entry) |
| r4x | d4c | 6.5,5.0,0.0 | 0.350 | 30 | 3.84 | 24 | 34.49 | 49.5 | whole vs gcd: 9.87e-16 (max-entry); split vs gcd: 7.44e-16 (max-entry); whole vs split: 9.91e-16 (max-entry) |
| r4x | d8c | 10.5,9.0,0.0 | 0.208 | 20 | 1.73 | 18 | 23.74 | 42.6 | whole vs gcd: 4.42e-16 (max-entry); split vs gcd: 4.19e-16 (max-entry); whole vs split: 6.72e-16 (max-entry) |
| r4x | d16c | 18.5,17.0,0.0 | 0.114 | 16 | 1.18 | 15 | 17.52 | 40.5 | whole vs gcd: 1.96e-16 (max-entry); split vs gcd: 3.68e-16 (max-entry); whole vs split: 5.52e-16 (max-entry) |
| r4xyz | x4c | 6.5,0.0,0.0 | 0.666 | -1 | NaN | 49 | 111.67 | NaN |  |
| r4xyz | x4e | 6.5,1.5,0.0 | 0.649 | -1 | NaN | 45 | 109.86 | NaN |  |
| r4xyz | x4v | 6.5,1.5,1.5 | 0.633 | 54 | 12.18 | 43 | 98.75 | 594.6 | whole vs gcd: 1.13e-15 (max-entry); split vs gcd: 1.06e-15 (max-entry); whole vs split: 6.77e-16 (max-entry) |
| r4xyz | x6c | 8.5,0.0,0.0 | 0.509 | 40 | 6.88 | 33 | 93.77 | NaN | whole vs split: 5.11e-16 (max-entry) |
| r4xyz | x6e | 8.5,1.5,0.0 | 0.502 | 38 | 6.19 | 32 | 79.30 | NaN | whole vs split: 5.49e-16 (max-entry) |
| r4xyz | x6v | 8.5,1.5,1.5 | 0.494 | 38 | 5.82 | 31 | 77.71 | 682.3 | whole vs gcd: 1.42e-15 (max-entry); split vs gcd: 7.20e-16 (max-entry); whole vs split: 7.32e-16 (max-entry) |
| r4xyz | x8c | 10.5,0.0,0.0 | 0.412 | 32 | 4.07 | 26 | 70.81 | NaN | whole vs split: 5.38e-16 (max-entry) |
| r4xyz | x8e | 10.5,1.5,0.0 | 0.408 | 32 | 3.97 | 26 | 67.38 | NaN | whole vs split: 1.96e-15 (max-entry) |
| r4xyz | x8v | 10.5,1.5,1.5 | 0.404 | 30 | 3.57 | 26 | 67.07 | 547.9 | whole vs gcd: 1.34e-15 (max-entry); split vs gcd: 3.48e-16 (max-entry); whole vs split: 1.17e-15 (max-entry) |
| r4xyz | x16c | 18.5,0.0,0.0 | 0.234 | 20 | 1.72 | 19 | 42.58 | NaN | whole vs split: 6.06e-16 (max-entry) |
| r4xyz | x16e | 18.5,1.5,0.0 | 0.233 | 20 | 1.69 | 19 | 39.49 | NaN | whole vs split: 4.39e-16 (max-entry) |
| r4xyz | x16v | 18.5,1.5,1.5 | 0.233 | 20 | 1.77 | 19 | 43.48 | 632.6 | whole vs gcd: 3.89e-16 (max-entry); split vs gcd: 3.89e-16 (max-entry); whole vs split: 4.93e-16 (max-entry) |
| r4xyz | x32c | 34.5,0.0,0.0 | 0.126 | 16 | 1.22 | 16 | 33.34 | NaN | whole vs split: 3.16e-16 (max-entry) |
| r4xyz | x32e | 34.5,1.5,0.0 | 0.125 | 16 | 1.15 | 16 | 33.36 | NaN | whole vs split: 2.30e-16 (max-entry) |
| r4xyz | x32v | 34.5,1.5,1.5 | 0.125 | 16 | 1.15 | 16 | 32.97 | 628.7 | whole vs gcd: 4.15e-16 (max-entry); split vs gcd: 3.18e-16 (max-entry); whole vs split: 4.88e-16 (max-entry) |
| r4xyz | d4c | 6.5,6.5,0.0 | 0.471 | 36 | 5.45 | 29 | 78.32 | NaN | whole vs split: 6.27e-16 (max-entry) |
| r4xyz | d4e | 6.5,6.5,1.5 | 0.465 | 36 | 5.87 | 28 | 74.38 | 572.8 | whole vs gcd: 5.43e-16 (max-entry); split vs gcd: 3.91e-16 (max-entry); whole vs split: 6.32e-16 (max-entry) |
| r4xyz | d8c | 10.5,10.5,0.0 | 0.292 | 24 | 2.34 | 21 | 52.89 | NaN | whole vs split: 3.98e-16 (max-entry) |
| r4xyz | d8e | 10.5,10.5,1.5 | 0.290 | 24 | 2.35 | 21 | 51.83 | 575.2 | whole vs gcd: 2.63e-16 (max-entry); split vs gcd: 2.08e-16 (max-entry); whole vs split: 3.83e-16 (max-entry) |
| r4xyz | d16c | 18.5,18.5,0.0 | 0.166 | 18 | 1.48 | 17 | 36.58 | NaN | whole vs split: 3.99e-16 (max-entry) |
| r4xyz | d16e | 18.5,18.5,1.5 | 0.165 | 18 | 1.43 | 17 | 34.43 | 685.6 | whole vs gcd: 4.45e-16 (max-entry); split vs gcd: 6.06e-16 (max-entry); whole vs split: 4.22e-16 (max-entry) |
| r8x | x4c | 8.5,0.0,0.0 | 0.555 | 54 | 11.98 | 44 | 72.96 | 73.4 | whole vs gcd: 8.98e-16 (max-entry); split vs gcd: 9.47e-16 (max-entry); whole vs split: 4.22e-16 (max-entry) |
| r8x | x6c | 10.5,0.0,0.0 | 0.449 | 42 | 7.12 | 35 | 49.77 | 82.4 | whole vs gcd: 1.28e-15 (max-entry); split vs gcd: 6.67e-16 (max-entry); whole vs split: 1.77e-15 (max-entry) |
| r8x | x8c | 12.5,0.0,0.0 | 0.377 | 34 | 4.94 | 31 | 39.21 | 79.4 | whole vs gcd: 3.62e-16 (max-entry); split vs gcd: 1.25e-15 (max-entry); whole vs split: 1.46e-15 (max-entry) |
| r8x | x16c | 20.5,0.0,0.0 | 0.230 | 24 | 2.29 | 23 | 25.00 | 72.1 | whole vs gcd: 4.61e-16 (max-entry); split vs gcd: 1.63e-15 (max-entry); whole vs split: 2.09e-15 (max-entry) |
| r8x | x32c | 36.5,0.0,0.0 | 0.129 | 18 | 1.38 | 18 | 18.37 | 87.7 | whole vs gcd: 4.03e-16 (max-entry); split vs gcd: 7.44e-16 (max-entry); whole vs split: 4.51e-16 (max-entry) |
| r8x | d4c | 8.5,5.0,0.0 | 0.478 | 44 | 7.77 | 38 | 51.57 | 93.2 | whole vs gcd: 7.48e-16 (max-entry); split vs gcd: 1.28e-15 (max-entry); whole vs split: 7.53e-16 (max-entry) |
| r8x | d8c | 12.5,9.0,0.0 | 0.306 | 28 | 3.43 | 27 | 29.24 | 92.9 | whole vs gcd: 3.98e-16 (max-entry); split vs gcd: 3.09e-16 (max-entry); whole vs split: 4.93e-16 (max-entry) |
| r8x | d16c | 20.5,17.0,0.0 | 0.177 | 20 | 1.69 | 20 | 21.23 | 87.4 | whole vs gcd: 3.50e-16 (max-entry); split vs gcd: 3.76e-16 (max-entry); whole vs split: 4.67e-16 (max-entry) |
| r8xyz | x4c | 8.5,0.0,0.0 | 0.917 | -1 | NaN | -1 | NaN | NaN |  |
| r8xyz | x4e | 8.5,3.5,0.0 | 0.848 | -1 | NaN | -1 | NaN | NaN |  |
| r8xyz | x4v | 8.5,3.5,3.5 | 0.792 | -1 | NaN | -1 | NaN | 4245.9 |  |
| r8xyz | x6c | 10.5,0.0,0.0 | 0.742 | -1 | NaN | -1 | NaN | NaN |  |
| r8xyz | x6e | 10.5,3.5,0.0 | 0.704 | -1 | NaN | -1 | NaN | NaN |  |
| r8xyz | x6v | 10.5,3.5,3.5 | 0.671 | -1 | NaN | -1 | NaN | 4294.2 |  |
| r8xyz | x8c | 12.5,0.0,0.0 | 0.624 | 56 | 13.10 | -1 | NaN | NaN |  |
| r8xyz | x8e | 12.5,3.5,0.0 | 0.600 | 54 | 11.97 | -1 | NaN | NaN |  |
| r8xyz | x8v | 12.5,3.5,3.5 | 0.580 | 50 | 10.26 | 52 | 122.78 | 5210.4 | whole vs gcd: 9.60e-13 (max-entry); split vs gcd: 9.26e-13 (max-entry); whole vs split: 6.27e-13 (max-entry) |
| r8xyz | x16c | 20.5,0.0,0.0 | 0.380 | 32 | 4.06 | 31 | 99.92 | NaN | whole vs split: 2.71e-16 (max-entry) |
| r8xyz | x16e | 20.5,3.5,0.0 | 0.375 | 30 | 3.78 | 30 | 73.94 | NaN | whole vs split: 1.40e-15 (max-entry) |
| r8xyz | x16v | 20.5,3.5,3.5 | 0.370 | 30 | 3.68 | 30 | 74.82 | 4908.2 | whole vs gcd: 5.48e-16 (max-entry); split vs gcd: 1.14e-15 (max-entry); whole vs split: 8.54e-16 (max-entry) |
| r8xyz | x32c | 36.5,0.0,0.0 | 0.214 | 22 | 2.23 | 23 | 52.57 | NaN | whole vs split: 5.54e-16 (max-entry) |
| r8xyz | x32e | 36.5,3.5,0.0 | 0.213 | 22 | 2.36 | 22 | 53.01 | NaN | whole vs split: 7.41e-16 (max-entry) |
| r8xyz | x32v | 36.5,3.5,3.5 | 0.212 | 22 | 2.11 | 22 | 44.73 | 5579.7 | whole vs gcd: 4.18e-16 (max-entry); split vs gcd: 5.91e-16 (max-entry); whole vs split: 3.74e-16 (max-entry) |
| r8xyz | d4c | 8.5,8.5,0.0 | 0.648 | -1 | NaN | -1 | NaN | NaN |  |
| r8xyz | d4e | 8.5,8.5,3.5 | 0.623 | 56 | 13.85 | 50 | 152.47 | 4899.7 | whole vs gcd: 3.94e-12 (max-entry); split vs gcd: 5.01e-13 (max-entry); whole vs split: 3.80e-12 (max-entry) |
| r8xyz | d8c | 12.5,12.5,0.0 | 0.441 | 36 | 5.34 | 35 | 82.61 | NaN | whole vs split: 6.99e-16 (max-entry) |
| r8xyz | d8e | 12.5,12.5,3.5 | 0.433 | 36 | 5.57 | 34 | 79.80 | 4866.9 | whole vs gcd: 1.43e-15 (max-entry); split vs gcd: 8.45e-16 (max-entry); whole vs split: 6.89e-16 (max-entry) |
| r8xyz | d16c | 20.5,20.5,0.0 | 0.269 | 24 | 2.42 | 25 | 52.76 | NaN | whole vs split: 2.58e-16 (max-entry) |
| r8xyz | d16e | 20.5,20.5,3.5 | 0.267 | 24 | 2.59 | 25 | 54.00 | 5162.6 | whole vs gcd: 7.84e-16 (max-entry); split vs gcd: 7.95e-16 (max-entry); whole vs split: 3.21e-16 (max-entry) |
| r16x | x4c | 12.5,0.0,0.0 | 0.689 | -1 | NaN | -1 | NaN | 170.3 |  |
| r16x | x6c | 14.5,0.0,0.0 | 0.594 | -1 | NaN | -1 | NaN | 150.3 |  |
| r16xyz | x4c | 12.5,0.0,0.0 | 1.178 | -1 | NaN | -1 | NaN | NaN |  |
| r16xyz | x4e | 12.5,7.5,0.0 | 1.010 | -1 | NaN | -1 | NaN | NaN |  |
| r16xyz | x4v | 12.5,7.5,7.5 | 0.898 | -1 | NaN | -1 | NaN | 33441.7 |  |
| r16xyz | x6c | 14.5,0.0,0.0 | 1.015 | -1 | NaN | -1 | NaN | NaN |  |
| r16xyz | x6e | 14.5,7.5,0.0 | 0.902 | -1 | NaN | -1 | NaN | NaN |  |
| r16xyz | x6v | 14.5,7.5,7.5 | 0.819 | -1 | NaN | -1 | NaN | 34019.9 |  |
| r16xyz | x8c | 16.5,0.0,0.0 | 0.892 | -1 | NaN | -1 | NaN | NaN |  |
| r16xyz | x8e | 16.5,7.5,0.0 | 0.812 | -1 | NaN | -1 | NaN | NaN |  |
| r16xyz | x8v | 16.5,7.5,7.5 | 0.751 | -1 | NaN | -1 | NaN | 33810.5 |  |
| r16xyz | x16c | 24.5,0.0,0.0 | 0.601 | -1 | NaN | -1 | NaN | NaN |  |
| r16xyz | x16e | 24.5,7.5,0.0 | 0.575 | 54 | 12.37 | -1 | NaN | NaN |  |
| r16xyz | x16v | 24.5,7.5,7.5 | 0.551 | 52 | 10.74 | -1 | NaN | 31846.5 | whole vs gcd: 1.12e-09 (max-entry) |
### Near band f index 3: kap = 1, 2, 3 (x family) and the diagonal kap = 1, 2; times in us at the stated load

| shape | off | R/g | rho_w | L_w | t_w | err_w (max,ent) | rho_p | L_p | t_p | err_p | rho_gcd | sub-routes 1/2/3 | N_s | t_gcd | err_gcd | cross-route agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r2x | x1c | 2.5,0.0,0.0 | 0.825 | -1 | NaN | - | 0.522 | 51 | 120.57 | - | 0.866 | 1/1/0 | 2 | 142.8 | - | split vs gcd: 3.71e-15 (max-entry) |
| r2x | x2c | 3.5,0.0,0.0 | 0.589 | 48 | 9.56 | - | 0.333 | 31 | 60.29 | - | 0.577 | 2/0/0 | 2 | 37.6 | - | whole vs gcd: 3.32e-16 (max-entry); split vs gcd: 8.85e-16 (max-entry); whole vs split: 6.64e-16 (max-entry) |
| r2x | x3c | 4.5,0.0,0.0 | 0.458 | 34 | 4.76 | - | 0.243 | 24 | 41.51 | - | 0.433 | 2/0/0 | 2 | 31.8 | - | whole vs gcd: 6.63e-16 (max-entry); split vs gcd: 4.83e-16 (max-entry); whole vs split: 7.24e-16 (max-entry) |
| r2x | d1c | 2.5,2.0,0.0 | 0.644 | 56 | 12.43 | - | 0.397 | 36 | 65.00 | - | 0.612 | 2/0/0 | 2 | 32.9 | - | whole vs gcd: 7.40e-16 (max-entry); split vs gcd: 1.11e-15 (max-entry); whole vs split: 8.63e-16 (max-entry) |
| r2x | d2c | 3.5,3.0,0.0 | 0.447 | 34 | 4.75 | - | 0.243 | 24 | 37.77 | - | 0.408 | 2/0/0 | 2 | 32.7 | - | whole vs gcd: 5.64e-16 (max-entry); split vs gcd: 5.64e-16 (max-entry); whole vs split: 2.82e-16 (max-entry) |
| r2xyz | x1c | 2.5,0.0,0.0 | 1.039 | -1 | NaN | - | 0.577 | -1 | NaN | - | NaN | offlat | 8 | NaN | - |  |
| r2xyz | x1e | 2.5,0.5,0.0 | 1.019 | -1 | NaN | - | 0.548 | 56 | 177.22 | - | NaN | offlat | 8 | NaN | - |  |
| r2xyz | x1v | 2.5,0.5,0.5 | 1.000 | -1 | NaN | - | 0.522 | 52 | 171.98 | - | 0.866 | 4/4/0 | 8 | 501.4 | - | split vs gcd: 1.43e-15 (max-entry) |
| r2xyz | x2c | 3.5,0.0,0.0 | 0.742 | -1 | NaN | - | 0.346 | 32 | 99.99 | - | NaN | offlat | 8 | NaN | - |  |
| r2xyz | x2e | 3.5,0.5,0.0 | 0.735 | -1 | NaN | - | 0.340 | 32 | 98.50 | - | NaN | offlat | 8 | NaN | - |  |
| r2xyz | x2v | 3.5,0.5,0.5 | 0.728 | -1 | NaN | - | 0.333 | 31 | 95.44 | - | 0.577 | 8/0/0 | 8 | 113.8 | - | split vs gcd: 2.13e-16 (max-entry) |
| r2xyz | x3c | 4.5,0.0,0.0 | 0.577 | 44 | 7.61 | - | 0.247 | 25 | 74.80 | - | NaN | offlat | 8 | NaN | - | whole vs split: 4.57e-16 (max-entry) |
| r2xyz | x3e | 4.5,0.5,0.0 | 0.574 | 44 | 7.86 | - | 0.245 | 25 | 74.84 | - | NaN | offlat | 8 | NaN | - | whole vs split: 4.73e-16 (max-entry) |
| r2xyz | x3v | 4.5,0.5,0.5 | 0.570 | 44 | 8.05 | - | 0.243 | 25 | 75.91 | - | 0.433 | 8/0/0 | 8 | 99.0 | - | whole vs gcd: 4.19e-16 (max-entry); split vs gcd: 2.80e-16 (max-entry); whole vs split: 6.29e-16 (max-entry) |
| r2xyz | d1c | 2.5,2.5,0.0 | 0.735 | -1 | NaN | - | 0.408 | 38 | 93.50 | - | NaN | offlat | 8 | NaN | - |  |
| r2xyz | d1e | 2.5,2.5,0.5 | 0.728 | -1 | NaN | - | 0.397 | 37 | 99.22 | - | 0.612 | 8/0/0 | 8 | 120.6 | - | split vs gcd: 6.34e-16 (max-entry) |
| r2xyz | d2c | 3.5,3.5,0.0 | 0.525 | 40 | 6.05 | - | 0.245 | 24 | 63.92 | - | NaN | offlat | 8 | NaN | - | whole vs split: 3.54e-16 (max-entry) |
| r2xyz | d2e | 3.5,3.5,0.5 | 0.522 | 38 | 5.26 | - | 0.243 | 24 | 63.02 | - | 0.408 | 8/0/0 | 8 | 91.5 | - | whole vs gcd: 3.63e-16 (max-entry); split vs gcd: 3.63e-16 (max-entry); whole vs split: 2.11e-16 (max-entry) |
| r4x | x1c | 3.5,0.0,0.0 | 0.821 | -1 | NaN | - | 0.522 | 51 | 140.08 | - | 0.866 | 3/1/0 | 4 | 149.7 | - | split vs gcd: 2.31e-15 (max-entry) |
| r4x | x2c | 4.5,0.0,0.0 | 0.638 | -1 | NaN | - | 0.364 | 36 | 72.48 | - | 0.577 | 4/0/0 | 4 | 66.9 | - | split vs gcd: 1.77e-15 (max-entry) |
| r4x | x3c | 5.5,0.0,0.0 | 0.522 | 44 | 8.09 | - | 0.299 | 30 | 51.05 | - | 0.433 | 4/0/0 | 4 | 46.6 | - | whole vs gcd: 7.22e-16 (max-entry); split vs gcd: 4.51e-16 (max-entry); whole vs split: 1.08e-15 (max-entry) |
| r4x | d1c | 3.5,2.0,0.0 | 0.713 | -1 | NaN | - | 0.432 | 43 | 77.67 | - | 0.612 | 4/0/0 | 4 | 61.3 | - | split vs gcd: 2.04e-15 (max-entry) |
| r4x | d2c | 4.5,3.0,0.0 | 0.531 | 46 | 8.91 | - | 0.321 | 32 | 49.42 | - | 0.408 | 4/0/0 | 4 | 51.1 | - | whole vs gcd: 1.12e-15 (max-entry); split vs gcd: 6.98e-16 (max-entry); whole vs split: 1.12e-15 (max-entry) |
| r4xyz | x1c | 3.5,0.0,0.0 | 1.237 | -1 | NaN | - | 1.453 | -1 | NaN | - | NaN | offlat | 64 | NaN | - |  |
| r4xyz | x1e | 3.5,1.5,0.0 | 1.137 | -1 | NaN | - | 1.049 | -1 | NaN | - | NaN | offlat | 64 | NaN | - |  |
| r4xyz | x1v | 3.5,1.5,1.5 | 1.058 | -1 | NaN | - | 0.839 | -1 | NaN | - | 0.866 | 60/4/0 | 64 | 989.8 | - |  |
| r4xyz | x2c | 4.5,0.0,0.0 | 0.962 | -1 | NaN | - | 0.872 | -1 | NaN | - | NaN | offlat | 64 | NaN | - |  |
| r4xyz | x2e | 4.5,1.5,0.0 | 0.913 | -1 | NaN | - | 0.748 | -1 | NaN | - | NaN | offlat | 64 | NaN | - |  |
| r4xyz | x2v | 4.5,1.5,1.5 | 0.870 | -1 | NaN | - | 0.665 | -1 | NaN | - | 0.577 | 64/0/0 | 64 | 735.7 | - |  |
| r4xyz | x3c | 5.5,0.0,0.0 | 0.787 | -1 | NaN | - | 0.623 | -1 | NaN | - | NaN | offlat | 64 | NaN | - |  |
| r4xyz | x3e | 5.5,1.5,0.0 | 0.760 | -1 | NaN | - | 0.572 | -1 | NaN | - | NaN | offlat | 64 | NaN | - |  |
| r4xyz | x3v | 5.5,1.5,1.5 | 0.735 | -1 | NaN | - | 0.533 | 56 | 129.98 | - | 0.433 | 64/0/0 | 64 | 725.2 | - | split vs gcd: 3.46e-16 (max-entry) |
| r4xyz | d1c | 3.5,3.5,0.0 | 0.875 | -1 | NaN | - | 0.782 | -1 | NaN | - | NaN | offlat | 64 | NaN | - |  |
| r4xyz | d1e | 3.5,3.5,1.5 | 0.837 | -1 | NaN | - | 0.638 | -1 | NaN | - | 0.612 | 64/0/0 | 64 | 777.6 | - |  |
| r4xyz | d2c | 4.5,4.5,0.0 | 0.680 | -1 | NaN | - | 0.469 | 46 | 109.15 | - | NaN | offlat | 64 | NaN | - |  |
| r4xyz | d2e | 4.5,4.5,1.5 | 0.662 | -1 | NaN | - | 0.432 | 42 | 101.97 | - | 0.408 | 64/0/0 | 64 | 670.9 | - | split vs gcd: 2.78e-16 (max-entry) |
| r8x | x1c | 5.5,0.0,0.0 | 0.858 | -1 | NaN | - | 0.644 | -1 | NaN | - | 0.866 | 7/1/0 | 8 | 215.6 | - |  |
| r8x | x2c | 6.5,0.0,0.0 | 0.726 | -1 | NaN | - | 0.546 | -1 | NaN | - | 0.577 | 8/0/0 | 8 | 109.7 | - |  |
| r8x | x3c | 7.5,0.0,0.0 | 0.629 | -1 | NaN | - | 0.474 | 50 | 82.16 | - | 0.433 | 8/0/0 | 8 | 90.8 | - | split vs gcd: 1.83e-15 (max-entry) |
| r8x | d1c | 5.5,2.0,0.0 | 0.806 | -1 | NaN | - | 0.624 | -1 | NaN | - | 0.612 | 8/0/0 | 8 | 89.0 | - |  |
| r8x | d2c | 6.5,3.0,0.0 | 0.659 | -1 | NaN | - | 0.511 | 56 | 91.43 | - | 0.408 | 8/0/0 | 8 | 86.5 | - | split vs gcd: 1.10e-15 (max-entry) |
| r8xyz | x1c | 5.5,0.0,0.0 | 1.417 | -1 | NaN | - | 3.317 | -1 | NaN | - | NaN | offlat | 512 | NaN | - |  |
| r8xyz | x1e | 5.5,3.5,0.0 | 1.196 | -1 | NaN | - | 2.258 | -1 | NaN | - | NaN | offlat | 512 | NaN | - |  |
| r8xyz | x1v | 5.5,3.5,3.5 | 1.053 | -1 | NaN | - | 0.962 | -1 | NaN | - | 0.866 | 508/4/0 | 512 | 6158.8 | - |  |
| r8xyz | x2c | 6.5,0.0,0.0 | 1.199 | -1 | NaN | - | 1.990 | -1 | NaN | - | NaN | offlat | 512 | NaN | - |  |
| r8xyz | x2e | 6.5,3.5,0.0 | 1.056 | -1 | NaN | - | 1.401 | -1 | NaN | - | NaN | offlat | 512 | NaN | - |  |
| r8xyz | x2v | 6.5,3.5,3.5 | 0.954 | -1 | NaN | - | 0.897 | -1 | NaN | - | 0.577 | 512/0/0 | 512 | 4895.6 | - |  |
| r8xyz | x3c | 7.5,0.0,0.0 | 1.039 | -1 | NaN | - | 1.421 | -1 | NaN | - | NaN | offlat | 512 | NaN | - |  |
| r8xyz | x3e | 7.5,3.5,0.0 | 0.942 | -1 | NaN | - | 1.010 | -1 | NaN | - | NaN | offlat | 512 | NaN | - |  |
| r8xyz | x3v | 7.5,3.5,3.5 | 0.867 | -1 | NaN | - | 0.821 | -1 | NaN | - | 0.433 | 512/0/0 | 512 | 5639.4 | - |  |
| r8xyz | d1c | 5.5,5.5,0.0 | 1.002 | -1 | NaN | - | 1.683 | -1 | NaN | - | NaN | offlat | 512 | NaN | - |  |
| r8xyz | d1e | 5.5,5.5,3.5 | 0.914 | -1 | NaN | - | 0.872 | -1 | NaN | - | 0.612 | 512/0/0 | 512 | 4239.9 | - |  |
| r8xyz | d2c | 6.5,6.5,0.0 | 0.848 | -1 | NaN | - | 1.010 | -1 | NaN | - | NaN | offlat | 512 | NaN | - |  |
| r8xyz | d2e | 6.5,6.5,3.5 | 0.792 | -1 | NaN | - | 0.718 | -1 | NaN | - | 0.408 | 512/0/0 | 512 | 5179.8 | - |  |
| r16x | x1c | 9.5,0.0,0.0 | 0.907 | -1 | NaN | - | 0.791 | -1 | NaN | - | 0.866 | 15/1/0 | 16 | 280.7 | - |  |
| r16x | x2c | 10.5,0.0,0.0 | 0.821 | -1 | NaN | - | 0.716 | -1 | NaN | - | 0.577 | 16/0/0 | 16 | 176.7 | - |  |
| r16x | x3c | 11.5,0.0,0.0 | 0.749 | -1 | NaN | - | 0.654 | -1 | NaN | - | 0.433 | 16/0/0 | 16 | 164.3 | - |  |
| r16xyz | x1c | 9.5,0.0,0.0 | 1.550 | -1 | NaN | - | 7.079 | -1 | NaN | - | NaN | offlat | 4096 | NaN | - |  |
| r16xyz | x1e | 9.5,7.5,0.0 | 1.216 | -1 | NaN | - | 4.764 | -1 | NaN | - | NaN | offlat | 4096 | NaN | - |  |
| r16xyz | x1v | 9.5,7.5,7.5 | 1.034 | -1 | NaN | - | 0.991 | -1 | NaN | - | 0.866 | 4092/4/0 | 4096 | 42564.4 | - |  |
| r16xyz | x2c | 10.5,0.0,0.0 | 1.402 | -1 | NaN | - | 4.247 | -1 | NaN | - | NaN | offlat | 4096 | NaN | - |  |
| r16xyz | x2e | 10.5,7.5,0.0 | 1.141 | -1 | NaN | - | 2.955 | -1 | NaN | - | NaN | offlat | 4096 | NaN | - |  |
| r16xyz | x2v | 10.5,7.5,7.5 | 0.986 | -1 | NaN | - | 0.974 | -1 | NaN | - | 0.577 | 4096/0/0 | 4096 | 43103.3 | - |  |
| r16xyz | x3c | 11.5,0.0,0.0 | 1.280 | -1 | NaN | - | 3.034 | -1 | NaN | - | NaN | offlat | 4096 | NaN | - |  |
| r16xyz | x3e | 11.5,7.5,0.0 | 1.072 | -1 | NaN | - | 2.131 | -1 | NaN | - | NaN | offlat | 4096 | NaN | - |  |
| r16xyz | x3v | 11.5,7.5,7.5 | 0.941 | -1 | NaN | - | 0.951 | -1 | NaN | - | 0.433 | 4096/0/0 | 4096 | 39974.1 | - |  |

### Far offsets f index 3: kap >= 4 (whole-box route), times and agreement

| shape | off | R/g | rho_w | L_w | t_w | L_p | t_p | t_gcd | agreement |
|---|---|---|---|---|---|---|---|---|---|
| r2x | x4c | 5.5,0.0,0.0 | 0.375 | 28 | 3.07 | 21 | 33.71 | 32.5 | whole vs gcd: 4.41e-16 (max-entry); split vs gcd: 4.41e-16 (max-entry); whole vs split: 7.16e-16 (max-entry) |
| r2x | x6c | 7.5,0.0,0.0 | 0.275 | 22 | 2.15 | 17 | 25.18 | 26.1 | whole vs gcd: 4.05e-16 (max-entry); split vs gcd: 5.40e-16 (max-entry); whole vs split: 4.05e-16 (max-entry) |
| r2x | x8c | 9.5,0.0,0.0 | 0.217 | 20 | 1.76 | 15 | 21.72 | 26.4 | whole vs gcd: 3.25e-16 (max-entry); split vs gcd: 2.60e-16 (max-entry); whole vs split: 5.20e-16 (max-entry) |
| r2x | x16c | 17.5,0.0,0.0 | 0.118 | 14 | 1.04 | 12 | 15.63 | 26.8 | whole vs gcd: 1.72e-16 (max-entry); split vs gcd: 7.86e-16 (max-entry); whole vs split: 6.21e-16 (max-entry) |
| r2x | x32c | 33.5,0.0,0.0 | 0.062 | 12 | 0.88 | 11 | 14.17 | 24.1 | whole vs gcd: 3.59e-16 (max-entry); split vs gcd: 4.86e-16 (max-entry); whole vs split: 4.67e-16 (max-entry) |
| r2x | d4c | 5.5,5.0,0.0 | 0.277 | 22 | 2.28 | 17 | 25.18 | 25.5 | whole vs gcd: 9.67e-16 (max-entry); split vs gcd: 1.16e-15 (max-entry); whole vs split: 2.90e-16 (max-entry) |
| r2x | d8c | 9.5,9.0,0.0 | 0.158 | 16 | 1.27 | 14 | 19.45 | 23.8 | whole vs gcd: 4.17e-16 (max-entry); split vs gcd: 1.30e-15 (max-entry); whole vs split: 1.54e-15 (max-entry) |
| r2x | d16c | 17.5,17.0,0.0 | 0.084 | 14 | 1.07 | 11 | 13.87 | 22.7 | whole vs gcd: 3.71e-16 (max-entry); split vs gcd: 6.12e-16 (max-entry); whole vs split: 2.72e-16 (max-entry) |
| r2xyz | x4c | 5.5,0.0,0.0 | 0.472 | 34 | 4.76 | 21 | 58.78 | NaN | whole vs split: 1.27e-15 (max-entry) |
| r2xyz | x4e | 5.5,0.5,0.0 | 0.470 | 34 | 4.79 | 21 | 60.49 | NaN | whole vs split: 5.93e-16 (max-entry) |
| r2xyz | x4v | 5.5,0.5,0.5 | 0.469 | 34 | 4.69 | 21 | 60.31 | 97.8 | whole vs gcd: 6.07e-16 (max-entry); split vs gcd: 2.43e-16 (max-entry); whole vs split: 4.85e-16 (max-entry) |
| r2xyz | x6c | 7.5,0.0,0.0 | 0.346 | 26 | 2.58 | 18 | 49.66 | NaN | whole vs split: 8.31e-16 (max-entry) |
| r2xyz | x6e | 7.5,0.5,0.0 | 0.346 | 26 | 2.48 | 18 | 46.26 | NaN | whole vs split: 7.01e-16 (max-entry) |
| r2xyz | x6v | 7.5,0.5,0.5 | 0.345 | 26 | 2.57 | 18 | 48.17 | 95.8 | whole vs gcd: 2.84e-16 (max-entry); split vs gcd: 2.13e-16 (max-entry); whole vs split: 4.26e-16 (max-entry) |
| r2xyz | x8c | 9.5,0.0,0.0 | 0.273 | 22 | 2.17 | 16 | 41.90 | NaN | whole vs split: 2.64e-16 (max-entry) |
| r2xyz | x8e | 9.5,0.5,0.0 | 0.273 | 22 | 2.57 | 16 | 45.11 | NaN | whole vs split: 1.99e-16 (max-entry) |
| r2xyz | x8v | 9.5,0.5,0.5 | 0.273 | 22 | 2.27 | 16 | 41.43 | 75.1 | whole vs gcd: 4.01e-16 (max-entry); split vs gcd: 2.68e-16 (max-entry); whole vs split: 2.03e-16 (max-entry) |
| r2xyz | x16c | 17.5,0.0,0.0 | 0.148 | 16 | 1.27 | 13 | 31.48 | NaN | whole vs split: 4.41e-16 (max-entry) |
| r2xyz | x16e | 17.5,0.5,0.0 | 0.148 | 16 | 1.79 | 13 | 34.62 | NaN | whole vs split: 5.17e-16 (max-entry) |
| r2xyz | x16v | 17.5,0.5,0.5 | 0.148 | 16 | 1.28 | 13 | 31.66 | 81.6 | whole vs gcd: 3.20e-16 (max-entry); split vs gcd: 3.11e-16 (max-entry); whole vs split: 1.98e-16 (max-entry) |
| r2xyz | x32c | 33.5,0.0,0.0 | 0.078 | 12 | 0.89 | 11 | 25.82 | NaN | whole vs split: 5.05e-16 (max-entry) |
| r2xyz | x32e | 33.5,0.5,0.0 | 0.078 | 12 | 0.90 | 11 | 25.13 | NaN | whole vs split: 2.00e-16 (max-entry) |
| r2xyz | x32v | 33.5,0.5,0.5 | 0.078 | 12 | 0.88 | 11 | 25.94 | 77.6 | whole vs gcd: 3.22e-16 (max-entry); split vs gcd: 4.00e-16 (max-entry); whole vs split: 2.00e-16 (max-entry) |
| r2xyz | d4c | 5.5,5.5,0.0 | 0.334 | 24 | 2.23 | 18 | 42.75 | NaN | whole vs split: 3.32e-16 (max-entry) |
| r2xyz | d4e | 5.5,5.5,0.5 | 0.333 | 24 | 2.11 | 18 | 44.41 | 82.9 | whole vs gcd: 6.71e-16 (max-entry); split vs gcd: 4.48e-16 (max-entry); whole vs split: 2.24e-16 (max-entry) |
| r2xyz | d8c | 9.5,9.5,0.0 | 0.193 | 18 | 1.41 | 14 | 33.16 | NaN | whole vs split: 2.53e-16 (max-entry) |
| r2xyz | d8e | 9.5,9.5,0.5 | 0.193 | 18 | 1.37 | 14 | 33.09 | 91.3 | whole vs gcd: 2.56e-16 (max-entry); split vs gcd: 5.09e-16 (max-entry); whole vs split: 5.09e-16 (max-entry) |
| r2xyz | d16c | 17.5,17.5,0.0 | 0.105 | 14 | 0.99 | 12 | 24.82 | NaN | whole vs split: 6.18e-16 (max-entry) |
| r2xyz | d16e | 17.5,17.5,0.5 | 0.105 | 14 | 0.97 | 12 | 26.09 | 81.3 | whole vs gcd: 4.95e-16 (max-entry); split vs gcd: 4.95e-16 (max-entry); whole vs split: 2.76e-16 (max-entry) |
| r4x | x4c | 6.5,0.0,0.0 | 0.442 | 36 | 5.22 | 27 | 39.39 | 37.4 | whole vs gcd: 7.82e-16 (max-entry); split vs gcd: 3.13e-16 (max-entry); whole vs split: 6.26e-16 (max-entry) |
| r4x | x6c | 8.5,0.0,0.0 | 0.338 | 28 | 3.11 | 23 | 30.23 | 43.6 | whole vs gcd: 3.55e-16 (max-entry); split vs gcd: 8.88e-16 (max-entry); whole vs split: 5.33e-16 (max-entry) |
| r4x | x8c | 10.5,0.0,0.0 | 0.274 | 24 | 3.03 | 20 | 27.35 | 45.2 | whole vs gcd: 9.73e-16 (max-entry); split vs gcd: 3.27e-16 (max-entry); whole vs split: 1.14e-15 (max-entry) |
| r4x | x16c | 18.5,0.0,0.0 | 0.155 | 18 | 1.38 | 16 | 16.98 | 41.2 | whole vs gcd: 2.53e-16 (max-entry); split vs gcd: 1.74e-16 (max-entry); whole vs split: 3.48e-16 (max-entry) |
| r4x | x32c | 34.5,0.0,0.0 | 0.083 | 14 | 1.02 | 13 | 14.00 | 36.1 | whole vs gcd: 8.83e-16 (max-entry); split vs gcd: 9.88e-16 (max-entry); whole vs split: 1.01e-15 (max-entry) |
| r4x | d4c | 6.5,5.0,0.0 | 0.350 | 30 | 3.48 | 24 | 28.36 | 37.7 | whole vs gcd: 5.04e-16 (max-entry); split vs gcd: 2.02e-15 (max-entry); whole vs split: 2.39e-15 (max-entry) |
| r4x | d8c | 10.5,9.0,0.0 | 0.208 | 20 | 1.80 | 18 | 19.96 | 43.9 | whole vs gcd: 4.22e-16 (max-entry); split vs gcd: 2.73e-16 (max-entry); whole vs split: 2.75e-16 (max-entry) |
| r4x | d16c | 18.5,17.0,0.0 | 0.114 | 16 | 1.21 | 14 | 15.21 | 42.5 | whole vs gcd: 1.40e-16 (max-entry); split vs gcd: 3.75e-16 (max-entry); whole vs split: 3.80e-16 (max-entry) |
| r4xyz | x4c | 6.5,0.0,0.0 | 0.666 | -1 | NaN | 49 | 106.18 | NaN |  |
| r4xyz | x4e | 6.5,1.5,0.0 | 0.649 | -1 | NaN | 45 | 104.62 | NaN |  |
| r4xyz | x4v | 6.5,1.5,1.5 | 0.633 | 54 | 11.13 | 43 | 96.00 | 686.8 | whole vs gcd: 6.23e-16 (max-entry); split vs gcd: 1.28e-16 (max-entry); whole vs split: 6.23e-16 (max-entry) |
| r4xyz | x6c | 8.5,0.0,0.0 | 0.509 | 38 | 5.65 | 32 | 67.99 | NaN | whole vs split: 1.99e-16 (max-entry) |
| r4xyz | x6e | 8.5,1.5,0.0 | 0.502 | 38 | 5.69 | 32 | 69.08 | NaN | whole vs split: 4.28e-16 (max-entry) |
| r4xyz | x6v | 8.5,1.5,1.5 | 0.494 | 38 | 5.47 | 31 | 70.32 | 595.9 | whole vs gcd: 5.79e-16 (max-entry); split vs gcd: 1.24e-16 (max-entry); whole vs split: 5.79e-16 (max-entry) |
| r4xyz | x8c | 10.5,0.0,0.0 | 0.412 | 30 | 3.53 | 26 | 57.26 | NaN | whole vs split: 3.47e-16 (max-entry) |
| r4xyz | x8e | 10.5,1.5,0.0 | 0.408 | 30 | 3.31 | 26 | 52.88 | NaN | whole vs split: 5.47e-16 (max-entry) |
| r4xyz | x8v | 10.5,1.5,1.5 | 0.404 | 30 | 3.45 | 26 | 59.82 | 659.7 | whole vs gcd: 3.85e-16 (max-entry); split vs gcd: 3.87e-16 (max-entry); whole vs split: 2.40e-16 (max-entry) |
| r4xyz | x16c | 18.5,0.0,0.0 | 0.234 | 20 | 1.66 | 19 | 40.42 | NaN | whole vs split: 3.65e-16 (max-entry) |
| r4xyz | x16e | 18.5,1.5,0.0 | 0.233 | 20 | 1.76 | 19 | 37.55 | NaN | whole vs split: 8.99e-16 (max-entry) |
| r4xyz | x16v | 18.5,1.5,1.5 | 0.233 | 20 | 1.69 | 19 | 41.60 | 561.2 | whole vs gcd: 4.08e-16 (max-entry); split vs gcd: 3.29e-16 (max-entry); whole vs split: 5.55e-16 (max-entry) |
| r4xyz | x32c | 34.5,0.0,0.0 | 0.126 | 16 | 1.18 | 15 | 28.77 | NaN | whole vs split: 3.39e-16 (max-entry) |
| r4xyz | x32e | 34.5,1.5,0.0 | 0.125 | 16 | 1.24 | 15 | 28.23 | NaN | whole vs split: 3.39e-16 (max-entry) |
| r4xyz | x32v | 34.5,1.5,1.5 | 0.125 | 16 | 1.28 | 15 | 29.67 | 604.4 | whole vs gcd: 5.53e-16 (max-entry); split vs gcd: 5.36e-16 (max-entry); whole vs split: 1.65e-16 (max-entry) |
| r4xyz | d4c | 6.5,6.5,0.0 | 0.471 | 36 | 5.24 | 28 | 63.35 | NaN | whole vs split: 7.15e-16 (max-entry) |
| r4xyz | d4e | 6.5,6.5,1.5 | 0.465 | 34 | 4.58 | 28 | 66.92 | 664.4 | whole vs gcd: 3.92e-16 (max-entry); split vs gcd: 1.92e-16 (max-entry); whole vs split: 3.81e-16 (max-entry) |
| r4xyz | d8c | 10.5,10.5,0.0 | 0.292 | 24 | 2.27 | 21 | 45.81 | NaN | whole vs split: 3.30e-16 (max-entry) |
| r4xyz | d8e | 10.5,10.5,1.5 | 0.290 | 24 | 2.34 | 21 | 44.88 | 556.2 | whole vs gcd: 3.40e-16 (max-entry); split vs gcd: 3.40e-16 (max-entry); whole vs split: 3.37e-16 (max-entry) |
| r4xyz | d16c | 18.5,18.5,0.0 | 0.166 | 18 | 1.47 | 16 | 34.67 | NaN | whole vs split: 2.59e-16 (max-entry) |
| r4xyz | d16e | 18.5,18.5,1.5 | 0.165 | 18 | 2.24 | 16 | 31.78 | 563.4 | whole vs gcd: 7.85e-16 (max-entry); split vs gcd: 5.26e-16 (max-entry); whole vs split: 2.61e-16 (max-entry) |
| r8x | x4c | 8.5,0.0,0.0 | 0.555 | 54 | 12.52 | 43 | 62.19 | 74.7 | whole vs gcd: 1.64e-15 (max-entry); split vs gcd: 1.77e-15 (max-entry); whole vs split: 1.20e-15 (max-entry) |
| r8x | x6c | 10.5,0.0,0.0 | 0.449 | 40 | 6.13 | 35 | 42.77 | 84.0 | whole vs gcd: 7.95e-16 (max-entry); split vs gcd: 1.19e-15 (max-entry); whole vs split: 3.98e-16 (max-entry) |
| r8x | x8c | 12.5,0.0,0.0 | 0.377 | 34 | 4.62 | 30 | 33.79 | 84.7 | whole vs gcd: 6.85e-16 (max-entry); split vs gcd: 2.30e-16 (max-entry); whole vs split: 4.56e-16 (max-entry) |
| r8x | x16c | 20.5,0.0,0.0 | 0.230 | 22 | 2.08 | 22 | 23.13 | 70.7 | whole vs gcd: 7.51e-16 (max-entry); split vs gcd: 1.08e-15 (max-entry); whole vs split: 1.82e-15 (max-entry) |
| r8x | x32c | 36.5,0.0,0.0 | 0.129 | 18 | 1.53 | 17 | 16.72 | 80.9 | whole vs gcd: 1.39e-15 (max-entry); split vs gcd: 4.14e-16 (max-entry); whole vs split: 1.07e-15 (max-entry) |
| r8x | d4c | 8.5,5.0,0.0 | 0.478 | 44 | 8.59 | 38 | 49.99 | 91.1 | whole vs gcd: 1.36e-15 (max-entry); split vs gcd: 5.81e-16 (max-entry); whole vs split: 1.07e-15 (max-entry) |
| r8x | d8c | 12.5,9.0,0.0 | 0.306 | 28 | 3.08 | 26 | 25.62 | 93.2 | whole vs gcd: 7.11e-16 (max-entry); split vs gcd: 3.97e-16 (max-entry); whole vs split: 7.11e-16 (max-entry) |
| r8x | d16c | 20.5,17.0,0.0 | 0.177 | 20 | 1.72 | 19 | 19.03 | 90.9 | whole vs gcd: 9.19e-16 (max-entry); split vs gcd: 2.65e-16 (max-entry); whole vs split: 1.18e-15 (max-entry) |
| r8xyz | x4c | 8.5,0.0,0.0 | 0.917 | -1 | NaN | -1 | NaN | NaN |  |
| r8xyz | x4e | 8.5,3.5,0.0 | 0.848 | -1 | NaN | -1 | NaN | NaN |  |
| r8xyz | x4v | 8.5,3.5,3.5 | 0.792 | -1 | NaN | -1 | NaN | 6090.5 |  |
| r8xyz | x6c | 10.5,0.0,0.0 | 0.742 | -1 | NaN | -1 | NaN | NaN |  |
| r8xyz | x6e | 10.5,3.5,0.0 | 0.704 | -1 | NaN | -1 | NaN | NaN |  |
| r8xyz | x6v | 10.5,3.5,3.5 | 0.671 | -1 | NaN | -1 | NaN | 4328.2 |  |
| r8xyz | x8c | 12.5,0.0,0.0 | 0.624 | 56 | 12.38 | -1 | NaN | NaN |  |
| r8xyz | x8e | 12.5,3.5,0.0 | 0.600 | 52 | 17.50 | -1 | NaN | NaN |  |
| r8xyz | x8v | 12.5,3.5,3.5 | 0.580 | 50 | 9.50 | 52 | 103.85 | 5324.9 | whole vs gcd: 1.60e-15 (max-entry); split vs gcd: 1.58e-15 (max-entry); whole vs split: 4.03e-16 (max-entry) |
| r8xyz | x16c | 20.5,0.0,0.0 | 0.380 | 30 | 3.56 | 30 | 60.49 | NaN | whole vs split: 3.82e-16 (max-entry) |
| r8xyz | x16e | 20.5,3.5,0.0 | 0.375 | 30 | 3.42 | 29 | 59.69 | NaN | whole vs split: 9.80e-16 (max-entry) |
| r8xyz | x16v | 20.5,3.5,3.5 | 0.370 | 30 | 3.49 | 29 | 62.03 | 4641.7 | whole vs gcd: 5.97e-16 (max-entry); split vs gcd: 5.22e-16 (max-entry); whole vs split: 3.24e-16 (max-entry) |
| r8xyz | x32c | 36.5,0.0,0.0 | 0.214 | 20 | 1.69 | 21 | 38.39 | NaN | whole vs split: 5.26e-16 (max-entry) |
| r8xyz | x32e | 36.5,3.5,0.0 | 0.213 | 20 | 3.00 | 21 | 39.68 | NaN | whole vs split: 3.67e-16 (max-entry) |
| r8xyz | x32v | 36.5,3.5,3.5 | 0.212 | 20 | 1.80 | 21 | 38.40 | 5090.7 | whole vs gcd: 7.15e-16 (max-entry); split vs gcd: 1.07e-15 (max-entry); whole vs split: 5.37e-16 (max-entry) |
| r8xyz | d4c | 8.5,8.5,0.0 | 0.648 | -1 | NaN | -1 | NaN | NaN |  |
| r8xyz | d4e | 8.5,8.5,3.5 | 0.623 | 56 | 12.02 | 50 | 118.70 | 5106.3 | whole vs gcd: 1.36e-15 (max-entry); split vs gcd: 6.78e-16 (max-entry); whole vs split: 1.19e-15 (max-entry) |
| r8xyz | d8c | 12.5,12.5,0.0 | 0.441 | 34 | 4.39 | 34 | 69.57 | NaN | whole vs split: 3.44e-16 (max-entry) |
| r8xyz | d8e | 12.5,12.5,3.5 | 0.433 | 34 | 4.62 | 33 | 69.95 | 4894.1 | whole vs gcd: 8.34e-16 (max-entry); split vs gcd: 1.25e-15 (max-entry); whole vs split: 4.16e-16 (max-entry) |
| r8xyz | d16c | 20.5,20.5,0.0 | 0.269 | 24 | 2.43 | 24 | 45.44 | NaN | whole vs split: 7.29e-16 (max-entry) |
| r8xyz | d16e | 20.5,20.5,3.5 | 0.267 | 24 | 2.47 | 24 | 46.11 | 4957.1 | whole vs gcd: 7.46e-16 (max-entry); split vs gcd: 8.78e-16 (max-entry); whole vs split: 4.45e-16 (max-entry) |
| r16x | x4c | 12.5,0.0,0.0 | 0.689 | -1 | NaN | -1 | NaN | 160.9 |  |
| r16x | x6c | 14.5,0.0,0.0 | 0.594 | -1 | NaN | -1 | NaN | 158.5 |  |
| r16xyz | x4c | 12.5,0.0,0.0 | 1.178 | -1 | NaN | -1 | NaN | NaN |  |
| r16xyz | x4e | 12.5,7.5,0.0 | 1.010 | -1 | NaN | -1 | NaN | NaN |  |
| r16xyz | x4v | 12.5,7.5,7.5 | 0.898 | -1 | NaN | -1 | NaN | 34016.5 |  |
| r16xyz | x6c | 14.5,0.0,0.0 | 1.015 | -1 | NaN | -1 | NaN | NaN |  |
| r16xyz | x6e | 14.5,7.5,0.0 | 0.902 | -1 | NaN | -1 | NaN | NaN |  |
| r16xyz | x6v | 14.5,7.5,7.5 | 0.819 | -1 | NaN | -1 | NaN | 34828.7 |  |
| r16xyz | x8c | 16.5,0.0,0.0 | 0.892 | -1 | NaN | -1 | NaN | NaN |  |
| r16xyz | x8e | 16.5,7.5,0.0 | 0.812 | -1 | NaN | -1 | NaN | NaN |  |
| r16xyz | x8v | 16.5,7.5,7.5 | 0.751 | -1 | NaN | -1 | NaN | 50978.6 |  |
| r16xyz | x16c | 24.5,0.0,0.0 | 0.601 | 56 | 11.92 | -1 | NaN | NaN |  |
| r16xyz | x16e | 24.5,7.5,0.0 | 0.575 | 52 | 10.35 | -1 | NaN | NaN |  |
| r16xyz | x16v | 24.5,7.5,7.5 | 0.551 | 48 | 9.14 | -1 | NaN | 39755.0 | whole vs gcd: 4.25e-11 (max-entry) |

## 3e. est-ratio scan at f = 1 + 0.1i and 0.37 (estratio2.jl -> estratio_f2.md, estratio_f3.md; fixed n-cut; {x}, {x,y,z})

Per on-lattice offset of the matrix: est_X/max|G| (G the gcd average), Lambda of the gcd sum, whole vs gcd where the whole
box converges, and each route against the reference where one exists (the lateral-0 x family at kap 1, 2, 4, 8, 32).
f = 1 + 0.1i: 104 rows; est_X/max|G| 0.43 .. 4.77, Lambda 1.00 .. 2.29, whole vs gcd <= 3.3e-15, worst vs reference
whole 1.4e-15, gcd 9.8e-16, split 2.6e-15.  f = 0.37: 104 rows; est_X/max|G| 0.26 .. 4.87, Lambda 1.00 .. 2.15,
whole vs gcd <= 2.4e-15, worst vs reference whole 2.4e-15, gcd 2.7e-15, split 2.5e-15.  (f = 1: estratio_f1.txt, S2c/S2d:
est/max|G| 0.50 .. 5.0, Lambda 1.0 .. 2.2.)  The "cancellation finding" of the r4-r16 rows is therefore only that
Lambda grows with the ratio and the number of coarse axes (1.0 at r2, 1.1-1.3 at r8, 1.5-2.3 at r16 xyz): the gcd
average loses at most one bit to interference at these cell sizes, at all three frequencies.

### f = 1 + 0.1i
| shape | off | R/g | rho_w | L_w | est_X/max\|G\| | Lambda_gcd | whole vs gcd | err_w vs ref | err_gcd vs ref | err_p vs ref |
|---|---|---|---|---|---|---|---|---|---|---|
| r2x | x1c | 2.5,0.0,0.0 | 0.825 | -1 | 1.67 | 1 | - | - | 5.8e-16,1.3e-15 | 8.6e-16,8.6e-16 |
| r2x | x2c | 3.5,0.0,0.0 | 0.589 | 48 | 1.98 | 1 | 1.11e-15 | 3.4e-16,5.1e-16 | 7.7e-16,1.1e-15 | 6.1e-16,1.5e-15 |
| r2x | x3c | 4.5,0.0,0.0 | 0.458 | 34 | 2.18 | 1 | 7.54e-16 | 6.1e-16,1.6e-15 | 9.8e-16,9.8e-16 | 1.2e-15,1.2e-15 |
| r2x | x4c | 5.5,0.0,0.0 | 0.375 | 28 | 2.32 | 1 | 4.11e-16 | 2.6e-16,6.2e-16 | 3.0e-16,6.6e-16 | 1.4e-15,1.4e-15 |
| r2x | x6c | 7.5,0.0,0.0 | 0.275 | 22 | 2.54 | 1 | 1.99e-16 | 5.0e-16,5.5e-16 | 3.8e-16,6.7e-16 | 1.9e-16,3.3e-16 |
| r2x | x8c | 9.5,0.0,0.0 | 0.217 | 20 | 2.73 | 1 | 2.88e-16 | 3.2e-16,4.2e-16 | 4.4e-16,4.4e-16 | 2.2e-16,2.2e-16 |
| r2x | x16c | 17.5,0.0,0.0 | 0.118 | 16 | 2.14 | 1 | 6.97e-16 | 5.9e-16,5.9e-16 | 2.9e-16,2.9e-16 | 5.8e-16,9.4e-16 |
| r2x | x32c | 33.5,0.0,0.0 | 0.062 | 12 | 1.53 | 1 | 5.91e-16 | 3.8e-16,3.8e-16 | 2.3e-16,4.5e-16 | 2.2e-16,6.2e-16 |
| r2x | d1c | 2.5,2.0,0.0 | 0.644 | 56 | 2.97 | 1 | 6.50e-16 | - | - | - |
| r2x | d2c | 3.5,3.0,0.0 | 0.447 | 34 | 3.43 | 1 | 3.45e-16 | - | - | - |
| r2x | d4c | 5.5,5.0,0.0 | 0.277 | 22 | 3.84 | 1 | 6.20e-16 | - | - | - |
| r2x | d8c | 9.5,9.0,0.0 | 0.158 | 16 | 2.67 | 1 | 4.45e-16 | - | - | - |
| r2x | d16c | 17.5,17.0,0.0 | 0.084 | 14 | 1.76 | 1 | 3.90e-16 | - | - | - |
| r2xyz | x1v | 2.5,0.5,0.5 | 1.000 | -1 | 2.38 | 1 | - | - | - | - |
| r2xyz | x2v | 3.5,0.5,0.5 | 0.728 | -1 | 2.36 | 1 | - | - | - | - |
| r2xyz | x3v | 4.5,0.5,0.5 | 0.570 | 44 | 2.41 | 1 | 8.67e-16 | - | - | - |
| r2xyz | x4v | 5.5,0.5,0.5 | 0.469 | 34 | 2.48 | 1 | 7.11e-16 | - | - | - |
| r2xyz | x6v | 7.5,0.5,0.5 | 0.345 | 26 | 2.63 | 1 | 1.39e-16 | - | - | - |
| r2xyz | x8v | 9.5,0.5,0.5 | 0.273 | 22 | 2.79 | 1 | 8.26e-16 | - | - | - |
| r2xyz | x16v | 17.5,0.5,0.5 | 0.148 | 16 | 2.15 | 1 | 6.73e-16 | - | - | - |
| r2xyz | x32v | 33.5,0.5,0.5 | 0.078 | 14 | 1.53 | 1 | 2.59e-16 | - | - | - |
| r2xyz | d1e | 2.5,2.5,0.5 | 0.728 | -1 | 3.26 | 1 | - | - | - | - |
| r2xyz | d2e | 3.5,3.5,0.5 | 0.522 | 40 | 3.59 | 1 | 2.08e-16 | - | - | - |
| r2xyz | d4e | 5.5,5.5,0.5 | 0.333 | 26 | 3.89 | 1 | 4.68e-16 | - | - | - |
| r2xyz | d8e | 9.5,9.5,0.5 | 0.193 | 18 | 2.62 | 1 | 3.27e-16 | - | - | - |
| r2xyz | d16e | 17.5,17.5,0.5 | 0.105 | 14 | 1.75 | 1 | 2.69e-16 | - | - | - |
| r4x | x1c | 3.5,0.0,0.0 | 0.821 | -1 | 1.2 | 1 | - | - | 4.9e-16,1.0e-15 | 5.9e-16,1.4e-15 |
| r4x | x2c | 4.5,0.0,0.0 | 0.638 | -1 | 1.66 | 1 | - | - | 5.6e-16,1.0e-15 | 4.3e-16,9.5e-16 |
| r4x | x3c | 5.5,0.0,0.0 | 0.522 | 44 | 1.97 | 1.01 | 4.84e-16 | 1.0e-15,1.0e-15 | 7.6e-16,9.2e-16 | 1.2e-15,1.2e-15 |
| r4x | x4c | 6.5,0.0,0.0 | 0.442 | 36 | 2.19 | 1.01 | 6.65e-16 | 5.8e-16,1.2e-15 | 2.9e-16,6.1e-16 | 9.1e-16,1.2e-15 |
| r4x | x6c | 8.5,0.0,0.0 | 0.338 | 28 | 2.51 | 1.01 | 5.88e-16 | 4.3e-16,4.3e-16 | 3.5e-16,5.0e-16 | 9.0e-16,9.0e-16 |
| r4x | x8c | 10.5,0.0,0.0 | 0.274 | 24 | 2.75 | 1.02 | 5.99e-16 | 4.8e-16,4.8e-16 | 5.3e-16,5.3e-16 | 6.9e-16,8.1e-16 |
| r4x | x16c | 18.5,0.0,0.0 | 0.155 | 18 | 2.09 | 1.02 | 7.01e-16 | 5.4e-16,5.4e-16 | 1.7e-16,1.7e-16 | 1.1e-15,1.1e-15 |
| r4x | x32c | 34.5,0.0,0.0 | 0.083 | 14 | 1.54 | 1.02 | 3.42e-16 | 3.8e-16,3.8e-16 | 1.5e-16,5.1e-16 | 3.1e-16,5.1e-16 |
| r4x | d1c | 3.5,2.0,0.0 | 0.713 | -1 | 2.71 | 1.1 | - | - | - | - |
| r4x | d2c | 4.5,3.0,0.0 | 0.531 | 46 | 3.38 | 1.07 | 3.39e-16 | - | - | - |
| r4x | d4c | 6.5,5.0,0.0 | 0.350 | 30 | 3.86 | 1.04 | 9.87e-16 | - | - | - |
| r4x | d8c | 10.5,9.0,0.0 | 0.208 | 20 | 2.56 | 1.01 | 4.42e-16 | - | - | - |
| r4x | d16c | 18.5,17.0,0.0 | 0.114 | 16 | 1.75 | 1.01 | 1.96e-16 | - | - | - |
| r4xyz | x1v | 3.5,1.5,1.5 | 1.058 | -1 | 3.21 | 1.27 | - | - | - | - |
| r4xyz | x2v | 4.5,1.5,1.5 | 0.870 | -1 | 3.1 | 1.1 | - | - | - | - |
| r4xyz | x3v | 5.5,1.5,1.5 | 0.735 | -1 | 3 | 1.03 | - | - | - | - |
| r4xyz | x4v | 6.5,1.5,1.5 | 0.633 | 54 | 2.95 | 1.02 | 1.13e-15 | - | - | - |
| r4xyz | x6v | 8.5,1.5,1.5 | 0.494 | 38 | 2.98 | 1.02 | 1.42e-15 | - | - | - |
| r4xyz | x8v | 10.5,1.5,1.5 | 0.404 | 30 | 3.07 | 1.04 | 1.34e-15 | - | - | - |
| r4xyz | x16v | 18.5,1.5,1.5 | 0.233 | 20 | 2.12 | 1.03 | 3.89e-16 | - | - | - |
| r4xyz | x32v | 34.5,1.5,1.5 | 0.125 | 16 | 1.54 | 1.02 | 4.15e-16 | - | - | - |
| r4xyz | d1e | 3.5,3.5,1.5 | 0.837 | -1 | 3.81 | 1.12 | - | - | - | - |
| r4xyz | d2e | 4.5,4.5,1.5 | 0.662 | -1 | 4.02 | 1.05 | - | - | - | - |
| r4xyz | d4e | 6.5,6.5,1.5 | 0.465 | 36 | 3.86 | 1.05 | 5.43e-16 | - | - | - |
| r4xyz | d8e | 10.5,10.5,1.5 | 0.290 | 24 | 2.49 | 1.02 | 2.63e-16 | - | - | - |
| r4xyz | d16e | 18.5,18.5,1.5 | 0.165 | 18 | 1.74 | 1.02 | 4.45e-16 | - | - | - |
| r8x | x1c | 5.5,0.0,0.0 | 0.858 | -1 | 0.74 | 1.01 | - | - | 3.9e-16,8.4e-16 | - |
| r8x | x2c | 6.5,0.0,0.0 | 0.726 | -1 | 1.23 | 1.02 | - | - | 6.1e-16,8.9e-16 | - |
| r8x | x3c | 7.5,0.0,0.0 | 0.629 | -1 | 1.62 | 1.03 | - | - | 6.5e-16,7.2e-16 | 2.6e-15,2.6e-15 |
| r8x | x4c | 8.5,0.0,0.0 | 0.555 | 54 | 1.94 | 1.04 | 8.98e-16 | 5.9e-16,1.0e-15 | 3.9e-16,5.4e-16 | 6.8e-16,1.2e-15 |
| r8x | x6c | 10.5,0.0,0.0 | 0.449 | 42 | 2.42 | 1.08 | 1.28e-15 | 1.2e-15,1.5e-15 | 4.4e-16,4.7e-16 | 9.0e-16,1.2e-15 |
| r8x | x8c | 12.5,0.0,0.0 | 0.377 | 34 | 2.78 | 1.16 | 3.62e-16 | 7.3e-16,7.4e-16 | 4.4e-16,4.4e-16 | 1.4e-15,1.4e-15 |
| r8x | x16c | 20.5,0.0,0.0 | 0.230 | 24 | 2.07 | 1.09 | 4.61e-16 | 3.7e-16,3.7e-16 | 4.2e-16,4.2e-16 | 1.7e-15,1.7e-15 |
| r8x | x32c | 36.5,0.0,0.0 | 0.129 | 18 | 1.61 | 1.1 | 4.03e-16 | 4.5e-16,4.5e-16 | 1.3e-16,4.7e-16 | 7.9e-16,7.9e-16 |
| r8x | d1c | 5.5,2.0,0.0 | 0.806 | -1 | 2.04 | 1.21 | - | - | - | - |
| r8x | d2c | 6.5,3.0,0.0 | 0.659 | -1 | 2.99 | 1.2 | - | - | - | - |
| r8x | d4c | 8.5,5.0,0.0 | 0.478 | 44 | 3.36 | 1.08 | 7.48e-16 | - | - | - |
| r8x | d8c | 12.5,9.0,0.0 | 0.306 | 28 | 2.43 | 1.05 | 3.98e-16 | - | - | - |
| r8x | d16c | 20.5,17.0,0.0 | 0.177 | 20 | 1.77 | 1.06 | 3.50e-16 | - | - | - |
| r8xyz | x1v | 5.5,3.5,3.5 | 1.053 | -1 | 4.44 | 1.77 | - | - | - | - |
| r8xyz | x2v | 6.5,3.5,3.5 | 0.954 | -1 | 4.45 | 1.53 | - | - | - | - |
| r8xyz | x3v | 7.5,3.5,3.5 | 0.867 | -1 | 4.32 | 1.4 | - | - | - | - |
| r8xyz | x4v | 8.5,3.5,3.5 | 0.792 | -1 | 4.19 | 1.35 | - | - | - | - |
| r8xyz | x6v | 10.5,3.5,3.5 | 0.671 | -1 | 3.64 | 1.24 | - | - | - | - |
| r8xyz | x8v | 12.5,3.5,3.5 | 0.580 | 50 | 3.12 | 1.16 | 5.66e-16 | - | - | - |
| r8xyz | x16v | 20.5,3.5,3.5 | 0.370 | 30 | 2.17 | 1.12 | 5.48e-16 | - | - | - |
| r8xyz | x32v | 36.5,3.5,3.5 | 0.212 | 22 | 1.64 | 1.11 | 4.18e-16 | - | - | - |
| r8xyz | d1e | 5.5,5.5,3.5 | 0.914 | -1 | 4.48 | 1.42 | - | - | - | - |
| r8xyz | d2e | 6.5,6.5,3.5 | 0.792 | -1 | 4.43 | 1.31 | - | - | - | - |
| r8xyz | d4e | 8.5,8.5,3.5 | 0.623 | 56 | 3.36 | 1.1 | 9.04e-16 | - | - | - |
| r8xyz | d8e | 12.5,12.5,3.5 | 0.433 | 36 | 2.42 | 1.09 | 1.48e-15 | - | - | - |
| r8xyz | d16e | 20.5,20.5,3.5 | 0.267 | 24 | 1.81 | 1.1 | 7.84e-16 | - | - | - |
| r16x | x1c | 9.5,0.0,0.0 | 0.907 | -1 | 0.429 | 1.04 | - | - | 4.2e-16,9.0e-16 | - |
| r16x | x2c | 10.5,0.0,0.0 | 0.821 | -1 | 0.84 | 1.09 | - | - | 6.3e-16,7.8e-16 | - |
| r16x | x3c | 11.5,0.0,0.0 | 0.749 | -1 | 1.26 | 1.16 | - | - | 7.0e-16,7.7e-16 | - |
| r16x | x4c | 12.5,0.0,0.0 | 0.689 | -1 | 1.66 | 1.23 | - | - | 3.6e-16,5.3e-16 | - |
| r16x | x6c | 14.5,0.0,0.0 | 0.594 | -1 | 2.38 | 1.4 | - | - | 3.1e-16,3.4e-16 | - |
| r16x | x8c | 16.5,0.0,0.0 | 0.522 | 54 | 2.65 | 1.43 | 6.52e-16 | 8.7e-16,8.7e-16 | 3.4e-16,3.8e-16 | - |
| r16x | x16c | 24.5,0.0,0.0 | 0.352 | 34 | 2.42 | 1.46 | 1.11e-15 | 1.4e-15,1.4e-15 | 3.5e-16,3.5e-16 | - |
| r16x | x32c | 40.5,0.0,0.0 | 0.213 | 26 | 2.1 | 1.52 | 5.90e-16 | 8.1e-16,9.8e-16 | 2.5e-16,4.7e-16 | - |
| r16x | d1c | 9.5,2.0,0.0 | 0.888 | -1 | 1.29 | 1.32 | - | - | - | - |
| r16x | d2c | 10.5,3.0,0.0 | 0.789 | -1 | 2.24 | 1.38 | - | - | - | - |
| r16x | d4c | 12.5,5.0,0.0 | 0.640 | -1 | 2.73 | 1.23 | - | - | - | - |
| r16x | d8c | 16.5,9.0,0.0 | 0.458 | 46 | 2.47 | 1.28 | 6.26e-16 | - | - | - |
| r16x | d16c | 24.5,17.0,0.0 | 0.289 | 30 | 2.02 | 1.3 | 9.12e-16 | - | - | - |
| r16xyz | x1v | 9.5,7.5,7.5 | 1.034 | -1 | 4.77 | 2.29 | - | - | - | - |
| r16xyz | x2v | 10.5,7.5,7.5 | 0.986 | -1 | 4.65 | 2.06 | - | - | - | - |
| r16xyz | x3v | 11.5,7.5,7.5 | 0.941 | -1 | 4.46 | 1.92 | - | - | - | - |
| r16xyz | x4v | 12.5,7.5,7.5 | 0.898 | -1 | 4.26 | 1.83 | - | - | - | - |
| r16xyz | x6v | 14.5,7.5,7.5 | 0.819 | -1 | 3.88 | 1.73 | - | - | - | - |
| r16xyz | x8v | 16.5,7.5,7.5 | 0.751 | -1 | 3.57 | 1.67 | - | - | - | - |
| r16xyz | x16v | 24.5,7.5,7.5 | 0.551 | 52 | 2.79 | 1.59 | 2.07e-15 | - | - | - |
| r16xyz | x32v | 40.5,7.5,7.5 | 0.352 | 32 | 2.22 | 1.56 | 3.33e-15 | - | - | - |
| r16xyz | d1e | 9.5,9.5,7.5 | 0.957 | -1 | 4.56 | 1.97 | - | - | - | - |
| r16xyz | d2e | 10.5,10.5,7.5 | 0.885 | -1 | 4.25 | 1.8 | - | - | - | - |
| r16xyz | d4e | 12.5,12.5,7.5 | 0.767 | -1 | 3.7 | 1.65 | - | - | - | - |
| r16xyz | d8e | 16.5,16.5,7.5 | 0.601 | -1 | 3.01 | 1.54 | - | - | - | - |
| r16xyz | d16e | 24.5,24.5,7.5 | 0.415 | 38 | 2.4 | 1.52 | 2.99e-15 | - | - | - |

done load [3.26220703125, 3.169921875, 2.9560546875]

### f = 0.37
| shape | off | R/g | rho_w | L_w | est_X/max\|G\| | Lambda_gcd | whole vs gcd | err_w vs ref | err_gcd vs ref | err_p vs ref |
|---|---|---|---|---|---|---|---|---|---|---|
| r2x | x1c | 2.5,0.0,0.0 | 0.825 | -1 | 1.42 | 1 | - | - | 2.7e-15,2.7e-15 | 1.5e-15,3.1e-15 |
| r2x | x2c | 3.5,0.0,0.0 | 0.589 | 48 | 1.66 | 1 | 3.32e-16 | 5.2e-16,5.2e-16 | 7.4e-16,7.4e-16 | 1.5e-16,2.2e-16 |
| r2x | x3c | 4.5,0.0,0.0 | 0.458 | 34 | 1.81 | 1 | 6.63e-16 | 1.1e-15,1.1e-15 | 6.0e-16,6.0e-16 | 3.6e-16,6.8e-16 |
| r2x | x4c | 5.5,0.0,0.0 | 0.375 | 28 | 1.93 | 1 | 4.41e-16 | 5.3e-16,1.2e-15 | 2.4e-16,5.4e-16 | 6.8e-16,6.8e-16 |
| r2x | x6c | 7.5,0.0,0.0 | 0.275 | 22 | 2.12 | 1 | 4.05e-16 | 3.8e-16,8.1e-16 | 2.4e-16,2.4e-16 | 4.5e-16,1.2e-15 |
| r2x | x8c | 9.5,0.0,0.0 | 0.217 | 20 | 2.25 | 1 | 3.25e-16 | 2.0e-16,5.6e-16 | 3.1e-16,5.3e-16 | 5.7e-16,5.7e-16 |
| r2x | x16c | 17.5,0.0,0.0 | 0.118 | 14 | 2.6 | 1 | 1.72e-16 | 3.1e-16,3.1e-16 | 4.7e-16,4.7e-16 | 3.1e-16,3.1e-16 |
| r2x | x32c | 33.5,0.0,0.0 | 0.062 | 12 | 2.96 | 1 | 3.59e-16 | 4.5e-16,4.5e-16 | 3.6e-16,3.6e-16 | 3.0e-16,3.0e-16 |
| r2x | d1c | 2.5,2.0,0.0 | 0.644 | 56 | 2.36 | 1 | 7.40e-16 | - | - | - |
| r2x | d2c | 3.5,3.0,0.0 | 0.447 | 34 | 2.65 | 1 | 5.64e-16 | - | - | - |
| r2x | d4c | 5.5,5.0,0.0 | 0.277 | 22 | 3.1 | 1 | 9.67e-16 | - | - | - |
| r2x | d8c | 9.5,9.0,0.0 | 0.158 | 16 | 3.82 | 1 | 4.17e-16 | - | - | - |
| r2x | d16c | 17.5,17.0,0.0 | 0.084 | 14 | 4.12 | 1 | 3.71e-16 | - | - | - |
| r2xyz | x1v | 2.5,0.5,0.5 | 1.000 | -1 | 2.05 | 1 | - | - | - | - |
| r2xyz | x2v | 3.5,0.5,0.5 | 0.728 | -1 | 2 | 1 | - | - | - | - |
| r2xyz | x3v | 4.5,0.5,0.5 | 0.570 | 44 | 2.03 | 1 | 4.19e-16 | - | - | - |
| r2xyz | x4v | 5.5,0.5,0.5 | 0.469 | 34 | 2.08 | 1 | 6.07e-16 | - | - | - |
| r2xyz | x6v | 7.5,0.5,0.5 | 0.345 | 26 | 2.2 | 1 | 2.84e-16 | - | - | - |
| r2xyz | x8v | 9.5,0.5,0.5 | 0.273 | 22 | 2.3 | 1 | 4.01e-16 | - | - | - |
| r2xyz | x16v | 17.5,0.5,0.5 | 0.148 | 16 | 2.61 | 1 | 3.20e-16 | - | - | - |
| r2xyz | x32v | 33.5,0.5,0.5 | 0.078 | 12 | 2.96 | 1 | 3.22e-16 | - | - | - |
| r2xyz | d1e | 2.5,2.5,0.5 | 0.728 | -1 | 2.56 | 1 | - | - | - | - |
| r2xyz | d2e | 3.5,3.5,0.5 | 0.522 | 38 | 2.77 | 1 | 3.63e-16 | - | - | - |
| r2xyz | d4e | 5.5,5.5,0.5 | 0.333 | 24 | 3.18 | 1 | 6.71e-16 | - | - | - |
| r2xyz | d8e | 9.5,9.5,0.5 | 0.193 | 18 | 3.86 | 1 | 2.56e-16 | - | - | - |
| r2xyz | d16e | 17.5,17.5,0.5 | 0.105 | 14 | 4.06 | 1 | 4.95e-16 | - | - | - |
| r4x | x1c | 3.5,0.0,0.0 | 0.821 | -1 | 0.961 | 1 | - | - | 2.3e-15,2.3e-15 | 1.2e-15,2.4e-15 |
| r4x | x2c | 4.5,0.0,0.0 | 0.638 | -1 | 1.33 | 1 | - | - | 7.3e-16,7.3e-16 | 2.5e-15,2.5e-15 |
| r4x | x3c | 5.5,0.0,0.0 | 0.522 | 44 | 1.59 | 1 | 7.22e-16 | 3.0e-16,6.8e-16 | 4.6e-16,4.6e-16 | 8.2e-16,8.9e-16 |
| r4x | x4c | 6.5,0.0,0.0 | 0.442 | 36 | 1.77 | 1 | 7.82e-16 | 3.9e-16,8.9e-16 | 4.0e-16,4.0e-16 | 4.6e-16,1.1e-15 |
| r4x | x6c | 8.5,0.0,0.0 | 0.338 | 28 | 2.03 | 1 | 3.55e-16 | 1.6e-16,4.3e-16 | 2.6e-16,2.6e-16 | 6.3e-16,6.3e-16 |
| r4x | x8c | 10.5,0.0,0.0 | 0.274 | 24 | 2.2 | 1 | 9.73e-16 | 7.5e-16,7.5e-16 | 2.2e-16,4.2e-16 | 3.8e-16,5.1e-16 |
| r4x | x16c | 18.5,0.0,0.0 | 0.155 | 18 | 2.6 | 1 | 2.53e-16 | 1.5e-16,3.2e-16 | 2.5e-16,4.2e-16 | 4.1e-16,5.3e-16 |
| r4x | x32c | 34.5,0.0,0.0 | 0.083 | 14 | 2.88 | 1 | 8.83e-16 | 5.1e-16,5.1e-16 | 3.9e-16,3.9e-16 | 8.3e-16,8.3e-16 |
| r4x | d1c | 3.5,2.0,0.0 | 0.713 | -1 | 2.07 | 1.06 | - | - | - | - |
| r4x | d2c | 4.5,3.0,0.0 | 0.531 | 46 | 2.56 | 1.03 | 1.12e-15 | - | - | - |
| r4x | d4c | 6.5,5.0,0.0 | 0.350 | 30 | 3.15 | 1 | 5.04e-16 | - | - | - |
| r4x | d8c | 10.5,9.0,0.0 | 0.208 | 20 | 3.89 | 1 | 4.22e-16 | - | - | - |
| r4x | d16c | 18.5,17.0,0.0 | 0.114 | 16 | 3.99 | 1 | 1.40e-16 | - | - | - |
| r4xyz | x1v | 3.5,1.5,1.5 | 1.058 | -1 | 2.73 | 1.35 | - | - | - | - |
| r4xyz | x2v | 4.5,1.5,1.5 | 0.870 | -1 | 2.63 | 1.15 | - | - | - | - |
| r4xyz | x3v | 5.5,1.5,1.5 | 0.735 | -1 | 2.53 | 1.05 | - | - | - | - |
| r4xyz | x4v | 6.5,1.5,1.5 | 0.633 | 54 | 2.47 | 1.01 | 6.23e-16 | - | - | - |
| r4xyz | x6v | 8.5,1.5,1.5 | 0.494 | 38 | 2.45 | 1 | 5.79e-16 | - | - | - |
| r4xyz | x8v | 10.5,1.5,1.5 | 0.404 | 30 | 2.49 | 1 | 3.85e-16 | - | - | - |
| r4xyz | x16v | 18.5,1.5,1.5 | 0.233 | 20 | 2.69 | 1 | 4.08e-16 | - | - | - |
| r4xyz | x32v | 34.5,1.5,1.5 | 0.125 | 16 | 2.89 | 1 | 5.53e-16 | - | - | - |
| r4xyz | d1e | 3.5,3.5,1.5 | 0.837 | -1 | 2.88 | 1.08 | - | - | - | - |
| r4xyz | d2e | 4.5,4.5,1.5 | 0.662 | -1 | 3.11 | 1.02 | - | - | - | - |
| r4xyz | d4e | 6.5,6.5,1.5 | 0.465 | 34 | 3.46 | 1 | 3.92e-16 | - | - | - |
| r4xyz | d8e | 10.5,10.5,1.5 | 0.290 | 24 | 4.04 | 1 | 3.40e-16 | - | - | - |
| r4xyz | d16e | 18.5,18.5,1.5 | 0.165 | 18 | 3.83 | 1 | 7.85e-16 | - | - | - |
| r8x | x1c | 5.5,0.0,0.0 | 0.858 | -1 | 0.529 | 1 | - | - | 2.0e-15,2.0e-15 | - |
| r8x | x2c | 6.5,0.0,0.0 | 0.726 | -1 | 0.891 | 1 | - | - | 4.4e-16,4.4e-16 | - |
| r8x | x3c | 7.5,0.0,0.0 | 0.629 | -1 | 1.19 | 1 | - | - | 5.7e-16,5.7e-16 | 1.3e-15,1.3e-15 |
| r8x | x4c | 8.5,0.0,0.0 | 0.555 | 54 | 1.44 | 1 | 1.64e-15 | 1.5e-15,1.5e-15 | 1.4e-16,3.3e-16 | 1.6e-15,1.6e-15 |
| r8x | x6c | 10.5,0.0,0.0 | 0.449 | 40 | 1.8 | 1 | 7.95e-16 | 5.5e-16,5.5e-16 | 2.5e-16,2.5e-16 | 9.4e-16,1.3e-15 |
| r8x | x8c | 12.5,0.0,0.0 | 0.377 | 34 | 2.05 | 1 | 6.85e-16 | 9.5e-16,9.5e-16 | 2.6e-16,3.6e-16 | 4.9e-16,6.8e-16 |
| r8x | x16c | 20.5,0.0,0.0 | 0.230 | 22 | 2.58 | 1.01 | 7.51e-16 | 6.8e-16,6.8e-16 | 1.3e-16,2.5e-16 | 1.1e-15,1.1e-15 |
| r8x | x32c | 36.5,0.0,0.0 | 0.129 | 18 | 2.75 | 1.01 | 1.39e-15 | 1.3e-15,1.3e-15 | 3.0e-16,3.0e-16 | 2.7e-16,2.7e-16 |
| r8x | d1c | 5.5,2.0,0.0 | 0.806 | -1 | 1.42 | 1.15 | - | - | - | - |
| r8x | d2c | 6.5,3.0,0.0 | 0.659 | -1 | 2.15 | 1.13 | - | - | - | - |
| r8x | d4c | 8.5,5.0,0.0 | 0.478 | 44 | 3.07 | 1.11 | 1.36e-15 | - | - | - |
| r8x | d8c | 12.5,9.0,0.0 | 0.306 | 28 | 3.97 | 1.06 | 7.11e-16 | - | - | - |
| r8x | d16c | 20.5,17.0,0.0 | 0.177 | 20 | 3.74 | 1 | 9.19e-16 | - | - | - |
| r8xyz | x1v | 5.5,3.5,3.5 | 1.053 | -1 | 3.43 | 1.8 | - | - | - | - |
| r8xyz | x2v | 6.5,3.5,3.5 | 0.954 | -1 | 3.54 | 1.51 | - | - | - | - |
| r8xyz | x3v | 7.5,3.5,3.5 | 0.867 | -1 | 3.45 | 1.31 | - | - | - | - |
| r8xyz | x4v | 8.5,3.5,3.5 | 0.792 | -1 | 3.32 | 1.19 | - | - | - | - |
| r8xyz | x6v | 10.5,3.5,3.5 | 0.671 | -1 | 3.12 | 1.06 | - | - | - | - |
| r8xyz | x8v | 12.5,3.5,3.5 | 0.580 | 50 | 3.01 | 1.01 | 1.60e-15 | - | - | - |
| r8xyz | x16v | 20.5,3.5,3.5 | 0.370 | 30 | 2.95 | 1.01 | 5.97e-16 | - | - | - |
| r8xyz | x32v | 36.5,3.5,3.5 | 0.212 | 20 | 2.77 | 1.02 | 7.15e-16 | - | - | - |
| r8xyz | d1e | 5.5,5.5,3.5 | 0.914 | -1 | 3.33 | 1.28 | - | - | - | - |
| r8xyz | d2e | 6.5,6.5,3.5 | 0.792 | -1 | 3.69 | 1.18 | - | - | - | - |
| r8xyz | d4e | 8.5,8.5,3.5 | 0.623 | 56 | 4.05 | 1.08 | 1.36e-15 | - | - | - |
| r8xyz | d8e | 12.5,12.5,3.5 | 0.433 | 34 | 4.39 | 1.03 | 8.34e-16 | - | - | - |
| r8xyz | d16e | 20.5,20.5,3.5 | 0.267 | 24 | 3.49 | 1.01 | 7.46e-16 | - | - | - |
| r16x | x1c | 9.5,0.0,0.0 | 0.907 | -1 | 0.255 | 1 | - | - | 2.0e-15,2.0e-15 | - |
| r16x | x2c | 10.5,0.0,0.0 | 0.821 | -1 | 0.507 | 1 | - | - | 4.7e-16,4.7e-16 | - |
| r16x | x3c | 11.5,0.0,0.0 | 0.749 | -1 | 0.766 | 1 | - | - | 4.9e-16,4.9e-16 | - |
| r16x | x4c | 12.5,0.0,0.0 | 0.689 | -1 | 1.01 | 1 | - | - | 2.5e-16,3.4e-16 | - |
| r16x | x6c | 14.5,0.0,0.0 | 0.594 | -1 | 1.42 | 1.01 | - | - | 2.6e-16,2.6e-16 | - |
| r16x | x8c | 16.5,0.0,0.0 | 0.522 | 52 | 1.74 | 1.01 | 7.29e-16 | 9.6e-16,9.6e-16 | 2.3e-16,4.1e-16 | - |
| r16x | x16c | 24.5,0.0,0.0 | 0.352 | 34 | 2.51 | 1.03 | 2.39e-15 | 2.4e-15,2.4e-15 | 1.6e-16,2.5e-16 | - |
| r16x | x32c | 40.5,0.0,0.0 | 0.213 | 24 | 2.58 | 1.04 | 1.17e-15 | 1.5e-15,1.5e-15 | 3.6e-16,3.6e-16 | - |
| r16x | d1c | 9.5,2.0,0.0 | 0.888 | -1 | 0.771 | 1.21 | - | - | - | - |
| r16x | d2c | 10.5,3.0,0.0 | 0.789 | -1 | 1.44 | 1.23 | - | - | - | - |
| r16x | d4c | 12.5,5.0,0.0 | 0.640 | -1 | 2.57 | 1.24 | - | - | - | - |
| r16x | d8c | 16.5,9.0,0.0 | 0.458 | 44 | 3.61 | 1.13 | 7.51e-16 | - | - | - |
| r16x | d16c | 24.5,17.0,0.0 | 0.289 | 28 | 3.34 | 1.02 | 4.51e-16 | - | - | - |
| r16xyz | x1v | 9.5,7.5,7.5 | 1.034 | -1 | 4.19 | 2.15 | - | - | - | - |
| r16xyz | x2v | 10.5,7.5,7.5 | 0.986 | -1 | 4.48 | 1.87 | - | - | - | - |
| r16xyz | x3v | 11.5,7.5,7.5 | 0.941 | -1 | 4.49 | 1.65 | - | - | - | - |
| r16xyz | x4v | 12.5,7.5,7.5 | 0.898 | -1 | 4.42 | 1.49 | - | - | - | - |
| r16xyz | x6v | 14.5,7.5,7.5 | 0.819 | -1 | 4.21 | 1.29 | - | - | - | - |
| r16xyz | x8v | 16.5,7.5,7.5 | 0.751 | -1 | 4.02 | 1.18 | - | - | - | - |
| r16xyz | x16v | 24.5,7.5,7.5 | 0.551 | 48 | 3.65 | 1.1 | 1.47e-15 | - | - | - |
| r16xyz | x32v | 40.5,7.5,7.5 | 0.352 | 30 | 2.66 | 1.07 | 7.26e-16 | - | - | - |
| r16xyz | d1e | 9.5,9.5,7.5 | 0.957 | -1 | 4 | 1.54 | - | - | - | - |
| r16xyz | d2e | 10.5,10.5,7.5 | 0.885 | -1 | 4.49 | 1.45 | - | - | - | - |
| r16xyz | d4e | 12.5,12.5,7.5 | 0.767 | -1 | 4.87 | 1.32 | - | - | - | - |
| r16xyz | d8e | 16.5,16.5,7.5 | 0.601 | 56 | 4.84 | 1.21 | 2.32e-15 | - | - | - |
| r16xyz | d16e | 24.5,24.5,7.5 | 0.415 | 34 | 3.11 | 1.04 | 1.78e-15 | - | - | - |

done load [3.28515625, 3.1728515625, 2.9560546875]

## 2e. Full n-cut vectors l = 0..56, old vs fixed (ncutfull.jl -> ncutfull.txt; f = 1, no tables needed)

Rows: OLD = what the S2/S2c runs used (max.() swallow, rNc incl. piece radii); FIXED whole-box radii = the vector farx.jl now
uses (cutMax, rNc = smallest whole-box radius); FIXED incl. piece radii = cutMax with the old rNc (rejected: N = 15-25 from
certifying the whole-box series at a radius where rho > 1); the two constituents at nMax 12 (-1 = not certifiable at 12).

### r8xy  rNc old (incl. pieces) = 1.66 g, whole-box rNc = 11.07 g, rHi = 3298.3 g
l:                        0   2   4   6   8  10  12  14  16  18  20  22  24  26  28  30  32  34  36  38  40  42  44  46  48  50  52  54  56
OLD (S2/S2c runs):        9   9   9   9   9   9   9  10  10  10  10  11  11  11  11  12  12  12  12   0   0   0   0   0   0   0   0   0   0
FIXED whole-box radii:    9   9   8   8   7   7   6   6   6   6   5   5   5   4   4   4   4   4   3   3   3   3   2   2   2   2   2   1   1
FIXED incl. piece radii:  9   9   9   9   9   9   9  10  10  10  10  11  11  11  11  12  12  12  12  13  13  13  14  14  14  14  15  15  15
at rHi alone (nMax 12):   9   9   8   7   6   5   3   2   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
at rNc old alone (nMax 12):   9   9   9   9   9   9   9  10  10  10  10  11  11  11  11  12  12  12  12  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1
### r8xyz  rNc old (incl. pieces) = 1.66 g, whole-box rNc = 12.5 g, rHi = 3990.6 g
l:                        0   2   4   6   8  10  12  14  16  18  20  22  24  26  28  30  32  34  36  38  40  42  44  46  48  50  52  54  56
OLD (S2/S2c runs):       10   9   9   9  10  10  10  10  11  11  11  11  12  12  12   0   0   0   0   0   0   0   0   0   0   0   0   0   0
FIXED whole-box radii:   10   9   9   8   8   7   7   6   6   6   6   5   5   5   5   4   4   4   4   3   3   3   3   3   2   2   2   2   2
FIXED incl. piece radii: 10   9   9   9  10  10  10  10  11  11  11  11  12  12  12  13  13  13  14  14  14  15  15  15  16  16  16  17  17
at rHi alone (nMax 12):  10   9   8   7   6   5   4   3   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
at rNc old alone (nMax 12):   9   9   9   9  10  10  10  10  11  11  11  11  12  12  12  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1
### r16xyz  rNc old (incl. pieces) = 1.66 g, whole-box rNc = 25.62 g, rHi = 7537.9 g
l:                        0   2   4   6   8  10  12  14  16  18  20  22  24  26  28  30  32  34  36  38  40  42  44  46  48  50  52  54  56
OLD (S2/S2c runs):       12  12  12  10   9   8   7   6   4   3   2   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
FIXED whole-box radii:   13  12  11  11  10  10   9   9   8   8   7   7   7   6   6   6   5   5   5   4   4   4   4   3   3   3   2   2   2
FIXED incl. piece radii: 13  12  12  13  13  14  14  15  15  16  16  17  17  18  18  19  19  20  20  21  21  22  22  23  23  24  24  25  25
at rHi alone (nMax 12):  -1  12  11  10   9   8   7   6   4   3   2   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
at rNc old alone (nMax 12):  12  12  12  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1
### r2x  rNc old (incl. pieces) = 1.66 g, whole-box rNc = 3.2 g, rHi = 1055.5 g
l:                        0   2   4   6   8  10  12  14  16  18  20  22  24  26  28  30  32  34  36  38  40  42  44  46  48  50  52  54  56
OLD (S2/S2c runs):        7   6   6   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5
FIXED whole-box radii:    7   6   5   5   5   5   4   4   4   4   4   3   3   3   3   3   3   2   2   2   2   2   2   1   1   1   1   1   1
FIXED incl. piece radii:  7   6   6   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5
at rHi alone (nMax 12):   7   6   5   4   3   2   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
at rNc old alone (nMax 12):   6   6   6   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5   5
### r4xyz  rNc old (incl. pieces) = 1.66 g, whole-box rNc = 6.84 g, rHi = 2217.0 g
l:                        0   2   4   6   8  10  12  14  16  18  20  22  24  26  28  30  32  34  36  38  40  42  44  46  48  50  52  54  56
OLD (S2/S2c runs):        8   7   7   7   7   7   7   7   8   8   8   8   8   8   8   9   9   9   9   9   9  10  10  10  10  10  10  11  11
FIXED whole-box radii:    8   7   7   6   6   6   5   5   5   5   4   4   4   4   3   3   3   3   3   3   2   2   2   2   2   1   1   1   1
FIXED incl. piece radii:  8   7   7   7   7   7   7   7   8   8   8   8   8   8   8   9   9   9   9   9   9  10  10  10  10  10  10  11  11
at rHi alone (nMax 12):   8   7   6   5   4   3   2   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
at rNc old alone (nMax 12):   8   7   7   7   7   7   7   7   8   8   8   8   8   8   8   9   9   9   9   9   9  10  10  10  10  10  10  11  11

Reading: at l >= 38 (r8xy), 30 (r8xyz), 22 (r16xyz) the OLD vector is 0 -- max(-1, 0) -- so those columns were contracted
with the n = 0 term alone where the fixed vector asks for 1-4 terms at the whole-box radius (11-27 g); at l = 0 r16xyz
lost the far-end 13.  That is the whole content of the S2c disagreements.  The fixed vectors are monotone decreasing in
l as the series c_ln ~ k^{2n}/(2l+2n+1)!! predicts; the OLD ones increased with l up to nMax and then dropped to 0.

## 5c. Bit-identity of the equal-cell paths, re-run against the current farfield.jl (bitid.jl -> bitid2.out, load 3.5-3.7)

farfield.jl (notes/farfield/farfield.jl, 1488 lines, unchanged this round) as module Old, farx.jl (2079 lines, with cutMax,
the whole-box-radius n-cut, farBlockX! and its helpers) as module New, one process:

| check | result |
|---|---|
| tabWhl(L = 10) / tabOct(L = 7) GeoBox fields, farMomW(80), farMomO(72), nCutVec(56) Float64 arrays, 4 shapes | 0 mismatches |
| farTensor at the 30 offsets x 4 shapes of S5, UInt64 bit patterns of re/im of all 9 entries + route kind | 120/120 identical; routes (i)/(ii)/(iii) 108/12/0 |
| Lw/Lo of the four shapes, old/new | 56/56 in both |

cutMax changes farSetup's result only when the -1 sentinel would have been swallowed, which at the four shipped shapes
(nCut <= 6, nMax 12) never happens; the equal-cell tables, moments, cuts and tensors are the same bits as before.

## 6. What farfield.jl must gain: farx.jl vs farfield.jl (definition-level diff; 110 definitions unchanged, 0 removed)

Added (name, lines, farx.jl line): wgtX 2 :37, wgtTrpX 1 :39, tabPce 39 :288, RRR 1 :370, xKey 3 :371, lamX 1 :668,
cutMax 1 :889, trpAB 2 :1546, typIdx/typOf/pceList/pceTyp 7 :1549-1556, FrqSetX 32 :1558, pceGeo 7 :1592, BndX 21 :1602,
bndX 21 :1624, bndLw 7 :1647, bndLp 7 :1656, farSetupX 64 :1665, boundLX 7 :1731, boundLpce 12 :1740, tnsWhlX! 10 :1754,
tnsPceX! 19 :1766, onLat 2 :1787, tnsGcd! 31 :1791, gcdCost 1 :1824, farRouteX 16 :1827, farWsX 2 :1845, farTensorX 8 :1849,
FACESX/boxFaceQ/facePairX 6 :1874-1879, MOMCX 1 :1882, tnsKsrX 39 :1884, xLat 6 :1925, xMult 1, boxSumX! 25 :1933,
xLevel 14 :1960, farBlockX! 105 :1975.  Total 520 added lines.
Changed (name, old -> new lines): tabWhl 45 -> 48 (W = wgtTrpX(a, b), vt = prod(b); tabWhl(s, ..) forwards with a = 0),
tabOct 37 -> 2 (forwards to tabPce with a = 0, typ RRR), ShpTab 8 -> 10 (a, typ), SHPC key (a, b, typ, nMax), shpFile
2 -> 5 (xKey suffix, equal-cell names unchanged), farShape 64 -> 68 (ta, tb, typ; whl/oct slots may be skipped),
farMomW 18 -> 19 (a, b measures; forwards), farMomO 13 -> 14 (al, bt, hf; forwards), nCutVec 22 -> 24 (vt, vs, rd
separated; forwards), farSetup 73 -> 75 (cutMax).  283 -> 266 lines.
Production subset (what a cross-scale fill needs): wgtX/wgtTrpX/lamX, tabWhl(ta, tb, ..), ShpTab/SHPC/shpFile/farShape with
(a, b), farMomW(a, b), nCutVec(vt, vs, rd), cutMax in farSetup, FrqSetX/BndX/bndX/bndLw/farSetupX(split = false, gcd = false)/
boundLX/tnsWhlX!, trpAB, xLat/xMult/boxSumX!/xLevel, farBlockX!, and tnsKsrX + facePairX for the route-3 fallback
(~330 lines).  The 27-split (tabPce, pieces of FrqSetX, bndLp/boundLpce/tnsPceX!) and the per-offset gcd average
(tnsGcd!, farRouteX, farTensorX) are measurement scaffolding: they converge nowhere in the realistic near band (S4)
or cost 160x the box sum, and the pieces carry no k-series remainder certificate (S2d).
Open items: (a) tnsKsrX has no disk cache and no reference check (no block offset reached route 3); (b) the a posteriori
escalation leaves 0.13% of the realistic block (rho_w 0.55-0.60) at L = 56 with a proven bound of 1-4 tol (S4b) -- the
remedy is LMAX > 56 or routing that band to the box sum (its fine box would grow from 33^3 to ~41^3); (c) a Theorem B
k-series remainder for the piece tables if the split is ever wanted; (d) est_X under-states max|T| by 2-4x on the r16x
pair at f = 1 and 0.37 (S3b/S3d), harmless for the certificate but worth a note in the document; (e) the cross-scale
table cache grew to 883 MB / 86 files (n13, n15, n17, n19, n25 variants from the runs before the n-cut fix; only n12
and n13 are needed).

## Where I stopped
All five items of the resume brief are done: S3b/S3c/S3d (272 + 64 + 64 references, worst route error 5.1e-15), S2d/S2e
(attribution: whole-box k-series under-truncation, gcd Lambda 1.1-2.2), S3e/S2f (est-ratio scan and the f = 1+0.1i / 0.37
driver), S4b (farBlockX!: 1,404,604 offsets in 6.39 s single-thread / 2.19 s on 12 threads, 13/13 references to
<= 3.5e-15; r = 4 xyz block 1692 offsets in 1.56 s, 9/9 references to <= 8.7e-16), S5c/S6.  Not done: nothing from the
brief; the open items above are outside it.  Two incidents: the first relaunch of attrib.jl/recheck2.jl at 00:03 was
killed at ~00:08 with the rest of the session (no error output), and my first fixed versions had two script bugs
(pad length, a piece type without a table) fixed at 07:39; every number above is from the runs after 07:39, load
2.8-3.9, except S2f (first incarnation, load 10.4-13.1).
