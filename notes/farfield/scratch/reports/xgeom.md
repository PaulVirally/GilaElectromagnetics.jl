# xgeom: what unequal-cell pairs Gila produces, and how it computes them today

Scripts (all in SCRATCH/xwork/xgeom/): enum.jl (exact-rational reproduction of genEgoCrc!/genEgoExt!/sepGrd/grdSel
and the egoFunExtCnt!/egoFunOut! criteria; output enum_out.md), instr.jl + override.jl (logged copies of egoFunOut!
and egoCntOut! evaluated inside GilaVacuum, then real GlaVacOprMem builds; output instr_out.md), offs.jl (offset list),
acc.jl (Gila tensors via egoFunOut! with genEgoCrcExt!'s arguments; output gila_out.txt), ref.jl (220-bit references
through xref/refx.jl, appended to xwork/xref/reftensors_x.txt), cmp.jl (errors; output cmp_out.md), acc2.jl (Gila at every
key of the shared cache; output cmp_all.md / cmp_all.txt), nest.jl (route (b) nested-refinement composite; output nest_out.md).
g = 1/32, f = 1, source = coarse cell at the origin, target = fine cell at R (R = target centre - source centre, the
value grdSel returns and egoFunOut! receives).

## 0. Where the unequal-cell path is actually reached (read from src, confirmed by instr.jl below)

- GlaCmpOprVac (glaCmpOpr.jl:223-237, `_cmpBlk`): a region pair with `trgReg.scl != srcReg.scl` whose bounding boxes
  touch (`_cntChk`, closed-box test, so face, edge AND corner contact) is built by `_sndBlk` (:212-221): the coarse
  region is remeshed at the fine scale (`GlaVol(cel .* rat, sclFin, org)`) and an EQUAL-cell external operator
  GlaOprVac(trgFin, srcFin) with the contact correction is wrapped in GlaSndOprVac.  genEgoCrcExt! with unequal
  cells is therefore NEVER called for the coarse-fine block of a face-sharing composite (test/cmpOprTest.jl:100
  asserts exactly this: `cmpBlkCnt(mnyOpr) == (2, 0, 0, 2)`).
- The unequal-cell genEgoCrcExt! runs for: (a) GlaOprVac{T}(trgVol, srcVol) / GlaVacOprMem on two GlaVols of
  different scale, touching or not (test/crsSclTest.jl, all six cross-scale testsets); (b) composite region pairs of
  different scale that do NOT touch: nested refinement (fine core inside a medium ring inside a coarse box: core vs
  box) and two-body operators GlaCmpOprVac(trgCvl, srcCvl).  In (b) the nearest pair is at least one intermediate
  region away, i.e. never the "one fine cell of face gap" case; that case exists only through route (a).
- Everything below enumerates what genEgoCrcExt! evaluates when it IS given the touching pair (route (a), the
  test-suite route), because that is the block whose near band the root asked to size; the composite's actual
  block for the same geometry is the equal-cell fine-mesh external (for fine 64^3 g vs coarse 4^3 of 16 g: a
  64^3-vs-64^3 equal-cell external at g, circulant 128^3, 2.1e6 offsets, all equal-cell).

## 1. Geometry: offsets of the coarse-fine block (enum.jl, exact rationals)

Construction (as crsSclTest.jl): coarse GlaVol (2,2,2) cells of r g on the axes of A (g elsewhere) centred at 0;
fine GlaVol of g with the same physical extent, centred at (2 r_x g, 0, 0), so the two share the full yz face.
GlaCmpVol([fine, coarse]) validates every geometry (tiling, parity, common lattice).  GlaExtInf: the FINE volume is
split into trgDiv = r^{|A|} interleaved partitions with grid pitch r g (per-partition cells (2,2,2)); the coarse
volume has one partition; each partition pair is a (2,2,2)x(2,2,2) block, circulant (4,4,4), 27 non-pad offsets.

Index arithmetic reproduced: sepGrd(trg,src,0) = trg.start-src.start : trg.step : trg.stop-src.start;
sepGrd(trg,src,1) = trg.start-src.stop : src.step : trg.start-src.start; grdSel(ind) = trgGrd[ind] for
ind <= indSpt, zero pad at indSpt+1, srcGrd[ind-indSpt-1] above (so the duplicate trg.start-src.start of the
source grid is skipped; per axis trgCel+srcCel-1 distinct offsets).  Per pair: contact iff the PARTITION-level
flag (cntChk: box separation < cntTol + (sT+sS)/2 on all axes; fitChk: integer scale ratio and lower corners an
integer number of gcd cells apart) AND |R_i| < cntTol + (sT_i+sS_i)/2 on all axes (egoFunExtCnt!:411);
otherwise sep = max_i round(|R_i| / max(sT_i,sS_i)) (egoFunOut!:443), sep <= 1 -> egoSrfAdp!, else egoSrfFxd!
with quadOrd(sep, 2 pi |f| max scale).

| geometry (coarse cell) | partitions (cnt flag true) | offsets | contact | adaptive | fixed ord 9 / ord 7 |
|---|---|---|---|---|---|
| r=2 x   (1/16,1/32,1/32) | 2 (2)     | 54     | 9   | 9     | 18 / 18 |
| r=2 xy  (1/16,1/16,1/32) | 4 (4)     | 108    | 12  | 24    | 36 / 36 |
| r=2 xyz (1/16)^3         | 8 (8)     | 216    | 16  | 56    | 72 / 72 |
| r=4 x   (1/8,1/32,1/32)  | 4 (3)     | 108    | 9   | 27    | 36 / 36 |
| r=4 xy  (1/8,1/8,1/32)   | 16 (12)   | 432    | 18  | 126   | 144 / 144 |
| r=4 xyz (1/8)^3          | 64 (48)   | 1728   | 36  | 540   | 576 / 576 |
| r=8 x   (1/4,1/32,1/32)  | 8 (5)     | 216    | 9   | 63    | 72 / 72 |
| r=8 xy  (1/4,1/4,1/32)   | 64 (40)   | 1728   | 30  | 546   | 576 / 576 |
| r=8 xyz (1/4)^3          | 512 (320) | 13824  | 100 | 4508  | 4608 / 4608 |
| r=16 x  (1/2,1/32,1/32)  | 16 (9)    | 432    | 9   | 135   | 144 / 144 |
| r=16 xy (1/2,1/2,1/32)   | 256 (144) | 6912   | 54  | 2250  | 2304 / 2304 |
| r=16 xyz (1/2)^3         | 4096 (2304) | 110592 | 324 | 36540 | 36864 / 36864 |

Closed forms that the table obeys: offsets = 27 r^{|A|}, all distinct; contact = prod over lateral axes of
(r_i + 2) (the fine centre must satisfy |R_i| <= (r_i+1)/2 g: r_i+2 half-integer positions) times 1 in x;
adaptive = offsets with all |R_i| < 1.5 max(sT_i, sS_i) minus contact.  Partitions whose fine cells are more than
r/2 fine cells from the face have the partition-level flag false and go through egoFunExt! (no contact test);
this changes nothing because none of their pairs touch.

Nearest non-contact offset: R = ((r+3)/2 g, 0, 0), |R| = 2.5, 3.5, 5.5, 9.5 g for r = 2, 4, 8, 16; in coarse cells
(r+3)/(2r) = 1.25, 0.875, 0.6875, 0.594, all rounding to sep = 1 -> adaptive.  On the x axis the fine cell is
adaptive for every face gap kap <= r-1 (R_x/(r g) = (r+1+2 kap)/(2r) < 1.5) and enters the fixed rule at kap = r
with sep = 2, ord 9.  For r = 16 the adaptive band is 15 fine cells = 0.94 coarse cells deep on axis and
(|R_y| < 24 g) wide.

Lattice: every offset has 2R/g integer and R/g NOT integer on the coarsened axes: R_x = (r+1+2 kap)/2 g is a
half-integer multiple of g for even r (the cell-centre lattices of the two grids are offset by g/2).  Exact values
of the 20 nearest non-contact offsets, r=2 x (R in lambda): (5/64, 0, 0), (5/64, +-1/32, 0), (5/64, 0, +-1/32),
(5/64, +-1/32, +-1/32) [these 9 adaptive, |R|/g = 2.5, 2.693, 2.872], then (7/64, 0, 0), (7/64, +-1/32, 0),
(7/64, 0, +-1/32), (7/64, +-1/32, +-1/32), (9/64, 0, 0), (9/64, 1/32, 0) [fixed, sep 2, ord 9].  r=16 xyz: the
nearest is (19/64, +-1/64, +-1/64) (R/g = (9.5, +-0.5, +-0.5), |R| = 9.526 g); the full lists are in enum_out.md.
Since g = 1/32 and r is a power of two every offset is a dyadic rational, so the Float64 values grdSel produces are
exact and no classification depends on rounding.

rho_whole = |b|/|R| with b = (sT+sS)/2 per axis, over the non-contact offsets (bins 0-.3-.45-.6-.8-1-1.5-3):
| geometry | adaptive rho range | fixed rho range | histogram |
|---|---|---|---|
| r=2 x    | 0.718-0.825 | 0.310-0.589 | 0, 26, 10, 8, 1, 0, 0 |
| r=2 xyz  | 0.600-1.000 | 0.351-0.728 | 0, 60, 60, 60, 16, 4, 0 |
| r=4 x    | 0.506-0.821 | 0.212-0.442 | 40, 32, 9, 17, 1, 0, 0 |
| r=4 xyz  | 0.455-1.213 | 0.278-0.662 | 68, 776, 436, 288, 92, 32, 0 |
| r=8 x    | 0.407-0.858 | 0.171-0.377 | 108, 54, 18, 18, 9, 0, 0 |
| r=8 xyz  | 0.391-1.406 | 0.244-0.623 | 2264, 5728, 3256, 1752, 528, 196, 0 |
| r=16 x   | 0.366-0.907 | 0.155-0.352 | 243, 90, 45, 27, 18, 0, 0 |
| r=16 xyz | 0.362-1.545 | 0.228-0.601 | 25572, 44240, 24520, 11236, 3244, 1440, 16 |
So the whole-box trapezoid expansion (needs rho < ~0.6) covers every FIXED-rule offset of the x-only shapes and
almost all of the xyz ones (max 0.60-0.73), but only part of the adaptive band: in r=16 xyz 15936 of 110268
non-contact offsets have rho >= 0.6 and 1456 have rho >= 1 (no convergence at all): those are the gcd-split /
k-series candidates.

Edge- and corner-adjacent pairs.  The farthest contact offsets are exactly R_i = +-(sT_i+sS_i)/2 on every axis:
r=2 x: R/g = (3/2, +-1, +-1) (corner contact, |R| = 2.062 g); r=16 xyz: (17/2, +-17/2, +-17/2) (|R| = 14.72 g).
|R_i| - (sT_i+sS_i)/2 = 0 exactly, and the test is `<` against cntTol + (sT+sS)/2 with cntTol = 1e-8 lambda =
3.2e-7 g, so equality is classified CONTACT and egoCntOut! averages gcd-cell self-tensors for them; the sep <= 1
test never sees an edge- or corner-touching pair.  The first non-contact lateral offset is one fine cell further
((r+1)/2 + 1 in g), sep = round((r+3)/(2r)) = 1 -> adaptive.  In the swapped orientation (coarse target) the same
offsets appear with R -> -R (instr.jl below).

## 3. Realistic blocks (item 3; what genEgoCrcExt! would evaluate through route (a))

| block | partitions | offsets (all distinct) | contact | adaptive (sep 1) | fixed ord 9 (sep 2) | ord 7 (sep 3-4) | ord 6 (sep 5+) | distinct abs-R patterns (non-contact) |
|---|---|---|---|---|---|---|---|---|
| fine 64^3 of g vs coarse 4^3 of 16 g = (1/2)^3, face | 4096 | 1404928 | 324 | 36540 | 167936 | 598016 | 602112 | 351151 |
| fine 64^3 of g vs coarse 8^3 of 8 g = (1/4)^3, face   | 512  | 1728000 | 100 | 4508  | 20992  | 140288 | 1562112 | 431975 |
Sep histogram (16 g): sep 1: 36540, 2: 167936, 3: 397312, 4: 200704, 5-7: 200704 each.  (8 g): sep 1: 4508,
2: 20992, 3: 49664, 4: 90624, 5: 143872, 6: 209408, 7: 287232, 8-15: 115200 each.
The adaptive band is the same 36540 (16 g) / 4508 (8 g) offsets as in the 2^3 geometry: it depends only on the
face and the ratio.  By face gap kap (fine cells), 16 g: kap 0 (lateral neighbours of the touching layer): 1980,
kap 1..15: 2304 each; 8 g: kap 0: 476, kap 1..7: 576 each.
quadOrd floor: kScl = 2 pi f max(scale) = 3.14 (16 g) and 1.57 (8 g), so ordFlr = 6 everywhere and the 16 g block
trips the "coarser than lambda/4, outside the tested quadrature range" warning (glaVacOprMemGen.jl:811): the fixed
rule's schedule was measured on cells <= lambda/4, and the coarse cell here is lambda/2.
rho_whole: 16 g block: 1217468 offsets below 0.3, 140504 in [0.3,0.45), 30696 in [0.45,0.6), 15936 at or above
0.6 (1456 at or above 1.0); 8 g block: 1699552 / 21264 / 4608 / 2476 (196 at or above 1.0).
Cost today (single thread, from acc.jl timings below): adaptive 36540 x ~5-40 ms = 3-25 min; fixed 1.37e6 x
(13 / 4.7 / 2.5 ms for ord 9 / 7 / 6) = 0.6 + 0.8 + 0.4 h = ~1.8 h single-thread for the 16 g block.

## 2a. Gila's per-offset cost today (acc.jl, JULIA_NUM_THREADS=1, load average 6.1-6.2 throughout)

egoFunOut! wall time (min of two calls after compile), the adaptive rule egoSrfAdp! timed separately:
| class | offsets | egoFunOut! time |
|---|---|---|
| adaptive, kap=1 axial, x shapes (r=2,4,16)     | R=(r+3)/2 g            | 16.3, 16.6, 17.1 ms (egoSrfAdp! 16.0, 19.5, 16.9 ms) |
| adaptive, kap=1 axial, xyz shapes (r=2,4,16)   |                        | 39.1, 54.0, 42.2 ms |
| adaptive, kap=2..8 axial, r=4 x / r=16 x       |                        | 3.3, 1.6 / 4.3, 2.3, 1.6, 1.05, 0.94 ms |
| adaptive, kap=2..8 axial, r=4 xyz / r=16 xyz   |                        | 10.4, 4.0 / 16.8, 10.0, 7.9, 5.8, 5.6 ms |
| adaptive, corner-corner kap 1 / 2, xyz          | r=2: 34 / -, r=4: 35 / 8.7, r=16: 32 / 13.6 ms | |
| adaptive, diagonal kap 1 / 2, xyz               | r=2: 7.9 / -, r=4: 10.6 / 2.9, r=16: 15.8 / 7.9 ms | |
| fixed ord 9 (sep 2) | every shape | 12.6-13.4 ms |
| fixed ord 7 (sep 3-4) | | 4.6-4.8 ms |
| fixed ord 6 (sep 5-7) | | 2.5-2.6 ms |
| fixed ord 5 (sep 8-16) | | 1.22-1.25 ms |
| fixed ord 4 (sep 17) | | 0.51 ms |
The adaptive rule is only expensive at the touching-layer offsets (kap = 1, 16-54 ms); from kap = 2 on it is
FASTER than the order-9 fixed rule (1-17 ms), which is the signature of hcubature stopping after one or two
Genz-Malik steps at cubRelTol = 1e-6 / cubAbsTol = 1e-9 (the same behaviour quadOrd's comment reports for
separated equal cells).  Its accuracy is in 2b.

## 1b. Validation of the enumeration against Gila's real fill (instr.jl, JULIA_NUM_THREADS=1)

egoFunOut! and egoCntOut! were redefined inside GilaVacuum (override.jl: verbatim copies plus a log push) and
GlaVacOprMem(CPUKerOpt{Float64}(), trg, src) built for five pairs; the logged (R, class, sep, ord) set equals
enum.jl's in every case:
| fill | build time (load) | logged = enum offsets | missing / extra / class disagreements | contact / adaptive / fixed |
|---|---|---|---|---|
| r=2 x, fine target (4,2,2) vs coarse (2,2,2)   | 34.4 s incl. compile (5.98) | 54 = 54     | 0 / 0 / 0 | 9 / 9 / 36 |
| r=2 x swapped, coarse target, fine source       | 0.74 s (5.98)               | 54 = 54     | 0 / 0 / 0 | 9 / 9 / 36 |
| r=4 xyz, fine (8,8,8) vs coarse (2,2,2)         | 61.7 s (8.92)               | 1728 = 1728 | 0 / 0 / 0 | 36 / 540 / 1152 |
| r=8 x, fine (16,2,2) vs coarse (2,2,2)          | 3.05 s (8.92)               | 216 = 216   | 0 / 0 / 0 | 9 / 63 / 144 |
| r=16 x, fine (32,2,2) vs coarse (2,2,2)         | 6.57 s (8.76)               | 432 = 432   | 0 / 0 / 0 | 9 / 135 / 288 |
(the swapped orientation logs R -> -R of the same set, so the enumeration for the coarse-target block is the
mirror image and the class counts are identical.)
In-fill per-offset times (single thread): egoFunOut! adaptive median 1.1-12 ms (max 67-75 ms at the kap = 1
offsets), fixed median 12.8-14.3 ms (ord 9 dominates the median), egoCntOut! 15-63 us (it only sums stored
gcd-cell self tensors; the cost sits in genEgoSlf! on the (r+1)-cell contact volume, run once per fill).
The r=4 xyz fill: 1728 offsets, 3.5 s adaptive + 12.6 s fixed + the contact-volume self build.

Composite dispatch confirmed: GlaCmpOprVac{Float64}(GlaCmpVol([fine, coarse])) for r=2 on x gives block types
[GlaOprVac GlaSndOprVac; GlaSndOprVac GlaOprVac]; the (fine, coarse) block's inner operator has trgVol.cel = (4,2,2)
scl = g and srcVol.cel = (4,2,2) scl = g (the coarse (2,2,2) of (1/16,1/32,1/32) remeshed), and the build made
126 = 2 x 63 = 2 x (4+4-1)(2+2-1)(2+2-1) egoFunOut!/egoCntOut! calls, i.e. exactly the two equal-cell inner
blocks and zero unequal-cell evaluations.

## 2b. Gila's tensors today against the 220-bit references (cmp.jl -> cmp_out.md; acc2.jl -> cmp_all.md, cmp_all.txt)

References: all 91 acc.jl offsets were in xwork/xref/reftensors_x.txt (nothing had to be added; 47 of them were
computed by ref.jl, rule B ord 32 grd 2, logged in ref_log.txt with tB32 = 5-182 s at load 6.5-12.4).  Rule checks
in ref_log.txt: |A44 - A64| 1.6e-64 (r=2 x, 446 s); |A44 - B32| 1.9e-58 (r=2 x) and 1.6e-58 (r=4 x); |B32 - B40|
2.9e-61 (r=2 xyz), 2.4e-59 (r=4 xyz), 6.8e-53 (r=16 xyz, 660 s).  acc2.jl then evaluated egoFunOut! (the
genEgoCrcExt! arguments, CPUKerOpt(f, 48, false)) at EVERY key of the shared cache (356 distinct keys: 276 at f = 1,
40 at 1+0.1i, 40 at 0.37; coarse shapes r = 2, 4, 8, 16 on x, xy, xyz) so the r = 8 shapes (lambda/4 cells) that
acc.jl did not carry are covered too; timings in cmp_all.md at load 2.5 (acc.jl's at 6.1-6.2).

Error metric: max-entry-normalised |G - Gref|_max / |Gref|_max and worst per-entry |G_ij - Gref_ij| / |Gref_ij| over
entries with |Gref_ij| > 1e-30 |Gref|_max (the entries that vanish by symmetry are 1e-64..1e-70 in the reference and
1e-17 in Gila: pure Float64 noise, excluded).

Per class, f = 1, the 91 acc.jl offsets (shapes r = 2, 4, 16 on x and xyz):
| class | n | max-entry-norm err min / median / max | worst per-entry max | where the max sits |
|---|---|---|---|---|
| adaptive (sep 1) | 31 | 3.6e-9 / 9.1e-7 / 1.30e-5 | 1.94e-5 | cc2 r=4 xyz, R/g = (4.5, 1.5, 1.5) |
| fixed ord 9 (sep 2) | 18 | 2.3e-14 / 3.2e-13 / 1.77e-4 | 2.28e-4 | dg1 r=16 x, R/g = (9.5, 2, 0), sS/g = (16, 1, 1) |
| fixed ord 7 (sep 3-4) | 23 | 3.9e-14 / 1.1e-13 / 1.48e-4 | 1.71e-4 | dg2 r=16 x, R/g = (10.5, 3, 0) |
| fixed ord 6 (sep 5-8) | 9 | 4.7e-14 / 2.8e-13 / 2.24e-5 | 2.30e-5 | dg4 r=16 x, R/g = (12.5, 5, 0) |
| fixed ord 5 (sep 8-16) | 8 | 2.0e-14 / 5.5e-14 / 2.0e-13 | 4.2e-13 | |
| fixed ord 4 (sep 17) | 2 | 5.8e-14 / - / 1.0e-13 | 3.2e-13 | |

Per shape, f = 1 (cmp_all.md; n = offsets in the cache; "rod" = r on x only, "slab" = r on x, y, "cube" = r on x, y, z):
| sS/g | adaptive: n, min / median / max | fixed ord 9: n, min / max | ord 7: max | ord 6: max | ord 5 / 4: max |
|---|---|---|---|---|---|
| (2,1,1) rod   | 1: 4.2e-6                    | 4: 5.6e-14 / 8.8e-13 | 1.7e-12 | 2.8e-13 | 2.0e-13 / 7.5e-14 |
| (2,2,1) slab  | 3: 6.9e-7 / 1.5e-6 / 7.6e-6  | 5: 5.4e-14 / 5.8e-13 | 2.7e-13 | 1.2e-13 | 1.3e-13 / 6.2e-14 |
| (2,2,2) cube  | 5: 9.9e-8 / 2.6e-7 / 4.8e-6  | 8: 2.4e-14 / 8.3e-13 | 4.0e-13 | 1.1e-13 | 1.2e-13 / 1.0e-13 |
| (4,1,1) rod   | 3: 8.5e-7 / 3.1e-6 / 3.5e-6  | 3: 1.3e-13 / 2.6e-9  | 3.4e-9  | 3.4e-10 | 3.4e-11 / - |
| (4,4,1) slab  | 8: 1.3e-6 / 2.0e-6 / 3.3e-5  | 3: 1.1e-13 / 2.9e-12 | 4.5e-13 | 1.6e-13 | 7.1e-14 / - |
| (4,4,4) cube  | 14: 3.0e-7 / 1.8e-6 / 1.3e-5 | 6: 3.2e-14 / 4.1e-12 | 6.3e-13 | 5.1e-14 | 2.2e-14 / - |
| (8,1,1) rod   | 4: 1.1e-7 / 8.3e-7 / 3.3e-6  | 2: 1.0e-12 / 1.42e-6 | 1.38e-6 | 1.5e-7  | - |
| (8,8,1) slab  | 11: 2.7e-7 / 3.2e-6 / 2.28e-4 | 3: 4.2e-13 / 6.4e-12 | 2.5e-13 | 1.1e-13 | - |
| (8,8,8) cube  | 18: 9.7e-8 / 1.1e-6 / 2.3e-5 | 5: 3.7e-14 / 8.1e-12 | 4.0e-13 | 3.5e-14 | - |
| (16,1,1) rod  | 6: 1.1e-7 / 8.5e-7 / 3.2e-6  | 3: 2.2e-13 / 1.77e-4 | 1.48e-4 | 2.24e-5 (ax120: 1.7e-11) | - |
| (16,16,1) slab | 14: 7.9e-9 / 2.5e-6 / 1.49e-4 | 3: 2.8e-13 / 4.5e-12 | 3.5e-13 | - | - |
| (16,16,16) cube | 24: 2.7e-9 / 3.5e-7 / 7.6e-6 | 6: 1.5e-14 / 5.6e-12 | 5.9e-13 | 1.7e-11 (ax120) | - |
Other frequencies (1+0.1i, 0.37; rods and cubes r = 2..16, 80 offsets): adaptive 2.0e-8..6.3e-6, fixed 3.2e-14..1.5e-11
(the 1.5e-11 is (8,8,8) ord 9 at f = 0.37), no new failure mode.

What the numbers say.
- The adaptive rule (egoSrfAdp!, hcubature at cubRelTol 1e-6 / cubAbsTol 1e-9) is 1e-7..1e-5 wherever it is used,
  median 1e-6, i.e. it delivers its rtol and nothing more; its best cases (3e-9) are the deep-band axial offsets of
  the 16-cube where hcubature happened to converge.  106 of the 111 adaptive-class offsets at f = 1 are above the
  1e-8 bar (the 5 below, 2.7e-9..7.9e-9, are kap = 8 offsets of the 16-cube and the 16-slab); 13 are above 1e-5
  and 2 above 1e-4 (slabs (8,8,1) at R/g = (8.5, 0, 0): 2.28e-4 and (16,16,1) at (12.5, 0, 0): 1.49e-4).  Of the
  165 fixed-class offsets at f = 1, 7 are above 1e-8, all of them side-displaced rods of r = 8 and 16 (next item).
- The fixed rule on cubes and slabs is 1.5e-14..8e-12 at every order, on every shape from lambda/16 to lambda/2
  cells: the order-9 rule at sep 2 is 2.4e-14..8.3e-13 on the 2-cube, 3.2e-14..4.1e-12 on the 4-cube, 3.7e-14..
  8.1e-12 on the 8-cube (lambda/4) and 1.5e-14..5.6e-12 on the 16-cube (lambda/2).  The worst per-entry values in
  this family (2e-12..2.8e-11) are small entries (|Gref| 1e-3 of the max) of the ord 5/4 rules at sep 16-17.
- The fixed rule FAILS on rods when the fine cell is displaced sideways: the classification sep =
  max_i round(|R_i| / max(sT_i, sS_i)) reaches 2 through the thin axis (|R_y| = 2 g against max scale g) while the
  fine cell sits 1.5 g from the rod's 16 g long side face, and 9 Gauss-Legendre points along a lambda/2 panel whose
  singularity is 1.5 g away give 1.77e-4 (r=16), 1.42e-6 (r=8), 2.6e-9 (r=4), 8.8e-13 (r=2).  Along the axis the
  same rods are fine (2e-13..1e-12).  The equal-cell quadOrd schedule was measured for cubes; the rod aspect ratio
  is the untested dimension, not kScl.
- At 4 lambda (ax120, R/g = 128.5, sep 8, ord 6) both r=16 shapes are at 1.7e-11 max-entry (2.1e-10 per entry):
  order 6 across a lambda/2 panel (pi radians of phase) is the kScl > 1.6 regime quadOrd warns about; on the
  lambda/4 cells the same order at sep 8-16 is 1e-13.

Sign convention (cmp_out.md, last block).  Flipping R_y (dg1 r=2 x, R/g = (2.5, 2, 0) -> (2.5, -2, 0), fixed rule):
G12 ratio flipped/original = -1.000000 in Gila and -1.000000 in the reference; G11, G22, G33 ratio +1 in both; G13,
G23 vanish in both (reference 5e-68..2e-67, Gila 1e-17 noise, ratio meaningless).  Flipping R_z (cc1 r=4 xyz,
(3.5, 1.5, 1.5) -> (3.5, 1.5, -1.5), adaptive rule): G13 and G23 ratio -1.0 in both, G12 and the diagonal +1 in
both.  Gila-flipped against reference-flipped: 8.1e-13..8.8e-13 (fixed) and 1.5e-8..2.8e-6 (adaptive), the same
as the unflipped errors.  So Gila and refB agree that G_ab(R) with R = target centre - source centre changes sign
with R_a R_b for a != b, and both use the same R (the fine target at R, the coarse source at the origin, kernel
evaluated at R + d); no orientation or transposition fix is needed on either side.

## 4. Measured accuracy of Gila's two separated rules on unequal cells, f = 1 (acc2.jl, cmp_all.md)

egoSrfAdp! runs at cubRelTol = 1e-6 / cubAbsTol = 1e-9 (glaVacOprMemGen.jl:6-8, :777).  kap = face gap in fine
cells (max over axes of (|R_i| - (sT_i + sS_i)/2) / g); axial = R lateral (0, 0); corner = fine cell over a corner
sub-cell of the coarse face; diagonal = fine cell past the coarse edge.  Entries: max-entry-normalised error
(worst per-entry in brackets where it differs by more than 2x).
| rule | offsets | (2,2,2) lambda/16 | (4,4,4) lambda/8 | (8,8,8) lambda/4 | (16,16,16) lambda/2 | rods (r,1,1) | slabs (r,r,1) |
|---|---|---|---|---|---|---|---|
| adaptive, kap 1 axial     | (r+3)/2 g on x            | 9.9e-8 (2.5e-7) | 5.2e-6 | 9.9e-7 | 4.4e-7 | 4.2e-6, 3.5e-6, 3.3e-6, 3.2e-6 (r = 2, 4, 8, 16) | 6.9e-7, 2.8e-6 (8.7e-6), 1.3e-5 (6.3e-5), 1.1e-5 (3.2e-5) (r = 2, 4, 8, 16) |
| adaptive, kap 1 corner / corner-corner | lateral (r-1)/2 g | 4.8e-6 | 1.8e-6 | 2.5e-6 / 1.1e-6 | 2.4e-6 / 4.6e-7 | - | 7.6e-6 (2.2e-5), 3.7e-6 (1.8e-5), 2.7e-6 (1.1e-5), 2.5e-6 |
| adaptive, kap 1 diagonal  | past the edge             | 2.6e-7 | 2.3e-6 (4.5e-6) | 7.6e-6 / 7.8e-6 | 2.8e-6 / 1.9e-6 | fixed rule (below) | 1.5e-6 (3.0e-6), 1.6e-6, 6.3e-5 (9.6e-5), 2.6e-5 (5.9e-5) |
| adaptive, kap 2 axial     |                           | - | 3.0e-7 | 4.8e-7 | 2.8e-8 | 8.5e-7, 8.3e-7, 8.5e-7 (r = 4, 8, 16) | 2.0e-6 (6.6e-6), 3.0e-6 (9.4e-6), 1.2e-6 (r = 4, 8, 16) |
| adaptive, kap 2 corner / diagonal |                   | - | 1.3e-5 (1.9e-5) / 3.3e-6 | 1.7e-6..6.4e-6 / 9.7e-8..8.8e-7 | 2.3e-7..2.5e-6 / 1.9e-6..3.4e-6 | - | 3.3e-5 (1.3e-4), 2.4e-5 (6.4e-5), 1.8e-5 (3.2e-5) / 1.3e-6, 2.7e-7, 3.2e-6 |
| adaptive, kap 3           |                           | - | 6.7e-7 | 7.3e-7..4.1e-6 | 6.4e-8..6.5e-7 | 3.1e-6, 2.8e-6, 2.7e-6 | 1.2e-6..3.2e-6 (6.1e-6) |
| adaptive, kap 4           |                           | - | - | 7.3e-7..2.3e-5 | 2.8e-7..7.6e-6 | 1.1e-7 (r = 8, 16) | 2.28e-4 (4.3e-4) at (8.5,0,0); 1.49e-4 (2.1e-4) at (12.5,0,0); 1.8e-6..4.6e-5 |
| adaptive, kap 6 / 8 (r = 16 only) |                  | - | - | - | 3.5e-8 / 2.7e-9..3.4e-8 | 1.3e-7 / 9.1e-7 | - / 7.9e-9..1.8e-7 |
| fixed ord 9, sep 2 axial  | first fixed offset (kap = r) | 8.3e-13 (2.4e-12) | 4.1e-12 (9.4e-12) | 8.1e-12 | 5.6e-12 | 5.6e-14, 3.2e-13, 1.0e-12, 4.8e-13 (r = 2, 4, 8, 16) | 5.1e-13 (1.4e-12), 2.9e-12 (6.9e-12), 6.4e-12, 4.5e-12 |
| fixed ord 9, sep 2 corner / diagonal |                | 4.1e-13 (2.2e-12) / 5.5e-14 | - | 1.6e-12..3.8e-12 / 3.7e-14 | 5.1e-13..2.1e-12 / 1.5e-14 | side-displaced: 8.8e-13, 2.6e-9, 1.42e-6, 1.77e-4 (r = 2, 4, 8, 16) | 5.8e-13..1.7e-12 (3.9e-12) / 1.1e-13..4.2e-13 |
| fixed ord 9, sep 2 second layer (kap = r + 1 ... 2r - 1) | | 6.2e-14 | 3.2e-14 | - | 2.3e-14 | 1.0e-13, 1.3e-13, 2.2e-13 | 6.3e-14 (r = 2) |
| fixed ord 7 / 6 / 5 / 4 (sep 3-4 / 5-8 / 9-16 / 17), all offsets | | <= 4.0e-13 / 1.1e-13 / 1.2e-13 / 1.0e-13 | <= 6.3e-13 / 5.1e-14 / 2.2e-14 | <= 4.0e-13 / 3.5e-14 | <= 5.9e-13 / 1.7e-11 (4 lambda) | rods r = 8, 16 side-displaced: 1.4e-6 / 1.5e-7, 1.5e-4 / 2.2e-5; else <= 1.7e-12 | <= 4.5e-13 |
Reading: (i) the adaptive rule is 1e-7..1e-5 at every kap, not just at kap 1; the median over the 111 adaptive-class
offsets is 1.2e-6 and it stops at one or two Genz-Malik steps from kap 2 on (times 1-17 ms, section 2a), so the
touching layer is not the accuracy problem, the tolerance is.  (ii) The order-9 fixed rule at sep 2 is 1e-12..8e-12
on the lambda/4 and lambda/2 cubes and slabs, the same as on the lambda/16 and lambda/8 cubes: the "outside the
tested range" warning is not accompanied by a loss at sep 2.  The loss the warning anticipates appears at the far
end (order 6 over a lambda/2 panel at 4 lambda: 1.7e-11) and, much larger, on rods displaced sideways (1.8e-4),
where no order in the schedule would help (the panel is 16 g long and the singularity is 1.5 g from it).

## 5. What Gila needs from the cross-scale library (summary for the root)

Block types and where each class of pair is evaluated (sections 0, 1, 3 above; nest.jl -> nest_out.md for (b)):
1. Composite (GlaCmpOprVac) region pair of different scale whose boxes TOUCH (face, edge or corner): _sndBlk
   remeshes the coarse region at the fine scale and builds an EQUAL-cell external with the contact correction;
   the unequal-cell code is never called.  Needs from the library: nothing cross-scale; the equal-cell fine
   offsets of farfield.jl (for a 64^3 fine region against a 4^3 coarse neighbour of 16 g: a 64^3 x 64^3 equal-cell
   block at g, 2.1e6 offsets).
2. Route (a): GlaOprVac / GlaVacOprMem on two GlaVols of different scale (the test-suite route; touching or not).
   Per cell pair: contact if |R_i| < cntTol + (sT_i + sS_i)/2 on all axes (egoCntOut!, gcd-cell self-tensor
   averaging, unchanged); else adaptive if sep = max_i round(|R_i| / max(sT_i, sS_i)) <= 1; else fixed with
   quadOrd(sep, kScl).  Classes as (R, sT, sS): sT = g, sS = r g on the coarsened axes, R = ((r+1)/2 g + kap g,
   lat_y, lat_z) with lat on the half-lattice of the coarse cell (all 27 r^{|A|} - contact combinations of the
   circulant), plus the mirror block (coarse target) with R -> -R and the factor V_s / V_t.  Counts for the
   touching 2^3-coarse geometries are the table of section 1 (offsets 27 r^{|A|}, contact prod(r_i + 2) over
   lateral axes); for the two realistic face-sharing blocks (section 3):
   | block | contact (unchanged) | adaptive (sep 1) | fixed ord 9 | ord 7 | ord 6 | ord 5 | rho_whole >= 0.6 / >= 1 |
   |---|---|---|---|---|---|---|---|
   | fine 64^3 of g vs coarse 4^3 of 16 g (lambda/2), face | 324 | 36540 | 167936 | 598016 | 602112 | 0 | 15936 / 1456 |
   | fine 64^3 of g vs coarse 8^3 of 8 g (lambda/4), face   | 100 | 4508  | 20992  | 140288 | 1562112 | 0 | 2476 / 196 |
   The library must therefore supply, for route (a): every non-contact (R, g, r g) of the block.  Whole-box
   trapezoid expansion (rho = |b|/|R| < ~0.6) covers all fixed-class offsets of the rod shapes (rho <= 0.59) and all
   but 15936 (1.1 %) of the 1.40e6 offsets of the lambda/2 block; the adaptive band (36540 offsets, rho 0.36-1.55,
   kap 0..15 fine cells) needs the gcd average (r^{|A|} = 4096 equal-cell fine tensors per offset at r = 16, 512 at
   r = 8) or the k-series where nothing converges (rho >= 1: 1456 offsets at r = 16, 196 at r = 8).  Today these
   36540 offsets cost 3-25 min single-thread at 1e-7..1e-4 accuracy (sections 2a, 4); the fixed 1.37e6 cost ~1.8 h
   at 1e-12..1e-11 (cubes) but 1e-4 on side-displaced rods.
3. Route (b): composite region pairs of different scale that do NOT touch (nested refinement core vs outer
   shell; two-body operators).  nest.jl built the composite fine core 32^3 of g (1 lambda) + ring + outer shell of
   coarse cells r g for r = 2, 4, 8 and enumerated the core-vs-shell blocks with enum.jl's exact reproduction of
   genEgoCrcExt!.  A ring ONE coarse cell thick is impossible in a legal GlaCmpVol: chkCmpVol demands even cell
   counts per region and chkParCmpVol demands an even per-partition cell sum for every region pair, so a
   one-coarse-cell ring of any finer scale (two layers of (r/2) g, four of (r/4) g, ...) gives per-partition
   count 1 against the shell's even count, an odd sum, and is rejected ("(1, 18, 18) cells per partition in
   region 2 plus (2, 22, 22) in region 8 gives (3, 40, 40)", nest_out.md, all three r).  The thinnest legal gap
   between a fine region and a non-touching coarse region is TWO coarse cells (ring of coarse cells, two thick,
   or four layers of (r/2) g).  With that ring, core vs shell (6 slabs, 13 regions validated by GlaCmpVol):
   | r (coarse cell) | offsets (all distinct, all non-contact) | adaptive | fixed by order | nearest R/g, abs R | sep, ord at nearest | rho_whole range | rho >= 0.6 |
   |---|---|---|---|---|---|---|---|
   | 2 (1/16)  | 1118192 | 0 | ord 4: 386384, 5: 699360, 6: 24672, 7: 7776 | (5.5, +-0.5, +-0.5), 5.545 g | 3, 7 | 0.039-0.4685 | 0 |
   | 4 (1/8)   | 1528704 | 0 | ord 5: 1269120, 6: 197376, 7: 62208 | (10.5, +-0.5, +-0.5), 10.52 g | 3, 7 | 0.055-0.4115 | 0 |
   | 8 (1/4)   | 2616320 | 0 | ord 6: 2118656, 7: 497664 | (20.5, +-0.5, +-0.5), 20.51 g | 3, 7 | 0.076-0.3800 | 0 |
   Closed forms: nearest |R_x| = (5r+1)/2 g, sep = round((5r+1)/(2r)) = 3 for every r >= 2, max rho = sqrt(3)(r+1)/2
   / sqrt(((5r+1)/2)^2 + 1/2) = 0.4685, 0.4115, 0.3800 (r = 2, 4, 8; 0.363 at r = 16).  Route (b) therefore never
   produces an adaptive-class or a rho >= 0.6 offset: the whole-box trapezoid expansion alone serves every
   composite non-touching cross-scale block, at 1-2.6e6 offsets per 1-lambda core.  (Had the one-coarse-cell ring
   been legal, the nearest offset would have been (3r+1)/2 g, sep 2 -> ord 9, rho 0.728, 0.662, 0.623 for r = 2,
   4, 8: a thin band above 0.6 that the gcd average would have had to take.)
4. Lattice.  Every non-contact R has 2R/g integer (routes (a) and (b), all shapes, nest_out.md and enum_out.md).
   R_i/g is a half-integer on every coarsened axis whenever r_i is even: R_i = (r_i + 1)/2 g + kap g, and the
   lateral positions (r_i - 1)/2 g - j g are half-integers too, so on a cubic coarse cell of even r EVERY component
   of R is a half-integer multiple of g (nest_out.md: "all R/g half-integer on every axis true" for all three r).
   Only odd r (legal for GlaExtInf, never used in Gila's power-of-two refinements) puts R on the integer g lattice.
   Consequences: a fine equal-cell egoToe indexed by integer offsets D cannot be indexed by R/g for any unequal pair
   of even ratio, and a whole-box unequal-cell table would have to be keyed by 2R/g (the half-lattice), which is
   what the block circulant index already does (grdSel positions, section 1), so Gila's storage is unaffected.  The
   gcd average is unaffected as well and is exactly where the half-integers cancel: with the coarse source split
   into sub-cells at c'_j = (-(r-1)/2 + j) g, j = 0..r-1 (half-integers for even r), G(R; g, r g) = sum_j G(R - c'_j;
   g, g) and R_x - c'_j = (r + kap - j) g runs over the INTEGER fine offsets kap + 1 .. kap + r; laterally
   (r-1)/2 - j' + lat/g is an integer for the same reason.  So the gcd average of an even-ratio pair is a sum of
   r^{|A|} entries of the fine equal-cell egoToe at integer D (the nearest being D_x = kap + 1 = 2 for the kap = 1
   layer, the one-fine-cell-gap offset that farfield.jl routes to the k-series), with weights +1 (fine target) or
   +1/N_t (coarse target, N_t = r^{|A|} target sub-cells), and no half-integer index ever appears in it.

Stopped here: items 1-3 of the resume brief are done (2b, 4, 5 written; scripts cmp.jl, acc2.jl, nest.jl in
xwork/xgeom with sentinels cmp.done, acc2.done, nest.done; load averages 2.5-6.0 during this session's runs).  Not
done: the slender-pair shapes (1/32,1/32,1/512) of the brief (no Gila evaluation requested for them here), and
no reference was added to the cache by this incarnation (91 of 91 present).
