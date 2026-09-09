# xnear: the near field of unequal cells -- the thin-indicator branch on cross-scale geometries

Working dir: `notes/farfield/scratch/xwork/xnear/` (all paths below relative to it unless absolute).
Copy of the library under test: `moments_x.jl` (byte-identical copy of `notes/moments/moments.jl`; not modified).
Machine load: stated with every timing (12-core M3 Pro shared with other agents; load average 4-7 throughout).

## 0. What the thin branch is, and the one-line finding

`box3` (moments.jl:998) routes every 3D box first through `slvAxs`: an axis with `2 (hi - lo) < hi` (strict), i.e.
`lo > hi/2`, is "thin" and is integrated by a Gauss-Legendre rule (`box3Slv`, order `slvOrd` = 11-14 in Float64) against
the 2D closed forms (`psiBox`/`wgt2`) at the node heights, or, with two thin axes, a tensor rule against the 1D
closed form `seg1`. Off the thin gate the box goes to `domAxs` (binomial series), `thnAxs`, or the divergence-identity
closed forms (`kBox3`/`wmBox3`/`wpBox3`).

Finding (part 1): the gate is not an off-lattice exotic. It fires on every SEPARATED pair, equal cells included
(`hi = 2 lo` is the ONLY separated lattice shape that escapes it, i.e. a face gap of exactly one cell), so the k-series
route (iii) of farfield.jl already runs through `box3Slv` for 99% of its perpendicular boxes at |D| >= 3 in units of
the cell; on cross-scale TOUCHING pairs it fires as well once a ratio is >= 2 (never for equal touching cells, which is
the "never reached on the lattice" statement of moments.tex sec. "The off-lattice thin-indicator branch" -- that
statement is about the fifteen touching sub-geometries only).

## 1. Census of the thin branch over the gcd-lattice geometries (script `geom.jl`, 50 s at load 4.2)

Geometry: fine target g = (1/32)^3 at centre R, coarse source r g on the axes of the set A at the origin,
r in {2, 4, 8, 16}, A in {x}, {x,y}, {x,y,z} (12 shapes) plus 1x1x1 (equal cells) as the control. Fine-centre
positions per axis, in units of g, on the gcd lattice (fine centres at half-integers on a coarse axis of even r,
integers on an equal axis), R_i >= 0 by reflection symmetry:
  L (lateral, inside the coarse shadow): 1/2, 3/2, ..., (r-1)/2  (r even) or 0 (r = 1);
  T (touching, kap = 0): (r+1)/2;   S (separated): (r+1)/2 + kap, kap in {1, 2, 3, 4, 6, 8, 16, 32}.
All positions with at least one T or S axis (overlapping fine/coarse cells are not produced by Gila), all 36 face pairs,
`pairBxs` exact (Rational breakpoints), each 3D box classified by the branch `box3` takes. Position type = the multiset
of axis types, e.g. TLL = face contact, TTL = edge contact, TTT = corner contact, SLL = face-separated inside the
shadow, SSS = separated on all three axes. Full table (234 classes): `out/census.txt`; every face pair with a thin box:
`out/thinpairs.txt` (one line each, 7 columns + min ratio).

Condensed per shape (perpendicular face pairs only; parallel pairs decompose into 2D boxes and never reach the branch):

shape | contact | perp face pairs | 3D boxes | slv1 | slv2 | thin frac | dom | closed | fp with >=1 thin | fp frac | min (hi-lo)/hi
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
1x1x1 | touch | 168 | 336 | 0 | 0 | 0.000 | 0 | 336 | 0 | 0.000 | -
1x1x1 | sep | 23808 | 47616 | 6750 | 40500 | 0.992 | 0 | 366 | 23700 | 0.995 | 0.0294
2x1x1 | touch | 168 | 416 | 64 | 0 | 0.154 | 0 | 352 | 48 | 0.286 | 0.333
2x1x1 | sep | 23808 | 56320 | 7986 | 47850 | 0.991 | 0 | 484 | 23682 | 0.995 | 0.0286
2x2x1 | touch | 168 | 508 | 144 | 8 | 0.299 | 0 | 356 | 88 | 0.524 | 0.333
2x2x1 | sep | 23808 | 65408 | 9396 | 55412 | 0.991 | 0 | 600 | 23690 | 0.995 | 0.0286
2x2x2 | touch | 168 | 612 | 240 | 30 | 0.441 | 0 | 342 | 120 | 0.714 | 0.333
2x2x2 | sep | 23808 | 74880 | 10848 | 63330 | 0.991 | 0 | 702 | 23712 | 0.996 | 0.0286
4x1x1 | touch | 240 | 656 | 160 | 0 | 0.244 | 0 | 496 | 120 | 0.500 | 0.2
4x1x1 | sep | 26112 | 64000 | 11358 | 51750 | 0.986 | 90 | 802 | 25916 | 0.992 | 0.027
4x4x1 | touch | 336 | 1212 | 504 | 56 | 0.462 | 0 | 652 | 280 | 0.833 | 0.2
4x4x1 | sep | 28608 | 84288 | 17988 | 64548 | 0.979 | 144 | 1608 | 28472 | 0.995 | 0.027
4x4x4 | touch | 456 | 2064 | 1032 | 318 | 0.654 | 0 | 714 | 456 | 1.000 | 0.2
4x4x4 | sep | 31296 | 108960 | 27036 | 79218 | 0.975 | 168 | 2538 | 31296 | 1.000 | 0.027
8x1x1 | touch | 384 | 1136 | 352 | 0 | 0.310 | 0 | 784 | 216 | 0.562 | 0.111
8x1x1 | sep | 30720 | 79360 | 16852 | 61050 | 0.982 | 392 | 1066 | 30428 | 0.990 | 0.0244
8x8x1 | touch | 816 | 3388 | 1704 | 248 | 0.576 | 0 | 1436 | 728 | 0.892 | 0.111
8x8x1 | sep | 39360 | 128192 | 35942 | 87858 | 0.966 | 720 | 3672 | 39144 | 0.995 | 0.0244
8x8x8 | touch | 1464 | 7560 | 4056 | 1758 | 0.769 | 0 | 1746 | 1464 | 1.000 | 0.111
8x8x8 | sep | 49728 | 197856 | 71676 | 118122 | 0.959 | 960 | 7098 | 49728 | 1.000 | 0.0244
16x1x1 | touch | 672 | 2096 | 736 | 0 | 0.351 | 0 | 1360 | 408 | 0.607 | 0.0588
16x1x1 | sep | 39936 | 110080 | 25090 | 82950 | 0.981 | 542 | 1498 | 39572 | 0.991 | 0.0204
16x16x1 | touch | 2352 | 10812 | 6024 | 1016 | 0.651 | 0 | 3772 | 2200 | 0.935 | 0.0588
16x16x1 | sep | 65472 | 240576 | 72460 | 159540 | 0.964 | 1110 | 7466 | 65192 | 0.996 | 0.0204
16x16x16 | touch | 5208 | 28920 | 15864 | 8094 | 0.828 | 0 | 4962 | 5208 | 1.000 | 0.0588
16x16x16 | sep | 100416 | 458592 | 187140 | 254034 | 0.962 | 1608 | 15810 | 100416 | 1.000 | 0.0204

(`thnAxs` never fires on any of these geometries: its threshold 1e-4 needs a 100:1 aspect.)

Where the thin boxes come from (exact, from the pairBxs pieces; all lengths in g, fine centre at R):
* Fine-width indicator axis (target extended, source face plane at s): `[R_i - s - 1/2, R_i - s + 1/2]`, width 1,
  thin iff `lo > 1`: the fine cell's near edge more than one fine cell from the source face plane. For a TOUCHING
  fine cell the perpendicular source faces are the near one (lo = 0, harmless) and the FAR one, `[r, r+1]`, ratio
  1/(r+1): 1/3, 1/5, 1/9, 1/17 for r = 2, 4, 8, 16 -- these are all the touching-class thin boxes in the table, and r = 1
  gives [1, 2], exactly on the strict gate, which is why equal touching cells never enter. For a SEPARATED fine cell
  with face gap kap the near face gives [kap, kap+1], on the gate at kap = 1 and thin for kap >= 2: the equal-cell
  lattice at |D| = 2 escapes, |D| >= 3 does not.
* Coarse-width indicator axis (target degenerate, source extended): `[R_i + sig/2 - r/2, R_i + sig/2 + r/2]`, width r,
  thin iff lo > r: the target face plane more than one coarse cell beyond the source's far face (kap >= r).
* Trapezoid axis (both extended), pieces from the exact convolution of [-1/2, 1/2] and [-r/2, r/2] shifted by R_i:
  ramps `[R_i + (r-1)/2, R_i + (r+1)/2]` (width 1, weight linear) are thin iff `R_i > 1 - (r-1)/2`, so for every
  separated or laterally shifted position once r >= 4 (r = 4, R_i = 1/2: [2, 3]); the plateau
  `[R_i - (r-1)/2, R_i + (r-1)/2]` (width r-1, constant weight) is thin iff R_i > 3(r-1)/2, i.e. laterally separated by
  more than one coarse cell.
Smallest ratio per shape = the fine-width indicator at kap = 32 against the far face, `[r + 32, r + 33]`, ratio
1/(r + 33): 1/34 (r=1), 1/35 (2), 1/37 (4), 1/41 (8), 1/49 (16); it shrinks like 1/(gap in fine cells) without bound,
which is exactly the regime Gila hands to `egoSrfFxd!` today and that the k-series route (iii) would take over.

Two thin axes (`slv2`) dominate the separated classes (85% of the thin boxes): a box whose fine-width indicator AND
whose coarse-width indicator (or a ramp) are both away from the origin. Three thin axes occur too (SSS positions);
`slvAxs` returns the two thinnest and the third keeps its 1D closed form (`seg1`), which is cancellation-free.

## 3. The 3D centred Taylor series (`box3tay.jl`, function `box3Tay(m, lo, hi, cw, dw)`, generic in T)

Box `prod [lo_i, hi_i]` in the positive octant, weight `prod_i (cw_i + dw_i u_i)` (pairBxs gives at most one dw != 0;
the code is generic in all three). Centre `u_m = (lo+hi)/2`, half-widths `h = (hi-lo)/2`, `p = u - u_m`,
`rho^2 = |u_m|^2`:

    r^m = (rho^2 + 2 u_m p + 2 v_m q + 2 w_m s + p^2 + q^2 + s^2)^{m/2} = sum_{ijk} c_{ijk} p^i q^j s^k,
    W = sum_{ijk} c_{ijk} mu_i^{(1)} mu_j^{(2)} mu_k^{(3)},
    mu_n^{(a)} = int_{-h}^{h} (alpha_a + beta_a p) p^n dp = alpha_a 2h^{n+1}/(n+1) (n even), beta_a 2h^{n+2}/(n+2) (n odd),
    alpha_a = cw_a + dw_a u_m,a  (the weight at the centre, >= 0),  beta_a = dw_a.

Coefficients from `g d_p f = (m/2) f d_p g`, f = g^{m/2}, and its q- and s-mirrors; for i >= 1 (the entry of level
n = i+j+k from levels n-1 and n-2 only):

    rho^2 i c_{ijk} = (m - 2(i-1)) u_m c_{i-1,j,k} + (m - i + 2) c_{i-2,j,k}
                      - 2 v_m i c_{i,j-1,k} - 2 w_m i c_{i,j,k-1} - i c_{i,j-2,k} - i c_{i,j,k-2},
    rho^2 j c_{0jk} = (m - 2(j-1)) v_m c_{0,j-1,k} + (m - j + 2) c_{0,j-2,k} - 2 w_m j c_{0,j,k-1} - j c_{0,j,k-2},
    rho^2 k c_{00k} = (m - 2(k-1)) w_m c_{0,0,k-1} + (m - k + 2) c_{0,0,k-2},        c_000 = rho^m.

(The 2D recurrence of moments.tex is the j-mirror with the w terms dropped.) The table is built one level at a time
as an (n+1) x (n+1) triangle `c_{i,j,n-i-j}`; the level sums are stored, the sum is formed smallest level first, and
the same five controls as box2Tay are kept: power-of-two prescale so that max(u_m) = O(1) (unscale by
`2^{e (m+6)}` because the three weight factors are scaled as lengths), two consecutive levels below `eps(T)/16`
(odd levels vanish identically for constant weights), the projection from the geometric-mean level ratio at n >= 10
abandoning attempts needing more than 140 levels, and the final guard `sum |terms| <= 4 |sum|`.

Polydisc gate. On the octant box every term of `t = 2 u_m . x + |x|^2` is nonnegative, so the expansion of
`(rho^2 + t)^{m/2}` in powers of t, re-expanded in x, is ABSOLUTELY convergent iff `t(h) < rho^2`, i.e.

    sum_i (u_m,i^2 - 2 u_m,i h_i - h_i^2) = sum_i (lo_i^2 + 2 lo_i hi_i - hi_i^2)/2 > 0   <=>   lam > 1,

lam the positive root of `|h|^2 lam^2 + 2 (u_m . h) lam - rho^2 = 0` (the quantity box2Tay gates at 1/2). Per axis
the margin is `+a^2/2` for [a, 2a], `~lo^2` for a thin [lo, lo+1], and `-H^2/2` for a corner axis [0, H]: a thin axis
buys convergence, a long corner axis spends it. The gate is kept at lam > 1/2 as in 2D because the rearranged series
converges conditionally somewhat beyond lam = 1 (measured: [1,2]x[0,1]x[0,1], lam = 0.87, converges in 82 levels at
m = -1 with condition 1.05).

Smoke test (`taytest.jl`, `out/taytest.txt`; reference = 320-bit graded Gauss rule of mom.jl `boxReg!`, ordN 44 vs 64
agreeing to 1e-84 or better on every box), m in {-1, 0, 3, 12}, all eight boxes: box3Tay within one eps of the
reference in all 32 cases (digits lost 0.00), the GL branch 0.0-0.85; levels 10-82 in Float64; at 320 bits the series
reaches 3e-95 (the reference's own level) where it converges and is rejected by the projection where it would need
> 140 levels (as documented for the 2D one). Cost at m = 3 (load 5.9): one thin axis [16,17]x[0,8.5]x[0,7.5]: box3 (GL, 9
nodes x psiBox) 27.2 us, box3Tay 22.3 us; two thin axes [7.5,8.5]x[0,1]x[7.5,8.5]: box3 (81 nodes x seg1) 3.4 us, box3Tay
6.8 us.

### 3a. A 3D-only convergence trap, found by the sweep and fixed (`dbg.jl`, `out/boxval_v1.txt`)

The first per-box sweep (2734 distinct boxes, part 3b) returned three boxes with 11.1-13.2 digits lost by the series
at m = -1 while the condition number read exactly 1.0 and the level count 3: `[2,3]^3` (2x1x1, R = (5/2, 2, 2)),
`[1,2]x[0,3]x[0,3]` twice (4x4x4, R = (7/2, 3/2, 3/2)), all in units of g, all with centre u_m = (a, a, a). Instrumented
level sums (`dbg.jl`): level 1 = 0 (odd moments vanish for constant weights), level 2 = 0 EXACTLY, level 4 = 1.1e-4 of
the running sum. At m = -1 the function 1/r is harmonic, so c_200 + c_020 + c_002 = 0, and on the diagonal direction the
three pure second derivatives are equal, hence each is zero: box2Tay's stopping rule "two consecutive levels below
eps/16" is satisfied by an identically-zero odd level plus an accidentally-zero even level and the series stops at
n = 3 with the m = -1 seed wrong by 3e-5 (cube) and 3e-3. The 2D code cannot hit this: the restriction of 1/r to a
plane is not harmonic in (p, q) (f_pp + f_qq = -f_cc), and box2Tay's 3x3 weight table always contains the
c_11 mu_1 nu_1 term of the linear weights, which is nonzero. Fix in `box3tay.jl`: THREE consecutive small levels and
n >= 4. After the fix the cube gives 7.047930526956486e-6 against 7.047930526956487e-6 (320 bits) in 25 levels, and
the corner-type box at lam = 0.52 is rejected (falls back to the closed forms, 0.54 digits). The whole sweep was rerun
with the fixed rule (part 3b); `out/boxval_v1.txt` keeps the pre-fix data.

## 2. Digits lost in Float64 on the face pairs of unequal cells (`facedig.jl`, tier A; `out/facedig.txt`, one line per face pair)

Protocol as in moments.tex sec. "Digits lost in Float64": `pairMoments(A, B, 12)` in Float64 against the 320-bit
evaluation of the same formulas at the same (exactly representable, g = 1/32) panels; digits lost =
log10(|x64 - x320| / (eps |x320|)), clipped at 0. Sample: 8 shapes (1x1x1 control, 2x1x1, 4x1x1, 16x1x1, 4x4x1, 4x4x4,
16x16x1, 16x16x16) x 11 fine-cell positions (T-ctr: touching, fine cell over the centre-most sub-cell; T-crn: touching
over the corner sub-cell; S1/S2/Sr/S32-ctr: face gap 1, 2, r, 32 fine cells; S1-crn; TT-diag, SS1-diag, SSr-diag:
edge-diagonal positions; SSS1: separated on all three axes) x 36 face pairs = 2772 face pairs (positions coinciding
for r = 1 deduplicated). Box class of a face pair: 2D = parallel faces (never thin); closed = perpendicular, no box in
the thin branch; thin = perpendicular with >= 1 box through `box3Slv`. Tier A ran 20 min at load 9-11.

shape | contact | box class | face pairs | worst digits lost | where (position, F/Fp, m) | mean Float64 us per face pair
--- | --- | --- | --- | --- | --- | ---
1x1x1 | touch | 2D | 24 | 0.29 | TT-diag xyL/xyL m=12 | 672
1x1x1 | touch | closed | 48 | 0.90 | TT-diag yzU/xzL m=-1 | 207
1x1x1 | sep | 2D | 60 | 0.23 | S1-ctr xzL/xzU m=12 | 570
1x1x1 | sep | closed | 10 | 0.90 | SS1-diag yzL/xzU m=-1 | 266
1x1x1 | sep | thin | 110 | 0.81 | SS1-diag xzU/xyL m=12 | 433
2x1x1 | touch | 2D | 24 | 0.23 | TT-diag yzL/yzU m=12 | 866
2x1x1 | touch | closed | 24 | 0.74 | TT-diag yzL/xzL m=12 | 182
2x1x1 | touch | thin | 24 | 0.51 | TT-diag xzU/xyL m=-1 | 323
2x1x1 | sep | 2D | 72 | 0.23 | SS1-diag yzL/yzU m=12 | 750
2x1x1 | sep | closed | 20 | 0.90 | SS1-diag xzL/yzU m=-1 | 293
2x1x1 | sep | thin | 124 | 0.76 | SSr-diag xyL/yzU m=12 | 442
4x1x1 | touch | 2D | 24 | 0.23 | TT-diag yzL/yzU m=12 | 929
4x1x1 | touch | closed | 24 | 0.70 | T-ctr xzL/yzU m=12 | 184
4x1x1 | touch | thin | 24 | 0.68 | TT-diag xzU/yzL m=12 | 267
4x1x1 | sep | 2D | 84 | 0.23 | SS1-diag yzL/yzU m=12 | 799
4x1x1 | sep | closed | 28 | 0.90 | SS1-diag xzL/yzU m=-1 | 224
4x1x1 | sep | thin | 140 | 0.93 | SSr-diag yzU/xzL m=0 | 444
16x1x1 | touch | 2D | 24 | 0.82 | T-ctr xzL/xzU m=-1 | 642
16x1x1 | touch | closed | 24 | 0.70 | T-ctr xzL/yzU m=12 | 221
16x1x1 | touch | thin | 24 | 0.66 | TT-diag xzU/yzL m=7 | 219
16x1x1 | sep | 2D | 84 | 0.99 | SS1-diag xyL/xyL m=-1 | 937
16x1x1 | sep | closed | 28 | 0.90 | SS1-diag xzL/yzU m=-1 | 245
16x1x1 | sep | thin | 140 | 0.93 | SSr-diag yzU/xzL m=0 | 432
4x4x1 | touch | 2D | 36 | 0.11 | T-ctr yzU/yzU m=12 | 1295
4x4x1 | touch | closed | 12 | 0.70 | T-ctr yzL/xzU m=12 | 223
4x4x1 | touch | thin | 60 | 0.61 | T-ctr xyL/xzL m=11 | 365
4x4x1 | sep | 2D | 96 | 0.18 | S1-crn xyL/xyL m=-1 | 1292
4x4x1 | sep | closed | 15 | 0.70 | S1-crn xzU/yzU m=12 | 535
4x4x1 | sep | thin | 177 | 0.93 | Sr-ctr yzU/xzL m=0 | 628
4x4x4 | touch | 2D | 36 | 0.20 | T-ctr yzU/yzL m=3 | 1952
4x4x4 | touch | thin | 72 | 0.61 | T-ctr xzL/xyL m=6 | 605
4x4x4 | sep | 2D | 96 | 0.24 | SS1-diag yzU/yzL m=3 | 1927
4x4x4 | sep | thin | 192 | 0.90 | Sr-ctr yzU/xzL m=0 | 991
16x16x1 | touch | 2D | 36 | 0.82 | T-crn xzL/xzU m=-1 | 1113
16x16x1 | touch | closed | 10 | 0.63 | T-crn yzU/xzU m=12 | 254
16x16x1 | touch | thin | 62 | 0.71 | T-crn xzU/xyL m=12 | 304
16x16x1 | sep | 2D | 96 | 0.45 | SS1-diag xyL/xyL m=-1 | 1317
16x16x1 | sep | closed | 10 | 0.69 | SS1-diag yzU/xzU m=12 | 403
16x16x1 | sep | thin | 182 | 0.86 | SSr-diag yzU/xyL m=-1 | 470
16x16x16 | touch | 2D | 36 | 0.30 | T-crn xzU/xzU m=11 | 1868
16x16x16 | touch | thin | 72 | 0.71 | T-ctr xzL/yzL m=12 | 404
16x16x16 | sep | 2D | 96 | 0.24 | S2-ctr xzU/xzU m=-1 | 2041
16x16x16 | sep | thin | 192 | 0.73 | SS1-diag xyL/yzL m=10 | 627

Worst digits lost per m over all 2772 face pairs, by box class:

class | m=-1 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- 
thin | 0.86 | 0.93 | 0.74 | 0.73 | 0.78 | 0.75 | 0.79 | 0.72 | 0.68 | 0.78 | 0.82 | 0.76 | 0.74 | 0.81
closed | 0.90 | 0.00 | 0.30 | 0.30 | 0.33 | 0.47 | 0.51 | 0.53 | 0.55 | 0.61 | 0.61 | 0.66 | 0.68 | 0.74
2D | 0.99 | 0.00 | 0.21 | 0.00 | 0.24 | 0.22 | 0.27 | 0.18 | 0.35 | 0.24 | 0.30 | 0.27 | 0.30 | 0.29

GLOBAL worst 0.99 digits (16x1x1, SS1-diag, xyL/xyL, m = -1, a PARALLEL pair, i.e. the 2D integrator, not the thin
branch); worst thin-branch face pair 0.93 (4x1x1 / 16x1x1 / 4x4x1 SSr-diag and Sr-ctr, yzU/xzL, m = 0, two thin axes);
worst closed-form 3D pair 0.90 (the m = -1 seed of Vp-like corner geometries, identical for equal and unequal cells).
Zero face pairs above one digit. For comparison the equal-cell TOUCHING lattice (moments.tex) is at 0.91 (3D) / 0.87
(2D); the equal-cell SEPARATED pairs measured here (1x1x1 sep thin, 110 pairs, 99% through the thin branch) are at
0.81. So: the cross-scale geometries do NOT lose more than the lattice ones, and the thin branch itself (0.93 worst,
m = 0) is at the level of the closed forms it bypasses (0.90-0.96 per box, part 3b). Float64 wall time per face pair
(load 9-11, so upper bounds): 2D pairs 0.6-2.0 ms, closed 3D 0.18-0.54 ms, thin 3D 0.22-0.99 ms; the parallel pairs
are the expensive ones (the 2D hybrid series), not the thin branch.

Relevance to Gila: farfield.jl's route (iii) calls `pairMoments` at KSRPRC = 128 bits (farfield.jl:1244, tnsKsr), where
the GL order of the thin branch is ~26-30 and the loss is 0 at the 1e-16 level whatever the geometry; the Float64
numbers above matter only if a Float64 near-field fill through moments.jl is ever wired.

### 3b. Per-box sweep of box3Tay against the 320-bit reference (`boxval.jl`, `out/boxval.txt`, aggregate `agg_box.py` -> `out/boxval_agg.txt`)

Sample: every distinct 3D box of the tier-A perpendicular face pairs (8 shapes x 11 positions x 24 perpendicular pairs, pairBxs
exact), 2734 boxes: slv1 1307, slv2 921 (thin branch, 2228 = 81%), dom 15, cls 491 (divergence-identity closed forms), thn 0.
Reference: `box3` at 320 bits (the same formulas; slvOrd gives 66 GL nodes per thin axis there at the gate ratio, 14 in Float64, 28 at 128 bits), cross-checked on 193 boxes against the independent
320-bit graded quadrature of mom.jl (`boxReg!` ordN 44, `boxDuf!` ordX 30 / ordE 44 for corner boxes): max relative
difference 3.9e-43, so the reference is good to far below Float64. Both Float64 evaluations at all m in -1..12 per box;
digits lost = log10(|x64 - x320| / (eps |x320|)) clipped at 0. Stopping rule of the series: three consecutive levels below
eps/16 and n >= 4 (sec. 3a); polydisc gate lam > 1/2; projection abandons attempts needing > 140 levels. 479 s at load 10.9.
Times are per single (box, m) call, Float64, at load 10-11 (upper bounds; the ratios are what matters).

branch | boxes | series converged all 14 m | partial (some m) | rejected (all m) | worst box3 digits (m) | worst series digits (m) | median / p90 / max levels | median cnd | max cnd | median us box3 | median us series | ratio series/box3 median / p90 / max
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
slv1 | 1307 | 1236 (94.6%) | 49 | 22 (1.7%) | 0.95 (12) | 0.35 (12) | 35 / 89 / 139 | 1.033 | 3.40 | 8.2 | 26.3 | 3.2 / 28.5 / 2126
slv2 | 921 | 919 (99.8%) | 2 | 0 | 0.96 (3) | 0.18 (12) | 25 / 39 / 138 | 1.078 | 2.02 | 6.1 | 12.7 | 2.0 / 5.3 / 55
dom | 15 | 15 | 0 | 0 | 0.22 (1) | 0.22 (9) | 43 / - / 119 | 1.007 | 1.05 | 3.1 | 26.8 | -
cls | 491 | 150 (30.5%) | 134 | 207 (42.2%) | 1.15 (-1) | 0.56 (12) | 83 / - / 139 | 1.014 | 3.78 | 4.8 | 298.6 | ~60

Digits. On the 2228 thin-branch boxes the GL branch (box3Slv) loses 0.51 (slv1) / 0.71 (slv2) digits at the median, 0.66 / 0.87
at p90, 0.95 / 0.96 at worst (52 boxes above 0.9), its worst m being 12, 11, -1, 10 in that order (the m = -1 seed and the high
powers). The series, where it converges, is at ONE eps: median 0.00, p90 0.01, worst 0.35 (slv1) / 0.18 (slv2), no box above 0.5;
the "worst m" of the series is m = -1 in 1979 of 2206 converged boxes, i.e. its loss is the single rounding of the seed
rho^m. The two agree to 3.2e-15 relative at worst (`tay-vs-box3` column), consistent with the GL loss. On the 491 ordinary
(closed-form) boxes the series also beats the closed forms where it converges (0.56 vs 1.15; the five boxes with >= 1.0 digits
lost by the closed forms are the m = -1 seeds of the Vp-like corner boxes [1,2]x[1,2]x[0,1] and [4,8]x[1,2]x[1,2], the same
0.9-1.1 per-box loss moments.tex measured on the equal-cell lattice) but costs 60x (median 299 us vs 4.8 us, 83 levels median).

Condition number (sum |terms| / |sum|, converged thin boxes): min 1.000, median 1.038, p90 1.455, p99 2.151, max 3.40;
202 boxes above 1.5, 37 above 2, 3 above 3; the guard (<= 4) never fired on a converged box. The rearranged series does not
cancel: the digits lost track cnd only through the seed.

Rejection (polydisc gate / projection). Thin boxes: 22 of 2228 (1.0%) rejected at every m, all at lam 0.463-0.497 (i.e. by the
gate lam > 1/2, not by the projection), all in the r = 16 shapes: a thin fine-width indicator [1,2] or [2,3] next to a long
corner or coarse axis, e.g. [1,2]x[1,17]x[2,3] (16x16x1, 16x1x1) and [2,3]x[0,7..9]x[0,7..9] (16x16x16). 51 boxes (2.3%) converge
only for the low m (nc = 13, 12, 11, 9, 7: the high m fail the projection first), lam 0.507-0.696, again a thin axis beside a
corner axis [0,16] or [0,4]. Everything at lam >= 0.7 converged at all 14 m. Ordinary boxes: rejected 207 / 491, exactly the
boxes at lam 0.414-0.500 (the corner boxes [0,H]x... whose lam = sqrt2 - 1 = 0.414 for a cube at the origin), converged for
lam 0.503-1.243. Coverage per shape (thin boxes, all 14 m): 1x1x1 43/43, 2x1x1 71/71, 4x1x1 77/77, 4x4x1 203/203, 4x4x4 646/648,
16x1x1 68/83, 16x16x1 249/265, 16x16x16 798/838.

Convergence vs lam (thin boxes): lam in [0.5,0.8): 177 boxes, 126 full / 51 partial, median 109 levels (max 139); [0.8,1.0): 106,
all full, median 65; [1.0,1.2): 119, median 53 (max 64); [1.2,1.5): 389, median 39; [1.5,2): 350, 33; [2,3): 403, 27; [3,5): 249,
22; [5,10): 199, 17; [10,100): 214, 13 (max 15). The worst series digits per bin: 0.35, 0.26, 0.28, 0.19, 0.18, 0.08, 0.05, 0.11,
0.06; the worst GL digits per bin 0.87-0.96 in every bin from 0.8 up (the GL loss does not depend on lam).

Cost. Per call the series is 2-3x the GL branch at the median (26 vs 8 us with one thin axis, 13 vs 6 with two) and the tail is
long: the cost is O(N^2) per level for the (n+1)x(n+1) triangle, measured 5.5 us (< 20 levels), 16 (20-40), 45 (40-60), 93
(60-80), 224 (80-100), 430 (100-120), 543 us (120-139) per call; the worst ratio, 2126x, is [0,16]x[7,8]x[0,1] (16x16x1, lam
0.672, 115 levels: 800 us against 2.8 us). Summed over all 2228 thin boxes (14 m each) the series costs 161 ms against 25.5 ms
for GL: 6.3x, with 271 boxes (12%) above 10x. Gating the series on lam (series iff slvAxs fires and lam >= L, GL otherwise),
from the same per-box data:

gate L | boxes to series | to GL | worst digits of the mix | max levels | total us (mix) | total us (GL only)
--- | --- | --- | --- | --- | --- | ---
0.5 (as run) | 2155 | 73 | 0.86 | 139 | 99917 | 25544
0.8 | 2029 | 199 | 0.87 | 105 | 59790 | 25544
1.0 | 1923 | 305 | 0.94 | 64 | 47872 | 25544
1.2 | 1804 | 424 | 0.94 | 49 | 41244 | 25544
1.5 | 1415 | 813 | 0.96 | 41 | 31984 | 25544
2.0 | 1065 | 1163 | 0.96 | 32 | 27911 | 25544

No gate makes the series cheaper than GL in aggregate (1.1-3.9x), and any gate above 0.8 hands the boxes it excludes back to GL
at 0.94-0.96 digits, so the worst case of the mix is the GL worst case: the series improves the median and the tail below the
gate only. The gate lam >= 1 (absolute convergence) caps the levels at 64 and the per-call cost at ~100 us for 1.9x the GL total.

The three trap boxes of sec. 3a (`out/boxval_v1.txt`: 13.19, 13.19, 11.10 digits lost with the two-level rule, levels 14, 14, 22)
read 0.00, 0.00, 0.00 digits with the fixed rule in 15, 15 (7 of 14 m; [1,2]x[0,3]x[0,3] is a cls box at lam 0.52) and 25 levels
([2,3]^3, all 14 m, cnd 1.000).

## 2b. Tier B: the face-pair digits against the independent quadrature reference (`facedig.jl`, `out/facedigB.txt`)

Reduced set: 14 (shape, position) classes x 36 face pairs = 504 face pairs; reference `pairMom` of mom.jl at 320 bits (graded
Gauss, ordN 44 / ordX 30 / ordE 44), cached in `cache/facemom320.txt`; ran 1874 s at load 6.3-8.5. Agreement of the two 320-bit
references (pairMoments-320 of tier A vs pairMom, max over m of the relative difference): 4e-84 to 8e-91 on separated pairs,
6e-60 on the touching parallel pairs, 2.4e-60 (1x1x1), 1.2e-49 (4x1x1), 3.9e-45 (16x1x1), 2.1e-43 to 7.1e-43 (4x4x4) on the
touching perpendicular pairs -- the last three are the Duffy rule's own convergence on panels of aspect 4-16 (ordX 30), 27+
orders below Float64. The digits lost vs pairMom are IDENTICAL to those vs pairMoments-320 on every one of the 504 pairs
(column 13 = column 9 of `out/facedigB.txt`): the tier-A numbers of sec. 2 are reference-independent. On this set: thin class worst
0.70 (4x4x4 SS1-diag xyU/yzL m = 10), closed 0.70 (m = 12), 2D 0.82 (16x1x1 T-ctr xzL/xzU m = -1); per m the thin class is
0.48-0.70 across m = -1..12 with no m standing out.

## 3c. The hybrid at face-pair level (`hybrid.jl`, `out/hybrid.txt`; per-m rerun `hybrid_m.jl`, `out/hybrid_m.txt`)

pairMoments(A, B, 12) in Float64 on the 1848 PERPENDICULAR face pairs of the tier-A set (8 shapes x 11 positions x 24; the
parallel pairs never reach box3 and are untouched), three box3 policies, digits lost against pairMoments at 320 bits, wall
time per face pair (all 14 moments) at load 8.5-9.6:
  GL     = the shipped box3 (thin axes by Gauss-Legendre);
  SER    = box3Tay wherever slvAxs fires and the series converges, box3 (GL) otherwise; the non-thin boxes keep their closed forms;
  SERALL = box3Tay attempted on EVERY 3D box (closed forms only on rejection).
Pair class: thin = the pair has >= 1 box in the thin branch (1595 pairs), closed = none (253). 843 s.

class | contact | pairs | worst digits GL | SER | SERALL | median us GL | SER | SERALL | mean us GL | SER | SERALL | SER boxes series / rejected / closed | SERALL series / rejected
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
thin | touch | 338 | 0.71 | 0.71 | 0.71 | 271 | 776 | 2362 | 352 | 2730 | 5593 | 11500 / 64 / 6412 | 13400 / 4576
thin | sep | 1257 | 0.93 | 0.63 | 0.63 | 463 | 630 | 768 | 531 | 1419 | 2777 | 50250 / 598 / 9324 | 57240 / 2932
closed | touch | 142 | 0.90 | 0.90 | 0.70 | 209 | 205 | 2081 | 224 | 231 | 4255 | 0 / 0 / 4144 | 1578 / 2566
closed | sep | 111 | 0.90 | 0.90 | 0.68 | 231 | 215 | 4063 | 364 | 322 | 6245 | 0 / 0 / 3276 | 2780 / 496

Per shape (thin pairs only): shape | pairs | worst GL | worst SER | mean us GL | mean us SER
1x1x1 | 110 | 0.81 | 0.07 | 478 | 528
2x1x1 | 148 | 0.76 | 0.51 | 499 | 573
4x1x1 | 164 | 0.93 | 0.29 | 470 | 661
16x1x1 | 164 | 0.93 | 0.63 | 403 | 1399
4x4x1 | 237 | 0.93 | 0.38 | 467 | 903
4x4x4 | 264 | 0.90 | 0.55 | 756 | 1898
16x16x1 | 244 | 0.86 | 0.71 | 357 | 1803
16x16x16 | 264 | 0.73 | 0.71 | 453 | 4053

Reading. (a) Where a pair is served ENTIRELY by the series (1037 of the 1595 thin pairs: no closed-form box, nothing rejected)
the worst digits lost drop from 0.93 (GL) to 0.23 (SER). (b) The remaining 558 thin pairs mix series boxes with closed-form
corner boxes, and their worst case, 0.71, is the closed-form boxes' (16x16x1 T-crn / TT-diag xzU/xyL: 14 series boxes + 28
closed boxes, 0.71 under all three policies; SERALL rejects the same 28 at lam 0.41-0.50 and falls back to the closed forms), so
the series cannot lower the face-pair worst case below the closed forms it sits next to: the global worst goes 0.93 -> 0.90 (SER,
a closed-only pair) -> 0.71 (SERALL). (c) SER is better than GL by > 0.1 digits on 1189 pairs, worse by > 0.1 on 48 (by > 0.2 on
24, max +0.37: 4x4x4 S1-ctr xyL/yzU 0.02 -> 0.31, TT-diag 0.03 -> 0.40; these are pairs where GL happened to lose nothing and
the series' seed rounding on 14 boxes shows), so per pair the two are within 0.4 digits of each other in either direction and
the series wins on the 0.9-digit tail only. (d) Cost: SER/GL median 1.47x, p90 6.5x, max 150x; SER is faster than GL on 416 of
the 1595 thin pairs (the two-thin-axis boxes at lam > 2: 13 levels against 81 GL nodes x seg1) and 4-9x slower at the mean
on the r = 16 shapes, where the lam 0.5-0.8 boxes ([0,16]x[7,8]x[0,1] and kin) run 100+ levels; SERALL is 2.5x at the median and
35x at p90. The digit gain per pair is confined to the thin pairs' tail (0.93 -> 0.63 for separated pairs, none for touching ones,
whose 0.71 is closed-form) and buys nothing at the face-pair worst case below 0.7.

### 3d. Per-m digits of the hybrid, with the lam >= 1 regime policy (`hybrid_m.jl`, `out/hybrid_m.txt`, `agg_hybm.py` -> `out/hybrid_m_agg.txt`)

Same 1848 perpendicular face pairs and reference as sec. 3c, per-m digits kept, four policies: GL, SER, SERALL as above and
SER1 = box3Tay iff slvAxs fires AND lam >= 1 (absolute convergence of the polydisc expansion, <= 64 levels), GL otherwise.
719 s at load 6.0 (the four Float64 policies are timed back to back per pair, so their ratios are comparable).

Worst digits lost per m over the 1595 thin pairs:

policy | m=-1 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
GL | 0.86 | 0.93 | 0.74 | 0.73 | 0.78 | 0.75 | 0.79 | 0.72 | 0.68 | 0.78 | 0.82 | 0.76 | 0.74 | 0.81
SER | 0.55 | 0.00 | 0.52 | 0.38 | 0.48 | 0.36 | 0.57 | 0.45 | 0.48 | 0.59 | 0.56 | 0.51 | 0.63 | 0.71
SER1 | 0.56 | 0.48 | 0.57 | 0.54 | 0.56 | 0.58 | 0.57 | 0.60 | 0.59 | 0.62 | 0.56 | 0.62 | 0.63 | 0.71
SERALL | 0.48 | 0.00 | 0.52 | 0.38 | 0.48 | 0.36 | 0.57 | 0.45 | 0.48 | 0.59 | 0.56 | 0.51 | 0.63 | 0.71

Mean digits lost over the thin pairs at m = -1, 0, 3, 6, 9, 12: GL 0.21, 0.18, 0.20, 0.23, 0.25, 0.28; SER 0.02, 0.00, 0.01, 0.02,
0.03, 0.03; SER1 0.05, 0.03, 0.04, 0.05, 0.05, 0.06; SERALL 0.01, 0.00, 0.01, 0.01, 0.02, 0.02. On the 253 closed-only pairs GL,
SER and SER1 are identical by construction (0.90 at m = -1, 0.30-0.74 for m >= 1) and SERALL takes the m = -1 seed from 0.90 to
0.48 and m = 12 from 0.74 to 0.70. The GL worst per m is 0.68-0.93 with m = 0 worst (the two-thin-axis boxes); the series
policies remove the m = 0 loss entirely (0.00) and leave 0.36-0.71 at the other m, all of it from the closed-form boxes in
mixed pairs (sec. 3c (b)): the per-m worst of SER for m >= 1 equals the per-m worst of the closed forms in the same pairs.

class | contact | pairs | worst GL | SER | SER1 | SERALL | median us GL | SER | SER1 | SERALL | mean us GL | SER | SER1 | SERALL | SER1 boxes series / not attempted
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
thin | touch | 338 | 0.71 | 0.71 | 0.71 | 0.71 | 229 | 627 | 556 | 1971 | 309 | 2300 | 754 | 4874 | 9408 / 8568
thin | sep | 1257 | 0.93 | 0.63 | 0.63 | 0.63 | 386 | 554 | 510 | 656 | 451 | 1328 | 650 | 2369 | 46396 / 13776
closed | touch | 142 | 0.90 | 0.90 | 0.90 | 0.70 | 160 | 135 | 134 | 1672 | 165 | 160 | 160 | 3140 | 0 / 4144
closed | sep | 111 | 0.90 | 0.90 | 0.90 | 0.68 | 191 | 186 | 186 | 2638 | 236 | 230 | 232 | 5569 | 0 / 3276

Per shape, thin pairs: shape | n | worst GL | SER | SER1 | SERALL | mean us GL | SER | SER1 | SERALL
1x1x1 | 110 | 0.81 | 0.07 | 0.07 | 0.00 | 352 | 392 | 390 | 553
2x1x1 | 148 | 0.76 | 0.51 | 0.51 | 0.00 | 359 | 441 | 426 | 866
4x1x1 | 164 | 0.93 | 0.29 | 0.39 | 0.21 | 345 | 415 | 406 | 1240
16x1x1 | 164 | 0.93 | 0.63 | 0.63 | 0.63 | 330 | 1184 | 282 | 2053
4x4x1 | 237 | 0.93 | 0.38 | 0.62 | 0.27 | 411 | 777 | 611 | 3427
4x4x4 | 264 | 0.90 | 0.55 | 0.55 | 0.28 | 693 | 1693 | 1061 | 5231
16x16x1 | 244 | 0.86 | 0.71 | 0.71 | 0.71 | 336 | 2008 | 575 | 2712
16x16x16 | 264 | 0.73 | 0.71 | 0.71 | 0.71 | 405 | 3615 | 1090 | 3942

Global worst: GL 0.93, SER 0.90, SER1 0.90, SERALL 0.71. Time ratios to GL per pair (median / p90 / max): SER 1.48 / 6.8 / 286;
SER1 1.23 / 3.5 / 47; SERALL 2.4 / 33 / 330. SER1 rejects nothing (every attempted box converges at all 14 m: 55,804 series
boxes, 0 rejections; 22,344 thin boxes at lam < 1 left to GL) and serves 907 pairs entirely by the series, worst 0.23. Pairs
better than GL by > 0.1 digits: SER 1189, SER1 1069, SERALL 1413; worse by > 0.1: 48, 46, 49 (max +0.37). The lam >= 1 gate
removes the 100+-level tail (16x1x1: 1184 -> 282 us, faster than GL's 330; 16x16x16: 3615 -> 1090) at no cost in worst digits;
it does not move the face-pair worst case, which the closed-form corner boxes fix at 0.71 (touching) and 0.90 (closed-only).

## 4. Recommendation

**Keep the quadrature branch (`box3Slv`) as moments.jl's thin-axis integrator; do not replace it by the series; no change to the
`box3` dispatch.** The numbers:

1. Accuracy of what ships. Per box the GL branch loses 0.51 / 0.71 digits at the median and 0.95 / 0.96 at worst (one / two thin
   axes, 2228 boxes, sec. 3b); per face pair 0.93 at worst over 1595 thin pairs and 2772 pairs in all, zero pairs above one digit
   (sec. 2, confirmed against the independent quadrature reference on 504 pairs, sec. 2b). That is the same level as the
   divergence-identity closed forms it bypasses (1.15 per box, 0.90 per face pair) and as the equal-cell lattice (0.91 3D / 0.87
   2D in moments.tex). The cross-scale geometries, ratios 2-16, gaps 1-32 fine cells, add nothing to the loss.
2. What the series buys. Where it converges it is at one eps per box (worst 0.35, median 0.00, cnd <= 3.4), and on the 1037 face
   pairs served entirely by it the worst case falls 0.93 -> 0.23. But (a) it needs a fallback: 1.0% of the thin boxes are rejected
   and 2.3% converge only for the low m (all at lam < 0.7, thin axis beside a long corner or coarse axis, r = 16), so `box3Slv`
   stays in the code either way; (b) the face-pair worst case does not follow the box: 558 thin pairs also hold closed-form corner
   boxes at lam 0.41-0.50 where no series converges, pinning them at 0.71, and the closed-only pairs stay at 0.90, so the global
   worst moves 0.93 -> 0.90 (SER) or -> 0.71 (SERALL), less than one binary digit; (c) cost: 2-3x per call at the median, 6.3x in
   aggregate, 2126x at worst (115 levels at lam 0.67), 1.5x / 6.5x (median / p90) per face pair, 4-9x at the mean on the r = 16
   shapes; SERALL 2.5x / 35x.
3. Both by regime. Gating the series at lam >= 1 (absolute convergence, <= 64 levels) or lam >= 2 (<= 32 levels) costs 1.9x / 1.1x
   the GL total and leaves the worst case at 0.94 / 0.96 because the excluded boxes go back to GL (sec. 3b table). The gate only
   improves the median (0.5-0.7 -> 0.0), which nothing downstream is asking for (at face-pair level, sec. 3d: SER1 = series iff
   slvAxs and lam >= 1 costs 1.23x GL at the median, 3.5x at p90, 47x at worst, and its worst digits are SER's, 0.90 global,
   0.63 / 0.71 on separated / touching thin pairs, against GL's 0.93 / 0.71): the 1e-8 publishability bar is 8 decades away
   and the near-field fill that would run moments.jl in Float64 does not exist. A second code path for a median is not justified.
   If a Float64 near-field fill is ever wired and its error budget is counted in eps, the drop-in is `box3tay.jl` (90 lines,
   generic in T) gated at slvAxs AND lam >= 1, GL otherwise, WITH the three-level stopping rule (sec. 3a: the two-level rule of
   box2Tay loses 11-13 digits on 1/r at diagonal centres u_m = (a,a,a) because level 2 vanishes identically).
4. What changes in moments.jl: nothing in the code. In moments.tex, the sentence "never reached on the lattice" about the thin
   branch must be scoped to the fifteen touching sub-geometries, and this sentence added for the document:
   **"Off the touching lattice the thin-indicator branch is the common case, not the exception: every separated cell pair with
   a face gap of two or more cells (equal cells at |D| >= 3) and every touching pair of ratio >= 2 sends 96-99% of its
   perpendicular boxes through `box3Slv`, where the Gauss-Legendre rule loses at most 0.93 digits per face pair in Float64
   (0.96 per box, 2228 boxes, 2734-box sweep against a 320-bit reference) -- the same level as the closed forms it replaces."**
5. What the equal-cell k-series route (iii) of farfield.jl sees (KSRPRC = 128 bits, `tnsKsr`, farfield.jl:1244-1400). At
   |D| >= 3 it already runs 99% of its perpendicular boxes through `box3Slv` (sec. 1, 1x1x1 sep: 47,616 boxes, 0.992 thin, only
   |D| = 2 escapes because [1,2] sits exactly on the strict gate). At 128 bits `slvOrd` takes 28 GL nodes per thin axis at the gate
   ratio and fewer as the box thins (rho grows with lo/hi), and the branch loses 0.00-0.99 of 38.2 digits against 320 bits on the
   seven face pairs measured (`ksr128.jl`, `out/ksr128.txt`; worst 5.8e-38 relative on the two-thin-axis pair 4x1x1 SSr-diag
   yzU/xzL, the same pair that is worst in Float64), i.e. 22 orders below route (iii)'s 1e-16 target: the thin branch is
   invisible to the k-series route, and its 320-bit values agree with the independent graded quadrature to 3.9e-43 on 193
   boxes. Its cost at 128 bits is 27-416 ms per face pair (load 5.7; 0.2-0.5 ms in Float64), dominated by the 28^2 = 784
   `seg1` evaluations of the two-thin-axis boxes (416 ms on the slv2,slv2 pair against 56-157 ms for slv1 pairs), which is
   where route (iii)'s seconds-per-offset go; the series would not help there either (at 128 bits it needs ~2x the Float64 levels
   and the same O(N^2)-per-level table). Nothing in the k-series route changes.

Stopped here: all three parts done (3b, 2b/3c/3d, 4); nothing pending, no Julia process left running. Files: report sections
3b/2b/3c/3d/4 above; new scripts `hybrid_m.jl`, `agg_hybm.py`, `ksr128.jl`; outputs `out/hybrid_m.txt`, `out/hybrid_m_agg.txt`,
`out/ksr128.txt`. Load averages during this incarnation's runs: 2.4-6.2.
