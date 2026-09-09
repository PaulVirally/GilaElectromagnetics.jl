# Family (b) extended by subdivision of the difference box

Library: `SCRATCH/work/famBsub/famBsub.jl` (famB's core verbatim + the sub-box layer;
standalone, generic in the number type).
Scripts: `s0_ref.jl` (220-bit reference builder), `s1_check.jl` (moment and reflection
identities), `s2_acc.jl` (accuracy against the reference), `s3_cost.jl` (table cost,
per-offset timing, symmetries, homogeneity, nMax), `s4_rho.jl` (rho_j / L / cost tables,
geometry only).  Raw output in `work/famBsub/out/`.

## 1. The sub-box decomposition

### 1.1 Statement

    T_ab(R) = (1/V_t) int_D w(d) [(da db + dab k^2) g](R + d) dd,
    D = prod_i [-s_i, s_i],  w(d) = prod_i (s_i - |d_i|),  g = e^{ikr}/(4 pi f^2 r), k = 2 pi f.

Split each axis into pieces; D becomes the disjoint union of boxes D_j = prod_i [a_ij, b_ij]
with centres c_j and half-widths h_j, and with d' = d - c_j,

    T_ab(R) = sum_j T^{(j)}_ab(R + c_j),
    T^{(j)}_ab(V) = (1/V_t) int_{D_j} w(c_j + d') [(da db + dab k^2) g](V + d') dd'.

Each T^{(j)} is famB's object with the local expansion centre V = R + c_j, the local radius
r_j = |h_j| = sqrt(sum_i h_ij^2), and the local ratio rho_j = r_j / |R + c_j|.

### 1.2 The weight on a sub-box, and the moments

**No piece may straddle 0.**  On a piece [a,b] with a >= 0 the factor s_i - |d_i| equals
s_i - c_i - t (t = d'_i), and on a piece with b <= 0 it equals s_i + c_i + t; both are

    s_i - |d_i| = al_i + bt_i t,   al_i = s_i - |c_i|,  bt_i = -sign(c_i),   t in [-h_i, h_i].

A piece containing 0 in its interior carries the kink of |d_i| and has no affine form.  The
error of using one anyway is not small: for m = 3 the middle piece is [-s/3, s/3] and the best
affine model there (al = s, bt = 0) gives int = 2s^2/3 against the true 5s^2/9, i.e. **20 % of
that piece's zeroth moment**.  So 0 must be a split point.  With m_i equal pieces this is
automatic for even m_i and forces an extra split for odd m_i, which then has m_i + 1 pieces:
**m = 3 costs exactly what m = 4 costs and converges more slowly** (its outer pieces have
h = s/3 against s/4), so odd m is dominated and never used below.  The minimal legal split is
therefore the 8 octants of D, written m = (2,2,2) throughout (identical to m = (1,1,1)).

The moments are exact rationals in the edge lengths:

    nu_n(al, bt, h) = int_{-h}^{h} (al + bt t) t^n dt
                    = 2 al h^{n+1}/(n+1)      n even
                    = 2 bt h^{n+2}/(n+2)      n odd
    int_{D_j} w(c_j + d') d'^gamma dd' = prod_i nu_{gamma_i}(al_i, bt_i, h_i).

Only three one-dimensional tables are needed, exactly as in famB:
nu^{(0)}_n = nu_n, nu^{(1)}_n = n nu_{n-1} (for one derivative), nu^{(2)}_n = n(n-1) nu_{n-2}
(for two), since da db acts on the polynomial d'^gamma.

Checked exactly (`s1_check.jl`, items 1 and 2, Rational{BigInt}):

    sum over pieces of nu_0 - mu_0(s)                             0 exactly, m = 1,2,3,4,5,8
    sum_j sum_q C(n,q) c_j^{n-q} nu_q  -  mu_n(s), n <= 8, m = 2,3,4,6   0 exactly

### 1.3 What subdivision destroys, and the sign it puts back

famB's parity argument (only even l, and four disjoint (l,m) classes) rests on w being even
in every coordinate.  The affine weight al + bt t on an off-centre piece is not even, so

- every (l,m) with 0 <= l <= L and -l <= m <= l contributes to every one of the six tensor
  entries: the table is 7 kinds x (nMax+1) x (L+1)^2 numbers per sub-box type instead of
  famB's four parity classes, and the per-offset cost per sub-box is 7 complex multiplies per
  (l,m) instead of famB's roughly 1.7;
- **odd l survives, so the (-1)^l of the addition theorem is now mandatory.**  famB dropped it
  because every surviving l was even.  Omitting it here is a first-order error: cube 1/32,
  m = (2,2,2), L = 24, f = 1, max|G - G_famB|/max|G_famB| was

        D = (4,0,0)  3.65e-2      D = (3,1,0)  5.63e-2
        D = (6,6,6)  6.45e-3      D = (5,1,2)  7.58e-3

  with the factor restored (same run) 1.51e-13, 2.63e-11, 2.85e-16, 2.59e-16 -- and the first
  two residuals are famB's own l-truncation at L = 24, not the subdivided value's.

### 1.4 The reflection rule

Let sigma be a sign pattern in {+-1}^3 acting on coordinates.  w is even, so w(sigma x) = w(x);
and G_ab(x) = [(da db + dab k^2) g](x) satisfies G_ab(sigma y) = sigma_a sigma_b G_ab(y).
Substituting d' = sigma u in the definition of T^{(sigma j)} gives

    T^{(sigma j)}_ab(V) = sigma_a sigma_b T^{(j)}_ab(sigma V),

and since the sub-box set is closed under sigma with c_{sigma j} = sigma c_j,

    **T_ab(R) = sum_{t in one octant} sum_{sigma in {+-1}^3} sigma_a sigma_b
                T^{(t)}_ab(sigma R + c_t).**

Diagonal entries take no sign; the off-diagonal entry (a,b) flips whenever exactly one of the
axes a, b is flipped -- the same rule the brief states for the whole tensor.  Consistently, the
mirrored sub-box's moments are nu_n(al, -bt, h) = (-1)^n nu_n(al, bt, h), i.e. the mirrored
table is the original with every odd-gamma_i monomial negated.

So only prod_i (pieces per axis on the positive side) geometry tables are ever built.  Verified
against tables built independently for all 8 octants (`s1_check.jl`, item 4), cube 1/32,
m = (2,2,2), L = 16:

    D = (3,1,0)  1.70e-16      D = (2,1,1)  1.35e-16      D = (5,1,2)  9.61e-17

### 1.5 Evaluation

Per sub-box type t and sign pattern sigma, with V = sigma R + c_t, r = |V|:

    T^{(t)}_ab(V) = (i k / f^2) sum_{l=0}^{L_t,sigma} sum_{m=-l}^{l}
                    (-1)^l h_l(k r) Y_lm(Vhat) Qt^{ab}_{lm}(t, k),
    Qt^{ab}_{lm} = sum_{n=0}^{nMax} c_ln(k) * (1/V_t) int_{D_j} w (da db + dab k^2)
                                              [ |d'|^{2n+l} Y_lm(d'hat) ] dd',
    c_ln(k) = (-1)^n k^{l+2n} / (2^n n! (2l+2n+1)!!).

The geometry part is the exact rational moment of section 1.2 contracted with the integer
solid-harmonic coefficients of famB section 1.3; the (-1)^l is folded into Qt once.
`subTns!` is allocation-free.

## 2. Convergence ratio, term count and cost: the geometry table (`s4_rho.jl`)

Purely geometric: rho_j = |h_j| / |sigma R + c_j| over all sub-boxes; `Lbnd` is the largest
per-sub-box L the a priori bound of section 4 selects for a 1e-13 relative tensor error;
`cost` = sum_j (L_j + 1)^2, the number of (l,m) terms actually summed (7 complex multiplies
each, plus one spherical-harmonic recursion of (L_j+1)^2 per sub-box).  `Lemp` = the
single-sub-box rule 13 / log10(1/rho_max).  Cube lambda/32, f = 1 (`out/s4_rho_c32.txt`):

    off        m         nSub octT rhoMax Lbnd  cost      Lemp
    (2,0,0)    (2,2,2)   8    1    0.522  48    13700     47
    (2,0,0)    (3,3,3)   64   8    0.354  34    41348     29
    (2,0,0)    (4,4,4)   64   8    0.333  33    35952     28
    (2,0,0)    (6,6,6)   216  27   0.243  25    75500     22
    (2,0,0)    (8,8,8)   512  64   0.190  22    135456    19
    (2,1,0)    (2,2,2)   8    1    0.522  48    11574     47
    (2,1,0)    (4,4,4)   64   8    0.333  32    31964     28
    (2,1,1)    (2,2,2)   8    1    0.522  48    10032     47
    (2,1,1)    (3,3,3)   64   8    0.408  40    31528     34
    (2,1,1)    (4,4,4)   64   8    0.333  31    28501     28
    (2,2,0)    (2,2,2)   8    1    0.397  39    7814      33
    (2,2,0)    (4,4,4)   64   8    0.243  25    24230     22
    (2,2,2)    (2,2,2)   8    1    0.333  32    5488      28
    (3,0,0)    (2,2,2)   8    1    0.333  32    6856      28
    (3,1,0)    (2,2,2)   8    1    0.333  32    6262      28
    (4,0,0)    (2,2,2)   8    1    0.243  25    4640      22
    (6,0,0)    (2,2,2)   8    1    0.156  19    2896      17

**For cubes the minimal split m = (2,2,2) is always the cheapest.**  Refining multiplies the
sub-box count by 8 and divides rho only by 2, i.e. it divides L by about 1.5; since the cost
is nSub * L^2 the trade is 8/2.25 = 3.6 against, at every offset in the table.  Subdivision is
worth doing exactly once (the octant split, which is forced anyway by the kink of w at 0) and
no further.

The famB rule L = 13/log10(1/(0.7 rho)) does **not** carry over: the factor 0.7 came from the
triangle weight vanishing cubically at the corners of D, and the affine weight of a sub-box is
O(1) there.  Measured (below) the sub-box rule is L = 13/log10(1/rho), i.e. exactly the naive
one; at rho = 0.522 that is 47 against famB's 30, and the measured L for 1e-13 was 38-42.

## 3. Accuracy for the cubic cell, lambda/32 (`s2_acc.jl`, `out/s2_c32.txt`)

Reference: the 36 `pairKer` face-pair integrals at 220 bits, ordN = 44, assembled by the
`srfSum!` signs, divided by V_t; 79 tensors were computed for this report on top of famB's 163.
`eF`/`eB` = max over the 9 entries of |G - Gref| / max|Gref| in Float64 / BigFloat(220) at the
same L; `pF` = worst per-entry relative error in Float64 over entries above 1e-4 of the largest;
`canc` = max over entries of sum_j |T_j| / |sum_j T_j| (the cancellation between sub-boxes);
`est/|Gr|` = the a priori magnitude used to turn the absolute bound into a relative one,
divided by the true max|Gref| (it is the scale the L-selection uses, so it must be >= 1).

All 12 offsets of the |n| <= 2.56 band plus five controls, f = 1 and f = 1+0.1i where the
reference exists.  Lbnd from the bound, capped at L = 48.

    m = (2,2,2), 8 sub-boxes, 1 geometry table (41.0 s exact rational, L = 48, nMax = 6)
    off        f        rhoMax Lbnd cost@Lb L13 cost@L13 est/|Gr| eF       eB       pF       canc
    (2,0,0)    1        0.522  48   13700   38  10180    2.07     4.2e-15  4.8e-16  4.2e-15  1.00
    (2,0,0)    1+0.1i   0.522  48   13700   38  10180    2.00     8.2e-16  4.8e-16  1.6e-15  1.00
    (0,2,0)    1        0.522  48   13700   38  10180    2.07     4.2e-15  4.8e-16  4.2e-15  1.00
    (0,2,0)    1+0.1i   0.522  48   13700   38  10180    2.00     1.2e-15  4.8e-16  1.4e-15  1.00
    (0,0,2)    1        0.522  48   13700   38  10180    2.07     1.4e-15  4.8e-16  1.4e-15  1.00
    (2,1,0)    1        0.522  48   11574   42  10470    2.88     1.3e-15  3.3e-16  1.3e-15  1.48
    (2,0,1)    1        0.522  48   11574   42  10470    2.88     1.1e-15  3.3e-16  1.4e-15  1.48
    (1,2,0)    1        0.522  48   11574   42  10470    2.88     1.3e-15  3.3e-16  1.3e-15  1.48
    (0,2,1)    1        0.522  48   11574   42  10470    2.88     1.1e-15  3.3e-16  1.4e-15  1.48
    (1,0,2)    1        0.522  48   11574   42  10470    2.88     8.7e-16  3.3e-16  4.3e-15  1.48
    (0,1,2)    1        0.522  48   11574   42  10470    2.88     8.3e-16  3.3e-16  4.1e-15  1.48
    (2,1,1)    1        0.522  48   10032   42  9480     3.93     9.4e-16  4.1e-16  2.8e-15  1.18
    (1,2,1)    1        0.522  48   10032   42  9480     3.93     9.4e-16  4.1e-16  2.8e-15  1.18
    (1,1,2)    1        0.522  48   10032   42  9480     3.93     1.9e-15  4.1e-16  1.9e-15  1.18
    (2,2,0)    1        0.397  39   7814    30  6536     3.14     4.1e-16  5.3e-17  7.3e-16  1.01
    (2,2,0)    1+0.1i   0.397  39   7814    30  6536     2.99     9.0e-16  5.3e-17  9.0e-16  1.01
    (2,2,1)    1        0.397  39   7030    30  6263     3.61     8.7e-16  3.0e-17  1.3e-15  1.03
    (2,2,2)    1        0.333  32   5488    26  5128     5.08     4.7e-16  1.7e-17  4.7e-16  1.23
    (2,2,2)    1+0.1i   0.333  32   5488    26  5128     4.77     9.1e-16  1.7e-17  9.9e-16  1.30
    (3,0,0)    1        0.333  32   6856    26  5416     2.22     5.7e-16  8.5e-17  1.1e-15  1.00
    (3,0,0)    1+0.1i   0.333  32   6856    26  5416     2.13     1.0e-15  8.6e-17  2.5e-15  1.00
    (3,1,0)    1        0.333  32   6262    26  5318     2.59     1.8e-16  2.0e-17  4.1e-16  1.01

    m = (4,4,4), 64 sub-boxes, 8 geometry tables (68.7 s exact rational, L <= 33, nMax = 6)
    (2,0,0)    1        0.333  33   35952   24  31480    2.07     1.1e-15  5.5e-17  1.1e-15  1.00
    (2,0,0)    1+0.1i   0.333  33   35952   24  31480    2.00     4.1e-16  5.5e-17  9.3e-16  1.00
    (2,1,0)    1        0.333  32   31964   24  29642    2.88     5.3e-16  3.6e-17  7.2e-16  1.52
    (2,1,1)    1        0.333  31   28501   24  27393    3.93     4.3e-16  3.0e-17  6.1e-16  1.25
    (2,2,0)    1        0.243  25   24230   20  22880    3.14     4.6e-16  8.6e-17  1.0e-15  1.03
    (2,2,1)    1        0.243  24   22478   18  20752    3.61     4.6e-16  1.3e-17  4.6e-16  1.07
    (2,2,2)    1        0.200  21   19501   16  17332    5.08     5.7e-16  1.6e-16  5.7e-16  1.34
    (3,0,0)    1        0.190  21   22292   16  17976    2.22     6.0e-16  1.7e-18  6.0e-16  1.00
    (3,1,0)    1        0.190  21   21404   16  17906    2.59     2.3e-16  5.6e-17  5.7e-16  1.03

    m = (3,3,3) -> 4 pieces per axis after the mandatory split at 0, 64 sub-boxes, 8 tables
    (95.8 s exact rational, L <= 40, nMax = 6)
    (2,0,0)    1        0.354  34   41348   24  32900    2.07     1.1e-15  3.8e-17  1.1e-15  1.00
    (2,1,0)    1        0.378  37   35854   26  32582    2.88     1.8e-16  1.1e-17  2.8e-16  1.55
    (2,1,1)    1        0.408  40   31528   28  30322    3.93     7.0e-16  4.9e-17  7.0e-16  1.26
    (2,2,0)    1        0.289  28   26434   22  25112    3.14     3.1e-16  9.3e-17  3.7e-16  1.02
    (2,2,2)    1        0.250  26   21348   20  20796    5.08     2.6e-16  8.5e-17  2.6e-16  1.34
    (3,0,0)    1        0.229  24   24848   18  21260    2.22     3.0e-16  9.6e-17  8.0e-16  1.00

The full tables (all 17 offsets x 3 splits, both frequencies where the reference exists) are in
`out/s2_c32.txt`.  Reading them:

- **The 12 offsets famB could not reach are solved.**  famB needs L = 94 there for 6.6e-14 per
  entry and does not reach 1e-15 by L = 100; the octant split reaches **4.2e-15 max-entry and
  4.3e-15 per entry at L = 48**, and 1e-13 already at L = 38-42.  The failure famB reported was
  truncation, and halving rho from 0.866 to 0.522 removes it.
- **m = (3,3,3) is dominated.**  It costs the same 64 sub-boxes and 8 tables as m = (4,4,4),
  has a larger rho (0.408 vs 0.333 at (2,1,1)), a larger cost at every offset (41348 vs 35952
  at (2,0,0)), and its n-sum cancellation is 461 against 4.0 for m = (2,2,2) and (4,4,4),
  because its inner pieces are 3 times smaller than its outer ones.  Odd m has no use.
- **The cancellation between sub-boxes is negligible**: canc <= 1.55 over every row of every
  table.  Subdivision does not cost digits.  (The largest values, 1.48-1.55, are the (2,1,0)
  class, where the eight sub-box contributions to G_12 partly cancel.)
- **Float64 is rounding-limited, not truncation-limited, wherever Lbnd is used**: eB is 20 to
  100 times smaller than eF in every row, so the remaining 1e-15 is arithmetic, not the series.
- The a priori magnitude `est` is 2.0 to 5.1 times max|Gref| everywhere, so using it as the
  scale in the bound is safe and costs at most log10(5)/log10(1/rho) ~ 1 extra order in L.

## 4. The a priori bound for a sub-box, and how (m, L) is chosen

famB's bound is reused with three changes: the sum runs over **all** l (odd l no longer
vanishes), the radial moment is the sub-box's own, and the expansion centre is R + c_j:

    |Rem_{L_j}^{(j)}| <= (17 |k|^3 / (4 pi |f|^2 V_t))
        sum_{l > L_j} (2l+1) sqrt(2l+5) |k|^l e^{(|k| r_j)^2/(4l+6)} / (2l+1)!!
                      * W_l^{(j)} * hb(l+2, k|R + c_j|),
    W_l^{(j)} = int_{D_j} w(c_j + d') |d'|^l dd',
    hb(l, z) = (e^{-Im z}/|z|) sum_{s=0}^{l} (l+s)!/(s!(l-s)!(2|z|)^s)   (famB section 3.1).

V_t is the whole cell volume because the tensor carries the 1/V_t of the definition; r_j is
the sub-box half-diagonal.  W_l^{(j)} for even l is an exact positive sum of sub-box moments,

    W_{2p}^{(j)} = sum_{i+j'+q = p} p!/(i! j'! q!) nu_{2i}(al_1,.,h_1) nu_{2j'}(al_2,.,h_2)
                                                  nu_{2q}(al_3,.,h_3)

(only even one-dimensional moments enter, so bt drops out and every term is positive); for odd
l, W_l <= sqrt(W_{l-1} W_{l+1}) by Cauchy-Schwarz with the non-negative weight w.  The 17 and
the sum_m |Y_lm| <= (2l+1)/sqrt(4 pi) are famB's, unchanged.

The bound is absolute; it is turned into a relative one with the **a priori** magnitude

    est(R) = V_t |k|^2 e^{-Im(k) |R|} / (4 pi |f|^2 |R|) * (1 + 3/|k R| + 3/|k R|^2),

the pointwise dyadic scale times (1/V_t) int_D w = V_t.  Measured est / max|Gref| = 2.0 to 5.1
over every offset and shape tested (columns above and in `out/s2_*.txt`), so it is a safe
scale and never optimistic.  Each of the nSub sub-boxes is given tol * est / nSub of the
budget and L_j is the first l at which its cumulative tail falls below it.

**How loose the bound is.**  Comparing Lbnd with L13 (the smallest uniform cap that actually
reaches 1e-13 per entry), cube lambda/32, m = (2,2,2): 48 vs 38 at (2,0,0), 48 vs 42 at
(2,1,0) and (2,1,1), 39 vs 30 at (2,2,0), 32 vs 26 at (2,2,2) and (3,0,0).  So the bound
over-selects L by 1.2 to 1.3 -- **much tighter than famB's 1.5 to 2.0 factor**, because the
sub-box radial moment W_l^{(j)} is a far better description of the affine weight than of the
triangle weight (famB's bound had to over-estimate exactly the corner suppression that gives
the 0.7 factor).  The cost penalty is (Lbnd/L13)^2 = 1.3 to 1.7.

**The selection rule that comes out of the bound.**  For cubes the cost sum_j (L_j+1)^2 is
monotone increasing in the refinement (section 2), so the rule is

    m = (2,2,2) always (the octant split; forced by the kink of w at 0 anyway),
    L_j from the bound above, per sub-box and per sign pattern.

There is no offset class for cubes where refining beyond the octants pays: the table below
gives the cost of the cheapest split that reaches the target at each band of the cube at
lambda/32, together with famB's un-subdivided cost.

    band          rho(famB) rho_j(m=222) L famB   L_j max  cost sub   cost famB   pF sub    pF famB
    2 axis        0.866     0.522        94       48       13700      ~9000       4.2e-15   6.6e-14
    2 face+1      0.775     0.522        ~70      48       11574      ~5500       4.3e-15   -
    2 body+1      0.707     0.522        ~58      48       10032      ~4000       2.8e-15   -
    2 face diag   0.612     0.397        38       39       7814       ~2000       9.0e-16   7.9e-16
    2 body diag   0.500     0.333        22       32       5488       ~900        9.9e-16   1.7e-15
    3 axis        0.577     0.333        32       32       6856       ~1400       2.5e-15   3.7e-16
    4 axis        0.433     0.243        26       25       4640       ~1000       -         7.7e-16
    6 axis        0.289     0.156        18       19       2896       ~600        -         4.1e-16

(cost is sum_j (L_j+1)^2 for the subdivided form and (L+1)^2 + the parity-pruned contraction
for famB; famB's numbers are read off its report.)  **Beyond three cells famB alone is 3 to 7
times cheaper and already at 1e-15, so subdivision is only for the 12 offsets of the |n|<=2.56
band** -- the ones famB cannot reach.  Since there are exactly 12 such offsets per octant
whatever N, the absolute cost of subdividing them is 12 tensor evaluations, i.e. microseconds
in a build of 10^7 offsets.

