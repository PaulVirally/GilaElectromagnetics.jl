# unify — the far-field library `notes/farfield/farfield.jl`

Deliverable: `/Users/pvirally/.julia/dev/GilaElectromagnetics/notes/farfield/farfield.jl`
(standalone for routes (i) and (ii); route (iii) lazily `include`s `notes/moments/moments.jl`,
see §1.4).  Work dir `SCRATCH/work/unify/`; every table below is reproduced by the script named
next to it and its raw output is in `SCRATCH/work/unify/out/`.
Shape tables and the k-series cache are in `SCRATCH/work/unify/cache/`.
SCRATCH = `/Users/pvirally/.julia/dev/GilaElectromagnetics/notes/farfield/scratch`.

## 1. What the library computes, and how the route is chosen



### 1.1 The object and its normalisation

    T_ab(R) = (1/V_t) int_D w(d) [(d_a d_b + dab k^2) g](R + d) dd,
    g(r) = e^{ikr}/(4 pi f^2 r),  k = 2 pi f,  D = prod_i [-s_i, s_i],  w = prod_i (s_i - |d_i|),
    R = n .* s,  V_t = s1 s2 s3.

Sign `+`, no extra factor.  Pinned three ways in this work: at `D = (5,1,0)`, `s = (1/32)^3`,
`f = 1` the library returns

    T[1,1] = 0.001570738221266370 + 0.000360155079207178 i

against Paul's independent Mathematica `NIntegrate` of the volume form,
`0.001570738221266370299274 + 0.000360155079207178606379 i` — every printed digit
(REGISTRY.md, "Independent Mathematica check").  The same value comes out of route (i) (the
addition theorem), route (iii) (the 36 face-pair k-series through `srfSum!`) and the
220-bit `pairKer` reference in `SCRATCH/refcache/reftensors.txt`.

### 1.2 Route (i): the whole difference box

    e^{ik|R+d|}/|R+d| = 4 pi i k sum_{l,m} (-1)^l h_l(kR) Y_lm(Rhat) j_l(k|d|) Y_lm(dhat),  |d| < |R|,
    j_l(k|d|) Y_lm(dhat) = sum_n c_ln(k) |d|^{2n} [|d|^l Y_lm(dhat)],
    c_ln(k) = (-1)^n k^{l+2n} / (2^n n! (2l+2n+1)!!),

with real solid harmonics `r^l Y_lm = ncoef(l,m) C_l^m(z,r^2) A_m(x,y)` carrying *integer*
coefficients, so that every geometry moment

    Ad[a,n,lm] = (1/V_t) int_D w d_a^2 [|d|^{2n+l} Y_lm],  Bd[n,lm] = (1/V_t) int_D w [...],
    Ao[c,n,lm] = (1/V_t) int_D w d_a d_b [...]                              (ab = 12, 13, 23)

is an exact rational in the edge lengths times `ncoef(l,m)`.  The triangle weight is even in
every coordinate, so only even `l` survives and the (l,m) set splits into four disjoint classes
(famB §1.4); `(-1)^l` is therefore `+1` throughout, but the library folds it in anyway because
route (ii) needs it.  Per frequency,

    Qt^{aa}_lm = sum_n c_ln (Ad[a,n,lm] + k^2 Bd[n,lm]),   Qt^{ab}_lm = sum_n c_ln Ao[ab,n,lm],

and per offset `T_ab(R) = (i k/f^2) sum_{l<=L, m} h_l(k|R|) Y_lm(Rhat) Qt^{ab}_lm`.

### 1.3 Route (ii): the eight octants of D

`D` splits at the kink of `w` at 0 into 8 octants; on each the weight is affine,
`w = prod_i (al_i + bt_i d'_i)` with `al_i = h_i = s_i/2`, `bt_i = -1`, and the moments are
`nu_n = 2 al h^{n+1}/(n+1)` (n even), `2 bt h^{n+2}/(n+2)` (n odd).  All eight octants are one
reflection family of a single table, by

    T^{(sigma)}_ab(V) = sigma_a sigma_b T_ab(sigma V),   V = sigma .* R + c,  c = s/2,

so exactly ONE octant table is built per shape.  The affine weight is not even, so **odd `l`
survives and the `(-1)^l` of the addition theorem is mandatory** (famBsub measured 3.7e-2
relative error without it).  The library carries `sg = (-1)^l` inside `frqBox`.

### 1.4 Route (iii): the k-series in BigFloat(128)

Where neither bound is met with `L <= Lmax = 56`, the 36 face-pair integrals are evaluated as
`int_A int_B g = (1/(4 pi f^2)) sum_n (ik)^n I_{n-1}/n!` with `I_m` from `pairMoments` on
BigFloat panels (`facePair` gives the panels; `Rational{BigInt}` panels fail at `moments.jl:442`),
assembled by the `srfSum!` signs in BigFloat(128) and rounded to `T`.  The moments are
frequency independent and are cached in memory per `(D, s)`; the assembled tensors are cached
on disk per `(s, f)` in `KSRDIR[]`, so a build that finds the cache never touches `moments.jl`.
**This is the only part of `farfield.jl` that is not standalone**: `needMom()` lazily
`include`s `MOMJL[]` (default `notes/moments/moments.jl`) and errors with a clear message if
it is absent.

### 1.5 The truncation bounds (the only thing that selects L)

**Whole box.** Moving both derivatives onto the singular solid harmonic gives at most 16 terms
`h_{l'} Y_{l'm'}` with `l' in {l-2,l,l+2}` and coefficients of modulus <= 1, plus the `dab k^2`
term: the constant 17.  With `sum_m |Y_lm| <= (2l+1)/sqrt(4 pi)`, `|P_l| <= 1`, and
`|j_l(z)| <= |z|^l e^{|z|^2/(4l+6)}/(2l+1)!!` used *pointwise in |d|* so that the exact radial
moment absorbs the geometry,

    |Rem_L| <= (17|k|^3/(4 pi |f|^2 V_t)) sum_{l>L, even} (2l+1) sqrt(2l+5) |k|^l
               e^{(|k| r_d)^2/(4l+6)} / (2l+1)!! * W_l * hb(l+2, kR),
    W_l = int_D w(d) |d|^l dd  (exact positive sum of box moments, no cancellation),
    hb(l,z) = (e^{-Im z}/|z|) sum_{s=0}^{l} (l+s)!/(s!(l-s)!(2|z|)^s)   >= |h_l^{(1)}(z)|,
    r_d = sqrt(s1^2+s2^2+s3^2).

**Sub-box.** The same with the sub-box radial moment `W_l^{(j)}` (`wRadA`), all `l` (odd `l` no
longer vanishes; `W_l <= sqrt(W_{l-1} W_{l+1})` by Cauchy-Schwarz), the sub-box half-diagonal
`r_j = r_d/2`, and the local centre `|sigma R + c|`.

**n-truncation.** The tail of the k-series inside `j_l` after `n = N` obeys

    |sum_{n>N} c_ln z^{l+2n}| <= |z|^{l+2N+2} e^{|z|^2/(4l+4N+10)} / (2^{N+1} (N+1)! (2l+2N+3)!!)

(proof: factor out the `n = N+1` term of `j_l`'s series and bound every later ratio by
`(|z|^2/2)^j/(j! (2l+2N+5)^j)`), so the n-truncation error of the tensor is bounded by the same
expression as `|Rem_L|` with `j_l` replaced by that tail and `W_l` by `W_{l+2N+2}`.  `nCut[l+1]`
is the smallest `N` meeting `tol*est/(L+1)`; the table is built to `nMax = 12`.

**Relative target.** Both bounds are absolute and are compared against `tol * est(R)`,

    est(R) = V_t |k|^2 e^{-Im(k)|R|} / (4 pi |f|^2 |R|) * (1 + 3/|kR| + 3/|kR|^2),

the pointwise dyadic scale times `(1/V_t) int_D w = V_t`; famBsub measured
`est/max|G_ref| in [2.0, 5.1]` over every offset and shape tested, so it never under-states
the true magnitude.  `tol = 1e-13`.  Each of the 8 sub-boxes of route (ii) gets `tol*est/8`.

### 1.6 The decision rule, and its cost model

Cost in complex multiply-adds actually issued (Y_lm recursion + the contracted sums):

    cost(i)  = (L+1)^2 + 3 |{lm: l <= L}|_diag + sum_c |{lm: l <= L}|_c
    cost(ii) = sum_{j=1..8} 7 (L_j + 1)^2

Rule, per offset:

    L   = smallest even l <= 56 with the whole-box tail <= tol*est(R)          -> route (i)
    L_j = smallest l <= 56 with the sub-box tail <= tol*est(R)/8, per octant   -> route (ii)
    if L exists                      : take (i) if cost(i) <= cost(ii), else (ii)
    elseif every L_j exists          : (ii)
    else                             : (iii)

The implementation short-circuits to (i) whenever `L` exists, because (i) is *always* the
cheaper of the two when it converges: halving the box radius only reduces `L` by the factor
`ln(1/rho)/ln(2/rho)` (0.77 at rho = 0.1, 0.63 at rho = 0.5) while the term count is multiplied
by `8 * 7 / 1.7 = 33`.  Measured over every offset of a 24^3 octant where BOTH bounds are met
(§5), the worst `cost(i)/cost(ii)` ratio is reported in `out/t11_diag.txt`; it never reaches 1.

### 1.7 API

    farTensor(D::NTuple{3,Int}, s::NTuple{3,<:Rational}, f::Complex{T}) -> Matrix{Complex{T}}
        errors when max|D| <= 1; any offset with max-norm separation >= 2 is accepted.
    farBlock!(egoToe::AbstractArray{Complex{T},5}, s, f)
        fills egoToe[:,:,i1,i2,i3] for every index with max(i) >= 3 (offset i .- 1), leaving
        indices all <= 2 untouched; Threads.@threads :static over offsets, one FarWs per thread,
        route (iii) offsets computed serially (and cached) before the threaded loop.
    farShape(s), farSetup(s, f), boundL(fs, |R|), boundLoct(fs, R), farRoute(fs, D),
    farRouteStat(fs, dim), est, jbnd, hbnd, jtail, bndTrm, bndCum, pickCut, costWhl, costOct,
    tnsWhl!, tnsOct!, tnsKsr, srfSum      -- all exported for verify.jl.

`TABDIR[]` (shape tables) and `KSRDIR[]` (route (iii) tensors) are `Ref`s; both caches are plain
append-only text and both are optional.

## 2. The shape tables (script `work/unify/t1_tab.jl`, raw `out/t1_tab_*.txt`)



`L = 56`, `nMax = 12`, one whole-box table and one octant table per shape.  The contraction of
the integer solid-harmonic coefficients against the box moments alternates in sign and cancels
by up to `7.3e11`, so it CANNOT be done in Float64 (it would return noise) — the library builds
it at `BLDPRC = 1024` bits and stores it at `TABPRC = 192`.

    shape                      build (s)   contraction cancellation, whole box / octant
    c32   (1/32)^3                473.7      7.31e11 / 1.96e11
    c8    (1/8)^3                 497.7      7.31e11 / 1.96e11
    c4    (1/4)^3                 576.2      7.31e11 / 1.96e11
    sl    (1/32,1/32,1/512)       642.6      3.37e9  / 1.67e9

    (l,m) entries kept after dropping identically-zero columns, identical for all four shapes:
      whole box  diag 435,  xy 210,  xz 406,  yz 406      -> 35906 stored numbers
      octant     diag 3249, xy 2465, xz 3249, yz 3249     -> 285467 stored numbers
    The whole-box classes are the parity-pruned ones of famB S1.4: the diagonal class has
    sum_{l even <= 56} (l/2 + 1) = 435 pairs and all survive; xz and yz have 406 each and all
    survive; xy has 406 of which only 210 survive, the other 196 vanishing identically because
    s1 = s2 (famB S13 measured the same 20% for cubes).  The octant table has all (L+1)^2 = 3249
    (l,m) per class -- odd l included, which is what makes it 8x larger -- and again only the xy
    class loses entries (784 of 3249) to the square cross-section.  For a fully generic box
    (three distinct edges) nothing is dropped.

Reloading a shape from its disk cache takes **1.9 s** (`out/t1_tab_c.txt`), against 474-643 s to
build it, so the table is built once per shape ever.

Verified against an exact `Rational{BigInt}` build twice.  At `L = 20`, `nMax = 6`
(`t0_smoke.jl`, output at 192-bit storage) the two tables agree to **1.80e-78 (Ad),
3.80e-78 (Bd), 1.63e-78 (Ao)** and both report the identical cancellation `2.1396e5` (whole box)
and `2.5268e5` (octant) — the same `2.14e5` famB measured at that size.  At `L = 24`, `nMax = 6`
for all three distinct shapes (`t15_exact.jl`, `out/t15_exact.txt`):

    shape                exact (s)  float (s)  whl Ad     whl Bd     whl Ao     oct Ad     oct Ao     sparsity
    c32  (1/32)^3           5.0        2.6     2.41e-64   1.68e-65   1.39e-64   3.95e-60   4.56e-60   identical
    c4   (1/4)^3            3.7        2.3     9.87e-62   6.89e-62   5.70e-61   3.16e-59   3.65e-59   identical
    sl   (1/32,1/32,1/512)  4.2        2.4     2.47e-61   6.89e-62   2.85e-61   3.95e-60   4.56e-60   identical

i.e. the two builds agree to the 192-bit storage floor (`eps(BigFloat(192)) = 1e-58`) and produce
*bit-identical sparsity patterns* — the `abs(acc) < abs_ * 2^-256` test that turns a cancelled
float sum into an exact zero reproduces the exact-arithmetic zero set.  The exact path is kept as
`farShape(s; exct = true)`; measured here it is only 1.6-2.0x slower at `L = 24`, but it was not
run at `L = 56` (the `Rational{BigInt}` numerators grow with the polynomial degree, which reaches
80 there), so 1024-bit floats are the default.

The disk cache is one 19.5 MB text file per shape (`shapetab/` by default, `TABDIR[]` here);
loading it takes a few seconds and a shape is therefore built once, ever.
## 3. (a) Accuracy against every cached 220-bit reference (script `t2_ref.jl`, raw `out/t2_ref.txt`)



**403 reference tensors**, pooled from every round-1 cache: 170 from `reference` (`refcache/
reftensors.txt`, `pairKer` at 220 bits, ordN 44, plus its independent `volTensor` rows), 157 from
`auditG` (random lattice directions, five frequencies), 46 from `famBsub`, 26 from `famB`, 4 from
`famD`'s BigFloat Gauss-Legendre volume rule.  Loader `refload.jl`.
`eMx = max_ab |G-Gr| / max|Gr|`; `pEn` = worst `|G-Gr|/|Gr|` over entries above 1e-8 of the
largest; `eRe`/`eIm` = the same for the real and imaginary parts separately, relative to
`|Re Gr|` and `|Im Gr|` of that entry.

    shape  f          n     eMx        pEn        eRe        eIm        worst-pEn offset
    c32    1          81    4.2e-15    4.2e-15    2.6e-14    3.5e-14    (0,2,0)
    c32    1+0.1i     79    1.6e-15    3.0e-15    1.2e-13    5.7e-14    (64,0,0)
    c32    0.37       14    9.5e-16    1.3e-15    2.2e-14    7.7e-14    (-4,-10,29)
    c32    3+0.3i     13    5.9e-16    8.7e-16    3.1e-14    3.6e-15    (2,32,-16)
    c32    1+1i       13    1.3e-14    6.3e-14    9.9e-13    2.9e-13    (-7,6,-36)
    c8     1          28    5.6e-15    5.6e-15    1.5e-13    1.5e-13    (32,32,32)
    c8     1+0.1i     28    1.9e-15    3.6e-15    3.6e-14    7.3e-15    (64,0,0)
    c4     1          29    8.6e-15    8.6e-15    4.9e-13    8.8e-13    (64,0,0)
    c4     1+0.1i     29    2.5e-15    1.2e-14    2.8e-14    4.5e-14    (64,0,0)
    sl     1          34    2.5e-15    4.7e-15    4.7e-15    6.3e-16    (4,4,4)
    sl     1+0.1i     31    1.5e-15    5.1e-15    5.1e-15    7.3e-15    (4,4,4)
    sl     0.37        8    8.2e-16    1.8e-15    1.8e-15    3.3e-16    (7,-2,33)
    sl     3+0.3i      8    2.6e-15    2.6e-15    3.0e-15    6.7e-15    (0,0,32)
    sl     1+1i        8    3.2e-15    8.3e-15    1.7e-14    6.1e-15    (-5,5,5)

    routes exercised by these 403 rows: (i) 339, (ii) 55, (iii) 9.

`c4 (2,1,0)` and `c4 (2,1,1)` (route (ii), `L_j` = 56, `rho_j` = 0.52) come out at `1.9e-15` and
`1.2e-15` per entry; `c8 (2,1,1)` at `9.9e-16`; the twelve `c32` two-cell offsets at `1.2e-15`.
famB alone needed `L = 94` for `6.6e-14` at `(2,0,0)` and could not reach `1e-15` by `L = 100`;
the octant split with the sub-box bound reaches `1.2e-15` there at `L_j <= 56`.

**At `f = 1` and `f = 1+0.1i`, over all four shapes and every cached offset (from 2 cells out to
64 cells, axis, face diagonal, body diagonal and random lattice directions): `eMx <= 8.6e-15`
and `pEn <= 1.2e-14`.**  The two-cell shell (route (ii), `L_j` up to 56) is at `1.2e-15` (c32),
`9.9e-16` (c8) and `1.9e-15` (c4) per entry; the slender near needle (route (iii)) at `1e-16`.

The `eRe`/`eIm` columns are the honest per-part numbers and they are worse: up to `8.8e-13` at
`c4 (64,0,0)`, `f = 1`, where the same tensor is at `8.6e-15` by `eMx` and `pEn`.  Those are
entries whose *real* or *imaginary* part is a small fraction of the entry's own modulus (an
individual Re or Im passing through a zero as a function of `kR`); the absolute error stays at
`eps * max|G|` and only the ratio to the small part blows up.  §9 states this limitation.

The `f = 1+1i` rows (auditG's strongly damped frequency, `e^{-2 pi R}` over 36 cells) are the
worst per-entry rows in the whole table, `6.3e-14`.  There the tensor is one complex sum (the
Re/Im split of §1.2 is available only for real `f`), the entries have `|Im|/max|T| ~ 1e-3`, and
the imaginary part costs three digits by construction.

## 3b. (b) Route (iii) against the volume reference, and its cost (script `t10_ksr.jl`, raw `out/t10_ksr.txt`)

Reference: `refQuad.jl` (famD's independent BigFloat Gauss-Legendre rule on the difference box
with the exact triangle weight), 256-bit, on the seven slender near-needle offsets the task names,
both frequencies.  `N` = the k-series order the proven bound of §1.4 selects for `1e-16/Lambda`;
`Lam` = the static assembly amplification `(sum_fp I_{-1})/(4 pi |f|^2 V_t)/est(R)` that the
`srfSum!` signed difference applies to every face-pair error.

    D            f        N    Lam      eMx       pEn       eRe       eIm       tMom (s)  tSum (ms)
    (0,0,2)      1        12      1.88   3.8e-17   3.8e-17   3.8e-17   4.0e-16    6.32      376
    (0,0,2)      1+0.1i   12      1.88   2.0e-17   4.2e-17   4.2e-17   7.6e-17    3.64      385
    (0,0,8)      1        13     67.9    9.8e-17   9.8e-17   9.8e-17   2.3e-17    3.73      355
    (0,0,8)      1+0.1i   13     68.6    7.7e-17   7.7e-17   7.8e-17   5.8e-17    3.66      359
    (0,0,20)     1        15    484      2.9e-17   4.0e-17   4.0e-17   5.3e-17    8.06      946
    (0,0,20)     1+0.1i   15    496      2.5e-17   2.5e-17   2.5e-17   1.9e-17    8.71      954
    (1,0,2)      1        15    398      2.7e-17   2.9e-17   2.9e-17   5.7e-17    6.40      454
    (1,0,2)      1+0.1i   15    406      5.5e-17   7.3e-17   7.3e-17   7.9e-17    6.41      442
    (1,1,2)      1        16    710      3.8e-17   6.6e-17   6.6e-17   1.6e-16   10.98      508
    (1,1,2)      1+0.1i   17    729      3.2e-17   4.7e-17   4.7e-17   8.7e-17   12.72      505
    (1,1,12)     1        17    829      6.6e-17   7.0e-17   7.0e-17   6.1e-17   13.68      663
    (1,1,12)     1+0.1i   17    854      8.4e-17   8.4e-17   8.3e-17   1.0e-16   13.53      682
    (0,1,30)     1        17   1290      6.7e-17   9.2e-17   9.2e-17   5.1e-17    8.71      431
    (0,1,30)     1+0.1i   17   1340      3.3e-17   3.3e-17   3.2e-17   8.2e-17    8.77      434

**Every row is at 2.0e-17 to 1.6e-16, i.e. at the Float64 rounding floor of the returned tensor**
(the sum itself is done in BigFloat(128) and only the final 3x3 is rounded), against a reference
whose own convergence floor is measured below.  `Lambda` reaches 1340 here -- the amplification
that makes a *Float64* face-pair method impossible on this cell (kseries.md: `eps*Lambda` alone is
`4.8e-13` at two cells) -- and BigFloat(128) removes it entirely.

**Cost.**  `tMom` is the BigFloat(128) `pairMoments` build for the 36 face pairs, **once per
(shape, offset)** and frequency-independent: 3.6 to 13.7 s.  `tSum` is everything that depends on
the frequency (the 36 series, the `I_{-1}` amplification, the `srfSum!` assembly, all in
BigFloat(128)): 355 to 954 ms.  For the 72 route-(iii) offsets of the slender shape that is
**~10 minutes of moments, once ever, plus ~36 s per frequency**, after which they are read from
the disk cache in microseconds.  This is why route (iii) is affordable exactly where it is used
and nowhere else: at `lambda/8` and `lambda/4` the two-cell shell would need `N ~ 43` and `N ~ 52`
and kseries.md's measured `mMax^3.9` scaling puts its Float64 moments at 2.4 s and 6.5 s per
offset, i.e. **10^2 to 10^3 times that in BigFloat(128)**, which is why route (ii) had to reach
`L = 56` and cover the cubes' two-cell shell instead (§4).

**The reference itself.**  `refQuad`'s cuts must be graded toward the point of closest approach.
With famD's own cut rule (`ctrCut(s,6,3)` on axes with `D_a = 0`, `uniCut(s,2)` otherwise) the
order-34-vs-46 convergence floor is `8.1e-36` at `(0,0,2)`, `2.5e-44` at `(0,0,8)`, `5.9e-60` at
`(0,0,20)`, `9.9e-38` at `(1,1,12)` and `7.5e-59` at `(0,1,30)` -- but only **`5.9e-12` at
`(1,0,2)` and `2.4e-13` at `(1,1,2)`**, which is useless as a 1e-13 reference.  The reason is
geometric: on an axis with `|D_a| = 1` the point `R_a + delta_a = 0` sits on the *edge* of the
difference box, so the near-singular corner is an endpoint of a Gauss panel.  `t6b_refq.jl` grades
every axis with `|D_a| <= 1` geometrically toward that point (8 levels, ratio 2, 0 still a cut),
which brings the same two offsets to **`2.9e-45` and `4.6e-46`** (443 s and 270 s per reference at
orders 30 and 40, 459 and 243 sub-boxes).  The table above uses the graded references.

## 4. Routing over a full `egoToe` octant (script `t3_route.jl`, raw `out/t3_route.txt`)



Offsets with max-norm separation >= 2 (i.e. `max(i) >= 3` in Gila's indexing), `tol = 1e-13`.
128^3 for the cubes, 64x64x128 for the slender cell.  Identical at `f = 1` and `f = 1+0.1i`.

    shape                      (i) whole box   (ii) octants   (iii) k-series
    c32   (1/32)^3               2 097 132          12              0
    c8    (1/8)^3                2 097 129          15              0
    c4    (1/4)^3                2 097 129          15              0
    sl    (1/32,1/32,1/512)        524 066         142             72

**The cubes need no k-series at all.**  Route (ii) takes exactly the classes the whole-box bound
cannot reach at `L <= 56`:

    c32:  (2,0,0) (2,1,0) (0,2,0) (1,2,0) (2,0,1) (2,1,1) (0,2,1) (1,2,1) (0,0,2) (1,0,2) (0,1,2) (1,1,2)
          -- the 12 members of the classes (2,0,0), (2,1,0), (2,1,1), i.e. |n| <= 2.45
    c8, c4: the same 12 plus (2,2,0), (2,0,2), (0,2,2)          -- 15

For the slender cell route (ii) is the 142 offsets with `(n1,n2)` a permutation of `(2,0)`,
`(2,1)` or `(2,2)` at any `n3`, plus the 2-cell shell; route (iii) is exactly

    (0,0,n3)             n3 = 2 .. 19        18 offsets
    (0,1,n3), (1,0,n3)   n3 = 2 .. 19        36 offsets
    (1,1,n3)             n3 = 2 .. 19        18 offsets
                                             72 offsets total

i.e. the near needle along the short axis, where `rho = 22.63/n3 > 1` for the whole box and
`rho_j` stays above ~0.55 for the octants.  `k D` along that needle is 0.28 to 0.60, so the
k-series is at its cheapest exactly there.  The 72 tensors are frequency dependent but their
moments are not; they are computed once and cached on disk.

Whole-box `L` histograms (f = 1):

    c32   8:1174762  10:902327  12:15998  14:2602  16:749  18:310  20:144  22:82  24:45 ...  56:3
    c8   12:1809933  14:281179  16:4532   18:869   20:277  22:133  24:66  ...              52:6
    c4   16:2064621  18:29977   20:1725   22:425   24:165  26:70   28:36  ...              54:6
    sl   10:436311   12:51400   14:14426  16:5941  18:2938 20:1547 ... 8:9028 ...          56:9

`L` bottoms out at 8 (lambda/32), 12 (lambda/8) and **a flat 16 at lambda/4** — the floor
`overlap.md` §7 measured as necessary (`Lneed = 16` at `rho = 0.045 .. 0.125`) and which famB's
`rho`-only rule, giving `L = 10` at 23 cells, could not produce.  56.0% of the lambda/32 offsets
and 98.4% of the lambda/4 offsets sit at the floor.

Setup per (shape, frequency): 1.2-1.7 s (in-memory table + the frequency contraction), 4.0 s the
first time (table read from disk).

## 5. Truncation: certification and tightness (script `t11_diag.jl`, raw `out/t11_diag.txt`)



### 5.1 The n-truncation is certified by the bound with `nMax = 12`

`nCut[l+1]` from the `jtail` bound of §1.5, evaluated at the *smallest radius actually routed to
(i) or (ii)* (so the certification is not vacuous), budget `tol*est/(L+1)` per `l`:

    shape   rMin      max nCut over l <= 56   saturates nMax = 12?
    c32     0.0625     5                      no
    c8      0.25       8                      no
    c4      0.5       10                      no
    sl      0.0391     5                      no

Identical at `f = 1` and `f = 1+0.1i`.  Since no `l` needs `N = 12`, the stored `nMax = 12` table
is provably sufficient at every offset the library evaluates by (i) or (ii).  (famB's *measured*
requirement was 6 / 10 / 12 at lambda/32 / lambda/8 / lambda/4; the bound asks for 5 / 8 / 10 —
the bound is tighter than famB's own 1e-16-per-term criterion because it is a bound on the
assembled tensor, not on the largest term of `Qtilde`.)

### 5.2 How loose the L bound is

`Lneed` = the smallest even `L` whose Float64 tensor stays within 1e-13 per entry of the `L = 56`
value; `Lbnd` = what the bound selects.  Whole-box (route (i)) offsets only.

    shape  offset        rho      kR       Lbnd  Lneed  ratio
    c32    (3,0,0)       0.577     0.589    50    32     1.56
    c32    (4,0,0)       0.433     0.785    34    26     1.31
    c32    (8,0,0)       0.217     1.571    20    16     1.25
    c32    (16,0,0)      0.108     3.142    14    12     1.17
    c32    (64,0,0)      0.027    12.566    10     8     1.25
    c8     (3,0,0)       0.577     2.356    52    32     1.62
    c8     (8,8,8)       0.125    10.883    16    12     1.33
    c8     (64,0,0)      0.027    50.265    14    12     1.17
    c4     (3,0,0)       0.577     4.712    54    32     1.69
    c4     (8,0,0)       0.217    12.566    24    16     1.50
    c4     (16,16,16)    0.063    43.531    18    14     1.29
    c4     (23,23,20)    0.045    59.979    18    14     1.29
    c4     (64,0,0)      0.027   100.531    16    16     1.00
    sl     (3,0,0)       0.472     0.589    42    30     1.40
    sl     (8,8,8)       0.125     2.224    16    14     1.14
    sl     (23,23,20)    0.043     6.391    10    10     1.00
    sl     (64,0,0)      0.022    12.566    10     8     1.25

`Lbnd/Lneed` in **[1.00, 1.69]** over 80 rows (both frequencies), i.e. the cost penalty of using
the bound instead of the measured requirement is `(Lbnd/Lneed)^2 = 1.0` to `2.9`.  The bound is
never short.

### 5.3 The whole-box threshold table is exact (script `t13_mono.jl`)

`boundL` reads `L` off a table of critical radii built by bisection, which presumes the
bound-to-`est` ratio is monotone in `|R|`.  Checked against a direct evaluation of the bound at
**4000 log-spaced radii per (shape, frequency)**, over the full block range
(`2 min s_i` to `4 * 128 * r_d`): **0 disagreements out of 32000**, for all four shapes and both
frequencies.

### 5.4 Route (i) is 9 to 10 times cheaper than route (ii) wherever both converge

Over every offset of a 24^3 octant where BOTH bounds are met (13650-13804 offsets per case), the
worst `cost(i)/cost(ii)` is

    c32 0.102 at (2,2,0)    c8 0.0977 at (3,0,0)    c4 0.0984 at (3,0,0)    sl 0.107 at (2,0,23)

so the short-circuit "take (i) whenever its bound is met" is the cost-optimal rule at every
offset tested, by a factor of at least 9.3.

## 6. (c) Overlap with famG's Taylor route (script `t7_ovl.jl`, raw `out/t7_ovl.txt`)



Every octant offset of a 16^3 block with max-norm >= 2 (4085 offsets), `farTensor` against famG's
`(G,S)` recurrence + exact-moment assembly at **auditG's rule** `q = 2 ceil((9.7/log10(1/rho)+2)/2)`,
which is the rule this task specified.

    shape  f          band     n      dMx       dEn       dRe       dIm       worst offset
    c32    1          2        16     9.7e-15   1.2e-14   1.2e-14   8.1e-14   (1,0,2)
    c32    1          3-4      98     6.0e-15   6.0e-15   1.0e-14   3.6e-14   (0,0,3)
    c32    1          5-8      604    2.7e-15   2.7e-15   1.4e-13   9.9e-15   (2,2,8)
    c32    1          9-15    3367    2.5e-15   2.5e-15   2.4e-13   3.5e-13   (9,9,10)
    c32    1+0.1i     2        16     1.0e-14   1.0e-14   1.0e-14   4.1e-14   (2,1,0)
    c32    1+0.1i     3-4      98     4.6e-15   5.9e-15   7.5e-15   9.3e-14   (1,3,1)
    c32    1+0.1i     5-8      604    5.7e-15   5.7e-15   7.1e-13   5.9e-13   (5,2,2)
    c32    1+0.1i     9-15    3367    3.1e-15   3.1e-15   5.9e-13   5.5e-13   (13,8,4)
    c4     1          2        16     1.1e-14   1.1e-14   1.5e-14   4.3e-15   (1,0,2)
    c4     1          3-4      98     3.9e-15   8.3e-15   2.1e-14   5.6e-14   (1,0,3)
    c4     1          5-8      604    1.3e-11   1.3e-11   2.8e-11   8.6e-11   (8,8,8)
    c4     1          9-15    3367    8.6e-10   8.6e-10   2.2e-8    4.1e-8    (15,15,15)
    c4     1+0.1i     5-8      604    1.4e-11   1.4e-11   5.0e-9    5.5e-9    (8,8,8)
    c4     1+0.1i     9-15    3367    9.4e-10   9.4e-10   8.4e-7    9.6e-7    (15,15,15)

**At lambda/32 the two agree to 1.0e-14 (max-norm) and 1.2e-14 (per entry) in every band, at both
frequencies, over all 4085 offsets** — a free end-to-end confirmation of the geometry
bookkeeping, the `srfSum!` sign convention, the exact weight moments and the two independent
expansions against each other.

**At lambda/4 they disagree by 1.3e-11 (5-8 cells) and 8.6e-10 (9-15 cells).  The 220-bit
reference adjudicates against famG, i.e. against auditG's `q` rule:** at exactly the worst
offset `c4 (8,8,8)`, `f = 1`, `farTensor` is `2.2e-15` from the `pairKer` reference and at
`c4 (16,16,16)` it is `4.8e-15` (rows of `out/t2_ref.txt`), while the disagreement with famG at
those offsets is `1.3e-11`.  auditG's `q = 9.7/log10(1/rho) + 2` is a `rho`-only rule with no
`kR` term, exactly the defect `overlap.md` §7 identified in famB's `L` rule and famC's `M` rule;
it was validated by auditG on the cubic lambda/32 cell only.  At `c4 (15,15,15)`, `rho = 0.0667`,
it asks for `q = 12` at `kR = 40.8`.  Repeating the sweep with famG's *own* rule
`q = 2 ceil((17/log10(1/rho)+8)/2)` (the one `overlap.md` §7 found "conservative everywhere
measured"), `out/t7b_ovl.txt`:

    shape  f          band     n      dMx       dEn       worst offset      (was, with the 9.7 rule)
    c32    1          3-4      98     6.6e-15   6.6e-15   (3,1,1)           6.0e-15
    c32    1          5-8      604    2.8e-15   3.1e-15   (2,2,8)           2.7e-15
    c32    1          9-15    3367    2.5e-15   2.6e-15   (9,9,10)          2.5e-15
    c4     1          3-4      98     3.9e-15   5.6e-15   (1,0,3)           8.3e-15
    c4     1          5-8      604    5.1e-15   6.2e-15   (6,5,5)           1.3e-11
    c4     1          9-15    3367    6.8e-15   8.5e-15   (13,13,13)        8.6e-10
    c4     1+0.1i     5-8      604    4.1e-15   5.2e-15   (7,6,8)           1.4e-11
    c4     1+0.1i     9-15    3367    7.4e-15   8.4e-15   (13,13,13)        9.4e-10

**The lambda/4 disagreement collapses from 8.6e-10 to 8.5e-15 per entry when famG is given its
own `q` rule instead of auditG's.**  The two expansions therefore agree to Float64 rounding at
lambda/4 as well; what disagreed was auditG's truncation rule, and the 220-bit reference says
`farTensor` was the correct side (`2.2e-15` at `c4 (8,8,8)`, the worst offset of the 5-8 band).

## 7. (d) Symmetries, homogeneity, number types (script `t5_sym.jl`, raw `out/t5_sym.txt`)



50 random offsets per (shape, frequency) drawn from `[-40,40]^3` with `max|n| >= 2`, fixed seed.
`rel(A,B) = max|A-B|/max|B|`.

    shape  f          D -> -D   transpose   axis reflections   axis permutation   homogeneity 3.7
    c32    1          0.0       0.0         0.0                1.05e-15           1.20e-15
    c32    1+0.1i     0.0       0.0         0.0                8.06e-16           1.29e-15
    c4     1          0.0       0.0         0.0                1.04e-15           1.06e-14
    c4     1+0.1i     0.0       0.0         0.0                1.15e-15           9.18e-15
    sl     1          0.0       0.0         0.0                n/a                1.50e-15
    sl     1+0.1i     0.0       0.0         0.0                n/a                1.11e-15

`D -> -D`, the transpose and the three axis reflections are **bit-exact** — as auditG warned,
they are exact by construction here (only even `l` survives the whole box, and the octant table
is *defined* by the reflection rule), so they prove bookkeeping, not accuracy.  The honest test
is the axis permutation for the cubic cells, which mixes different `(l,m)` entries of the table:
**1.15e-15**.  It is not applicable to the slender cell (`s1 = s2 != s3`) and the script reports
`0.0` there because the test is skipped.
Homogeneity `T(lam f, s/lam) = lam^{-2} T(f, s)` at `lam = 3.7` (a non-dyadic rational, so the
scaled shape is a genuinely different geometry table, built at `L = 20`, `nMax = 6` on both
sides so the two truncations match): **1.5e-15** (lambda/32, slender), **1.1e-14** (lambda/4).

Number type, at `D = (5,1,0)`, `s = (1/32)^3`, `f = 1`:

    Complex{Float64}  in -> ComplexF64        out
    Complex{Float32}  in -> ComplexF32        out,  3.07e-8 from the Float64 value (eps(Float32) = 1.2e-7)
    Complex{BigFloat} in -> Complex{BigFloat} out at the ambient precision (160 bits checked),
                                                  5.06e-16 from the Float64 value

so `T` is genuinely generic: `Float32` in gives `Float32` out at the `Float32` rounding floor, and
`BigFloat` in gives a `BigFloat` result at the ambient precision, with the geometry table served
from its 192-bit store and the frequency contraction done at `max(192, precision(T)+64)` bits.

## 7b. The double-double phase seed and the Re/Im split (script `t12_seed.jl`, raw `out/t12_seed.txt`)



Both seeds feed the *same* truncated series and are compared against a BigFloat(220) evaluation
of that same series, so the column is the seed's contribution and nothing else.

    shape  offset          f        kR       L    dd seed    naive seed   gain
    c32    (32,32,32)      1        10.9     10   2.29e-16   1.00e-15     4.4
    c32    (127,127,127)   1        43.2      8   1.07e-14   7.49e-15     0.7
    c32    (127,127,127)   1+0.1i   43.4      8   8.60e-16   6.14e-15     7.1
    c32    (100,70,30)     1+0.1i   24.8      8   2.06e-16   7.58e-16     3.7
    c4     (32,32,32)      1        87.1     16   3.40e-16   7.81e-15    23
    c4     (64,64,64)      1       174.1     16   4.33e-16   1.57e-14    36
    c4     (127,0,0)       1       199.5     16   1.39e-16   2.45e-15    18
    c4     (127,127,127)   1       345.5     16   6.44e-14   6.43e-14     1.0
    c4     (127,127,127)   1+0.1i  347.3     16   3.47e-15   4.66e-14    13
    c4     (100,70,30)     1+0.1i  198.4     16   1.74e-15   5.74e-15     3.3

At lambda/32 (`kR <= 43`) the seed is worth a factor 1 to 7; at lambda/4 with a 128^3 grid
(`kR` up to 348) it is worth **13 to 36**, and without it the far tensor at 64 cells is at
`1.6e-14` from a seed error alone.  The two rows where it buys nothing (`c4 (127,127,127)` and
`c32 (127,127,127)` at `f = 1`, both `6.4e-14` / `1.1e-14` either way) are not seed-limited: at
`kR = 345` famB measured the l-sum cancellation of the axial diagonal entry as `c_l ~ 0.67 kR`,
i.e. `~230`, and `eps * 230 = 5e-14` — that is the floor those two rows sit on, and it is a
property of the expansion, not of the seed.

**The Re/Im split is structural, not an extra code path.**  For real `f` every `c_ln(k)` and
every geometry moment is real, so the contracted table `Qtilde` is stored as `Float64`
(`eltype(fs.whl.Qd) == Float64`, measured) and the prefactor is exactly imaginary
(`prf = 0.0 + 6.283185307179586im`, measured).  The accumulator `sum h_l (Y_lm Q_lm)` is then a
`Complex * Real` product in every term: its real part accumulates only `j_l` contributions (from
Miller's downward recurrence) and its imaginary part only `y_l` contributions (from the upward
recurrence), with no cross-contamination, and the final multiply by a pure imaginary `prf` is an
exact swap-and-negate.  That is exactly "two separate real sums", obtained at half the flops of a
complex sum.  For complex `f` the table is `ComplexF64` and the sum is genuinely complex.

## 7c. (e) Anti-Hermitian positivity of a 6^3 block (script `t8_pos.jl`, raw `out/t8_pos.txt`)



`GlaVol((6,6,6), s)`, `CPUKerOpt(f, 48, false, CPU())`, `egoFunInn!` over all Toeplitz offsets and
`egoFunSng!` (`wekTrp`) over the eight offsets with all indices <= 2, then
`egoToe[a,a,1,1,1] -= 1/f^2` — Gila's own build.  The second build is that block with every offset
of max-norm separation >= 2 replaced by `farBlock!`; the contact and touching-shell entries are
identical in both.  The dense 648x648 matrix uses `egoToeCrc!`'s rule
`M[(c,a),(c',b)] = egoToe[a,b,|d|+1] sigma_a sigma_b`, `sigma_m = sign(d_m)`.  `M` is complex
symmetric to 1.3e-15, so `(M - M')/(2i) = Im M`.

    shape f        build      lam_min           lam_max     eps*lam_max   negatives
    c32   1        Gila       -4.468094e-15     0.0772947   1.716e-17     248 / 648
    c32   1        farfield   -2.448448e-15     0.0772947   1.716e-17     201 / 648
    c32   1+0.1i   Gila       +2.8550812464276e-03  0.1881490  4.178e-17   0 / 648
    c32   1+0.1i   farfield   +2.8550812464274e-03  0.1881490  4.178e-17   0 / 648

    sl    1        Gila       -4.864140e-14     0.0050611   1.124e-18     282 / 648
    sl    1        farfield    -1.802381e-14     0.0050611   1.124e-18     270 / 648
    sl    1+0.1i   Gila       -6.812225e-03     0.1953457   4.338e-17    36 / 648
    sl    1+0.1i   farfield   -1.267514e-02     0.1953534   4.338e-17    31 / 648

    max |Gila - farBlock!| / max entry over the separated offsets:
      c32, f = 1       5.58e-13 at (0,5,5)
      c32, f = 1+0.1i  3.48e-13 at (4,4,0)
      sl,  f = 1       6.16e-01 at (0,0,2)     <- Gila, not this library

At `f = 1` the operator is lossless, `Im M` is positive **semi**-definite with a large null space,
so the meaningful number is the most negative eigenvalue: `farBlock!` moves it from `-4.47e-15`
to `-2.45e-15` (a factor 1.8 closer to zero) and cuts the negative count from 248 to 201.  Both
remain 143 and 260 times the Float64 floor `eps*lam_max = 1.7e-17` because the residual is
dominated by the contact and touching-shell entries, which are Gila's in both builds and are not
what this library replaces.  At `f = 1+0.1i` the identity term dominates and the two cubic builds agree
to 13 digits in `lam_min`.  The `5.6e-13` gap between the two cubic builds is Gila's quadrature
error, not this library's: §3 puts `farTensor` at `4.2e-15` against the 220-bit reference at
lambda/32, and `overlap.md` §9 measured Gila at `3e-13 .. 6e-13` there.

**Two things the slender rows say, and neither is about this library.**  (1) The
`max |Gila - farBlock!|` of **0.616 at (0,0,2)** is Gila's fixed Gauss rule being wrong by 62% of
the largest entry on the slender short axis — the same failure the `reference` agent measured
(0.79 relative at `n = 2`) and `overlap.md` §9 confirmed against the 220-bit reference
(`4.7e-7` at `(0,0,36)`, `1.1e-5` per entry).  §3 verifies `farTensor` at `1e-16 .. 5e-15` at
those offsets against `pairKer` and against `refQuad`.  (2) At `f = 1+0.1i` the slender block's
`Im M` is **strongly indefinite in both builds**: `lam_min = -6.8e-3` (Gila) and `-1.27e-2`
(farfield) against `lam_max = 0.195`, i.e. 3.5% and 6.5% of the spectrum radius, 14 orders above
any rounding floor.  That is a property of the 1:1:16 discretization (a 1/512-thin plate cell
with the identity term `Im(-1/f^2) = +0.196`), not of the far-field evaluation; the two builds
differ by a factor 1.9 for the same reason as (1).  Whether Gila's *contact* integrals for such
a cell are the cause was not investigated here.

## 8. Defects found in the source libraries



1. **famB's published `L` rule has no `kR` term and under-selects.**  `L = 13/log10(1/(0.7 rho))`
   is a `rho`-only rule; `overlap.md` §7 measured it short by 6 levels at `lambda/4`, 23 cells
   (`Lneed = 16` vs `Lrule = 10`, error 5.5e-10) and by 14 levels on the slender needle
   (`Lneed = 54` vs 40 at `(0,0,36)`, error 1.12e-11).  This library never uses that rule; it
   uses `remBndAll`, which does carry `kR` through `hb(l+2, kR)`.  Its `lambda/4` histogram
   (§4) bottoms out at a flat `L = 16` at every separation, which is exactly the floor
   `overlap.md` says is needed and which the `rho`-only rule cannot produce.
2. **famB's tail bound is a partial sum, not a tail.**  `remBndAll(lTop, ...)` returns
   `cum[i] = sum_{L < l <= lTop}`, so `cum[end] = 0` and the bound is vacuous at `l = lTop`.
   Called with `lTop = L` (the natural reading) it certifies `L = lTop` for free.  Fixed here by
   summing to `L + LEXT` with `LEXT = 16` and only offering cuts at `l <= L`; the residual
   `l > L + 16` part is below 1e-40 of the retained tail at every shape and frequency used.
3. **famB's `geoTab` accepts Float64 edge lengths with no guard.**  The contraction cancels by
   `2.1e5` at `L = 20`, `5.8e8` at `L = 40` and (measured here) `7.3e11` at `L = 56`; a Float64
   table is pure noise beyond `L ~ 26`.  `farfield.jl` takes `NTuple{3,Rational{BigInt}}` only
   and always contracts at 1024 bits.
4. **famB's `bslj!` normalises on `sin(z)/z` and `cos(z)` recomputed from the Float64 `z`.**
   At `lambda/4` and 128 cells `kR` reaches 348 and the Float64 rounding of `z` alone puts
   `z*eps = 7.7e-14` of phase error into the seed — at the 1e-13 target.  Replaced here by
   `phsSeed`, which carries `|R|` and the argument `2 f |R|` in double-double and reduces mod 2
   exactly (auditG's `phsDD`, 12 flops).  §7 measures what this buys.
5. **The `(-1)^l` of the addition theorem.**  famB drops it (correctly: the triangle weight kills
   every odd `l`), famBsub restores it (mandatory: the affine sub-box weight does not).  A reader
   who takes famB's `qTab` and feeds it a non-parity-pruned table gets a first-order error
   (famBsub measured 3.7e-2 at `(4,0,0)`).  `farfield.jl` folds `(-1)^l` into `frqBox` for both
   tables, so the two paths cannot diverge.
6. **A bug this work introduced and then found, worth recording because it is easy to repeat:**
   sizing a per-thread workspace vector by `Threads.nthreads()` and indexing it with
   `Threads.threadid()` is wrong in Julia 1.12 -- `nthreads()` counts the `:default` pool while
   `threadid()` is a global id that also numbers the interactive threads.  With
   `JULIA_NUM_THREADS=8` it raised `BoundsError` at index 9.  `Threads.maxthreadid()` is the
   correct size; the threaded/serial bit-identity check of §13 is what caught it.
7. **`WRADC::Dict{Any,Any}` in famB/famBsub is a global untyped cache and is not thread safe.**
   `farfield.jl` has no global mutable state in the per-offset path; the shape cache `SHPC` is
   guarded by a `ReentrantLock` and is only touched at setup.
8. **`pairMoments` with `Rational{BigInt}` panels fails** (`sqrt` at `moments.jl:442`); route
   (iii) converts the exact panels to BigFloat first, which is exact for these lengths.
9. Not used here, hence not fixed: famC's `gTab` overflows Float64 when `L log10(1/R0) > 308`
   (`overlap.md` §4), and famC's `M` rule has no `kR` term either.

No sign error, normalisation error or implementation error was found in famB, famBsub or the
k-series assembly: the three agree with each other, with the 220-bit `pairKer` reference and
with Paul's Mathematica evaluation of the volume form to the numbers in §3.

## 9. Honest statement of where `farTensor` does not meet 1e-13



Every number in this section is the worst case over the runs of §3-§7.

**Direct answer: there is no offset class, at `f = 1` or `f = 1+0.1i`, at any of the four shapes,
at which `farTensor` fails 1e-13 per entry.  The worst per-entry error over the 403 cached
220-bit references is 1.2e-14 (`c4 (64,0,0)`, `f = 1+0.1i`), and the worst max-norm error is
8.6e-15.  Three qualifications, all measured:**

- **Strongly complex frequency.**  At `f = 1+1i` (auditG's stress frequency, well outside Gila's
  usual range) the worst per-entry error is `6.3e-14` over 21 rows, still inside 1e-13; the worst
  *per-part* error is `9.9e-13`.  I did not test beyond `Im f / Re f = 1`.
- **The imaginary part in relative terms.**  For real `f` the library computes `Re T` from `y_l`
  and `Im T` from `j_l` as two independent real sums (§1.2, §7), so the radiative part keeps its
  own relative accuracy.  For complex `f` the tensor is one complex sum and the imaginary part
  is accurate only to `eps |Re T|`: whenever `|Im T_ab| / max|T| < 1e-3` the relative error of
  that entry's imaginary part is about `1e-16 / (|Im|/max|T|)`, i.e. above 1e-13 when the ratio
  falls below 1e-3.  This is a property of the formulation, not of the expansion (auditG §2c
  reached the same conclusion for the Taylor route); it is unavoidable in Float64 without
  splitting the complex-frequency sum, which no route in the registry does.
- **Entries suppressed by symmetry** (exact zeros on axis offsets) carry absolute error only;
  they are excluded from every per-entry column and reported through the max-norm column.

The three worst `eRe`/`eIm` rows of §3, entry by entry (`t14_split.jl`, `out/t14_split.txt`),
show exactly what the per-part columns are measuring:

    c4 (64,0,0), f = 1, max|Gr| = 2.487e-3      (route (i), L = 16, kR = 100.5)
      [1,1]  |G|/mx 0.0199   Re/mx  1.9e-4   Im/mx -0.0199   rel 4.8e-15  Re 4.9e-13  Im 1.1e-15
      [2,2]  |G|/mx 1.0      Re/mx  1.0      Im/mx  9.8e-3   rel 8.6e-15  Re 4.0e-16  Im 8.8e-13
      [3,3]  |G|/mx 1.0      Re/mx  1.0      Im/mx  9.8e-3   rel 8.6e-15  Re 2.3e-16  Im 8.8e-13

    c8 (32,32,32), f = 1                        (route (i), L = 14, kR = 43.5)
      every entry has |Re|/mx and |Im|/mx of order 0.2-1.0, and every Re and Im error is <= 1.5e-14

    c32 (64,0,0), f = 1+0.1i, max|Gr| = 1.367e-5
      [1,1]  |G|/mx 0.159    Re/mx -3.6e-3   Im/mx -0.159    rel 3.0e-15  Re 1.2e-13  Im 1.5e-15

Every `eRe`/`eIm` above 1e-13 in the whole of §3 is a part carrying `<= 1e-2` of the entry's own
modulus, with the *absolute* error still at `eps * max|G|`; wherever both parts are O(1) fractions
of the entry (the `c8 (32,32,32)` row) both per-part errors are at `1.5e-14`.  **There is no
offset class at `f = 1` or `f = 1+0.1i` where an entry, or a part of an entry that carries more
than 1% of that entry, exceeds 1.2e-14 relative.**

## 10. Files



    notes/farfield/farfield.jl        the library (deliverable)
    work/unify/t0_smoke.jl            sign/normalisation pin, BigFloat(1024) vs exact-rational tables
    work/unify/t1_tab.jl              builds and times the four shape tables      -> out/t1_tab_*.txt
    work/unify/refload.jl             loader for every round-1 220-bit reference cache
    work/unify/t2_ref.jl              (a) farTensor vs the cached references      -> out/t2_ref.txt
    work/unify/t3_route.jl            (2) routing counts over a full octant       -> out/t3_route.txt
    work/unify/t4_needL.jl            what L the bounds want, table-free diagnostic
    work/unify/t5_sym.jl              (d) symmetries, homogeneity, number types   -> out/t5_sym.txt
    work/unify/t6_refq.jl             builds the missing refQuad volume references -> cache/refq_*.txt
    work/unify/t7_ovl.jl              (c) overlap against famG route (i)          -> out/t7_ovl.txt
    work/unify/t8_pos.jl              (e) anti-Hermitian positivity of a 6^3 block -> out/t8_pos.txt
    work/unify/t9_cost.jl             (f) timing and the term histogram           -> out/t9_cost.txt
    work/unify/t10_ksr.jl             (b) route (iii) vs refQuad, and its cost    -> out/t10_ksr.txt
    work/unify/t11_diag.jl            n-truncation certification, bound tightness, cost model
    work/unify/t12_seed.jl            the double-double phase seed, the Re/Im split
    work/unify/t13_mono.jl            monotonicity of the whole-box threshold table
    work/unify/cache/                 shape tables (19.5 MB each), route (iii) tensors, refq references
## 11. What remains uncertain



- **The bound's constant 17** rests on the standard differentiation relations for solid harmonics
  having coefficients of modulus <= 1 (famB §8 makes the same caveat).  It was never violated in
  any run here, and the measured `Lbnd/Lneed` ratios of §5 quantify how much it over-selects.
- **`est(R)` is an a priori scale, not a bound on `max|T|`.**  famBsub measured `est/max|G_ref|`
  in `[2.0, 5.1]` over every offset tested and this work adds the rows of §3; if a shape existed
  for which `est` under-stated the true magnitude the selected `L` would be too small.  A proof
  that `est >= max_ab |T_ab|` for every separated pair is not in hand.
- **`W_l` for odd `l`** uses `W_l <= sqrt(W_{l-1} W_{l+1})` (Cauchy-Schwarz with the non-negative
  weight), which is exact for the even case and a genuine bound for the odd; no tightness claim.
- **The whole-box threshold table** assumes the bound-to-`est` ratio is monotone in `|R|`.  §6
  checks that assumption against a direct evaluation at 4000 radii per (shape, frequency); it is
  a check, not a proof.
- **Aspect ratios other than 1:1:1 and 1:1:16 were not verified against a reference.**  The
  library is shape-generic (three independent rational edges) and the bound is shape-generic, but
  the only references that exist are for the four shapes plus auditG's random-aspect cases.
- **No GPU path, and no wiring into `GlaVacOprMem`.**  `farBlock!` fills a Gila-layout `egoToe`
  and was checked against a Gila-built block in §7(e), but the build itself is a later task.
## 12. Summary of the numbers



    sign/normalisation vs Paul's Mathematica volume form at (5,1,0), 1/32, f = 1  every printed digit
    BigFloat(1024) geometry table vs exact Rational{BigInt}, L = 20 / L = 24      1.80e-78 / 4.6e-59
    sparsity pattern of the float build vs the exact build                        identical, 3 shapes
    geometry contraction cancellation, L = 56: whole box / octant                 7.31e11 / 1.96e11
    shape table build (L = 56, nMax = 12), whole + octant, per shape              474-643 s, 19.5 MB
    reload of a shape table from disk                                             1.9 s
    (l,m) entries stored, whole box / octant                                      1457 / 12212
    accuracy vs 403 cached 220-bit references, f = 1 and 1+0.1i, four shapes:
      max over the 9 entries, relative to the largest entry                       <= 8.6e-15
      per entry, entries above 1e-8 of the largest                                <= 1.2e-14
      worst case in the whole table (f = 1+1i, |Im|/max ~ 1e-3, per entry)        6.3e-14
    routing, 128^3 octant (64x64x128 slender):  (i) / (ii) / (iii)
      cube lambda/32, lambda/8, lambda/4                          2 097 132 / 12-15 / 0
      slender (1/32,1/32,1/512)                                     524 066 / 142 / 72
    whole-box L floor: lambda/32 / lambda/8 / lambda/4                            8 / 12 / 16
    n-truncation certified by the bound at nMax = 12 (max nCut 5/8/10/5)          never saturates
    bound tightness Lbnd/Lneed over 80 rows                                       1.00 - 1.69
    threshold table vs a direct bound evaluation, 32000 radii                     0 disagreements
    cost(i)/cost(ii) where both bounds are met, 24^3 octant                       <= 0.107
    overlap vs famG (its own q rule), 4085 offsets, lambda/32 and lambda/4        <= 1.0e-14
    route (iii) vs the graded BigFloat GL volume reference, 14 rows                2.0e-17 - 1.6e-16
    route (iii) cost: BigFloat(128) moments (once) / series+assembly per frequency 3.6-13.7 s / 0.36-0.95 s
    refQuad floor with famD cuts / with cuts graded to the near point, (1,0,2)     5.9e-12 / 2.9e-45
    D -> -D, transpose, axis reflections                                          0.0 (bit-exact)
    axis permutation (cubes), homogeneity at lam = 3.7                            1.15e-15, 1.1e-14
    Float32 in -> Float32 out (3.07e-8), BigFloat in -> BigFloat out (5.1e-16)    generic in T
    anti-Hermitian lam_min, 6^3 at lambda/32, f = 1: Gila / farfield              -4.47e-15 / -2.45e-15
    double-double seed gain at lambda/4, kR = 87 .. 200                           18 - 36x
    farBlock! 128^3 at lambda/32, single thread, load 2.95                        2.05 s, 976 ns/offset
    farBlock! 128^3 at lambda/32, 8 threads, load 3.92                            0.335 s, 160 ns/offset
    farBlock! threaded vs serial                                                  0.0 (bit-identical)
    mean terms summed per offset, 128^3 at lambda/32                              185
## 13. (f) Timing and the term histogram (script `t9_cost.jl`, raw `out/t9_cost.txt`)


`farBlock!` filling a whole `egoToe` block, **single-threaded** (`JULIA_NUM_THREADS=1`), M3 Pro,
**load average 2.95** (one other Julia process and Spotlight indexing the 19.5 MB shape tables;
no other agent's job was running).  Best of three calls after a warm-up; the frequency
contraction and the shape table are done once outside the timed region, the route-(iii) offsets
are pre-filled from the disk cache.  "terms" = complex multiply-adds actually issued
(`costWhl`/`costOct`), which for real `f` is two real multiplies each.

    shape  N     f        offsets filled   time       ns/offset   mean terms/offset
    c32    32    1            32 760       0.0337 s    1029        335.0
    c32    64    1           262 136       0.2460 s     939        237.1
    c32    128   1         2 097 144       2.0471 s     976        185.0
    c32    32    1+0.1i       32 760       0.0308 s     939        335.2
    c4     32    1            32 760       0.0571 s    1742        684.5
    sl     32    1            32 760       0.0474 s    1448        634.2

**The whole 128^3 far field at lambda/32 is 2.05 s single-threaded**, against the ten to thirty
minutes the task statement quotes for the quadrature path, and 5.3 ns per summed term.  The
per-offset cost falls with N (1029 -> 939 -> 976 ns) because the mean term count falls (335 ->
237 -> 185) as more of the block sits at the `L = 8` floor.

Histogram of terms summed per offset, 128^3, lambda/32 (the `L` histogram of §4 in cost units):

    152 terms : 1 174 762 offsets  (L = 8)      1096 :   45      2631 :   12
    223       :   902 327          (L = 10)     1275 :   22      2904 :    1
    307       :    15 998          (L = 12)     1467 :   24      3491 :    3
    405       :     2 602          (L = 14)     1673 :   15      3805 :    6
    516       :       749          (L = 16)     1892 :    9      4473 :    6
    641       :       310                       2125 :   12      5576 :    3
    779       :       144                       route (ii) : 12 offsets, ~2e4 terms each
    931       :        82

Multi-threaded, `JULIA_NUM_THREADS=8`, load average 3.92 (same script, same block):

    shape  N     f        time       ns/offset   speedup over 1 thread
    c32    32    1        0.0068 s    207         5.0
    c32    64    1        0.0444 s    170         5.5
    c32    128   1        0.3352 s    160         6.1
    c4     32    1        0.0118 s    359         4.9
    sl     32    1        0.0102 s    312         4.6

**0.335 s for the whole 128^3 far field on 8 threads**, and `farBlock!` threaded against the same
block filled serially is **bit-identical (max |diff| = 0.0)** on a 12^3 block.

(That check found a real bug in the first version of `farBlock!`: the per-thread workspace vector
was sized `Threads.nthreads()`, which counts only the `:default` pool, while `Threads.threadid()`
is a global id that includes the interactive threads -- a `BoundsError` at index 9 with 8 default
threads.  It is sized `Threads.maxthreadid()` now.)

At lambda/4 the histogram bottoms at 641 terms (`L = 16`) for 25006 of 32760 offsets, and the
per-offset time is 1.7x the lambda/32 one.  The slender 32^3 row includes its 72 route-(iii) and
126 route-(ii) offsets; the 72 k-series tensors are computed once (from the disk cache in the
timed region, see §3b for their cost) and the remaining 32562 offsets carry the whole 1448 ns
average, whose mean term count 634 is high because a slender block has many offsets at large
`rho` in the transverse plane.

## 14. Cost model summary (terms actually summed per offset)



`cost(i) = (L+1)^2 + 3|{lm : l <= L}|_diag + sum_c |{lm : l <= L}|_c`, one complex multiply-add
per term (two real multiplies each for real `f`, six for complex `f`), plus the `h_l` recurrence
(`L` steps) and, for real `f`, Miller's downward recurrence for `j_l` (`L + 10 + 2x + O(sqrt L)`
real steps).  Evaluated at the `L` each shape's histogram bottoms out at:

    lambda/32   L =  8   cost(i) = 156 terms      56.0% of the 128^3 offsets sit here
    lambda/8    L = 12   cost(i) = 313 terms      86.3%
    lambda/4    L = 16   cost(i) = 532 terms      98.4%
    two-cell shell (route (ii), L_j up to 56): cost(ii) = sum_j 7 (L_j+1)^2, order 2e4 terms,
      at 12-15 offsets per octant whatever N

## 15. Reproducing everything



    ENV=notes/farfield/scratch/env ; U=notes/farfield/scratch/work/unify
    JULIA_NUM_THREADS=1 julia --startup-file=no --project=$ENV $U/t1_tab.jl c32 c8 c4 sl   # ~500 s each
    JULIA_NUM_THREADS=1 julia --startup-file=no --project=$ENV $U/t2_ref.jl                # table (a)
    ... (one line per script of §10)

Each script sets `TABDIR[]` and `KSRDIR[]` to `work/unify/cache/` and writes its table to
`work/unify/out/`.  The shape tables must exist first (t1_tab.jl); everything else reads them.

**Where the tables live.**  `farfield.jl` defaults to `TABDIR[] = notes/farfield/shapetab` and
`KSRDIR[] = notes/farfield/ksrcache`, creating them on first use; the four shape tables built here
are 19.5 MB each (75 MB total) and were deliberately left in `work/unify/cache/` rather than
committed under `notes/farfield/`.  Either point `TABDIR[]` at that directory

    TABDIR[] = ".../notes/farfield/scratch/work/unify/cache"
    KSRDIR[] = ".../notes/farfield/scratch/work/unify/cache"

or let the library rebuild them (474-643 s per shape, once ever).  Nothing else is needed: with
`TABDIR[]` and `KSRDIR[]` populated, `farfield.jl` never touches `moments.jl` or any other file.
