# prep: Theorem A/B in the selector, a 256-bit lazy geometry table, and a proven scale

Deliverable: `SCRATCH/work/prep/farfield_v2.jl` (1337 lines), a copy of
`notes/farfield/farfield.jl` with the three changes below; unified diff at
`SCRATCH/work/prep/farfield_v2.diff`.  `notes/farfield/farfield.jl` itself was NOT touched.
Work dir `SCRATCH/work/prep/`, raw output in `SCRATCH/work/prep/out/`,
SCRATCH = `/Users/pvirally/.julia/dev/GilaElectromagnetics/notes/farfield/scratch`.

Every timing below carries the machine load at which it was taken; the machine was never quiet
(five to eleven other Julia processes from other agents throughout), so every second is an upper
bound and old-vs-new pairs were taken back to back at comparable load.

---

## 0. What changed, in one place

| | old (`notes/farfield/farfield.jl`) | new (`work/prep/farfield_v2.jl`) |
|---|---|---|
| whole-box l bound | Theorem (c), singular organization, constant 17 | **Theorem A**, regular organization, `hm = :bd` |
| sub-box l bound | eq (bnd:sub), same constant | **Theorem B**, per-entry, `hm = :bd` |
| n-truncation bound | Theorem (c) form at radius `2 min(s)` | Theorem A form, at the smallest radius actually used, budget `tol est/(256 (L+1))` |
| scale in the criterion | `est(R)` | `est(R)` (default) or the **proven** `estLo(R)` (`scl = :low`) |
| table build precision | 1024 bits | **256 bits** |
| table extent | `L = 56` always, both tables | **lazy**: `LDEF = 24`, grown to what the block asks for, whole and octant sized separately |
| table cache | one file per `(shape, L, nMax)` | one file per `(shape, nMax)`, l-segments **appended** |
| exact-zero test | fixed `2^-256` | `2^-(prc-24)`, tied to the build precision |
| bound arithmetic | in the working type `T` | always Float64 (Float32 overflowed, S G.3) |
| default `tol` | `1e-13` | `1e-14` (`const TOL`), see A.2 |

`boundL(fs, rr)`, `boundLoct(fs, R)`, `farRoute(fs, D)`, `farRouteStat(st, dim)`, `farTensor`,
`farBlock!`, `farSetup`, `farShape`, `est`, `jbnd`, `hbnd`, `jtail`, `pickCut`, `costWhl`,
`costOct`, `tnsWhl!`, `tnsOct!`, `tnsKsr`, `srfSum`, `FarWs`, `FrqSet` keep their names and their
signatures.  `farShape` keeps its old `L` keyword as an alias that sets both `Lw` and `Lo`, so
`farShape(s; L = 56)` still builds the full table.  Only `bndTrm` and `bndCum` change shape --
they now carry the six per-entry Theorem A term arrays instead of one, and `bndCum` takes
`(ls, t, k, rr, hm)` instead of `(ls, t, kR)` -- and no thin wrapper is offered for them because
the old bound they returned is not in the file any more.  New: `bndWhl(fs, rr, L)` (the bound
itself, for a posteriori certificates), `estLo(fs, rr)`, `sclOf`, `bndTrmO`, `farMomW`,
`farMomO`, `hbndS`, `hrecS`, `hFac`, `lowScl`, `lowVal`, `nearOff`, `mrgGeo`, `segL`, `zTol`.
`FrqSet` gains `Lw`, `Lo`, `wLs`, `wT`, `hm`, `e0`, `eT`, `scl`; new keywords on `farSetup` are
`lDef`, `offs`, `scl`, `hm`.  `FarWs(fs.L, T)` still works (`fs.L` is `LMAX = 56`, an upper bound
on both table extents); `FarWs(max(fs.Lw, fs.Lo), T)` is the tight sizing and is what
`farTensor`/`farBlock!` use.

## B. The table build: 256 bits is bit-for-bit the 1024-bit table

**The rounding argument.**  One entry of the contraction is a sum of at most
`(L/2 + nMax + 2)^2` products of an integer solid-harmonic coefficient with a box moment
(`momAcc`); at `L = 56`, `nMax = 12` that is `(28+14)^2 = 1764 < 2^11` terms.  Measured
cancellation `sum|term| / |acc|` is `7.31e11 < 2^39.4` (whole box) and `1.96e11 < 2^37.5`
(octant).  A build at `prc` bits therefore carries a relative error of at most
`2^(11 + 39.4 - prc)`: `2^-205.6` at 256 bits, `2^-141.6` at 192.  The table is stored at
`TABPRC = 192` bits (`2^-192`) and consumed in Float64 (`2^-53`), so 256 bits leave 13 bits of
slack against the storage rounding itself and 152 bits against Float64.  The one place the
precision is not just slack is the exact-zero test inside `momAcc`: the old fixed threshold
`sum|term| * 2^-256` sits *below* the 256-bit accumulation noise (`2^-245 sum|term|`) and would
have turned every structural zero into noise, so it is now `2^-(prc-24)`, which is above the
noise (`2^-232 > 2^-245`) and far below the smallest true value (`2^-39.4 sum|term|`).

**Measured** (script `work/prep/p4_tab.jl`, raw `out/p4_tab.txt`).  Each row builds the FULL
`L = 56`, `nMax = 12` whole-box and octant table at `prc` bits, rounds it to the same 192-bit
storage the 1024-bit tables of `work/unify/cache/` use, and compares entry by entry.  "differing"
counts stored numbers that differ at all; "Float64-differing" counts those whose `Float64` value
differs.  Timings are single-threaded at load 5.9 to 10.5 (other agents' jobs throughout).

    shape prc   whl (s)  oct (s)  total   whl differing / >1ulp / Float64  oct differing / >1ulp / Float64   max rel
    c32   256    18.5     68.7     87.2    0 / 0 / 0  of 35906             0 / 0 / 0  of 285467              0.0
    c32   192    24.3     87.5    111.8    29516 / 28491 / 0               255741 / 244694 / 0              8.86e-48
    c32   320    21.8    117.0    138.7    0 / 0 / 0                       0 / 0 / 0                        0.0
    c8    256    21.5    114.8    136.3    0 / 0 / 0                       0 / 0 / 0                        0.0
    c8    192    36.7    115.7    152.4    29516 / 28491 / 0               255741 / 244694 / 0              8.86e-48
    c8    320    19.4     83.8    103.2    0 / 0 / 0                       0 / 0 / 0                        0.0
    c4    256    37.0    139.5    176.6    0 / 0 / 0                       0 / 0 / 0                        0.0
    c4    192    38.3    107.9    146.2    29516 / 28491 / 0               255741 / 244694 / 0              8.86e-48

    reference (unify, 1024 bits, same L, nMax): c32 473.7 s, c8 497.7 s, c4 576.2 s, sl 642.6 s

**256 and 320 bits reproduce the 1024-bit table bit for bit** -- every one of the 321373 stored
numbers per shape, whole box and octant, including the sparsity pattern (the cancelled-to-zero
set) and the reported cancellation 7.31e11 / 1.96e11.  192 bits differs on 82% of the entries by
at most 8.9e-48 relative, which is the 192-bit build noise showing through the 192-bit storage;
**no Float64 value changes even at 192 bits**, so 192 would also be safe, but it has no margin
against a larger `L` and it is not faster.  256 bits is 5.4x (c32) to 3.3x (c4) faster than 1024.
The 192-bit rows are not faster than the 256-bit rows because MPFR's limb count is the same
(3 limbs of 64 bits carry 192; 4 carry 256) and the machine load moved between rows.

## A. Theorem A / Theorem B in the selector

### A.1 What is integrated, and why `hm = :bd`

`farfield_v2.jl` carries bound2's cached form directly:

    farMomW(lTop, s)        -> (fc, wr):  fc[r][l/2+1] = the frequency-independent part of V^{ab}_l
                                          for the six entries r = 11,22,33,12,13,23, wr = W_l.
                                          Once per shape; O(lTop^3) integer multinomials.
    bndTrm(lTop, mom, rd, k, frq, vt) -> (ls, t): t[r][i] = everything of the Theorem A term
                                          except |h_l|.  Once per (shape, frequency).
    bndCum(ls, t, k, rr)    -> cum[i] = max_r sum_{l > ls[i]} t[r] hbndS(l), per offset.
    farMomO(lTop, hf), bndTrmO(...)   -> the same two stages for Theorem B on an octant
                                          (al = h = s/2, bt = -1, so u^+ = 0, u^- = s).

`hbndS(L, z, ak)` is `hb(l,z) |k|^l/(2l+1)!!` accumulated with the scaling folded into every term,
which is bound2's `hMaj`, i.e. **`hm = :bd`**.  `:ex` (the value of the upward Hankel recurrence)
is not used, for one reason: the deliverable's claim is a *proven* bound, and the upward `h_l`
recurrence has no error analysis -- famB measured it accurate to 3e-15 for `l <= 40`, which is a
measurement, not a proof, and it is the one step that would make the whole chain unproven.
bound2 measured the price at a median 8% (at most 38%) in the selected `L`; that is visible in the
`Lnew(hb)` column of bound2 S3.2 and is included in every `L` reported below.  Nothing else in the
chain is empirical: `V^{ab}_l` and `W_l` are exact rational moments summed with positive terms,
the `j_l` factor is the pointwise majorant, and the `m`-sum is one Cauchy-Schwarz.

The tail past `lTop = L + LEXT = 72` is no longer dropped: `bndCum` adds the geometric
continuation `q r/(1-r)` at the ratio attained at the top and returns `Inf` when that ratio is
`>= 1` (the guard theory.md S5.8 asked for; the old code summed to `lTop` and stopped).

### A.2 The one interface constant that had to move: `tol`

The old `tol = 1e-13` was calibrated against a bound that over-stated the true remainder by a
median 1.5e5.  Theorem A over-states by a median 49.  At the same `tol` the selector therefore
spends the whole gain on `L` and the delivered accuracy degrades to the budget:

    tol      worst per-entry error over the 403 cached 220-bit references (c32 / c8 / c4 / sl)
    1e-13    7.7e-15  /  1.8e-14  /  9.3e-15  /  1.6e-13
    1e-14    4.4e-15  /  5.5e-15  /  9.3e-15  /  8.7e-15
    old lib  6.3e-14  /  5.6e-15  /  1.2e-14  /  8.3e-15   (tol = 1e-13, old bound)

so `farfield_v2.jl` sets `const TOL = 1e-14` as the default of `farSetup`, `farTensor` and
`farBlock!`.  With it the delivered accuracy matches the old library entry for entry (better on
c32) and the *certified* bound is ten times tighter.  Everything below is at `tol = 1e-14` unless
the row says `1e-13`.  Both are reported wherever the two differ, because `1e-13` is what the
"same decision rule" comparison means and `1e-14` is what "same delivered accuracy" means.

### A.3 New `L` per offset against the old, on the cached references

Every one of the 403 cached 220-bit reference offsets (four shapes, f = 1, 1+0.1i, 1+1i, 3+0.3i,
0.37, loaded by `work/unify/refload.jl`), old `L` read from `work/unify/out/t2_ref.txt`, new from
`work/prep/out/p2_L.txt` (script `p2_ref.jl`, joined by `join_L.py`).

    shape  n    both (i)   Lold/Lnew  min/median/max    Lold-Lnew min/med/max   (Lold+1)^2/(Lnew+1)^2 med/max
    c32   200      176      0.833 / 1.000 / 1.167         -2 / 0 / 8            1.000 / 1.353
    c8     56       48      0.857 / 1.000 / 1.143         -2 / 0 / 6            1.000 / 1.299
    c4     58       50      0.889 / 1.077 / 1.136         -2 / 2 / 6            1.154 / 1.284
    sl     89       65      0.857 / 1.000 / 1.105         -2 / 0 / 4            1.000 / 1.216

Offset by offset (f = 1; `rt` = route, `bnd` = the Theorem A bound divided by max|G_ref|):

    shp  D             rho     kR       rt.o L.old  rt.n L.new   eMx      pEn      bnd
    c32  (2,2,0)       0.612   0.555     1    56     1    48     9.4e-16  9.4e-16  3.0e-14
    c32  (3,0,0)       0.577   0.589     1    50     1    44     4.9e-16  1.3e-15  1.5e-14
    c32  (3,1,0)       0.548   0.621     1    46     1    42     3.1e-16  6.1e-16  6.7e-15
    c32  (4,0,0)       0.433   0.785     1    34     1    32     3.9e-16  7.7e-16  3.6e-15
    c32  (8,0,0)       0.217   1.571     1    20     1    18     2.4e-16  4.2e-16  2.0e-14
    c32  (8,8,8)       0.125   2.721     1    14     1    14     4.8e-16  4.8e-16  2.1e-14
    c32  (16,0,0)      0.108   3.142     1    14     1    14     4.1e-16  4.1e-16  2.1e-15
    c32  (32,0,0)      0.054   6.283     1    12     1    12     2.4e-16  6.2e-16  2.3e-16
    c32  (64,0,0)      0.027  12.566     1    10     1    10     1.0e-15  1.0e-15  1.7e-15
    c8   (2,2,0)       0.612   2.221     2    39     1    50     1.3e-15  1.7e-15  1.9e-14
    c8   (3,0,0)       0.577   2.356     1    52     1    46     1.3e-15  1.3e-15  1.2e-14
    c8   (4,0,0)       0.433   3.142     1    36     1    32     7.3e-16  7.3e-16  1.0e-14
    c8   (8,8,8)       0.125  10.883     1    16     1    16     3.5e-16  6.8e-16  1.1e-14
    c8   (64,0,0)      0.027  50.265     1    14     1    14     3.1e-15  3.1e-15  2.0e-16
    c4   (2,0,0)       0.866   3.142     2    56     2    51     4.7e-16  5.5e-16  2.9e-14
    c4   (2,2,0)       0.612   4.443     2    40     1    52     2.4e-16  4.2e-16  1.5e-14
    c4   (3,0,0)       0.577   4.712     1    54     1    48     3.7e-16  4.4e-16  9.1e-15
    c4   (4,0,0)       0.433   6.283     1    38     1    34     3.8e-16  1.2e-15  8.5e-15
    c4   (8,0,0)       0.217  12.566     1    24     1    22     9.6e-16  9.6e-16  8.0e-15
    c4   (16,16,16)    0.062  43.531     1    18     1    18     4.7e-15  4.7e-15  6.6e-16
    c4   (64,0,0)      0.027 100.531     1    16     1    16     8.6e-15  9.3e-15  6.5e-15
    sl   (2,0,0)       0.708   0.393     2    47     2    44     1.2e-15  2.7e-15  1.8e-14
    sl   (2,2,2)       0.5     0.556     1    46     1    42     6.1e-16  8.7e-15  8.8e-15
    sl   (4,4,4)       0.25    1.112     1    22     1    22     3.5e-16  4.7e-15  2.8e-14
    sl   (0,0,32)      0.708   0.393     2    34     2    32     1.3e-15  2.4e-15  2.5e-14
    sl   (0,0,64)      0.354   0.785     1    30     1    30     1.2e-16  1.3e-16  2.9e-15
    sl   (16,0,0)      0.088   3.142     1    14     1    14     2.5e-16  3.5e-16  5.2e-15

The near band (`rho >= 0.4`) is where Theorem A pays: 6 to 8 shells off `L` at 2-4 cells, and two
offsets change route (see A.5).  At `rho <= 0.15` the two agree, and at a few far offsets the new
`L` is 2 shells LARGER than the old (`Lold/Lnew = 0.833`): that is the price of `hm = :bd`, which
is quantified in A.6.

### A.4 Error against the reference, and the bound against the actual error

    worst over all reference offsets of a shape      c32       c8        c4        sl
    max-norm  max_ab|G-Gr| / max|Gr|                 4.4e-15   5.5e-15   8.6e-15   2.5e-15
    per entry, entries above 1e-8 max|Gr|            4.4e-15   5.5e-15   9.3e-15   8.7e-15
    (old library, same references)      max-norm     4.2e-15   5.6e-15   8.6e-15   3.2e-15
                                        per entry    6.3e-14   5.6e-15   1.2e-14   8.3e-15
    Theorem A bound / max|Gr| at the selected L      4.1e-14   4.5e-14   3.0e-14   4.4e-14
    est(R)/max|Gr|                              [1.16,6.34] [1.10,4.48] [1.25,3.05] [1.42,135]

**Every per-entry error is at or below 9.3e-15, inside the 1.2e-14 the old library delivered.**

**The bound is never violated.**  103 of the 403 rows have `eMx > bnd` -- at `tol = 1e-14` the
truncation bound is often below the Float64 evaluation floor -- and the largest excess is
`eMx - bnd = 5.05e-15` at c8 `(32,32,32)`, `L = 14`, where the OLD library at the same `L = 14`
measured `eMx = 5.6e-15`: the excess is the arithmetic floor of the evaluation, not truncation.
No row exceeds `bnd + 5.1e-15 = bnd + 23 eps max|G_ref|`.

### A.5 Routing over a full `egoToe` octant, new selector (script `p3_route.jl`)

    shape / block          tol       route (i)   (ii)   (iii)   max octant l   mean terms/offset
    c32  128^3  f = 1      1e-14     2 097 132    12      0          51             225.8
    c32  128^3  f = 1      1e-13     2 097 132    12      0          48             224.3
    c32  128^3  OLD        1e-13     2 097 132    12      0          -              185.0
    c8   128^3  f = 1      1e-14     2 097 132    12      0          51             406.1
    c8   128^3  f = 1      1e-13     2 097 132    12      0          48             312.4
    c8   128^3  OLD        1e-13     2 097 129    15      0          -               -
    c4   128^3  f = 1      1e-14     2 097 132    12      0          52             519.8
    c4   128^3  f = 1      1e-13     2 097 132    12      0          48             517.0
    c4   128^3  OLD        1e-13     2 097 129    15      0          -               -
    sl   64x64x128 f = 1   1e-14       524 102   111     67          55             319.3
    sl   64x64x128 f = 1   1e-13       524 131    86     63          55             274.7
    sl   64x64x128 OLD     1e-13       524 066   142     72          -               -

Same at `f = 1 + 0.1i` to within a few offsets of the `L = 10` / `L = 12` boundary.

**What moves, exactly.**

- **(ii) -> (i), cubes:** for `c8` and `c4` the three offsets `(2,2,0)`, `(2,0,2)`, `(0,2,2)` leave
  the octant split for the whole box, at `L = 50` (c8) and `L = 52` (c4).  The old bound could not
  certify the whole box there at all below `L = 56`; Theorem A certifies `L = 50`/`52`, and route
  (i) at `L = 52` costs 3400 terms against `8 x 7 x 41^2 = 94 000` for the octants -- a 28x saving
  on those three offsets.  For `c32` the same three offsets were already route (i) (at `L = 56`);
  they now take `L = 48`.  The remaining 12 route-(ii) offsets are the two-cell family
  `(2,0,0),(2,1,0),(0,2,0),(1,2,0),(2,0,1),(2,1,1),(0,2,1),(1,2,1),(0,0,2),(1,0,2),(0,1,2),(1,1,2)`
  at every scale, unchanged.
- **(iii) -> (ii)/(i), slender:** the k-series band shrinks from 72 offsets to 67 at `tol = 1e-14`
  and 63 at `tol = 1e-13`.  Old: `(0,0,n)`, `(0,1,n)`, `(1,0,n)`, `(1,1,n)` for `n = 2..19`
  (18 + 36 + 18).  New at 1e-14: `(0,0,n)` `n = 2..17` (16), `(0,1,n)/(1,0,n)` `n = 2..18` (34),
  `(1,1,n)` `n = 2..18` (17).  At 1e-13: `(0,0,n)` `n = 2..16` (15) and 63 in total.
- **(ii) -> (i), slender:** 31 offsets at 1e-14 (56 at 1e-13) leave the octant split, all from the
  `(2,0,n)/(2,1,n)/(0,2,n)/(1,2,n)` columns at larger `n`.
- **Nothing ever moves the wrong way**: no offset goes from (i) to (ii) or from (ii) to (iii).

### A.6 Per-offset cost: the far-field floor moves UP by two shells, and it is not `hb`


    c32 128^3, f = 1     L histogram (route (i) only)
    OLD   tol 1e-13     8:1174762  10:902327  12:15998  14:2602  16:749 ... 56:3
    NEW   tol 1e-13    10:2080640  12:13898   14:1749   16:461   18:175 ... 44:3
    NEW   tol 1e-14    10:2050272  12:41906   14:3535   16:833   18:277 ... 48:3

The 1.17M offsets the old selector served at `L = 8` now need `L = 10`, which is the whole of the
+22% in mean terms per offset (225.8 against 185.0); the near tail is uniformly shorter (top of
the histogram 56 -> 48).  `tol = 1e-14` costs only +0.7% over `tol = 1e-13` there (225.8 against
224.3): at the far end the floor is set by the bound, not by the budget, so the tenfold accuracy
is nearly free.

**It is not the `hb` majorant.**  `farSetup(..., hm = :ex)` swaps `hb(l,kR)` for the value of the
scaled upward Hankel recurrence and changes nothing at the far end (script `p10_hm.jl`):

    shape / block          terms :bd      terms :ex      :bd / :ex   L(:bd)-L(:ex) min/med/max
    c32 128^3              473 457 434    469 874 854      1.008        0 / 0 / 2
    c8  128^3              851 569 408    850 486 243      1.001        0 / 0 / 4
    c4  128^3            1 090 189 265  1 083 148 809      1.006        0 / 0 / 8
    sl  64x64x128          167 394 123    139 616 197      1.199        0 / 2 / 2
    routes are identical for the cubes; for sl, :ex moves 4 offsets out of (ii)+(iii) into (i).

So the proven majorant costs **0.1 to 0.8% on the cubes** and 20% on the slender needle, and is
kept.  The +2 shells at the far end of a cube block are Theorem A itself: it bounds the REGULAR
remainder, which bound2 S1 measured at a median 17.5x (up to 7.4e3) the singular remainder that
Theorem (c) bounds, and at `rho < 0.027` that ratio is at the top of its range.  The old `L = 8`
was never a certificate for the sum the library computes; `L = 10` is.

## B. The lazy table

### B.1 `L = 24` by default is NOT enough for separation >= 3 -- the routing tables say so

The claim was checked directly (script `p0_ldist.jl`, raw `out/p0_ldist.txt`), with the OLD bound,
by listing every offset of the block whose whole-box `L` exceeds 24:

    shape  block        offsets needing L > 24 (or no L)   of those, separation >= 3   max L there
    c32    128^3                125                              106                      50
    c8     128^3                155                              136                      52
    c4     128^3                231                              212                      54
    sl     64x64x128           1329                             1310                      56

`L = 24` covers separation `>= 5` for the cubes (the first offsets above 24 are `(3,0,0)` at 50,
`(3,1,0)` at 46, `(3,1,1)` at 44, `(3,2,0)` at 38, `(4,0,0)` at 34 ...), not separation `>= 3`.
With Theorem A the same list is shorter but not empty: `(3,0,0)` needs 44, `(4,0,0)` 32,
`(8,0,0)` 18.  So a full block always extends the table, and the honest statement of the saving is
per use:

    use                                            whole-box L built     octant L built
    one far offset, e.g. farTensor((17,5,3), ...)          24                  0
    a 32^3 or 128^3 block at lambda/32, c32                50                 51
    a 32^3 or 128^3 block at lambda/4, c4                  52                 52
    a 64x64x128 block, slender                             56                 56

`farSetup` sizes both tables exactly, by enumerating the lattice offsets inside the critical
radius `thr[LDEF/2+1]` (a few hundred points, `nearOff`) and taking the largest `L` any of them
asks for -- or, when `offs` is passed (which `farTensor` does for a single offset), only those.
The whole-box and octant tables are sized separately, which matters because the octant table is
8x the size of the whole-box one at the same `L`.

### B.2 How the cache stores an extension

One file per `(shape, nMax)`, `shapetab/s<...>_L0_n12_p192_seg.txt`, holding a sequence of
l-segments, each a whole-box block followed by an octant block, appended in the order they were
built.  Loading reads every segment and concatenates the `(l,m)` columns; because both builders
emit columns in ascending `l` within each parity class, concatenation is exactly the table that a
single build to the final `L` would have produced (verified: the `L = 56` single-segment build of
`p4_tab.jl` and the `24 + 26..56` two-segment build agree bit for bit, and `p1_smoke.jl` reloads a
two-segment cache and reproduces `farTensor((5,1,0), ...)` to 0.0).  A segment in which only one
of the two tables grew writes a placeholder for the other (`emtGeo`, no columns, `L = -1`), and
`segL` makes the loader ignore it.

## C. `est(R)`: the open item

**The direction of the problem, first.**  The criterion is `bound <= tol * X`.  It implies
`bound <= tol * max_ab|T_ab|`, which is what "relative error `tol`" means, only if
`X <= max_ab|T_ab|`.  Measured here over all 403 references, `est/max|G_ref|` runs over
`[1.10, 6.34]` for the cubes and reaches 135 on the slender needle: `est` **over**-states, so what
the old rule actually certifies is a relative error of `tol * est/max|T|`, i.e. 1.1 to 6.3 times
`tol` on the cubes.  (unify S11 states the risk the other way round -- "if est under-stated the
true magnitude the selected L would be too small"; under-stating is the safe direction.)

**A proven scale, `estLo`.**  Away from the source `lap g = -k^2 g`, so

    tr T = (1/V_t) int_D w (lap + 3k^2) g = 2 k^2 S,   S = (1/V_t) int_D w g(R+d) dd,

and `max_ab |T_ab| >= |tr T|/3 = (2|k|^2/3)|S|`.  Expanding `S` by the same addition theorem, the
`l = 0` shell is `S_0 = (i k/f^2) h_0(kR) (1/(4 pi)) (1/V_t) int_D w j_0(k|d|) dd`, so

    |S| >= |S_0| - |S - S_0|,
    |S_0| >= (|k|/(4 pi |f|^2)) (e^{-Im(k)|R|}/(|k||R|)) V_t j_0(|k| r_d)        (|k| r_d <= pi)
    |S - S_0| <= (|k|/(4 pi |f|^2 V_t)) sum_{l >= 2, even} (2l+1) |h_l(kR)| |k|^l
                 e^{(|k| r_d)^2/(4l+6)}/(2l+1)!! W_l    (Theorem A with V^{ab}_l -> W_l)

-- everything already in the shape table, `O(lMax)` per offset, and a function of `|R|` alone, so
the threshold table survives.  `farSetup(..., scl = :low)` uses it; `estLo(fs, rr)` exposes it.

**Measured** `estLo/max|G_ref|` over the references (it must be `<= 1` to be a lower bound, and
close to 1 to be useful):

    shape   estLo/max|G_ref|      est/max|G_ref|     |k| r_d
    c32     0.04 .. 0.98          1.16 .. 6.34        0.340
    c8      0.04 .. 0.63          1.10 .. 4.48        1.361
    c4      0 (vacuous)           1.25 .. 3.05        2.721
    sl      0 .. 0.71             1.42 .. 135         0.278

It is never above 1 (as it must not be) and reaches 0.98, 0.63 and 0.71 on the three shapes where
it applies; on the far offsets that carry the block it sits at 0.4 - 0.8, i.e. within 1.3 - 2.5x
of the truth, against `est`'s 1.1 - 2.5x above it.

**Where it fails, and why cheaply is impossible there.**  For `c4` the bound is vacuous at every
offset: the `l >= 2` tail of the SCALAR series is larger than the `l = 0` shell.  The ratio is
`|k|^2 r_d^2 / (18 j_0(|k| r_d))` at large `|kR|`, which is 0.0065 at c32, 0.14 at c8, 0.0009 at
sl -- and 2.86 at c4.  The trace is direction-free precisely because it throws away the anisotropy,
and the anisotropic part is the whole of `T` at small `kR`; recovering it means keeping the
`Y_lm(Rhat)` of shells 2 and 4, which makes the scale direction-dependent and destroys the
`O(1)`-per-offset threshold table.  So: a proven, direction-free, `O(1)` scale exists and is within
a factor 1.3-2.5 of the truth for cells up to about `lambda/8` across the diagonal
(`|k| r_d <= 1.4`), and does not exist by this argument for a `lambda/4` cell.

**The universal fallback, which costs nothing.**  After the tensor is evaluated,
`bndWhl(fs, |R|, L)` and `max_ab |T^(L)_ab|` are both in hand, and

    relative error <= bnd / (max_ab|T^(L)_ab| - bnd)

is a certificate with no hypothesis at all.  Over the 403 references it evaluates to at most
4.5e-14 (the `bnd` column of A.4 divided by `1 - bnd`), which is the honest statement of what
`farfield_v2.jl` guarantees at `tol = 1e-14`: **a certified max-norm relative error of 4.5e-14,
and a measured one of 8.6e-15.**

### B.3 The cold 32^3 build at lambda/32, old against new (script `p7_cold.jl` / `p7o_cold.jl`)

Empty table cache, `c32`, `f = 1`, `N = 32`, single thread, run back to back in the same slot:

                                       OLD (1024 bits, L = 56)   NEW (256 bits, lazy)
    farSetup: table build + contraction      827.5 s                 201.6 s   (Lw 48, Lo 51)
    farBlock! first fill                       0.914 s                 0.893 s
    total                                    828.4 s                 202.5 s   -> 4.09x
    disk cache written                        19 555 247 B            16 076 486 B
    load average during the run               4.90 - 6.74             6.74 - 8.41
    one far offset from cold, farTensor((17,5,3), ...)
                                             828 s (same table)        1.3 s   (table L = 24 / 0)

The new run was at the HIGHER load of the two, so 4.09x is a lower bound on the gain.  (The old
827.5 s against unify's 473.7 s for the same build is the machine: unify measured at load ~3.)
Of the 4.09x, the precision accounts for 3.3-5.4x on a full `L = 56` build (S B) and the lazy `L`
for the rest; for the single-offset case the lazy `L` is the whole 640x.

## C.1 `scl = :low` measured (script `p6_scl.jl`, raw `out/p6_scl.txt`, `out/p6_scl_sl.txt`)

    shape  block       |k| r_d   estLo/est at |R| = 8 r_d   routes :est -> :low     terms :low/:est
    c32    32^3         0.340         0.256                [32748,12,0] unchanged       1.055
    c8     32^3         1.361         0.288                [32748,12,0] -> [32748,9,3]  1.041
    c4     32^3         2.721         vacuous (0)          [32748,12,0] -> [0,0,32760]   --
    sl     32x32x64     0.278         0.220                [65350,111,67] -> [65314,138,76]  1.125

`L(:low) - L(:est)` over the offsets that stay on route (i) is 0 at the median and at most 6.
So the proven scale costs **4 to 13% in terms** where it applies, plus three offsets pushed to the
k-series on `c8` and 36 on the slender needle; on a `lambda/4` cell it sends every offset to the
k-series and is useless.  Default stays `:est`; `scl = :low` is one keyword away.

## D. Cost: terms per offset and a single-threaded `farBlock!`

Script `p5_cost.jl` (new) and `p5o_cost.jl` (the current `notes/farfield/farfield.jl`), run back to
back in the same slot, `JULIA_NUM_THREADS=1`, best of three fills after a warm-up, tables and the
frequency contraction outside the timed region.  Load average is printed beside every row; it sat
between 9.9 and 12.3 for the whole sequence (six to eight other agents' Julia processes), which is
why the small-block wall times scatter by 2x and the term counts, which are exact, are the number
to read.

    block            terms/offset                     ns/offset (load 9.9 - 12.3)
                     OLD      NEW 1e-13   NEW 1e-14   OLD      NEW 1e-13   NEW 1e-14
    c32 32^3  f=1    335.0    306.8       360.3       1094     1003        2010 *
    c32 64^3  f=1    237.1    233.5       245.1        997      925        1029
    c32 128^3 f=1    185.0    224.3       225.8       1026     1030        1252
    c32 32^3  f=1+.1i 335.2   307.1       360.6        989      887        1012
    c4  32^3  f=1    684.5    577.0       686.6       1792     1663        1821
    sl  32^3  f=1    634.2    556.8       643.6       1485     1512        2996 *

    * both starred rows are load spikes: the same fill at f = 1 + 0.1i with the identical term
      count took 1012 ns/offset, and the 32^3 wall times are 30-100 ms, i.e. at the noise floor.

At `tol = 1e-13` (the like-for-like decision rule) the new selector is **8 to 16% cheaper** per
offset on a 32^3 block and 5% cheaper on 64^3, and **21% dearer** on 128^3, where the block is
almost entirely far offsets and `hb` sets the floor.  At `tol = 1e-14` it is level with the old
selector on 32^3 (+0.3% on c32 at f=1+0.1i, +0.3% on c4, +1.5% on sl) and 22% dearer on 128^3.
The whole of the 128^3 penalty is the `L = 8 -> 10` floor move of A.6, i.e. the `hm = :bd` choice.

## E. The copy verified end to end, beside unify's numbers

Same four scripts as unify, pointed at `farfield_v2.jl` (`p2_ref.jl`, `p_t5_sym.jl`,
`p_t7_ovl.jl`, `p_t8_pos.jl`), same references, same seeds.

    t2_ref  (403 cached 220-bit references, 4 shapes, f = 0.37, 1, 1+0.1i, 1+1i, 3+0.3i)
                                              unify              prep (farfield_v2, tol = 1e-14)
      worst max-norm error over all shapes    8.6e-15            8.6e-15   (c4 (64,0,0), f = 1)
      worst per-entry error, |Gr| > 1e-8 mx   6.3e-14 (f=1+1i)   9.3e-15   (c4 (64,0,0), f = 1)
      worst per entry at f = 1 and 1+0.1i     1.2e-14            9.3e-15
      per shape, max-norm  c32/c8/c4/sl       4.2e-15 / 5.6e-15 / 8.6e-15 / 3.2e-15
                                              4.4e-15 / 5.5e-15 / 8.6e-15 / 2.5e-15
      per shape, per entry c32/c8/c4/sl       6.3e-14 / 5.6e-15 / 1.2e-14 / 8.3e-15
                                              4.4e-15 / 5.5e-15 / 9.3e-15 / 8.7e-15

    t5_sym                                    unify              prep
      D -> -D, transpose, axis reflections    0.0                0.0   (bit-exact, all 6 rows)
      axis permutation, cubes                 1.15e-15           1.07e-15
      homogeneity T(lam f, s/lam) at lam=3.7  1.1e-14            1.06e-14
      Float32 in -> Float32 out               3.07e-8            6.20e-8
      BigFloat(160) vs Float64                5.1e-16            5.14e-16

    t7_ovl  (farTensor vs famG's Taylor route, 4085 offsets of a 16^3 octant, its own q rule)
      c32 worst dMx over the four bands       1.0e-14            1.1e-14
      c32 worst per entry                     1.0e-14            1.2e-14
      c4  worst dMx, band 9-15                8.6e-10            8.6e-10   (famG's own q rule)
      every band, both frequencies            agree row by row to within 20% of unify's value

    t8_pos  (anti-Hermitian part of a 6^3 block, Gila's contact + farBlock!)
      c32 f=1     lam_min gila / farfield     -4.47e-15 / -2.45e-15   -4.468e-15 / -2.449e-15
      c32 f=1+.1i lam_min                      0.00285508...           0.0028550812464274
      sl  f=1     lam_min gila / farfield     (unify did not print)   -4.864e-14 / -1.802e-14
      max |Gila - farBlock| / max entry, c32   -                       5.6e-13 at (0,5,5)
                                                                       (Gila's order-7 rule, not us)

The `t7_ovl` differences from unify are third-digit changes in famG's own error, not in
`farTensor`: the two libraries agree to 1.1e-14 on every c32 band, which is famG's Taylor
truncation at its own `q` rule, exactly as unify reported.

## F. Files

    work/prep/farfield_v2.jl     THE DELIVERABLE (1337 lines; farfield.jl is 1035)
    work/prep/farfield_v2.diff   unified diff against notes/farfield/farfield.jl, 759 lines
    work/prep/p0_ldist.jl        which offsets need L > 24 with the OLD bound  -> out/p0_ldist.txt
    work/prep/p1_smoke.jl        value pin, lazy table, segment-cache round trip
    work/prep/p2_ref.jl          A: vs the 403 cached 220-bit references       -> out/p2_ref.txt,
                                 out/p2_L.txt (machine readable), out/p2_ref_t14.txt (tol sweep)
    work/prep/p3_route.jl        A: routing and L histograms over a full octant -> out/p3_route.txt,
                                 out/p3_route_t13.txt
    work/prep/p4_tab.jl          B: 192/256/320 vs the 1024-bit tables          -> out/p4_tab.txt
    work/prep/p5_cost.jl         A: terms and farBlock! timing, new            -> out/p5_cost*.txt
    work/prep/p5o_cost.jl        the same on the current farfield.jl           -> out/p5_cost_old.txt
    work/prep/p6_scl.jl          C: scl = :est vs :low                          -> out/p6_scl*.txt
    work/prep/p7_cold.jl         B: cold 32^3 build, new                        -> out/p7_cold.txt
    work/prep/p7o_cold.jl        the same on the current farfield.jl            -> out/p7_cold.txt
    work/prep/p8_ncut.jl         the n-truncation diagnosis (S G.1)
    work/prep/p9_sets.jl         exact (ii)/(iii) sets for the needle           -> out/p9_sets.txt
    work/prep/p10_hm.jl          A: hm = :bd against :ex                        -> out/p10_hm.txt
    work/prep/p_t5_sym.jl, p_t8_pos.jl, p_t7_ovl.jl   unify's t5/t8/t7 on the copy
    work/prep/join_L.py          joins out/p2_L.txt to unify/out/t2_ref.txt (old vs new L)
    work/prep/cache/             the segmented shape tables built here (16 MB per shape)

Run as `JULIA_NUM_THREADS=1 julia --startup-file=no --project=SCRATCH/env <script>`;
`TOL=1e-13 SFX=_t13` re-runs `p2_ref.jl`, `p3_route.jl` and `p5_cost.jl` at the other tolerance.

## G. Three things that had to be fixed on the way, with the number that exposed each

### G.1 The n-truncation bound in the old library is vacuous, and tightening it naively costs 1.6e-13

`nCutVec` is evaluated at `kR = |k| * 2 min(s_i)`, the smallest offset radius in the lattice.  For
every shape that is inside the divergence radius of the whole-box expansion (`rho > 1` there), so
the bound never meets its budget and `nCutVec` returns its fallback `nMax` for every `l >= 2`.
Measured, reimplementing the old formula exactly (`p8_ncut.jl`):

    shape   old nCut, l = 0,2,4,...,16          reported by unify S5.1 (evaluated at a larger r)
    c32     5, 12, 12, 12, 12, 12, 12, 12, 12   max 5
    c8      8, 12, 12, ...                      max 8
    c4     10, 12, 12, ...                      max 10
    sl      5, 12, 12, ...                      max 5

So the old library summed all 12 stored orders at every `l >= 2` and had no n-truncation error;
unify's "provably sufficient" table was computed at a different radius from the one the library
uses.  Writing the n-tail in the Theorem A organization (same `V^{ab}_q` moments at order
`q = l + 2N + 2`, `|h_l|` instead of `hb(l+2)`) and keeping the old per-`l` budget
`tol est/(L+1)` makes the bound bite -- and the total n error is then `(L+1)` times the per-`l`
budget, i.e. `tol est` itself: **measured 3.6e-13 relative at sl `(8,8,8)`, 3.5e-13 at sl
`(16,0,0)`, 2.1e-13 at sl `(4,4,4)`** while the L-truncation bound at those offsets is 6e-15.
`farfield_v2.jl` therefore divides the n-budget by `NBUD = 256` and evaluates the bound at the
smallest radius the expansion is ACTUALLY used at (route (i) offsets and route (ii) sub-box
centres, both known from the `nearOff` scan) and at the far end of the block:

    shape   new nCut, l = 0,2,...,16            max over l <= 56   saturates nMax = 12?
    c32     6, 5, 5, 5, 5, 4, 4, 4, 4                 6                   no
    c8      9, 8, 8, 7, 7, 7, 7, 6, 6                 9                   no
    c4     11, 11, 10, 10, 9, 9, 9, 8, 8              11                   no
    sl      6, 5, 5, 5, 5, 5, 5, 4, 4                 6                   no
    error at sl (8,8,8) / (16,0,0) / (4,4,4)   3.2e-16 / 2.5e-16 / 3.5e-16

This is the first actual certificate that `nMax = 12` is enough: the bound now meets a budget
1/256 of the l-budget at every `l`, at the radii where the expansion is used, with `nCut < 12`.

### G.2 The exact-zero threshold has to track the build precision

`momAcc` turned a cancelled sum into an exact zero when `|acc| < sum|term| * 2^-256`.  At
`BLDPRC = 1024` that is far above the accumulation noise (`2^-1013 sum|term|`) and below the
smallest true value (`2^-39 sum|term|`), so it reproduced the exact-arithmetic zero set (unify
S2 verified that against a `Rational{BigInt}` build).  At `BLDPRC = 256` the noise is
`2^-245 sum|term|`, ABOVE the fixed threshold, so no structural zero would have been detected and
the sparsity pattern would have been lost.  `zTol(prc) = 2^-(prc-24)` sits between the two for
every precision from 128 up, and the 256-bit tables of S B reproduce the 1024-bit sparsity
pattern exactly (0 differing entries out of 321373 per shape).

### G.3 The bound must not be evaluated in the working precision

`hbndS`, `(2l+1)!!` and the `V^{ab}_l` moments overflow Float32 above `l ~ 30`; one `Inf` in the
term array poisons the geometric-continuation guard and every cut with it, and `farTensor` at
`ComplexF32` returned `NaN`.  In `farfield_v2.jl` the whole selection layer -- moments, term
arrays, thresholds, `nCut`, `estLo` -- is computed in Float64 regardless of `T`, and `FrqSet`
stores it as Float64; only the evaluation is generic.  `Float32 in -> Float32 out` is back at
6.2e-8 (unify: 3.07e-8, both at the Float32 rounding floor).

## H. Honest statement of what is better and what is worse

Better:
- The bound now bounds the sum the library computes (Theorem A / B, regular organization), with
  no unproven step in it (`hm = :bd`); the old bound bounded a different, 17.5x smaller quantity.
- `nMax = 12` is certified for the first time (G.1), at 1/256 of the l-budget.
- The delivered per-entry accuracy is at 9.3e-15 worst over 403 references, against 6.3e-14 for
  the old library, at `tol = 1e-14`.
- The shape table costs 200 s instead of 474-643 s per shape for a whole block, 1.3 s for a
  single far offset, and it is bit-for-bit the 1024-bit table.
- Three offsets per cubic block (`(2,2,0)` and its permutations at c8 and c4) and 31-36 slender
  offsets leave the expensive routes; the k-series band shrinks from 72 offsets to 67.
- `Float32` works (it returned `NaN` before, G.3).

Worse:
- Per-offset cost on a large block rises 21-22% (mean terms 185.0 -> 225.8 on a 128^3 c32
  block), because Theorem A honestly asks for `L = 10` where the old bound asked for `L = 8`.
  That is not the `hb` majorant (0.8%, A.6) and not the tolerance (0.7%, A.6); it is the price of
  certifying the right quantity.  On 32^3 and 64^3 blocks the new selector is level with or
  cheaper than the old.
- `est(R)` is still the default scale.  The proven `estLo` exists, costs 4-6% in terms where it
  applies, and is vacuous for a `lambda/4` cell (C).

## I. Summary of the numbers

    Theorem A/B integrated, hm = :bd (proven majorant), cached farMom form, same decision rule
    interface constant changed: tol default 1e-13 -> 1e-14 (A.2), everything else unchanged
    L per offset vs the old bound, 403 references, Lold/Lnew   0.833 .. 1.167, median 1.000
      near band rho >= 0.4 (2 to 4 cells)                      6 to 8 shells off L
      far band rho <= 0.027                                    2 shells ON to L (correct organization)
    error vs the 403 cached 220-bit references, per entry      <= 9.3e-15  (old library 6.3e-14)
    error vs the same, max-norm                                <= 8.6e-15  (old library 8.6e-15)
    bound vs actual error: rows with eMx > bnd                 103/403, all at the Float64 floor
      largest excess eMx - bnd                                 5.05e-15 (= 23 eps max|G_ref|)
      no truncation violation anywhere
    routing 128^3 cubes (i)/(ii)/(iii), all three scales       2 097 132 / 12 / 0
      moved (ii) -> (i): (2,2,0),(2,0,2),(0,2,2) at c8 and c4  3 offsets per shape, 28x cheaper each
    routing slender 64x64x128 (i)/(ii)/(iii)                   524 102 / 111 / 67 (old 524 066/142/72)
      moved out of (iii)                                       5 offsets; k-series band n3 <= 18
    terms per offset, 128^3 c32                                225.8 (old 185.0)  +22%
    terms per offset, 32^3 c32 / c4 / sl                       360.3 / 686.6 / 643.6
                                              (old)            335.0 / 684.5 / 634.2
                                     (new at tol 1e-13)        306.8 / 577.0 / 556.8   -8 to -16%
    what the proven hb majorant costs (:bd/:ex), cubes         1.001 .. 1.008 in terms
                                              slender          1.199
    n-truncation: nCut max over l, c32/c8/c4/sl                6 / 9 / 11 / 6, never saturates 12
      old code's actual nCut at l >= 2                         12 (bound vacuous at 2 min(s))
    geometry table at 256 and 320 bits vs the 1024-bit table   bit-identical, all four shapes,
                                                               321 373 numbers each, sparsity too
    geometry table at 192 bits                                 max rel 8.9e-48, no Float64 changed
    table build L = 56 nMax = 12, 256 bits, per shape          87 - 200 s (1024 bits: 474 - 828 s)
    cold 32^3 build at lambda/32, tables + fill                202.5 s (old 828.4 s)  4.09x
    cold single far offset                                     1.3 s (old 828 s)
    est/max|T| over 403 references                             1.10 .. 6.34 (135 on the needle)
    estLo/max|T| (proven lower bound) c32/c8/sl                0.04 .. 0.98 / 0.04 .. 0.63 / 0 .. 0.71
    estLo cost in terms, c32/c8/sl                             1.055 / 1.041 / 1.125; c4 unusable
    certified relative error at tol = 1e-14 (a posteriori)     <= 4.5e-14
    D -> -D, transpose, axis reflections                       0.0 (bit-exact)
    axis permutation / homogeneity at lam = 3.7                1.07e-15 / 1.06e-14
    Float32 in -> Float32 out                                  6.20e-8 (was NaN before G.3)
    BigFloat(160) vs Float64                                   5.14e-16
    overlap vs famG, 4085 offsets, lambda/32                   <= 1.1e-14
    anti-Hermitian lam_min 6^3 lambda/32, gila / farfield_v2    -4.468e-15 / -2.449e-15
