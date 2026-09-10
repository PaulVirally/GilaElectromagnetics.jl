# merge — the final `notes/farfield/farfield.jl`

`notes/farfield/farfield.jl` is now `work/prep/farfield_v2.jl` plus the six changes below.
Backups of the three files I was allowed to touch, as they were before this session:
`work/merge/{farfield_orig.jl, verify_orig.jl, bench_orig.jl}`.  Work dir `work/merge/`, scripts
`m0_smoke.jl` .. `m9_mom.jl`, raw output `work/merge/out/`.  Nothing under `src/`, `test/`,
`Project.toml`, `Manifest.toml` was touched.

Machine: M3 Pro, 12 cores, 36 GB, Julia 1.12.6.  At most two Julia processes of mine at a time;
the load average is on every timing line.

**Verdict in one paragraph.**  D9 is fixed by making the table's `k`-series order a function of
`|k| r_d` and refusing above `NCAP = 40`: audit2's worst row goes from `9.6e-3` to `1.71e-13` per
entry and the `lambda/2` cube from `3.0e-12` to `3.8e-15`.  D2 and D10 are confirmed fixed by v2:
`c128` goes from a flat `4.65e-13` at every separation to `<= 4.3e-15`, `r1 (48,48,48)` from
`1.23e-13` to `3.2e-15`.  The bound now has **no truncation violation** over audit2's 60 random
cases (`bound/actual` min `0.118`, median `4.02`, max `54.5`; all 15 sub-unit ratios are the
rounding floor at `<= 16.4 eps max|T|`).  D4 is fixed (309-312 ns per offset off a 988-1095 ns
fill) and route (iii) is threaded (633 s -> 128 s cold on the slender block, bitwise identical).
The five API gaps are closed and `verify.jl` a,b,c,d,e,g,i and `bench.jl` at 16^3/32^3 re-run
clean.  D8's memory is **not** fixed: the build is 4.5x faster but still peaks at 21-25 GB, and
S4.2 says exactly which two functions own it.  And a finding: replacing Gila's contact and
touching-shell entries by the exact `k`-series makes the slender block's anti-Hermitian part
**positive definite** at `f = 1+0.1i` (`+3.25e-4` against `-1.465e-2`), so the indefiniteness
audit2 and `unify.md` both attributed to the discretization is Gila's order-9 contact rule.


## 0. What changed, and where

| | `farfield_v2.jl` | `farfield.jl` (final) |
|---|---|---|
| table k-series order | compile-time `NMAX = 12`, `nCutVec` can only lower it | `NMAX` is a **floor**; `farSetup` raises the table to the `N` the `jtail` bound needs at this (shape, frequency), up to `NCAP = 40`, and **errors** above that |
| certified-`N` check | none | `nCutVec` returns `-1` when the budget is not met; `farSetup` refuses; `ckCut(fs)` refuses in `farTensor`/`farBlock!` |
| `nCut` length | `L+1`, silently short when a cached table carries more shells | padded to `max(L, Lw, Lo)+1` (a latent `BoundsError`, hit by `verify.jl` part g) |
| `farRoute` | always evaluates `costWhl`/`costOct` | `cst::Bool = false` keyword; the cost is computed only when asked |
| route (iii) in `farBlock!` | serial loop | `Threads.@threads :static`, `MOMC` and the moments load behind `MOMLK`, `fs.ksr`/`fs.ksrI`/the cache file behind `fs.lk`, one ambient `setprecision` for the region |
| `needMom` | `getglobal(Main, ...)` after a run-time `Base.include` (Julia 1.12 world-age warning) | the two functions are fetched once through `invokelatest` and held in `MOMF` |
| `momAcc` | one BigFloat allocation per term | an in-place MPFR method on four registers, bit-identical (28518 pairs, `m9_mom.jl`) |
| table directory | `TABDIR[]` only | `dir` keyword on `farShape`, `farSetup`, `farTensor`, `farBlock!`; `TABDIR[]` default `notes/farfield/shapetab/`, created on demand |
| `farRouteStat` | `(cnt, hst, ksr)` | `(cnt, hst, ksr, oct)` |
| route (iii) cache line | `d1 d2 d3` + 18 numbers | `d1 d2 d3 N Lambda` + 18 numbers; 21-token lines still read, `fs.ksrI[D] = (N, Lambda)` |
| block partition | docstring only | `isFarInd(i)` / `isFarInd(i1,i2,i3)`, used by `farBlock!` and `farRouteStat` |

New names: `NCAP`, `ckCut`, `isFarInd`, `momReg`, `mpfrMul!`, `mpfrAdd!`, `mpfrSetZ!`, `mpfrAbs!`,
`mpfrZero!`, `MOMLK`, `MOMF`.  `FrqSet` gains `nNd`, `nMx`, `ksrI`, `lk`.  Everything `verify.jl`
and `bench.jl` used keeps its name; only `farRouteStat`'s arity and `nCutVec`'s last argument
(`nMax` -> `nCap`, and it may now return -1) changed, and both callers were updated.

`notes/farfield/shapetab/` now exists and holds four symlinks to the `L = 56, nMax = 12` tables in
`scratch/work/unify/cache/`, renamed to the segmented spelling (`..._L0_n12_p192_seg.txt`); the
old one-shot files are byte-identical to a single l-segment, so they load unchanged.

## 1. Defect D9: the `nMax = 12` cap

### 1.1 The mechanism, and what replaces it

The geometry table stores `Ad[:, 1:nMax+1, :]`, the first `nMax+1` terms of the `k`-power series of
`j_l(k|d|)`.  `jtail(l, N, |k| r_d) ~ (|k| r_d)^{l+2N+2}`, so the order the series needs is set by
`|k| r_d`, `r_d = sqrt(s1^2+s2^2+s3^2)` — a **shape and frequency** quantity, not a constant.  In
v2 `nCutVec` started at `fill(nMax, lTop+1)` and could only lower entries, so a shape whose bound
asked for more than 12 got 12 and no warning (audit2 §1.5: `9.6e-3` per entry at `|k| r_d = 8.96`).

Now `nCutVec` returns `-1` where the budget is not met within `nCap`, `farSetup` re-runs it against
`NCAP = 40` (widening the moment table to `L + 2 NCAP + 2` only then), takes `nNd = max(nCut)`,
builds the table at `max(NMAX, nNd)` — `NMAX = 12` is kept as a floor so that a table already on
disk at 12 is reused — and stores `(nNd, nMx)` in the `FrqSet`.  `ckCut` re-checks them in
`farTensor` and `farBlock!`.  If the bound is not met at `N = 40` the call **errors**:

    farSetup: the k-series of j_l is not certified within N = 40 at |k| r_d = ... (cell ..., f = ...);
    the cell is too large in wavelengths for this expansion

### 1.2 The (|k| r_d, N) relation (`m1_ncap.jl`, `out/m1_ncap.txt`, table-free, 70 rows)

`N` is the largest `nCut` over `l = 0..56` at `tol = 1e-14`, `NBUD = 256`, with `offs` = the six
audit offsets of the shape.  `Lw`, `Lo` are the table extents the same call asks for.

    |k| r_d   0.03  0.09  0.28  0.45  0.68  1.11  1.58  2.23  2.72  2.89  3.16  4.03
    N            4     5     6     7     8     9    11    12    12    14    16    14
    |k| r_d   4.46  4.78  5.44  5.80  8.96  10.9  21.9
    N           18    18    16    19    25    24    39

    (the non-monotone pairs are different shapes at the same |k| r_d: N also depends weakly on the
    aspect ratio through V^{ab}_l, e.g. ex3 (aspect 1020) needs 16 at 3.16 where nd4 needs 14, and
    the cubes c2/c1 need 16/14 at 5.44/4.03 where the needles r3/r4 need 18/19 at 4.46/5.80)

A cell inside Gila's stated range (`longest edge <= lambda/4`, so `|k| r_d <= 2.72`) needs
`N <= 12`: **`NMAX = 12` is exactly right for Gila and wrong for anything coarser**, which is why
the cap was invisible until audit2 went outside the range.  `NCAP = 40` covers `|k| r_d <= 22`,
i.e. a cubic cell of one wavelength at `f = 2`.

### 1.3 The four failing rows of audit2's sweep, re-measured (`m2_d9.jl`, `out/m2_d9.txt`)

References are audit2's own 220-bit graded volume rule (`work/audit2/refcache/reftensors.txt`,
its `vol` records), read but not recomputed.  `pEn` = worst per-entry `|G-Gr|/|Gr|` over entries
above `1e-8 max|Gr|`; `eMx` = `max_ab |G-Gr| / max|Gr|`.  The offsets are audit2's six per shape;
the two nearest ones route to (iii) (the BigFloat k-series, 10-76 s each) and audit2 already
verified that route at `8.3e-18 .. 3.6e-15`, so they are listed but not recomputed here.

    shape  f        |k| r_d  N   rows  eMx       pEn       eRe       eIm       audit2 pEn   gain
    r3     2+0.2i    8.96    25   4     4.91e-15  1.71e-13  4.58e-14  9.28e-12  9.6e-03      5.6e10
    r4     2+0.2i    5.80    19   4     7.29e-15  1.02e-13  3.58e-13  1.86e-14  6.4e-07      6.3e6
    r2     2+0.2i    4.78    18   4     3.47e-15  2.10e-14  3.78e-14  2.05e-14  1.3e-09      6.2e4
    r3     1+0.1i    4.48    18   4     3.81e-15  3.29e-14  3.42e-13  2.99e-13  1.8e-10      5.5e3
    r3     1         4.46    18   4     5.28e-15  2.71e-14  5.04e-14  1.88e-14  1.7e-10      6.3e3
    r4     1         2.89    14   4     1.59e-15  8.88e-15  4.92e-15  1.37e-14  9.3e-14      10
    r2     1         2.38    13   4     1.00e-15  1.70e-15  1.18e-14  7.97e-15  8.2e-14      48
    r3     0.37      1.65    12   4     1.59e-15  1.87e-15  3.46e-15  6.00e-15  1.2e-14      6.4
    r4     0.37      1.07    10   4     1.03e-15  1.74e-15  2.71e-15  1.31e-14  1.1e-14      6.3
    r2     0.37      0.88    10   4     1.16e-15  7.75e-15  7.79e-15  1.38e-13  2.3e-14      3.0
    c2     1         5.44    16   5     3.79e-15  3.79e-15  2.39e-14  2.00e-14  7.2e-13      190

**Every row is now at or below `1.71e-13` per entry and `7.3e-15` in max-norm**, against
`1.7e-10 .. 9.6e-3` before.  The `<= 1e-13` per-entry target is met on 9 of the 11 rows; the two
misses are `r4 f = 2+0.2i` at `1.02e-13` (offset `(7,-3,2)`) and `r3 f = 2+0.2i` at `1.71e-13`
(offset `(16,16,16)`), both on entries carrying a small fraction of `max|G|` — the max-norm errors
of those two rows are `7.29e-15` and `4.91e-15`.  These are aspect-87.7 and aspect-678 cells with
a longest edge of `0.92` and `1.43` wavelengths, three to six times outside Gila's range, and what
is left there is the per-part/small-entry floor of audit2's item 4, not the `k`-series (the `eIm`
column shows it: `9.3e-12` on an imaginary part that carries `< 1e-3` of its entry).

The 44 route-(i)/(ii) rows of this table cost 5.0 us to 0.42 ms each; `farTensor` on the near
offsets `(0,0,2)`, `(0,0,4)` of `r3`, `r2`, `r4` still routes to (iii) and is not re-timed here.

### 1.4 The `lambda/2` cube, measured directly (audit2 §1.7's five offsets)

`c2 = (1/2,1/2,1/2)`, `f = 1`, `|k| r_d = 5.44`, `N` needed `16`, table rebuilt at `nMax = 16`,
`Lw = 38`, `Lo = 53`: **171 s** of build+setup, peak RSS 22.5 GB (see §4.3), 21.7 MB on disk.

    offset       route  L/N   eMx        pEn        audit2 (nMax = 12)  audit2 bound
    (2,0,0)      ii     53    3.79e-15   3.79e-15   1.95e-17 (route iii) 6.10e-17
    (3,3,3)      i      32    6.03e-16   6.03e-16   7.22e-13             1.76e-14
    (4,0,0)      i      38    1.44e-15   1.44e-15   3.01e-12 (Re 9.0e-11) 1.74e-14
    (7,-3,2)     i      28    2.54e-15   2.71e-15   2.20e-12             7.07e-15
    (16,16,16)   i      22    3.54e-16   6.83e-16   7.20e-13             1.94e-15

so the four route-(i) offsets improve by **265x to 2100x** and the whole `lambda/2` cube is now at
`3.8e-15` per entry.  `(2,0,0)` moved from route (iii) to route (ii) because the octant table is
now built to `Lo = 53` (v2's `nCut` fix plus the lazy sizing); it agrees with the reference to
`3.79e-15`, four orders worse than the k-series' `1.95e-17` but two orders inside the target, and
it costs 0.9 ms instead of tens of seconds.

## 2. Defects D2 (`lambda/128`) and D10 (`L` under-selected far out)

Both come from v2 (the `nCut` budget re-evaluated at the radius the expansion is actually used at,
divided by `NBUD = 256`, plus `TOL = 1e-14` and Theorem A).  Measured on exactly audit2's cases,
against exactly audit2's references (`m3_d2.jl`, `out/m3_d2.txt`).

### 2.1 `c128 = (1/128)^3`, `f = 1`, the `(n,n,n)` and `(n,0,0)` scans of audit2 §1.6c

    D            L   nNd  eMx       pEn       audit2 pEn | D          L   nNd  eMx       pEn       audit2
    (3,3,3)      24   5   4.61e-16  4.32e-15  4.7e-13    | (4,0,0)    30   5   1.58e-15  1.58e-15  1.2e-14
    (4,4,4)      20   5   2.30e-16  2.25e-15  4.7e-13    | (8,0,0)    18   5   7.36e-16  7.87e-16  5.2e-14
    (6,6,6)      16   5   1.58e-16  9.50e-16  4.7e-13    | (16,0,0)   14   5   4.28e-16  4.28e-16  2.2e-13
    (8,8,8)      14   5   4.83e-16  7.09e-16  4.7e-13    | (32,0,0)   10   5   2.01e-16  3.48e-16  3.6e-13
    (12,12,12)   12   5   2.03e-16  3.55e-16  4.7e-13    | (64,0,0)   10   5   4.94e-16  4.94e-16  4.6e-13
    (16,16,16)   12   5   2.29e-16  2.62e-16  4.7e-13    | (128,0,0)   8   5   4.11e-16  4.11e-16  9.6e-13
    (24,24,24)   10   5   1.92e-16  1.92e-16  4.7e-13
    (32,32,32)   10   5   4.44e-16  4.44e-16  4.7e-13
    (48,48,48)    8   5   2.96e-16  5.38e-16  4.7e-13
    (64,64,64)    8   5   8.53e-16  8.53e-16  4.6e-13

**16 of 16 rows: `2.6e-16` to `4.3e-15` per entry, against a flat `4.65e-13` before** — a factor
110 to 1800, and the `kR`-independence audit2 diagnosed (the same `4.65e-13` at every separation
from 3 to 128 cells) is gone.  `nCut` is now `5` at `l = 0` where the old code used `2..4`.
Over the four sweep frequencies and six offsets of `c128` (24 more rows) the worst per entry is
`4.32e-15` (at `f = 1`, `(3,3,3)`) and the worst max-norm `3.95e-15`; audit2's sweep figure for
this shape was `4.8e-13`.

### 2.2 `gen = (1/16,1/32,1/64)`, the second D2 shape

24 sweep rows plus `(64,64,64)` and `(128,0,0)`: worst per entry **`2.96e-15`** (at `(128,0,0)`,
`L = 10`), worst max-norm `2.38e-15`.  audit2: `1.4e-13` over the sweep, `1.01e-12` at
`(128,0,0)`, i.e. a factor **340** there.  The two `gen` offsets that route to (iii) are at
`8.5e-18 .. 9.7e-17`.

### 2.3 D10: `L` under-selected at large separation

    shape  D              L    nNd   eMx       pEn       eRe       eIm       audit2 pEn
    r1     (48,48,48)     12    8    4.15e-16  3.22e-15  7.02e-14  2.17e-15  1.23e-13
    r1     (64,64,64)     12    8    2.45e-16  3.83e-15  1.59e-15  6.80e-15  1.19e-13
    c32    (128,0,0)      10    6    1.91e-15  2.38e-15  5.94e-14  4.78e-14  7.20e-14

`r1 (48,48,48)` and `(64,64,64)` were `L = 10` and are now `L = 12`; `c32 (128,0,0)` was `L = 8`
and is now `L = 10`.  Per entry **`3.8e-15` against `1.2e-13`**, i.e. audit2's "falls to
`3.8-5.2e-15` when 12 shells are added" is what the selector now does on its own — with two
shells, not twelve, because Theorem A bounds the sum the code actually performs.  The `eRe` column
still carries `5.9e-14`/`7.0e-14`: those are real parts holding `< 1e-2` of their entry's modulus
(audit2 item 4, unfixed and unfixable in Float64 here).

## 3. The bound: audit2 task 7 re-run (`m7_bnd.jl`, `out/m7_bnd.txt`)

Same 60 cases (`MersenneTwister(778899)` over the twelve audit shapes plus `c32`, `c4`;
`D` uniform in `[-20,20]^3`; `f` over `{1, 1+0.1i, 0.37, 2+0.2i, 1+1i}`), same 220-bit volume
references from audit2's cache, final library at `tol = 1e-14`.  `bound` is the library's own
Theorem A/B tail at the `L` it used (`bndWhl` for route (i), the sum over the eight octants for
route (ii)); `actual = max_ab |G - Gref|`.  Two columns are new: `act/flr` with
`flr = eps * max|Gref|`, and `amp = max_r sum|term| / |sum|` of the `l`-sum the library actually
performs (computed from `fs.whl`/`fs.oct` with the same `h_l` and `Y_lm` the evaluation uses).

Each shape's table was built once at the `nMax` its five frequencies need (`m7a_need.jl`:
`c128 6, pl8 12, nd4 14, gen 9, slX/Y/Z 8, ex3 16, r1 9, r2 18, r3 25, r4 19, c32 8, c4 16`).

    bound/actual over the 60:        min 0.118    median 4.02    max 54.5    violations (<1): 15
    audit2, same 60, old library:    min 9.96e-10 median 7.48    max 419     violations: 19
    max_ab |G - Gref| / max|Gref|:   <= 3.65e-15 over the 60 (audit2: up to 1.38e-4)
    actual / (eps max|Gref|):        min 0.636   median 2.3     max 16.4
    amp = sum|term|/|sum|:           min 1.27    median 3.7     max 3.14e3 (r4 (13,-6,2), f=2+0.2i)

### Classification of the 15 remaining violations

**All 15 are (a), the rounding floor.  (b) truncation not covered by the bound: zero.  (c)
something else: zero.**  The 15 are listed in full in `out/m7_bnd.txt`; their errors are
`0.964` to `15.6` times `eps * max|Gref|` and their `l`-sum amplifications `1.78` to `388`:

    shape D              f         L    bnd/act   act/flr   amp     eMx
    r3    (-14, 13, -5)  1         28   0.717     15.6      388     3.46e-15
    r3    (12, 9, -10)   1+1i      32   0.122     10.5      45.1    2.34e-15
    r2    (17, 19, 15)   1+1i      24   0.840      6.09     36.2    1.35e-15
    pl8   (-17,-11,-7)   1         16   0.293      5.56      3.50   1.24e-15
    slX   (-2, 13, 1)    1+1i      16   0.635      4.53      2.79   1.01e-15
    pl8   (20, 14, -8)   0.37      14   0.118      2.82      2.70   6.27e-16
    r1    (7, -19, 10)   2+0.2i    16   0.119      2.76     14.5    6.13e-16
    r1    (-1,-17, 11)   1+0.1i    14   0.883      2.78      6.03   6.17e-16
    ex3   (-17, -5, 9)   0.37      18   0.154      2.35      6.81   5.21e-16
    gen   (-14,-12, 4)   0.37      14   0.319      2.33      1.78   5.17e-16
    r1    (-3, -8, -5)   1         18   0.881      2.05      2.84   4.55e-16
    gen   (-11, 16, -9)  2+0.2i    16   0.526      1.69      2.71   3.75e-16
    pl8   (-18, 17, 15)  1+0.1i    16   0.296      1.21      1.90   2.70e-16
    slX   (9, 9, 11)     1+1i      16   0.739      1.11      3.47   2.47e-16
    slX   (-17,-19,-8)   1         14   0.913      0.964     2.72   2.14e-16

In every one of them the Theorem A bound is *below* the Float64 evaluation floor of the answer
(`bound = 1.8e-22 .. 1.8e-19` against `eps max|Gref| = 2e-22 .. 3e-19`), so what it fails to cover
is rounding, not truncation.  audit2's kind (a) — "the answer is wrong, not the bound loose",
9 cases, `bound/actual` down to `9.96e-10` — is **empty**: `r3 (18,-14,-10)` went from `2.43e-3` to
`54.5`, `r3 (-14,13,-5)` from `9.60e-3` to `0.717`, `c128 (-18,7,-13)` from `3.86e-2` to `11.5`,
`r4 (13,-6,2)` from `1.50e-4` to `37.4`, `r3 (20,-2,15)` and `(4,-5,2)` and `(12,9,-10)` likewise.
audit2's kind (b), four cases at `63-396 eps`, is also empty: the largest rounding error in the
whole sample is now **16.4 eps**, on `r4 (13,-6,2)` at `f = 2+0.2i`, whose `l`-sum amplification is
`3.14e3` (that row does not violate: `bound/actual = 37.4`).

**The sentence the document can carry**: *the truncation error is certified by the bound (no
violation over the 60 cases is a truncation violation, and the largest truncation-side ratio is
`bound/actual = 54.5`); the rounding error is measured at `<= 16.4 eps` times the largest tensor
entry, with an `l`-sum amplification of at most `3.1e3`.*  The corresponding a posteriori
certificate is `bound + 20 eps max|T|`, which no row of the 60 exceeds (audit2 needed
`bound + several hundred eps` and still failed on 13 rows).

## 4. Cost

### 4.1 D4: the cost model nobody reads (`m4_cost.jl`, `out/m4_cost.txt`)

`farRoute` gained `cst::Bool = false`; `cst = true` is exactly the old behaviour.  Single thread,
`c32`, `f = 1`, best of three, load 3.56-3.67 throughout:

    block   offsets   farRoute cst=false  cst=true   diff     farBlock! after   before      saved
    32^3     32 760      46.4 ns          358.4 ns   312 ns   0.0257 s          0.0359 s    28.5%
                                                              783 ns/offset     1095 ns/offset
    64^3    262 136      33.3 ns          342.7 ns   309 ns   0.1780 s          0.2591 s    31.3%
                                                              679 ns/offset      988 ns/offset

audit2 measured `farRoute` at 360 ns with 25 ns of it used and put the waste at 330-390 ns, i.e.
35-42% of the build; the measured saving is **312 and 309 ns per offset, 28.5% and 31.3%** of the
whole far-field fill.  The residual 33-46 ns is `boundL`'s threshold scan plus the tuple work
(audit2's "25 ns").  Extrapolated to a `128^3` doubled grid (2 097 144 offsets) the far field is
**1.42 s** instead of 2.07 s, single-threaded at load 3.6.

### 4.2 D8: the geometry build's time and memory at 256 bits (`m5_tab.jl`)

Cold, one shape per process (`Sys.maxrss` and `/usr/bin/time -l` agree), `dir` pointed at an empty
directory, sized for a `32^3` block, single thread, load 3.4-4.1:

    shape  Lw  Lo  nMax  build+setup   peak RSS   cache      prep (S B.3, 256 bit, allocating momAcc)
    c32    48  51   12    44.5 s       23.4 GB    16.08 MB   201.6 s
    c8     50  51   12    44.6 s       21.8 GB    16.16 MB   -
    c4     52  52   12    53.2 s       25.1 GB    16.85 MB   -
    sl     56  55   12    64.7 s       25.3 GB    18.97 MB   -

and, from the D9 runs of S1 (larger `nMax`, so a larger table):

    r3   nMax 25  Lw 52 Lo 47   503 s    (peak RSS 18.9 GB, first build in that process)
    r2   nMax 18  Lw 42 Lo 41    71.6 s
    r4   nMax 19  Lw 42 Lo 40    74.6 s
    c2   nMax 16  Lw 38 Lo 53   171 s
    nd4  nMax 14, ex3 nMax 16, c4 nMax 16, r2/r3/r4 extended to L = 56: built inside `m7_bnd.jl`

**Time: fixed.**  `momAcc` now does its arithmetic in four preallocated MPFR registers instead of
allocating five limb buffers per term (`m9_mom.jl` checks it is bit-identical over 28518
`(acc, sum|term|)` pairs, both fields, at `nMax = 6, L = 16` and `nMax = 12, L = 20`).  The `c32`
cold build goes **201.6 s -> 44.5 s** (4.5x) and `c8` **83.0 s -> 44.6 s** (1.9x, measured back to
back in this session at load 3.5-4.0).  audit2's D8 range was 298-1770 s at 1024 bits.

**Memory: NOT fixed, and I can now say where it is.**  `m10_alloc.jl` measures the bytes allocated
per `(l,m)` column of the octant table at `L = 50`, `nMax = 12`, 256 bits:

    l        hrmPly!      momAcc x91    mulRsq! x12
    10       0.73 MB      (first call)  1.71 MB
    30       2.20 MB      9.2 kB        1.79 MB
    50       3.78 MB      9.2 kB        2.03 MB

`momAcc`, the loop that runs `3e8` times, now allocates **9 kB per column**; the remaining
2.4-5.8 MB per column is the `BigInt` churn of `hrmPly!` (the integer solid-harmonic polynomial)
and `mulRsq!` (multiplication by `r^2`).  Over the 2601 columns of an `L = 50` octant table that
is **~15 GB of GMP garbage**, which is what the peak RSS tracks.  Two attempts to make the GC give
it back failed, both measured on the `c8` cold build:

    GC.gc(false) every 16 columns   peak RSS 21.4 GB (was 21.3)   time 387 s (was 83)   -> reverted
    --heap-size-hint=3G             peak RSS 22.7 GB              time 147 s

so the pages are not lazily-collected slack that a hint can shrink; freeing them needs `hrmPly!`
and `mulRsq!` to stop allocating, exactly as `momAcc` now does (`Base.GMP.MPZ.mul!`/`add!`/`set!`
are the in-place calls). **That is the one D8 item I did not do**, and 21-25 GB per build stands.

### 4.3 D3: route (iii) threaded (`m6_thr.jl`, `out/m6_thr_t1.txt`, `out/m6_thr_t8.txt`)

`farBlock!` used to fill the route-(iii) offsets in a serial loop between its two threaded loops,
because `MOMC`, `fs.ksr` and the run-time `include` of `moments.jl` are shared.  They are now behind
`MOMLK` (the moments cache and the one-time load) and `fs.lk` (the tensor cache and its file), the
expensive `pairMoments` call is made **outside** both locks, and the whole region runs under one
ambient `setprecision(BigFloat, KSRPRC)` because `setprecision` is process-global, not task-local
(checked: a `Threads.@spawn`ed `setprecision(BigFloat, 111)` changes the main task's precision).

Cold `64x64x128` slender block, empty `KSRDIR`, `f = 1`, 524 280 offsets of which 67 route (iii):

    threads  cold farBlock!   per (iii) offset   warm farBlock!   peak RSS   load
    1        633.0 s          9.45 s             0.539 s          1.10 GB    3.21 3.38 3.80
    8        128.0 s          1.91 s             0.141 s          1.19 GB    7.54 5.55 4.63

**4.95x on 8 threads** (62% parallel efficiency), so BigFloat allocation contention does *not*
make threading useless here; the loss is that plus the `:static` imbalance of 67 items over 8
threads (9 chunks, the longest offset setting the tail).  The warm fill is 3.8x, which is the
ordinary far-field scaling.  audit2's serial figure was 684 s under load; 633 s at load 3.2 is the
same number.

**The threaded route (iii) is bitwise identical to the serial one**: the 67 tensors the two runs
wrote to their (separate, initially empty) caches, printed at 192 bits, compare equal line for line
after sorting.  The cost discontinuity of D3 itself is unchanged: `(1,1,19)` still costs
`10^5` times its neighbour `(1,1,20)`; only the wall time of a whole build improved.

## 5. The five API gaps of `deliver.md`

1. **A `dir` keyword and a table directory that exists.**  `farShape`, `farSetup`, `farTensor` and
   `farBlock!` take `dir::AbstractString = TABDIR[]`; `shpFile(s, nMax, dir)` uses it and
   `mkpath(dir)` runs on the first write.  `TABDIR[]` still defaults to
   `notes/farfield/shapetab/`, which now exists and holds the four `L = 56, nMax = 12` tables as
   symlinks under the segmented name.  `verify.jl` and `bench.jl` look for `*_p192_seg.txt` (the
   old glob `*_p192.txt` no longer matches a segmented cache) and `mkpath` the default if nothing
   is found.  `m5_tab.jl` uses `dir` to build into a fresh directory per shape, which is what the
   cold-build timings of S4.3 are.
2. **World age.**  `needMom()` now takes `MOMLK`, includes `moments.jl` once, and fetches
   `facePair`/`pairMoments` through `Base.invokelatest(getglobal, Main, :name)` into `MOMF[]`.
   The Julia 1.12 warning ("this code will error in future versions of Julia") is gone from the
   route-(iii) runs of `m6_thr.jl` and `m8_pos.jl`.  `moments.jl` is still loaded lazily, so
   `farfield.jl` stays standalone for the cubes.
3. **`farRouteStat` returns the (ii) offsets**: `(cnt, hst, ksr, oct)`.  Both `verify.jl` call
   sites updated (`cnt, hs, ksr, _ =`).
4. **The route-(iii) cache stores `N` and `Lambda`**: the line is now
   `d1 d2 d3 N Lambda re im ...` (23 tokens) and `fs.ksrI[D] = (N, Lambda)` after either a compute
   or a cache read.  21-token lines written by the previous format are still read, with
   `(N, Lambda) = (-1, NaN)`, so the 145 tensors in `notes/farfield/ksrcache/` keep working.
5. **`isFarInd`**: `isFarInd(i::NTuple{3,Int})` and `isFarInd(i1, i2, i3)`, `max(i) >= 3`, used by
   `farBlock!` and `farRouteStat` themselves so the predicate and the partition cannot drift.

### 5.1 `verify.jl` changes and its re-run

Beyond the three signature updates, `verify.jl` part (c) now falls back to the cached **tensor**
reference when the 36 face-pair values are absent (`refOf`, already in the file for part (d)):
`pairErr`, `fp` and `amp` print `-`, the assembled `tMx`/`tEn` columns are computed against the
tensor, and a new `src` column says which cache the row used.  Part (e)'s `e_ncut` table was
rewritten around `fs.nCut`/`fs.nNd`/`fs.nMx` (it used to call `nCutVec` with the pre-v2 argument
list) and now prints `|k| r_d` beside them.

The slender short-axis rows of part (c) that used to print "no cached reference" now carry
numbers, from `refill.md`'s four `tns:refQuad` references (Gila's own error, not the library's):

    sl, f = 1, ax3 = (0,0,n)    n = 2      3        4        6        8        16       32       64
    Gila tMx                    7.89e-1  2.27e-1  3.68e-2  4.50e-3  1.79e-3  2.97e-6  8.94e-7  9.12e-9
    Gila tEn                    1.60e+0  4.62e-1  7.51e-2  9.25e-3  3.70e-3  6.35e-6  2.12e-6  2.69e-8
    src                         pairs    refQuad  pairs    refQuad  pairs    pairs    refQuad  refQuad

so the `n = 3` and `n = 6` gaps in audit2's and `reference.md`'s picture of Gila's short-axis
error are filled: it is monotone, `0.79 -> 2.3e-1 -> 3.7e-2 -> 4.5e-3 -> 1.8e-3 -> 3.0e-6`.

### 5.2 The runs (single thread, load on each line)

    part  wall    load at start   headline
    a      7.8 s  3.73 3.35 3.09  divergence identity 0//1 in all 18 rows; mu_n 0 disagreements
    b     14.7 s  3.75 3.36 3.09  volume form vs 36 face pairs 1.05e-49 worst; farTensor 4.40e-15
    c      3.3 s  2.34 2.48 2.83  Gila today; slender long axis amplification 4090 -> 165000,
                                  tensor 1.17e-11 .. 1.87e-11; short axis as above
    d     19.5 s  4.01 3.44 3.12  427 rows vs the 220-bit references: eMx <= 8.60e-15,
                                  per entry <= 9.34e-15  (v2/old library: 8.6e-15 / 1.2e-14)
    e     48.9 s  5.27 4.17 3.47  nNd/nMx 6/12 (c32), 9/12 (c8), 12/12 (c4), 6/12 (sl);
                                  routing 128^3 2097132/12/0, slender 64x64x128 524102/111/67;
                                  worst digits lost 1.59 (c4 (64,0,0))
    g     30.2 s  4.55 3.92 3.35  D->-D, transpose, reflections 0.0; permutation 1.63e-15;
                                  homogeneity 6.55e-15; Float32 6.20e-8; BigFloat(160) 5.14e-16
    i     23.4 s  5.30 4.41 3.59  mean terms/offset over a 32^3 octant: 360 (c32), 480 (c8),
                                  686 (c4), 644 (sl); farBlock! 32^3 0.0416-0.0481 s

Part (f) and part (h) were not re-run (the brief's list is a,b,c,d,e,g,i; (h) is 553 s of which
492 s is Gila's own `wekTrp`).  Every part ran clean; the only failure found was the `nCut`-length
`BoundsError` in part (g), which is the latent v2 bug fixed in S0.

### 5.3 `bench.jl` (single thread, load 4.89 at start, 115 s total for both sizes)

    N     before: contact / far / rest / total     after: contact / table / far / total
    16     29.05 / 6.45 / 0.06 / 35.56 s           29.05 / 0.70 / 0.005 / 29.81 s
    32     29.05 / 23.20 / 0.09 / 52.34 s          29.05 / 0.70 / 0.026 / 29.85 s

    far fill 1271x (16^3), 893x (32^3); whole build 1.2x and 1.8x -- the rest is wekTrp at
    intOrd 48 (29.05 s of the 29.81 s and 29.85 s totals)
    staged "before" egoFur vs GlaVacOprMem's own:      0.000e+00 (bit-identical)
    "after" vs "before" egoFur, 4^3 / 16^3 / 32^3:     5.316e-14 / 2.060e-14 / 1.663e-14
    both operators on one random vector, 16^3 / 32^3:  2.550e-14 / 1.319e-14
    worst separated offset, Gila vs farBlock!:         5.579e-13 at (0,5,5) -- Gila's rule, not this
    quadrature far fill 32^3: 20.54 s, 3.827e8 kernel evaluations, 53.7 ns/eval

`deliver.md` measured 0.042 s for the `32^3` far fill under load 8-10; it is 0.026 s here, which
is the D4 saving plus the quieter machine.

## 6. The slender block's anti-Hermitian part: it is Gila's contact path (`m8_pos.jl`)

audit2 §8 found the `(6,6,12)` slender block's `Im M` indefinite at the percent level with **both**
builds at `f = 1+0.1i` and concluded the far field cannot be the cause.  Test: keep `farBlock!` on
every separated offset and replace the **eight near offsets** (the contact cell `(0,0,0)` and the
seven touching-shell offsets with all `|n_i| <= 1`) by the exact `k`-series of
`notes/moments/moments.jl` — `momentSeries` on the 36 `facePair` panels at 192 bits, `mMax = 30`,
assembled by the `srfSum!` signs, then the identity term `-1/f^2` on the diagonal of `(0,0,0)`.
Series convergence: worst last-term/total over the 288 panel pairs `6.6e-36`.  The eight offsets
cost 117 s at `f = 1+0.1i` and 114 s at `f = 1`.

How far Gila's order-9 rule is from the exact value on those eight offsets, `max_ab |Gila -
moments| / max_ab |Gila|`:

    offset   (0,0,0)  (1,0,0)  (0,1,0)  (1,1,0)  (0,0,1)  (1,0,1)  (0,1,1)  (1,1,1)
    rel      0.127    0.0244   0.0244   7.4e-5   1.17     0.0365   0.0365   8.21e-5

i.e. **117% on the touching shell along the short axis and 12.7% on the contact cell itself**,
which is `notes/moments/moments.tex`'s 1.6e-2 failure of the order-9 rule on slender cells, worse
than documented because this cell is 1:1:16 and the offending pair is the one across the 1/512 face.

    (6,6,12) slender block, dense 1296 x 1296, A = (M - M')/(2i)

    f = 1+0.1i                          lam_min          lam_max      negatives   lam_min/lam_max
      Gila everywhere                   -9.93932e-03     0.1953792    108/1296    -5.09e-2
      Gila contact + farBlock!          -1.46500e-02     0.1953886     76/1296    -7.50e-2
      moments contact + farBlock!       +3.25013e-04     0.1953884      0/1296    +1.66e-3

    f = 1 (lossless; eps*lam_max = 2.25e-18)
      Gila everywhere                   -6.55324e-14     0.01011658   596/1296    -6.48e-12
      Gila contact + farBlock!          -1.92133e-14     0.01011658   576/1296    -1.90e-12
      moments contact + farBlock!       -4.63544e-18     0.01011658   605/1296    -4.58e-16

**With exact contact integrals the anti-Hermitian part of the slender operator is positive
definite at `f = 1+0.1i` (`lam_min = +3.25e-4`, no negative eigenvalue) and positive semi-definite
to the Float64 floor at `f = 1` (`lam_min = -4.6e-18 = -2.1 eps lam_max`, against `-6.6e-14 =
-29000 eps lam_max` for Gila's own build).**  So the `-1.465e-2` audit2 measured is not a property
of the 1:1:16 discretization, as `unify.md` §7c and audit2 §8 both concluded: it is Gila's
order-9 touching-shell and contact quadrature, and it is 45 times larger than the whole
anti-Hermitian spectrum's true distance from zero.  The far-field replacement is what exposes it
(it moves `lam_min` from `-9.9e-3` to `-1.5e-2` because it removes the compensating error at
separation 2), and the contact replacement removes it entirely.

## 7. Files, and the diff

    notes/farfield/farfield.jl   1479 lines (v2: 1337, the previous deliverable: 1035)
    notes/farfield/verify.jl      958 lines (was 954)
    notes/farfield/bench.jl       194 lines (unchanged but for the table-directory glob)
    notes/farfield/shapetab/      four symlinks to the L = 56 nMax = 12 tables (segmented names)

    work/merge/farfield_final.diff  343 lines, vs work/prep/farfield_v2.jl
    work/merge/verify.diff          114 lines
    work/merge/bench.diff             8 lines
    work/merge/{farfield,verify,bench}_orig.jl   the three files as they were before this session

    work/merge/m0_smoke.jl    value pin, route, isFarInd, farRouteStat, setprecision task-locality
    work/merge/mbase.jl       common loader: audit2's reference cache, needCut (table-free selector)
    work/merge/m1_ncap.jl     the (|k| r_d, N) relation, 70 rows          -> out/m1_ncap.txt
    work/merge/m2_d9.jl       D9: the failing sweep rows and the lambda/2 cube -> out/m2_d9.txt
    work/merge/m3_d2.jl       D2 and D10 on audit2's cases                 -> out/m3_d2.txt
    work/merge/m4_cost.jl     D4: farRoute with/without the cost model     -> out/m4_cost.txt
    work/merge/m5_tab.jl      D8: cold build time and peak RSS, one shape  -> out/m5_tab_*.txt
    work/merge/m6_thr.jl      D3: cold slender block at 1 and 8 threads    -> out/m6_thr_t*.txt
    work/merge/m7a_need.jl    N needed per shape over the five frequencies
    work/merge/m7_bnd.jl      audit2 task 7 re-run, with flr and amp       -> out/m7_bnd.txt
    work/merge/m8_pos.jl      positivity with moments.jl contact integrals -> out/m8_pos.txt
    work/merge/m9_mom.jl      in-place momAcc vs the allocating one, bit for bit
    work/merge/m10_alloc.jl   where the build's allocation goes

## 8. What is still not fixed, and what is not proven

1. **D8, the memory.**  21-25 GB peak RSS per geometry build stands (S4.2).  The cause is now
   measured (`hrmPly!` + `mulRsq!`, 2.4-5.8 MB of `BigInt` garbage per `(l,m)` column, ~15 GB over
   an `L = 50` octant table) and the two cheap remedies are measured not to work.  The fix is an
   in-place `Base.GMP.MPZ` rewrite of those two functions, which I did not do.  Two concurrent
   `farShape` calls on a 36 GB machine will still fight.
2. **D3, the cost discontinuity.**  Route (iii) is now threaded (633 s -> 128 s on the cold slender
   block) but still costs `1.9-9.5 s` per offset against `0.13 ms` for its lattice neighbour, and
   `tnsKsr` still has no cap: it raises only when `nCap = 200` series terms are not enough.  A
   `lambda/8` plate at `N = 28` will still take tens of seconds per offset.
3. **D5.**  `TABPRC = 192`, so a `BigFloat` result above 192 bits is not more accurate than 192
   bits.  Untouched.
4. **D6, partially.**  `MOMC`, `MOMF` and the moments load are now behind `MOMLK`, and `fs.ksr`,
   `fs.ksrI` and the route-(iii) cache file behind `fs.lk`, so `farTensor` called from several
   threads with one shared `FrqSet` is safe; `SHPC` was already behind `SHPLK`.  What is **not**
   safe is `setprecision`: it is process-global, so a user who runs `farTensor` on threads at
   different `BigFloat` precisions can still corrupt a route-(iii) result.  `farBlock!` pins the
   precision for its own threaded region; a bare threaded `farTensor` loop does not.
5. **Per-part accuracy (audit2 item 4).**  Unchanged and unfixable in Float64 here: an entry's real
   or imaginary part that carries `< 1e-3` of `max|T|` keeps only absolute accuracy `eps max|T|`.
   It is what the `eRe`/`eIm` columns of S1.3, S2.3 and S5.2 still show at `1e-13 .. 9e-12`.
6. **`est(R)` is still the default scale** and still over-states `max|T|` by 1.1-6.3 (135 on the
   needle), so what the rule certifies is `tol * est/max|T|`; `scl = :low` (the proven `estLo`) is
   one keyword away and vacuous at `lambda/4`.  Unchanged from `prep.md`.
7. **`NCAP = 40` is a policy, not a theorem.**  Above `|k| r_d ~ 22` the library refuses instead of
   being wrong, which is the fix D9 needed, but nothing says the expansion is the right tool there;
   the `lambda/2` and `lambda/1` cubes are simply expensive (`N = 16` and `24`, table 171 s and up).
8. **The `N` values of S1.2 are the bound's, not measured optima.**  `jtail` is a majorant; the
   measured errors of S1.3 (`3e-16 .. 1.7e-13`) are 2-4 orders below the `tol * est` budget the cut
   was sized against, so `N` is over-selected by an unmeasured margin.
9. **Route (iii) offsets of the D9 shapes were not re-verified** (`(0,0,2)`, `(0,0,4)` of `r2`,
   `r3`, `r4`, 24 tensors at 10-76 s each): audit2 §11.1 verified that route at
   `8.3e-18 .. 3.6e-15` per entry on 88 rows and nothing in this session touched it.
10. **Parts (f) and (h) of `verify.jl` were not re-run** (the brief's list excluded them); (h) is
    the positivity part, whose slender rows S6 supersedes anyway, and (f) is the famG overlap.
11. **The `moments.jl` contact rebuild of S6 is a diagnosis, not a deliverable.**  It costs 115 s
    per (shape, frequency) for eight offsets and is not wired into `farBlock!` or Gila; and I
    verified its series only by its own last-term ratio (`6.6e-36`) and by the fact that it makes
    `Im M` positive definite, not against an independent 220-bit contact reference.

## 9. Summary of the numbers

    D9  N needed vs |k| r_d (table-free, 70 rows)        4 (0.03) .. 12 (2.7) .. 25 (9.0) .. 39 (21.9)
        NMAX = 12 is exactly Gila's range               |k| r_d <= 2.72 at longest edge = lambda/4
        r3/r2/r4 at |k| r_d 4.46 .. 8.96, per entry     1.7e-10 .. 9.6e-3  ->  2.7e-14 .. 1.71e-13
        lambda/2 cube, four route-(i) offsets           7.2e-13 .. 3.0e-12 ->  6.0e-16 .. 2.7e-15
        above NCAP = 40 the library errors               "not certified within N = 40 at |k| r_d ="
    D2  c128 (n,n,n) and (n,0,0), 16 rows, per entry    4.7e-13 flat -> 2.6e-16 .. 4.3e-15
        c128 sweep, 24 rows                             4.8e-13 -> 4.32e-15
        gen (128,0,0)                                   1.01e-12 -> 2.96e-15
    D10 r1 (48,48,48) / (64,64,64) / c32 (128,0,0)      1.23e-13 / 1.19e-13 / 7.2e-14
                                                        -> 3.22e-15 / 3.83e-15 / 2.38e-15
    bound, audit2's 60 cases       bound/actual         min 0.118  median 4.02  max 54.5, 15 < 1
        (old: min 9.96e-10, 19 < 1)  all 15 are the rounding floor, none is truncation
        rounding error over the 60                      <= 16.4 eps max|Gref| (audit2: 396 eps)
        l-sum amplification sum|term|/|sum|             1.27 .. 3.14e3
        max-norm error over the 60                      <= 3.65e-15 (audit2: up to 1.38e-4)
    D4  farRoute 32^3 / 64^3                            358 -> 46 ns, 343 -> 33 ns
        farBlock! 32^3 / 64^3, 1 thread, load 3.6       1095 -> 783 ns, 988 -> 679 ns/offset
    D3  cold slender 64x64x128, 1 / 8 threads           633 s / 128 s (4.95x), warm 0.539 / 0.141 s
        threaded route (iii) vs serial                  bitwise identical, 67 tensors
    D8  cold table build c32 / c8 / c4 / sl at 256 bit  44.5 / 44.6 / 53.2 / 64.7 s
        (c32 was 201.6 s in prep, c8 83.0 s here before the in-place momAcc)
        peak RSS                                        23.4 / 21.8 / 25.1 / 25.3 GB  (NOT fixed)
        in-place momAcc vs allocating                   bit-identical, 28518 pairs; whole c8 table
                                                        byte-identical to the pre-change build
    verify.jl a/b/c/d/e/g/i                             7.8/14.7/3.3/19.5/48.9/30.2/23.4 s
        (d) 427 references                              eMx <= 8.60e-15, per entry <= 9.34e-15
        (g) permutation / homogeneity / Float32          1.63e-15 / 6.55e-15 / 6.20e-8
        (e) nNd/nMx c32,c8,c4,sl                        6/12, 9/12, 12/12, 6/12
    bench.jl 16^3 / 32^3, 1 thread                      far fill 1271x / 893x, build 1.2x / 1.8x
        egoFur after vs before                          2.060e-14 / 1.663e-14
        operator on a random vector                     2.550e-14 / 1.319e-14
    slender (6,6,12) Im M lam_min at f = 1+0.1i         Gila -9.94e-3, +farBlock! -1.465e-2,
                                                        +moments contact +3.25e-4 (0 negatives)
        at f = 1                                        -6.55e-14 -> -1.92e-14 -> -4.64e-18
        Gila's contact error vs the exact k-series       12.7% at (0,0,0), 117% at (0,0,1)
