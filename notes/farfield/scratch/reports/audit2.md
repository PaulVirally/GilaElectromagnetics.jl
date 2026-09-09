# audit2 — adversarial audit of `notes/farfield/farfield.jl`

Work dir `SCRATCH/work/audit2/`.  `farfield.jl` was **not** modified.  Reference: an independent
220-bit graded Gauss–Legendre rule on the difference box (`volTensor` of `work/reference/ref.jl`,
copied unmodified to `work/audit2/myref.jl` with only its cache directory changed), which evaluates
`(1/V_t) int_D w(d) [(d_a d_b + dab k^2) g](R+d) dd` directly — no addition theorem, no moments, no
face pairs, no shared code with `farfield.jl`.  Its own convergence floor is measured per case
(order `o` vs `o-6`) and is reported in every table.  Over the 115 references of the routing and
frequency sections it is **3.3e-22 at worst and 1e-43 or better in 90 % of cases**; over the 288
references of the shape sweep of §1 it is **median 6.0e-50 and below 1e-20 in 280 of 288**, the
eight exceptions being the extreme-aspect needle offsets named in §1.3 (four of them at 3.6e-15,
which is then the floor of those four rows and not the library's error).

Machine: M3 Pro, 12 cores.  Two Julia processes of this agent at any time, as instructed; other
agents' processes were running throughout and the load average is recorded next to every timing.

## Reading order

Sections were appended as they were measured, so the file is not in numerical order.  In order:
**0** (early defect list, superseded by 11.3), **1** (arbitrary cell shapes: 1.1-1.2 the twelve
shapes, 1.3 the reference, 1.4 the result, 1.5 the `nMax = 12` cap, 1.6/1.6b/1.6c the `lambda/128`
failure and its cause, 1.7 cells above `lambda/4` and `s = 1/2`), **2** (routing boundary),
**3** (layout and signs vs Gila), **4** (threading and determinism), **5** (number types),
**6** (frequencies), **7** (the bound), **8** (anti-Hermitian positivity, 8.1 the reflection
identity), **9** (cost), **10** (scripts), **11** (verdict).

## 0. Summary of defects found

| # | defect | evidence |
|---|---|---|
| D1 | **`Float32` is not supported: the tail bound overflows to `NaN` and every offset falls through to route (iii)** | §5 |
| D2 | **the n-truncation cut `nCutVec` saturates silently at `nMax = 12`** — no error, no warning, the bound is simply not met | §5, §6 |
| D3 | **route (iii) has no cost bound: 10.7 s to 65.8 s per offset, a factor 5e4–5e5 cost discontinuity across one lattice step** | §2 |
| D4 | `farRoute` computes a cost model (`costWhl`/`costOct`) that the routing rule never uses, on every offset | §9 |
| D5 | BigFloat output above 192 bits is not more accurate than 192 bits (the geometry table is stored at `TABPRC = 192`) | §5 |
| D6 | `MOMC`, `fs.ksr` and `needMom()`'s runtime `Base.include` are unguarded global mutable state; `farTensor` is not thread safe when route (iii) is reachable | §4 |
| D7 | a Julia 1.12 world-age deprecation warning is emitted by route (iii) ("this code will error in future versions of Julia") | §5 |
| D8 | shape-table build peaks at **9.2 GB** of RSS and takes 307–560 s per shape | §1, §9 |

*This table was written before the shape sweep.  It is superseded by the complete list in §11.3,
which adds two defects (D9, the `nMax = 12` cap, the most serious in this report; D10, `L`
under-selected at large separation), restates D2 as a two-sided failure, revises D8 to
**298–1770 s and 18.9 GB peak RSS**, revises D3's worst per-offset cost to **76.3 s**, and records
for each defect whether `work/prep/farfield_v2.jl` fixes it.*

## 2. The routing boundary (task 2) — `work/audit2/s10a_route.jl`, `s10b_bdry.jl`

### 2.1 Where the boundaries are

`farRoute` over `n3 = 0..48` for every `(n1,n2)` with `n1,n2 <= 3`, `tol = 1e-13`
(`out/s10a_route.txt`).  **Identical at `f = 1` and `f = 1+0.1i`, and identical for the two shapes
except for one lattice step in the `(2,n2)` classes** — as it must be, since
`rho = r_d/|R| = 22.63/n3` is the same number for `(1/32,1/32,1/512)` and `(1/8,1/8,1/128)`.

    class          slZ = (1/32,1/32,1/512)              pl8 = (1/8,1/8,1/128)
                   (iii)      (ii)        (i)           (iii)      (ii)        (i)
    (0,0,n3)       2..19      20..39      >= 40         2..19      20..39      >= 40
    (1,0,n3)       2..19      20..35      >= 36         2..19      20..36      >= 37
    (1,1,n3)       2..19      20..31      >= 32         2..19      20..32      >= 33
    (2,0,n3)       -          0..22       >= 23         -          0..23       >= 24
    (2,1,n3)       -          0..15       >= 16         -          0..16       >= 17
    (2,2,n3)       -          -           all           -          -           all
    (3,*,n3)       -          -           all           -          -           all

So route (iii) is used on exactly 72 offsets per shape (4 classes x n3 = 2..19) — the same 72 for
the `lambda/8` plate as for the `lambda/32` needle.

### 2.2 Both sides of every boundary, against the 220-bit volume reference

`eMx` = max over the 9 entries of `|G-Gr|/max|Gr|`; `pEn`, `eRe`, `eIm` per entry (entries above
1e-8 of the largest), `Re`/`Im` relative to that entry's own `Re`/`Im`.  `bound` is the library's
own predicted absolute error at the `L`/`L_j`/`N` it actually used (`bndCum` at that `L`, summed
over the 8 octants for route (ii); `1e-16*est` for route (iii), which is what `tnsKsr`'s `N`
guarantees).  `t` is the wall time of the single `farTensor` call.

    shape f       D            rt  L/N  rho    eMx      pEn      eRe      eIm      bnd/act   t
    slZ   1       (0,0,19)     iii  15  1.19   5.1e-17  5.1e-17  5.1e-17  2.2e-17     5.99   10.7  s
    slZ   1       (0,0,20)     ii   53  1.13   2.5e-15  2.5e-15  2.5e-15  6.3e-16   115      0.23 ms
    slZ   1       (1,0,19)     iii  16  0.912  6.7e-17  9.3e-17  9.3e-17  5.6e-17     5.10   10.3  s
    slZ   1       (1,0,20)     ii   54  0.884  1.7e-15  1.0e-14  1.1e-14  2.7e-16   146      0.15 ms
    slZ   1       (1,1,19)     iii  17  0.767  2.4e-17  4.3e-17  4.3e-17  6.8e-17    17.0    13.6  s
    slZ   1       (1,1,20)     ii   55  0.75   1.6e-15  4.4e-15  4.4e-15  2.3e-16   183      0.13 ms
    slZ   1       (2,0,22)     ii   34  0.583  5.6e-16  8.1e-16  8.1e-16  3.0e-16   373      0.10 ms
    slZ   1       (2,0,23)     i    56  0.575  4.4e-16  7.1e-16  7.3e-16  3.0e-16   492      0.06 ms
    slZ   1       (2,1,15)     ii   40  0.584  1.1e-15  1.3e-15  1.3e-15  1.8e-16   158      0.10 ms
    slZ   1       (2,1,16)     i    56  0.578  1.7e-15  2.8e-15  2.8e-15  2.8e-16   230      0.04 ms
    pl8   1       (0,0,19)     iii  22  1.19   7.8e-17  7.8e-17  8.0e-17  5.1e-17     4.84   13.2  s
    pl8   1       (0,0,20)     ii   54  1.13   1.6e-15  1.6e-15  1.7e-15  1.6e-16   140      0.19 ms
    pl8   1       (1,0,19)     iii  25  0.912  6.4e-17  7.7e-17  8.9e-17  6.8e-17     7.52   25.5  s
    pl8   1       (1,0,20)     ii   55  0.884  3.2e-16  3.7e-16  4.0e-16  2.6e-16   960      0.16 ms
    pl8   1       (1,1,19)     iii  28  0.767  6.4e-17  6.8e-17  6.9e-17  6.3e-17     9.43   65.8  s
    pl8   1       (1,1,20)     ii   55  0.75   3.9e-16  4.3e-16  4.3e-16  2.7e-16  1010      0.13 ms
    pl8   1       (2,0,23)     ii   34  0.575  6.1e-16  7.1e-16  8.4e-16  2.9e-16   415      0.10 ms
    pl8   1       (2,0,24)     i    56  0.566  2.0e-15  2.4e-15  4.5e-15  3.0e-15    89.4    0.07 ms
    pl8   1       (2,1,16)     ii   39  0.578  3.1e-16  4.8e-16  4.9e-16  3.3e-16   650      0.10 ms
    pl8   1       (2,1,17)     i    56  0.572  2.7e-15  2.8e-15  2.8e-15  2.9e-15   137      0.04 ms

and the same 20 rows at `f = 1+0.1i` (`out/s10b_bdry.txt`), whose worst numbers are
`eMx 2.4e-15`, `pEn 3.6e-15`, `eRe 1.17e-14` (pl8 (2,0,24)), `eIm 4.8e-15`, `bnd/act >= 5.32`.

**Every boundary offset on both sides meets 1e-13 per entry, and per part.**  Worst per-entry over
the 40 rows: `1.0e-14` (slZ (1,0,20), route (ii)); worst per-part `1.17e-14` (pl8 (2,0,24), Re).
**The bound is never violated**: `bound/actual` ranges from **4.84** (pl8 (0,0,19), route (iii)) to
**1010** (pl8 (1,1,20), route (ii)).

### 2.3 What the boundary costs (defect D3)

The accuracy is continuous across the boundary; the **cost is not**.  At `(1,1,19)` on the
`lambda/8` plate one offset costs **65.8 s**; its neighbour `(1,1,20)` costs **0.13 ms**.  That is a
factor of **5.1e5** across one lattice step, and it is a property of the routing rule, not of the
machine: route (iii) runs `pairMoments` at BigFloat(128) on 36 face pairs to order `N`, and `N`
grows with `|k| D_max`, which is 16x larger for the plate than for the needle at the same `rho`:

    shape  D           N    first call (moments cold)   second frequency (moments cached in MOMC)
    slZ    (0,0,19)    15    10.7 s                      1.22 s
    slZ    (1,0,19)    16    10.3 s                      0.80 s
    slZ    (1,1,19)    17    13.6 s                      0.67 s
    pl8    (0,0,19)    22    13.2 s                     15.0   s   (recomputed: see below)
    pl8    (1,0,19)    25    25.5 s                      0.94 s
    pl8    (1,1,19)    28    65.8 s                      0.71 s

At 72 route-(iii) offsets per shape this is **~13 min of BigFloat moments for the `lambda/32`
needle and, extrapolating the measured `(1,1,19)` cost, of the order of 30–50 min for the
`lambda/8` plate**, once per shape.  `unify.md` §3b quotes 3.6–13.7 s per offset, measured on the
`lambda/32` needle only; the `lambda/8` plate is a shape squarely inside Gila's stated range
(`lambda/128`..`lambda/4`) and costs up to **5x more per offset**, at `N = 28` rather than 15–17.
No cap and no fallback exists: `tnsKsr` raises only when `nCap = 200` terms are not enough.

## 3. Layout and sign equivalence with Gila (task 3) — `work/audit2/s3_gila.jl`

`egoToe` built twice for the same volume: once by Gila's own path (`GlaVol`, `CPUKerOpt(f,48,false,
CPU())`, `sepGrd(vol,vol,0)`, `egoFunInn!` on every Toeplitz index, `egoFunSng!`/`wekTrp` on the
indices all `<= 2`), once by taking that array, overwriting every entry with `max(i) >= 3` with
`NaN` and calling `farBlock!` on it.  The eight contact blocks are therefore bit-identical by
construction and only the separated offsets are compared.  Both arrays are then run through Gila's
own `egoToeCrc!` onto the doubled grid.

    band = max-norm separation in cells;  maxrel = max_ab |Gila - farBlock!| / max_ab |Gila|

    c32, 10^3 cells, lambda/32, f = 1                 f = 1+0.1i
    band  maxrel     at                               maxrel     at
    2     3.16e-13   (2,0,0)                          2.81e-13   (0,2,0)
    3     4.50e-13   (3,3,3)                          2.25e-13   (3,3,3)
    4     4.74e-13   (4,4,4)                          3.48e-13   (4,4,0)
    5     5.58e-13   (0,5,5)                          2.98e-13   (5,5,5)
    6     4.55e-13   (2,4,6)                          3.78e-13   (3,3,6)
    7     4.16e-13   (7,0,0)                          4.16e-13   (5,6,7)
    8     3.47e-13   (8,7,7)                          2.95e-13   (6,8,0)
    9     2.85e-13   (9,7,7)                          2.97e-13   (9,9,7)

    sign-flipped entries (|A+B| < 1e-6|A| on an entry above 1e-3 of the largest): 0 out of 8748
    farBlock! output symmetry max|G - G^T|/max|G|: exactly 0.0
    circulant arrays (20^3 grid, egoToeCrc!): max|CG - CF| / max|CG| = 9.81e-15 (f = 1),
      8.68e-15 (f = 1+0.1i); entries differing by more than 1e-10 of max|CG| = 0.67: 0 out of 72000

**No sign error, no index transposition, no layout mismatch.**  The band figures 2.3e-13 to 5.6e-13
are Gila's own fixed-rule error (`reference.md` measured Gila at 5.0e-14..8.7e-13 at `lambda/32`
against the 220-bit reference, and §6 of `unify.md` puts `farTensor` at 4.2e-15 there), i.e. the
difference is Gila's, not the library's.  The circulant comparison collapses to 1e-14 relative
because `max|CG|` is the self term (0.67) while the far entries are 1e-3 or smaller.

    slender 6x6x12 at (1/32,1/32,1/512), f = 1
    band   2       3       4       5       6       7        8        9        10       11
    maxrel 0.616   0.226   0.0367  0.0193  0.0045  0.00543  0.00179  5.95e-4  1.94e-4  5.76e-5
    argmax (0,0,2) (0,0,3) (0,0,4) (0,0,5) (0,0,6) (0,0,7)  (0,0,8)  (0,0,9)  (0,0,10) (0,0,11)
    sign-flipped entries: 0     farBlock! asymmetry: 0.0
    circulant: max|CG-CF|/max|CG| = 0.0555, 204 of 20736 entries differ by more than 1e-10 of
      max|CG| = 0.825; worst absolute difference 0.0458

**Every one of those differences is at `(0,0,n3)`, and it is Gila that is wrong there**, not
`farBlock!`: §2 verifies `farTensor` at `(0,0,19)` to 5.1e-17 against the 220-bit volume reference,
and `reference.md` already measured Gila's fixed rule at 0.79 (n = 2), 1.8e-3 (n = 8) relative on
this shape.  Excluding the short-axis needle would leave the c32-like 1e-13 agreement; the run
reports the maximum over the whole band, so the needle dominates every band up to 11.  The
important audit result is the second line: **no sign-flipped entry and no index mismatch anywhere,
including on the block where the two builds differ by 62 %.**

## 5. Genericity in the number type (task 5) — `work/audit2/s5_gen.jl`, `out/s5_gen.txt`

All rows at `s = (1/32)^3`, `D = (5,1,0)`, `f = 1`, where the Float64 library returns
`T[1,1] = 0.0015707382212663695 + 0.00036015507920717844im` by route (i) at `L = 28`.

### 5.1 Float32 (defect D1) — the arithmetic is fine, the *bound* overflows

    Float32:  every entry of fs.thr is Inf          -> boundL always returns -1
              37 of the 73 sub-box bound terms fs.oT are NaN
              farRoute((5,1,0)) = (3, -1)           -> route (iii), the BigFloat k-series
              one farTensor call: 20.9 s            (Float64, route (i): ~1 us)
              value: 0.0015707383f0 + 0.00036015507f0im, 3.07e-8 from Float64 = 0.257 ulp(Float32)

The cause, from a direct evaluation of `bndTrm` in both types:

    l                    0          10         20         30        40    50    60    70
    Float64 term    2.334e-2   2.842e-18  5.118e-39  1.401e-61   2.3e-85  4e-110 1e-135 9e-162
    Float32 term    2.334e-2   2.842e-18  5.118e-39  0.0         NaN     NaN    NaN    NaN

`bndTrm` forms `pf * (2l+1) * sqrt(2l+5) * abs(k)^l * exp(...) / df * wl` left to right.  In
Float32, `abs(k)^l = 6.283^l` overflows at `l = 49` and the double factorial `df = (2l+1)!!`
overflows at `l = 29`; between 30 and 48 the term is `finite/Inf = 0` (harmless) and from `l = 49`
it is `Inf/Inf = NaN`.  `bndCum` accumulates from the top, so **one NaN poisons every cumulative
tail**, every `cum[i] <= bud` test is false, `whlThr` leaves the whole threshold table at `Inf`,
and `boundLoct` returns `-1` in every octant.  Every offset therefore falls through to route (iii).

Consequences, all measured, none of them documented in `unify.md` (whose §7 reports only
"Complex{Float32} in -> ComplexF32 out, 3.07e-8 from the Float64 value ... so `T` is genuinely
generic"):

- a Float32 `farBlock!` on a 128^3 block would take **20.9 s x 2.1e6 offsets = 1.4 years**;
- Float32 silently requires `notes/moments/moments.jl` and the whole BigFloat k-series machinery,
  which `unify.md` §1.4 calls "the only part of `farfield.jl` that is not standalone";
- route (ii) is unreachable in Float32, so if route (iii) ever failed there is no fallback.

**The Float32 arithmetic itself is correct.**  Forcing route (i) by hand at the `L = 28` that the
Float64 bound selects gives `6.2e-8 = 0.52 ulp(Float32)` from the Float64 value, and the entry-wise
Float32 errors through route (iii) are 0.12 to 0.35 ulp:

    entry   (1,1)     (1,2)     (2,1)     (2,2)     (3,3)
    rel     3.07e-8   3.05e-8   3.05e-8   1.43e-8   4.21e-8
    ulp32   0.257     0.256     0.256     0.120     0.353

so the fix is confined to `bndTrm`/`bndCum` (evaluate the bound in `Float64` or in logs regardless
of `T`), not to the expansion.

### 5.2 BigFloat — generic, but capped at 192 bits (defect D5)

    precision   route   eltype(out)         precision(out)   |G_big - G_float64| / max
    128 bit     (1,28)  Complex{BigFloat}   128              5.06e-16
    192 bit     (1,28)  Complex{BigFloat}   192              5.06e-16
    256 bit     (1,28)  Complex{BigFloat}   256              5.06e-16

    BigFloat(128) vs BigFloat(256):  3.15e-38     eps(128 bit) = 2.94e-39
    BigFloat(192) vs BigFloat(256):  3.95e-58     eps(192 bit) = 1.59e-58

The 128-bit answer differs from the 256-bit answer **in the 38th digit, not the 16th** — the
BRIEF's test for silently-Float64 constants passes: no untyped constant is used in the generic path.
A grep of every numeric literal in `farfield.jl` confirms it: `T(pi)` or `BigFloat(pi)` at lines
102, 376, 464, 487, 550, 569, 576, 659, 750, 907, 911, 919; the only bare `pi` (line 668) and the
only bare `0.0`/`2*fr` (lines 648, 664, 666) are inside the **Float64-specific method** of
`phsSeed`, where Float64 is the correct type; `0.30103` and `1.6` (lines 700–701) are dimensionless
heuristics for Miller's starting index and feed an `Int`.

But the 192-vs-256 row is the ceiling: **asking for more than 192 bits buys nothing**, because
`farShape` stores the geometry table at `TABPRC = 192` and `frqBox` contracts 192-bit inputs at
`max(192, precision(T)+64)` bits.  A BigFloat(256) result advertises 77 digits and carries 58.
`unify.md` §7 states "BigFloat in gives a BigFloat result at the ambient precision", which is true
of the *type* and false of the *accuracy*.

### 5.3 A Julia 1.12 deprecation (defect D7)

Route (iii) prints, on its first call,

    Julia 1.12 has introduced more strict world age semantics for global bindings.
    !!! This code may malfunction under Revise.
    !!! This code will error in future versions of Julia.
    Hint: Add an appropriate `invokelatest` around the access to this binding.

from `needMom()`'s `getglobal(Main, :facePair)` / `getglobal(Main, :pairMoments)` after a runtime
`Base.include(Main, MOMJL[])` (farfield.jl lines 844–850).  The call sites are wrapped in
`Base.invokelatest`, the *binding reads* are not.

## 6. Frequencies (task 6) — `work/audit2/s11_frq.jl`, `out/s11_frq.txt`

`s = (1/32)^3`, four offsets, reference = the 220-bit volume rule at order 32 (self-convergence
1.7e-47 to 1.9e-64 on these offsets).  `bnd` is the library's own predicted absolute truncation
error at the `L` it used; `act = max|G-Gr|`.

    f          D            kR      rt  L    eMx      pEn      eRe      eIm      bnd/act  max|Gr|
    0.05       (2,0,0)      0.0196  ii  55   4.1e-15  4.1e-15  4.1e-15  2.1e-16   24.3     7.77
    0.05       (5,1,0)      0.0501  i   28   9.5e-16  1.2e-15  1.2e-15  1.9e-16   16.4     0.453
    0.05       (16,16,16)   0.272   i   10   2.8e-15  3.6e-15  3.8e-15  8.1e-17   78.3     1.51e-3
    0.05       (32,0,0)     0.314   i   10   6.9e-16  1.2e-15  1.2e-15  8.2e-17   28.5     2.04e-3
    10         (2,0,0)      3.93    ii  56   9.7e-16  9.7e-16  1.9e-15  1.1e-16  252       1.07e-3
    10         (5,1,0)      10.0    i   32   2.0e-15  2.5e-15  2.4e-15  3.1e-15   20.5     4.31e-4
    10         (16,16,16)   54.4    i   18   6.0e-16  1.2e-15  2.8e-16  1.4e-15  321       5.33e-5
    10         (32,0,0)     62.8    i   18   1.9e-15  1.1e-14  7.5e-13  1.2e-13   37.6     6.88e-5
    1+3i       (2,0,0)      1.24    ii  55   3.7e-15  3.7e-15  5.3e-15  2.3e-15   39.8     1.35e-3
    1+3i       (5,1,0)      3.17    i   28   5.1e-16  8.1e-16  4.0e-15  6.0e-16   84.9     4.25e-5
    1+3i       (16,16,16)   17.2    i   14   7.4e-16  1.2e-15  1.0e-15  1.3e-15    5.26    6.16e-12
    1+3i       (32,0,0)     19.9    i   14   1.2e-15  3.5e-15  2.7e-15  7.6e-14    0.692   6.73e-13
    -1         (2,0,0)      0.393   ii  55   4.2e-15  4.2e-15  4.2e-15  2.5e-16   32.3     2.08e-2
    -1         (5,1,0)      1.0     i   28   5.7e-16  1.7e-15  2.2e-15  4.7e-16   49.2     1.61e-3
    -1         (16,16,16)   5.44    i   12   5.1e-16  9.6e-16  5.5e-16  2.9e-15    6.43    7.36e-5
    -1         (32,0,0)     6.28    i   12   5.0e-16  1.3e-15  8.1e-15  7.4e-16    0.869   9.44e-5
    1-0.1i     (2,0,0)      0.395   ii  55   1.4e-15  1.9e-15  1.9e-15  1.9e-15  100        2.07e-2
    1-0.1i     (5,1,0)      1.01    i   28   6.8e-16  1.0e-15  1.1e-15  8.4e-16   43.2     1.68e-3
    1-0.1i     (16,16,16)   5.47    i   12   3.7e-16  5.7e-16  5.8e-16  3.9e-16    9.03    1.27e-4
    1-0.1i     (32,0,0)     6.31    i   12   4.7e-16  1.3e-15  3.9e-15  2.6e-15    0.959   1.74e-4
    1          (32,0,0)     6.28    i   12   5.0e-16  1.3e-15  8.1e-15  7.4e-16    0.869   9.44e-5

Findings, one per frequency the task names.

- **`f = 0.05` (small `kR`, large `h_l`).**  Works.  `est` and the bound both lose their `k`
  dependence as `kR -> 0` (the `3/(kR)^2` term of `est` and the `(2l+3)!!/(kR)^{l+3}` growth of
  `hbnd` cancel the `|k|^l W_l` of `bndTrm`), and `thr` comes out at a *fixed* set of radii
  independent of `f`: `L = 4` beyond `|R| = 23.5`, `L = 6` beyond 3.66, `L = 8` beyond 1.49.
  Worst error 4.1e-15, `bnd/act >= 16.4`.  No underflow, no overflow.
- **`f = 10` (`kR = 62.8` at 32 cells).**  Works to 1.1e-14 per entry, but **`nCut` saturates at
  `nMax = 12` for at least one `l` and the library says nothing** (defect D2): `nCutVec` initialises
  `v = fill(nMax, lTop+1)` and only lowers an entry when the `jtail` bound is met, so a saturated
  entry is indistinguishable from a certified one.  At this frequency the `lambda/32` cell is
  `0.3125 lambda` across — outside Gila's stated `lambda/128..lambda/4` — and the per-*part* errors
  break 1e-13 at `(32,0,0)`: `eRe = 7.5e-13`, `eIm = 1.2e-13`, with `eMx = 1.9e-15`.  That is the
  known "one complex sum" limitation (`unify.md` §9) appearing at a frequency `unify.md` did not
  test; the seed and the `L = 18` floor are not the cause (`eMx` is at rounding).
- **`f = 1+3i` (strong damping).**  Works; no underflow.  At `(32,0,0)` the tensor is
  `6.7e-13` in modulus (`e^{-2 pi * 3 * 1}` over 32 cells) and the library still returns
  `1.2e-15` relative.  `est(R)` carries the same `exp(-Im(k)|R|)`, so the *relative* tolerance is
  preserved as the field decays: at `(16,16,16)` `est` has fallen by `e^{-17.2}` and `L` still
  comes out at 14.  Nothing breaks.  **But this is where the bound is smallest relative to
  rounding, and `bnd/act = 0.692 < 1`** — see below.
- **`f = -1` (negative real).**  Not an error, and not wrong: `prf = 0 - 6.2832i` (the conjugate of
  the `f = +1` prefactor), `hFill!` takes the real branch with a negative argument, `bsljR!` and
  `bsly!` are correct for `x < 0`, and the resulting tensor matches the 220-bit reference to
  `5.0e-16 .. 4.2e-15` — **the same numbers, entry by entry, as `f = +1`**, i.e. the library
  returns exactly the conjugate, which is what Gila's `cispi(2 r f)` convention demands.
- **`f = 1-0.1i` (gain medium, `Im f < 0`).**  Nothing assumes `Im f >= 0`.  `est` and `hbnd` both
  carry `exp(-Im(k) r) = exp(+|Im k| r)`, which is the correct growth of `|h_l|`; `hnkUp!` is used
  (the complex branch) and remains stable.  Errors `3.7e-16 .. 3.9e-15`, `bnd/act >= 0.959`.

**The bound is not a bound on the delivered error, only on the truncation** (see also §7).  Four of
these 21 rows have `bnd/act < 1`: `1+3i (32,0,0)` 0.692, `f = 1` and `f = -1` `(32,0,0)` 0.869,
`1-0.1i (32,0,0)` 0.959.  In every one of them the predicted truncation bound is *four orders of
magnitude below the Float64 rounding floor* of the answer (`f = 1`, `(32,0,0)`: bound `4.13e-20`,
actual `4.75e-20`, `eps * max|Gr| = 2.1e-20`), so what the bound fails to cover is rounding, not
truncation.  `bound + eps*max|Gr|` is never violated in any row.

## 1. Arbitrary cell shapes (task 1)

### 1.1 The twelve shapes and what their geometry tables cost — `work/audit2/s1_tab.jl`

Random shapes as `auditG` did them: `s = (1/32) 10^u`, `u_i ~ U[-1.5,1.5]`, `MersenneTwister(20260907)`,
rounded to dyadic rationals keeping 21 bits of relative precision (`shapes.jl`).  Every table is
built by the deliverable itself, `farShape(s)` at its defaults `L = 56`, `nMax = 12`, `BLDPRC = 1024`,
stored at `TABPRC = 192`.

    name  s                                asp     max edge  build (s)  cnc whole  cnc octant  peak RSS
    c128  (1/128, 1/128, 1/128)              1.0    0.0078     [cached]   7.31e11   1.96e11     0.39 GB
    pl8   (1/8, 1/8, 1/128)                 16.0    0.125      307        3.37e9    1.67e9      9.15 GB
    nd4   (1/4, 1/64, 1/64)                 16.0    0.25       (see 9)    3.68e2    1.63e6      3.50 GB
    gen   (1/16, 1/32, 1/64)                 4.0    0.0625     493        1.90e12   8.31e10     4.93 GB

(the remaining rows are filled in below; `build (s)` for `nd4` is contaminated by this agent's
SIGSTOP scheduling and is re-measured cleanly in §9.)

Two things worth recording before any accuracy number.  **The contraction cancellation inside the
exact geometry table is strongly shape dependent and reaches `1.9e12`** — larger than the `7.3e11`
`unify.md` §2 measured on cubes, and 5e9 times larger than `nd4`'s `368`.  The library's choice to
contract at 1024 bits is therefore not merely prudent; at `1.9e12` a Float64 table would carry 4
correct digits.  **And the build peaks at 9.15 GB of RSS for a single shape** (defect D8): two
concurrent `farShape` calls on a 36 GB machine are already half its memory, and the 19.5 MB on-disk
table gives no hint of it.

## 4. Threading and determinism (task 4) — code reading; measurements in §4.2

### 4.1 Every piece of global mutable state in `farfield.jl`

    name        line  what                                     guarded?
    SHPC        253   Dict, shape tables                       yes, ReentrantLock SHPLK, farShape only
    SHPLK       255   the lock                                 -
    TABDIR      254   Ref, table directory                     no  (setup-time)
    KSRDIR      840   Ref, route (iii) tensor cache directory   no  (setup-time)
    MOMJL       839   Ref, path to moments.jl                   no  (setup-time)
    MOMC        842   Dict, BigFloat face-pair moments          NO   -- written by tnsKsr
    fs.ksr      518   Dict inside FrqSet, route (iii) tensors   NO   -- written by ksrCached!
    Main        848   Base.include(Main, MOMJL[]) at run time   NO   -- and world-age fragile

`farBlock!` is safe by construction: the first `@threads :static` loop only writes `kn[q]`, `lw[q]`,
`lo[q]` at its own index and calls `farRoute`, which allocates but shares nothing; the route-(iii)
offsets are then filled **serially** (lines 1001–1005), so `MOMC`, `fs.ksr` and `needMom()` are only
ever touched from the main thread; the second `@threads :static` loop reads `st.ksr` and writes
disjoint views of `egoToe`, with one `FarWs` per thread taken from a vector sized
`Threads.maxthreadid()` (the fix `unify.md` §8.6 describes).  I found no race in `farBlock!`.

**`farTensor` is a different matter (defect D6).**  It calls `ksrCached!` directly (line 980), which
mutates `fs.ksr` and, through `tnsKsr`, the global `MOMC`, and may run `Base.include(Main, ...)`.
Any user who threads over `farTensor` calls sharing one `FrqSet` — the obvious way to use the
documented API, and the only way to avoid re-running `farSetup` — races on two unguarded `Dict`s
whenever any offset routes to (iii).  For the cubes that never happens; for the slender needle and
for `(1/8,1/8,1/128)` it happens on 72 offsets per shape.  Nothing in the code or the docstrings
says so.

## 7. The bound (task 7) — method

Two quantities are compared at every case: **`bound`**, the library's own predicted absolute error
at the `L`, `{L_j}` or `N` it actually selected — `bndCum(ls, t, k|R|)` evaluated at that `L` for
route (i), the sum over the eight octants of `bndCum(fs.oLs, fs.oT, k|v_j|)` at each `L_j` for
route (ii), and `1e-16 * est(R)` for route (iii) (which is exactly what `tnsKsr`'s choice of `N`
guarantees, since `N` is picked so that `ksrBnd * amp <= 1e-16` and the absolute tensor bound is
`ksrBnd * amp * est`) — against **`actual = max_ab |G - G_ref|`** from the independent 220-bit
volume rule.  Script `work/audit2/bnd.jl` (`predErr`), driven by `s7_bnd.jl`.


### 1.2 The twelve shape tables, complete (`work/audit2/s1_tab.jl`, `out/s1_tab.txt`)

`build (s)` is wall time of `farShape(s)` at the deliverable's defaults (`L = 56`, `nMax = 12`,
contraction at `BLDPRC = 1024` bits, stored at `TABPRC = 192`), measured with this agent's second
Julia process and other agents' processes running (load 3-6); `cnc` is the library's own reported
worst `sum|term|/|acc|` inside the exact contraction; `rss` is the peak resident set of the build
process.  `c128` and `slZ` are reloads of a table already on disk (1.5 s, 1.8 s) — every other row
is a real build.

    name  s                                          asp     max edge   build(s)  cnc whl   cnc oct   peak RSS
    c128  (1/128, 1/128, 1/128)                        1.0    0.00781    [1.51]    7.31e11   1.96e11   0.39 GB
    pl8   (1/8, 1/8, 1/128)                           16.0    0.125       307      3.37e9    1.67e9    9.15 GB
    nd4   (1/4, 1/64, 1/64)                           16.0    0.25       1770      3.68e2    1.63e6    3.50 GB
    gen   (1/16, 1/32, 1/64)                           4.0    0.0625      493      1.90e12   8.31e10   4.93 GB
    slX   (1/512, 1/32, 1/32)                         16.0    0.0312      490      7.31e9    2.38e11  10.10 GB
    slY   (1/32, 1/512, 1/32)                         16.0    0.0312      341      7.31e9    2.38e11   9.94 GB
    slZ   (1/32, 1/32, 1/512)                         16.0    0.0312     [1.80]    3.37e9    1.67e9    0.39 GB
    ex3   (1/4, 1/4096, 1/4096)                     1020      0.25        396      4.00e5    1.22e6    1.79 GB
    r1    (1405325/2^27, 1365905/2^24, 478835/2^26)   11.4    0.0814      867      1.88e7    2.54e7    9.96 GB
    r2    (43029/2^25, 1445443/2^28, 1587547/2^22)   295      0.379       306      4.30e5    1.97e5    8.38 GB
    r3    (743409/2^20, 1960455/2^26, 1122655/2^30)  678      0.709       361      4.35      9.73      8.98 GB
    r4    (1926005/2^22, 1244365/2^26, 351275/2^26)   87.7    0.459       317      3.92      2.21e1    8.64 GB

`r1`-`r4` are random shapes `s = (1/32) 10^u`, `u_i ~ U[-1.5, 1.5]`, `MersenneTwister(20260907)`,
snapped to dyadic rationals with 21 bits of relative precision (`work/audit2/shapes.jl`); their
aspect ratios 11.4 to 678 and the deliberate `ex3` at 1020 cover the required 1e-3..1e3 band
(`asp` is `max s / min s`; the reciprocal is the same shape permuted, and `slX`/`slY`/`slZ` verify
that the three permutations are handled alike).  The build cost is 306-1770 s **per shape** and the
peak memory is 8-10 GB for eight of the twelve — defect D8, and it is not a property of the slender
shape only: `slX` (1/512 first) peaks at 10.1 GB while `slZ` (1/512 last) is a 0.39 GB reload, so
the axis order alone changes nothing but the cache hit.

### 1.3 The sweep: what was compared against what

288 tensors: 12 shapes x 6 offsets x 4 frequencies `f = 1`, `1+0.1i`, `0.37`, `2+0.2i`.
Offsets per shape (`work/audit2/offsets.jl`), chosen relative to the shape rather than to the
lattice, so that every shape gets its own worst direction:

    2*e(argmin s)   4*e(argmin s)   2*e(argmax s)   (3,3,3)   (7,-3,2)   (16,16,16)

`e(argmin s)` is the axis of the **shortest** edge, along which the difference box reaches furthest
relative to `|R|`; `e(argmax s)` the longest.  For `c128` (a cube) the first and third coincide, so
`(2,0,0)` appears twice and the shape contributes 5 distinct offsets, not 6.

**Reference.**  `pairKer` was not used: `reference.md` and `famD` both record that it stalls
(> 15 min/offset) on slender pairs, and eight of these twelve shapes are slender.  Every reference
here is my own 220-bit graded Gauss-Legendre volume rule (`work/audit2/myref.jl`, an unmodified copy
of `work/reference/ref.jl` `volTensor`, order 32, box split at 0 and graded to the near corner),
which shares no code with `farfield.jl`, and which `reference.md` measured against the `pairKer`
face-pair reference at 3.7e-65..1e-49.  Its own convergence is measured per case as order 32 vs
order 26 (`out/s2_ref.txt`, 288 rows, 5183 s of BigFloat): **median 6.0e-50, and below 1e-20 in 280
of 288 cases.**  The eight exceptions are the extreme-aspect needle offsets

    r3 (0,0,2), all four frequencies   3.55e-15 .. 3.61e-15    (3.83e6 nodes, 50-57 s each)
    r2 (2,0,0), all four frequencies   8.89e-19 .. 8.91e-19    (3.50e6 nodes)

so for the four `r3 (0,0,2)` rows the reference itself is only good to 3.6e-15 and the reported
`eMx 3.55e-15 .. 3.59e-15` there is **the reference's floor, not the library's error** — that
offset is `|R| = 2 s_3 = 0.0021` against a box of diameter 0.71, `rho = 339`, and nothing short of
the k-series resolves it.  Those four rows are excluded from every "worst case" below and named
where they matter.

### 1.4 Result, per shape (`work/audit2/s6_shape.jl`, full 288-row table `out/s6_shape.txt`)

`eMx` = `max_ab |G - Gr| / max_ab |Gr|`; `pEn` = worst per-entry `|G-Gr|/|Gr|` over entries above
1e-8 of the largest; `eRe`, `eIm` the same for the real and imaginary parts separately, each
relative to that part's own value; `rt` the routes used ((i) whole box, (ii) octants, (iii) BigFloat
k-series); `bnd/act` the library's own predicted error over the true error, smallest over the 24
rows of the shape.

    shape  asp   rt    eMx      pEn      eRe      eIm     min bnd/act   worst row
    c128     1   1,2   4.1e-13  4.8e-13  8.7e-13  4.7e-12   0.26        (3,3,3)      f=1+0.1i  L=26 rt(i)
    pl8     16   1,2,3 6.4e-15  1.1e-14  2.1e-14  2.7e-14   1.9         (16,16,16)   f=0.37    L=12 rt(i)
    nd4     16   1,3   2.5e-15  1.5e-14  3.6e-13  8.6e-14   0.76        (16,16,16)   f=2+0.2i  L=20 rt(i)
    gen      4   1,3   4.4e-14  1.4e-13  9.8e-13  4.8e-13   0.37        (16,16,16)   f=1+0.1i  L=12 rt(i)
    slX     16   1,2,3 1.6e-15  4.3e-15  4.3e-15  8.5e-14   4.3         (3,3,3)      f=0.37    L=28 rt(i)
    slY     16   1,2,3 1.5e-15  4.3e-15  4.3e-15  2.4e-14   4.3         (3,3,3)      f=0.37    L=28 rt(i)
    slZ     16   1,2,3 1.9e-15  4.1e-15  4.1e-15  2.2e-14   6.8         (2,0,0)      f=0.37    L=47 rt(ii)
    ex3   1020   1,3   3.6e-15  3.2e-14  3.4e-13  8.8e-14   0.73        (3,3,3)      f=0.37    L=32 rt(i)
    r1    11.4   1,3   8.9e-14  1.3e-13  4.7e-13  1.2e-13   0.21        (16,16,16)   f=0.37    L=12 rt(i)
    r2     295   1,2,3 3.4e-11  1.3e-09  6.0e-09  1.3e-10   3.7e-3      (16,16,16)   f=2+0.2i  L=24 rt(i)
    r3     678   1,2,3 1.4e-04  9.6e-03  8.9e-03  3.0e-01   1.4e-09     (16,16,16)   f=2+0.2i  L=32 rt(i)
    r4    87.7   1,2,3 1.6e-08  6.4e-07  8.2e-07  4.3e-07   2.4e-05     (16,16,16)   f=2+0.2i  L=26 rt(i)

**Three shapes fail 1e-13 by six to eleven orders of magnitude, and one more (`c128`, a cube at
lambda/128, the finest cell Gila claims) fails it by a factor 4.8.  The library reports no error and
its own bound is violated by up to 7.2e8.**

### 1.5 The large-cell failure: the k-series of `j_l` is hard-capped at `nMax = 12`

(The `c128` and `gen` rows below sit at the *small* end of this table and fail for the opposite
reason — the same cut applied too aggressively; that is §1.6c.  Everything from `r3 f = 1` down is
the mechanism of this section.)

Sorting the 48 (shape, frequency) pairs by `|k| r_d`, `r_d = sqrt(s1^2+s2^2+s3^2)` the half-diagonal
of the difference box, makes the whole sweep monotone in a single parameter.  `s_max/lambda` is the
longest edge in wavelengths at that frequency.

    shape f        |k|r_d  s_max/lam    eMx      pEn      eRe      eIm    min bnd/act
    c128  0.37     0.0315   0.00289   3.8e-15  3.0e-14  3.0e-14  1.5e-15    30
    c128  1        0.085    0.00781   4.1e-13  4.7e-13  4.7e-13  4.7e-13     0.92
    c128  1+0.1i   0.0854   0.00785   3.8e-13  4.8e-13  8.7e-13  1.4e-12     0.89
    slX   0.37     0.103    0.0116    1.1e-15  4.3e-15  4.3e-15  6.3e-16     4.3
    slY   0.37     0.103    0.0116    1.2e-15  4.3e-15  4.3e-15  4.5e-16     4.3
    slZ   0.37     0.103    0.0116    1.9e-15  4.1e-15  4.1e-15  4.5e-16     6.8
    gen   0.37     0.166    0.0231    1.5e-14  2.1e-14  7.8e-14  2.0e-14     0.37
    c128  2+0.2i   0.171    0.0157    4.8e-15  4.8e-15  4.9e-15  4.7e-12     0.26
    r1    0.37     0.192    0.0301    8.9e-14  1.3e-13  4.7e-13  1.2e-13     0.21
    slX   1        0.278    0.0312    1.6e-15  3.9e-15  3.9e-15  5.4e-16    16
    slY   1        0.278    0.0312    1.5e-15  3.9e-15  3.9e-15  5.4e-16    16
    slZ   1        0.278    0.0312    1.3e-15  3.4e-15  3.4e-15  5.4e-16    16
    slX   1+0.1i   0.279    0.0314    1.2e-15  3.3e-15  3.3e-15  8.5e-14    23
    slY   1+0.1i   0.279    0.0314    1.3e-15  3.3e-15  3.3e-15  3.8e-15    23
    slZ   1+0.1i   0.279    0.0314    1.2e-15  2.9e-15  3.0e-15  4.8e-15    24
    pl8   0.37     0.411    0.0463    6.4e-15  1.1e-14  1.2e-14  1.3e-14     1.9
    gen   1        0.450    0.0625    4.3e-14  1.3e-13  7.5e-13  3.2e-13     0.74
    gen   1+0.1i   0.452    0.0628    4.4e-14  1.4e-13  9.8e-13  4.8e-13     0.71
    r1    1        0.518    0.0814    1.5e-15  1.5e-15  1.5e-15  2.2e-15     2.6
    r1    1+0.1i   0.520    0.0818    1.0e-15  1.7e-15  2.0e-15  2.7e-14     2.1
    slX   2+0.2i   0.559    0.0628    1.3e-15  2.9e-15  2.9e-15  2.4e-14    78
    slY   2+0.2i   0.559    0.0628    1.3e-15  2.9e-15  2.9e-15  2.4e-14    66
    slZ   2+0.2i   0.559    0.0628    8.0e-16  2.8e-15  2.7e-15  2.2e-14    18
    ex3   0.37     0.581    0.0925    2.3e-15  3.2e-14  3.4e-13  2.1e-14     0.73
    nd4   0.37     0.583    0.0925    2.5e-15  1.0e-14  2.4e-13  1.4e-14     0.76
    r2    0.37     0.880    0.140     1.4e-15  2.3e-14  2.4e-14  1.8e-12    29
    gen   2+0.2i   0.904    0.126     8.0e-16  1.5e-15  1.1e-14  1.2e-15     6.5
    r1    2+0.2i   1.04     0.164     9.7e-15  7.9e-14  5.8e-14  1.1e-13     4.4
    r4    0.37     1.07     0.170     1.0e-15  1.1e-14  2.9e-14  9.0e-15    22
    pl8   1        1.11     0.125     1.9e-15  5.7e-15  1.6e-14  4.8e-15     7.5
    pl8   1+0.1i   1.12     0.126     2.1e-15  6.9e-15  1.1e-14  2.7e-14     7.2
    ex3   1        1.57     0.250     1.1e-15  2.4e-15  4.6e-14  8.8e-14    32
    nd4   1        1.58     0.250     2.4e-15  4.9e-15  2.4e-14  8.6e-14    12
    ex3   1+0.1i   1.58     0.251     3.6e-15  4.6e-15  9.2e-15  5.7e-14    19
    nd4   1+0.1i   1.58     0.251     2.1e-15  3.2e-15  2.4e-14  7.3e-15    19
    r3    0.37     1.65     0.262     3.6e-15  1.2e-14  1.2e-14  6.1e-15    62
    pl8   2+0.2i   2.23     0.251     1.0e-15  1.5e-15  2.1e-14  1.3e-15     2.3
    r2    1        2.38     0.379     4.6e-15  8.2e-14  2.4e-13  4.5e-14     8.0
    r2    1+0.1i   2.39     0.380     4.6e-15  8.7e-14  4.9e-13  6.1e-14     6.6
    r4    1        2.89     0.459     4.3e-15  9.3e-14  6.9e-14  1.2e-13     2.8
    r4    1+0.1i   2.90     0.462     4.0e-15  9.0e-14  4.4e-14  2.3e-13     2.8
    ex3   2+0.2i   3.16     0.503     1.2e-15  9.3e-15  7.9e-15  2.4e-14    36
    nd4   2+0.2i   3.17     0.503     8.9e-16  1.5e-14  3.6e-13  7.3e-14    19
    ---- above this line every per-entry error is <= 1.4e-13 ------------------------------
    r3    1        4.46     0.709     4.8e-12  1.7e-10  4.4e-10  2.5e-10     4.4e-3
    r3    1+0.1i   4.48     0.713     5.0e-12  1.8e-10  5.1e-11  3.4e-09     4.4e-3
    r2    2+0.2i   4.78     0.761     3.4e-11  1.3e-09  6.0e-09  1.3e-10     3.7e-3
    r4    2+0.2i   5.80     0.923     1.6e-08  6.4e-07  8.2e-07  4.3e-07     2.4e-5
    r3    2+0.2i   8.96     1.425     1.4e-04  9.6e-03  8.9e-03  3.0e-01     1.4e-9

The break is at `|k| r_d ~ 3.2`, it is sharp, and the growth above it is a clean power law:
`pEn = 1.7e-10, 1.3e-9, 6.4e-7, 9.6e-3` at `|k| r_d = 4.46, 4.78, 5.80, 8.96`, i.e.
`pEn ~ (|k| r_d)^25.6` over five and a half decades.  **That exponent is `2 nMax + 2 = 26`**: the
leading term of the library's own `jtail(l, N, z) ~ z^{l+2N+2}` remainder for the `k`-power series of
`j_l(k|delta|)` truncated at `N = nMax = 12`, which is exactly the truncation baked into the geometry
table (`tabWhl`/`tabOct` fill `Ad[:, 1:nMax+1, :]` and nothing beyond).

Three independent checks confirm the mechanism, not just the correlation.

1. **The bound the library itself carries knows the answer and is never consulted.**
   `nCutVec` (farfield.jl:544) computes the smallest `N` meeting the `jtail` budget per `l`, but it
   initialises `v = fill(nMax, lTop+1)` and lowers an entry only when the budget is met, so
   saturation at `nMax` is indistinguishable from certification (defect D2).  Evaluating the same
   `jtail` criterion with the cap raised to 60 (`work/audit2/s16_ncut.jl`, `out/s16_ncut.txt`) gives
   the `N` actually needed for the low-`l` shells:

        shape f       |k|r_d   N needed   nMax used   measured pEn
        c4    1        2.72       10         12        (not in the sweep; c4 rows of s7 are 1e-15)
        r4    1        2.89       11         12        9.3e-14
        nd4   2+0.2i   3.17       12         12        1.5e-14
        r3    1        4.46       14         12        1.7e-10
        r2    2+0.2i   4.78       15         12        1.3e-09
        c2    1        5.44       14         12        (lambda/2 cube, no table built; see 1.7)
        r4    2+0.2i   5.80       16         12        6.4e-07
        r3    2+0.2i   8.96       21         12        9.6e-03

   Every (shape, frequency) whose `N needed` is `<= 12` meets 1e-13; every one whose `N needed`
   exceeds 12 fails, and the shortfall `N_need - 12` orders the failures correctly.

2. **Route (ii) is immune by exactly the predicted factor.**  An octant of `D` has half-diagonal
   `r_d/2`, so its `jtail` is smaller by `2^-(2 nMax + 2) = 2^-26 = 1.5e-8`.  At the same shape and
   frequency:

        shape f        route (ii) offset          pEn        route (i) offset           pEn      ratio
        r4    2+0.2i   (2,0,0)   L_j <= 40      2.31e-14    (16,16,16)  L = 26        6.43e-07   2.8e7
        r3    2+0.2i   (2,0,0)   L_j <= 44      6.43e-10    (16,16,16)  L = 32        9.59e-03   1.5e7
        r2    2+0.2i   (0,0,2)   L_j <= 39      3.55e-15    (3,3,3)     L = 40        1.64e-10   4.6e4

   against a predicted `6.7e7`.  Route (iii) (the BigFloat k-series) is immune outright: its 88 rows
   are `8.3e-18 .. 3.6e-15` max-norm and `<= 2.1e-14` per entry at every shape and frequency,
   `bnd/act 4.99 .. 1.24e4`, `N` from 9 to 48.

3. **It is not rounding.**  Re-running the four worst cases with `Complex{BigFloat}` input at 192
   bits (`work/audit2/s17_diag.jl`, `out/s17_diag.txt`) reproduces the same error to three digits;
   the Float64 and BigFloat answers differ by 1e-16 of the largest entry.

**Defect D9 (new, and the most serious in this report): the geometry table truncates the `k`-power
series of `j_l` at a compile-time constant `nMax = 12`; `farSetup` computes a per-`l` cut `nCut` that
can only lower it, never raise it, and its own `jtail` bound is evaluated at a radius (`2 min(s)`)
where it is vacuous, so the cut saturates silently.  The library is therefore correct only for
`|k| r_d <~ 3.2` and, above that, returns an answer whose error grows as `(|k| r_d)^26` while the
reported bound — which covers the `l`-truncation only — falls.  Worst observed: 9.6e-3 per entry with
a reported bound 7.2e8 times smaller.**

`farfield_v2.jl` **does not fix this.**  `prep.md` fixes the *vacuity* of `nCutVec` (it re-evaluates
at the true radius with `budget/256` and reports "nCut max 6/9/11/6, nMax = 12 certified") — that is
a fix for shapes at `|k| r_d <= 3`, and it makes the cut meaningful, but `NMAX` is still 12
(`farfield_v2.jl:266`) and `nCutVec` still cannot exceed it, so on `r2`, `r3`, `r4` at
`f = 2+0.2i` v2 would certify a cut it cannot honour, or (with the tighter budget) select
`N = nMax` and again report success.  A correct fix needs `nMax` chosen from `|k| r_d` at table build
time, and an error when the required `N` exceeds the table's.

### 1.6 The one failure that is not D9: `c128` at `lambda/128`

`c128` is a cube of edge `1/128`, `|k| r_d = 0.085` at `f = 1` — the n-series there is converged at
`N = 4` (s16: `nCut in (2,4)`, not saturated).  Yet:

    c128, f = 1, route (i)          eMx       diag [1,1]        offdiag [1,2]
    (3,3,3)    L = 26              2.0e-14   rel 4.63e-13      rel 2.95e-16
    (4,0,0)    L = 34              6.0e-15   rel 4.41e-15, Im 4.64e-13
    (16,16,16) L = 10              4.1e-13   rel 4.65e-13      rel 2.50e-15
    (2,0,0)    route (ii)          1.9e-15   rel 8.15e-16      -

    the same four rows in Complex{BigFloat}(192) input (s17_diag.jl):
    (3,3,3) 1.99e-14   (4,0,0) 5.91e-15   (16,16,16) 4.06e-13   (2,0,0) 1.69e-16

**The error is 4.65e-13 relative, identical at `L = 10`, 26 and 34, confined to the three diagonal
entries (the off-diagonals are at 2.5e-15), unchanged by BigFloat arithmetic, and absent from route
(ii).**  Independence of `L` rules out the `l`-truncation; independence of precision rules out
Float64 rounding; the off-diagonals being clean and route (ii) being clean localise it to the
**diagonal columns (`Ad`, `Bd`) of the whole-box geometry table for this shape**.  `c128` is the
shape with the largest whole-box contraction cancellation of the twelve (`cnc_whl = 7.31e11`) and
`momAcc` (farfield.jl:126) contains a hard-coded exact-zero threshold

        abs(acc) < abs_ * S(2)^-256 && (acc = zero(S))

which is a fixed `2^-256` relative cut applied inside a 1024-bit contraction, i.e. it discards
entries that still carry 770 correct bits, and — because `cmx` is updated only when `acc != 0` — a
discarded entry never appears in the reported `cnc`.  `prep.md` lists "exact-zero threshold must
track precision" among its fixes, so `farfield_v2.jl` may remove this one; **I could not confirm
that it fixes `c128`, because v2 was not in scope for this audit's measurements.**  What is measured
here is that `farTensor` on a `lambda/128` cube — inside Gila's stated range — is wrong by
**4.8e-13 per entry and 4.7e-12 on the imaginary part**, and that `unify.md` never tested that shape
(its four shapes were `1/32`, `1/8`, `1/4` and the slender `1/32,1/32,1/512`).

### 4.2 The measurement (`work/audit2/s4_thr.jl`, `s4b_cmp.jl`, `out/s4_thr.txt`)

`farBlock!` on a `12^3` block at `s = (1/32)^3`, `f = 1` (1720 offsets with `max(i) >= 3`, all route
(i)/(ii)) and on the slender `6x6x12` block at `(1/32, 1/32, 1/512)`, `f = 1` (where 40 offsets take
route (iii)), each written to a raw binary and compared byte for byte:

    threads   maxthreadid   repeat call bitwise identical (c32 12^3 / slender 6x6x12)
     1          1            true / true
     4          8            true / true
    12         18            true / true

    c32 12^3     1 vs 4 threads: bitwise identical (max|d| 0.0)   1 vs 12: bitwise identical (0.0)
    slZ 6x6x12   1 vs 4 threads: bitwise identical (0.0)          1 vs 12: bitwise identical (0.0)

**`farBlock!` is bitwise deterministic in the number of threads and across repeated calls, on both a
route-(i)/(ii) block and a block that uses route (iii).**  That is what the code structure predicts:
the two `@threads :static` loops write only to disjoint indices, each thread takes its own `FarWs`
from a vector sized by `Threads.maxthreadid()` (18 at `nthreads = 12`, so the `threadid()` index is
in range), and the route-(iii) offsets are filled by a **serial** loop between them.

### 4.3 The race in `farTensor` is real but did not manifest (`work/audit2/s4c_race.jl`)

The documented way to use the library on a handful of offsets is `farTensor(D, s, f; fs = fs)` with a
shared `FrqSet`.  That path calls `ksrCached!` directly, which inserts into `fs.ksr` (a plain `Dict`)
and, through `tnsKsr`, into the global `MOMC` (another plain `Dict`), and may run
`Base.include(Main, MOMJL[])` — none of it locked (defect D6).  Test: six route-(iii) offsets
`(0,0,14..19)` of the slender shape at `f = 1`, `KSRDIR[]` pointed at an empty directory so the
BigFloat path really runs, driven from `Threads.@threads :static` with 6 threads and one shared
`FrqSet`, five independent trials:

    exceptions 0/6 in every trial; every tensor bitwise equal to the serial value from the disk
    cache; 6 lines written to the one cache file each time (no duplicated work, no truncated line);
    8.7 s wall for the six against ~10 s each serially.

**So the unsynchronised `Dict` insertions did not corrupt anything in 30 concurrent insertions.**
That is not a proof of safety — `Base.Dict` rehashing under concurrent `setindex!` is undefined
behaviour in Julia, and the `getglobal(Main, :facePair)` after a runtime `Base.include` is
world-age fragile (defect D7, its warning is printed on the first route-(iii) call in every one of
these runs) — but the honest statement is: **the hazard is in the source, and I could not make it
fire.**  `farBlock!` avoids it by construction; `farTensor` does not, and nothing in the docstring
says so.  `farfield_v2.jl` does not change this: `MOMC` is still an unguarded global
(`farfield_v2.jl:1143`) and the `getglobal` reads are still outside `invokelatest`
(`farfield_v2.jl:1150`).

### 7.1 Sixty random cases (`work/audit2/s7_bnd.jl`, `out/s7_bnd.txt`)

60 cases drawn with `MersenneTwister(778899)`: shape uniform over the twelve of §1.2 plus `c32` and
`c4` (13 of the 14 were drawn), `D` uniform in `[-20,20]^3` with `max|D| >= 2`, `f` uniform over
`{1, 1+0.1i, 0.37, 2+0.2i, 1+1i}`, rejecting cases whose reference would need more than 6e6 nodes.
All 60 landed on route (i), `L` from 10 to 36, `rho` from 0.050 to 0.333.  Reference: `volTensor` at
order 32 (3.0-3.9 s each).  `bound` is `bndCum(ls, t, k|R|)` evaluated at the `L` `farRoute` actually
selected — the library's own predicted **absolute** truncation error; `actual = max_ab |G - Gr|`.

    bound/actual over all 60:      min 9.96e-10   median 7.48   max 419      violations (<1): 19
    over the 45 cases excluding
    the three broken shapes:       min 0.0887     median 8.82   max 419      violations (<1): 10

The 19 violations split into three kinds, and only the first two are the bound's fault:

**(a) 9 cases where the answer is wrong, not the bound loose** — `r3` (5 cases), `r4` (1), `c128`
(1), plus `r3` twice more at `f = 1+1i`.  These are defect D9 (§1.5) and the `c128` table defect
(§1.6); the truncation the bound describes is not the truncation that dominates.

    shape D              f           L    bound      actual     bound/actual   eMx
    r3    (20,-2,15)     2+0.2i      32   7.03e-28   7.06e-19   9.96e-10       1.38e-04
    r3    (12,9,-10)     1+1i        28   5.62e-43   1.36e-37   4.14e-06       1.22e-09
    r3    (4,-5,2)       1+1i        36   2.74e-26   1.24e-21   2.21e-05       1.04e-09
    r4    (13,-6,2)      2+0.2i      26   6.22e-22   4.14e-18   1.50e-04       1.53e-08
    r3    (18,-14,-10)   1           24   7.96e-21   3.28e-18   2.43e-03       4.84e-12
    r3    (-14,13,-5)    1           24   4.04e-20   4.21e-18   9.60e-03       4.84e-12
    r3    (-13,-1,-6)    1+0.1i      24   2.36e-22   1.56e-20   1.51e-02       5.00e-12
    c128  (-18,7,-13)    1           12   9.86e-20   2.55e-18   3.86e-02       2.45e-13
    r3    (-4,6,-17)     1+0.1i      32   3.83e-19   2.79e-18   1.37e-01       4.62e-12

**(b) 4 cases where the delivered answer is right to 1e-13 but the bound is still exceeded by 3 to
11 times, because the error is rounding and the bound covers truncation only.**  Expressed in units
of `eps * max|G_ref|`:

    r1   (-6,12,-3)    f=0.37   bound/actual 0.089   error 8.8e-14 = 396 eps
    gen  (-4,6,-15)    f=1      bound/actual 0.328   error 5.5e-14 = 246 eps
    gen  (-16,17,-17)  f=0.37   bound/actual 0.304   error 1.5e-14 =  66 eps
    gen  (-17,20,7)    f=0.37   bound/actual 0.132   error 1.4e-14 =  63 eps

**(c) 6 cases at the Float64 floor** (`slX` x2, `ex3`, `pl8` x2, `r2`), error 6 to 28 `eps`,
`bound/actual` 0.26 to 0.88 — the bound is simply below the rounding floor of the answer, as §6
already found at `(32,0,0)`.

**Answer to task 7.  The bound is violated in 19 of 60 random cases; `min bound/actual = 9.96e-10`,
`max = 419`.  Adding the rounding floor `eps * max|G|` repairs kinds (c) and (b) only up to about
30 eps, so even the repaired statement `bound + 30 eps max|G|` fails on the four kind-(b) cases and
on all nine of kind (a).  There is no offset in this sample where the library's reported bound can
be used as a certificate without adding a term it does not model — the k-series truncation of `j_l`
at `nMax = 12` (D9) — and a rounding allowance of several hundred `eps`.**  Where neither applies
(the 41 non-violating rows) the bound is loose by 1.21x to 419x, median 7.5x, which is consistent
with `bound2.md`'s "median 49 for the old theorem" only because `s7` measures the bound at the `L`
the library *chose*, not at the `L` the bound would need.

## 8. Anti-Hermitian positivity (task 8) — `work/audit2/s8_pos.jl`, `out/s8_pos.txt`

Two blocks, each built twice.  **Gila build**: `GlaVol(cel, s, (0,0,0))`, `CPUKerOpt(f, 48, false,
CPU())`, `egoFunInn!` on every Toeplitz index, `egoFunSng!` (`wekTrp`, `intOrd = 48`) on the eight
indices with all components `<= 2`, then `egoToe[a,a,1,1,1] -= 1/f^2`.  **farfield build**: that same
array with every index of max-norm separation `>= 2` overwritten by `farBlock!`; the eight contact
blocks are then identical by construction (verified: `contact blocks identical: 8/8` in all four
cases).  The dense `3N x 3N` matrix uses `egoToeCrc!`'s rule
`M[(c,a),(c',b)] = egoToe[a,b,|d|+1] sigma_a sigma_b`, `sigma_m = sign(d_m)`; `A = (M - M')/(2i)`.

    block            f        build      lam_min            lam_max      eps*lam_max   negatives
    sl 6x6x12        1        Gila       -6.553240722e-14   0.010116581  2.25e-18      596 / 1296
    (1/32,1/32,1/512) 1       farfield   -1.921332788e-14   0.010116581  2.25e-18      576 / 1296
    sl 6x6x12        1+0.1i   Gila       -9.9393219287e-03  0.195379202  4.34e-17      108 / 1296
                              farfield   -1.4650014925e-02  0.195388599  4.34e-17       76 / 1296
    c4 6^3           1        Gila       -6.398730928e-15   2.389663556  5.31e-16       58 /  648
    (1/4,1/4,1/4)    1        farfield   -5.009706905e-15   2.389663556  5.31e-16       49 /  648
    c4 6^3           1+0.1i   Gila       +7.124128212914e-3 1.745212265  3.88e-16        0 /  648
                              farfield   +7.124128212912e-3 1.745212265  3.88e-16        0 /  648

    max |Gila - farBlock!| / max entry over the separated offsets, and where:
      sl  f = 1        6.16e-01 at (0,0,2)      <- Gila's fixed rule, see below
      sl  f = 1+0.1i   6.16e-01 at (0,0,2)
      c4  f = 1        8.74e-14 at (0,0,2)
      c4  f = 1+0.1i   1.01e-13 at (0,0,2)

    M_ab(R) = M_ba(-R), measured as max|M - M^T| / max|M|:
      sl  f = 1       Gila 3.07e-15   farfield 1.17e-15
      sl  f = 1+0.1i  Gila 1.41e-15   farfield 1.41e-15
      c4  f = 1       Gila 1.47e-15   farfield 1.00e-15
      c4  f = 1+0.1i  Gila 8.27e-16   farfield 8.27e-16

Reading, one line per question asked.

- **`lambda/4` cube, real `f`.**  Both builds are at the Float64 floor: `lam_min` is `-12.0 eps
  lam_max` (Gila) and `-9.4 eps lam_max` (farfield); `farBlock!` improves it by 1.28x and removes 9
  of the 58 negative eigenvalues.  The two builds' separated entries agree to `8.7e-14` — Gila's own
  fixed-rule error at `lambda/4`, consistent with `reference.md`'s 1.6e-15..2.8e-13 there.  **At
  `lambda/4` the far field is no longer what limits positivity**; the contact and touching-shell
  integrals, identical in both builds, are.
- **`lambda/4` cube, complex `f`.**  `Im M` is positive definite (`lam_min = +7.12e-3`, no negative
  eigenvalues) and the two builds agree to 13 digits (`2.9e-13` relative in `lam_min`).
- **Slender `6x6x12`, real `f`.**  `farBlock!` moves `lam_min` from `-6.55e-14` to `-1.92e-14`, a
  factor **3.4** closer to zero, and removes 20 of 596 negative eigenvalues; both remain far above
  the floor (`-8540 eps lam_max` and `-2500 eps lam_max`) because the residual sits in the contact
  blocks, which are Gila's in both builds.  This is a larger improvement than `unify.md` §7c
  measured on the `6^3` blocks (1.8x on `c32`, 2.7x on its slender block); the 12-cell long axis
  gives the far field more weight.
- **Slender `6x6x12`, complex `f`: `Im M` is strongly indefinite in both builds and `farBlock!`
  makes it worse, not better** — `lam_min` goes from `-9.94e-3` (Gila) to `-1.465e-2` (farfield)
  against `lam_max = 0.195`, i.e. from 5.1% to 7.5% of the spectral radius, 12 orders above any
  rounding floor.  The negative count falls (108 -> 76) while the worst negative grows by 1.47x.
  Since §1 verifies `farTensor` on this shape at `<= 4.1e-15` per entry and §3 verifies that Gila's
  own rule is wrong by 62% at `(0,0,2)` on it, **the more accurate operator is the more indefinite
  one**: the indefiniteness is a property of the 1:1:16 discretization plus Gila's contact integrals
  for such a cell, not of the far-field evaluation, and correcting the far field exposes it rather
  than causing it.  `unify.md` §7c saw the same sign (`-6.8e-3 -> -1.27e-2` on `6^3`) and drew the
  same conclusion; this run confirms it on a second block shape and quantifies it: **the anti-
  Hermitian part of Gila's slender-cell operator at `f = 1+0.1i` is not usable as a positive form,
  and no far-field accuracy fixes that.**
- **`M_ab(R) = M_ba(-R)`** holds to `8.3e-16 .. 3.1e-15` in every case, and the farfield build is
  never worse than the Gila build.  The test is weak by construction on the separated entries —
  `farBlock!`'s 3x3 output is exactly symmetric (§3 measured `max|G - G^T| = 0.0`) and
  `egoToeCrc!`'s sign rule then makes `M` symmetric identically — so what the residual measures is
  Gila's contact blocks, which are the same in both builds except for rounding.

### 1.7 Cells above `lambda/4`, and what happens at `s = 1/2`

Seven of the 48 (shape, frequency) pairs have a longest edge above `lambda/4`.  Sorted by longest
edge, with the measured worst per-entry error and the controlling parameter `|k| r_d`:

    shape f        s_max/lambda   |k| r_d   pEn        verdict
    nd4   2+0.2i   0.503          3.17      1.5e-14    passes
    ex3   2+0.2i   0.503          3.16      9.3e-15    passes
    r3    1        0.709          4.46      1.7e-10    fails by 1.7e3
    r2    2+0.2i   0.761          4.78      1.3e-09    fails by 1.3e4
    r4    2+0.2i   0.923          5.80      6.4e-07    fails by 6.4e6
    r3    2+0.2i   1.425          8.96      9.6e-03    fails by 9.6e10
    r3    1+0.1i   0.713          4.48      1.8e-10    fails by 1.8e3

**The longest edge is not the parameter; `|k| r_d` is.**  `nd4` at `0.503 lambda` and `ex3` at
`0.503 lambda` pass because they are a needle and a fibre — two of their edges are 1/64 and 1/4096 of
a wavelength, so `r_d` is barely larger than the long edge and `|k| r_d` stays at 3.17.  `r4` at
`0.459 lambda` and `f = 1` (`|k| r_d = 2.89`) passes; the *same shape* at `f = 2+0.2i`
(`|k| r_d = 5.80`) is wrong by 6.4e-7.  The consequence for Gila's stated range is clean:

> for any cell whose **longest edge** is at most `lambda/4`, `r_d <= sqrt(3) s_max <= 0.433 lambda`
> and `|k| r_d <= 2.72`, which is below the measured break at 3.2 — **D9 cannot bite inside
> `lambda/128 .. lambda/4`**, and the margin in `|k| r_d` is 3.2/2.72 = 1.18, i.e. a factor
> `1.18^26 = 1.3e2` in the n-truncation error.

**At `s = 1/2` (a cubic cell of half a wavelength, `|k| r_d = 5.44` at `f = 1`) — measured.**  The
geometry table for `(1/2, 1/2, 1/2)` was built for this question (288 s, 19.2 GB peak RSS) and the
tensor compared against the `pairKer` face-pair reference at 220 bits
(`work/audit2/s26_half2.jl`, `out/s26_half2.txt`):

    c2 = (1/2,1/2,1/2), f = 1, |k| r_d = 5.44, nCut min/max 7/12, saturated at nMax for 14 of 57 l
      offset       route  L/N  reference self-conv   eMx       pEn       eRe       eIm       bound     bnd/act
      (2,0,0)      (iii)  54   3.82e-48              1.95e-17  5.61e-17  2.46e-17  5.65e-17  6.10e-17  19.1
      (3,3,3)      (i)    34   9.24e-64              7.22e-13  7.22e-13  7.29e-13  7.10e-13  1.76e-14  0.565
      (4,0,0)      (i)    42   3.53e-57              5.08e-13  3.01e-12  8.97e-11  1.86e-14  1.74e-14  0.427
      (7,-3,2)     (i)    28   7.20e-65              5.32e-13  2.20e-12  2.57e-12  1.38e-12  7.07e-15  0.341
      (16,16,16)   (i)    22   1.32e-64              7.20e-13  7.20e-13  7.18e-13  7.22e-13  1.94e-15  0.333
      each of the four route-(i) rows with nCut forced to nMax = 12: eMx unchanged to three digits

Two independent references were used and agree: the graded 220-bit volume rule (all five rows,
`work/audit2/s18_half.jl`, `out/s18_half.txt`) and the `pairKer` face-pair rule (the four route-(i)
rows, `s26_half2.jl`, `out/s26_half2.txt`, self-convergence 1.9e-65 to 4.7e-65); every entry above
agrees between them **to all three printed digits**.  `pairKer` did not return within 25 min on
`(2,0,0)`, which is why that row is the volume rule alone.

**The `(2,0,0)` row is the reversal made explicit.**  At `s = 1/2` the two-cell offset does not
converge on route (i) or (ii) and falls through to the BigFloat k-series, which returns
`5.6e-17` per entry — four orders *better* than the target — while the far offsets on route (i) are
wrong by `7e-13` to `3e-12`.  On a `lambda/2` cell the near field is exact and the far field is not.

**A `lambda/2` cubic cell is wrong by `5.1e-13 .. 7.2e-13` in max-norm and `7.2e-13 .. 3.0e-12` per
entry at every separation tested (3, 4, 7.9 and 27.7 cells along four different directions) — 7 to
30 times the target, in the real and the imaginary part alike, with one real part at `9.0e-11` —
and forcing every `nCut` up to the table's own `nMax = 12` changes nothing, because 12 is the
ceiling.  The library's reported bounds, `1.94e-15` to `1.76e-14`, are 1.8 to 3.0 times smaller than
the errors it delivers.**  This is D9 measured directly rather than extrapolated, on a cube, at `f = 1`.
It also corrects the two estimates below: the `est`-relative prediction (1.2e-12) is right to within
1.7x, and the extrapolation from the extreme-aspect `r3`/`r4` rows (3e-8 to 1e-7) is far too
pessimistic for a cube, because `pEn` on those shapes divides by entries that are themselves tiny.

What the library says about itself at this shape, evaluated table-free (`work/audit2/s16_ncut.jl`,
`s20_ntail.jl`, `out/s16_ncut.txt`, `out/s20_ntail.txt`):

    c2 = (1/2,1/2,1/2), f = 1, |k| r_d = 5.44
      offset      route  L    est        n-tail at nMax=12   n-tail at N=20   l-truncation bound
                                         (relative to est)   (relative)       it actually reports
      (2,0,0)     (ii)/(iii): route (i) does not converge (rho = 0.866)
      (2,2,0)     (ii)/(iii)
      (3,3,3)     (i)   34   0.181      1.21e-12            6.12e-27         9.74e-14
      (7,-3,2)    (i)   28   0.112      1.18e-12            6.04e-27         6.29e-14
      (16,16,16)  (i)   22   0.0293     1.15e-12            5.94e-27         6.62e-14
      nCut saturated at nMax = 12 for 14 of the 57 l values

    and at f = 2+0.2i (an entire wavelength per cell, |k| r_d = 10.9):
      (3,3,3)     (i)   46   0.00632    5.99e-04            1.10e-13         1.33e-14

So at `s = 1/2` the library's own `jtail` bound puts the n-truncation at **1.2e-12 of `est`, about
20 times the `l`-truncation error it reports, and (since `est/max|T|` is 1.1 to 6.3 for cubes) 2e-13
to 1.2e-12 of the largest tensor entry — 2 to 12 times the 1e-13 target, with no warning.**
Interpolating the *measured* rows at the same `|k| r_d` (`r3` at 4.46 giving 1.7e-10 per entry, `r4`
at 5.80 giving 6.4e-7) with the `(|k| r_d)^26` law brackets a `lambda/2` cube at **3e-8 to 1e-7 per
entry**, i.e. five to six orders above the target; the two estimates differ because `pEn` divides by
an individual entry, not by `est`.  Either way the answer at `s = 1/2` is: **route (i) is wrong,
route (ii) is not** (the octant halves `r_d`, giving `|k| r_d/2 = 2.72`, exactly the safe value), and
because at `s = 1/2` the 2-cell offsets are the ones that take route (ii), **the near offsets are
right and the far ones are wrong** — the reverse of the usual expectation, and a good reason to
force route (ii) (or, as the measured `(2,0,0)` row shows, route (iii)) for large cells until `nMax`
is made shape-dependent.  At `s = 1` the failure is 6e-4 of `est` and even `N = 20` only just
reaches 1.1e-13.

### 1.6b The `c128` defect, closed out: it is `farfield.jl`, not the reference and not my table

Three checks, each of which could have exonerated the library.

**(a) A third, fully independent reference.**  `refPairs` (`pairKer` of `notes/gen/ref/mom.jl` on the
36 face pairs at 220 bits, ordN 44) summed by `srfSum!`'s signs, against my graded volume rule
(`work/audit2/s21_pk.jl`, `out/s21_pk.txt`):

    c128 (16,16,16)   pairKer(44) vs pairKer(32)  7.3e-62    pairKer vs volTensor  1.81e-61
    c128 (3,3,3)      pairKer(44) vs pairKer(32)  9.7e-63    pairKer vs volTensor  9.22e-63
    c32  (16,16,16)   pairKer(44) vs pairKer(32)  9.1e-63    pairKer vs volTensor  5.33e-63

    farfield route (i)  vs pairKer:  c128 (16,16,16) 4.06e-13 (diag 4.65e-13);
                                     c128 (3,3,3)    2.01e-14 (diag 4.63e-13);
                                     c32  (16,16,16) 6.38e-16 (diag 6.38e-16)
    farfield route (ii) vs pairKer:  c128 (16,16,16) 3.51e-14 (diag 4.01e-14);
                                     c128 (3,3,3)    1.72e-15; c32 (16,16,16) 2.14e-16

Two references built by different mathematics (face pairs of `g` vs the volume form of
`(d_a d_b + dab k^2) g`) agree to **1.8e-61**, and both disagree with `farfield.jl`'s whole-box route
by 4.06e-13.

**(b) `L` sweep** (`work/audit2/s19_c128.jl`, `out/s19_c128.txt`): at `c128 (16,16,16)` the
whole-box error is `4.06e-13` at `L = 10, 20, 30, 40` and `56` — five values, identical to three
digits — and the octant error is `3.51e-14` at all five.  At `c128 (4,0,0)` both converge normally
(`6.32e-7 -> 7.29e-12 -> 5.97e-15` as `L` goes 10, 20, 30) and stop at `5.97e-15` (whole) /
`6.39e-16` (octant).  `c32 (16,16,16)` and `c32 (4,0,0)` are at `4e-16 .. 6.4e-16`.  **The residual
is not the `l`-truncation.**

**(c) My cached table is not corrupt.**  `farShape((1/128)^3)` rebuilt from scratch into an empty
directory (`work/audit2/s22_reb.jl`, `out/s22_reb.txt`) takes **298 s and 18.9 GB peak RSS**, reports
the same `cnc_whl = 7.31e11`, `cnc_oct = 1.96e11`, and reproduces the cached 321391-line file with
**0 differing lines**.  The rebuilt table gives the same 4.06e-13 and 2.01e-14.  (The `build_s 1.51`
in §1.2 was therefore a disk load of an already-present file, not a 1.5 s build; the correct
one-time table cost for every shape, `c128` included, is the 298-1770 s of §9.)

**So: `farTensor` at `s = (1/128)^3`, `f = 1`, on a shape inside Gila's stated range, returns a
diagonal wrong by `4.65e-13` relative on route (i) and `4.0e-14` on route (ii), reproducibly, in
Float64 and in BigFloat, at every `L`.  The cause is in the geometry table or in the small-`kR`
special-function evaluation, and §1.6c narrows it.**

## 9. Cost honesty (task 9) — `work/audit2/s9_cost.jl`, `out/s9_cost.txt`

Single-threaded (`JULIA_NUM_THREADS=1`), `farfield.jl` unmodified, load average recorded on the same
line.  This agent's second Julia process and other agents' processes were running; the machine was
never quiet, and `uptime` load 2.3-4.7 on 12 cores means 3 of 12 cores were busy elsewhere.

### 9.1 Cubic `lambda/32`, the shape the deliverable is fastest on

    block   offsets filled   best of 3   per offset   load
    32^3     32 760          0.0332 s    1010 ns      2.28
    64^3    262 136          0.245  s     936 ns      2.28
    farSetup with the table already on disk: 2.57 s (19.5 MB read and parsed)

Extrapolating `936 ns` to the `128^3` doubled grid (2 097 144 offsets with `max(i) >= 3`) gives
**1.96 s**, consistent with `unify.md`'s 2.05 s.  The claim "a `128^3` self-volume far field in
seconds, single-threaded" is true for a cubic `lambda/32` cell, **provided the shape table already
exists**; see 9.3.

### 9.2 The slender block with the route-(iii) offsets

`(1/32, 1/32, 1/512)`, `64 x 64 x 128` (524 280 offsets with `max(i) >= 3`), `f = 1`:

    routing:  route (i) 524 066   route (ii) 142   route (iii) 72
    COLD  (empty ksr cache, the 72 tensors computed from moments.jl in BigFloat(128))   684 s
    WARM  (the 72 in the FrqSet's in-memory Dict)                                         0.503 s
    DISK  (a fresh FrqSet reading the ksr cache the cold run wrote; setup 0.848 s)        0.507 s
    fraction of the cold time spent in route (iii): 0.999      cold/warm ratio: 1360

**72 offsets out of 524 280 — 0.014 % of the work — take 99.9 % of the time, 684 s against 0.503 s
for the other 524 208.**  That is 9.5 s per route-(iii) offset on this shape, and §2.3 measured 13 to
66 s per offset on the `lambda/8` plate, where `N` reaches 28.  The 0.503 s warm number (960 ns per
offset, the same as the cube) is the honest steady-state cost; the 684 s is the honest cost of the
first build of a new (shape, frequency), and it is **not** amortised over frequencies for free —
`ksrCached!` keys its disk cache on the frequency, so a second frequency repays the `pairMoments`
part only through the in-process `MOMC` (§2.3: 0.67-1.22 s per offset instead of 10-14 s, but only
inside one Julia session).

### 9.3 One-time table cost per shape, and where the per-offset time goes

    farShape at the defaults (L = 56, nMax = 12, contraction at 1024 bits, stored at 192):
      measured build time     298 s (c128, clean rebuild, §1.6b) .. 1770 s (nd4, under load)
                              306-867 s for the other ten shapes of §1.2
      peak RSS                18.9 GB (c128 clean rebuild)   8-10 GB for eight of the twelve
      table on disk           19.5-22.0 MB, 321 391 lines;  reload 1.5-2.6 s

A `32^3` far field at `lambda/32` therefore costs **0.033 s with the table and 298 s without it**, a
factor 9000; the deliverable's headline "seconds, not minutes" holds only from the second run
onwards, and only if the table directory survives.  `farfield_v2.jl` addresses this (256-bit
contraction and lazy `L`: `prep.md` reports 87-200 s per shape and a 202 s cold `32^3` build against
828 s) but does not remove it.

Per-offset breakdown at `lambda/32`, cubic (1000 calls, best of 5):

    offset          L    farRoute   boundL (the part used)   costWhl (the part never read)  tnsWhl!
    (5,1,0)         28   362 ns     26.8 ns                  395 ns                         2300 ns
    (30,20,10)      10   361 ns     25.3 ns                  392 ns                          580 ns
    (127,127,127)    8   360 ns     25.0 ns                  391 ns                          720 ns

**Defect D4 quantified.**  `farRoute` costs 360 ns per offset; the only part of its result any caller
uses is the `(kind, L, Lc)` that `boundL`/`boundLoct` produce in **25 ns**.  The remaining ~335 ns is
dominated by `costWhl`/`costOct`, whose value is discarded at all three call sites (`farfield.jl`
lines 977, 998, 1029 all destructure it as `_`).  Against the measured **936 ns per offset** for a
`64^3` block that is **about 36 % of the entire far-field build time spent computing a number nobody
reads**.  (The standalone `costWhl` timing, 391-395 ns, slightly exceeds `farRoute`'s own 360 ns —
the two microbenchmarks were taken under load and are not additive to better than 20 % — so the
honest range for the waste is 330-390 ns per offset, 35-42 % of the build.)  **`farfield_v2.jl` does
not fix it**: `farRoute` there still calls `costWhl`/`costOct` (`farfield_v2.jl:923, 928`) and its
three call sites (1279, 1300, 1331) still discard the value.

### 1.6c Cause found: `nCut` **over**-truncates as often as it saturates (D2, second face)

`farSetup` calls `nCutVec(L, s, k, frq, k * 2 min(s), tol, nMax)` — one per-`l` cut, chosen against
the budget `tol * est(2 min(s)) / (L+1)` at the *nearest possible* radius, then applied unchanged to
every offset in the block.  The shipped vectors (`work/audit2/s25_ncut2.jl`, `out/s25_ncut2.txt`):

    c128, f = 1:  nCut = 3,4,3,4,3,4,3,4,3,3,3,...,3,2,3,2,3,...,2   (min 2, max 4)
    c32,  f = 1:  nCut = 5,5,5,5,5,5,4,5,4,5,4,...,3,...,2,3,2,3,2   (min 2, max 5)
    gen,  f = 1:  nCut = 5,6,6,6,...,7,...,8,...,9                   (min 5, max 9)

Replacing `nCut` by `fill(nMax, L+1)` — the same table, the same `L`, the same arithmetic, only the
`k`-series of `j_l` summed to all 13 stored terms instead of 3 to 5:

    shape  offset        L    whole box: as shipped -> nCut = nMax    octants: as shipped -> nMax
    c128   (16,16,16)    22   4.06e-13  ->  2.29e-16                  3.51e-14  ->  2.24e-16
    c128   (32,32,32)    22   4.65e-13  ->  3.56e-16                  4.01e-14  ->  1.68e-16
    c128   (3,3,3)       38   2.01e-14  ->  2.95e-16                  1.72e-15  ->  6.73e-16
    gen    (16,16,16)    24   4.25e-14  ->  3.64e-16                  4.94e-16  ->  2.48e-16
    c32    (16,16,16)    24   4.48e-16  ->  5.90e-16                  1.74e-16  ->  1.74e-16

**The `c128` error is `nCut` cutting the `j_l` series at 2 to 4 terms.  It is a 1800x loss on that
shape and it costs nothing to avoid — the terms are already in the table.**  The same mechanism
costs `gen` a factor 117.  `c32` is unaffected because its cut, though also 2 to 5, happens to land
where the next term is already below rounding.

Why it is systematic rather than random: the truncation is an error of the *coefficients* `Q_lm`, so
it is a fixed **relative** error of the tensor at every offset.  A scan over `(n,n,n)` and `(n,0,0)`
at `f = 1` (`work/audit2/s24_kr.jl`, `out/s24_kr.txt`), each offset evaluated at the `L` the library
picks and again at `L + 12`, shows exactly that — the per-entry error is the same number, `4.65e-13`,
at `kR` from 0.255 to 5.44, and adding 12 shells changes nothing:

    c128    (3,3,3)  (4,4,4)  (6,6,6)  (8,8,8) (12^3)  (16^3)  (24^3)  (32^3)  (48^3)  (64^3)
    kR       0.255    0.34     0.51     0.68    1.02    1.36    2.04    2.72    4.08    5.44
    eMx     2.0e-14  3.5e-14  7.7e-14  1.3e-13 2.7e-13 4.1e-13 4.7e-13 4.7e-13 4.7e-13 4.6e-13
    pEn     4.7e-13  4.7e-13  4.7e-13  4.7e-13 4.7e-13 4.7e-13 4.7e-13 4.7e-13 4.7e-13 4.6e-13
    pEn@L+12 identical to three digits in every column

    c128 (n,0,0):  n = 4      8       16      32      64      128
                 pEn 1.2e-14 5.2e-14 2.2e-13 3.6e-13 4.6e-13 9.6e-13     (worst of the whole sweep)
    c32  (n,n,n) and (n,0,0), same scan: eMx <= 2.6e-15, pEn <= 7.2e-14 (the one 7.2e-14 is at
         (128,0,0), L = 8, and falls to 5.2e-15 at L = 20 — see the third mechanism below)

`4.65e-13` is not an accident either: it is `tol * est/max|T|` with `tol = 1e-13` and
`est/max|T| = 4.6` for this shape (§1.4 records `est/mx` 1.6-6.3 for `c128`).  **`nCut` is doing
exactly what it was told: it meets an absolute budget `tol * est`, and `est` over-states `max|T|` by
up to 6.3x, so the delivered per-entry accuracy is `6.3e-13`, not `1e-13`.**

**A third, distinct mechanism: `L` is under-selected at large separations.**  Independent of `nCut`,
`r1` at `(48,48,48)` and `(64,64,64)` and `c32` at `(128,0,0)` improve by 30x when 12 shells are
added:

    shape  offset        L chosen   pEn at L    pEn at L+12
    r1     (48,48,48)      10       1.23e-13     3.81e-15
    r1     (64,64,64)      10       1.19e-13     3.82e-15
    c32    (128,0,0)        8       7.20e-14     5.18e-15

which is the same `est`-over-states-`max|T|` effect acting through `whlThr` instead of `nCutVec`, and
is what `prep.md` reports as "far `rho <= 0.027` needs 2 shells MORE".

**Verdict on D2, restated.**  One bug, two opposite failures: `nCutVec`'s budget is evaluated at
`2 min(s)`, a radius at which the expansion often does not even converge, so the cut it returns is
meaningless — sometimes far too small (`c128`: 4.65e-13, `gen`: 4.25e-14), sometimes saturated at
`nMax` with no certificate (`r2`, `r3`, `r4`: up to 9.6e-3, defect D9).  **`farfield_v2.jl` fixes the
over-truncation half** (`prep.md`: the cut re-evaluated at the true radius with `budget/256`, giving
`nCut` max 6/9/11/6, plus `TOL = 1e-14`; that is a 256x tighter budget on top of a 10x tighter
tolerance, which would put `c128` at ~2e-16) **and does not fix the under-truncation half**, because
`NMAX` is still the compile-time constant 12 and `nCutVec` can still only lower it.

### 8.1 `T_ab(R) = T_ba(-R)` at the tensor level (`work/audit2/s23_sym.jl`, `out/s23_sym.txt`)

The dense-matrix test of §8 is weak because `egoToeCrc!`'s sign rule makes `M` symmetric whenever the
3x3 block is.  The sharp test is the offset reflection itself: for each of the eight sign patterns
`sigma`, `T_ab(sigma R)` must equal `sigma_a sigma_b T_ab(R)`, and one reflected offset is then
checked against **its own** 220-bit reference so that "both wrong the same way" is excluded.

    shape f        D          route   max|T(sg D) - sg_a sg_b T(D)|/max|T|   max|T-T^T|   T(-D1,D2,D3) vs its own reference
    c32   1        (5,1,0)    (i)     0.0                                    0.0          5.67e-16
    c32   1        (3,3,3)    (i)     0.0                                    0.0          3.86e-16
    c32   1        (2,1,1)    (ii)    2.69e-16                               0.0          7.41e-16
    slZ   1+0.1i   (5,1,0)    (i)     0.0                                    0.0          5.39e-16
    slZ   1+0.1i   (2,2,2)    (i)     0.0                                    0.0          7.45e-16
    slZ   1+0.1i   (0,1,20)   (ii)    1.56e-16                               0.0          1.64e-15

Route (i) is exact by construction (the whole-box table keeps even `l` only, and `Y_lm(-u) =
(-1)^l Y_lm(u)`), so those zeros prove nothing; route (ii), which reflects sub-boxes and keeps odd
`l`, gets it right to `1.6-2.7e-16`, and the reflected offsets match independent references to
`3.9e-16 .. 1.6e-15`.

## 10. Scripts and raw outputs of this report

    work/audit2/base.jl        loads notes/farfield/farfield.jl unmodified, points TABDIR/KSRDIR at work/audit2
    work/audit2/shapes.jl      the twelve audit shapes (four random, MersenneTwister(20260907))
    work/audit2/offsets.jl     the six per-shape offsets and the four frequencies
    work/audit2/bnd.jl         predErr: the library's own predicted error at the route/L it used
    work/audit2/myref.jl       unmodified copy of work/reference/ref.jl (refPairs/refSum/volTensor)
    s0_smoke.jl  s3_gila.jl    sign/normalisation pin; Gila-layout comparison            -> out/s3_gila.txt
    s1_tab.jl                  builds and times the twelve shape tables                  -> out/s1_tab.txt
    s2_ref.jl                  builds the 288 220-bit references, with self-convergence  -> out/s2_ref.txt
    s5_gen.jl                  Float32 / BigFloat genericity                             -> out/s5_gen.txt
    s6_shape.jl                the main 288-tensor accuracy sweep                        -> out/s6_shape.txt
    s7_bnd.jl                  60 random (shape, offset, f): bound vs true error         -> out/s7_bnd.txt
    s8_pos.jl                  anti-Hermitian positivity, Gila vs farBlock!              -> out/s8_pos.txt
    s9_cost.jl                 32^3, 64^3, slender 64x64x128 cold/warm, per-offset split -> out/s9_cost.txt
    s10a/s10b                  routing boundaries and both sides of each                 -> out/s10*.txt
    s11_frq.jl                 five frequencies incl. f = -1, 1-0.1i, 10                 -> out/s11_frq.txt
    s4_thr.jl, s4b_cmp.jl      1/4/12-thread bitwise determinism                         -> out/s4_thr.txt
    s4c_race.jl                threaded farTensor with a shared FrqSet (D6)              -> out/s4_race.txt
    s16_ncut.jl                nCut saturation and the N the jtail bound really needs    -> out/s16_ncut.txt
    s17_diag.jl                per-entry Float64 vs BigFloat(192) breakdown              -> out/s17_diag.txt
    s18_half.jl, s26_half2.jl  the lambda/2 cube (volume and pairKer references)         -> out/s18_half.txt, out/s26_half2.txt
    s19_c128.jl                L sweep, whole box vs octants vs reference                -> out/s19_c128.txt
    s20_ntail.jl              predicted n-truncation at nMax = 12 per shape and offset   -> out/s20_ntail.txt
    s21_pk.jl                  pairKer (third, independent) reference                    -> out/s21_pk.txt
    s22_reb.jl                 clean rebuild of the c128 table, byte comparison          -> out/s22_reb.txt
    s23_sym.jl                 T_ab(sigma R) = sigma_a sigma_b T_ab(R), reflected refs   -> out/s23_sym.txt
    s24_kr.jl                  (n,n,n) and (n,0,0) scans at L and L+12                   -> out/s24_kr.txt
    s25_ncut2.jl               the same tables with nCut forced to nMax                  -> out/s25_ncut2.txt
    sum6.py                    aggregation of the 288-row sweep
    tab/                       the twelve geometry tables (19.5-22.0 MB each)
    refcache/reftensors.txt    636 cached 220-bit references

## 11. Verdict

**Question.** Does `notes/farfield/farfield.jl` deliver `1e-13` per entry — and, for real `f`, per
part — for every cell shape from `lambda/128` to `lambda/4`, aspect ratios `1e-3` to `1e3`, every
separation from 2 cells outward, at real and complex frequency?

**Answer: no.  Per entry it misses by up to 4.8x inside that range, and per part by up to 18x.  The
misses are not random: three named mechanisms account for every one of them, two of the three are
already fixed in `work/prep/farfield_v2.jl`, and the third — a compile-time cap `nMax = 12` on the
`k`-series of `j_l` — is not fixed anywhere and destroys the answer completely (9.6e-3 per entry)
as soon as `|k| r_d` exceeds about 3.2, which is just outside the stated range and inside the range
of a `lambda/2` cell.**

What was measured: 288 tensors (12 shapes x 6 shape-adapted offsets x 4 frequencies) against
independent 220-bit references whose own convergence is median 6.0e-50; 60 further random
(shape, offset, frequency) cases for the bound; 40 boundary offsets; 21 frequency cases including
`f = -1`, `1-0.1i`, `10`, `1+3i`; two positivity blocks; two threading blocks; and four scans over
separation at fixed shape.  Aspect ratios covered: 1, 4, 11.4, 16 (three axis permutations), 87.7,
295, 678, 1020.

### 11.1 What holds

- **Excluding `c128` and `gen`, and restricted to `|k| r_d <= 3.2` (which every shape with a longest
  edge `<= lambda/4` satisfies), over 210 tensors: max-norm `<= 8.9e-14`, per entry `<= 1.3e-13`
  (one row: `r1 (16,16,16)`, `f = 0.37`), Re `<= 4.9e-13`, Im `<= 1.8e-12`.**  The per-entry target
  is met to within 1.3x, at every aspect ratio from 1 to 1020, in all three axis permutations, at
  all four frequencies, along axis, face-diagonal, body-diagonal, generic and 2-, 4-, 16-cell
  separations.
- **Route (iii) (the BigFloat k-series) is the most accurate part of the library**: 88 rows,
  max-norm `8.3e-18 .. 3.6e-15`, per entry `<= 2.1e-14`, `bound/actual` 4.99 to 1.24e4, `N` from 9
  to 48, at 0.053 s to 76.3 s per offset.  It never failed on accuracy; it is the cost defect D3.
- **Layout, signs, symmetry, determinism.**  No sign error, no index transposition against Gila
  (§3); `T_ab(sigma R) = sigma_a sigma_b T_ab(R)` to `2.7e-16` including the octant route, and the
  reflected offsets match their own references to `1.6e-15` (§8.1); `farBlock!` is bitwise identical
  at 1, 4 and 12 threads and across repeated calls (§4.2).
- **Frequency.**  `f = -1`, `f = 1-0.1i` (gain), `f = 0.05`, `f = 1+3i` (heavy damping) all behave;
  the tensor at `f = -1` is the exact conjugate of `f = +1` (§6).
- **Cost.**  `936-1010 ns` per offset single-threaded at `lambda/32`, i.e. a `128^3` far field in
  `1.96 s` — provided the shape table exists.

### 11.2 The exceptions, stated precisely

1. **`s = (1/128)^3`, a cubic cell at `lambda/128` — the finest cell Gila claims.**  Per entry
   `4.65e-13` at *every* separation tested from 3 to 128 cells and at every `kR` from 0.255 to 5.44,
   Re and Im alike, rising to `9.62e-13` at `(128,0,0)`; max-norm up to `4.65e-13`.  Cause: `nCut`
   truncating the `k`-series of `j_l` at 2 to 4 of the 13 stored terms (§1.6c); forcing
   `nCut = nMax` gives `2.3e-16`.  Confirmed against two independent 220-bit references agreeing to
   `1.8e-61`, in Float64 and BigFloat, at `L = 10 .. 56`, and with a table rebuilt from scratch that
   matches the cached one byte for byte.  **`farfield_v2.jl` fixes this.**
2. **`s = (1/16, 1/32, 1/64)` (`gen`, aspect 4).**  Same mechanism, smaller: per entry `1.4e-13` in
   the sweep, `1.65e-13` at `(64,64,64)` and `1.01e-12` at `(128,0,0)`; forcing `nCut = nMax` gives
   `3.6e-16`.  **Fixed in `farfield_v2.jl`.**
3. **Large separations on otherwise clean shapes**: `r1 (48,48,48)` and `(64,64,64)` at `1.23e-13`
   and `1.19e-13` per entry, `c32 (128,0,0)` at `7.2e-14`; all fall to `3.8-5.2e-15` when 12 shells
   are added, so `L` is under-selected, for the same reason (`est` over-states `max|T|` by up to
   6.3x, so `tol * est` is a looser criterion than `tol` per entry).  **Fixed in `farfield_v2.jl`**
   (`TOL = 1e-14` plus Theorem A; `prep.md` explicitly records "far `rho <= 0.027` 2 shells MORE").
4. **Per-part accuracy (Re and Im separately) is not delivered at `1e-13`, at real frequency or
   complex.**  Worst per-part errors over the sweep, excluding `c128`/`gen`: Re `4.9e-13`
   (`r2 (16,16,16)`, `f = 1+0.1i`), Im `1.8e-12` (`r2 (16,16,16)`, `f = 0.37`, a **real**
   frequency), and `4.7e-12` (`c128 (7,-3,2)`, `f = 2+0.2i`).  In every case the offending part
   carries less than about `1e-3` of the entry's own modulus and its **absolute** error is at
   `eps * max|T|`.  `unify.md` §9 states that for real `f` the Re/Im split makes each part keep its
   own relative accuracy; the `r2` `f = 0.37` row (Im error `1.8e-12` with max-norm `1.4e-15`)
   shows that is not true when the small part is small relative to the *largest entry of the
   tensor* rather than to its own entry.  **Not fixed in `farfield_v2.jl`, and not fixable in
   Float64 without splitting the sums by magnitude.**
5. **Outside `lambda/4`: total failure above `|k| r_d ~ 3.2` (defect D9).**  Measured per entry
   `1.7e-10` (`|k| r_d = 4.46`), `1.3e-9` (4.78), `6.4e-7` (5.80), `9.6e-3` (8.96), growing as
   `(|k| r_d)^25.6` (the exponent `2 nMax + 2 = 26`), with the library's own bound understating the
   error by up to `7.2e8` and no warning of any kind.  A `lambda/2` cubic cell sits at
   `|k| r_d = 5.44`, inside this band; its table was built (288 s, 19.2 GB) and five offsets measured
   against two independent 220-bit references that agree to all printed digits: **`7.2e-13` to
   `3.0e-12` per entry on the four route-(i) offsets `(3,3,3)`, `(4,0,0)`, `(7,-3,2)`, `(16,16,16)`
   — one real part at `9.0e-11` — against reported bounds 1.8 to 3.0 times smaller, and unchanged
   when every `nCut` is forced to the table's `nMax = 12`** — the cap itself, not the cut, is the
   limit.  The one offset that escapes is `(2,0,0)`, which route (i) cannot reach and which the
   BigFloat k-series answers to `5.6e-17`: on a `lambda/2` cell the near field is exact and the far
   field is wrong.  **Not fixed in `farfield_v2.jl`.**
6. **Cost, not accuracy, on the slender needle.**  The `(0,0,n)` offsets with `n <= 19` take route
   (iii) and meet the target (`<= 2.1e-14`) at **0.05 to 76 s per offset**: 684 s of a 684.5 s cold
   `64x64x128` build (99.9 %) for 72 of 524 280 offsets.  **Not fixed in `farfield_v2.jl`.**
7. **One reference limit of my own, stated so it is not mistaken for a result**: at `r3 (0,0,2)`
   (aspect 678, `|R| = 2 s_3 = 0.0021` against a box of diameter 0.71) my volume reference converges
   only to `3.6e-15`, and the four rows there report `3.55-3.59e-15`, which is that floor.

### 11.3 Defect list, with `farfield_v2.jl` status

| # | defect | measured | fixed in v2? |
|---|---|---|---|
| D1 | `Float32`: `bndTrm` overflows (`(2l+1)!!` at `l=29`, `|k|^l` at `l=49`), one `NaN` poisons every cumulative tail, `boundL` returns -1 everywhere, every offset falls to route (iii) | 20.9 s for one tensor instead of 1 us; a `128^3` Float32 block would take 1.4 years | **yes** (`prep.md`: selection layer forced to Float64, `Float32 -> 6.2e-8`) |
| D2 | `nCutVec`'s budget is evaluated at `2 min(s)`, a radius where the expansion often does not converge; the cut it returns is meaningless in both directions | over-truncation `4.65e-13` (`c128`), `4.25e-14` (`gen`); saturation at `nMax` for 8 of the 24 (shape, f) pairs with `|k| r_d >= 1.1` | **half**: the over-truncation is fixed (`budget/256` at the true radius, `TOL = 1e-14`); the saturation is not |
| D9 | `nMax = 12` is a compile-time cap the cut can only lower; above `|k| r_d ~ 3.2` the `j_l` `k`-series is under-resolved and the error grows as `(|k| r_d)^26` | `7.2e-13 .. 3.0e-12` per entry on a `lambda/2` cube at four offsets (measured, §1.7), `1.7e-10 .. 9.6e-3` on the extreme-aspect shapes; bound/actual down to `1.4e-9` | **no** (`NMAX` still 12 at `farfield_v2.jl:266`) |
| D10 | `L` under-selected at large separation (the same `est`-over-states-`max\|T\|` cause acting through `whlThr`) | `1.23e-13` per entry at `r1 (48,48,48)`, `-> 3.8e-15` at `L+12` | **yes** (`TOL = 1e-14` + Theorem A) |
| D3 | route (iii) has no cost bound and no fallback | 0.05 s to **76.3 s** per offset (`r3 (0,0,4)`, `f = 2+0.2i`, `N = 48`); 99.9 % of a cold slender build | **no** |
| D4 | `farRoute` computes `costWhl`/`costOct` on every offset; all three call sites discard it | 330-390 ns of 936 ns per offset = **35-42 % of the whole far-field build** | **no** (v2 lines 923, 928, 1279, 1300, 1331) |
| D5 | BigFloat output above 192 bits is no more accurate than 192 (`TABPRC = 192`) | 128 vs 256 bits: `3.15e-38`; 192 vs 256: `3.95e-58` | **no** (`TABPRC` still 192 at v2:261) |
| D6 | `MOMC`, `fs.ksr` and `needMom()`'s runtime `Base.include` are unguarded globals; `farTensor` is not thread safe when route (iii) is reachable | 5 trials x 6 concurrent route-(iii) offsets: 0 exceptions, results bitwise correct — the hazard is real in the source and did not fire | **no** (v2:1143, 1150) |
| D7 | Julia 1.12 world-age warning from `getglobal(Main, :facePair)` after a runtime `Base.include` ("this code will error in future versions of Julia") | printed on the first route-(iii) call in every run of this audit | **no** |
| D8 | shape-table build cost and memory | **298 s and 18.9 GB peak RSS** for a clean `c128` build; 306-1770 s for the other eleven; 19.5-22.0 MB on disk | **partly** (`prep.md`: 256-bit contraction and lazy `L`, 87-200 s per shape; peak memory not reported there) |

### 11.4 The one-line answer

With `farfield_v2.jl`'s fixes applied, the library would meet `1e-13` per entry everywhere I
measured **inside** `lambda/128 .. lambda/4` — items 1, 2 and 3 above are exactly what v2 addresses,
and items 4 (per-part on a part carrying `< 1e-3` of its entry) and 7 (my reference) are not
correctable by it.  It would still be **wrong without warning** for any cell with `|k| r_d > 3.2`
(`r_d = sqrt(s1^2+s2^2+s3^2)`): a cube coarser than `0.294 lambda = lambda/3.4`, a plate coarser than
`0.36 lambda`, a needle coarser than `0.51 lambda = lambda/2` on its long edge.  That is the one
defect in this report that nothing in the current tree fixes.  A cap check — `error` when the `jtail`
bound needs `N > nMax` — is three lines and would turn a silent `9.6e-3` into a refusal.
