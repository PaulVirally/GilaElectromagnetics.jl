# deliver — `notes/farfield/verify.jl` and `notes/farfield/bench.jl`

Everything below is under `REPO/notes/farfield/`.  Work dir `SCRATCH/work/deliver/`
(`smoke.jl`, `fillksr.jl`, `fillref.jl`, `rqtest.jl`, `runall.sh`, `runfin.sh` and their logs).
SCRATCH = `REPO/notes/farfield/scratch`.  Machine note: every time below was measured while the
root's own 128^3 runs were on the machine (load average 8-10, 10 GB resident in another Julia
process), so they are upper bounds, not quiet-machine numbers.

## 1. Files written

    verify.jl                954 lines   the nine verification parts (a)-(i)
    bench.jl                 194 lines   the before/after build comparison
    ref.jl                   231 lines   copy of SCRATCH/work/reference/ref.jl with REPODIR and
                                         CACHEDIR repointed (repo root is ../.., cache is
                                         ./refcache); refPairs, refSum, refTensor, volTensor, SUMIDX
    refquad.jl               107 lines   the graded Gauss-Legendre volume reference, standalone
                                         (no face pairs, no shared code with ref.jl): refQuad(D,s,f)
    mkrefcache.jl            152 lines   the converter: every round-1 cache -> refcache/reftensors.txt
    crosscheck/famG.jl       362 lines   copy of SCRATCH/work/famG/famG.jl (taylorSys, farTen), used
                                         by part (f) only; famC.jl is not used and not copied
    refcache/reftensors.txt  2.9 MB      1059 records, one format (below)
    ksrcache/*.txt           3 files     145 route-(iii) tensors: 72 slender at f = 1, 72 at 1+0.1i,
                                         1 cubic; farfield.jl reads them through KSRDIR[]
    tables/                  26 files    a_rho a_band a_mu a_div | b_volume | c_gila_<shp>_<r|c> (8)
                                         | d_ref | e_terms e_ncut e_route e_digits | f_bands f_ksr
                                         f_overlap | g_sym | h_pos | i_cost | bench_1 bench_12

`farfield.jl` was not modified.

### The cache format (one record per line)

    <kind>|D=d1,d2,d3|s=n//d,n//d,n//d|f=<re>;<im>|p=<bits>|n=<ord>|<v1> <v2> ...   v = <re>;<im>

`kind` is `pairs` (36 values, `I_FF'/V_t`, fp = 6(F-1)+F'), `vol` (9 entries of the volume rule,
column major) or `tns:<src>` (9 entries of another agent's 220-bit tensor, src in famB famBsub
auditG famG famD famDQ refQuad).  It is `ref.jl`'s own format with the third kind added, so the
reference agent's file is copied **verbatim** (no reprint, no precision loss) and `ref.jl` reads
and appends to it unchanged.  Record counts: 240 from the reference agent (pairs + vol), 361 famB
/ famBsub, 365 auditG / famG, 63 famD, 14 refQuad, plus the 16 built here (section 4);
duplicate keys are dropped (the reference agent's 242 lines hold 240 distinct records).
`verify.jl` prefers `pairs` (36 x pairKer + srfSum!), then `vol`, then `tns:refQuad`, then
`tns:famDQ`, then any other agent's, and prints which one it used in the `src` column.
`mkrefcache.jl` is idempotent: it keeps every record its sources do not provide (so the 16
references `ref.jl` appended survive a rebuild) and merges, rather than overwrites, the
`ksrcache/` files.  Route (iii) tensors: 72 slender offsets at f = 1 and 72 at f = 1+0.1i, the
second frequency completed here (unify's cache had 22 of them), 8-17 s of BigFloat(128) moments
each.

## 2. `verify.jl`: what each part produces, and what it measured

    JULIA_NUM_THREADS=1 julia --startup-file=no --project=SCRATCH/env verify.jl <part>

Each part writes `tables/<part>_*.txt` and the same text to stdout.  Times are single-threaded,
with every cache present, on the loaded machine described above.

| part | s | tables | what it produces |
|---|---|---|---|
| a | 5.8 | a_rho a_band a_mu a_div | geometry, exact |
| b | 14.4 | b_volume | the volume identity and the normalisation |
| c | 4.5 | c_gila_<shp>_<r,c> (8) | Gila's far field today |
| d | 22.4 | d_ref | farTensor vs every cached reference |
| e | 85.7 | e_terms e_ncut e_route e_digits | term counts from the bounds |
| f | 89.3 | f_bands f_ksr f_overlap | the near/far join |
| g | 37.7 | g_sym | invariances, homogeneity, number types |
| h | 553 | h_pos | anti-Hermitian positivity |
| i | 23.9 | i_cost | cost per offset, term histogram |

Part (h) is the only one near the ten-minute mark and 492 s of its 553 s is **Gila's** side:
`wekTrp` at `intOrd = 48` costs 32-33 s per (shape, frequency) and the six cases pay it six times
(35.8, 65.1, 38.1, 67.6, 128.9, 205.7 s for the Gila build of each case, the last two including
the (6,6,12) fill).  `farBlock!` on those blocks is milliseconds.

### (a) geometry, exact rational arithmetic — `a_rho`, `a_band`, `a_mu`, `a_div`

- `a_rho`: rho and rho_oct = max over the 8 octants of (r_d/2)/|sigma R + s/2|, for the classes
  axis, face diagonal, body diagonal, (n,1,0), (n,2,1) at n = 2,3,4,6,8,16,32,64, cubic (scale
  invariant) and slender.  Reproduces geometry.md exactly (0.866025, 0.522233 at (2,0,0); 0.577350
  at (3,0,0); 0.216506 at (8,0,0)).  rho/rho_oct rises from 1.54 to 1.98 with separation: the
  octant split buys a factor 2 in rho only in the far field, 1.54-1.66 at two cells.
  Slender short axis: rho < 1 first at n = 23, rho < 0.5 at n = 46; rho_oct < 1 at n = 12,
  rho_oct < 0.5 at n = 23.
- `a_band`: counts of egoToe octant offsets with rho above 0.5/0.3/0.2/0.1.  Cube: **30 / 134 /
  429 / 3089 at N = 32, 64 and 128 alike** — the slow band is a fixed set (geometry.md's
  37/141/436/3096 counts the 7 touching offsets with max n_i = 1, which the far field never sees).
  The slender cell is the exception and the table says so: 242/682/1417/5458 at N = 32 against
  288/1192/3618/19736 at N = 128, because rho > 1 along (0,0,n) up to n = 22.
- `a_mu`: `wgtT` of farfield.jl against the same integral evaluated as two elementary halves,
  both in `Rational{BigInt}`: **0 disagreements over 5 edge lengths x n = 0..24**.
- `a_div`: the divergence identity on F = |x-y|^{2j}, j = 1,2,3, at D = (2,0,0), (3,1,0), (2,2,2),
  cubic and slender: the 36 face-pair moments (exact multinomial integrator on the `facePair`
  panels) summed by the `srfSum!` signs against the difference-box form with the exact weight
  moments.  **max |difference| = 0//1 exactly in all 18 rows.**  Sample values (divided by V_t, as
  Gila stores them): c32 (2,0,0) j = 1 gives G[1,1] = -1//8192 = -4 mu_0^3/V_t, geometry.md's
  hand check -1//268435456 times 1/V_t.

### (b) the volume identity and the normalisation — `b_volume`

20 rows: every cached `vol` record that also has a `pairs` record (the 18 offsets of
reference.md section 5 plus c32 (5,1,0) and c32 (8,8,8)).  Columns ePP/ePM/eMP/eMM are the four
sign conventions of the volume form against `refTensor`, entPP the worst per-entry ePP, then
farTensor against both.

    (+diag, +off) agrees with the 36 face pairs to 3.7e-65 .. 1.05e-49 (the 1.05e-49 is the
    volume rule's own truncation at the slender (0,0,2)); every other convention is wrong by
    0.567 .. 2.0.  farTensor against the same references: <= 4.23e-15 (max norm) and per entry,
    and against volTensor the same to the last digit (fVol = fMx in every row).

### (c) Gila's far field today — `c_gila_<shape>_<r|c>`

`egoSrfFxd!` + `srfSum!` at Gila's own `quadOrd` order against the cached 220-bit `pairs`
references, per shape and frequency: worst face pair, assembly amplification and the entry that
attains it, assembled tensor error in both norms.  The cubic tables reproduce reference.md
line for line (c32 f = 1: per-pair 1.1e-15 .. 8.9e-14, amplification 36 .. 1300, tensor
5.0e-14 .. 4.7e-13; c4: amplification 5 .. 163, tensor 1.6e-15 .. 2.8e-13).

**New rows (16 references built for this table, see section 5):** the slender long axes at
separations 3, 6, 32 and 64, which reference.md did not have.  The amplification keeps growing:

    sl, f = 1, (n,0,0)   n = 2     3      4      6      8      16     32     64
    amplification        4300   10400  18700  29800  30600  39900  82700  167000
    tensor err (entry)   8.0e-12 1.4e-11 2.0e-11 3.4e-11 3.4e-11 1.3e-11 1.1e-11 1.1e-11
    worst face pair      7.7e-14 1.1e-14 3.9e-15 6.0e-15 3.9e-15 2.7e-15 2.0e-15 2.6e-15

so Gila's slender long-axis far field is 1e-11 accurate at every separation out to 64 cells, from
face pairs that are themselves good to 1e-14: **the amplification reaches 1.67e5**.  The short
axis is unchanged and unchanged in its verdict: 0.789 relative at (0,0,2), 3.68e-2 at (0,0,4),
1.79e-3 at (0,0,8), 2.97e-6 at (0,0,16).  Four rows (ax3 at n = 3, 6, 32, 64) print
"no cached reference": `pairKer` on those panels ran for more than 25 minutes without returning.

### (d) the new method against every cached reference — `d_ref`

423 rows: shape, offset, frequency, rho, kR, route, L, eMx, per-entry, per-Re, per-Im, whether the
offset is on the item-3 grid, and which cache the reference came from.  Summary per (shape, f):

    shp  f          n     eMx       pEn       eRe       eIm
    c32  1          81    4.23e-15  4.23e-15  2.64e-14  3.49e-14
    c32  1+0.1i     79    1.57e-15  3.00e-15  1.17e-13  5.65e-14
    c32  1+1i       13    1.30e-14  6.32e-14  9.92e-13  2.86e-13
    c8   1          28    5.61e-15  5.61e-15  1.48e-13  1.54e-13
    c4   1          29    8.60e-15  8.60e-15  4.90e-13  8.80e-13
    c4   1+0.1i     29    2.50e-15  1.17e-14  2.76e-14  4.47e-14
    sl   1          49    2.50e-15  4.71e-15  4.71e-15  6.28e-16
    sl   1+0.1i     48    2.44e-15  5.09e-15  5.05e-15  7.27e-15
    (plus f = 0.37, 3+0.3i rows for c32 and sl)

**At f = 1 and 1+0.1i, over four shapes and 2 to 64 cells: eMx <= 8.6e-15, per entry <= 1.2e-14.**
The 6.3e-14 worst row is c32 at f = 1+1i.  The eRe/eIm columns above 1e-13 are entries whose real
or imaginary part carries less than 1% of the entry's own modulus (unify.md section 9).

### (e) term counts from the bounds — `e_terms`, `e_ncut`, `e_route`, `e_digits`

- `e_terms`: 192 rows (4 shapes x 2 frequencies x 3 directions x 8 separations) of Lwb (whole box),
  its cost in complex multiply-adds, Loct = max_j L_j and costOct, and the route taken.  c32 axis:
  Lwb = -1 (no certificate at L <= 56) / 50 / 34 / 24 / 20 / 14 / 12 / 10 at n = 2 .. 64, costs
  4473 -> 223 terms; Loct at n = 2 is 55 with 116480 terms, which is why route (ii) is used only
  where (i) has no certificate.  c4 bottoms out at Lwb = 16 (cost 516) and the slender long axis
  at 10.
- `e_ncut`: the n-truncation from the `jtail` bound at the smallest radius actually routed to (i)
  or (ii): max nCut = 5 (c32), 8 (c8), 10 (c4), 5 (slender), **never the stored nMax = 12**, so the
  n-series is certified everywhere the library evaluates, at both frequencies.
- `e_route`: routing counts at 32^3, 64^3, 128^3 (cubes) and 16x16x32, 32x32x64, 64x64x128
  (slender), both frequencies, plus the whole-box L histogram of the largest block.
  128^3: **2 097 132 / 12 / 0** (c32), 2 097 129 / 15 / 0 (c8 and c4); slender 64x64x128:
  **524 066 / 142 / 72**.  Identical at f = 1 and 1+0.1i.  The L histogram bottoms at 8 (c32),
  12 (c8), 16 (c4), 8-10 (slender), and 56.0% / 86.3% / 98.4% of the offsets sit at that floor.
  Setup per (shape, frequency) 0.6-1.6 s.
- `e_digits`: the same route and the same L in Float64 and in BigFloat(192), so truncation cancels
  and only rounding is left.  **Worst over the grid: 1.59 digits (c4 (64,0,0), kR = 100.5,
  L = 16); the two-cell octant route loses 1.28 digits; most rows are 0.0-0.9.**

### (f) the join — `f_bands`, `f_ksr`, `f_overlap`

- `f_bands`: the k-series band recomputed **only on the offsets the router sends to (iii)**.  The
  three cubic shapes have none (the table says so explicitly).  The slender cell has 72, listed one
  by one with Dmax, x = |k| Dmax, the assembly amplification Lambda = (sum_fp I_{-1})/(4 pi |f|^2
  V_t)/est(R), eps*Lambda, and the k-series order N the proven bound selects for 1e-16/Lambda at
  both frequencies: **x in [0.280, 0.607], N in [12, 17], Lambda 1.88 .. 1038, eps*Lambda up to
  2.30e-13.**  That last number is the reason route (iii) is done in BigFloat(128): a Float64
  face-pair k-series on this set is already 2.3e-13 wrong from rounding alone.  (theory.md's
  independent estimate from the same bound was 13-16 terms with a conservative Lambda = 1e5.)
- `f_ksr`: route (iii) against the volume reference on the seven near-needle offsets, both
  frequencies, with the route farTensor actually takes in a column: **route (iii) is at
  2.3e-17 .. 1.6e-16 in all ten of its rows** (matching unify.md's 2.0e-17 .. 1.6e-16).  Two of
  the seven offsets, (0,0,20) and (0,1,30), lie outside the (iii) set (which is n3 = 2..19) and
  are routed to the octant split: they come out at 4.5e-16 .. 2.5e-15, and the forced `tnsKsr`
  value at the same offsets is 2.5e-17 .. 6.7e-17, so the two routes agree with each other and
  with the reference.  Those two forced `tnsKsr` calls are the only thing `verify.jl` recomputes
  (about 20 s of part (f)'s 108 s): being outside the routed (iii) set, they are not in `ksrcache/`.
- `f_overlap`: farTensor against famG's (G,S) Taylor recurrence at famG's own order rule, over
  every offset of a 16^3 octant with max-norm >= 2, c32 and c4, both frequencies (4085 offsets
  per case):

        shp  f        band    n     dMx       dEn       worst offset
        c32  1        2       7     9.61e-15  9.61e-15  (2,2,2)
        c32  1        3-4     98    6.64e-15  6.64e-15  (3,1,1)
        c32  1        5-8     604   2.80e-15  3.09e-15  (2,2,8)
        c32  1        9-15    3367  2.52e-15  2.59e-15  (9,9,10)
        c4   1        9-15    3367  6.83e-15  8.48e-15  (13,13,13)
        c4   1+0.1i   9-15    3367  7.42e-15  8.42e-15  (13,13,13)

  i.e. **two independent expansions agree to 1.1e-14 per entry at lambda/32 and lambda/4**, which
  is the free end-to-end check of the geometry bookkeeping, the srfSum! sign and the normalisation.

### (g) invariances — `g_sym`

50 random offsets per (shape, frequency) from [-40,40]^3 (fixed LCG seed, no RNG dependency):
D -> -D, transpose and the three axis reflections are **bit-exact** (0.0) — they are exact by
construction in this expansion and prove bookkeeping, not accuracy; the axis permutation, which
mixes different (l,m) columns, is **<= 1.38e-15**; homogeneity T(lam f, s/lam) = lam^-2 T(f, s) at
lam = 3.7 is **<= 6.8e-15**.  Number type: ComplexF32 in gives ComplexF32 out at 3.07e-8 of the
Float64 value (eps(Float32) = 1.19e-7), Complex{BigFloat} in gives 160-bit out, 5.06e-16 from the
Float64 value.

### (h) anti-Hermitian positivity — `h_pos`

6^3 blocks at lambda/32 and lambda/4 and a (6,6,12) slender block, Gila's own build against the
same block with every offset of max-norm separation >= 2 replaced by `farBlock!`; the contact and
touching-shell entries are Gila's in both.

    case                lam_max     lam_min (Gila)    lam_min (farfield)  negatives  eps*lam_max
    c32 6^3   f = 1     7.729e-02   -4.468094e-15     -2.448448e-15       248 -> 201  1.72e-17
    c4  6^3   f = 1     2.389664    -6.398731e-15     -5.009707e-15        58 ->  49  5.31e-16
    sl  6x6x12 f = 1    1.011658e-2 -6.553241e-14     -1.921333e-14       596 -> 576  2.25e-18
    c32 6^3   f = 1+0.1i            +2.855081e-03     +2.855081e-03 (13 digits)  0/648
    c4  6^3   f = 1+0.1i            +7.124128e-03     +7.124128e-03              0/648
    sl  6x6x12 f = 1+0.1i           -9.939322e-03     -1.465001e-02       108 -> 76

At real f the operator is lossless, Im M is positive semi-definite with a large null space, and
`farBlock!` moves the most negative eigenvalue **1.8x (c32), 1.3x (c4) and 3.4x (slender) closer
to zero**; both remain above the Float64 floor because the residual is dominated by the contact
and touching-shell entries, which are Gila's in both builds.  The slender f = 1+0.1i block is
strongly indefinite in both builds (-1e-2 against lam_max = 0.195): a property of the 1:1:16
discretization, not of the far field.  `max |Gila - farBlock!|` over the separated offsets is
5.58e-13 (c32), 8.74e-14 (c4) and **0.616 at the slender (0,0,2)**, which is Gila's fixed rule,
not this library (part (f) puts farTensor at 4e-17 there).

### (i) cost — `i_cost`

Route (i) with BenchmarkTools, single thread, zero allocations, at c32 and c4, both frequencies:

    L        4     8     12    16    20    30    40    56
    terms    51    152   307   516   779   1673  2904  5576
    ns       313   423   633   896   1279  2571  5025  9644     (c32, f = 1)
    ns/term  6.1   2.8   2.1   1.7   1.6   1.5   1.7   1.7

Route (ii) at the offsets that need it: (2,0,0) 116480 terms in 96.6 us, (2,1,1) 76160 in 63.4 us,
0.83-0.85 ns/term.  Term histogram over a 32^3 octant: mean 335 terms/offset (c32), 480 (c8), 685
(c4), 634 (slender, including its 72 route-(iii) and 126 route-(ii) offsets).  `farBlock!` over a
whole block, single-threaded on the loaded machine: 32^3 in 0.037 s (1119 ns/offset), 64^3 in
0.35 s (1337 ns/offset), c4 32^3 in 0.059 s, slender 32^3 in 0.050 s.

## 3. `bench.jl`: the before/after build

    JULIA_NUM_THREADS=<n> julia --startup-file=no --project=SCRATCH/env bench.jl 32/32 64/32 128/32 32/8 32/4

`N/den` is an N^3 self volume of (1/den)^3 cells; a 4^3 build of both sides warms up the compiler
first; output is appended to `tables/bench_<n>.txt`.  Nothing under `src/` is touched.

**Before** = the stages of `genEgoCrcSlf!` called one by one through
`GilaElectromagnetics.GilaVacuum`: `wekTrp` (contact integrals, `intOrd = 48`), the threaded
`egoFunInn!` loop over every Toeplitz offset, `egoFunSng!` on the eight offsets with all indices
<= 2, the identity term, `egoToeCrc!` into the doubled circulant, `gthEgoCmp!`, `genEgoFur`.
Since `wekTrp` is memoized on (scale, order, frequency), it is timed once per shape and that time
is carried into both totals.  **After** = the same, with `farBlock!(egoToe, s, f)` in place of the
`egoFunInn!` loop (`farSetup` timed separately as the one-time table), everything else identical.
Both end in `GlaVacOprMem(cmpInf, egoFur, vol, vol)`, so both are usable operators.

Three checks, printed for every configuration:

    the staged "before" egoFur against GlaVacOprMem(cmpInf, vol) itself:  0.000e+00 (bit-identical)
    the "after" egoFur against the "before" egoFur, 16^3 at lambda/32:    2.060e-14
        (4^3: 5.314e-14;  32^3: 1.664e-14)
    both operators on the same random vector, 16^3:  max|dy|/max|y| =     2.544e-14
        (4^3: 2.572e-14;  32^3: 1.341e-14)

The `egoFur` difference is Gila's quadrature error, not the library's: the worst separated offset
of the 16^3 block is 5.576e-13 at (0,5,5), and part (d) puts `farTensor` at 4.2e-15 against the
220-bit reference there.

Timings measured here (the machine was carrying the root's own runs; 64^3 and 128^3 were left to
the root as instructed):

    N     threads   before: contact / far / rest / total       after: contact / table / far / total
    16      1       32.29 / 7.14 / 0.07  / 39.50 s             32.29 / 0.70 / 0.007 / 33.06 s
    32      1       32.29 / 24.75 / 0.11 / 57.15 s             32.29 / 0.77 / 0.042 / 33.18 s
    16     12        6.81 / 1.65 / 0.06  /  8.52 s              6.81 / 0.71 / 0.002 /  7.58 s
    32     12        6.81 / 5.81 / 0.07  / 12.69 s              6.81 / 0.73 / 0.010 /  7.61 s

    far fill 588x (32^3, 1 thread), 577x (32^3, 12 threads), 1018x / 794x at 16^3
    whole build 1.7x at 32^3 either way -- because with the far field gone the build is
    **entirely the contact integrals**: 32.3 s of the 33.2 s single-threaded total is wekTrp at
    intOrd 48, and 6.8 s of 7.6 s on 12 threads.

Band attribution of the raw quadrature fill (single thread only, one `@belapsed` per band times the
band population), 32^3 at lambda/32:

    order  offsets  us/offset  band s   kernel evals
    4      27855    523.3      14.577   2.567e+08
    5      4570     1206.6     5.514    1.028e+08
    6      218      2473.9     0.539    1.017e+07
    7      98       4639.1     0.455    8.471e+06
    9      19       12671.9    0.241    4.488e+06
    serial far fill 21.33 s over 3.827e+08 kernel evaluations, 55.7 ns/eval

which reproduces the root's `bench_before.jl` (20.04 s, 3.827e8 evaluations, 52.4 ns/eval).

## 4. New references and caches built here

- 16 new 220-bit `pairs` references (`SCRATCH/work/deliver/fillref.jl`, `refPairs` of `ref.jl`):
  the slender long axes (n,0,0) and (0,n,0) at n = 3, 6, 32, 64, both frequencies, 32-43 s each.
  They are in `refcache/reftensors.txt` and are what fills the new rows of part (c).
- The slender short axis (0,0,n) at n = 3, 6, 32, 64 was attempted and **abandoned**: `pairKer`
  did not return on (0,0,3) in 25 minutes (the same stall famD reported).  Those four rows of
  part (c) print "no cached reference".  `refquad.jl` is the way to build them.
- 50 route-(iii) tensors added to `ksrcache/` (`SCRATCH/work/deliver/fillksr.jl`): the slender set
  at f = 1+0.1i was only 22 of 72 in unify's cache.  Both frequencies are now complete (72 + 72),
  10-15 s of BigFloat(128) moments per offset, which is why part (h) fell from 711 s to 553 s.
- `refquad.jl` was checked against the cached 220-bit face-pair reference at c32 (8,0,0):
  **3.2e-45** at order 16, 192 bits (self-convergence order 12 vs 16: 2.5e-33), 88 s.  At the
  slender-type near case (1,0,2) of a cubic cell it self-converges only to 4.0e-19 at order 16, so
  the near-singular offsets need order 30-40 at 256 bits, as unify measured (270-443 s each).

## 5. What `farfield.jl` lacks (nothing was changed in it)

1. **The shape tables have no home.**  `TABDIR[]` defaults to `notes/farfield/shapetab`, which does
   not exist; the four 19.5 MB tables live in `SCRATCH/work/unify/cache`.  Both deliverables
   therefore carry a three-candidate search (`$FARFIELD_TABDIR`, `notes/farfield/shapetab`,
   `notes/farfield/scratch/work/unify/cache`) and pick the first directory that contains a
   `*_p192.txt`.  Either the 78 MB of tables get committed under `notes/farfield/shapetab/`, or
   `farShape`/`farSetup`/`farTensor`/`farBlock!` should take a `dir` keyword; a global `Ref` that
   every caller has to set before the first call is easy to get wrong (a missing one costs 500 s
   per shape, silently).
2. **`needMom()` trips Julia 1.12's world-age rule.**  `tnsKsr` -> `needMom()` does
   `Base.include(Main, MOMJL[])` and then calls the two functions through `invokelatest`, which
   works but prints two "access to binding `Main.facePair` in a world prior to its definition
   world" warnings the first time route (iii) is hit inside a function, and the warning says it
   "will error in future versions of Julia".  Loading `moments.jl` once at include time (or
   holding the two functions in `Ref`s filled at load) would remove it.
3. **No accessor for the route decision in bulk.**  `farRouteStat(fs, dim)` returns counts, the
   whole-box L histogram and the (iii) offsets, but not the (ii) offsets, so part (e) and the
   cost tables re-walk the block with `farRoute` to get them.  Returning the (ii) list (or a
   per-offset kind array) would make the block diagnostics one call.
4. **The k-series amplification is computed but not exposed.**  `tnsKsr` returns `(G, N, amp, GB)`,
   but `ksrCached!` and `farTensor` drop `N` and `amp`, and the on-disk cache stores only the
   tensor.  Part (f) has to rebuild `Lambda` from `pairMoments(..., -1)` on the 36 face pairs
   (1.7 s per offset) to report the band.  Writing `N` and `amp` into the cache line would make
   the band table free.
5. **`farBlock!` names its own index convention only in the docstring** ("max(i) >= 3, offset
   i .- 1").  Since the complement of that set is exactly Gila's `egoFunSng!` set, a tiny exported
   predicate (`isFarInd(i)`) would let a caller assert the partition instead of re-deriving it.
6. Minor: `farTensor` errors when `max|D| <= 1` rather than returning `NaN`, which is right, but
   `farBlock!` silently leaves those entries untouched -- correct and documented, worth an
   `@assert` in the caller's tests (bench.jl checks the resulting `egoFur` instead).

Nothing else was missing: `est`, `jbnd`, `hbnd`, `jtail`, `bndTrm`, `bndCum`, `pickCut`,
`boundL`, `boundLoct`, `costWhl`, `costOct`, `srfSum`, `panSpan`, `ksrOrd`, `tnsWhl!`, `tnsOct!`,
`tnsKsr`, `FarWs`, `farSetup`, `farShape`, `SGN8`, `LMAX`, `NMAX`, `TABPRC`, `KSRPRC` are all
reachable from `verify.jl` and were all needed by one part or another.

## 6. Running it

    ENV=REPO/notes/farfield/scratch/env
    cd REPO/notes/farfield
    julia --startup-file=no --project=$ENV mkrefcache.jl          # once, rebuilds refcache/ + ksrcache/
    for p in a b c d e f g h i; do
      JULIA_NUM_THREADS=1 julia --startup-file=no --project=$ENV verify.jl $p
    done
    JULIA_NUM_THREADS=1  julia --startup-file=no --project=$ENV bench.jl 32/32 64/32 128/32 32/8 32/4
    JULIA_NUM_THREADS=12 julia --startup-file=no --project=$ENV bench.jl 32/32 64/32 128/32 32/8 32/4

`verify.jl` with no argument runs every part in one process.  Parts (c) and (h) and `bench.jl`
need `using GilaElectromagnetics`; the others need only the caches.  `bench.jl` appends, so delete
`tables/bench_<n>.txt` before a clean run.  Set `FARFIELD_TABDIR` if the shape tables move.

## 7. Summary of the numbers this deliverable produces

    exact-rational divergence identity, 18 rows (2 shapes x 3 offsets x j = 1,2,3)   0//1 exactly
    mu_n closed form vs the two elementary halves, 5 lengths x n = 0..24             0 disagreements
    volume form vs the 36 face pairs, 20 offsets, (+diag,+off)                       3.7e-65 .. 1.05e-49
    the three wrong sign conventions                                                 0.567 .. 2.0
    Gila today, cubes: per-pair / amplification / tensor per entry                   1.1e-15 .. 1.1e-13 /
                                                                                     5 .. 1300 / 4.5e-15 .. 7.9e-13
    Gila today, slender long axis, amplification at n = 2 .. 64                      4300 .. 167000
    Gila today, slender long axis, tensor per entry                                  8.0e-12 .. 3.4e-11
    Gila today, slender short axis (0,0,n), n = 2/4/8/16                             0.789 / 3.7e-2 / 1.8e-3 / 3.0e-6
    farTensor vs 423 cached 220-bit references, f = 1 and 1+0.1i                     eMx <= 8.6e-15, entry <= 1.2e-14
    farTensor at f = 1+1i (13 rows)                                                  entry <= 6.3e-14
    L from the bound, c32 axis n = 2 .. 64                                           -1, 50, 34, 24, 20, 14, 12, 10
    n-truncation nCut, max over l, four shapes                                       5 / 8 / 10 / 5 (nMax = 12)
    routing at 128^3 (i)/(ii)/(iii): c32, c8, c4                                     2097132/12/0, .../15/0, .../15/0
    routing at 64x64x128, slender                                                    524066 / 142 / 72
    Float64 digits lost at the same L (BigFloat(192) control), worst over the grid   1.59
    k-series band on the routed (iii) set: x, N, Lambda, eps*Lambda                  0.280-0.607, 12-17, 1.9-1038, 2.3e-13
    route (iii) vs the graded volume reference, 10 rows                              2.3e-17 .. 1.6e-16
    farTensor vs famG's Taylor route, 4085 offsets x 4 cases                         <= 1.1e-14 per entry
    axis permutation / homogeneity at lam = 3.7                                      1.38e-15 / 6.8e-15
    D -> -D, transpose, reflections                                                  0.0 (exact by construction)
    anti-Hermitian lam_min, 6^3 lambda/32, f = 1: Gila -> farfield                    -4.47e-15 -> -2.45e-15
    cost per offset, route (i) at L = 8 / 16 / 56                                    423 / 896 / 9644 ns
    cost per offset, route (ii) at the two-cell shell                                63-105 us
    farBlock! over 32^3 / 64^3, single thread (loaded machine)                       0.037 s / 0.35 s
    staged "before" egoFur vs GlaVacOprMem's own                                     0.000e+00 (bit-identical)
    "after" vs "before" egoFur at 4^3 / 16^3 / 32^3                                  5.31e-14 / 2.06e-14 / 1.66e-14
    both operators on one random vector, 16^3 / 32^3                                 2.54e-14 / 1.34e-14
    far fill 32^3, 1 thread: 24.75 s -> 0.042 s                                      588x
    whole build 32^3, 1 thread: 57.15 s -> 33.18 s                                   1.7x, and 32.3 s of the
                                                                                     remainder is wekTrp at intOrd 48
