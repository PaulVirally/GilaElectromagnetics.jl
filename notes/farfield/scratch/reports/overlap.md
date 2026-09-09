# overlap — the three far-field expansions (b), (g), (c) compared against each other and against Gila

Work dir `SCRATCH/work/overlap/`.  Libraries `famB/famB.jl`, `famG/famG.jl`, `famC/famC.jl` were
included unmodified; nothing outside `work/overlap/` was touched.  Every number below is Float64
unless it says otherwise.  Timings are deliberately absent (other agents were running).

## 1. The three calls, verbatim (`work/overlap/common.jl`)

Common: `s` the edge tuple in wavelengths, `R = (n1 s1, n2 s2, n3 s3)`, `f` the complex frequency,
`rho = sqrt(s1^2+s2^2+s3^2)/|R|`.  All three return the same object,
`T_ab = (1/V_t) int_D w(delta) [(d_a d_b + dab k^2) g](R+delta) d delta`, with no sign or scale
adjustment (checked at `(5,1,0)`, `s=(1/32)^3`, `f=1`: the three agree with each other to `2.7e-15`
and with Gila to `6.4e-14`, the number quoted in the BRIEF).

    (b) famB   gt = geoTab(sQ, Lmax, nMax, BigFloat)              # sQ exact Rational{BigInt}
               qt = qCvt(qTab(gt, Complex{BigFloat}(f)), Float64) # geometry+n-series in BigFloat
               farTns!(G, qt, FarWs(Lmax,Float64), R, 2pi*f; Lcut = L, fix = true)
               L  = 2*cld(ceil(Int, 14/log10(1/(0.7*rho))), 2)    # its rule, rounded up to even
               nMax = 6 (lambda/32 and slender), 12 (lambda/4)    # its n-truncation table
               not converged if rho >= 1 or L > 40

    (g) famG   a  = first(taylorSys(R, f, q+2)); farTen(a, s, f, q)
               q  = 2*ceil(Int, (17/log10(1/rho) + 8)/2)          # exactly its s_ver.jl rule
               not converged if q > 60 (the criterion this task specifies); its own script caps at 96,
               and 96 is what was used to evaluate wherever q <= 96

    (c) famC   b  = bndC(R, s, f; ax); ax = argmin_ax b.tauXi
               M  = ceil(Int, 14/log10(1/tauXi)); Q = ceil(Int, 14/log10(1/tauT)) + ceil(Int, kR0/8)
               farC(R, s, f; ax = ax, M = M, Q = Q).ten
               not converged if tauXi >= 1 (numerically >= 1-1e-9) or M > 400

    Gila       opt = GV.CPUKerOpt(f,48,false,GV.CPU()); egoSrfFxd!(..., quadOrd(max|n|, 2pi|f|max s));
               srfSum!  — the BRIEF's snippet verbatim.

The published famB and famC rules carry the numerator 13 (target `1e-13`); this task asks for
`1e-14`, so 14 was used.  That choice is *generous* to both: section 7 shows both rules are already
short of `1e-14` at 14, and would be shorter at 13.  famG's rule is used exactly as coded.

## 2. Cubic cell 1/32, all 13797 octant offsets `0<=n_i<=23`, `max n_i>=3`

`X-Y` is `max over the 9 entries of |X-Y| / max|Y|`; `X-Xg` is the same against Gila.
Full tables incl. worst offsets in `out/cub32r_tab.txt`, `out/cub32c_tab.txt`.

**f = 1**

    max-norm, |X-Y|/max|Y|
    band        b-g       b-c       g-c       b-Gila    g-Gila    c-Gila     nOff
    3        2.47e-14  4.05e-14  1.58e-14   4.50e-13  4.50e-13  4.50e-13      37
    4        6.00e-15  3.17e-15  4.96e-15   4.73e-13  4.72e-13  4.73e-13      61
    5-6      3.79e-15  8.03e-15  9.75e-15   5.58e-13  5.58e-13  5.58e-13     218
    7-8      2.83e-15  1.52e-15  2.90e-15   4.16e-13  4.16e-13  4.16e-13     386
    9-16     2.31e-15  1.89e-15  2.60e-15   4.00e-13  4.01e-13  4.00e-13    4184
    17-23    2.42e-15  1.66e-15  2.48e-15   3.09e-13  3.09e-13  3.10e-13    8911

    per entry, |X-Y|/|Y| over entries above 1e-4 of the largest (i.e. not suppressed by symmetry)
    3        3.58e-14  4.14e-14  1.58e-14   7.86e-13  7.85e-13  7.85e-13
    4        2.43e-14  2.44e-14  4.96e-15   6.00e-13  6.01e-13  6.00e-13
    5-6      7.53e-15  8.03e-15  9.75e-15   9.95e-13  9.95e-13  9.95e-13
    7-8      7.04e-15  4.97e-15  2.90e-15   2.73e-12  2.73e-12  2.73e-12
    9-16     8.84e-15  9.21e-15  2.99e-15   1.20e-11  1.20e-11  1.20e-11
    17-23    3.20e-15  3.28e-15  3.26e-15   2.44e-11  2.44e-11  2.44e-11

    real part, max|ReX-ReY|/max|ReY|        imaginary part, max|ImX-ImY|/max|ImY|
    3        2.47e-14  4.06e-14  1.59e-14   |  1.56e-15  2.38e-15  2.52e-15
    4        6.07e-15  3.19e-15  5.22e-15   |  1.64e-15  3.34e-15  3.20e-15
    5-6      4.34e-15  8.43e-15  1.02e-14   |  2.51e-15  2.39e-15  2.70e-15
    7-8      3.57e-15  1.78e-15  3.40e-15   |  2.44e-15  1.87e-15  2.44e-15
    9-16     2.47e-15  2.35e-15  3.88e-15   |  2.54e-15  2.32e-15  2.86e-15
    17-23    3.06e-15  1.98e-15  3.12e-15   |  4.20e-15  4.34e-15  2.48e-15

    per entry, real part |ReX-ReY|/|ReY|    per entry, imaginary part
    3        5.73e-14  4.18e-14  1.59e-14   |  2.14e-14  2.83e-14  3.15e-14
    4        2.43e-14  2.44e-14  7.93e-15   |  1.82e-14  3.21e-14  4.25e-14
    5-6      5.13e-14  7.51e-14  2.41e-14   |  9.64e-15  2.25e-14  2.36e-14
    7-8      5.42e-14  5.67e-14  4.63e-14   |  2.48e-15  3.61e-15  4.65e-15
    9-16     2.25e-13  4.94e-13  5.58e-13   |  2.31e-13  2.40e-13  1.74e-13
    17-23    1.69e-12  1.01e-12  6.87e-13   |  2.79e-12  3.04e-12  3.46e-13

The last two columns of the per-part tables are the only ones above `1e-13`, and they are an
artefact of the `1e-4` entry threshold, not a disagreement: at 17-23 cells the *smaller* of Re/Im
of an entry can be `1e-4` of the largest part in the tensor, so a `4e-15` absolute agreement reads
as `3e-12` relative *to that part alone*.  The max-norm real/imaginary columns, which normalise on
the largest entry of the same part, never exceed `4.3e-15`.

**f = 1 + 0.1i** (same offsets, `out/cub32c_tab.txt`)

    max-norm  band   b-g       b-c       g-c       b-Gila    g-Gila    c-Gila
    3              3.06e-14  4.21e-14  1.15e-14   2.25e-13  2.26e-13  2.26e-13
    4              6.11e-15  5.62e-15  4.83e-15   3.48e-13  3.48e-13  3.48e-13
    5-6            5.05e-15  8.17e-15  6.40e-15   3.78e-13  3.78e-13  3.77e-13
    7-8            2.90e-15  2.21e-15  2.38e-15   4.16e-13  4.15e-13  4.16e-13
    9-16           2.90e-15  2.17e-15  3.24e-15   4.35e-13  4.36e-13  4.36e-13
    17-23          2.64e-15  1.41e-15  2.91e-15   2.53e-13  2.53e-13  2.53e-13
    real part max-norm, worst band value 4.16e-14 (b-c, band 3); imaginary 6.00e-14 (b-c, band 3).
    per-entry worst over all bands: b-g 4.07e-14, b-c 4.21e-14, g-c 1.15e-14.

**Offsets with a pairwise difference above 1e-13: zero, at both frequencies, in every band**
(`out/cub32r_tab.txt`, `out/cub32c_tab.txt`, section "disagreements", `count 0`).

Exclusions at 1/32 (identical at both frequencies, `out/cub32r_excl.txt`):
famB none; famG 18 offsets (`q > 60`): the classes `(3,0,0)`, `(3,1,0)`, `(3,2,0)`, `(3,1,1)`
(`rho > 0.4642`); famC 3 offsets, the class `(3,1,1)`, where `tauXi = 1` exactly on all three axes
(the same geometric identity famC's report identifies at 2 cells).  famG *was* evaluated on its
18 offsets because its own script caps at 96, not 60, and `q <= 80` there; that is what makes
`nOff = 37` in band 3 for `b-g`.  `b-c` and `g-c` in band 3 use 19 offsets — see section 6 for the
15 offsets famC returns `Inf/NaN` on.

## 3. Cubic cell 1/4, f = 1 — the three do NOT agree

    max-norm, |X-Y|/max|Y|                       (out/cub4r_tab.txt)
    band        b-g       b-c       g-c       b-Gila    g-Gila    c-Gila     nOff
    3        6.21e-15  6.34e-15  3.28e-15   1.23e-14  1.10e-14  1.07e-14      37
    4        3.38e-15  1.19e-15  2.96e-15   8.72e-15  9.05e-15  8.84e-15      61
    5-6      4.84e-15  1.65e-15  4.31e-15   1.78e-14  1.84e-14  1.78e-14     218
    7-8      1.96e-14  1.95e-14  4.38e-15   2.04e-14  1.07e-14  1.02e-14     386
    9-16     2.36e-12  2.49e-12  1.13e-12   2.36e-12  1.37e-14  1.12e-12    4184
    17-23    6.02e-10  6.05e-10  1.15e-10   6.02e-10  1.49e-14  1.15e-10    8911

    per entry (>1e-4 of the largest)             real part max-norm / imaginary part max-norm
    9-16     1.70e-11  1.70e-11  5.45e-12     |  1.29e-11 1.69e-11 1.05e-11 / 1.89e-11 3.70e-11 3.60e-11
    17-23    2.15e-9   2.19e-9   1.41e-10     |  2.53e-8  2.44e-8  2.13e-9  / 1.25e-8  1.22e-8  2.54e-9

**10597 of 13797 offsets disagree by more than 1e-13** (counts per band, `work/overlap/cnt.jl`):

    band     b-g>1e-13   b-c>1e-13   g-c>1e-13
    3-8         0/702       0/699       0/699
    9-16     1665/4184   1770/4184    505/4184
    17-23    8827/8911   8827/8911   6765/8911

The onset is sharp at max-norm 18 (worst `b-g` by max-norm shell: 3.33e-12 at 17, 5.22e-10 at 18),
which is where famB's `L` drops from 12 to 10.  `g-Gila` stays at `1.5e-14` throughout, so famG and
Gila are the pair that agrees; (b) and (c) are the outliers.  Worst offsets: `(16,16,22)` and its
permutations, `b-c = 6.05e-10`.  The complete list of the 10597 offsets is the "disagreements"
section of `out/cub4r_tab.txt`.

## 4. Slender cell (1/32, 1/32, 1/512): the excluded sets

All 10984 octant offsets `0<=n1,n2<=12`, `0<=n3<=64` (excluding the origin), f = 1.
Every excluded set is **contiguous in n3 for each (n1,n2)**, and only `(n1,n2)` with
`max(n1,n2) <= 3` is affected at all (`out/slender_excl.txt`):

    excluded n3 (from 0, inclusive) for each (n1,n2)
    (n1,n2)   famB  rho>=1 or L>40   famG  q*>60    famC  tauXi>=1     union
    (0,0)          1..35   (35)         1..48 (48)     1..17 (17)      1..48
    (0,1),(1,0)    0..31   (32)         0..45 (46)     0..17 (18)      0..45
    (0,2),(2,0)    0..15   (16)         0..35 (36)     0..5  (6)       0..35
    (0,3),(3,0)    -                    0..2  (3)      -               0..2
    (1,1)          0..27   (28)         0..42 (43)     0..25 (26)      0..42
    (1,2),(2,1)    -                    0..32 (33)     0..19 (20)      0..32
    (2,2)          -                    0..16 (17)     0..11 (12)      0..16
    totals          159                    344            143            344

famG's set contains both of the others, so the union is famG's: **the band the near field must
cover is exactly `q* > 60`, i.e. `rho > 0.4642`, 344 of 10984 offsets (3.1%)**.  Within that band
the ordering is not uniform: famC reaches much further in on the pure needle (`(0,0,n3)` from
n3=18 against famB's 36 and famG's 49) but *less* far in on `(1,1,n3)` (from 26 against famB's 28
… famB is better there) — famB and famC are not nested.

An additional class, not covered by any author's criterion: **famC returns `Inf/NaN` on 57 slender
offsets and 15 cubic-1/32 offsets that its own `tauXi < 1` test admits.**  `gTab` seeds
`g[l,0] = e^{ikR0} R0^{-l}` for `l` up to `L = 5+2M+2Q`, and `R0 < 1` in wavelengths, so the seed
overflows Float64 once `L*log10(1/R0) > 308`.  Measured: `(0,1,3)` cubic 1/32, `M=145, Q=30,
L=355, R0=0.0988, L*log10(1/R0) = 357`; slender `(0,0,18)`, `M=321, Q=275, L=1197, R0=0.0352,
L*log10(1/R0) = 1740`.  Affected slender offsets: `(0,0) n3 18..25`, `(0,1)/(1,0) 18..22`,
`(0,2)/(2,0) 6..12`, `(1,1) 26..30`, `(1,2)/(2,1) 20..24`, `(2,2) 12..21`.  Affected cubic-1/32
offsets: exactly the 15 members of the classes `(3,1,0)` (`M=145`), `(3,2,1)` (`M=274`) and
`(3,2,2)` (`M=177`).
This is an implementation defect (the fix is to factor `R0^{-l}` out of the table), not a
convergence failure.

## 5. Slender cell: pairwise tables on the offsets each pair admits

    max-norm, |X-Y|/max|Y|                        (out/slender_tab.txt)
    band(max|n|)  b-g       b-c       g-c       b-Gila    g-Gila    c-Gila     nOff
    <=4        2.05e-13  2.06e-13  2.38e-14   7.24e-11  3.91e-11  3.91e-11      74
    5-8        2.00e-13  1.95e-13  1.20e-14   5.58e-08  1.09e-09  1.09e-09     568
    9-16       1.99e-13  5.44e-12  8.31e-15   5.04e-08  1.53e-09  5.87e-08    2072
    17-32      6.87e-13  1.05e-11  4.65e-15   1.36e-06  1.60e-07  1.90e-06    2576
    33-48      4.66e-13  1.12e-11  4.74e-15   5.50e-07  1.60e-07  7.65e-07    2646
    49-64      4.63e-13  4.63e-13  5.33e-15   6.46e-08  6.46e-08  6.46e-08    2704

    per entry (>1e-4 of the largest)
    <=4        8.33e-13  8.33e-13  2.38e-14   3.48e-10  8.86e-11  8.86e-11
    5-8        6.98e-13  6.99e-13  1.20e-14   1.01e-07  7.04e-09  7.04e-09
    9-16       5.24e-13  1.84e-11  8.31e-15   1.47e-07  4.50e-09  1.02e-07
    17-32      3.52e-12  2.58e-11  5.31e-15   1.14e-05  5.89e-07  1.14e-05
    33-48      1.33e-12  1.47e-11  6.87e-15   1.63e-06  4.39e-07  1.83e-06
    49-64      1.00e-12  1.00e-12  5.33e-15   1.76e-07  1.76e-07  1.76e-07

    real part max-norm                          imaginary part max-norm
    <=4        2.05e-13  2.07e-13  2.39e-14  |  1.87e-15  1.44e-14  1.30e-14
    5-8        2.01e-13  1.96e-13  1.23e-14  |  2.26e-15  1.20e-14  1.24e-14
    9-16       1.99e-13  5.45e-12  8.73e-15  |  1.99e-15  1.37e-14  1.28e-14
    17-32      6.87e-13  1.05e-11  4.62e-15  |  1.85e-15  2.08e-14  9.06e-15
    33-48      4.68e-13  1.12e-11  4.73e-15  |  1.77e-15  9.58e-15  6.98e-15
    49-64      4.64e-13  4.65e-13  5.34e-15  |  2.03e-15  7.13e-15  7.59e-15

**`g-c` never exceeds 5.4e-14 anywhere on the slender cell**, in every band, per entry, and in each
part separately.  `b-g` and `b-c` exceed `1e-13` on 275 offsets (counts per band 10, 8, 13, 36, 62,
36 for `b-g`), always with `g-c` at `1e-15`: famB is the sole outlier every time.  The 275 offsets
are listed in `out/slender_tab.txt`; by `(n1,n2)` class they occupy `n3` ranges
`(0,0) 36..63`, `(0,1)/(1,0) 32..60`, `(0,2)/(2,0) 16..53`, `(1,1) 31..52`, `(1,2)/(2,1) 25..50`,
`(1,3)/(3,1) 0..34`, `(2,2) 17..43`, `(2,3)/(3,2) 20..39`, `(3,3) 11..20`, `(0,3)/(3,0) 15`.
Worst: `(0,0,36)` `b-c = 1.12e-11`, then `(0,1,32)` and `(1,0,32)` at `1.05e-11`.

## 6. Adjudication against the 220-bit reference

Cubic references: the 36 `pairKer` face pairs at 220 bits assembled by the BRIEF's `srfSum!` signs
(`out/adj_cub4.txt`, `out/adj_cub32.txt`).  Slender references: the BigFloat Gauss-Legendre volume
integral of `famD/refQuad.jl`, `uniCut(s,2)` cuts, orders 34 and 46; **the two orders agree to
2.2e-63 … 7.9e-65**, so the reference is exact at every level discussed (`out/adj_sld.txt`).
Errors are `max|X - ref| / max|ref|`.

    cubic 1/4, f = 1
    D            rho     | L  errB(L)   L=34 errB | q  errG(q)  q+16 errG | M  Q  errC   M+16,Q+16 | Gila
    (23,23,20) 0.04536   | 10 5.51e-10  4.11e-16  | 22 1.14e-15  1.14e-15 | 12 17 4.01e-11  7.67e-16 | 4.84e-15
    (23,23,23) 0.04348   | 10 4.66e-10  3.37e-15  | 22 4.18e-15  4.18e-15 | 12 17 5.85e-11  3.63e-15 | 3.78e-15
    (16,16,16) 0.0625    | 12 9.21e-13  6.02e-15  | 24 4.91e-15  4.91e-15 | 14 16 1.03e-12  4.53e-15 | 8.02e-15
    (12,5,9)   0.1095    | 14 1.02e-14  5.65e-15  | 26 3.75e-15  3.75e-15 | 16 16 3.66e-15  3.20e-15 | 2.84e-15
    (8,8,8)    0.125     | 14 1.45e-14  2.70e-15  | 28 2.89e-15  2.89e-15 | 20 16 2.04e-15  2.12e-15 | 7.03e-15

    which index of famC is at fault: at (23,23,20) M+16 alone gives 7.67e-16 while Q+16 alone
    leaves 4.01e-11 — it is the xi index M, not Q.  Same at (23,23,23) and (16,16,16).

    slender (1/32,1/32,1/512), f = 1
    D           rho    | L  errB(L)  L=70 errB | q  errG(q)  q+16 | M  Q  errC   M+16,Q+16 | Gila     | ref floor
    (0,0,36)  0.6292   | 40 1.12e-11  2.59e-16 | 94 5.48e-16 5.48e-16| 24 41 2.12e-16 5.89e-16| 4.74e-07 | 2.2e-63
    (0,0,37)  0.6121   | 40 3.74e-12  3.67e-16 | 88 1.65e-15 1.65e-15| 23 40 3.62e-16 8.41e-16| 4.04e-07 | 7.5e-64
    (3,3,11)  0.3294   | 22 1.98e-13  3.78e-16 | 44 6.35e-16 6.35e-16| 51 24 6.47e-16 3.64e-16| 6.03e-11 | 2.2e-64
    (3,1,8)   0.4422   | 28 1.73e-13  4.98e-16 | 56 2.95e-15 2.95e-15| 60 29 2.44e-15 7.90e-16| 8.26e-10 | 8.1e-64
    (3,2,20)  0.3710   | 24 2.00e-13  5.87e-16 | 48 1.25e-15 1.25e-15| 53 26 8.45e-16 7.03e-16| 1.94e-08 | 7.9e-65

    cubic 1/32, f = 1, the largest disagreement in the whole 1/32 sweep
    (0,0,3)   0.5774   | 36 3.06e-14  L=46: 2.01e-16 | 80 5.93e-15 | 47 31 9.91e-15 (Q+16 alone: 5.85e-16) | Gila 6.07e-14

**Verdict of the adjudication.  In every case where two of the three disagree above 1e-13, the
method at fault is the one whose truncation index was set too low by its own published rule, and
raising that index alone brings it to the reference at `3e-16 … 6e-15`.  No implementation error,
no normalisation error and no sign error was found in any of the three.**

    who is wrong, and by how much
    cubic 1/4, >= 9 cells   famB (L rule)   up to 5.5e-10 at (23,23,20), 6.05e-10 at (16,16,22)
                            famC (M rule)   up to 5.9e-11 at (23,23,23), 1.15e-10 max over the sweep
                            famG            correct: 1.1e-15 … 4.9e-15 at the same offsets
    slender, 275 offsets    famB (L rule)   up to 1.12e-11 at (0,0,36)
                            famG, famC      correct: 2.1e-16 … 3.0e-15 at the same offsets
    cubic 1/32              famB 3.06e-14 and famC 9.91e-15 at (0,0,3), both below the 1e-13 threshold

## 7. What the rules give against what is actually needed for 1e-14

`Lneed`/`qneed`/`Mneed` = smallest truncation from which the error against the 220-bit reference
**stays** below `1e-14` (downward scan; `work/overlap/needL.jl`, `out/need_*.txt`).  `Mneed` is
measured with `Q` raised by 24 so it isolates the `xi` index.

    case        D            rho     kR     Lrule Lneed | qrule qneed | Mrule Mneed  (Q rule)
    cube 1/32  (0,0,3)     0.5774   0.59      36   38   |   80   36   |   47   27     (31)
    cube 1/32  (3,3,3)     0.3333   1.02      24   22   |   44   20   |   72   46     (21)
    cube 1/32  (0,0,8)     0.2165   1.57      18   16   |   34   14   |   11    9     (17)
    cube 1/4   (23,23,20)  0.04536 59.98      10   16   |   22   18   |   12   16     (17)
    cube 1/4   (16,16,16)  0.0625  43.53      12   16   |   24   18   |   14   16     (16)
    cube 1/4   (8,8,8)     0.125   21.77      14   16   |   28   18   |   20   18     (16)
    slender    (0,0,36)    0.6292   0.44      40   54   |   94   52   |   24   20     (41)
    slender    (3,3,11)    0.3294   0.84      22   26   |   44   24   |   51   40     (24)
    slender    (3,1,8)     0.4422   0.63      28   30   |   56   30   |   60   42     (29)
    slender    (8,0,0)     0.1769   1.57      16   16   |   32   14   |    9    8     (17)

Three statements follow, each a correction to a published rule:

- **famB's `L = 13/log10(1/(0.7 rho))` is a `rho`-only rule and has no `kR` floor.**  At `lambda/4`
  it under-selects by 6, 4 and 2 levels at 23, 16 and 8 cells; the needed value is a flat
  `L = 16` over the whole `rho` range `0.045 … 0.125`, which is exactly the `L13 = 16` floor that
  famB's own section 3.4 `lambda/4` table reports and that its summary rule does not carry.
  It is also short by 2-14 levels at moderate `rho` (`38` needed vs `36` at `rho = 0.577`;
  `54` vs `40` at `rho = 0.629`).  The a priori bound `remBndAll` is the rule that does work at
  `lambda/4` (famB's section 11 used it), and it was not what its summary formula says.
- **famC's `M = 13/log10(1/tauXi)` also has no `kR` term**, and at `kR = 44 … 60` it under-selects
  by 2-4 levels (`16` needed vs `12` at `(23,23,20)`).  Its `Q` rule already carries a `kR/8` term
  and was *not* the failing index there; at `lambda/32` `(0,0,3)`, `kR = 0.59`, the failing index
  is `Q` instead (raising `Q` alone takes `9.91e-15` to `5.85e-16`).  Both indices are marginal at
  their own rule, in opposite regimes.
- **famG's `q = 17/log10(1/rho) + 8` is conservative everywhere measured** (`qneed <= qrule` in all
  ten rows, by 2 at `lambda/4` and by up to 44 at `lambda/32` 3 cells).  That margin is why famG is
  the method that never fails in this study, and it is also why it is the most expensive of the
  three at small separations.

## 8. Random signed offsets (sign handling)

30 offsets drawn uniformly from `[-40,40]^3` with `max|n_i| >= 3`, fixed seed, cubic 1/32,
both frequencies (`work/overlap/rnd.jl`, `out/rnd.txt`).  `Xr` is Gila's octant value at `|n|`
mapped by `G_ab(sigma R) = sigma_a sigma_b G_ab(R)`.

    f            max over the 30:  b-g       b-c       g-c       b-Xr
    1.0                          1.18e-15  1.20e-15  1.55e-15  1.96e-13
    1.0 + 0.1i                   1.27e-15  8.21e-16  1.32e-15  1.82e-13

Every one of the 60 (offset, frequency) cases is at `<= 1.6e-15` for all three pairs.  All three
implementations handle negative components identically and correctly, and all three reproduce the
axis-reflection rule that Gila's assembly uses.  `rho` on these offsets ranges 0.032-0.090, so this
tests sign bookkeeping, not truncation.

## 9. Gila's fixed Gauss-Legendre rule, as measured by this sweep

The last three columns of every table are a by-product, and they are not uniform:

    shape / frequency          max-norm |X - Gila|/max|Gila|, worst band     against the 220-bit ref
    cube 1/32, f = 1           5.58e-13 (5-6 cells)                          6.07e-14 at (0,0,3)
    cube 1/32, f = 1+0.1i      4.36e-13 (9-16 cells)                         -
    cube 1/4,  f = 1           2.04e-14 (7-8 cells), 1.5e-14 vs famG at 23   2.8e-15 … 8.0e-15
    slender,   f = 1           1.36e-06 (17-32 cells)                        4.7e-07 at (0,0,36)

So Gila is at `3e-13` … `6e-13` at `lambda/32` (the level the BRIEF quotes), **better than `1e-14`
at `lambda/4`** (its `quadOrd` floor is order 6 there, so the fine cells are the *harder* case for
it, not the coarse ones), and **catastrophically wrong on the slender cell**: `4.7e-7` at
`(0,0,36)`, `1.9e-8` at `(3,2,20)`, `8.3e-10` at `(3,1,8)`, `6.0e-11` at `(3,3,11)`, all verified
against the 220-bit reference.  Per entry the slender `X-Gila` reaches `1.1e-5`.  The slender cell
is not a case where "Gila is good to 1e-10 to 1e-13"; the far-field expansions are five to seven
orders better there than the code they are meant to replace.

## 10. Verdict

Where all three converge by their authors' own criteria and their truncations are actually
sufficient, the three implementations agree to Float64 rounding and nothing more: over the 13797
octant offsets of the cubic `lambda/32` cell at `f = 1` and `f = 1+0.1i` the worst pairwise
difference is `4.2e-14` (max-norm) and `4.2e-14` (per entry), attained only in the 3-cell band, and
from 4 cells outward the max-norm difference never exceeds `9.8e-15` and the per-entry difference
`2.4e-14`; over the 30 random signed offsets it is `1.6e-15`;
on the slender cell `g-c` never exceeds `5.4e-14` in any band, any entry or either part.  That is a
free end-to-end confirmation of the geometry bookkeeping, the `srfSum!` sign convention, the
normalisation `T = +(1/V_t) int w (d_a d_b + dab k^2) g`, the exact weight moments, famB's rational
solid-harmonic tables, famG's `(G,S)` recurrence, and famC's `xi` resummation, all against each
other and against the 220-bit reference at `3e-16 … 6e-15`.  What does **not** agree is the choice
of truncation: famB's `rho`-only `L` rule and famC's `tauXi`-only `M` rule both lack the `kR`
dependence they need, and they fail — famB by up to `1.1e-11` on the slender cell and `6.0e-10` on
the `lambda/4` cube, famC by up to `1.2e-10` on the `lambda/4` cube — on 10597 of 13797 `lambda/4`
offsets and 275 of 10893 slender offsets, in every case adjudicated as pure truncation, corrected
by raising the offending index alone.  famG's rule is the only one of the three that was never
short.  So: the three *expansions* agree to Float64 rounding everywhere all three converge; the
three *published truncation rules* do not, and two of them must acquire a `kR` term before any of
this can be used as an acceptance test for a build.

## Files

    work/overlap/common.jl      the three tensor(D,s,f) wrappers, the three rules, the Gila call
    work/overlap/sweep.jl       one sweep per (shape, frequency); writes out/<tag>.bin and _excl.txt
    work/overlap/analyze.jl     band tables (max-norm, per entry, real, imaginary) -> out/<tag>_tab.txt
    work/overlap/cnt.jl         counts of offsets above 1e-13 per band
    work/overlap/nan.jl, nan2.jl  the famC Inf/NaN class and its overflow diagnosis
    work/overlap/adjudicate.jl  220-bit adjudication -> out/adj_{cub32,cub4,sld}.txt
    work/overlap/needL.jl       rule vs needed truncation -> out/need_{cub32,cub4,sld}.txt
    work/overlap/rnd.jl         30 random signed offsets -> out/rnd.txt
    work/overlap/smoke.jl       the (5,1,0) sign/normalisation check
    work/overlap/out/*.bin      all four tensors for every offset of every sweep (B,G,C,Gila)
