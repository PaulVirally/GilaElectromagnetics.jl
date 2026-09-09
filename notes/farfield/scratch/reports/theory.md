# theory: the five LaTeX fragments, their bound checks, and the inconsistencies found

Work dir: `SCRATCH/work/theory/`.  SCRATCH = `/Users/pvirally/.julia/dev/GilaElectromagnetics/notes/farfield/scratch`.
All fragments use only the macros of `notes/integrals.tex` and `notes/moments/moments.tex`
(`\dd`, `\ii`, `\code`, `\Gs`, the theorem environments, `booktabs`, `enumitem`); no
`\documentclass`, no figure inputs, no dependence on any other agent's library.

## 1. Files

### Fragments (to be `\input` into `notes/farfield/farfield.tex`)
| file | lines | contents |
|---|---|---|
| `work/theory/volume.tex` | 398 | the volume formulation: `srfScl` -> `srfMat = I_FF'/V_t` (Lemma, proved from `egoSrfFxd!` + `cubFac` + `srfScl`), the divergence identity in both directions, `srfSum!` sign-by-sign against it, the difference box and `mu_n`, Theorem (the boxed volume form), the 4pi^2 / 1/k^2 statement, the exact-rational check, the 18-row sign-pinning table, the assembly amplification |
| `work/theory/expansion.tex` | 422 | addition theorem with `(-1)^l` and the convention; real solid harmonics as integer polynomials with the `C_l^m` recurrence; `c_ln`; derivatives on the regular side (evaluation) and on the singular side (bound); exact moments; parity selection of the four (l,m) classes; the l=2 remark; the Helmholtz identity term by term; octant sub-boxes with `nu_n` and the no-straddle lemma; the reflection rule; the Re/Im split for real f |
| `work/theory/bounds.tex` | 549 | `est(R)`; Theorems (a) j_l, (b) h_l, (c) l-truncation with W_l, (d) n-truncation, (e) k-series, the sub-box version; Proposition on the exact radius `|R|/|delta|` and the remark separating rho<1 from sqrt2 rho<1; a section on what is NOT proven |
| `work/theory/join.tex` | 312 | cost model, the selection rule, the band table for the four shapes, the 12 two-cell offsets, the slender enumeration, the rho tables and the fixed-size slow band |
| `work/theory/registry.tex` | 301 | families (a)-(g), (k) in the style of moments.tex Sec. 9, plus a subsection on three disagreements between rounds and their resolution |

### Test document and check scripts
`work/theory/test.tex` (same preamble as `moments.tex` minus `figs.tex`/`figgeo.tex`),
`test.pdf`.
`sph.jl` (shared special functions), `chkElem.jl` -> `out_elem.txt`,
`chk17.jl` -> `out_c17.txt`, `chkRem.jl` + `chkRem_fns.jl` -> `out_rem.txt`,
`chkKser.jl` -> `out_kser.txt`, `chkKserSl.jl` -> `out_kser_sl.txt`,
`chkRay.jl` -> `out_ray.txt`, `bands.jl` -> `out_bands.txt`,
`bands2.jl` -> `out_bands2.txt`.
Run as `JULIA_NUM_THREADS=1 julia --startup-file=no --project=SCRATCH/env <script>`.
Never more than two of my own Julia processes at a time.

## 2. pdflatex

`pdflatex -interaction=nonstopmode test.tex`, three passes, in `/Library/TeX/texbin`.

    exit status 0
    Output written on test.pdf (28 pages, 543910 bytes)
    errors                    0
    undefined references      0   (grep "LaTeX Warning: Reference" -> 0 hits)
    undefined citations       0
    Overfull \hbox            0
    Underfull \hbox/\vbox     0
    package/font warnings     0

Four defects were found and fixed inside the fragments (they are gone from the log above):
three Overfull \hbox of 29.1 pt (the boxed volume form), 16.0 pt (the Helmholtz identity)
and 68.6 pt (the list of twelve offsets), all repaired by breaking the display; and one
`Font shape OMS/cmtt/m/n undefined` from `\code{Rational\{BigInt\}}` -- the same construct
appears in `moments.tex`, so it is inherited, but I reworded it away.  Five hyperref
"Token not allowed in a PDF string" warnings from math in (sub)section titles were removed
with `\texorpdfstring`.  Pages 4-5, 16-17 and 21-22 of the PDF were inspected as rendered
images; the boxed equations, the exact-rational fraction table and the 43-row band table
all set correctly.

## 3. Bound checks: bound vs truth, in BigFloat

Every row below was produced by the script named; the assertion `bound >= truth` is inside
each loop, so a violation would abort the run.  None aborted.

### (a) |j_l(z)| <= |z|^l e^{|z|^2/(4l+6)}/(2l+1)!!   (`chkElem.jl`, 400 bit)
64 rows: l in {0,1,2,5,10,20,40,80} x z in {0.3, 1, 3, 10, 30, 3+0.3i, 10+3i, 30+30i}.
No violation.  bound/|j_l| in [1.0011, 1.5e119].  In the regime family (b) uses it
(|z| = |k| r_d <= 2.72):

    |z|=0.3 : 1.0305 (l=0), 1.0039 (10), 1.0011 (40), 1.00055 (80)
    |z|=1.0 : 1.4039 (l=0), 1.0445 (10), 1.0121 (40), 1.0062  (80)
    |z|=3.0 : 95.27  (l=0, j_0 near its zero at pi), 1.481 (10), 1.115 (40), 1.057 (80)

The 1e119 outliers are all |z|^2 >> l (z = 30+30i, l = 0), a regime the expansion never
enters.

### (b) |h_l(z)| <= (e^{-Im z}/|z|) sum_s (l+s)!/(s!(l-s)!(2|z|)^s)   (`chkElem.jl`)
72 rows, same l list, z in {0.3, 1, 3, 10, 30, 100, 3+0.3i, 10+3i, 30+30i}.
No violation.  bound/|h_l| in [1.0, 1.8e13]; equality at l = 0 (both sides are
e^{-Im z}/|z|).  Tight for |z| << l (1.349 at z=0.3, l=80), loose for |z| ~ l
(1.17e4 at z=10, l=80; 3.01e3 at z=100, l=40).  Monotonicity of hb in l, which the proof
of (c) uses, holds term by term.

### (c) the l-truncation bound with W_l   (`chkRem.jl`, 320 bit; `chk17.jl`)
The truth is `max_ab |T_ab - T_ab^(L)|` with T_ab from an independent BigFloat tensor
Gauss-Legendre rule on the 8 octants of D at order 26 (ord 20 vs 26 agrees to 1.3e-44
(c32 (4,0,0)), 2.3e-37 ((3,1,0)), 1.0e-66 ((8,8,8)), 4.3e-45 (c4 (4,0,0)), 1.3e-43 (sl)),
and T^(L) from the derivatives-on-the-singular-side form with I_lm by an order-14 octant
rule and (da db + dab k^2)(h_l Y_lm) by 320-bit central differences at step 1e-25.
79 rows, no violation.  bound/truth in [1.3e3, 3.0e7]:

    shape f        n         rho    kR     L    max|Rem_L|   bound      bnd/true
    c32   1        (4,0,0)   0.433  0.785   8   1.62e-9      5.13e-5    3.17e4
    c32   1        (4,0,0)   0.433  0.785  16   2.53e-13     1.39e-8    5.50e4
    c32   1        (4,0,0)   0.433  0.785  24   2.42e-17     5.36e-12   2.21e5
    c32   1+0.1i   (4,0,0)   0.433  0.789  16   2.50e-13     1.28e-8    5.10e4
    c32   1        (3,1,0)   0.548  0.621  16   1.16e-11     1.82e-6    1.57e5
    c32   1        (3,1,0)   0.548  0.621  24   7.09e-14     4.63e-9    6.54e4
    c32   1        (8,8,8)   0.125  2.721   8   1.29e-16     2.63e-11   2.04e5
    c32   1        (8,8,8)   0.125  2.721  16   4.23e-25     3.60e-19   8.51e5
    c4    1        (4,0,0)   0.433  6.283   8   4.19e-9      6.59e-3    1.57e6
    c4    1        (4,0,0)   0.433  6.283  16   4.17e-13     2.29e-6    5.50e6
    c4    1        (4,0,0)   0.433  6.283  24   3.31e-17     9.87e-10   2.99e7
    sl    1        (4,0,0)   0.354  0.785   8   4.44e-11     8.46e-7    1.91e4
    sl    1        (4,0,0)   0.354  0.785  16   1.62e-15     9.63e-11   5.93e4
    sl    1        (4,0,0)   0.354  0.785  24   1.49e-19     1.41e-14   9.47e4

famB reports 1.8e2-4e6 for the same quantity; my set includes lambda/4 at kR = 6.28, where
the hb majorant is at its loosest, hence the 3e7 top.  In L this is a 1.2-1.5x
over-selection.

The scale: est(R)/max_ab|T_ab| = 2.36, 2.26, 2.59, 3.77, 1.94, 2.29 on the six cases
(famBsub's "2.0 to 5.1" confirmed).

The two structural constants (`chk17.jl`, 320 bit):
- `sum_m |Y_lm(u)| <= (2l+1)/sqrt(4pi)`: 32 rows (4 directions x l in {0,1,2,4,8,20,40,60}),
  no violation, ratio 1.0 at l = 0 and then a direction-dependent constant --
  0.4869 (axis), 0.7175 (body diagonal), 0.6541 (generic) at l = 60.  Costs a factor <= 2,
  not a factor growing with l.
- `|(da db + dab k^2)(h_l Y_lm)(R)| <= 17 |k|^2 hb(l+2,|kR|) sqrt((2l+5)/4pi)`: 72 rows
  (n in {(2,0,0),(4,0,0),(3,1,0),(16,16,16)} at lambda/32 and {(4,0,0),(32,0,0)} at
  lambda/4, f = 1 and 1+0.1i, l in {0,2,4,8,16,24}, all 9 (a,b), all m), no violation,
  worst LHS/RHS = 3.145e-2, best 2.943e-5.  **The constant 17 is loose by 32x to 34000x**
  and is where most of the over-selection of the composite bound comes from.
- W_l = int_D w |delta|^l: the multinomial form against a 24-point BigFloat octant Gauss
  rule, l in {0,2,4,8,16,32}, cubic and slender: relative difference 1.2e-120 to 5.6e-119
  (i.e. exact).  W_0 = V_t^2 and 8 x (octant W_0) = V_t^2 exactly, checked for all four
  shapes in `bands.jl`.

### (d) the n-truncation, |j_l(z) - sum_{n<=N}| <= (|z|^l/(2l+1)!!) y^{N+1}/(N+1)! e^y,
     y = |z|^2/(2(2l+3))   (`chkElem.jl`)
48 rows: |z| in {0.34, 1.36, 2.72, 0.278} (the four physical |k| r_d) x l in {0,4,12,24}
x N in {2,4,6}.  No violation.  bound/truth in [1.122, 3268]; for l >= 12 it is
1.12-4.49 at every |z|, i.e. tight where it is used.  The 3268 is l = 0, N = 6, where
j_0 is nearly its own leading term.

### (e) the k-series bound   (`chkKser.jl`, `chkKserSl.jl`, 300 bit)
Truth = tail of the same series summed to mMax (60 for the cube, 36 for the slender cell),
moments from `notes/moments/moments.jl`, 8 face pairs per offset.  44 rows, no violation.

    shape n         x=|k|Dmax  N: 8     12     16     20     24      max_n |I_{n-1}|/(Dmax^n I_{-1})
    c32   (2,0,0)   0.6512     4.44   8.05   14.4   25.2   43.2     0.0802
    c32   (2,0,0)   0.6545*    4.46   8.07   14.4   25.3   43.3     0.0802
    c32   (3,1,0)   0.8781     4.07   6.83   10.3   14.7   20.5     0.1109
    c32   (3,1,0)   0.8825*    4.10   6.87   10.3   14.8   20.5     0.1109
    sl    (0,0,2)   0.2801     44.7   95.7  169.3  265.7    -       0.0198
    sl    (0,0,2)   0.2815*    44.8   95.9  169.5  266.0    -       0.0198
    sl    (0,0,8)   0.2988     28.5   63.3  117.2  192.6    -       0.0273
    sl    (0,0,8)   0.3003*    28.6   63.4  117.4  192.8    -       0.0273
    sl    (1,1,17)  0.5933      7.48  13.4   21.8   33.3    -       0.0595
    sl    (1,1,17)  0.5962*     7.51  13.4   21.9   33.4    -       0.0595
    (* = f = 1 + 0.1i; otherwise f = 1)

The pointwise moment inequality |I_{n-1}| <= Dmax^n I_{-1}, which is the whole content of
the proof, is satisfied with 10x-50x margin at every n <= mMax on every pair tested.

### (f) the two radii   (`chkRay.jl`, 300 bit)
Exact singularity radius: the roots of (R + t d).(R + t d) = 0 satisfy
|t_+| = |t_-| = |R|/|d| to a relative error <= 8.5e-91 at c32 (2,0,0), (2,2,0), (3,0,0),
c4 (2,0,0) and sl (0,0,16).  Cauchy on |t| = beta < 1/rho with M = max|phi|, coefficients
extracted on |t| = 0.9 with 512 points (alias (0.9/beta)^512 <= 1e-45):

    case          rho     beta    M          q     |a_q|      M beta^-q   |partial - phi(1)|/|phi(1)|
    c32 (2,0,0)   0.866   1.097   3.529e5     8    1418.4     1.683e5     5.949e-1
    c32 (2,0,0)                              16    1616.2     8.026e4     8.972e0
    c32 (2,0,0)                              32     229.6     1.826e4     1.879e0
    c32 (2,0,0)                              64       6.434   9.446e2     5.051e-3
    c32 (2,0,0)                             128       3.665e-3 2.529e0    1.754e-5
    c32 (2,2,0)   0.6124  1.1     6.568e2    64       2.463e-10 1.473e0   4.696e-13
    c32 (2,2,0)                             128       4.737e-23 3.305e-3  6.910e-25
    c32 (3,0,0)   0.5774  1.1     9.984e2   128       3.136e-26 5.024e-3  1.984e-28
    c4  (2,0,0)   0.866   1.097   7.906e2   128       7.419e-6  5.666e-3  3.269e-6
    sl  (0,0,16)  1.4156   --      --        --        --        --       radius 0.7064 < 1: DIVERGES

The observed ratio (|a_128|/|a_64|)^{1/64} = 0.887 at (2,0,0), tending to rho = 0.866.

### (g) not proven
Listed as measurements in `bounds.tex` Sec. "What is not proven": the exact-rational
geometry table (cancellation 2.1e5 at L=20, 5.8e8 at L=40 for cubes, 1.1e7 slender --
8.8 digits, which is why it is not built in Float64); the per-offset sums (l-sum
sum|t_l|/|T| <= 8.4 at lambda/32 and <= 67 at lambda/4, growing as 0.67 kR; n-sum
3.07-4.14; sub-box sum 1.0-1.55); the special functions (h_l upward recurrence 3e-15 for
l<=40, |z| in [0.3,300]; Miller j_l 6.8e-15; Y_lm absolute 6.2e-15); the phase seed
(2.15/2.99/3.35 digits at kR = 300/1000/3000, removable for 12 flops); and the end-to-end
accuracy (4.2e-15 over 163 tensors, rounding-limited).  All are cited as measurements.

## 4. Band selection computed from the bounds (`bands.jl`, `bands2.jl`, 256 bit)

Method: whole box (W) if L_wb <= L_max; else octant (O) if max_j L_j <= L_max; else
k-series (K).  L_max = 64.  Budget 1e-13 * est(R); sub-boxes get 1e-13 * est / 8.
Tails summed explicitly to l = 320 (180-220 in the slender sweep) plus a geometric
continuation at the ratio attained there; if that ratio is >= 1 the bound gives no
certificate and the entry is "-1" (this guard matters: without it a divergent term
sequence silently reports a finite L).

    shape dir       n     rho      kR       Lwb  cost_wb   Loct  cost_oct  method
    c32  axis        2  0.866    0.393     174 93936       55  203648    O
    c32  axis        3  0.5774   0.589      50 7830        32  84283     W
    c32  axis        4  0.433    0.785      34 3664        25  57248     W
    c32  axis        8  0.2165   1.571      20 1320        16  27236     W
    c32  axis       32  0.0541   6.283      12 526         10  12450     W
    c32  axis       64  0.0271   12.566     10 390          9  10400     W
    c32  facediag    2  0.6124   0.555      56 9802        39  95971     W
    c32  bodydiag    2  0.5      0.68       40 5040        32  67594     W
    c8   axis        2  0.866    1.571     176 96106       55  206820    O
    c8   axis        3  0.5774   2.356      52 8462        32  86772     W
    c8   axis       32  0.0541   25.133     14 688         12  17134     W
    c4   axis        2  0.866    3.142     182 102764      56  215604    O
    c4   axis        3  0.5774   4.712      54 9120        33  92628     W
    c4   axis       64  0.0271   100.531    16 874         14  22600     W
    c4   facediag    2  0.6124   4.443      60 11240       40  106341    W
    sl   longaxis    2  0.7078   0.393      90 25190       47  154116    O
    sl   longaxis    3  0.4719   0.589      42 5548        29  70375     W
    sl   facediag    2  0.5005   0.555      46 6640        34  76426     W
    sl   shortax     2  11.3248  0.025      -1 -           -1  -         K
    sl   shortax     8  2.8312   0.098      -1 -          207  3489254   K
    sl   shortax    16  1.4156   0.196      -1 -           70  459214    K
    sl   shortax    32  0.7078   0.393      90 25190       34  116833    O
    sl   shortax    64  0.3539   0.785      30 2870        21  45780     W
    (full 43-row table in out_bands.txt and in join.tex)

L is set by rho, not by kR: at a fixed offset in cells L_wb moves by at most 4 from
lambda/32 to lambda/4 while kR changes by 8x.  Beyond 8 cells L = 10-16, cost < 2 us.

Cost model (from famB/famBsub measurements, all taken under load and to be re-measured):
c_W(L) = 3.1 L^2 + 80 ns; c_O({L_j}) = 12.2 sum_j (L_j+1)^2 + 640 ns (7 complex multiplies
per (l,m) instead of 1.7); k-series 32 ms/offset in Float64, ~0.2 s/offset at 128 bits.
Gila today: 13.9 ms/offset at 2 cells, 0.55 ms at 32 cells, 23.3 s for a 32^3 block.

### The 12 two-cell offsets of a cubic grid
`{(2,0,0),(0,2,0),(0,0,2)} u {(2,1,0),(2,0,1),(1,2,0),(0,2,1),(1,0,2),(0,1,2)} u
 {(2,1,1),(1,2,1),(1,1,2)}` -- exactly the offsets with max n_i >= 2 and |n| <= 2.56.

    shape n class    rho     Lwb   Loct  L_j over the 8 sub-boxes
    c32   (2,0,0)    0.8660  174   55    31,31,31,31,55,55,55,55
    c32   (2,1,0)    0.7746  104   55    28,28,31,31,38,38,55,55
    c32   (2,1,1)    0.7071   78   56    26,28,28,32,32,38,38,56
    c8    (2,0,0)    0.8660  176   55    32,32,32,32,55,55,55,55
    c8    (2,1,1)    0.7071   78   56    26,29,29,32,32,39,39,56
    c4    (2,0,0)    0.8660  182   56    33,33,33,33,56,56,56,56
    c4    (2,1,1)    0.7071   82   56    27,30,30,33,33,39,39,56

All twelve have Lwb >= 78 > 64 and max_j L_j <= 56 <= 64, at all three cubic scales; the
next offset out, (2,2,0) with rho = 0.6124, has Lwb = 56/58/60 and is a whole-box offset.
**The bounds reproduce the empirically found split with no tuning.**

### The slender needle
577 non-touching offsets with n1,n2 <= 2 and n3 <= 64 were enumerated.  Those whose
whole-box AND octant bounds need L above the cap:

    cap L > 48 : 83 offsets = (0,0,n) 2<=n<=21 ; (0,1,n),(1,0,n),(1,1,n) 2<=n<=22
    cap L > 64 : 64 offsets = (0,0,n),(0,1,n),(1,0,n),(1,1,n) all with 2<=n<=17

(The task asked for the L > 48 set: **83 offsets**; the rule as I state it uses
L_max = 64, giving **64 offsets**.)  Every offset with n1 = 2 or n2 = 2 is already covered
by W or O, and so is every n3 >= 23 (resp. 18).  On that set |k|Dmax runs from 0.2801 at
(0,0,2) to 0.5977 at (1,1,17), and Theorem (e) with a conservative Lambda = 1e5 gives
N = 13 to 16 terms.  At ~0.2 s per offset in 128-bit arithmetic that is 13 s once per cell
shape (the moments are frequency independent).

## 5. Inconsistencies between reports, and my resolutions

1. **Does the Cartesian series converge at (2,0,0)?**  `famA.md` Sec. 5 says it is a
   divergent asymptotic series that "never becomes a convergent tail"; `famA.md` Sec. 10
   and the registry line say it converges with effective ratio ~0.73, consistent with
   famB's L = 94 -> 6.6e-14.  **Resolved: it converges.**  Along R + t*delta the two roots
   of z.z = 0 are complex conjugates with product |R|^2/|delta|^2 and non-positive
   discriminant, so |t_pm| = |R|/|delta| exactly (verified to 8.5e-91), i.e. the radius is
   1/rho > 1 whenever rho < 1.  The measured partial sums at the worst corner fall to
   1.75e-5 at q = 128 with ratio 0.887 -> 0.866.  The BigFloat sequence famA quotes
   (7.9e-12 at p=72 down to 2.2e-15 at p=120) is itself decreasing; it is slow
   non-monotone convergence plus rounding, not divergence.  What is genuinely unavailable
   is the Cauchy *bound*: the complex ball of radius r_d touches the null cone at
   r_d = |R|/sqrt2, so `sqrt2 rho < 1` is a statement about the bound and `rho < 1` is the
   statement about the series.  Both fragments say this explicitly.  The practical
   consequence (hand (2,0,0) to the octant split) is unchanged, but the distinction
   matters for the slender needle, where rho > 1 and the divergence is real.

2. **The sub-box term count: 48 vs 56.**  `famBsub.md` Sec. 4 records L_bnd = 48 at the
   two-cell offsets; my evaluation of the identical written bound, with the identical
   geometry (rho_j = 0.522 at (2,0,0) in both) and the identical budget tol*est/8, gives
   55-56.  The gap is a factor ~100 in the tail value, i.e. rho_j^7.  **Unresolved.**  It
   moves no band edge (both are below L_max = 64 and above the measured L_13 = 38-42), and
   I record the larger number because it is what the stated formula produces; the
   over-selection factor against measurement is 56/42 = 1.33 here, 48/42 = 1.14 there.
   Recorded as a Remark in `join.tex` and in the registry fragment.

3. **famB's bound/actual range.**  famB reports 1.8e2-4e6; I measure 1.3e3-3.0e7 on my own
   79 rows.  Not a contradiction: my set includes lambda/4 at kR = 6.28, where the hb
   majorant is loosest, and famB's lowest value comes from offsets with large kR that I did
   not repeat.  Same bound, wider sample.

4. **The (-1)^l in famB's evaluation formula.**  `famB.md` Sec. 1.2 establishes that the
   addition theorem written with P_l(Rhat.deltahat) needs (-1)^l, and Sec. 1.5/3.2 then
   drop it because only even l survive the triangle weight.  That is correct for the whole
   box but the *general* statement must carry it (famBsub confirms: omitting it on a
   sub-box is a 3.7e-2 error).  My fragments carry (-1)^l in every displayed formula and
   state where it is moot.

5. **Two different (l,m) selections in famB.**  Sec. 1.4's four surviving classes belong to
   the organization with the derivatives on the *regular* side (the Ad/Ao tables); the
   bound of Sec. 3.2 uses the organization with the derivatives on the *singular* side, in
   which only the diagonal class (l even, m even, cos family) survives in I_lm.  Both are
   right and they are not the same statement.  `expansion.tex` says so explicitly, because
   it is exactly the kind of thing a reimplementation gets wrong.

6. **NEW: for a cubic cell the whole l = 2 shell vanishes identically.**  Not in any
   report.  |delta|^2 Y_20 ~ 2 d3^2 - d1^2 - d2^2 and |delta|^2 Y_22 ~ d1^2 - d2^2, and
   int_D w d_i^2 = mu_2(s_i) prod_{j!=i} mu_0(s_j) is the same number for i = 1,2,3 when
   s1 = s2 = s3, so both integrate to zero.  Measured (`chkRem.jl`): the shells with
   max_m |I_lm| above 1e-40 of the largest are {0,4,6,8,...} for (1/32)^3 and (1/4)^3 and
   {0,2,4,6,...} for (1/32,1/32,1/512).  It is an unclaimed saving on the shapes Gila uses
   most; the term-count tables in the fragments are stated for the generic shape and do not
   exploit it.

7. **famG's q* rule vs auditG's.**  17/log10(1/rho) against 7 + 7.1/log10(1/rho); the audit
   is right (3.35x cost difference) and is a fit, not a bound.  Recorded in the registry
   fragment as a reason to prefer the multipole form, whose L comes from a theorem.

8. **A guard the earlier band scripts may not have.**  Evaluating the tail of the famB
   bound by summing to a finite l cap and subtracting is wrong when the term sequence
   grows: the "tail" hits zero at the cap and the routine reports a finite L for a
   divergent expansion.  With the cap at 60 my first slender sweep reported Lwb = Loct = 60
   for (0,0,2), rho = 11.3.  The fix (require the ratio at the top of the sum to be < 1 and
   add a geometric continuation) turns those into "no certificate", which is the truth.
   Anyone re-deriving the band tables should check for this.

## 6. What I did not do

- No timing of my own: every ns/ms figure in `join.tex` is quoted from famB, famBsub,
  kseries or auditG and is labelled as measured under load and needing re-measurement.
- The `chkKser.jl` sweep on c32 (8,8,8), c4 (2,0,0) and sl (2,0,0) was killed after ~1 h
  (pairMoments at mMax = 60 on 3D perpendicular pairs); the 44 rows reported are c32
  (2,0,0), c32 (3,1,0) and the three slender offsets, both frequencies.
- The est(R) scale is checked on 6 cases here (1.94-3.77) and quoted from famBsub for the
  wider set (2.0-5.1); I did not re-run the wider set.
