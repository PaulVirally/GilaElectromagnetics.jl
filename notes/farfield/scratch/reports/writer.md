# writer — notes/farfield/farfield.tex and farfield.pdf

Deliverables (all under `REPO/notes/farfield/`):

    farfield.tex        the main file: preamble, abstract, S1 "How to use it", then 13 \input's
    farfield.pdf        63 pages, 834 917 bytes, pdflatex three passes, 0 errors, 0 overfull boxes
    tex/volume.tex      \input, copied from SCRATCH/work/theory/volume.tex, unmodified
    tex/expansion.tex   \input, copied from SCRATCH/work/theory/expansion.tex, unmodified
    tex/bounds.tex      \input, from SCRATCH/work/theory/bounds.tex, 3 edits (below)
    tex/bounds2.tex     \input, from SCRATCH/work/bound2/bounds2.tex, 1 edit (below)
    tex/join.tex        \input, from SCRATCH/work/theory/join.tex, 7 edits (below)
    tex/registry.tex    \input, from SCRATCH/work/theory/registry.tex, 4 edits (below)
    tex/groundwork.tex  new (S2)
    tex/certificate.tex new (S7)
    tex/verify.tex      new (S9)
    tex/bench.tex       new (S10)
    tex/defects.tex     new (S12)
    tex/uncertain.tex   new (S13)
    tex/method.tex      new (S14)
    tex/mktab.py        the table generator: tables/*.txt -> tex/tab/*.tex (5 tables, 208 rows)
    tex/tab/*.tex       e_terms (96 rows, longtable), e_route (24), d_ref (14),
                        e_digits (66, longtable), i_cost (8)

Work dir `SCRATCH/work/writer/` holds only the rendered PNGs used to check the pages.
No Julia process of mine was run; every number came from the tables the root's
`final_runs.sh` produced (`== all done Mon Sep 7 16:32:45 EDT 2026`) or from a report.

## 1. Section list with page numbers

     1  p  4  How to use it                                (1.1 files, 1.2 API, 1.3 cost + RSS)
     2  p  5  Groundwork: the measurements that came first  (2.1 profile, 2.2 offsets/reflections,
                2.3 Gila's far error today, 2.4 the volume identity, 2.5 rho and the slow band,
                2.6 derivative conditioning, 2.7 exact moments, 2.8 the joint error target)
     3  p 10  The volume formulation                        (volume.tex; 3.6 is the sign pin +
                                                             Paul's Mathematica)
     4  p 15  The local expansion                           (expansion.tex; 4.7 octant sub-boxes)
     5  p 21  The a priori bounds                           (bounds.tex)
     6  p 28  A sharper l-truncation bound                  (bounds2.tex: Theorems A and B)
     7  p 34  What the bound certifies, and what it does not (7.1 the 60 cases, 7.2 X = 16.4 eps,
                                                              7.3 the a posteriori certificate)
     8  p 36  Band selection                                (join.tex; 8.4 the shipped selector,
                                                             8.6 the slender band)
     9  p 44  Verification                                  (9.1 vs the reference, 9.2 aspect ratios,
                9.3 symmetries, 9.4 digits lost, 9.5 positivity, 9.6 the overlap)
    10  p 49  The build, before and after                   (10.1 times, 10.2 attribution,
                                                             10.3 accuracy, 10.4 per-offset cost)
    11  p 52  The registry of approach families             (registry.tex; 11.10 round 2)
    12  p 58  Defects found on the way                      (12.1 Gila + round-1, 12.2 D1-D10)
    13  p 61  What remains uncertain
    14  p 63  Acknowledgement of method

Full subsection list is in `farfield.toc` (89 entries).

## 2. pdflatex log summary

    pdflatex three passes (rm'd .aux/.toc/.out before the first)
    errors (lines starting "!")            0
    Overfull \hbox / \vbox                 0
    Underfull \hbox                        3   (loose interword spacing in three paragraphs
                                                that contain long \texttt runs; cosmetic)
    "Reference ... undefined"              0
    "Citation ... undefined"               0
    LaTeX Warning lines                    0
    LaTeX Font Warning                     2   OMS/cmtt/m/n and OMS/cmtt/bx/n undefined,
                                               substituted by OMS/cmr — the braces inside
                                               \texttt{...} in the defect table; renders correctly
    pages                                  63
    file size                              834 917 B

Six overfull boxes existed on the first complete build and all six were fixed, not suppressed:
two verbatim lines in S1.2 (shortened comments), the eight-offset contact table and the
(6x6x12) positivity table of S9.5 (\small -> \footnotesize, shorter row label), the
before/after accuracy table of S10.3 (shorter row labels), the family summary table of S11
and the defect table of S12.2 (fixed-width `L{}` columns with \raggedright, via
`\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}`).

Rendered pages inspected as PNG at 68-70 dpi (`pdftoppm`): 1 (title + abstract), 4 (the API
table), 5, 6 (the profile and band tables), 33, 34 (the bound2 measured tables), 39 (the
96-row `e_terms` longtable), 44 (the `d_ref` table), 48 (the positivity tables), 50 (the
before/after timing tables), 52 (`i_cost` + the registry), 58, 59 (the defect table), 61,
63 (the method section). All tables sit inside the text block; the two longtables break
across pages with repeated headers.

## 3. Which table fed which number

Every number in the document comes from one of these files, and the text names the file.

| table file | fed |
|---|---|
| `scratch/work/root/bench_before_1.txt` | S2.1 the whole single-thread profile: 128^3 far fill 1058.37 s, full build 1311.87 s, 1.941e10 kernel evals, 54.5 ns/eval; the band attribution (order 4: 2 092 239 offsets, 502.7 us, 1051.8 s, 1.928e10 evals); 32^3 and 64^3 rows |
| `scratch/work/root/bench_before_12.txt` | S2.1 the 12-thread column: 4.42 / 29.81 / 262.99 / 10.42 / 20.99 s |
| `tables/bench_1.txt` (this run) | S10.1 every 1-thread before/after row (128^3: 28.59 / 1157.46 / 1189.69 -> 28.59 / 0.70 / 1.491 / 34.26 s, 34.7x); S10.1 far-fill factors 882/866/776/1348/1947x; S10.3 the egoFur and random-vector columns; S10.4 the 711 ns/offset and the 498.1 us order-4 comparison |
| `tables/bench_12.txt` (this run) | S10.1 every 12-thread row (128^3: 5.14 / 200.76 / 206.54 -> 5.14 / 0.73 / 0.200 / 6.73 s, 30.7x, far fill 1002x); S10.2 the attribution (5.14 s of 6.73 s is wekTrp; embed+gather+FFT 0.60 s); the abstract |
| `tables/a_rho.txt` | S2.5 the rho / rho_oct table, rho/rho_oct 1.48-1.98, the slender (0,0,n) row (11.32 -> 0.354) and its rho_oct row |
| `tables/a_band.txt` | S2.5 the fixed-size slow band: 30 / 134 / 429 / 3089 identically at N = 32, 64, 128; the slender exception 242..19 736 |
| `tables/a_mu.txt` | S2.7 the mu_n values and "0 disagreements over 5 lengths x n = 0..24" |
| `tables/a_div.txt` | S2.4 "max |difference| = 0//1 exactly in all eighteen rows" |
| `tables/b_volume.txt` | S3.6 (volume.tex's own table) and the cross-check that farTensor vs the same references is 4.40e-15 |
| `tables/c_gila_{c32,c8,c4,sl}_{r,c}.txt` | S2.3 the whole "far-field error today" table: per-pair 1e-15..9.5e-14, amp 4.8-1300 (cubes) and 4.3e3-1.67e5 (slender long axes), tensor tMx/tEn per shape and direction, and the slender short-axis row 0.789 / 0.227 / 3.68e-2 / 4.50e-3 / 1.79e-3 / 2.97e-6 / 8.94e-7 / 9.12e-9 with tEn up to 1.60 |
| `tables/d_ref.txt` | S9.1 the 14-row summary (generated as `tex/tab/d_ref.tex`) and "427 rows, eMx <= 8.60e-15, pEn <= 9.34e-15" |
| `tables/e_terms.txt` | S8.4 the 96-row shipped band-edge longtable (`tex/tab/e_terms.tex`), f = 1 rows only |
| `tables/e_route.txt` | S8.4 the routing counts (`tex/tab/e_route.tex`), tol = 1.0e-14; the L-floor percentages 97.8 / 99.6 / 97.5 % computed from its histogram lines |
| `tables/e_ncut.txt` | S5.5 Remark "N is adaptive": the shipped shapes' nNd = 6 / 9 / 12 / 6 against nMx = 12 |
| `tables/e_digits.txt` | S9.4 the 66-row longtable (`tex/tab/e_digits.tex`) and "worst 1.59 digits at c4 (64,0,0)" |
| `tables/f_bands.txt` | S8.6 the measured k-series band: x in [0.280, 0.602], N in [12, 17], Lambda <= 1002, eps*Lambda <= 2.23e-13 |
| `tables/f_ksr.txt` | S9.6 route (iii) at 2.3e-17..9.8e-17; the two seam offsets (0,0,20), (0,1,30) |
| `tables/f_overlap.txt` | S9.6 the overlap table: worst per entry 1.02e-14 over 16 340 offsets, per-part up to 8.9e-12 |
| `tables/g_sym.txt` | S9.3 the invariance table, permutation <= 1.63e-15, homogeneity <= 6.55e-15, Float32 6.20e-8, BigFloat(160) 5.14e-16 |
| `tables/h_pos.txt` | S9.5 the six-row positivity table (lam_min -4.4681e-15 -> -2.4488e-15 at c32; -6.3987e-15 -> -4.8810e-15 at c4; -6.553e-14 -> -1.921e-14 and -9.939e-3 -> -1.465e-2 on the slender block) |
| `tables/i_cost.txt` | S10.4 the per-L cost table (`tex/tab/i_cost.tex`), route (ii) 53-84 us, the mean-terms figures 360/473/687/644, and the block fills 767.6 / 706.4 / 1410.3 / 1220.1 ns/offset |

Numbers taken from **reports** rather than from a table file, each named in the text:

- `merge.md` S1.2/S1.3/S1.4: the (|k| r_d, N) relation (70 rows) in S5.5; the four re-measured
  D9 rows and the lambda/2 cube in S9.2; S3 the 60-case bound classification, X = 16.4 eps,
  amp <= 3.14e3, the 15 rounding-floor violations in S7.1; S4.2 the 21-25 GB / 44.5-64.7 s
  table builds in S1.3 and S13.3; S4.3 the threaded route (iii) 633 -> 128 s; S6 the whole
  contact-rule finding of S9.5 (117 % at (0,0,1), 12.7 % at (0,0,0), lam_min +3.25013e-4).
- `audit2.md`: S1.2 the twelve shapes and their aspect ratios; S1.4/S1.5 the (|k| r_d)^25.6 law
  and the pre-fix per-entry errors; S1.6c the c128 nCut mechanism; S2.2 the forty boundary rows;
  S3 the circulant/reflection check (9.81e-15, 0 sign flips of 8748); S5 the Float32 defect;
  S8.1 the reflection test on route (ii); S9 the D4 cost measurement; S11.3 the defect table.
- `prep.md` C: est/max|T| in [1.10, 6.34] and 135 on the needle; estLo and where it is vacuous;
  the a posteriori certificate 4.5e-14; A.2 tol 1e-13 -> 1.6e-13 per entry vs tol 1e-14 -> 9.3e-15.
- `unify.md` S8, S9, S11: the round-1 source defects of S12.1 and the per-part statement of S13.2.
- `famA.md` S1, `famG.md` S1, `auditG.md`: the three derivative-conditioning tables of S2.6.
- `geometry.md`: the error-target numbers of S2.8 (FFT 5.3e-15 / 2.9e-6, anti-Hermitian floor
  6.5e-14 relative, dfltPrc Float32, relTol sqrt(eps) = 1.49e-8 / 3.45e-4).
- `deliver.md`, `refill.md`: the cache contents and the four refQuad slender references.
- `famE.md`, `famB.md`, `famBsub.md`, `famC.md`, `famD.md`, `kseries.md`: the registry section,
  which is `registry.tex` plus the round-2 subsection.

## 4. What was changed in the reused fragments, and why

`bounds.tex`
1. A boxed lead paragraph: Theorem 5.3 is **not** the bound the code evaluates; Theorems A and B
   of S6 are, at tol = 1e-14, and the two bound different truncations.
2. New `Remark 5.9` after the n-truncation: N is adaptive (the (|k| r_d, N) table from merge S1.2),
   NMAX = 12 is a floor, the call errors above NCAP = 40, plus the shipped shapes' nNd from
   `e_ncut.txt`. The old text implied a fixed n_max of 6/10/12.
3. S5.8 "what is not proven": the table-build cost updated from 14.65 s / 41 s (L = 40 / 48,
   round 1) to 44.5-64.7 s and 21-25 GB, the cancellation from 5.8e8 to 7.31e11 (L = 56) and
   1.90e12 over the audit shapes, and the end-to-end statement to the 427-reference numbers.

`bounds2.tex`
4. The "bound in use" row label of the measured table renamed "the first-round bound" (it is no
   longer in use).

`join.tex`
5. Lead paragraph distinguishing the first-round band-edge table (tol 1e-13, L_max 64) from the
   shipped selector (Theorems A/B, tol 1e-14, L_max 56).
6. The rule box: tol is now a symbol, with a table giving TOL = 1e-14, LMAX = 56, NCAP = 40, and a
   paragraph on why the tolerance was tightened when the bound was sharpened.
7. L_max justification rewritten (56 not 64; table cost 44.5-64.7 s; octant reaches 4.4e-15 at
   L_j <= 52).
8. New S8.4 "The band edges the shipped selector produces": the `e_terms` and `e_route` tables.
9. S8.5: the shipped octant orders (50/50/51 at (2,0,0), 51/51/52 at (2,1,1)) and the routing
   count of exactly twelve; "10^7 offsets" -> "two million".
10. S8.6: the k-series band replaced by the measured one (67 offsets, x in [0.280, 0.602],
    N in [12, 17], Lambda <= 1002, eps*Lambda <= 2.23e-13, 1.9-9.5 s/offset, 128 s cold on
    8 threads) instead of the round-1 estimate (64/83 offsets, N = 13-16, Lambda = 1e5).
11. S8.7: "64 offsets per octant" -> the shipped 67 (iii) and 111 (ii).

`registry.tex`
12. A one-page summary table of the nine families with the number that decided each.
13. A caveat that every timing in the section is a round-1 figure measured under load, pointing at
    S10 for the deliverable's timings.
14. New S11.10 "The second round": the six passes (theory, unification, sharper bound, preparation,
    adversarial audit, merge) with the number that made each necessary.
15. "the 64 offsets of S8.6" -> 67; Theorem ref for the L rule moved to Theorem A.

## 5. Numbers I could not source, and what I did about them

- **None are left in the document.** Two candidates were removed rather than quoted:
  the round-1 claim "2.37 s for the whole 128^3 far field single-threaded" (famB, under load)
  survives only inside the registry, under the explicit caveat of change 13; and the round-1
  "L_j <= 48" of famBsub is not quoted at all, because `theory.md` and `e_terms.txt` disagree
  with it (that disagreement is recorded in join.tex's own Remark 8.x, which was already there).
- **One number is quoted from two runs and both are given**: the single-threaded 128^3 far fill
  is 1058.37 s in the baseline profile (`bench_before_1.txt`, the band attribution) and
  1157.46 s in the staged before/after loop (`tables/bench_1.txt`). S2.1 uses the first, S10.1
  the second, and the abstract uses the first. They differ because the first is
  `@belapsed` on one representative offset per band times the band population and the second is
  the actual serial loop.
- **The 1-thread benchmark ran at a decaying load.** `bench_1.txt`'s header records load
  11.05 8.65 6.07 at its start, because it began immediately after the 12-thread run. Its
  contact time is 28.59 s against 26.90 s in the quiet baseline and 5.14 s on twelve threads;
  S10.1 says so in the text. The far-fill numbers are unaffected at the 3 % level
  (32^3: 22.80 s here, 20.04-20.30 s quiet).
- **verify.jl part (h) row for c4 at f = 1 moved** between the merge run and this one
  (lam_min -5.009707e-15 -> -4.880966e-15, i.e. the improvement factor 1.28 -> 1.31); the
  document quotes this run.

## 6. Numbers the document leads with

    128^3 at lambda/32, 12 threads: far fill 200.76 s -> 0.200 s (1002x), build 206.54 -> 6.73 s (30.7x)
    128^3 at lambda/32, 1 thread:   far fill 1157.46 s -> 1.491 s (776x), build 1189.69 -> 34.26 s (34.7x)
    32^3 at lambda/8 / lambda/4, 12 threads: far fill 1607x / 1963x, build 3.0x / 4.0x
    accuracy vs 427 220-bit references: 8.60e-15 max-norm, 9.34e-15 per entry, <= 1.59 digits lost
    truncation certified (no violation over 60 random cases), rounding measured <= 16.4 eps max|T|
    a posteriori certificate over the 427 references                         4.5e-14
    routing, 128^3 cubes / 64x64x128 slender          2 097 132 / 12 / 0  and  524 102 / 111 / 67
    Gila before, per entry: cube 7.9e-13, slender long axis 3.4e-11, slender short axis 1.60
    Gila's order-9 contact rule on a slender touching shell                  117 % wrong
    slender Im M at f = 1+0.1i: -9.94e-3 (Gila) -> -1.465e-2 (+farfield) -> +3.25e-4 (+exact contact)
    one-time geometry table per shape                44.5-64.7 s, 19.5 MB, 21-25 GB peak RSS
