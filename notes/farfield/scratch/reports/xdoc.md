# xdoc — tex/crossscale.tex, the cross-scale updates of farfield.tex / registry / defects / uncertain, and farfield.pdf

Deliverables (all under `REPO/notes/farfield/`):

    tex/crossscale.tex   new, Section 11 "Unequal cells: the far field of a composite volume", 8 subsections,
                         pages 53-66 of the pdf (14 pages; the brief said 8-12, the two extra pages are the
                         27-split table with lattice rows, the accuracy tables and the item-5 subsection)
    farfield.tex         \input{tex/crossscale} after tex/bench, before tex/registry; abstract paragraph; date
                         8 September 2026; files table (2009 lines, pair tables), API table (farSetupX,
                         farRouteX, farTensorX, farBlockX!), usage snippet for a pair, cost paragraph for the
                         pair tables; TOC subsection-number width 2.3em -> 3.0em (12.10/12.11 collided)
    tex/registry.tex     four rows (u1)-(u4) in the family table; new S12.11 "The cross-scale round"
    tex/defects.tex      Gila item 10 (the two separated-pair rules on unequal cells); D11, D12 rows; paragraph
                         "D11 is D2 and D9 again"
    tex/uncertain.tex    est under-states on the r16 rod (S14.1); new S14.4 "Unequal cells" (three paragraphs)
    farfield.pdf         82 pages, 999614 bytes, pdflatex three passes, 0 errors, 0 overfull boxes

Work dir `scratch/xwork/xdoc/`: build_1..3.log (the three passes), pg-*.png (rendered pages 1, 4, 53, 54, 58,
62, 63 at 60 dpi, inspected).  No Julia process of mine was run; every number is from a report or a table file.
The verification agent xverify started at 12:36 (its scratch/xwork/xverify/ was empty at 12:33); XVERIFY_STATE

## 1. Section 11 and where every number comes from

11.1 The trapezoid weight (p53)
  a, b, b-a = min, b^2-a^2 = sT sS; w = triangle(b) - triangle(a); mu_n two-term form; parity; table /prod(b);
  swap identity; est with V_s ......................... reports/xtheory.md S1.1, S1.6; BRIEF_cross.md
  "loses at most 0.66 digits at ratio 16 (a/b = 15/17, n = 0)" .......... xtheory.md S1.1
  bit-identity of equal-cell tables at a = 0 ............................ xunify.md B (120/120), xtable.md S5

11.2 The bounds (p54-56) — Lemma 11.1 (ibp'), Theorem 11.2 (A'), Remarks 11.3 (sub-boxes, gcd), 11.4 (est, estLo,
  n-tail): text reused from scratch/xwork/xtheory/bounds_cross.tex, every formula checked against xtheory.md S1.1-1.6
  (mu_n, lambda_n = 2(b^{n+1}-a^{n+1})/(n+1), point masses 2b^{2i}+2a^{2i}, V^{ab}_l, V^{aa}_l, W_l, r_d = |b|,
  D_max = sqrt(sum (|R_i|+b_i)^2)); labels renamed bndx -> xs.
  "359 comparisons, 0 mismatches" ....................................... xtheory.md S3 (out_xbit.txt)
  "|k||b| 0.405 .. 2.89"; estLo fails at kap 1 for r >= 4 on xy/xyz and everywhere r = 16 xy/xyz;
  estLo/est 0.005-0.03 / 0.02-0.42 ...................................... xtheory.md S4 (out_xlow.md)
  exact-rational table, 132 rows all 0, the 7 shapes, case counts ...... xtheory.md S2 (out_xexact.txt)

11.3 Where Gila produces unequal pairs (p56-58)
  _sndBlk remesh, routes (a)/(b), two-cell ring, chkCmpVol/chkParCmpVol .. xgeom.md S0, S5 item 3 (nest_out.md)
  64^3 vs 4^3 composite = 64^3 x 64^3 equal-cell block, 2.1e6 offsets ... xgeom.md S0
  half-integer lattice, R_x - c'_j integer, nearest sub-offset max-norm 2, dyadic .. xgeom.md S1, S5 item 4
  classes: contact / adaptive (hcubature 1e-6 / 1e-9) / fixed; 27 r^|A|, prod(r_i+2), |R_i| < 1.5 max ..
                                                                         xgeom.md S1, S4
  "0 missing / 0 extra / 0 class disagreements on five builds" .......... xgeom.md S1b (instr_out.md)
  offsets table (8 rows) ................................................ xgeom.md S1 table (fixed = ord9 + ord7)
  nearest (r+3)/2 g = 1.25/0.875/0.6875/0.594 coarse cells; fixed starts at kap = r .. xgeom.md S1
  blocks table: 1404928/324/36540/1368064/15936/1456; 1728000/100/4508/1723392/2476/196;
  route (b) 1118192 / 1528704 / 2616320, all fixed, 0 adaptive, rho max 0.4685/0.4115/0.3800, sep 3 ..
                                                                         xgeom.md S3, S5 item 3
  "1.1 %" = 15936/1404604; Gila 1.8 h (13/4.7/2.5 ms; adaptive 1-54 ms) . xgeom.md S3, S2a
  accuracy table (356 keys): adaptive 2.7e-9..2.28e-4 median 1.2e-6, 106/111 > 1e-8, 13 > 1e-5, slabs (8,8,1)
  at (8.5,0,0) and (16,16,1) at (12.5,0,0); ord 9 1.5e-14..8.3e-12, lambda/2 cube 1.5e-14..5.6e-12; orders 7-4
  <= 6.3e-13 except 1.7e-11 (2.1e-10) at 4 lambda; rods 8.8e-13, 2.6e-9, 1.42e-6, 1.77e-4 (2.28e-4); f != 1
  80 offsets 2.0e-8..6.3e-6 / 3.2e-14..1.5e-11 ........................... xgeom.md S2b, S4 (cmp_all.md)
  adaptive faster than ord 9 from kap 2 (1-17 ms) ....................... xgeom.md S2a
  sign convention -1.000000 both sides ................................... xgeom.md S2b last block (cmp_out.md)

11.4 Three routes and the dropped one (p58-60)
  whole box 0.9-12.5 us for L = 12-56 (t_w(L) table); 1,387,696 of 1,404,604 = 98.8 %; 4.15 s; L <= 24 for
  872,468 (= 13636+539332+319500); band edge 0.58-0.60; 16,908 ........... xtable.md S4 (census.txt, costsum.out)
  box-sum formula, prod(nT+nS-1) adds ................................... xtable.md S4b, xunify.md A (boxSumX!)
  Lambda 1.0-2.3 (1.0 at r2, 1.1-1.3 at r8, 1.5-2.3 at r16 xyz) .......... xtable.md S3e, S2d (estratio_f1/2/3)
  naive 38.9 ms/offset, 658 s, 160x; box sum 0.95 s (35,929 offsets, 33^3) + 28.6 us/offset (0.48 s);
  identical at the offset checked ....................................... xtable.md S4
  box sum vs per-offset average on 340 offsets worst 9.3e-16 ............ xtable.md S4b (r4 block)
  route 3: 6-21 s per offset, N = 13-44, 103 of 438 records, 0 block offsets .. xunify.md C (route-3 tables), D.2
  27-split: 8 piece-type tables, Theorem B budget, reflection ............ xtable.md S1, S1b
  plateau radius (r-1) sqrt3 g/2 vs (r+3) g/2: 0.35/0.87/1.17/1.24 ....... REGISTRY.md cross-scale round-0 analysis
  split table rows: x1c/x2c/d1c of r2x, x1c/x1v of r2xyz, x1c/x1e of r4xy, x1c/x1v of r4xyz, r8xyz, r16xyz
  (rho_w, L_w, rho_p, L_p, t_p, t_w) .................................... xtable.md S2 rows (lines 148-260)
  "0 of 2213", 13.0 g vs 12.7 g ......................................... xtable.md S4
  5.4-8.9x = 56.5/9.0, 67.5/12.5, 40.3/4.55, 39.0/4.54 .................. xtable.md S2 (r2x x2c, d1c, x3c, d2c)
  split vs reference 1.5e-15 over 152 ................................... xtable.md S3b (recheck2_f1.md)
  routing rule (touch test, route 1/2/3 order) ........................... farfield.jl farRouteX :1672-1688

11.5 Two defects (p60-61)
  D11: 5.9e-14 .. 6.8e-10; max. swallow; rNc = 1.66 g piece centre, rho = 8; far radius 3300-7500 g; before/after
  table (r8xy x6e, r8xyz x8v, d4e, r16xyz x16v) .......................... xtable.md S2d table, S2e (ncutfull.txt)
  old / fixed n-cut vectors of r16xyz; N = 8 / 10 / >= 11 -> 1e-10 / 2e-14 / 3e-16; nMax 12 -> 13 ..
                                                                         xtable.md S2d, S2e; xunify.md E (n13)
  |k| r_d >~ 2.9; equal-cell N <= 6 ...................................... xtable.md S2d, S5c; xunify.md B
  D12: 4.68e-13 vs 3.4e-14 (14x), 5.12e-16, 602/406 columns, 2.80 vs 1.88 MB .. xunify.md D.0, open item 1

11.6 Verification (p61-64)
  cache 441 records / 438 keys / 3 duplicates / 416 brief keys; matrix; slender pairs .. xref.md S3
  rule A ordN 44 vs 64 1-7e-63; (5,1,0) to 0.0 .......................... xref.md S4, S1
  rule B 16.5 us/point, 0.4-20 Mpts, 5-330 s, 10-20x cheaper ............ xref.md S2, S6
  A vs B 1.6e-58..3.9e-57 at kap 1, 1-2e-63 from kap 2, floor 1-4e-64; ord 40 -> 1.4e-64; 6.8e-53 r16 xyz kap 1;
  >= 52 digits; slender 3.0e-50; gcd identity 3.2e-64; swap 7.0e-65 / 1.4e-64; 4.3 h .. xref.md S4, S5, S6
  per-f table (64/310/64; routes; worst; digits; swap; 0 violations) .... xunify.md C per-f table
  per-route table (244/91/103; medians/worsts; cert ratios) ............. xunify.md C per-route table
  bold summary 3.4e-15 / 4.7e-15 / 1.3 digits / 0 violations; bitwise 438/438; swap exactly 0 on 244, 4.9e-15
  (15 of 91), 1e-38..2.6e-36; route 3 cert 4.6-5.0e-15 on 99, 1.5e-14..8.1e-13 on 4, est 100-8100x ..
                                                                         xunify.md C
  prototype recheck 272 / 147 / 152 / 156, 1.4e-15 / 1.5e-15 / 5.1e-15; 64 + 64 at 2.6e-15 / 2.7e-15 ..
                                                                         xtable.md S3b, S3c, S3d
  bit-identity paragraph (five bodies, 34,974,803 bytes, 120/120, 108/12/0, 13,894 / 13,952) .. xunify.md A, B, B'
  verify.jl part (a), a_rho.txt 4 -> 81 lines ........................... xunify.md F
  blocks table (8.91/1.21/1.29, 4.86/0.99/1.09, 17.92/9.81/17.63, 10.71/5.69/7.13); 9 refs 8.7e-16; 13 refs 3.4e-15
                                                                         xunify.md D.1, D.2
  phases on 12 threads (2.94 / 0.67 / 1.08 / 1.00 wall = 10.99 + 0.26 CPU-s; 2.1 us; 3.2 us; 9.5 us; 4.6x, 5.2x)
                                                                         xunify.md D.2
  L histogram 20:13636 22:539332 24:319500 .. 56:1544 ..................... xunify.md D.2 (= xtable.md S4)
  certificates route 1 4.7e-15/8.0e-15/4.1e-14, 516,872 above tol; route 2 9.0e-15/1.2e-14/3.0e-14; 4-9x / 15-50x
                                                                         xunify.md D.2
  pair tables 3.3-5.7 s, 4.0 s reference, 0.4-2.5 MB vs 19.5 MB, +0.2-0.3 GB, n13 .. xunify.md E
  Gila today 3.5 s + 12.6 s (small block), 1.8 h (large) ................. xgeom.md S1b, S3
  adversarial paragraph ................................................. XVERIFY_PARA

11.7 The near field (p64-65)
  gate fires on every separated pair; [kap, kap+1]; |D| = 2 escapes; touching r >= 2; [r, r+1] .. xnear.md S0, S1
  128 bits: 0-0.99 of 38.2 digits, 5.8e-38 on seven pairs ................ xnear.md S4 item 5 (ksr128.txt)
  2772 face pairs, 504 vs mom.jl identical; 0.99 parallel, 0.93 thin (m = 0), 0.90 closed; 0 above 1 ..
                                                                         xnear.md S2, S2b
  per box 0.51 / 0.71 median, 0.95 / 0.96 worst, closed 1.15; 2228 of 2734 .. xnear.md S3b
  equal-cell lattice 0.91 / 0.87 ........................................ xnear.md S2 (quoting moments.tex)
  box3Tay: recurrence, polydisc gate lam > 1/2, harmonic trap (level 2 = 0, 3e-5 .. 3e-3, 11-13 digits, 3 boxes),
  three-level rule; 94.6 % / 99.8 %; rejected 1.0 % at lam 0.463-0.497; one eps (0.00 / 0.35, cnd <= 3.4);
  2-3x, 6.3x, 2126x (115 levels, lam 0.67); lam >= 1 caps 64 levels at 1.9x .. xnear.md S3, S3a, S3b
  face-pair level: 558 of 1595 pinned at 0.71, 0.93 -> 0.90 -> 0.71; SERALL 2.4x / 33x .. xnear.md S3c, S3d
  recommendation, moments.jl unchanged .................................. xnear.md S4

11.8 Open items (p65-66)
  est over-states 100/550/2500/8100, certs 1.5e-14..8.1e-13 vs 2.3e-17..9.0e-17 (2600-8900x) .. xunify.md C
  est under-states 0.50 (f = 1) / 0.25 (f = 0.37) on r16x x1c ............ xtable.md S3b, S3d
  1800 offsets (0.13 %) at rho 0.55-0.60, 1-4 tol, (24.5,7.5,7.5) at L 54 2.9e-16; 33^3 -> ~41^3 ..
                                                                         xtable.md S4b; xunify.md D.2, open item 6
  unlocked writer ........................................................ xunify.md open item 1
  selection 2.1 us + route 1.0 us = 4.2 s vs 4.4 s; 2.9 s serial on 12 threads; ~1 GB BigInt of 2.95 GB ..
                                                                         xunify.md D.2, open item 2
  route 3 6-21 s, 103 records, 0 block offsets, disk cache ............... xunify.md C, open item 4
  fine set 0.8-0.95 s serial, fill 0.01-0.04 s ........................... xunify.md D.2 (fine_t.jl), open item 5
  odd ratios only in the exact checks (ratio 3) .......................... xtheory.md S2 (v); xgeom.md S5 item 4

## 2. The other files

farfield.tex abstract paragraph: 132 identities (xtheory S2); ratio >= 4 divergence (xtable S2); Lambda 1.0-2.3
(xtable S3e); 438 / 3.4e-15 / 4.7e-15 / 0 violations (xunify C); 5.7 s on 12 threads warm (xunify D.2);
1.8 h, 1e-6, 1.8e-4 (xgeom S3, S2b).  Files/API tables: farfield.jl 2009 lines, cross-scale lines 1537-2009
(xunify A; `grep -c '' farfield.jl` = 2009); signatures from farfield.jl :1586-1589, :1672, :1862-1864,
:1922-1926; pair tables 0.4-2.5 MB, 3.3-5.7 s, 4.0 s, 0.2-0.3 GB (xunify E).
registry.tex rows/S12.11: 98.8 % (xtable S4), 2.4e-15 over 244 (xunify C), Lambda 1.0-2.3 (xtable S3e), 658 s vs
1.4 s (xtable S4), 0 of 2213 (xtable S4), 5.4-8.9x (xtable S2), 103 records / 1e-17 / 6-21 s / 0 block offsets
(xunify C, D.2), 0.87/1.17/1.24 (REGISTRY.md round-0), 1.5e-15 over 152 (xtable S3b), 5.9e-14..6.8e-10 and
Lambda 1.1-1.6 at those offsets (xtable S2d), 16,908 (xtable S4), 0.9-12.5 us (xtable S4).
defects.tex Gila item 10: 2.7e-9..2.28e-4, median 1.2e-6, 36,540, 1.77e-4, 1.5 g, 1.7e-11 (xgeom S2b, S3, S4);
D11 / D12 rows (xtable S2d, S2e; xunify B, D.0).
uncertain.tex: 0.50 / 0.25 (xtable S3b, S3d); estLo failures (xtheory S4); 2600-8900x, 1800 offsets (xunify C,
D.2); 103 records (xunify C); 4.2 s vs 4.4 s, ~1 GB (xunify D.2).

## 3. Changes to numbers already in the document

None of the equal-cell numbers changed.  farfield.jl's line count in S1.1 went from 1480 to 2009 (the file was
replaced by xunify at 09:29; 1488 lines before the cross-scale merge per xunify.md A, the document said 1480 from
the merge round's count).  The date on the title page is now 8 September 2026.

## 4. pdflatex log summary

    pdflatex three passes (rm'd .aux/.toc/.out before the first)
    errors ("!" lines)                     0
    Overfull \hbox / \vbox                 0   (eight on the first complete build, all fixed: the files table of
                                               S1.1 -> L{0.72\textwidth} column, two verbatim lines shortened,
                                               the blocks table header, the Gila-accuracy table -> three L{}
                                               columns, eq. (66) split (m moved to the text), the per-route
                                               accuracy table separated with L{} columns, the references column
                                               of the block-timing table moved into the text)
    "Reference undefined"                  0
    LaTeX Font Warning                     2   (OMS/cmtt, pre-existing; braces inside \texttt)
    pages                                  82 (was 63)
    file size                              999614 B

## 5. Numbers I could not source, and what I did

- The prompt's expectation for the 27-piece split is quoted as the prompt states it (three pieces per axis,
  reflection rule, plateau its own mirror); its convergence claim was never made by the prompt, so the section
  says only what was measured.
- xtable.md S2 timings were measured at load 8.5-12.6 and are quoted with that load; the ratios (5.4-8.9x) are
  what the text leans on.
- "faster than the order-9 fixed rule from kap 2 on (1-17 ms)": xgeom S2a's 1-17 ms range for kap 2..8 adaptive
  offsets against 12.6-13.4 ms for order 9; quoted as a range.
- The per-thread "fill 1.00 s wall for 10.99 CPU-s" is xunify D.2's own accounting (CPU-seconds summed over
  threads); the 3.2 us per whole-box expansion is the single-thread figure (4.41 s / 1.39e6) from the same table.

## 6. Final pass: xverify.md arrived (17:38), the verification text written

Polled `scratch/reports/xverify.md` for the `## DONE` marker with a shell loop (never `pgrep -f`); it appeared at
17:38 with the file at 470 lines, part (j) having finished after 17583 s single-threaded.  Read the whole report,
then made four changes.

1. `tex/crossscale.tex`, `\paragraph{The adversarial verification.}` (sec:xs:adv) -- the placeholder that said the
   report had not completed is replaced by the real text: the cache re-evaluation (439 records = 438 + one added
   meanwhile, four ways, 4.9 h single-threaded at load 3.2-11.5, routes 244/92/103, worst eMx 3.42e-15, worst per
   entry 4.71e-15, 1.3 digits, cert ratios 2.4/23.9/135, 4.0/29.2/673, 48.7/115/8930, 0 violations in 878 certified
   evaluations, farTensorX bitwise = farBlockX! on 439/439) as the INDEPENDENT reproduction of the xunify table
   already in the document, plus the new reflection identity (exact on route 1, 5.17e-15 route 2, 1e-36 route 3);
   the 47 new 220-bit records and the 66 adversarial offsets (50 referenced; worst 5.1e-15 / 5.3e-14, swap 3.5e-15,
   flip 8.4e-16, 0 violations) as an eight-row `tried / result` table; the exhaustive routing check (125,256
   separated offsets, 16/36/100/324 touching all refused, route 3 never, xLat/xNear right everywhere, nearest piece
   max-norm exactly 2, rho 0.59-0.68), bitwise farTensorX = farBlockX! on 15,616 offsets and the 1/4/12-thread
   bit-identity of the r=4 block (15,228 entries, 0 differ); the independent Gila comparison (whole r4 block 1692
   offsets, 2172-offset r16 sample: adaptive 1e-7..5e-4, fixed 2e-14..1.8e-12, library <= 3.4e-15, 6-8 us vs
   5-13 ms); and the three defects, stated as defects.
2. `tex/crossscale.tex`, new `\paragraph{D13: ...}` in what is now "Three defects found on the way" (the subsection
   title changed from "Two"): cause (farSetupX certifies [rNc, rHi] and no caller compared |R| against it),
   reproducer, 1.59e-11 under a 1.7e-14 certificate = a 1000x violation, the three-line fix including the third line
   without which a near-only set refuses everything, and the after numbers (bitwise equal to a fresh set, 1.21e-15
   vs the reference; kap 100 refused by name).
3. `tex/defects.tex`: the D1-D14 table was split at the midrule into two tables (D1-D10 from the equal-cell audit;
   D11-D14 from the cross-scale round and the adversarial verification) because the two new rows pushed it past a
   page -- that was the only overfull box the new text introduced.  D13 (fixed) and D14 (the Float32 NaN,
   pre-existing, equal-cell, NOT fixed) added with the report's numbers; the lead-in sentence and the "Four of these
   deserve a sentence" count corrected (it said "Three" with four paragraphs following, pre-existing).
4. `tex/uncertain.tex`: the coverage paragraph no longer says the adversarial report had not completed.  It now
   states what that round added (ratios 3 and 6, a pair coarse on a different axis in each cell, lambda/2 and lambda
   coarse cells, two complex frequencies, mixed signs) and what is still NOT verified: non-integer rational ratios
   (refused, not computed; the three-line gcd extension was measured at 1.4e-15 but not shipped), coarse cells above
   lambda and ratios above 32, and Float32 in the near band (NaN, D14).  Routing is no longer a sample.

Numbers I deliberately did not write: the per-group offset counts of part (k) (the report gives 66 total and 50
referenced but not a clean per-group split, and I did not want to derive them); "2172 offsets" as the size of the
whole Gila comparison (it is the r16 sample only -- the first draft said this and was corrected); and 1.5e-15 for
the rational-gcd copy, corrected to the report's 1.4e-15.

Final build (`rm`'d .aux/.toc/.out, four pdflatex passes; 1-min load 3.2-3.5 throughout, the user's REPL idle):

    pages                                  82   (80 before this pass, 63 at the first complete build)
    errors ("!" lines)                     0
    Overfull \hbox / \vbox                 0
    "Reference undefined"                  0
    LaTeX Font Warning                     2    (OMS/cmtt m and bx, pre-existing: braces inside \texttt)
    file size                              999614 B

Nothing under `src/`, `test/`, `notes/moments/` or `notes/farfield/farfield.jl` was touched; the only files this
pass changed are `tex/crossscale.tex`, `tex/defects.tex`, `tex/uncertain.tex` and the built `farfield.pdf`.
