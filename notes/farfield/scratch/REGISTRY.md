# Registry of approach families (root's file; updated after every round)
Status: OPEN (untested), ALIVE (measured, converging somewhere), BLOCKED(band) with the number that
blocked it, WON(band) with the verification number. Reopen only for a materially new mechanism.
NOTE 2026-09-06 13:10: machine crashed and rebooted; /tmp scratchpad lost (env, reports, partial
work of 7 agents, ~2 h). Rebuilt under notes/farfield/scratch. Round 1 relaunched 13:30.

## Round 0 (root's estimates before measurement)
- rho = sqrt(sum s_i^2)/|R|: cubes 0.87 (2 cells), 0.58 (3), 0.43 (4), 0.29 (6), 0.22 (8), 0.11 (16).
  Plain centre expansion needs L ~ 13/log10(1/rho): 215 (2), 55 (3), 36 (4), 24 (6), 20 (8), 13.5 (16).
- Cartesian monomial organization: absolute-convergence radius for axis offsets is
  |delta_1| + |delta_perp| < |R| (majorant of (R+delta_1)^{-2j-1} (delta_perp^2)^j), rho_abs =
  2.41/n vs rho = 1.73/n -> monomial cancellation ~0.145 p digits (4-5 digits at p = 30).
  Prediction: (a) in plain monomial form fails 1e-13 inside ~8 cells; an (l,m)-organized (b)
  confines the cancellation to precomputed geometry moments.
- Gila evaluates N^3 offsets (egoToe), not (2N)^3: 2.1e6 at 128^3; kernel evaluations ~3e10.
- Offsets with sep <= 4: 9^3 = 729 -> the near band is cheap whatever method; accuracy is the issue.

## Families
(a) Cartesian Taylor about the centre ............ OPEN  (famA, round 1, relaunched)
(b) Spherical addition theorem / multipole ....... OPEN  (famB, round 1, relaunched)
(c) One exact axis, two expanded ................. OPEN  (famC, round 1, relaunched)
(d) Subdivision of the difference box ............ OPEN  (famD, round 1, relaunched)
(e) Spectral / Ewald ............................. BLOCKED(all bands), see famE below
(f) Interpolation of the smooth far tensor ....... ALIVE but marginal, see famE below
(g) Derivative-free Taylor coefficients .......... OPEN  (famG, round 1, relaunched)
(k) k-series (moment series) on separated pairs .. OPEN  (kseries, round 1, relaunched)

## Groundwork
1 profile: partial timings lost in crash; redo when machine is quiet (needs quiet anyway)
2 offsets: Gila computes N^3 (egoToe) + reflections egoToeCrc!; to be re-verified in the clean profile
3 far error today, 4 volume identity: reference agent (relaunched)
5,7,8: DONE (geometry, digest in reports/geometry.md)   6: famA

## Round 1 results
### geometry (items 5,7,8) -- DONE before the crash, digest in reports/geometry.md
- rho tables confirmed; rho_m ~ rho/m. Slow band is a FIXED set of offsets independent of N:
  37 offsets with rho>0.5, 141 with rho>0.3, 436 with rho>0.2, 3096 with rho>0.1 (egoToe octant).
- Slender (1/32,1/32,1/512) along short axis: rho<1 only at n>=23, rho<0.5 at n>=46; all other
  offset classes rho<1 at n=2. Only rho_1D(short axis)=1/n shrinks there. kD along that needle is
  tiny (~0.6 at n=23) -> the k-series owns that needle.
- mu_n(s) = 2 s^{n+2}/((n+1)(n+2)) even n, 0 odd. Exact-rational identity: 36 signed face-pair
  moments = volume integral of (d_a d_b - delta_ab lap)|x-y|^{2j} EXACTLY (sign rule in BRIEF).
- pairMoments(..., Rational{BigInt}) fails (sqrt(off2) at moments.jl:442); use BigFloat panels.
- Error target: default storage Float32, default solver relTol sqrt(eps); FFT Float64 floor 5e-15;
  anti-Hermitian floor measured 6.5e-14 relative on 6^3. Far entries 2e-2 (2 cells) .. 2.4e-5
  (128 cells) vs self term O(1): per-entry relative accuracy is the honest target.
### famE (families e, f) -- DONE before the crash, digest in reports/famE.md
(e) BLOCKED(all bands) by structure: spectrum T^_ab = V_t prod sinc^2(xi_i s_i/2)
  (delta_ab k^2 - xi_a xi_b)/(f^2(|xi|^2-k^2)) verified to 3.4e-13 vs real space. Plain alias sum
  needs M ~ 4e11 per axis; Ewald damping fixes aliasing (M = 2 at eta s <= 2.5) but the pole at
  |xi| = k is fatal: the inverse DFT of the periodized spectrum is the P-periodization of a
  NON-decaying kernel, error e^{-Im k P s}; real f: O(1) error at every P; f = 1+0.1i needs
  P ~ 1651 (vs 256) for 1e-13 at N = 128 -> 768 GB. Even if fixed: ~1e11 flops vs 5e9 direct.
(f) ALIVE, marginal: analyticity radius = cell gap exactly; Chebyshev degree for 1e-13 at
  lambda/32: 18 (H = 32), 26 (H = 64); direct fraction 0.10 over 128^3 -> 9.6x at best, but
  interpolation arithmetic ~2.1e3 flops/offset: pays only if the direct method costs > 2.2e3
  flops/offset; equispaced nodes never reach 1e-13 (Runge); nothing for lambda/8, lambda/4 or
  the 2-8 cell band.

### famG (family g) -- DONE 16:35, report reports/famG.md. STRONG RESULT.
- Route (i), the (G,S) pair recurrence for the Taylor coefficients c_alpha of g(R+delta): G = g(R+delta),
  S = |R+delta| G; grad S = ik (R+delta) G, P grad G = (R+delta)(ik S - G), P = |R+delta|^2. Two-level
  recurrence, seeds a_0 = e^{ik rho}/(4 pi f^2 rho), b_0 = e^{ik rho}/(4 pi f^2), pivot = largest index.
  Float64 digits lost <= 1.4 for kR <= 3 through order 40; <= 2.15 through order 30 at all kR (the
  floor at kR = 100/300 is the seed e^{ik rho}, k rho eps). Cross-checked 4 routes to 1e-65 at 220 bits.
- Assembly T_ab = (1/V) sum_{beta even} mu_beta [(beta_a+1+delta_ab)(beta_b+1) c_{beta+e_a+e_b}
  + delta_ab k^2 c_beta]; SIGN: equals Gila's srfSum! output with no extra sign or factor.
- 68 cases (cubic 1/32, 1/4; (n,0,0),(n,n,0),(n,n,n), n = 3..16; slender 4 offsets; f = 1, 1+0.1i):
  max truncation 9.2e-15 at measured q*, max rounding 1.42 digits, max cancellation 38.7, max tensor
  error 9.8e-15 rel. to largest, worst per-entry 2.5e-14 over 368 entries. q* = 30 (3 cells), 24 (4),
  18 (6), 14 (8), 10 (16), 8 (16,16,16) at lambda/32; floor 18 at lambda/4 from k 2s = pi oscillation.
- Cost: 17 ns per coefficient, t(q) = 17 ns [(q+3)(q+4)(q+5)/6 + (q/2+1)(q/2+2)(q/2+3)/6]: 8.8 us at
  q = 10, 127 us at q = 30. Speedup vs egoSrfFxd!+srfSum!: 39x (3 cells) .. 151x (16 cells). NOT "a few
  hundred flops" (30k at q = 10) but 128^3 in seconds is within reach: typical far offset q* ~ 4-6.
- Loses only at slender (0,0,32): rho = 0.708, q* = 68, 1.2 ms (2.2x slower than Gila), accuracy fine.
- BLOCKED sub-routes with numbers: Helmholtz march from exact Cauchy data (3.5 digits at order 30);
  addition theorem expanded into Cartesian MONOMIALS (3.0 digits at order 16, amplification 10^{0.33 n},
  Legendre-in-monomials growth (1+sqrt2)^l) -- NB this is not the (l,m)-organized famB route; 1D ODE +
  angular recovery (cond 10^{0.33 n}, 65-125x more expensive); AD (correct, 37-94x slower in Float64).
- Root's round-0 prediction (0.145 p digits monomial cancellation) was WRONG: measured ~0.3-1.4 digits
  total. Reason: binom(-1/2,j) decay and the 2/((n+1)(n+2)) moment suppression damp the alternation.
- Still missing for (g)/(a): a PROVEN remainder bound (q* measured), the 2-cell band (q* would be ~215
  by rho^q, i.e. not with a plain centre expansion), slender short-axis band, and the actual build.

### kseries (family k) -- DONE 17:05, report reports/kseries.md (521 lines)
- Bound proved: |R_N| <= (I_{-1}/4 pi |f|^2) x^{N+1}/(N+1)! min(e^x, 1/(1 - x/(N+2))), x = |k| D_max;
  tight to +-3 terms over 233 rows. Assembly amplification Lambda = sum I_{-1}/(4 pi |f|^2 V_t)/max|G|:
  15-24 (lambda/4), 68-1405 (lambda/32), 2.2e3-8.1e4 (slender). Tensor error = alpha eps cnc Lambda,
  alpha in [0.0014, 0.65].
- Float64 band edges for 1e-13 on the assembled tensor: 4 cells (lambda/32), 3 (lambda/8), 2 (lambda/4),
  NONE for the slender cell (eps Lambda = 4.8e-13 already at 2 cells). BigFloat(128) moments: 1e-31..1e-28
  everywhere. BigFloat summation of Float64 moments helps only the slender cell (16-60x).
- Cost kills it beyond the touching shell: pairMoments scales mMax^2.6-5.1 (ladder re-run per m on 3D
  pairs); per offset 32 ms at 2 cells lambda/32 (Gila 13.9 ms), 671 ms at 32 cells (Gila 0.55 ms);
  band-edge cost 0.26 s (lambda/32, 4 cells), 2.4 s (lambda/8, 3), 6.5 s (lambda/4, 2) of Float64 moments.
- Shifted exponential: outer series 11 terms (lambda/32), 19-21 (lambda/4), cancellation <= 1.5 at every
  separation -- but the centred moments cannot be generated stably: naive binomial loses n log10(2.3N)
  digits; divergence recurrence subtracts n R_0/d per step; the q = (2 R.delta + |delta|^2)/R_0^2 route
  converges iff (2N+3)/N^2 < 1 (N >= 4 on axis), i.e. family (a)'s own radius. BLOCKED as a merge.
- Side findings: Gila's egoSrfFxd! on the slender cell at (0,0,8) has 1.79e-3 relative tensor error
  (order-5 rule; 2.8e-5 on the worst face pair). Float64 pairMoments loses 5.5 digits at mMax = 48.
- Root's reading for the join: the k-series is the tool only where its moments are cheap AND Lambda is
  small: the touching shell (already done in notes/moments) and possibly sep 2 at lambda/32. For sep 2
  at coarser cells and for slender short-axis offsets use subdivision (d) or exact-axis (c) on top of
  the (G,S) Taylor route; slender (0,0,n), n < 23: ~21 offsets only, BigFloat k-series (kD ~ 0.6) is
  acceptable there if (c)/(d) fail (cost ~0.2 s/offset once per shape).

### famC (family c) -- DONE 17:35, report reports/famC.md
- As specified (one axis exact with transverse coordinates frozen) it does not exist: the line integral
  of e^{ik rho} rho^{-l} is an incomplete-Hankel object for every l >= 1.
- Inverted roles WORK: r^2 = rho(t)^2 + xi with xi = 2 R_2 y + y^2 + 2 R_3 z + z^2 a polynomial in the two
  transverse coordinates -> expand F(r) = H(r^2) in xi (D = (1/2r) d/dr acts in closed form on
  e^{ikr} r^{-l}), transverse moments exact per level, the along-R axis carries a Taylor series whose
  coefficients come from the closed ODE system G_l' = ik(R_1+t) G_{l+1} - l (R_1+t) G_{l+2}. Ratios
  tau_xi = xi_max/rho_min^2 and tau_t = s_1/R_0; choose the axis with largest |R_i|.
- Accuracy <= 1.3e-13 per entry from 3 cells outward, all directions, lambda/32 and lambda/4, f = 1 and
  1+0.1i; digits lost <= 1.07; M*/Q*/us = 25/28/142 (3 cells), 13/22/53 (4), 7/14/18 (8). Comparable
  to famG route (i) (127/70/18 us). Slender (16,0,0): 9 us at 9.4e-14; needle (0,0,32): 150 us vs
  famG 1.2 ms; needle n <= 16 BLOCKED (tau_xi = 1.13 at n = 16).
- BLOCKED at 2 cells by an identity: tau_xi = 1 exactly even with two exact axes (nearest box point at
  distance s, transverse excursion s). Only subdivision moves it.
- Centred moments (r - R_0)^n CAN be generated cancellation-free by expanding rho itself
  (rho_q = -(1/2R_0) sum rho_i rho_{q-i}; <= 1.4 digits through n = 20) -- but that is the cell-size
  expansion in disguise; the moments ladder is a detour.
- Invariances bit-exact; homogeneity T(alpha s, f/alpha) = alpha^2 T(s, f). Q rule not proven (bound
  with e^{1.5 k R_0 theta} valid but 1.8x loose at kR = 22); M rule never violated in 110 cases.

### famB (family b) -- DONE 18:00, report reports/famB.md (709 lines). WINNER ON COST, equal accuracy.
- Sign +, no factor (bit-exact vs reference at (5,1,0)). Addition theorem with P_l(R^.delta^) needs
  (-1)^l (measured 1e-41 with it); moot: the triangle weight's parity kills every ODD l. Surviving
  classes: diag cos/m even; xy sin/m even >= 2; xz cos/m odd; yz sin/m odd.
- Real solid harmonics as integer polynomials (C_l^m recurrence), geometry table EXACT rational once per
  shape: 14.65 s at L = 40, nMax = 12 (cancellation 5.8e8 inside the table, hence exact arithmetic);
  frequency contraction 31 ms. Helmholtz identity holds to 1.7e-89.
- h_l upward recurrence accurate to 3e-15 for l <= 40, |z| in [0.3, 300], Im z in [0, 30] (contamination
  decays as j_l/j_0 while h_l grows); closed finite form WORSE (6e-10 at z = 30, l = 40). j_l = Re h_l
  destroyed for l > |z|; Miller downward gives j_l to 7e-15; the repair improves Im part up to 11x,
  costs 850 ns flat, needed for the last digit.
- Proven bound |Rem_L| <= (17|k|^3/(4 pi |f|^2 V_t)) sum_{l>L even} (2l+1) sqrt(2l+5) |k|^l
  e^{(|k| r_d)^2/(4l+6)}/(2l+1)!! W_l hb(l+2, kR), W_l = int_D w |delta|^l exact; over-selects L by
  1.5-2x (bound/actual 1.8e2-4e6). Measured L(1e-13) = 13/log10(1/(0.7 rho)) (weight vanishes at the
  corners -> 0.7 rho effective); L set by rho not kR. n_max = 6/10/12 at kd = 0.34/1.36/2.72.
- Conditioning: per-offset l-sum cancellation <= 8.4 (lambda/32), <= 67 (lambda/4, axial 11 entry);
  n-sum 3-4; all the big cancellation is in the exact table.
- Accuracy vs 220-bit reference, 163 tensors, from 3 cells out, all shapes (1/32, 1/8, 1/4, slender),
  all directions, both f: <= 4.2e-15, <= 1.3 digits lost, rounding-limited (BigFloat same L: 1e-19..1e-22).
  Symmetries and f-homogeneity bit-exact.
- Cost 391/1250/2731/4972 ns per offset at L = 10/20/30/40, zero allocations, t(L) = 3.1 L^2 + 80 ns;
  whole 128^3 far field at lambda/32: 2.37 s single-threaded with the Miller repair (0.59 s without),
  96.5% of offsets at L <= 8.
- Gaps (truncation, not conditioning): (1) the 12 octant offsets with |n| <= 2.56 (rho = 0.87): L = 94 for
  6.6e-14, not 1e-15 by L = 100; (2) slender needle (0,0,n), n < 23: divergent (rho > 1); n = 24: 1.3e-4
  at L = 40; n = 32: 1.5e-9; n = 64: 3.4e-16. -> subdivision (d) or k-series for these bands.
- Root's decision: family (b) is the far-field core of the deliverable; (g)/(a) and (c) become the
  independent cross-checks in the overlap. Near band pending famD.

### Independent Mathematica check (Paul, 2026-09-06 evening) -- item 4 by a third route
NIntegrate of the volume form (1/V_t) int_D w [(d_a d_b + delta_ab k^2) g](R + delta) at D = (5,1,0),
s = 1/32, f = 1, WorkingPrecision 30-40, box split at 0:
  T[1,2]: agrees with the 220-bit face-pair reference to 3.4e-14 (Mathematica's own error estimate 1e-14);
  T[1,1]: first run (GlobalAdaptive stalled, its estimate 1.4e-9) agreed to 4.6e-8; the rerun with a
  fixed 30-point Gauss-Kronrod rule per sub-box reproduced 0.001570738221266370299274 +
  0.000360155079207178606379 i, i.e. every printed digit. Sign +, no extra factor, normalization
  confirmed with no shared code. Paul's own Mathematica, not a sub-agent.

### Timing caveat (root, from Paul): every microsecond/second figure in the round-1 reports was
measured while 4-10 other agents' Julia processes were running. Relative comparisons on the same run are
fine; absolute build times MUST be re-measured on a quiet machine by the final bench (bench_before.jl,
bench.jl) before they go in the document.

### famA (family a, item 6) -- DONE 18:45, report reports/famA.md (618 lines)
- Sign + confirmed (8e-26 at (5,1,0), 1e-40 at (8,8,0)).
- Item 6: Hobson route (radial derivatives + nu-sum) loses 7.3-8.4 digits by |alpha| = 25 at kR <= 3:
  DISCARDED. Coupled Cartesian recurrence r^2 d_j u = ik x_j w - x_j u, d_j w = ik x_j u (same object
  as famG's (G,S)) loses 1.39 digits at p = 25, kR = 1 axis; < 2 for kR <= 3 and kR >= 30; peaks 4.9
  digits at kR ~ 10 on individual derivatives. The degree sums S_p lose 3-6 digits relative to
  THEMSELVES but |S_30|/|S_0| = 1e-9..1e-29 and sum|S_p|/|sum| = 1.0-1.74, so the accumulated tensor
  loses -1.06..+0.68 digits over the whole grid. Float64 suffices; no extended-precision build path.
- Bound proven with explicit constants (ball min-modulus |R| - rho for rho <= |R|/2, sqrt(|R|^2/2 -
  rho^2) beyond; Helmholtz alternative for the diagonal; tail sharpened by the moment weights);
  2.0-7.7x pessimistic in p. Its convergence criterion sqrt(sum s_i^2) < |R|/sqrt2 is the complex-ball
  (null-cone) criterion; the real series converges for rho < 1 but with effective ratio ~0.73 at
  (2,0,0): 7.7e-14 at p = 100 (consistent with famB's L = 94 -> 6.6e-14). 2^3 subdivision: 5.5e-14 at
  p = 36, 1.7e7 flops. Slender needle needs n >= 33 (390% error at n = 8, 1.6e-14 at n = 33).
- 185 offsets vs reference: digits lost -0.11..+1.84, floor 3e-16..1.5e-14; p_truncation saturates
  beyond ~16 cells at 8/12/16-18 (lambda/32, /8, /4): constant far cost 6408 flops, 7.7 us/offset
  (34-185x faster than egoSrfFxd!). Symmetries exact; homogeneity 1e-47.
- Root: (a) == (g) in substance; both are the independent cross-check of (b) in the overlap.

### famD (family d) -- DONE 19:05, report reports/famD.md (512 lines)
- Independent BigFloat Gauss-Legendre VOLUME reference (work/famD/refQuad.jl) matches the face-pair
  pairKer reference to 1e-63..1e-65 (cubes) and 1.4e-62 (slender): third confirmation of the volume
  form, sign +, normalization. pairKer stalls (> 15 min/offset) on slender cells -> verify.jl must use
  refQuad there. 62 references cached in work/famD.
- Bound derived (singularities on the complex line at |t| = |X|/|delta'| exactly), never violated in
  364 rows; looseness in degree median 2.29x (1.75..7.74): a certificate, not a selector.
- rho_j falls like 1/(m(n-1)+1), not 1/m: at n = 2, m = 1,2,3,4,6,8 -> 0.866, 0.522, 0.433, 0.333,
  0.243, 0.190. Optimum: 2 cells -> uniform m = 2 ((2,0,0)) or 3 ((2,1,1)); >= 3 cells -> m = 1.
  m = 8 at 2 cells costs 3.5x more than m = 2. Non-uniform splits gain 1.00-1.09 (cubes), 1.5 (slender).
  ODD m is a trap for offsets with a zero component (a sub-box centred on the zero axis: slender
  (0,0,2) u3 gives rho_j = 5.66 vs u2 0.99).
- Slender needle (0,0,n): n = 2 needs ~200 boxes for rho_j < 0.5 (10x10x2 graded, p = 25, 2.7e7 flops
  with the derivative-table base); n = 8: 8x8x1, p = 21, 3.1e6 flops. Converges; cost is the issue.
- Cancellation between sub-boxes 1.0-4.0; monomial level <= 0.62 digits. Float64-safe.
- With the Cartesian derivative-table base the 2-cell offsets cost ~2.8e6 flops (1e4x the budget);
  with the (b) base per sub-box cost is ~3 L^2 ns, so famBsub decides the real cost.

### overlap (b vs g vs c vs Gila) -- DONE 19:45, report reports/overlap.md
- Wrappers in work/overlap/common.jl. 13797 octant offsets x (cube 1/32 f = 1, f = 1+0.1i; cube 1/4
  f = 1), 10893 slender offsets, 30 random signed offsets.
- AGREE TO ROUNDING where properly truncated: cube 1/32 worst pairwise 4.2e-14 (3-cell band), <= 9.8e-15
  from 4 cells; zero above 1e-13 out of 27594; random signed <= 1.6e-15 (sign handling identical);
  slender g-c <= 5.4e-14 everywhere.
- The EMPIRICAL truncation rules of (b) and (c) are wrong at large kR: famB's rho-only L has no kR
  floor (5.5e-10 at lambda/4 (23,23,20) with L = 10; needs a flat L = 16 across rho = 0.045-0.125;
  slender (0,0,36) needs L = 54 not 40); famC's tau_xi-only M likewise (4e-11). famG's q rule never
  short. Every adjudicated disagreement (10597/13797 at lambda/4, 275/10893 slender) is pure
  truncation, fixed by raising one index. PRODUCTION RULE: select L from the proven bound (which
  carries hb(l, kR)), never from the rho-only shortcut.
- Slender excluded sets contiguous in n3 per (n1,n2); famG's q* > 60 set (344 offsets, rho > 0.464)
  contains famB's (159) and famC's (143).
- Defects: famC gTab overflows (seed R_0^{-l}) on 57 slender + 15 cubic offsets -> restrict/fix if
  (c) is used in verify.jl. Gila today: 3-6e-13 at lambda/32, < 1e-14 at lambda/4, but 4.7e-7 at
  slender (0,0,36) (1.1e-5 per entry).

### reference (items 3, 4) -- DONE overnight, report reports/reference.md (400 lines)
- Volume formula pinned: G_gila[a,b](R) = (1/V_t) int_D w [(d_a d_b + delta_ab k^2) g](R+delta), sign +
  on diagonal and off-diagonal, no extra factor; the three other sign conventions are off by
  0.57-2.0; volTensor (BigFloat GL, ord 32, graded, split at 0) vs refTensor: 3.7e-65..1e-49 at 18
  offsets, all shapes, both f. ref.jl API: refPairs, refSum, refTensor, volTensor; cache
  refcache/reftensors.txt (append-only text). 25 s per tensor under load, 250 s for slender (0,0,2).
- Gila TODAY (item 3): lambda/32: per-pair 1e-15..9e-14, amplification 36..1300 (axis entry grows
  as n^2; body diagonal constant 312), tensor 5.0e-14..8.7e-13 (not 1e-10). lambda/8: 14..326, tensor
  5e-15..3e-13. lambda/4: 5..163, tensor 1.6e-15..2.8e-13 (per-pair error grows with separation on
  the axis from phase rounding 2 pi f r eps). Amplification is a FINE-cell effect (~(R/s)^2).
  SLENDER: short axis (0,0,n) wrong by O(1): 0.79 (n=2), 3.7e-2 (4), 1.8e-3 (8), 3e-6 (16), 4.5e-6
  (20), 1.9e-6 (24) -- a 1/512 boundary layer across a 1/32 face; long axes amplification
  4.3e3..4.0e4 -> 3.2e-12..2e-11 despite per-pair 1e-15..8e-14.

### famBsub -- DONE overnight, report reports/famBsub.md (cubes only; slender section missing)
- Sub-box weight affine al_i + bt_i t (NO piece may straddle 0: odd m forces an extra split, so
  m = 3 costs what m = 4 costs); moments nu_n exact rationals; odd l survives -> the (-1)^l of the
  addition theorem is mandatory (3.7e-2 error without it); reflection rule T^{(sigma j)}_ab(V) =
  sigma_a sigma_b T^{(j)}_ab(sigma V) -> one octant of tables. Verified exactly.
- The 12 two-cell offsets SOLVED: m = (2,2,2), L_j <= 48 from the bound: 4.2e-15 max, 4.3e-15 per
  entry, both f; canc between sub-boxes <= 1.55; rounding-limited (BigFloat 20-100x smaller).
  Sub-box bound over-selects L by only 1.2-1.3 (vs 1.5-2 for the whole box). L rule on a sub-box is
  the naive 13/log10(1/rho_j) (no 0.7: affine weight is O(1) at corners).
- Octant split is the only split ever worth it for cubes (refining x8 divides rho by 2 -> L by 1.5
  -> cost x3.6). Beyond 3 cells famB alone is 3-7x cheaper. Table 41 s exact rational (L = 48).

### auditG -- DONE overnight, report reports/auditG.md (658 lines). Route (i) correct; caveats:
- Recurrence and assembly re-derived and confirmed to 1e-76 (256 bit) with an independent Taylor
  arithmetic and a Cauchy-integral extraction; trace identity tr T_q = 2 k^2 <g>_q holds to 2e-58.
- 290 (case, f) rows vs 220-bit reference: in Gila's range (cubic 1/32, both slender shapes, 200
  rows, |n| to 62, f in {1, 1+0.1i, 1+1i, 0.37, 3+0.3i}) max tensor error 4.8e-15, per entry 8.5e-15,
  q* <= 14 far out. Failures: (A) rho >= 1 (slender (0,0,n), n <= 22) diverges; boundary at n = 22.6
  exactly; within q <= 40 no (0,0,n) with n <= 39 reaches 1e-13; (B) k 2s >= 10 (cells > lambda/2,
  outside Gila): Float64 cancellation 10^{0.2 k 2s}; (C) rho -> 1: (2,0,0) needs q ~ 100 where
  cancellation is 1.8e4 -> best 1.1e-13 (U-curve) -> route (i) alone cannot do 2 cells in Float64.
- Im part: the tensor is one complex sum, so Im is accurate only to eps |Re|; relative Im error
  1e-13..4e-12 when |Im|/max < 1e-3 (f = 0.37, slender). NB family (b) separates Re (y_l) and Im
  (j_l) sums for real f -- structural advantage; make the unified code exploit it.
- Stress: order 50-60 at kR = 30 loses 6-9 digits (never needed: order is set by rho). Seed
  e^{ik rho}: double-double phase (12 flops, seed.jl phsDD) removes the k rho eps floor: kR = 3000
  3.35 -> 0.88 digits. USE IT for the h_0 seed of (b) at large kR (lambda/4, 128^3: kR ~ 350).
- Cost re-measured under load 20: 17.6-18.6 ns/coefficient confirmed; q rule 9.7/log10(1/rho)+2
  (famG's 17 is 3.35x too costly); 32^3 block 1.28 s vs Gila 23.3 s (18x), sweep avg 39 us/offset.
- Symmetry tests offset->-offset, reflections, transpose are EXACT BY CONSTRUCTION in Taylor routes
  and prove nothing; permutation symmetry is the honest one (1.8e-15 in range).
- Gila's fixed rule at slender (0,0,20),(0,0,24): 4.5e-6, 1.9e-6 (quadOrd keys on cells, not length).

## Root's design for round 2 (2026-09-07 morning)
FAR CORE: family (b) (famB.jl + famBsub octant split), L per offset from the PROVEN bound (kR-aware),
n_max from a proven bound, Miller j_l repair, double-double phase seed; Re/Im as separate real sums
for real f. NEAR-NEEDLE FALLBACK: BigFloat(128) k-series (moments.jl pairMoments on the 36 face
pairs, BigFloat assembly) wherever the (b)+octant bound needs L > L_max: that is the slender set
(n1, n2 in {0,1}, small n3), ~120 offsets, once per shape (frequency-independent moments), with the
kseries proven bound. Band selection = cheaper method meeting 1e-13 per the bounds. CROSS-CHECKS:
(g)/(a) and (c) in the overlap (agree to rounding, overlap report). Deliverables: farfield.jl,
verify.jl, bench.jl, farfield.tex.

### theory -- DONE 09:45, report reports/theory.md; fragments work/theory/{volume,expansion,bounds,join,registry}.tex
- 28 pages compile clean. Bound checks in BigFloat: (a) j_l 1.0006-1.5 in the used regime; (b) h_l
  equality at l = 0; (c) l-truncation end-to-end bound/truth 1.3e3-3.0e7 -- the constant 17 is loose
  by 32x-34000x (that is where over-selection lives; sum_m |Y_lm| costs <= 2x); (d) n-truncation
  1.12-4.5 for l >= 12; (e) k-series 4-266; (f) radius along R + t delta exactly |R|/|delta|: the
  series converges for rho < 1 (famA's sqrt2 rho < 1 is about the bound, not the series).
- Band edges from the bounds alone reproduce the empirical split: 12 two-cell offsets -> octant
  (L_wb >= 78, L_oct <= 56); (2,2,0) whole box; slender: k-series set with L > 48: 83 offsets =
  (0,0,n) n = 2..21 and (0,1,n),(1,0,n),(1,1,n) n = 2..22; with L > 64: 64 offsets (n <= 17), where
  |k| Dmax in [0.28, 0.60] -> 13-16 k-series terms.
- New: for a cubic cell the whole l = 2 shell vanishes identically (shells {0,4,6,8,...}).
- Inconsistency "famBsub L_j = 48 vs theory 56": resolved by root -- famBsub capped L at 48.
- Guard: a finite l-cap on a divergent tail reports a finite L; require ratio < 1 at the top.

### unify -- DONE 10:30, report reports/unify.md (866 lines); deliverable notes/farfield/farfield.jl (1035 lines)
- Routes: (i) whole box, (ii) octants, (iii) BigFloat(128) k-series; L from the proven bound with
  Lmax = 56, nCut from the n-tail bound, budget tol est(R), tol = 1e-13; (i) short-circuits when it
  converges (cost(i)/cost(ii) <= 0.107). Routing 128^3 cubes: (i) 2097132, (ii) 12-15, (iii) 0;
  slender 64x64x128: 524066 / 142 / 72 ((0,0,n),(0,1,n),(1,0,n),(1,1,n), n = 2..19).
- Accuracy vs 403 references (f = 1, 1+0.1i): <= 8.6e-15 max-norm, <= 1.2e-14 per entry; f = 1+1i
  6.3e-14; route (iii) vs graded volume reference 2e-17..1.6e-16; overlap vs famG <= 1.0e-14 over
  4085 offsets at lambda/32 and lambda/4; symmetries bit-exact, permutation 1.15e-15, homogeneity
  1.1e-14; Float32/BigFloat generic; positivity lambda_min -4.47e-15 (Gila) -> -2.45e-15.
- Speed: 128^3 far field lambda/32 2.05 s single thread (load 3), 0.335 s on 8 threads; threaded ==
  serial bitwise. Shape table build 474-643 s per shape (L = 56, nMax = 12, 1024-bit contraction),
  19.5 MB, reload 1.9 s -- THE one cost that undercuts "seconds" for a new shape; fix pass: 256-bit
  suffices (cancellation 7e11), lazy L. Re/Im as separate real sums for real f; for complex f the
  small part of an entry is accurate to eps|entry| only (formulation floor).
- Defects fixed vs sources: famB tail bound was a partial sum (vacuous at lTop); Float64 tables are
  noise beyond L ~ 26; bslj! seed loses 7.7e-14 phase at kR = 348 -> double-double seed;
  Threads.maxthreadid() vs nthreads(); famD refQuad cuts need grading to the near point.

### bound2 -- DONE 11:45, report reports/bound2.md; work/bound2/{bound2.jl, bounds2.tex}
- Two exact identities replace the loose steps: (1) integration by parts onto the weight
  (w_i'' = delta_{-s} - 2 delta_0 + delta_s, w = 0 on dD): the harmonics are never differentiated,
  two derivatives cost 4/(s_a s_b) instead of hb(l+2)/hb(l); (2) one joint Cauchy-Schwarz over m
  with sum_m Y_lm^2 = (2l+1)/(4 pi): removes the 17 and the sqrt(2l+5). Singular-side variant: exact
  l^2 Hessian bound via Schur (true/bound 0.71..1.0).
- 116 BigFloat rows, no violation: bound/truth median 49 (regular org.) vs 1.5e5 for the theorem in
  use (median gain 3233x); L over-selection vs measured 1.40 -> 1.14; per-offset cost gain median
  1.47x; (2,0,0) certified at L = 126 (old: none below 136); worst octant of a 2-cell offset L = 46
  instead of 55. boundL(s, R, f, L) 3.8-9.7 us per offset after a 31 ms per-shape farMom.
- IMPORTANT: the old theorem bounds the singular-side truncation; farfield.jl truncates the REGULAR
  side (tables Ad/Ao); the regular remainder is a median 17.5x larger at the same L. The old slack
  covered it; the new Theorem A bounds what is actually truncated. The fix pass must use it.

### deliver -- DONE 12:30, report reports/deliver.md; notes/farfield/{verify.jl, bench.jl, ref.jl, refquad.jl,
### mkrefcache.jl, crosscheck/famG.jl, refcache/reftensors.txt (1059 records), ksrcache/, tables/ (26 files)}
- verify.jl parts a-i run in 5.8/14/4.5/22/86/108/38/553/24 s (h = 492 s of Gila's wekTrp x6).
- Numbers: identity 0//1 in 18 rows; farTensor vs 423 references <= 8.6e-15 / 1.2e-14 per entry;
  digits lost <= 1.59; slender long-axis Gila amplification up to 167000 (n = 64); route (iii) vs
  volume reference 2.3e-17..1.6e-16; k-series band x in [0.28, 0.61], N 12-17, Lambda <= 1038.
- bench.jl 16^3: staged before == GlaVacOprMem bitwise; after vs before egoFur 2.06e-14 (Gila's
  error; worst separated offset 5.6e-13 at (0,5,5)); operator on a random vector 2.5e-14.
  32^3 single thread: far fill 24.75 s -> 0.042 s (588x); whole build 57.2 -> 33.2 s = wekTrp at
  intOrd 48 (32.3 s). 12 threads: 5.81 -> 0.010 s; build 12.7 -> 7.6 s.
- farfield.jl lacks: a table directory keyword (TABDIR default does not exist -> silent 500 s
  rebuild); needMom world-age warnings (load moments.jl at include or Ref the functions);
  farRouteStat should return (ii) offsets; ksr cache should store N and Lambda; an exported
  predicate for the farBlock! index partition.
- Missing references: slender short axis n = 3, 6, 32, 64 (pairKer stalls); refquad.jl is the tool
  (3.2e-45 vs face-pair reference at c32 (8,0,0)).

### prep -- DONE 13:40, report reports/prep.md; work/prep/{farfield_v2.jl, farfield_v2.diff}
- Theorem A (whole box) / B (octants) of bound2 integrated with hm = :bd (proven hb majorant; the
  Hankel recurrence has only a measurement); cost of :bd 1.001-1.008x (cubes), 1.2x (slender).
- TOL must become 1e-14: the old 1e-13 was calibrated against a bound loose by 1.5e5; with Theorem A
  (loose by 49) tol 1e-13 delivers 1.6e-13 per entry; tol 1e-14 gives 9.3e-15 per entry (old 6.3e-14),
  max-norm 8.6e-15, +0.7% terms. L_old/L_new 0.83-1.17; near band 6-8 shells fewer; far rho <= 0.027
  2 shells MORE (the old L = 8 was never a certificate of the regular-side sum). Bound never violated
  except at the Float64 floor (max excess 23 eps). Routing 128^3 cubes 2097132/12/0; (2,2,0) class
  moves (ii)->(i) at c8, c4; slender 524102/111/67 (band n3 <= 18). Mean terms 226 vs 185 (+22%).
- Tables: <= 2^11 terms x 2^39.4 cancellation -> 256 bits reproduce the 1024-bit table BIT FOR BIT
  (all 321373 numbers, 4 shapes); 192 bits: no Float64 value changed. Build 87-200 s vs 474-828 s;
  lazy L sized per use (24/0 lone far offset, 48/51 for a 32^3 block): cold 32^3 build 202 s vs
  828 s; one far offset from cold 1.3 s vs 828 s. The claim "L = 24 covers sep >= 3" is FALSE
  (106..1310 offsets need more; max 50-56).
- est(R) over-states (est/max|T| 1.1-6.3, 135 on the needle) -> rule certifies tol est/max|T|.
  Proven direction-free lower scale estLo from tr T = 2 k^2 S: 0.04-0.98 of max|T| but vacuous at
  lambda/4; kept as keyword scl = :low. A posteriori certificate bnd/(max|T^(L)| - bnd) <= 4.5e-14.
- Bugs found: nCutVec vacuous (evaluated at 2 min(s) where rho > 1 -> nMax for every l); fixed with
  budget/256 at the true radius: nCut max 6/9/11/6, nMax = 12 certified. Exact-zero threshold must
  track precision. Selection layer must be Float64: Float32 farTensor returned NaN (overflow of
  (2l+1)!! above l ~ 30); fixed, Float32 -> 6.2e-8.
- Verification on the copy: t2_ref 8.6e-15/9.3e-15; symmetries bit-exact; overlap <= 1.1e-14;
  positivity -2.449e-15.

### refill -- DONE 13:45, report reports/refill.md
- Eight slender short-axis volume references ((0,0,3) ord 28, (0,0,6) ord 24, (0,0,32) ord 20,
  (0,0,64) ord 16; f = 1, 1+0.1i; last-order changes 5e-37..9e-32) appended to
  notes/farfield/refcache/reftensors.txt as kind tns:refQuad. farTensor vs them: (0,0,32) route (ii)
  9.2e-16, (0,0,6) route (iii) 5.8e-17. verify.jl d picks them up; part (c) still prints "no cached
  reference" for slender ax3 sep 3/6/32/64 because it requires the 36 face-pair values -> merge
  agent: make part (c) fall back to the tensor reference (tensor error only) when pairs are absent.

### audit2 (adversarial audit of farfield.jl) -- DONE 14:20, report reports/audit2.md (1217 lines)
- 288-tensor shape sweep (12 shapes, aspect 1e-3..1e3, 6 offsets, 4 f): everything collapses onto
  |k| r_d (half-diagonal of D): |k| r_d <= 3.17 -> per entry <= 1.4e-13 (two shapes worse, see D2);
  |k| r_d = 4.5/4.8/5.8/9.0 -> 1.7e-10/1.3e-9/6.4e-7/9.6e-3, growing as (|k| r_d)^25.6.
- D9 (MOST SERIOUS, not fixed by v2): the k-series inside j_l is capped at nMax = 12 at table
  build; nCutVec can only lower it and its budget was evaluated at a vacuous radius -> silent
  saturation; the l-bound understates the error by up to 7.2e8; route (ii) immune by 2^26. Within
  Gila's range (<= lambda/4 cubes, k r_d <= 2.72) OK; lambda/2 cube 7e-13..3e-12 per entry.
  FIX (merge): adaptive N from the n-tail bound, table extended lazily, REFUSE if N > stored.
- D2: lambda/128 cube wrong by 4.65e-13 per entry at every separation (nCut cutting 2-4 of 13
  terms; forcing nCut = nMax -> 2.3e-16) -- v2 fixes. D10: L under-selected at large separation
  (r1 (48,48,48) 1.23e-13 -> 3.8e-15 at L+12) -- v2 fixes.
- Bound (task 7): violated in 19/60 random cases; bound/actual 1e-9..419 (median 7.5); 9 are the
  D9/c128 wrong answers, 4 "genuine" at 63-396 eps (rounding floor territory), 6 below rounding.
  -> the document must separate the certified truncation error from the measured rounding error.
- Threading: farBlock! bitwise identical at 1/4/12 threads; farTensor route (iii) has an unguarded
  cache race (MOMC, fs.ksr, Base.include) that did not fire in 30 trials -> lock it.
- Positivity: slender (6,6,12) f = 1: lambda_min -6.6e-14 (Gila) -> -1.9e-14; f = 1+0.1i BOTH
  builds indefinite at the percent level (-9.9e-3 Gila, -1.47e-2 farfield, lambda_max 0.195):
  the shared contact/touching-shell entries are the suspect (moments.tex: Gila's order-9 shell
  rule off by 1.6e-2 on slender cells). lambda/4 cube: both at the Float64 floor.
  M_ab(R) = M_ba(-R) to 3e-15.
- Cost: 32^3 0.033 s (1010 ns/offset), 64^3 0.245 s single thread; farRoute 360 ns/offset of which
  25 ns used (cost model for (ii) evaluated needlessly: 35-42% of the build); slender 64x64x128 cold
  684 s = 72 route-(iii) offsets serial (warm 0.5 s); table build 298-1770 s, 18.9 GB RSS at 1024
  bits (v2: 256 bits, 4x faster; RSS to be measured).
- Verdict on the audited version: misses 1e-13 per entry inside lambda/128..lambda/4 by up to 4.8x
  (per part 18x); items D2, D10 fixed by v2; D9 and the cost items go to the merge pass.

### merge -- DONE 15:40, report reports/merge.md (568 lines); FINAL notes/farfield/farfield.jl (1479 lines)
- = prep's v2 + D9 fix (N per shell from jtail, table extended lazily to NCAP = 40, REFUSES above;
  N: 4 at |k| r_d 0.03, 12 at 2.7 = Gila's range, 25 at 9.0, 39 at 21.9; audit2's failing rows
  1.7e-10..9.6e-3 -> 2.7e-14..1.7e-13; lambda/2 cube 7e-13..3e-12 -> 6e-16..3.8e-15), D2/D10
  confirmed fixed (c128 4.65e-13 -> <= 4.3e-15; r1 (48,48,48) -> 3.2e-15), cost model removed from
  farRoute (358 -> 46 ns; farBlock! 1095 -> 783 ns/offset), route (iii) threaded and locked (cold
  slender 64x64x128 633 s -> 128 s on 8 threads, bitwise identical), five API gaps closed, verify.jl
  part (c) fallback, in-place MPFR momAcc (tables c32/c8/c4/sl 44.5/44.6/53.2/64.7 s at 256 bits).
- Bound, audit2's 60 cases with the final library: bound/actual min 0.118 median 4.0 max 54.5;
  15 "violations", ALL at the rounding floor (bound below eps max|T|), ZERO truncation violations;
  rounding error <= 16.4 eps max|T| with l-sum amplification <= 3.1e3; max-norm error <= 3.65e-15.
  Sentence for the document: truncation certified by the bound; rounding measured <= 16.4 eps;
  a posteriori certificate bound + 20 eps max|T| never exceeded.
- verify (d) 427 references: eMx <= 8.6e-15, per entry <= 9.34e-15. bench 16^3/32^3: far fill
  1271x/893x; egoFur after vs before 2.06e-14/1.66e-14.
- FINDING: slender (6,6,12) Im M at f = 1+0.1i: Gila -9.9e-3; Gila contact + farBlock! -1.47e-2;
  moments.jl contact + farBlock! +3.25e-4 with ZERO negative eigenvalues (f = 1: -4.6e-18 =
  -2.1 eps lam_max). Gila's order-9 rule is off by 12.7% on the contact cell and 117% on the
  touching shell across the 1/512 face. The indefiniteness is the contact path, not the far field.
- Not fixed / not proven: peak RSS 21-25 GB per table build (BigInt churn in hrmPly!/mulRsq!);
  route (iii) 1.9-9.5 s per offset (cold), no cap; TABPRC 192 bits caps BigFloat results; global
  setprecision unsafe for bare threaded farTensor; per-part (Re/Im) floor eps max|T| at complex f;
  est over-states (rule certifies tol est/max|T|); NCAP = 40 a policy; N over-selected (jtail is a
  majorant); moments contact rebuild is a diagnosis, not a deliverable.

### FINAL (2026-09-07 16:40): clean runs + document
- final_runs (quiet machine): bench 12 threads: 128^3 lambda/32 far fill 200.8 s -> 0.200 s (1002x),
  whole build 206.5 -> 6.73 s (contact 5.1 s dominates); 64^3 25.9 -> 0.034 s; 32^3 4.29 -> 0.005 s;
  lambda/8, lambda/4 32^3 9.8/14.9 s -> 0.006/0.008 s. 1 thread: 128^3 far fill 1157 -> 1.49 s,
  build 1190 -> 34.3 s (contact 28.6 s); kernel evaluations before 1.94e10 at 53 ns. egoFur after vs
  before 1e-14..5e-14 (Gila's error). verify a-i all exit 0 (h = 7.5 min).
- Document notes/farfield/farfield.tex + tex/*.tex -> farfield.pdf, 63 pages, 0 errors, 0 overfull;
  writer's report reports/writer.md maps every number to its table file.
- shapetab/ symlinks replaced by copies (self-contained deliverable, 4 x 19.5 MB).

### MEM (2026-09-07 17:50): D8 memory fixed, root
- hrmPly!/mulRsq! rewritten with in-place Base.GMP.MPZ (set!/add!/mul!) on matrices of distinct
  BigInt objects (zeros(BigInt, n, n) aliases one object; bigMat/bigZero! replace it); one scratch
  BigInt threaded through tabWhl/tabOct. Original saved at work/root/mem/farfield_pre_mem.jl.
- Bit-identity (work/root/mem/bitid.jl): 24 tables (3 shapes x nMax 6/12 x whl/oct x lLo 0/>0),
  0 field mismatches. Cold cache files byte-count identical to merge's (16076486/16158624/
  16848551/18965309 B).
- Cold build, 1 thread, 32^3 sizing, load ~2 (work/merge/m5_tab.jl, out in work/root/mem/):
  c32 44.5 s 23.4 GB -> 14.3 s 0.661 GB; c8 44.6/21.8 -> 14.8/0.666; c4 53.2/25.1 -> 15.7/0.698;
  sl 64.7/25.3 -> 18.3/0.738. Smoke (5,1,0) unchanged.
- tex updated (farfield.tex sec:use:cost, bounds, defects D8, join, uncertain); pdf rebuilt,
  63 pages, 0 errors, 0 overfull. Audit shapes (aspect to 1020) not rebuilt: inferred, stated.

## Cross-scale round (unequal cells; prompt notes/PROMPT_crossscale.md; brief scratch/BRIEF_cross.md)
Root's round-0 analysis (2026-09-07 19:20):
- Weight per axis becomes the trapezoid (a, b) = (|sT-sS|/2, (sT+sS)/2); mu_n closed form; parity unchanged;
  Theorem A changes only its measures (ramp-only flat measure, point masses at +-a, +-b); Theorem B unchanged.
  est must scale with V_s (int w / V_t = V_s).  One table per unordered pair (swap symmetry).
- The prompt's 27-piece split: all-plateau piece radius |a| = (r-1) sqrt3 g / 2 vs nearest separated |R| ~
  (r+5) g / 2 (cubic, ratio r): rho_plateau = 0.35 (r=2), 0.87 (4), 1.17 (8), 1.24 (16) -> DIVERGES for r >= 8
  in the near band; must subdivide the plateau.  Gila's pairs are always integer ratios on the gcd lattice
  (GlaExtInf), so the gcd average G(R;sT,sS) = (1/N_t) sum_j sum_j' G(R + c_j - c'_j'; g, g) of EQUAL-cell
  tensors is the candidate near-band route: no new tables, no new bound, no cancellation (weights +1/N_t).
  Cost N_t N_s us per offset (4096 us at r = 16 cubic) -> too slow for the bulk, so the trapezoid whole-box
  table is still needed far out.  To be measured by xtable/xgeom.
- Round 0 agents: xtheory (moments, Theorem A', identities), xtable (trapezoid table + gcd average prototype,
  timing), xgeom (which pairs Gila produces, rho distributions, Gila's accuracy today), xref (220-bit
  references for the matrix, two independent rules), xnear (item 5: thin-indicator regime of moments.jl).

### Cross-scale round 0 results (2026-09-07 23:55; agents killed by the API limit at ~22:30, Julia jobs finished; relaunched to finish)
- xtheory DONE (reports/xtheory.md, xwork/xtheory/xbound.jl, bounds_cross.tex): trapezoid = triangle(b) - triangle(a), so
  mu_n(a,b) = wgtT(n,b) - wgtT(n,a); Lemma ibp' (ramps only; point masses at +-a, +-b); Theorem A' = Theorem A with new
  measures and r_d = |b|; Theorem B unchanged (beta = 0 plateau covered); est with V_s; estLo hypotheses fail at kap 1
  for r >= 4 on xy/xyz (scl = :est stays). 132 exact-rational identities all 0 (incl. swap and gcd decomposition);
  359 bitwise fallback checks 0 mismatches. rho/L per route on the matrix in xwork/xtheory/out_xrho.md.
- xgeom (partial, reports/xgeom.md): ARCHITECTURE FACT: GlaCmpOprVac remeshes a touching coarse region to the fine scale
  (_sndBlk) and builds an EQUAL-cell external block; unequal genEgoCrcExt! runs only for direct GlaOprVac(trg, src) on
  unequal GlaVols (test suite) and for NON-touching composite region pairs (nested refinement). Offsets = 27 r^{|A|} per
  2^3 test geometry, all on the half-integer gcd lattice (2R/g integer). Nearest non-contact |R| = (r+3)/2 g -> Gila's
  adaptive hcubature (rtol 1e-6); realistic 64^3 g vs 4^3 (16 g) block: 1.40e6 offsets, 324 contact, 36540 adaptive,
  1.37e6 fixed rule (1.8 h single thread today); rho_whole >= 0.6 for 15936, >= 1 for 1456. Gila accuracy vs reference:
  pending (cmp.jl).
- xtable (partial, reports/xtable.md, xwork/xtable/farx.jl): trapezoid whole-box table = tabWhl with W = wgtTrpX; stored
  / prod(b); equal-cell paths bit-identical (120/120 tensors). 27-split piece tables (8 types) built; gcd average
  implemented. Whole/split/gcd agree to ~1e-15 where all converge; vs reference 3.9e-16 (split) / 3.2e-15 (gcd) at the
  4 refs available then. The 27-split does NOT converge in the near band for r >= 4 xy/xyz (root's prediction held).
  Naive gcd average 38.9 ms/offset (4096 pieces at r = 16) = 658 s for the band; as a BOX SUM over the fine equal-cell
  egoToe (convolution with the coarse indicator) 1.4 s total; whole-box route 4.2 s for the 1.39e6 far offsets.
  OPEN: whole vs gcd disagreements 5.9e-14 .. 6.8e-10 at r = 8, 16 (lambda/4, lambda/2 coarse cells), to be attributed
  (root's hypothesis: gcd-sum cancellation Lambda and est_X over-stating |T| where the kernel oscillates across the
  coarse cell). Table build 20-60 s per pair, < 0.85 GB.
- xref (partial): rule B (graded trapezoid GL, ord 32) is 10-20x cheaper than rule A (pairKer, 5-11x the equal-cell cost)
  and is the production rule; A44 vs B32 1.6e-58 (r = 4), rule B ord 32 vs 40 6.8e-53 at the hardest offset (r16 xyz kap 1);
  A ordN 44 vs 64 at the 1e-63 floor for all ratios. 359 records cached (f = 1 complete for the trimmed matrix; 5 per
  shape at f = 1+0.1i, 0.37). Swap identity 7e-65.
- xnear (partial): the thin-indicator branch fires on EVERY separated pair (equal cells included; only hi = 2 lo escapes),
  so route (iii) already runs through it; digits lost <= 0.99 over 2772 cross-scale face pairs (thin branch worst 0.93),
  no worse than the lattice. 3D centred Taylor box3Tay written; a 3D-only stopping trap (harmonic 1/r: level 2 exactly 0)
  found and fixed (three consecutive small levels). Sweep and recommendation pending.
- xgeom DONE (2026-09-08 00:20, reports/xgeom.md 337 lines): Gila today on unequal separated pairs: the adaptive hcubature
  band (sep 1, rtol 1e-6) is 2.7e-9 .. 2.3e-4 (median 1.2e-6) at EVERY gap, not only the touching layer; the fixed rule
  is 1.5e-14 .. 8e-12 on cubes/slabs of lambda/16..lambda/2 (the "outside tested range" warning is harmless) but FAILS
  on rods with the fine cell displaced sideways: 8.8e-13, 2.6e-9, 1.4e-6, 1.8e-4 for r = 2, 4, 8, 16. Sign convention
  matches (flip R_y -> only G12 flips). Route (b) (nested refinement) needs a ring >= 2 coarse cells (chkParCmpVol),
  then rho_max = 0.47/0.41/0.38 (r = 2/4/8), zero adaptive offsets: the whole-box expansion alone serves route (b).
  R/g is half-integer on coarsened axes: key whole-box tables by 2R/g; the gcd sum runs over integer fine offsets.
- xnear DONE (2026-09-08 00:40, reports/xnear.md): item 5 answered. The thin-indicator GL branch is reached on every
  separated pair (gap >= 2) and every touching pair of ratio >= 2; Float64 loss <= 0.93 digits per face pair (2772
  pairs), tier B vs independent pairMom identical. 3D centred Taylor box3Tay (box3tay.jl) converges on 96.7% of 2734
  boxes at one eps but costs 1.5-6x GL, is rejected at the polydisc gate on corner boxes, and cannot lower the face-pair
  worst case (pinned by co-occurring closed-form corner boxes at 0.71-0.90): RECOMMENDATION keep box3Slv, no change to
  moments.jl. Route (iii) at 128 bits: loss 0-1 of 38 digits, nothing changes.
- xref DONE (2026-09-08 01:15, reports/xref.md 217 lines): cache xwork/xref/reftensors_x.txt = 441 records, 438 keys, ALL
  416 brief-matrix keys present (incl. 12 slender-pair records, f = 1+0.1i / 0.37 sets). Rule A (pairKer ordN 44) vs rule B
  (graded trapezoid GL ord 32): 1.6e-58 .. 3.9e-57 at kap 1 (rule B's ord-32 error; B40 reaches 1e-64), 1e-63 from kap 2.
  gcd identity to the 220-bit floor (3.2e-64 / 1.7e-64) when whole and pieces share a rule. Worst record 3e-50 (slender z
  kap 1). Rule B cost 16.5 us/point, 5-330 s per record; ~4.3 h machine time in total.
- xtable (2026-09-08 07:45, root's digest of its on-disk outputs; the agent stalled at 00:12 waiting for two Julia jobs
  that died together at 00:06-00:08 with no .done; its report lacks S3b/attribution/blockx; files in xwork/xtable/):
  * ATTRIBUTION (attrib.out): the whole-vs-gcd disagreements 5.9e-14 .. 3.2e-12 were the WHOLE-BOX route's k-series
    cut: farSetupX combined the two nCutVec evaluations with `max.`, which turns an uncertified -1 into the other
    radius' value; with `cutMax` (propagate -1, recompute at nCap) the table nMax rises to 15/17/25 for the lambda/4
    and lambda/2 coarse cells and the whole-box error drops to 3e-16 .. 1.2e-15 at every disputed offset. Lambda_gcd
    = 1.0-1.5 everywhere: root's cancellation hypothesis was WRONG; the gcd sum does not amplify. est_X/max|T| =
    1.5-5 (no over-statement problem). The SAME `max.` idiom is in farfield.jl's equal-cell farSetup (line ~949):
    latent defect D11, harmless at lambda/32-lambda/4 (both cuts certified) but must be fixed in the unify step.
  * S3b (recheck2_f1.md, 195 offsets, 10 shapes r2-r8 all sets + r16x partial, f = 1, vs the reference cache): worst
    whole 1.4e-15 (per entry 2.3e-15), split 1.5e-15 (2.0e-15), gcd 3.2e-15 (3.2e-15). r16xy/r16xyz and f != 1 rechecks
    not done (job died). estratio.out: whole-box bound/max|T| 1e-16..3e-14, est/max|T| 1.5-5, Lambda 1.0-2.2.
  * Decision (root): production cross-scale = trapezoid whole box (Theorem A') + gcd box-sum over the fine equal-cell
    block in the near band + k-series fallback; the 27-piece split is dropped from the deliverable (converges only where
    the whole box nearly does, 5-10x slower, diverges for r >= 4 xy/xyz near) and kept in xwork/xtable/farx.jl as the
    cross-check. r16 and complex-f verification move to the verify round.
- xtable DONE (2026-09-08 ~08:40, reports/xtable.md 1868 lines): S3b vs 272 referenced offsets at f = 1: whole box worst
  1.4e-15 (147 offsets), 27-split 1.5e-15 (152), gcd average 5.1e-15 (156, r16xyz 4096 terms); f = 1+0.1i / 0.37: 2.6e-15 /
  2.7e-15 over 64 each. Theorem A' bound <= 3.1e-14 max|T|; est_X/max|T| 0.26-5.8; Lambda_gcd 1.0-2.3. D11 confirmed with a
  second cause: in farSetup the near radius rNc is taken from OCTANT/piece centres (rho = 8 there, whole-box n-tail
  vacuous -> -1 -> masked by max. -> far-end N used, N = 0 at l >= 22-38). Fix: cutMax (-1 absorbing) AND whole-box n-cut
  from whole-box radii only; farfield.jl has the same two lines (:901-902), latent for the shipped shapes (fires at
  |k| r_d >~ 2.9). farBlockX! (farx.jl:1858-2079): realistic 1.40e6-offset block 6.39 s single / 2.19 s on 12 threads,
  13/13 refs <= 3.5e-15, certificates hold; 1800 offsets (rho 0.55-0.60) certified only to 1-4 tol at L = 56. r4xyz block
  1692 offsets 1.56 s, 9/9 refs <= 8.7e-16. Bit-identity vs current farfield.jl: 120/120. Table cache in xwork/xtable/
  shapetab grew to 883 MB (obsolete nMax variants) -> delete at cleanup.
- xunify DONE (2026-09-08 09:45, reports/xunify.md 520 lines): notes/farfield/farfield.jl is now the integrated file (2009
  lines; pre-cross copy xwork/xunify/farfield_pre_cross.jl). Equal-cell bodies changed: tabWhl (2 lines), farShape (key,
  file name, skip), farMomW (moment vectors), nCutVec (vt/vs split), farSetup (cutMax) + mrgGeo (drop duplicated
  columns, D12); bit-identity 120/120 tensors + two 12^3 blocks. Cross-scale API: farSetupX, farRouteX, farTensorX,
  farBlockX! (routes: 1 trapezoid whole box, 2 gcd box sum over one fine farBlock! egoToe, 3 k-series on unequal
  panels off the gcd lattice). 438/438 references: worst 3.4e-15 max-entry, 4.7e-15 per entry, 0 certificate
  violations, swap identity 0 / <= 4.9e-15 / 1e-36 by route, farTensorX == farBlockX! bitwise. Realistic 1.40e6-offset
  block: 17.9 s cold / 9.8 s warm single thread, 10.7 / 5.7 s on 12 threads (fill 1.0 s). verify.jl part a unchanged
  (a_rho.txt's staged copy was truncated at 4 lines; regenerated 81-line file left unstaged). Open: est_X over-states
  max|G| 100-8100x on the slender 4x4x1 thin axis (certificates loose, valid); unlocked table writer; exact-rational
  offset passes ~3 us/offset; 0.13% of the block certified to 1-4 tol.
- Round 2 launched 09:50: xverify (adversarial: all records, odd/mixed/non-integer ratios, lambda/2 and lambda coarse
  cells, complex f, Float32, routing/lattice logic, thread determinism, Gila before/after; verify.jl parts j-m) and
  xdoc (tex/crossscale.tex + registry/defects/uncertain/abstract/API updates; pdf rebuild).
- Cleanup 2026-09-08 17:40 (root): deleted the regenerable table caches of the two finished agents -- xwork/xtable/
  shapetab (870 MB, obsolete nMax variants), xtable/eqtab, eqtab2, and xunify/tab_{new,d,old,e,corrupt,attrib1,
  attrib2,old_fresh,new_fresh} (286 MB). notes/farfield 1.6 GB -> 470 MB. Every deleted file is a shape table
  rebuilt by farShape in 14-18 s; xunify/ksr_new and all reports, scripts and reference caches were kept. The
  shipped notes/farfield/shapetab/ (112 MB) stays gitignored, as in the equal-cell round.
- Round 2 interrupted 2026-09-08 17:2x by the Fable rate limit: both agents were killed, every Julia job of theirs
  had finished except verify.jl part j (orphaned, still running, 423/488 rows). Relaunched on Opus 5 at 17:35 as
  two resume agents: xverify (wait for part j, write Section 1 + the bug roll-up + `## DONE`) and xdoc (poll for
  that marker, write the adversarial paragraph into tex/crossscale.tex, rebuild the PDF). The stale xtable waiter
  shell (waiting on a bitid.done that bitid.jl never writes) was killed at 13:40.
- Root, 17:40-18:05: while deleting xverify's regenerable caches I also removed `xwork/xverify/bitid/`, which held the
  bit-identity reproducer and its output, not just tables. Both were regenerated: `bitid/bitid_v.jl` was rebuilt from
  `xwork/xunify/bitid.jl` (Old = xunify/farfield_pre_cross.jl, New = the current notes/farfield/farfield.jl with the
  D13 fix) and re-run from empty table dirs, which makes it a stronger check than the original (every table built
  fresh in both modules rather than copied in). Result, independent of the agent's run: (a) tabWhl/tabOct at
  BigFloat(256), farMomW(80) and nCutVec over four shapes, 0 mismatches; (b) farTensor 120/120 bitwise identical with
  equal Lw/Lo and nCut on all four shapes; (c) farBlock! on two 12^3 blocks, 13894 and 13952 nonzero entries, 0
  differing; (a') a fresh farShape build byte-identical on disk. Output `bitid/bitid.txt`, log `bitid/bitid_v.log`;
  the private table dirs were deleted again afterwards.
- xeq DONE (2026-09-08 18:55, reports/xeq.md): D15, the equal-cell twin of D13, found by asking whether D13's
  mechanism survives the port to `FrqSet`. It does. `farSetup` certifies the k-series cut `nCut` on
  `[rNc, rHi]` only -- an interval fixed by the offsets the set was BUILT for -- and `farRoute` compared no later
  request against it. Worst measured: 3.215e-08 at D = (2,2,6) on the (1/32,1/32,1/512) cell from a set built with
  an explicit empty `offs` at nBlk = 128, under a certificate of 6.465e-15 of max|G|, a 4.97e6x violation where a
  fresh set gives ~1e-15. Realistic form (set built for offs = ((100,0,0),), reused at every nearer offset its Lw
  admits): 9.444e-09 vs a 220-bit reference against a certificate of 9.087e-15, 1.04e6x, on all five shape/frequency
  cases. Fix (farfield.jl, 15 lines, xwork/xeq/fix.diff): FrqSet carries rNc/rHi; farSetup falls back to thr[end]
  when the built offsets certified nothing; a new `ckRad` raises from farRoute on the whole-box radius and on the
  eight octant vertex radii. It only raises, never changes a value. Proofs: bit-identity old vs new from empty table
  dirs (tabWhl/tabOct/farMomW/nCutVec 0 mismatches; farTensor 120/120 bitwise; two 12^3 farBlock! blocks 13894 and
  13952 nonzero, 0 differing; fresh farShape byte-identical), scan2/repro pre and post, and verify.jl parts a-i, k, l
  all exit 0 with the guard firing nowhere. The DEFAULT `offs = nothing` path is safe and was measured so, not
  argued: nearOff enumerates the whole block, and a full nBlk^3 sweep over four shapes x two frequencies x
  nBlk in {16,128} -- 16 785 152 offsets -- raises 0 times (xwork/xeq/dflt.txt). The one caller in the suite that
  DID fire it was verify.jl part (b), which cached one set per (shape, f) but sized nBlk on the first offset of the
  group and then reused it at (64,0,0) and (64,64,64); fixed in the caller (part (d) already did it right), and the
  two b_volume.txt rows improved, 1.13e-15 -> 1.01e-15 and 2.12e-15 -> 1.75e-15, so the reuse had been costing
  accuracy in the shipped table. All other table differences vs xwork/xeq/tables_before are timing and load columns.
  Renumbered: the first draft called this D14, which is taken by the Float32 NaN defect; it is D15 in the code
  comment, the report, tex/defects.tex and here. Residual: the interval is certified at its endpoints only, the
  thr[end] fallback is a floor and not a proof, and the octant half of ckRad is untested against a real failure
  (every measured octant escape already raises on the pre-existing Lo guard).
- FINAL, cross-scale round, 2026-09-08 18:50. Deliverables on branch paul-farfield: notes/farfield/farfield.jl (2029
  lines: equal-cell library unchanged bit for bit apart from the D15 guard, plus farSetupX / farRouteX / farTensorX /
  farBlockX! for unequal cells over three routes), verify.jl (parts a-i equal cell, j-m cross scale), tables/,
  farfield.tex + tex/crossscale.tex (83 pages, 0 errors, 0 overfull), and the reports in scratch/reports/.
  Accuracy: 439/439 cached 220-bit references to 3.42e-15 worst (4.71e-15 per entry), 66 adversarial offsets with 47
  new references to 5.1e-15, 0 certificate violations in 878 certified evaluations, routing exact on 125,256
  enumerated offsets, farBlockX! bitwise identical at 1, 4 and 12 threads and bitwise equal to farTensorX per offset.
  Cost: 6-8 us/offset against Gila's 5-13 ms, a 1.4e6-offset block in 8.5 s single-threaded / 5.2 s on 12 threads.
  Gila today, same offsets: adaptive rule 1e-7..5e-4, fixed rule 2e-14..1.8e-12.
  Defects this round: D11 (masked n-cut) and D12 (duplicated table segments) fixed during integration; D13 (a
  FrqSetX reused at an uncertified radius, 1.59e-11 under a 1.7e-14 certificate) fixed; D15 (its equal-cell twin,
  3.2e-8 under a 6.5e-15 certificate, up to 5e6x) fixed; D14 (Float32 NaN at L >= 28 from the overflowing h_l
  recursion) diagnosed and NOT fixed. Not done: wiring into src/ (notes/PLAN_wiring.md, whose star-2 item is now
  option (b), the trapezoid whole box, rather than option (a)); non-integer rational ratios (the three-line rational
  gcd is proven in xwork/xverify/farfield_gcd.jl but not shipped, since Gila's GlaExtInf refuses them too).
