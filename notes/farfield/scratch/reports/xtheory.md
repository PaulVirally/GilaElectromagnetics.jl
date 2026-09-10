# xtheory: trapezoid-weight bounds (Theorems A', B), exact identities, and where each route converges

Dir: notes/farfield/scratch/xwork/xtheory/. Files: `xbound.jl` (deliverable, include after farfield.jl),
`xexact.jl` -> `out_xexact.txt` (exact rational checks), `xbit.jl` -> `out_xbit.txt` (bitwise fallback tests),
`xrho.jl` -> `out_xrho.md` (rho and L per route on the test matrix), `bounds_cross.tex` (tex fragment).

## 1. Derivations

Notation: target edges sT at centre R, source edges sS at the origin; per axis a_i = |sT_i - sS_i|/2,
b_i = (sT_i + sS_i)/2, so b_i - a_i = min(sT_i, sS_i), b_i + a_i = max(sT_i, sS_i), b_i^2 - a_i^2 = sT_i sS_i.
D = prod [-b_i, b_i], V_t = prod sT_i, V_s = prod sS_i, r_d = |b| = max_{D} |delta|, rho = |b|/|R|.

### 1.1 The trapezoid is a difference of two triangles (the one fact that does all the work)

    w^{ab}(t) = (b - |t|)_+ - (a - |t|)_+  =  b - a  on |t| <= a,   b - |t| on a <= |t| <= b,   0 beyond.

Every LINEAR functional of the trapezoid weight is the triangle(b) functional minus the triangle(a) functional:

    mu_n(a,b) = int w t^n dt = 2 (b^{n+2} - a^{n+2}) / ((n+1)(n+2))   (even n; 0 odd)  = wgtT(n,b) - wgtT(n,a),
    w'  = -sign(t) [1_{|t|<b} - 1_{|t|<a}] = -sign(t) on the ramps, 0 on the plateau,
    w'' = (delta_{-b} - 2 delta_0 + delta_b) - (delta_{-a} - 2 delta_0 + delta_a) = delta_{-b} + delta_b - delta_{-a} - delta_a .

The brief's form 2(b-a)a^{n+1}/(n+1) + 2[b(b^{n+1}-a^{n+1})/(n+1) - (b^{n+2}-a^{n+2})/(n+2)] collapses to the
two-term form above (exact-rational check (i) below: identical for all (a,b), n <= 20).  a = 0 gives wgtT(n,b);
the point mass 2 delta_0 of the triangle is the a -> 0 limit of the two masses at +-a (2 a^{2i} -> 2 * 0^{2i}, i.e.
mass 2 at i = 0 and nothing above, which is exactly farfield.jl's `zro` vector).  int w = b^2 - a^2 = sT sS, so
(1/V_t) int_D w = V_s.  Digits lost in Float64 by the difference form: log10(1/(1 - (a/b)^{n+2})) <= 0.66 at
ratio 16 (a/b = 15/17, n = 0) and decreasing in n; exact in Rational.

For the bounds the measures must stay POSITIVE, so the two triangles are not subtracted there:
    |w'| dt   = flat measure on the ramps,      lambda_n(a,b) = int_{a<|t|<b} |t|^n dt = 2 (b^{n+1} - a^{n+1})/(n+1),
    |w''|     = delta_{+-b} + delta_{+-a},      moments 2 b^{2i} + 2 a^{2i}  (a = 0: 2 s^{2i} + 2 * 0^{2i}).

### 1.2 Lemma ibp' (integration by parts onto the trapezoid weight)

w = prod_i w_i^{a_i b_i}, extended by zero, is Lipschitz and vanishes on dD.  For phi in C^2 near the closure of D:
  (o) a != b, c the third axis:   int_D w d_a d_b phi = int_D sign(d_a) sign(d_b) 1_{a_a<|d_a|<b_a} 1_{a_b<|d_b|<b_b} w_c(d_c) phi
  (d) a = b, Q_a the rectangle in (b,c):
        int_D w d_a^2 phi = int_{Q_a} w_b w_c [ phi|_{d_a=b_a} + phi|_{d_a=-b_a} - phi|_{d_a=a_a} - phi|_{d_a=-a_a} ] .
Proof: exactly as Lemma ibp of bounds2.tex.  (o): one integration by parts in each of the two axes, both boundary
terms vanish (w = 0 on dD), w_a' w_b' = sign sign on the ramps and 0 elsewhere.  (d): one integration by parts in
axis a, then -int w_a' d_a phi = -int_{-b}^{-a} d_a phi + int_{a}^{b} d_a phi = phi(b)+phi(-b)-phi(a)-phi(-a) =
<w_a'', phi>.  When a_a = 0 the last two evaluations coincide and give -2 phi(0): the triangle lemma.
What changed: only the support of w' (ramps instead of the whole interval) and the location of the point masses
(+-a instead of the doubled 0).  Nothing else in the proof refers to the shape of w.

### 1.3 Theorem A' (whole trapezoid box, derivatives on the regular side)

Pair separated, rho = |b|/|R| < 1, T^{(L)} the truncation at l <= L of the expansion with derivatives on the regular factor:
    |T_ab - T_ab^{(L)}| <= (|k| / (4 pi |f|^2 V_t)) sum_{l > L, l even} (2l+1) |h_l(k|R|)| |k|^l e^{(|k| |b|)^2/(4l+6)} / (2l+1)!!  V^{ab}_l ,
    V^{ab}_l = sum_{i+j+q=p} p!/(i!j!q!) lambda_{2i}(a_a,b_a) lambda_{2j}(a_b,b_b) mu_{2q}(a_c,b_c)                 (a != b, l = 2p),
    V^{aa}_l = 2 sum_{i+j+q=p} p!/(i!j!q!) (b_a^{2i} + a_a^{2i}) mu_{2j}(a_b,b_b) mu_{2q}(a_c,b_c) + |k|^2 W_l,
    W_l     = sum_{i+j+q=p} p!/(i!j!q!) mu_{2i} mu_{2j} mu_{2q}   (all trapezoid moments).
Every term positive.  V_t in the prefactor is the TARGET volume (srfMat = I/V_t).
Proof, step by step against bounds2.tex thm:bnd2:reg (I checked each):
  1. d_R = d_delta on g(R + delta): unchanged.
  2. Uniform convergence of the addition-theorem series with all delta-derivatives on the closure of D: needs
     max_D |delta| < |R|, i.e. rho = |b|/|R| < 1 with r_d = |b| (the box is prod[-b_i,b_i]).  Only r_d changes.
  3. T_ab = (ik/(f^2 V_t)) sum (-1)^l psi_lm(R) M^{ab}_lm with M = int_D w (d_a d_b + delta_ab k^2) phi_lm: unchanged
     (it is the definition of the volume form with the new w).
  4. Cauchy-Schwarz over m and sum_m |psi_lm|^2 = (2l+1)/(4 pi) |h_l|^2: unchanged (no w involved).
  5. Lemma ibp' replaces Lemma ibp; Minkowski's integral inequality needs the measures to be positive, which they are
     (|w'| dt, |w''|, w dt; this is the only place w >= 0 is used).  Same argument, new measures.
  6. |j_l(kx)| <= (|k|x)^l e^{(|k| r_d)^2/(4l+6)}/(2l+1)!! (Theorem thm:bnd:j holds for every z; the exponential is
     increasing in |z|, so it is taken at r_d = |b| >= x on the support of every measure: ramps and points at
     +-a_i, +-b_i all lie in the closure of D).  Only r_d changes.
  7. Multinomial expansion of |delta|^{2p} against the product measures: unchanged; the one-dimensional moments
     are now lambda_n(a,b), mu_n(a,b), 2 b^{2i} + 2 a^{2i}.
  8. Odd l omitted because M^{ab}_lm = 0 for odd l: the parity argument of sec:exp:parity uses only "w is even in
     every coordinate", which the trapezoid is.  Unchanged; same four (l,m) classes.
Conclusion: only the measures (step 5/7) and r_d (steps 2/6) change; nothing else.  a = 0 on every axis returns
Theorem A term by term (bitwise, Section 3).

### 1.4 Theorem B (one affine sub-box) needs no change

Its hypotheses are: D_j = prod[-h_i,h_i], u_i = alpha_i + beta_i t >= 0 on [-h_i,h_i], V = R + c_j, r_j = |h|,
rho_j = r_j/|V| < 1.  The pieces of the 27-split are: ramp [a,b] -> centre (a+b)/2, h = (b-a)/2, u = h - t'
(alpha = h, beta = -1; the mirror ramp has beta = +1), which is the octant piece with s/2 -> (b-a)/2; plateau
[-a,a] -> h = a, alpha = b - a, beta = 0.  For beta = 0 the one-dimensional identities int u X' = u^+ X(h) - u^- X(-h)
- beta int X and int u X'' = u^+ X'(h) - u^- X'(-h) - beta (X(h) - X(-h)) hold with the beta terms absent, so
nu_i = pi_i = (b-a)(delta_h + delta_{-h}), the beta-measure is 0, and every step (Minkowski, Lemma msum for the one
surviving first derivative, the j_l majorant at r_j, the multinomial moments) is verbatim.  The proof never uses
beta != 0 or alpha = h.  An axis with a = 0 has no plateau: two ramps of h = b/2 (the octant split on that axis).
farMomOX(lTop, h, alpha, beta) in xbound.jl implements the general piece; farMomOX(h, h, -1) == farMomO(h) bitwise.

The gcd average G(R; sT, sS) = (1/N_t) sum_j sum_j' G(R + c_j - c'_j'; g, g) is a decomposition of the MEASURE
w^{ab} d delta into N_t N_s translated triangle measures of the cell g (sum of indicator convolutions), so each
piece is a whole-box Theorem A tensor at a real offset, with its own r_d = |g|, rho_jj' = |g|/|R + c_j - c'_j'|,
prefactor 1/V_g inside G(.; g, g) and the 1/N_t outside; errors add with weights 1/N_t, so a budget tol est/N_s
per piece certifies tol est for the sum (est = est_X of 1.6, which is sum over pieces of est_g up to the variation
of the pointwise scale across sub-cells).

### 1.5 The j_l pointwise bound and the k-series n-tail

j_l: Theorem thm:bnd:j is a statement about j_l alone; it enters only through x <= r_d on the support of the measures,
so r_d = |b| is the only change (whole box) and r_j = |h| on a piece.
n-tail (nCutVec): the remainder of the j_l power series after n = N is bounded pointwise by
|k|^q x^q e^{(|k| r_d)^2/(4l+4N+10)}/(2^{N+1}(N+1)!(2l+2N+3)!!), q = l + 2N + 2, and integrated against the SAME
positive measures gives Theorem A' with V^{ab}_l -> V^{ab}_q, i.e. jtail(l, N, |k| r_d)/r_d^q * V^{ab}_q with
r_d = |b| and V^{ab}_q from farMomWX.  The est inside the budget must be est_X (1.6), and the `rd = |s|`, `vt = prod(s)`
lines of nCutVec become rd = |b|, vt = V_t (prefactor) with est taking V_s.  Route (iii)'s k-series bound
ksrBnd(x, N) uses x = |k| D_max with D_max = max_{x in target, y in source} |x - y| = sqrt(sum_i (|R_i| + b_i)^2)
= max over the 36 face pairs of panSpan's `hi` (panSpan already takes unequal panels); its amplification Lambda
divides by est, which again must be est_X.

### 1.6 The scale est and the lower bound estLo

est is the pointwise dyadic scale |k|^2 e^{-Im k |R|}/(4 pi |f|^2 |R|) (1 + 3/|kR| + 3/|kR|^2) times (1/V_t) int_D w.
For the trapezoid (1/V_t) int_D w = prod (b_i^2 - a_i^2)/V_t = V_s, so
    est_X(R) = est(|R|, k, f, V_s)   (farfield.jl's est with vt := V_s = prod sS),
and by the swap identity est_X(R; sT, sS)/est_X(-R; sS, sT) = V_s/V_t = G(R; sT,sS)/G(-R; sS,sT), consistent.
estLo (lowScl/lowVal): tr T = 2 k^2 S with S = (1/V_t) int_D w g(R + delta); the l = 0 shell has
|S_0| >= (|k| e^{-Im k |R|}/(4 pi |f|^2 |k| |R|)) (1/V_t) int_D w |j_0(k|delta|)| >= (...) V_s j_0(|k| r_d)
using w >= 0 and j_0 positive and decreasing on [0, pi]: hypotheses are |k| r_d <= pi with r_d = |b| (holds for all
12 test shapes at g = 1/32, f = 1: |k||b| = 0.405 (r=2,{x}) ... 2.89 (r=16,{x,y,z}); table in out_xlow.md), and the
tail (Theorem A' sum with W_l in place of V^{ab}_l, prefactor 1/V_t, no derivative factor)
smaller than the leading shell.  So in lowScl: e0 = |k| V_s j_0(|k| |b|)/(4 pi |f|^2) (V_s, not vt), the tail
terms keep pf = |k|/(4 pi |f|^2 V_t) with the trapezoid W_l, rd = |b|.  Nothing else changes.

## 2. Exact-rational verification (`xexact.jl` -> `out_xexact.txt`; 132 rows, every difference exactly 0//1)

Polynomials over Rational{BigInt} in the six variables (x, y) or the three variables delta; box integrals by exact
antiderivatives; nothing floating anywhere.  Shapes (sT, sS), g = 1/32: (g,g,g)/(g,g,g) [a = 0]; (g,g,g)/(2g,g,g);
(g,g,g)/(16g,16g,16g) [a = 15/64, b = 17/64: a close to b]; (g,g,g)/(100g,g,4g) [a/b = 99/101]; (3g,5g,g)/(5g,3g,7g);
(g,g,16g)/(g,g,2g); (g,g,1/512)/(1/8,1/8,1/512) [slender].

| check | cases | quantity compared | result |
|---|---|---|---|
| (i) mu_n(a,b) | 7 shapes x 3 axes x n = 0..20 | two-term closed form vs brief's three-term form vs direct piecewise (plateau + two ramps) | all exactly 0 |
| (i') convolution identity | 7 shapes x 125 monomials (x-y-R)^alpha, alpha_i <= 4, R = (7/3,-2/5,11/7) | int_T int_S (x-y-R)^alpha  vs  prod_i mu_{alpha_i}(a_i,b_i) | all exactly 0 |
| (ii) IBP lemma | 7 shapes x 343 monomials delta^e, e_i <= 6, six (p,q) | int w d_p d_q phi vs (o) signed ramp x ramp x w_c, (d) point evaluations at +-b_p, +-a_p times mu mu | off-diag 0, diag 0 |
| (iii) V^{ab}_l, W_l | 7 shapes x l = 0,2,..,20 x 6 entries + W_l | farMomWX (multinomial code) vs expanding |delta|^l as a polynomial and integrating against piecewise-computed one-dim moments | all exactly 0 |
| (iv) swap | 6 shapes x 2 R x j = 1..3, F = |x-y|^{2j} and its six d_p d_q | raw(R; sT, sS) vs raw(-R; sS, sT) | all exactly 0 |
| (v) gcd | ratios 2, 3; axes {x},{x,y},{x,y,z}; source-coarse, target-coarse, mixed (x-coarse target vs y-coarse source); j = 1..3 | raw(R;sT,sS) vs sum_{j,j'} raw(R + c_j - c'_j'; g, g), and G = (1/N_t) sum sum G(.; g,g) | all exactly 0 (N_t N_s up to 27) |
| weight form | 7 shapes x j = 1..3 | (1/V_t) int_D w |R + delta|^{2j} with trapezoid moments vs raw6/V_t | all exactly 0 |

(The "mixed" rows of (v) are the same pair for the three axis sets; harmless repetition.)

## 3. Bitwise fallback tests (`xbit.jl` -> `out_xbit.txt`; 359 comparisons, 0 mismatches)

farMomWX(lTop, (0,0,0), s) == farMomW(lTop, s) for T in {Float64, BigFloat, Rational{BigInt}}, s in {(1/32)^3, (1/8)^3,
(1/32,1/32,1/512), (0.3, 1e-3, 7.0)}, lTop in {0, 2, 24, 56, 140}; wgtTrpX(0, s, dM) == wgtTrpT(s, dM) for dM in {0, 5, 80};
farMomOX(lTop, h, h, (-1,-1,-1)) == farMomO(lTop, h) for lTop in {0, 7, 56}.  Bitwise = same bitstring (Float64),
same value, precision and sign (BigFloat), equality (Rational).  Why it is exact: wgtT(n, 0) = 2*0^(n+2)/... = 0 and
x - 0 == x in every type; a^(2i) at a = 0 is [1, 0, 0, ...], which is farMomW's `zro`.

## 4. estLo hypotheses on the test matrix (`xlow.jl` -> `out_xlow.md`, f = 1, BigFloat(128))

|k||b| <= pi holds for all twelve shapes (0.405 at r=2 {x} to 2.89 at r=16 {x,y,z}; at f = 1+0.1i multiply by 1.005).
The tail-below-leading-shell hypothesis fails (estLo = 0) at kap = 1 for r >= 4 on {x,y} and {x,y,z} and at every
kap <= 32 for r = 16 on {x,y} and {x,y,z}; where it holds, estLo/est_X is 0.005-0.03 at kap = 1 and 0.02-0.42 at
kap = 32.  So scl = :est stays the default for cross-scale pairs as it is for equal cells.
