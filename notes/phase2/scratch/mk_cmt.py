import re
src = open("far_v2_raw.jl").read()
def rep(old, new, n=1):
    global src
    c = src.count(old)
    assert c == n, f"count {c} != {n} for:\n{old[:160]}"
    src = src.replace(old, new)

# ---- file header -------------------------------------------------------------
rep("""# Far field of Gila's discretized vacuum Green operator without numerical quadrature.
# T_ab(R) = (1/V_t) int_D w(d) [(da db + dab k^2) g](R+d) dd, g = e^{ikr}/(4 pi f^2 r), k = 2 pi f,
# D = prod_i [-s_i, s_i], w(d) = prod_i (s_i - |d_i|), R = n .* s, V_t = s1 s2 s3.  Sign +, no factor.
# (i)   whole box by the spherical addition theorem; exact rational geometry table, only even l.
# (ii)  the 8 octants of D with the affine weight; same theorem, all l, (-1)^l mandatory.
# (iii) BigFloat k-series on the 36 face pairs of notes/moments, where neither (i) nor (ii) converges.
# L per offset and per sub-box from the proven tail bound of Theorems A (whole box) and B (sub-box)
# of bounds2.tex, which bound the truncation this file performs (derivatives on the regular side);
# the route is the cheaper of (i), (ii) whose bound meets tol * est(R), else (iii).
# The geometry table is built lazily in l and extended in place; its k-series order N is chosen per
# (shape, frequency) from the n-tail bound (N rises with |k| r_d) and the library refuses to
# evaluate rather than truncate silently; generic in T, every constant typed.
# Unequal cells (target edges sT at R, source edges sS at 0) change only the weight: per axis the
# trapezoid w_i(t) = (b_i - |t|)_+ - (a_i - |t|)_+, a = |sT - sS|/2, b = (sT + sS)/2, D = prod[-b_i, b_i],
# (1/V_t) int_D w = V_s.  Routes: the whole trapezoid box (Theorem A', same proof with r_d = |b| and the
# ramp/point measures), the gcd average of equal-cell tensors of g = min(sT, sS) (a box sum over the
# fine egoToe), and the k-series on the 36 unequal face pairs.  Section "unequal cells" at the end.
""",
"""# Separated blocks of the discretized vacuum Green operator, in closed form.
# T_ab(R) = (1/V_t) int_D w(d) [(da db + dab k^2) g](R+d) dd, g = e^{ikr}/(4 pi f^2 r), k = 2 pi f,
# D = prod_i [-s_i, s_i], w(d) = prod_i (s_i - |d_i|), V_t = s1 s2 s3.  Three routes per offset:
#   1  the whole box by the spherical addition theorem, one expansion about R, even l only;
#   2  the 8 octants of D with the affine weight, same theorem, all l;
#   3  the k-series on the 36 face pairs of glaVacIntMom.jl, where neither expansion converges.
# The l per offset and per sub-box comes from the proven tail bounds of Theorems A (whole box) and
# B (sub-box); the route is the first of 1, 2 whose bound meets tol * est(R), else 3.  The geometry
# table is exact rational, built lazily in l and extended in place; its k-series order N is chosen
# per (shape, frequency) from the n-tail bound, and the library refuses to evaluate rather than
# truncate silently.  Generic in T, every constant typed.
# Unequal cells (target edges sT at R, source edges sS at 0) change only the weight: per axis the
# trapezoid w_i(t) = (b_i - |t|)_+ - (a_i - |t|)_+, a = |sT - sS|/2, b = (sT + sS)/2, D = prod[-b_i,
# b_i], (1/V_t) int_D w = V_s.  Their routes are the whole trapezoid box (Theorem A'), the gcd
# average of equal-cell tensors of g = min(sT, sS), and the k-series on the 36 unequal face pairs.
""")

# ---- section dividers --------------------------------------------------------
for d, keep in (("# ---- exact weight moments ---------------------------------------------------\n", ""),
                ("# ---- integer solid harmonic polynomials -------------------------------------\n", ""),
                ("# ---- geometry table ---------------------------------------------------------\n", ""),
                ("# ---- shape table, in-memory and on-disk cache -------------------------------\n", ""),
                ("# ---- frequency contraction --------------------------------------------------\n", ""),
                ("# ---- a priori bounds --------------------------------------------------------\n", ""),
                ("# ---- routing ----------------------------------------------------------------\n", ""),
                ("# ---- double-double phase seed ----------------------------------------------\n", ""),
                ("# ---- spherical Bessel / Hankel ---------------------------------------------\n", ""),
                ("# ---- per-offset evaluation --------------------------------------------------\n", ""),
                ("# ---- public API -------------------------------------------------------------\n", ""),
                ("# ---- unequal cells ----------------------------------------------------------\n", ""),
                ("# ---- public API, unequal cells ----------------------------------------------\n", "")):
    rep(d, keep)
rep("# ---- route (iii): the k-series on the 36 face pairs in BigFloat -------------\n", "")
rep("# ---- route 3: the k-series on the 36 unequal face pairs ---------------------\n", "")
rep("# ---- Theorem A (whole box) and Theorem B (one octant) of bounds2.tex -------\n",
    "# Theorem A (whole box) and Theorem B (one octant) of notes/farfield/farfield.pdf.\n")

# ---- in-place GMP / MPFR -----------------------------------------------------
rep("""# The polynomial matrices hold one distinct BigInt per entry (zeros(BigInt, ..) aliases a single
# object) and every write is an in-place GMP call: the allocating version left 2.4-5.8 MB of
# BigInt garbage per (l,m) column, 15 GB over an L = 50 octant table, which the GC did not return.
""",
"""# The polynomial matrices hold one distinct BigInt per entry -- zeros(BigInt, ..) would alias a
# single object -- and every write is an in-place GMP call; the allocating spelling leaves BigInt
# garbage the collector does not return.
""")
rep("""# The BigFloat build runs this loop ~3e8 times per shape and the generic method allocates five
# limb buffers per term; doing the same arithmetic in four preallocated registers is bit-identical
# and takes the cold c32 table from 202 s to 44 s.  (The build's former 21-25 GB peak RSS was the
# BigInt churn of hrmPly! and mulRsq!, now in place as well.)
""",
"""# In-place MPFR for momAcc's inner loop: the generic BigFloat method allocates five limb buffers
# per term, and the loop runs ~3e8 times per shape.  Bit-identical, ~5x faster.
""")
rep('''"Four scratch registers for the BigFloat method of momAcc, at the ambient precision."\n''',
    "# the four registers momAcc's BigFloat method works in, at the ambient precision\n")

# ---- constants ---------------------------------------------------------------
rep("""# NMAX is the FLOOR of the table's k-series order, not a cap: farSet raises it to whatever the
# n-tail bound asks for at this shape and frequency (N grows with |k| r_d: 12 covers every cell
# with a longest edge <= lambda/4, a lambda/2 cube at |k| r_d = 5.44 needs 16, |k| r_d = 8.96
# needs 25).  NCAP is the largest N the bound may ask for (|k| r_d ~ 22); beyond it farSet
# refuses rather than returning an uncertified value.
const NMAX = 12
const NCAP = 40""",
"""# NMAX is a FLOOR on the table's k-series order, not a cap: farSet raises it to whatever the
# n-tail bound asks for, which grows with |k| r_d.  Beyond NCAP farSet refuses rather than
# returning an uncertified value.
const NMAX = 12                     # enough for every cell with a longest edge <= lambda/4
const NCAP = 40                     # the largest N the n-tail bound may ask for, |k| r_d ~ 22""")
rep("""# The budget is tol * (a scale of the tensor).  1e-13 against the OLD bound, which over-stated the
# remainder by a median 1.5e5, delivered 1e-15 to 1e-14; against Theorem A, which over-states by a
# median 49, the same 1e-13 delivers 1.6e-13 on the worst entry (measured, prep report S A.3).
# 1e-14 reproduces the old library's delivered accuracy entry for entry and still selects a
# smaller or equal L at 96% of the reference offsets.
const TOL = 1e-14""",
"""# Theorem A over-states the remainder by a median 49, so tol = 1e-14 delivers 1e-15 to 1e-14 per
# entry.
const TOL = 1e-14                   # relative to est(R), the a priori scale of the tensor""")
rep("""# The geometry contraction alternates in sign and cancels by up to 7.3e11 = 2^39.4 at L = 56 while
# summing at most (L/2 + nMax + 2)^2 ~ 2^11 terms, so a build at prc bits keeps a relative error
# 2^(11 + 39.4 - prc): 2^-205 at BLDPRC = 256, far below the 2^-192 at which the table is stored
# and below the 2^-53 that matters.  Segments are built lazily in l and appended to the cache.
""",
"""# Segments are built lazily in l and appended to the cache.
""")
rep("const BLDPRC = 256                  # precision of the geometry contraction (cancellation ~7.3e11)",
    "# the geometry contraction cancels by up to 2^39.4 over at most 2^11 terms, so 256 bits leaves\n"
    "# a relative 2^-205, well under the 2^-192 the table is stored at\n"
    "const BLDPRC = 256")

# ---- docstrings -> comments --------------------------------------------------
DOC = [
 ('"The top l a segment actually carries; -1 for the placeholder written when only the other grew."',
  "# the top l a segment carries; -1 for the placeholder written when only the other table grew"),
 ('"Concatenate two l-segments of one table; the (l,m) columns of both are already l-ordered."',
  "# concatenate two l-segments of one table; both (l,m) column lists are already l-ordered."),
 ('"Geometry tables for shape s, whole box to Lw and octant to Lo, built lazily and cached."',
  "# geometry tables for shape s, whole box to Lw and octant to Lo, built lazily and cached"),
 ('"An l-segment holding nothing, written when only one of the two tables grew."',
  "# an l-segment holding nothing, written when only one of the two tables grew"),
 ('"hb(l, z) |k|^l/(2l+1)!! for l = 0..L, the scaling folded into each term so nothing overflows."',
  "# hb(l, z) |k|^l/(2l+1)!! for l = 0..L, the scaling folded into each term so nothing overflows"),
 ('"V^{ab}_l of Theorem A, split as fc[r][l/2+1] + (a==b ? |k|^2 W_l : 0); even l = 0..lTop."',
  "# V^{ab}_l of Theorem A, split as fc[r][l/2+1] + (a == b ? |k|^2 W_l : 0); even l = 0..lTop"),
 ('"Per-l, per-entry factors of the Theorem A tail, everything but |h_l|; even l only."',
  "# per-l, per-entry factors of the Theorem A tail, everything but |h_l|; even l only"),
 ('"Even-order radial moments of the four one-dimensional measures of an octant sub-box."',
  "# even-order radial moments of the four one-dimensional measures of an octant sub-box"),
 ('"|d|^q moment from an even-order table; odd q by Cauchy-Schwarz with the non-negative measure."',
  "# |d|^q moment from an even-order table; odd q by Cauchy-Schwarz with the non-negative measure"),
 ('"Per-l, per-entry factors of the Theorem B octant tail, everything but |h_l|; all l."',
  "# per-l, per-entry factors of the Theorem B octant tail, everything but |h_l|; all l"),
 ('"Cumulative tail after each l, max over the six entries, with a geometric continuation past lTop."',
  "# cumulative tail after each l, max over the six entries, continued geometrically past lTop"),
 ('"A priori magnitude est(R) used to turn the absolute tail bound into a relative one."', ""),
 ('"Smallest l <= lTop in `ls` whose cumulative tail is below `bud`; -1 if none."',
  "# smallest l <= lTop in ls whose cumulative tail is below bud; -1 if none is"),
 ('"Per-l n-truncation cut; nCut[l+1] is the smallest N <= nCap meeting the budget, -1 if none does."',
  "# per-l n-truncation cut: nCut[l+1] is the smallest N <= nCap meeting the budget, -1 if none does"),
 ('"Elementwise max of two n-cut vectors in which -1 (not certified within the cap) is absorbing."',
  "# elementwise max of two n-cut vectors, in which -1 (uncertified within the cap) is absorbing"),
 ('"Lattice offsets whose whole-box bound asks for more than `lDef` shells; the table is sized on them."',
  "# lattice offsets whose whole-box bound asks for more than lDef shells; the table is sized on them"),
 ('"Frequency-contracted tables, thresholds and route cache for shape s at frequency f."',
  "# frequency-contracted tables, thresholds and route cache for shape s at frequency f"),
 ('"Refuse to evaluate when the certified k-series order exceeds what the geometry table carries."',
  "# refuse to evaluate when the certified k-series order exceeds what the geometry table carries"),
 ('"Whole-box L for offset radius rr from the tabulated thresholds; -1 if L > Lmax is needed."',
  "# whole-box l for offset radius rr from the tabulated thresholds; -1 if more than LMAX is needed"),
 ('"The Theorem A bound itself on max_ab |T_ab - T_ab^(L)| at radius rr."',
  "# the Theorem A bound itself on max_ab |T_ab - T_ab^(L)| at radius rr"),
 ('"Per-sub-box L for the octant split at offset R; -1 in a slot means Lmax is not enough."',
  "# per-sub-box l for the octant split at offset R; -1 in a slot means LMAX is not enough"),
 ('"Leading factor and l >= 2 tail terms of the proven lower bound on max_ab |T_ab|."',
  "# leading factor and l >= 2 tail terms of the proven lower bound on max_ab |T_ab|"),
 ('"The proven lower bound itself at radius rr; 0 when the leading shell does not survive its tail."',
  "# the lower bound itself at radius rr; 0 when the leading shell does not survive its own tail"),
 ('"The scale the selector actually divides by: est (default) or the proven estLo."',
  "# the scale the selector divides by: est by default, else the proven lower bound"),
 ("\"Refuse a radius outside [rNc, rHi], where this set's k-series cut nCut is certified.\"",
  "# refuse a radius outside [rNc, rHi], where this set's k-series cut nCut is certified"),
 ('"Route for one offset: (kind, L, Lc, cost); kind 1 = whole box, 2 = octants, 3 = k-series."',
  "# route for one offset: (kind, L, Lc, cost); kind 1 = whole box, 2 = octants, 3 = k-series"),
 ('"|R| as a double-double from the exact squares of the components."',
  "# |R| as a double-double from the exact squares of the components"),
 ('"(|R|, e^{2 pi i f |R|}); the phase is argument-reduced in double-double for Float64."',
  "# (|R|, e^{2 pi i f |R|}); the phase argument is reduced in double-double for Float64"),
 ('"h_l^{(1)}(z) by the upward recurrence from h_0 = -i e^{iz}/z; e = e^{iz} supplied."',
  "# h_l^{(1)}(z) by the upward recurrence from h_0 = -i e^{iz}/z; e = e^{iz} is supplied"),
 ('"y_l(x) by the upward recurrence (dominant solution) from y_0 = -cos x/x."',
  "# y_l(x) by the upward recurrence, its dominant solution, from y_0 = -cos(x)/x"),
 ('"j_l(x) by the downward (Miller) recurrence, normalised on the larger of j_0, j_1."',
  "# j_l(x) by the downward (Miller) recurrence, normalised on the larger of j_0, j_1"),
 ('"h_l(kR) into ws.h; for real k the real part is j_l by Miller and the imaginary part y_l upward."',
  "# h_l(kR) into ws.h; for real k the real part is j_l by Miller and the imaginary part y_l upward"),
 ('"Real spherical harmonics Y_lm (no Condon-Shortley phase) up to L, by the normalised recursion."',
  "# real spherical harmonics Y_lm (no Condon-Shortley phase) up to L, by the normalised recursion"),
 ('"The six sums sum_{lm} h_l(k|v|) Y_lm(vhat) Q^{ab}_{lm} for entries 11, 22, 33, 12, 13, 23."',
  "# the six sums sum_{lm} h_l(k|v|) Y_lm(vhat) Q^{ab}_{lm}, for entries 11, 22, 33, 12, 13, 23"),
 ('"Route (i): the whole difference box, one expansion about R."',
  "# route 1: the whole difference box, one expansion about R"),
 ('"Route (ii): the 8 octants of D, reflected from one table by T^{(sg)}_ab(V) = sg_a sg_b T_ab(sg V)."',
  "# route 2: the 8 octants of D, reflected from one table by T^{(sg)}_ab(V) = sg_a sg_b T_ab(sg V)"),
 ('"max and min |x-y| over an axis-aligned panel pair."',
  "# max and min |x - y| over an axis-aligned panel pair"),
 ('"Route (iii): the 36 face-pair k-series in BigFloat(KSRPRC), assembled by the srfSum! signs."',
  "# route 3: the 36 face-pair k-series in BigFloat(KSRPRC), assembled by the srfSum! signs"),
 ('"True for the egoToe indices farBlk! fills: max(i) >= 3, i.e. max-norm separation >= 2 cells."',
  "# true for the egoToe indices farBlk! fills: max(i) >= 3, i.e. max-norm separation >= 2 cells"),
 ('"Far tensor for one offset D in cells of shape s at frequency f; max-norm separation >= 2 required."',
  "# far tensor for one offset D in cells of shape s at frequency f; max-norm separation >= 2\n# required.  A test entry point: generation fills whole blocks with farBlk!"),
 ('"Fill egoToe[:,:,i1,i2,i3] for every index with max(i) >= 3 (offset i .- 1), threaded over offsets."',
  "# fill egoToe[:, :, i1, i2, i3] for every index with max(i) >= 3 (offset i .- 1), threaded"),
 ('"Frequency-contracted trapezoid table, thresholds and caches of the pair (sT, sS) at f; the equal-cell set of g is built on first use."',
  "# frequency-contracted trapezoid table, thresholds and caches of the pair (sT, sS) at f"),
 ('"The equal-cell set of the gcd cell g, built on first use and shared by every route-2 evaluation."',
  "# the equal-cell set of the gcd cell g, built on first use and shared by every route-2 evaluation"),
 ("\"Integer gcd-lattice coordinate m of R (sub-offsets are m + j - j' in cells of g); nothing off lattice.\"",
  "# integer gcd-lattice coordinate m of R, in cells of g; nothing when R is off that lattice"),
 ('"Float64 offset vector of the integer gcd-lattice coordinate m; the inverse of latInd."',
  "# the offset vector of the gcd-lattice coordinate m, R = m g + (nT - nS) g/2; inverse of latInd"),
 ("\"Multiplicity of the sub-offset difference t = j - j' over j in 1:nT, j' in 1:nS.\"",
  "# multiplicity of the sub-offset difference t = j - j' over j in 1:nT, j' in 1:nS"),
 ("\"Smallest max-norm over the sub-offsets m + j - j'; the equal-cell routes need it >= 2.\"",
  "# smallest max-norm over the sub-offsets m + j - j'; the equal-cell routes need it >= 2"),
 ('"The cells touch or overlap when |R_d| <= b_d on every axis; exact in both representations."',
  "# the cells touch or overlap when |R_d| <= b_d on every axis; exact in both representations"),
 ('"Route for one offset: (kind, L); 1 whole trapezoid box at l = L, 2 gcd average, 3 k-series."',
  "# route for one offset: (kind, L); 1 whole trapezoid box at l = L, 2 gcd average, 3 k-series"),
 ('"Route 1: the whole trapezoid box, one expansion about R."',
  "# route 1: the whole trapezoid box, one expansion about R"),
 ("\"Theorem A' bound on max_ab |T_ab - T_ab^(L)| of the whole trapezoid box at radius rr.\"",
  "# the Theorem A' bound on max_ab |T_ab - T_ab^(L)| of the whole trapezoid box at radius rr"),
 ('"Certified absolute bound on max_ab |T_ab - G_ab| of the equal-cell tensor at D as routed (kind, L, Lc)."',
  "# certified absolute bound on max_ab |T_ab - G_ab| of the equal-cell tensor at D, as routed"),
 ('"Route 2 for one offset: (1/N_t) sum of the equal-cell tensors at |D|, reflected; returns the summed certificate."',
  "# route 2: (1/N_t) sum of the reflected equal-cell tensors at |D|; returns the summed certificate"),
 ("\"Face F of the target (edges sT at R) and face F' of the source (edges sS at 0) as panels.\"",
  "# face F of the target (edges sT at R) and face F' of the source (edges sS at 0) as panels"),
 ('"Route 3: the 36 unequal face-pair k-series in BigFloat(KSRPRC), srfSum! signs, divided by V_t."',
  "# route 3: the 36 unequal face-pair k-series in BigFloat(KSRPRC), srfSum! signs, over V_t"),
 ('"Route 3 tensor at RQ from the memory or disk cache, else computed and stored."',
  "# route 3 tensor at RQ from the memory or disk cache, else computed and stored"),
 ('"Block size in fine cells that covers the offsets Rs and their sub-offsets."',
  "# block size in fine cells that covers the offsets Rs and their sub-offsets"),
 ('"Far tensor of the pair (target edges sT centred at R, source edges sS at 0) at f; cert = true also returns the certified bound on max_ab |T_ab - G_ab|."',
  "# far tensor of the pair (target edges sT centred at R, source edges sS at 0) at f; with\n# cert = true it also returns the certified bound on max_ab |T_ab - G_ab|"),
 ('"The box sum (1/N_t) sum_t mult(t) T(m + t) over the fine egoToe, reflected; returns the summed certificate from cg."',
  "# the box sum (1/N_t) sum_t mult(t) T(m + t) over the fine egoToe, reflected; returns the\n# summed certificate from cg"),
 ('"Fill G[:, :, q] for every exact offset Rs[q] of the pair, threaded, the near band as one box sum over a fine egoToe; returns the route kinds (with cert = true also the certificates)."',
  "# fill G[:, :, q] for every exact offset Rs[q] of the pair, threaded, the near band as one box\n# sum over a fine egoToe; returns the route kinds, and with cert = true the certificates"),
]
for a, b in DOC:
    if b == "":
        rep(a + "\n", "")
    else:
        rep(a, b)

open("far_v2_cmt.jl", "w").write(src)
print("ok", len(src.splitlines()))
