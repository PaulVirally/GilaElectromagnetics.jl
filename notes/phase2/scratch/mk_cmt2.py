src = open("far_v2_cmt.jl").read()
def rep(old, new, n=1):
    global src
    c = src.count(old); assert c == n, f"count {c} != {n} for:\n{old[:160]}"
    src = src.replace(old, new)

rep("""# The n-truncation lives inside the same organization: replacing j_l by its series remainder after
# n = N and integrating the pointwise majorant against the same positive measures gives Theorem A
# with |k|^l e^{...}/(2l+1)!! -> bslTal(l,N,|k| r_d)/r_d^{l+2N+2} and V^{ab}_l -> V^{ab}_{l+2N+2}.
# The budget is tol est/(NBUD (lTop+1)), NBUD = 256, so that the whole n-truncation contributes at
# most tol/256 ~ 4e-16 of est: the l-truncation is what tol is spent on, and nCut is only there to
# certify that nMax stored orders are enough.  The bound is evaluated at the smallest radius the
# expansion is ACTUALLY used at (route (i) offsets and the sub-box centres of route (ii)) and at
# the far end of the block, since the bound-to-est ratio is decreasing in |R| for l >= 2 and
# increasing for l = 0; at 2 min(s) it is vacuous whenever rho > 1 there, which is why the old
# code silently returned nMax for every l >= 2 (measured: nCut = 12 for l >= 2, all four shapes).
const NBUD = 256
""",
"""# Truncating j_l's own series after n = N is Theorem A again, with |k|^l e^{..}/(2l+1)!! replaced
# by bslTal(l, N, |k| r_d)/r_d^{l+2N+2} and V^{ab}_l by V^{ab}_{l+2N+2}.  The budget is
# tol est/(NBUD (lTop+1)): tol is spent on the l-truncation, and nCut only certifies that the
# stored orders are enough.  The bound is evaluated at both ends of the radius range the expansion
# is actually used over, since its ratio to est falls with |R| for l >= 2 and rises for l = 0.
const NBUD = 256                    # so the whole n-truncation costs at most tol/256 of est
""")

rep("""    # The k-series order the table must carry is NOT a compile-time constant: bslTal grows as
    # (|k| r_d)^{2N+2}, so N rises with the cell size in wavelengths.  nMax is a floor (so that a
    # table already on disk at NMAX is reused); the cut is recomputed against nCap when the floor
    # does not meet the budget, and the moment table is widened only then.
    # -1 means "no N <= nc meets the budget" and must survive the max over the two radii: a plain
    # max.() swallowed it whenever the far end was met at nMax while the near end was not (D11)
""",
"""    # bslTal grows as (|k| r_d)^{2N+2}, so the order the table must carry rises with the cell size
    # in wavelengths.  nMax is only a floor, so a table already on disk at NMAX is reused; the cut
    # is recomputed against nCap, and the moment table widened, only when the floor falls short.
    # cutMax, not max.(): -1 means "no N <= nc meets the budget" and must survive the two radii.
""")

rep("""# est(R) is an a priori SCALE, not a proven lower bound on max_ab |T_ab|, and the certificate
# "relative error <= tol" needs one: bound <= tol * X implies bound <= tol * max|T| only if
# X <= max|T|.  Measured over 403 references, est/max|T| is 1.10 to 6.34 on the cubes (135 on the
# needle), i.e. est OVER-states and what the rule certifies is tol * est/max|T|, not tol.  estLo
# is a proven lower bound, from the trace:
# Helmholtz gives lap g = -k^2 g away from the source, so tr T = 2 k^2 S with
# S = (1/V_t) int_D w g(R+d) dd, and max_ab |T_ab| >= |tr T|/3 >= (2|k|^2/3)(|S_0| - tailS).
# S_0 is the l = 0 shell, |S_0| >= (|k|/(4 pi |f|^2)) (e^{-Im(k)|R|}/(|k||R|)) V_t j_0(|k| r_d)
# (j_0 is positive and decreasing on [0, pi], which is the hypothesis |k| r_d <= pi), and tailS is
# the Theorem A sum with V^{ab}_l replaced by W_l and no derivative factor.  It returns 0 when
# either hypothesis fails: |k| r_d > pi (e0 = 0), or the tail eating the leading shell, whose
# ratio is |k|^2 r_d^2/(18 j_0(|k| r_d)) at large |kR| -- 0.0065 at (1/32)^3, 0.14 at (1/8)^3,
# and 2.86 at (1/4)^3, where the bound is vacuous at every offset.  Hence scl = :est by default.
""",
"""# A proven lower bound on max_ab |T_ab|, from the trace.  Helmholtz gives lap g = -k^2 g away
# from the source, so tr T = 2 k^2 S with S = (1/V_t) int_D w g(R+d) dd, and
# max_ab |T_ab| >= |tr T|/3 >= (2|k|^2/3)(|S_0| - tailS), where S_0 is the l = 0 shell and tailS
# the Theorem A sum with V^{ab}_l replaced by W_l and no derivative factor.  It needs
# |k| r_d <= pi (j_0 positive and decreasing there) and the leading shell to survive its tail, and
# returns 0 otherwise -- which it does at every offset of a lambda/4 cell.  Hence scl = :est.
""")

rep("""# The cost model is diagnostic only: the rule short-circuits to (i) whenever (i) converges, so
# cost(i)/cost(ii) is never read.  Evaluating it anyway costs 309-312 ns of the 988-1095 ns per
# offset, i.e. 29-31% of a whole far-field build, so it is off by default.
# the k-series cut of both tables is certified on [rNc, rHi] only (D15, the equal-cell twin of D13:
# a set built for other offsets gave 3.2e-8 at a nearer one); a set from farTns/farBlk! covers
# its own offsets, so the check can fire only for a hand-built or reused set
""",
"""# The cost model is diagnostic only, and off by default: routing short-circuits to route 1
# whenever route 1 converges, so the ratio is never read, and evaluating it costs 30% of a build.
# A set's k-series cut is certified on [rNc, rHi] only, and reusing a set outside that interval
# returned a wrong tensor under a valid-looking certificate.  A set built by farTns or farBlk!
# covers its own offsets, so the check can fire only for a hand-built or reused one.
""")

rep("""# G(R; sT, sS) = (1/V_t) int_D w(d) [(da db + dab k^2) g](R+d) dd with the trapezoid weight of the
# pair, D = prod [-b_i, b_i], V_t = prod sT; (1/V_t) int_D w = V_s, so est takes V_s while the
# Theorem A'/B prefactor keeps V_t.  The whole-box table is keyed on (a, b), which is symmetric in
# the pair, and stored divided by prod(b); prod(b)/V_t is folded into prf.  G(R; sT, sS) =
# (V_s/V_t) G(-R; sS, sT).  Routes per exact offset R of the target centre (source at 0):
#   1  the whole trapezoid box (Theorem A', r_d = |b|, ramp and point measures), one expansion;
#   2  the gcd average G(R) = (1/N_t) sum_{j,j'} G(R + c_j - c'_j'; g, g), g = min(sT, sS) per axis,
#      N_t x N_s equal-cell tensors at INTEGER fine offsets (Gila's grids sit on the common gcd
#      lattice, so the half-integer R/g cancels in R + c_j - c'_j'); farTnsX sums farTns's
#      routes, farBlkX! takes it as a box sum over one fine egoToe filled by farBlk!;
#   3  the k-series on the 36 unequal face pairs in BigFloat(KSRPRC), off the lattice or when a
#      sub-offset would touch.
# Touching pairs (|R_i| <= b_i on every axis) are Gila's contact path and are refused.
""",
"""# G(R; sT, sS) = (1/V_t) int_D w(d) [(da db + dab k^2) g](R+d) dd with the trapezoid weight of the
# pair, D = prod [-b_i, b_i], V_t = prod sT; (1/V_t) int_D w = V_s, so est takes V_s while the
# Theorem A'/B prefactor keeps V_t.  The whole-box table is keyed on (a, b), symmetric in the
# pair, and stored divided by prod(b), with prod(b)/V_t folded into prf.  Routes per exact offset
# R of the target centre, source at 0:
#   1  the whole trapezoid box (Theorem A', r_d = |b|, ramp and point measures), one expansion;
#   2  the gcd average G(R) = (1/N_t) sum_{j,j'} G(R + c_j - c'_j'; g, g), g = min(sT, sS) per
#      axis: N_t x N_s equal-cell tensors at integer offsets in cells of g, since Gila's grids sit
#      on the common gcd lattice.  farTnsX sums farTns, farBlkX! takes a box sum over one fine
#      egoToe filled by farBlk!;
#   3  the k-series on the 36 unequal face pairs, off the lattice or when a sub-offset would touch.
# Touching pairs (|R_i| <= b_i on every axis) are Gila's contact path and are refused.
""")

open("far_v2_cmt.jl", "w").write(src)
print("ok", len(src.splitlines()))
