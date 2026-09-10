import re

src = open('notes/phase1/scratch/stage2.jl').read().split('\n')

# field renames on the OffDif struct
rep = [('d1o', 'dOne'), ('d1r', 'dRis'), ('d1f', 'dFal')]

HEAD = """# Closed-form moments I_m = int_A int_B |x - y|^m dS' dS of the face pairs of
# touching cuboid cells, m = -1, 0, 1, ...  The 4D integral collapses onto the
# separation w = x - y with a per-axis box-box convolution weight, so every pair
# is a positively weighted sum of axis-aligned boxes (parBxs): 2D with a frozen
# offset (box2), 3D with none (box3).  Generic in the number type."""

# comment lines inserted immediately before the anchor line
INS = {
'function asnD(': '# log1p branch below 1/2; the plain asinh difference cancels there',
'function cshMns(': '# all-positive series: d cosh d - sinh d cancels away for small d',
'function ladFal(': '# (i) h*zeta + C amplifies zeta by 2 delta, (ii) t2*zeta - (A2-A1) cancels by\n# 2 t2/(t2-t1); a box [p,2p] has delta <= log2/2, so (i) governs there',
'struct OffDif{T}': '# J[t1,t2;dh] - J[t1,t2;dl] without forming either J',
'@inline function pikCmb(': '# dh^2 J^h - dl^2 J^l and dh^2 D + ddsq J^l are both exact; take the smaller\n# term sum',
'function asnMns(': '# asinh(z) - z and x - atan(x), cancellation-free below 1/2',
'function potCrn(': '# the plain p asinh + q asinh - c atan has all three terms ~ pq/c when p,q << c;\n# regrouped, the leading p q (1/al + 1/be - 1/R) stands alone and is positive',
'@inline function atnY(': '# D_t atan(t s/(c R)) = atan(Y) with Y a ratio of positive quantities',
'function atnDD(': '# the four-corner difference where its amplification is small, else the u-first\n# or v-first subtraction formula; the two never degenerate together',
'function potBox(': '# u2 f(u2) - u1 f(u1) split as h (f2+f1) + um (f2-f1), the second factor an\n# exact offset difference',
'function box2Tay(': '# Exact, elementary and finitely truncated centre-Taylor seed -- not a\n# quadrature.  Covers every box whose distance to the origin exceeds its own\n# size, which is where the closed forms below pay their u2 v2/(hu hv) = 4, 8 or\n# 16 amplification.  Returns nothing and the caller falls back to them.',
'function logMns(': '# cancellation-free below 1/2',
'function powSum(': '# sum_{i=0}^{n-1} A^i B^{n-1-i}, so A^n - B^n = (A^2-B^2) powSum/(A+B)',
'function powSeg(': "# hi Sh^n - lo Sl^n re-associated as (hi-lo) Sh^n + lo (hi-lo)(hi+lo)\n# powSum(Sh,Sl,n)/(Sh+Sl): every term of the rung is positive.  One rounding per\n# rung does accumulate, log10((n+1)/2) + 0.08 digits, and psiBox, box3Mom and the\n# assembled I_m inherit it.",
'function powSegD(': '# grouped as d0^2 DJ_{n-2} + e2 J_{n-2}(d1), not d1^2 DJ + e2 J(d0): that keeps\n# the n = 1 step finite and unamplified as d0 -> 0',
'function wmnSed(': '# regrouped as (q-hi) J_{-1} + Ppos - hi (x - log1p x); the naive\n# q J_{-1} - (Sh-Sl) amplifies by 7, this by at most 2.6',
'function solAng(': '# van Oosterom-Strackee per triangle: a four-corner arctangent difference\n# cancels, and in the positive octant every dot product here is positive',
'function psiBox(': '# opposite-face pairs delta-grouped through powSegD, worth 0.3-0.4 digits\n# everywhere downstream',
'function box3Mom(': '# (m+3) K_m = sum_axes [hi_i Psi_m(hi_i) - lo_i Psi_m(lo_i)], delta-grouped',
'function box3Wmn(': '# div((q-u) X r^m) = [(m+4)(q-u) - q] r^m, so the near faces contribute their\n# own terms instead of being differenced away.  q >= hi[1]',
'function box3Wpl(': '# div((u-p) X r^m) = [(m+4)(u-p) + p] r^m.  p <= lo[1]',
'function polMom(': '# the naive antiderivative evaluates hi^{p+2}[1/(p+1) - 1/(p+2)] as written and\n# loses log10(4p) digits; expanded about the near end it is term-by-term\n# positive for either sign of d',
'function domAxs(': '# one dominant axis: the binomial expansion in (x^2+y^2)/z^2 <= 1/4 replaces the\n# divergence identity, whose face terms would be O(dist) larger than the integral',
'function thnAxs(': '# one negligible axis: term n of the mirror expansion weighs 1e-4n, so it needs\n# only 16 - 4n digits',
'function slvAxs(': '# thin axes, thinnest first.  The gate 2 delta < hi is strict, so hi = 2 lo --\n# the only shape an exact parBxs produces away from the origin -- keeps the\n# divergence identities; below it they amplify by hi/delta > 2 per thin axis',
'function slvOrd(': '# the gate bounds hf/md by 1/3, i.e. Bernstein rho >= 3 + sqrt(8), and an\n# n-point Gauss rule then converges like rho^{-2n} to eps(T)',
'function gauLeg(': '# hand-rolled Newton on the Legendre recurrence, so the rule is generic in T and\n# reaches eps(T) at any precision',
'function wplSeg(': '# every term of both forms is positive, so the smaller term sum cancels less',
'function box3Slv(': '# with two thin axes the two hi/delta amplifications multiply (6.9 digits\n# measured at delta/hi = 2.4e-4), so the thin axes are quadratured instead',
'function box3(': '# at most one axis may have dw != 0, and its weight is split as\n# cw + dw u = alp (hi-u) + bet (u-lo), both parts nonnegative',
'function axsCnv(': '# per-axis box-box convolution weight; (:frozen, offset) or\n# (:pieces, [(lo, hi, c, d)]) with weight c + d w on [lo, hi]',
'function parBxs(': '# Breakpoints are formed exactly in Rational{BigInt} and rounded to T once: two\n# that coincide mathematically but round an ulp apart emit a sliver box on which\n# every closed form amplifies by 1e16.  tol is the fallback for coordinates that\n# are not exactly representable, merging breakpoints closer than that.',
'const FACES': '# faces in the order yzL yzU xzL xzU xyL xyU: (degenerate axis, side)',
'scl2(x::AbstractFloat': '# exact multiplication by 2^k, whatever the panels are given as',
'parMom(pA, pB, mMax::Integer;': '# I_m for m = -1 .. mMax, out[m + 2] = I_m.  I_m(lam s) = lam^{m+4} I_m(s), so\n# the pair is prescaled to [1, 2) and each moment scaled back; both steps only\n# move exponents, so nothing overflows and nothing changes.',
'function momSer(': '# int_A int_B exp(2 pi i f r)/(4 pi r f^2) = (1/(4 pi f^2)) sum_n (2 pi i f)^n\n# I_{n-1}/n!.  rel = |last term|/|sum| is the term-count stopping rule, NOT an\n# error bound: it certifies term decay, not the conditioning of the partial sum.',
}

# inline comments: exact replacement text keyed by a unique substring of the code
INLINE = {
'H = L * (t2 + t1) / (A2 + A1)': '# A2 - A1, rationalized',
'dm1::T': '# asinh difference at m = -1, negative',
'h2 = ddsq / (A2h + A2l)': '# A2h - A2l',
'h1 = iszero(A1h + A1l)': '# A1h - A1l',
'gap = sq * (h2 + h1)': '# positive',
'lam > 1//2 || return nothing': '# certainly hopeless, do not even try',
'cf = Vector{Vector{T}}()': '# cf[n+1][i+1] = c_{i, n-i}',
'if n >= 3 && rel <= tol': '# two consecutive small levels',
'for n in nUse:-1:0, a in 1:3': '# smallest levels first',
'if n < -1                        #': '# J_{-3} exact and positive; below it, downwards',
'if n < -1                          #': '# downwards from the seed, low-order use only',
'v = T(NaN)': '# true value -Inf; killed by d0^2 = 0',
'pw(q) = q == 0': '# int u^{q-1}, lo > 0',
'iszero(bin) && break          #': '# even m: the expansion terminates at n = m/2',
'j = 6 - i - i2': '# the one axis left analytic',
'if tol !== nothing': '# merge near-coincident breakpoints',
'for (lo, hi, c, dd) in cmb': '# split at 0, then reflect',
'mo = isodd(mMax)': '# top odd m, >= -1',
'me = iseven(mMax)': '# top even m, or -2 (none)',
'out[2 * k - 2 + 1] += vo[k]': '# m = -1 + 2(k-1)',
'out[2 * k] += ve[k]': '# m = 2(k-1)',
}

# standalone comments kept inside function bodies, keyed by a substring
BODY = {
'# projection: single-level ratios oscillate': '        # single-level ratios oscillate, so project on the geometric mean',
'# route 1 integrates the u weight first': '        # both routes are exact; averaging restores the u<->v symmetry of the box',
'# the domain of convergence of the series does not depend on m': '        # the domain of convergence does not depend on m, so a series seed lets every\n        # rung come from the series too, and the ladder accumulates no rounding',
'# only the thin-axis expansion needs this': '        # only the thin-axis expansion needs this, and there t = 0 kills the Psi_{n-2}',
}

out = []
i = 0
pend_body = None
while i < len(src):
    ln = src[i]
    st = ln.strip()
    if st.startswith('#'):
        hit = None
        for k, v in BODY.items():
            if st.startswith(k):
                hit = v
                break
        if hit is not None:
            out.append(hit)
        # skip the whole comment block
        while i < len(src) and src[i].strip().startswith('#'):
            i += 1
        continue
    # anchor insertion
    for k, v in INS.items():
        if ln.startswith(k):
            out.append(v)
            break
    # inline comment handling
    if '#' in ln:
        code = ln[:ln.index('#')].rstrip()
        keep = None
        for k, v in INLINE.items():
            if k in ln:
                keep = v
                break
        if keep is None:
            ln = code
        else:
            pad = max(1, 80 - len(code) - len(keep))
            ln = code + ' ' * pad + keep
    out.append(ln)
    i += 1

txt = '\n'.join(out)
for a, b in rep:
    txt = re.sub(r'\b' + a + r'\b', b, txt)
txt = re.sub(r'\n{3,}', '\n\n', txt).strip('\n')
open('src/vacuum/glaVacOprMemMom.jl', 'w').write(HEAD + '\n\n' + txt + '\n')
