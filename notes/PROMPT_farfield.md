# Prompt: the far field of Gila without quadrature — a cell-size expansion for every separated pair

## Context and what already exists

Read `notes/integrals.pdf` (source `notes/integrals.tex`) and `notes/moments/moments.pdf`
(source `notes/moments/moments.tex`) fully before doing anything else. The first explains
where Gila's integrals come from and how the operator is assembled; the second gives closed
forms for every moment `I_m(F,F') = ∫_F ∫_{F'} |x−y|^m dS' dS` of every face pair of touching
cells, and shows that the power series in k built from those moments reproduces Gila's
contact and touching-shell integrals to the floating-point floor. That series is the right
tool when kD is at most a few units (D the diameter of the pair); it is the wrong tool for
separated cells, where kD grows linearly with the separation, the series needs about e·kD
terms, and it cancels about kD/ln 10 digits.

Today the separated pairs are done by a fixed tensor Gauss–Legendre rule whose order comes
from the measured schedule in `quadOrd` (9 points per dimension at 2 cells down to 4 beyond
16 cells), applied to each of the 36 face-pair integrals and assembled by `srfSum!` into the
3×3 tensor per relative offset. Two facts make this the bottleneck of Gila and the reason
for this task:

- Cost. A 128³ self-volume needs one tensor per offset of the doubled Toeplitz grid,
  about 1.7·10⁷ offsets, each from 36 face pairs at 4⁴ to 9⁴ kernel evaluations: of the
  order of 10¹¹ complex exponentials, i.e. ten to thirty minutes on a workstation.
- Accuracy. The face-pair rule is good to 10⁻¹² to 10⁻¹³ per pair, but `srfSum!` subtracts
  nearly equal face pairs and amplifies that by 10 to 1000, so the assembled far tensor is
  good to about 10⁻¹⁰. For separated cells the face-pair detour is unnecessary: it exists
  only to tame a singularity that a separated pair does not have.

## Task statement

**Replace the fixed Gauss–Legendre rules for separated cells by convergent elementary
expansions with a priori error control, so that every entry of Gila's discretized Green
operator, for every relative offset, is computed without numerical quadrature, to an
assembled-tensor accuracy of 10⁻¹³ or better in Float64, at a cost of at most a few hundred
floating-point operations per offset.** The result must cover every cell shape (three
independent edge lengths, aspect ratios from 10⁻³ to 10³ and the actual shapes
(1/32, 1/32, 1/32) and (1/32, 1/32, 1/512) in wavelengths), every separation from the
first non-touching offset outward, real and complex frequency, and must join seamlessly onto
the moment series of `notes/moments` in the near field. The final measure of success is a
128³ self-volume Green function at λ/32 built in seconds, not minutes, with every tensor entry
verified against a 220-bit reference to 10⁻¹³.

Partial progress does not count unless it implies exactly the resolution above. In particular:
an expansion whose truncation order is chosen by numerical experiment rather than by a proven
bound, a method that works only for cubic cells, a method verified only at one separation or
one frequency, a scheme that still calls a quadrature rule in some band of separations without
a proof that no expansion converges there, and a speedup measured on a toy problem rather
than on the actual `GlaVacOprMem` build, are all insufficient.

## Groundwork: measurements that must be made before any integral is evaluated

Everything below must be measured and written down first. Each item names what is unknown
today; do not guess it.

1. **Profile the actual build.** Time a `GlaVacOprMem` build for a self-volume at 32³, 64³ and
   128³ cells at λ/32 (and 32³ at λ/8 and λ/4), single-threaded and multi-threaded, and if the
   GPU path is available also on the GPU. Attribute the time to: contact integrals
   (`wekS`, `wekE`, `wekV`), touching-shell rule (order 9), separated pairs by `quadOrd` band
   (order 9, 7, 6, 5, 4), the `srfSum!` assembly, and the FFT of the Toeplitz block. Count
   the kernel evaluations per band. Confirm or refute the estimate of 10¹¹ evaluations. This
   profile is the baseline every later speedup is measured against.
2. **Count the distinct offsets Gila actually evaluates.** Determine whether the assembly
   exploits offset → −offset and the three axis reflections (the tensor components map onto
   each other up to signs) or evaluates all offsets of the doubled grid independently. The
   factor at stake is up to eight and it is independent of everything else in this task.
3. **Measure the true far-field error today.** For separations 2, 3, 4, 6, 8, 16, 32, 64
   cells along an axis, along a face diagonal and along the body diagonal, at λ/32, λ/8 and
   λ/4, at f = 1 and f = 1 + 0.1i, compute the assembled 3×3 tensor from `egoSrfFxd!` and
   compare with a 220-bit reference (`pairKer` of `notes/gen/ref/mom.jl` on each of the 36
   face pairs, then the same signed sum). Report per-pair error, assembly amplification, and
   tensor error. This fixes the accuracy target the expansion must beat and documents the
   amplification the volume formulation removes.
4. **Establish the volume formulation numerically.** For the same offsets confirm to 10⁻³⁰
   that the signed sum of the 36 face-pair integrals of g equals the volume–volume integral
   of ∂_a∂_b g (equation (7) of `integrals.tex` read backwards; the identity term is absent
   for separated cells). This is the formulation the far field will use; its normalization
   against Gila's conventions (`srfScl`, `celInv`, the 4π² and 1/f² factors) must be pinned
   down here, with numbers, not later.
5. **Measure the analyticity margins.** For each cell shape and separation, compute
   ρ = max|δ| / |R| where δ ranges over the difference set of the two cells (a box of edges
   2s_i centred at R) and R is the centre separation. Tabulate ρ over the lattice: it is the
   convergence ratio of any expansion in cell size, and it is 0.87 at two cells for cubes,
   about 0.43 at four, 0.22 at eight. For slender cells the relevant quantity is direction
   dependent; tabulate it per offset class.
6. **Measure the conditioning of the derivative recursion.** The expansion needs Cartesian
   derivatives ∂^α of g(r) = e^{ikr}/(4πf²r) at R up to total order 15 to 25. Implement the
   radial-derivative recursion and the Cartesian assembly (Hobson's formula, or the
   recursion for derivatives of radial functions), and measure in BigFloat versus Float64
   the digits lost as a function of order, kR and direction, for kR from 0.3 to 300. Whether
   this recursion is stable to order 25 at kR ≈ 1 determines the whole design.
7. **Measure the exact polynomial moments.** The volume-pair moments μ_α = ∫∫ δ^α are
   products of one-dimensional moments of the triangle weight of width 2s_i, exact rationals
   in the lengths; write them down, note which vanish by symmetry, and check that `moments.jl`'s
   even moments agree with them for the face-pair case.
8. **Fix the joint error target.** Decide, with numbers, what the assembled tensor accuracy
   must be so that the anti-Hermitian positivity check of Section 2.4 of `integrals.tex`
   and the solve accuracy are not limited by the kernel: the earlier work targeted 10⁻¹⁰;
   this task targets 10⁻¹³; state what the FFT and the solver can use.

## The integrals

For a source cell V_s centred at the origin and a target cell V_t centred at R (both with
edges s₁, s₂, s₃; a regular grid, so R is an integer vector of cell lengths), Gila needs

    (G₀)_ab(R) = (1/(k² V_t)) ∫_{V_t} ∫_{V_s} ∂_a ∂_b g(|x − y|) dy dx,   g(r) = e^{ikr}/(4π f² r),

for a ≠ b, and the same with −δ_ab ∇² replacing ∂_a∂_b for the diagonal (see equation (5)
of `integrals.tex`; with the Helmholtz equation the diagonal reduces to derivatives of g as
well, since ∇²g = −k²g away from the origin). Writing x − y = R + δ, δ ranges over the
difference box D = Π_i [−s_i, s_i] with the triangle weight w(δ) = Π_i (s_i − |δ_i|). So every
far entry is

    (G₀)_ab(R) = (1/(k² V_t)) ∫_D w(δ) (∂_a∂_b g)(R + δ) dδ,

an integral of a smooth function over a box with a product weight. The face-pair
decomposition into 36 integrals of g itself is equivalent and available as a check, but the
volume form has no cancelling signed sum.

## Approach families to keep alive

Keep a written registry grouped by mechanism. At least these families must be alive in the
first round, each on at least two separations and two cell shapes, and the far/near boundary
must be attacked by at least two of them.

(a) **Cartesian Taylor expansion about the centre separation.** ∂_a∂_b g(R + δ) =
    Σ_α (∂^{α+e_a+e_b} g)(R) δ^α / α!, so the entry is Σ_α (∂^{α+e_a+e_b} g)(R) μ_α / α!
    with the exact moments μ_α of item 7. Odd moments vanish, so only even α contribute,
    and for the diagonal entries the sum organizes by total degree. Deliver: the derivative
    recursion, an a priori remainder bound of the form C · ρ^{p+1}/(1−ρ) from the Cauchy
    estimate of g on the ball of radius |R| − max|δ| (g is analytic there), the number of
    terms needed for 10⁻¹³ as a function of ρ and kR, and the measured Float64 digits lost.
(b) **Spherical addition theorem.** e^{ik|R+δ|}/|R+δ| = ik Σ_l (2l+1) j_l(k|δ|) h_l^{(1)}(k|R|)
    P_l(cos θ) for |δ| < |R|. The singular factors h_l(kR) are elementary (spherical Hankel
    functions are finite sums of e^{ikR} times powers of 1/kR); the regular factors
    j_l(k|δ|) P_l(cos θ) are entire and, expanded in powers of k, become homogeneous
    polynomials in δ whose box integrals against w(δ) are exact moments. The regular part of
    each l is therefore a power series in k with exact rational-in-lengths coefficients,
    frequency-independent geometry data. Deliver the convergence in l (ratio about ρ), the
    number of k-terms per l, the conditioning of h_l at small kR (they grow like (kR)^{−l−1},
    and the cancellation between l terms must be measured), and a comparison with (a): they
    are the same expansion reorganized, and the reorganization may or may not be better
    conditioned.
(c) **One exact axis, two expanded.** Along one axis (the one with the largest s_i, or the
    one aligned with R) the integral of g against the triangle weight can be done in closed
    form using the moments machinery of `notes/moments` (the difference variable along one
    axis with the other two coordinates fixed gives elementary one-dimensional integrals of
    r^m; the k-series in that one variable converges for the small kD of one edge even when
    the whole pair is far). Expand only in the remaining two coordinates. This shrinks ρ to
    the two-dimensional ratio and is the natural tool for slender cells, where one edge is
    16 times shorter than the others and the isotropic ρ is dominated by the long edges.
(d) **Subdivision.** Split the difference box into 2³ or 4³ sub-boxes, each with its own
    centre R_j and a smaller ρ_j, and apply (a) or (b) per sub-box. Cost multiplies by the
    number of sub-boxes, convergence ratio divides by the subdivision factor. Find the
    optimum for each separation band. This is the direct way to push the far method inward
    to two and three cells.
(e) **Spectral/Ewald split.** The Fourier transform of g is 1/(|ξ|² − k²), and cell averaging
    is a product of sinc² factors in ξ. The discretized operator is therefore a sampled,
    filtered Green function, and its Toeplitz kernel might be obtained for all offsets at
    once by an FFT of an analytic spectral kernel, with the near field corrected in real
    space by the moment series. The obstruction is aliasing and the singularity at |ξ| = k.
    Deliver at least a quantitative statement of the aliasing error for Gila's grids and a
    decision whether this family is viable; if it is, it changes the cost model entirely
    (O(N log N) for the whole far field).
(f) **Interpolation of the smooth far tensor.** Away from the origin the tensor entries are
    analytic functions of R. Evaluate (a) on a coarse sub-lattice and interpolate (Chebyshev
    or polynomial in each coordinate) to the full lattice, with an a priori interpolation
    bound from the same analyticity radius. This trades accuracy for a further constant
    factor and is only worth it if (a) alone is not fast enough after profiling.
(g) **Derivative-free variants.** The high-order Cartesian derivatives of g are the
    conditioning risk of (a). Alternatives to measure: generate the Taylor coefficients of
    g(R + δ) directly by the recurrence that the ODE for e^{ikr}/r induces on its Taylor
    coefficients in each coordinate (a two-term recurrence, like the ones that made the
    moments cancellation-free), or by automatic differentiation in BigFloat once per offset
    class followed by rounding.

Do not tell most sub-agents which family currently looks best; preserve independence in the
first rounds. Do not let one family win because its formulas are elegant: an expansion whose
remainder bound is not proven, or whose Float64 evaluation loses three digits at order 20,
has not solved anything.

## Bridging the gap between near and far

The near field is the k-series of `notes/moments` (converges for kD small: all pairs within
about 15 cells at λ/32, 4 cells at λ/8, 2 cells at λ/4). The far field is any of (a)–(d)
(converges for ρ small: comfortable beyond 6 to 8 cells for cubes, slower closer in). The
overlap is wide at fine cells and vanishes at coarse cells; the task is to make the join
seamless and provable everywhere Gila operates (cells from λ/128 to λ/4). Ideas to develop,
each with a measured convergence table:

- **Use both and compare in the overlap.** Where both converge, evaluate both; their
  agreement is a free end-to-end check of everything from geometry bookkeeping to
  normalization, and its measured size becomes the acceptance test of the build.
- **Extend the k-series outward with a shifted exponential.** Write e^{ikr} = e^{ikR₀}
  e^{ik(r − R₀)} with R₀ = |R| and expand in (r − R₀), which is small over the pair even when
  kR₀ is large. The moments ∫∫ w (r − R₀)^n are combinations of the elementary moments I_j
  with large cancellation if formed naively; the question is whether they have a direct
  cancellation-free form (they are the moments of r about its own mean, and the centred
  moments of a far pair are what family (a) computes term by term). If they do, the k-series
  and the cell-size expansion become one object.
- **Extend the far expansion inward by subdivision (d)** until its ρ at two cells is below
  0.5, and measure the cost crossover against the k-series with extra terms.
- **Exact axis (c) in the transition band.** With one axis done exactly, the two-dimensional
  ρ is smaller than the three-dimensional one; measure whether this alone covers two to four
  cells at λ/4.
- **A priori band selection.** From the proven bounds (kD)^{N+1}/(N+1)! · e^{kD}/cos(kD) for
  the k-series and C ρ^{p+1}/(1 − ρ) for the cell-size expansion, derive for each cell shape
  and frequency the separation at which each method reaches 10⁻¹³ within a fixed term budget,
  and choose per offset by the cheaper method. The band edges must come out of the bounds,
  not out of trial and error; trial and error is only allowed to confirm them.
- **The Float64 question.** Both methods are sums of terms with a proven worst-case
  amplification (e^{kD} for one, Σ|terms|/|sum| bounded by 1/(1−ρ) for the other). Decide
  whether Float64 suffices everywhere, or whether the build should form the sums in
  double-double or 128-bit BigFloat and round (the moments are frequency-independent and
  cost microseconds, so a higher-precision build-time path is cheap); measure both.

## Resources and environment

- Julia is installed. Always run it as
  `JULIA_NUM_THREADS=1 julia --startup-file=no --project=<env> script.jl`; the
  `--startup-file=no` is essential. For a Gila-enabled environment, create a throwaway one
  that `Pkg.develop`s the repository. Never modify files under `src/`, `test/`,
  `Project.toml`, or `Manifest.toml`; wiring into Gila is a separate later task.
- `notes/moments/moments.jl`: `pairMoments(A, B, mMax)` gives every moment I_m of any
  axis-aligned rectangle pair in closed form, generic in the number type; `momentSeries`
  sums the k-series; `faceMoments`, `facePair`, `geomCode` do the lattice bookkeeping;
  `verify.jl` and `gilacmp.jl` show how the reference comparisons and the Gila comparisons
  were done. Use them; do not re-derive the near field.
- `notes/gen/ref/mom.jl`: the 220-bit evaluator (`pairMom`, `pairKer`). It is the reference
  for every face-pair integral; for the volume–volume integrals build the reference from the
  36 face pairs (item 4) or extend it, and validate the extension against the exact even
  moments first.
- `src/vacuum/glaVacOprMemGen.jl`: `egoSrfFxd!`, `quadOrd`, `cntOrd`, `pairListUn`,
  `egoFunSng!`, `srfSum!`, `srfScl` — the current far-field path and the normalizations.
  `notes/moments/gilacmp.jl` documents how to call it and how raw face-pair integrals are
  recovered from `srfMat`.
- Public search may be used for standard background (Cartesian and spherical multipole
  expansions of the Helmholtz Green function, Hobson's theorem, derivative recursions for
  radial functions, Ewald summation for the Helmholtz kernel, kernel-independent fast
  multipole methods), not for a solution to this exact problem.
- Write everything under your scratchpad except the final deliverables, which go in
  `notes/farfield/`: the Julia implementation, the verification and benchmark scripts, and
  the report.

## How to run the search

You are the root: plan, synthesize, challenge, redirect, audit. Launch sub-agents with
`model: "opus"` for derivations, conditioning analyses and difficult verification, and
`model: "sonnet"` for profiling, sweeps, tabulation and plotting. Tell every sub-agent to
write its report with a Bash heredoc if the Write tool refuses `.md` files, and to report
formulas, numbers and script paths, never status. Never read a sub-agent's transcript; read
its report. Keep the registry of families current after every round; mark a route blocked
for a band only with the measurement that blocked it, and reopen it only for a materially
new mechanism. Use adversarial sub-agents: every expansion must be checked against the
220-bit reference at every separation band, every cell shape including the slender one,
real and complex frequency, all offset directions (axis, face diagonal, body diagonal, and
random lattice directions), for the correct symmetry under offset → −offset and the axis
reflections, for homogeneity in f, and for the anti-Hermitian positivity of the assembled
operator on a small block. A method that reaches 10⁻¹³ at eight cells and 10⁻⁹ at three has
not solved the problem; say where it stands and keep going.

## Persistence

This is a hard problem in a different way from the near field: the mathematics is classical
(local expansions of the Helmholtz kernel are the core of every fast multipole method), and
the difficulty is conditioning, coverage of the transition band, slender cells, and turning
a convergent expansion into a provably bounded Float64 evaluation that is actually fast in
Gila's build. Expect the first derivative recursion to lose digits at high order, the first
remainder bound to be too loose by orders of magnitude, the slender cell to break the
isotropic ρ, and the transition band at λ/4 to resist. None of these is a reason to return.
Do not return with "the fixed rule is kept between two and four cells" unless you can show
that no expansion in the registry converges there within the term budget and that
subdivision does not fix it. Do not return with a speedup that was not measured on the
actual 128³ build. Spend at least eight hours before even thinking of returning.

## Final deliverable

Only at the end, when the method is complete and audited: `notes/farfield/farfield.jl`
(standalone, generic in the number type, every constant typed), `notes/farfield/verify.jl`
producing every table below, `notes/farfield/bench.jl` producing the build-time comparison,
and a LaTeX document `notes/farfield/farfield.tex` compiled to `farfield.pdf` with pdflatex
(check the rendered pages) containing: the groundwork measurements of the first section; the
integrals and the volume formulation with its normalization pinned to Gila's; each expansion
with its derivation, its a priori remainder bound, the term-count tables that follow from the
bound for 10⁻¹³ at every separation band and cell shape, and its measured Float64 digits
lost; the near/far join with the proven band edges; the verification tables against the
220-bit reference for all bands, shapes, directions and frequencies; the anti-Hermitian
positivity check; the measured build times before and after for 32³, 64³ and 128³ at λ/32
and the tensor accuracy before and after; the registry of approach families with what worked
and what was blocked and why; and an honest section on what remains uncertain. Attribute
yourself as the author in the same style as the earlier documents. Write it once, at the
end, from the audited results.
