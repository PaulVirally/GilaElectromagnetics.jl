# Prompt: closed forms for the remaining panel-pair integrals of GilaElectromagnetics

## Current task statement

The attached document `notes/integrals.pdf` (source `notes/integrals.tex`, in the
repository at `/Users/pvirally/.julia/dev/GilaElectromagnetics`) explains where the
integrals of the Julia package GilaElectromagnetics come from, which of them are
already in closed form, and which are not. Read it fully before doing anything else.
It is your specification, your literature review, and your record of what has already
been tried and measured. Do not redo work it reports as done and verified.

The objects of study are the moments

    I_m(F, F') = ∫_F ∫_{F'} |x − y|^m dS' dS,     m = −1, 0, 1, 2, 3, …

for pairs (F, F') of axis-aligned rectangles in R^3 that are faces of two touching or
coincident cuboid cells. Even m gives polynomials and is trivial. The content is in
odd m and in m = −1. Section 5.1 of the document proves constructively that every
I_m with integer m ≥ −1 is an elementary function of the edge lengths and offsets
(polynomials, square roots of sums of squares, asinh, atan). Section 5.2 exhibits the
result for the self face for every m at once. The rest has not been done.

Resolve the following completely:

**For each of the ten rectangle-pair geometries that occur between touching cells on
a regular grid, produce elementary closed forms for I_{−1} and for every odd moment
I_{2m+1}, m = 0, 1, 2, …, either as an explicit formula valid for general m or as a
verified recurrence in m with explicit elementary seeds, together with a
cancellation-free floating-point arrangement of each.**

The ten geometries, in the codes of Table 1 of the document, are:

- S: coincident faces (done for all m, Section 5.2 of the document; use it as the
  template and as a test of your methods, not as new work);
- Ef: coplanar faces sharing an edge (flat edge). I_{−1} is done (`rSrfEdgFlt`);
- Ec: perpendicular faces sharing an edge (cornered edge). I_{−1} is done
  (`rSrfEdgCrn`);
- Vc: coplanar faces sharing only a corner. Nothing done;
- Vp: perpendicular faces sharing only a corner. Nothing done;
- P1, P2: parallel faces one or two cell lengths apart, aligned;
- P1s, P2s: parallel faces one or two cell lengths apart, laterally shifted by one
  cell length;
- Xg: perpendicular faces that do not meet.

All formulas must hold for general rectangular cells with three independent edge
lengths s_1, s_2, s_3, in every orientation the tables of the document list. Assume
for the purposes of this task that the closed forms exist and can be found; the
theorem in Section 5.1 guarantees it, and the self-face case shows what they look
like.

A complete solution consists of exactly the following, for every one of the ten
geometries:

1. The closed form or recurrence, written out in LaTeX, with its derivation.
2. Numerical verification against an independent high-precision reference to at
   least 1e-30 relative, for m = −1, 0, …, 12, at cubic cells and at slender cells with
   aspect ratios 1e-3 and 1e3 in each independent direction, and at the actual cell
   shapes (1/32, 1/32, 1/32) and (1/32, 1/32, 1/512) in wavelengths.
3. A cancellation-free arrangement whose Float64 evaluation loses at most one decimal
   digit against the high-precision value over aspect ratios from 1e-6 to 1e6, with a
   measured digits-lost table.
4. A demonstration that the series of Section 5 of the document built from these
   moments reproduces Gila's `wekS`, `wekE`, `wekV` values, and the order-9 fixed-rule
   values for the non-touching pairs, to the accuracy of those numerics (about 1e-11
   to 1e-13; the series will be the more accurate side).
5. A standalone Julia file implementing every formula generically over the number type
   (so that BigFloat inputs give BigFloat results; type every constant, see the
   pitfall in Section 4.7 of the document), with the verification script that
   produces the tables above. Do not modify the package source; wiring into Gila is a
   separate later task.

Partial progress does not count unless it implies exactly the resolution above. In
particular: numerical quadrature of any moment, a series whose coefficients are
computed numerically, formulas valid only for cubic cells or only for one orientation,
formulas verified only at aspect ratio one, formulas for a subset of the geometries,
formulas left in a form that cancels catastrophically for slender panels, and
Mathematica-style outputs pasted without regrouping and without verification, are all
insufficient. The perpendicular geometries (Ec, Vp, Xg) are the hard part; do not
return with only the coplanar ones.

## Resources and environment

- Julia is installed. Always run it as
  `JULIA_NUM_THREADS=1 julia --startup-file=no --project=<env> script.jl`; the
  `--startup-file=no` is essential because the user's startup file activates a
  different project. The repository environment may fail to precompile (a stale
  Manifest); if it does, create a throwaway environment in your scratchpad that
  `Pkg.develop`s the repository path, and `Pkg.add` what you need there (SymPy or
  Symbolics for symbolic work, QuadGK, SpecialFunctions, Arblib if useful). Never
  modify files under `src/`, `test/`, `Project.toml`, or `Manifest.toml`.
- `notes/gen/ref/mom.jl` is the validated 220-bit numerical evaluator of all moments
  of any rectangle pair (`pairMom`), of the full kernel (`pairKer`), and of Gila's
  geometries (`pnlPair`, `grdScl`); Section 6.6 of the document describes how it
  works. It is your independent reference for every verification. `mom7.jl` holds the
  general self-face formula, its recurrence, and its cancellation-free version;
  `ref.jl` and `formsG.jl` hold the m = −1 closed forms; `slf.jl` and `dec.jl` the
  special-function reduction of the self face; `moments_report.md` the analysis of the
  self-face moments.
- No Mathematica is available to you. Symbolic work is by hand, by SymPy or Symbolics
  in a throwaway environment, or by numerical discovery: compute a moment to 200 bits at
  rational edge lengths and identify the rational coefficients of a conjectured basis
  (powers of the lengths, hypotenuses, asinh and atan of length ratios) by solving a
  linear system or by integer-relation detection, then prove the conjecture by
  derivation. The self-face analysis found its double-factorial coefficients exactly
  this way.
- Public search may be used for ordinary mathematical background (the Newtonian
  potential of polyhedra, box potentials, reduction formulas, incomplete cylindrical
  functions), not to look for a solution to this exact problem.
- Write everything under your scratchpad, except the final deliverables, which go in
  `notes/moments/` in the repository: the Julia implementation, the verification
  script, and the report.

## How to run the search

You are a Fable 5.1 agent with limited usage. You are the root: you plan, synthesize,
challenge, redirect, and audit. You do not do the tedious work yourself. Launch
sub-agents with `model: "opus"` for derivations and difficult verification, and
`model: "sonnet"` for mechanical transcription, sweeps, tabulation, and plotting. Ask
every sub-agent for a compact written report with the actual formulas, the actual
numbers, and the paths of its scripts; refuse status reports, optimism, and claims
that something is routine. Never read a sub-agent's transcript; read its report.

Manage the search with the following heuristics.

- Begin with a genuinely diverse portfolio of approach families, and keep a written
  registry of them grouped by mathematical mechanism, not by wording. At least these
  families should be alive in the first round:
  (a) polar and Duffy coordinates with exact radial integration and the sec-power
  reduction, generalizing Section 5.2 to the trapezoidal weights of the other coplanar
  geometries;
  (b) the divergence recursion of Section 5.1 on the four-dimensional product box,
  implemented as an actual symbolic program that lowers dimension and exponent
  together;
  (c) the box-potential route: reduce a perpendicular pair to
  ∫ (weight) · Φ_m(u) du with Φ_m the potential of a rectangle at a point with kernel
  r^m, derive Φ_m by the same means, then integrate by parts along the shared
  direction, as `rSrfEdgCrn` was derived;
  (d) recurrences in m obtained from derivatives with respect to the edge lengths or
  from the identity ∇·(x r^m) = (N + m) r^m, seeded by the known m = −1 and m = 1
  forms;
  (e) numerical discovery: high-precision values at rational geometry, a conjectured
  basis of elementary functions, exact rational coefficients recovered by linear
  algebra, then proof;
  (f) symbolic integration with SymPy or Symbolics on the reduced one- and
  two-dimensional integrals, for fixed m, followed by pattern extraction across m;
  (g) generating functions: sum the moments against t^m to a closed form in t and
  read the moments off, or relate the moment sequence to the special-function form
  of Section 5.3.
  Assign geometries and families so that each hard geometry (Ec, Vp, Xg) is attacked
  by at least two independent families in the first round.

- Do not tell most sub-agents which approach currently looks best. Preserve
  independence in early rounds so they do not all converge on the same attractive
  reduction.

- Do not let one family dominate because it gives elegant reductions. A reduction to
  a one-dimensional integral that no one can do in closed form is not progress on
  that geometry. A recurrence whose seed is unknown is not a solution.

- When an approach stalls on a geometry, mark that route blocked for that geometry
  in the registry and reassign. Reopen it only when a sub-agent proposes a materially
  new mechanism, substitution, or basis. Keep several incompatible routes alive
  through several rounds, and cross-pollinate only after independent sub-agents have
  developed them far enough to expose real strengths and gaps.

- Exploit structure across geometries. The coplanar cases are one family with
  different convolution weights; a formula for a general trapezoidal weight covers
  S, Ef, Vc and the coplanar parts at once. The perpendicular cases share the box
  potential Φ_m; Ec, Vp and Xg differ only in the limits of the outer integral.
  The parallel-offset cases P1, P2, P1s, P2s are the coplanar cases with a constant
  added to r^2, which is where the divergence recursion's c^2 term enters. Sub-agents
  who discover such a unification should say so explicitly.

- Use adversarial sub-agents throughout. Every candidate formula must be checked
  against `pairMom` at the extreme aspect ratios above, not only at cubes; checked for
  homogeneity of degree m + 4 in the lengths; checked for the symmetries the geometry
  has (for example l_b ↔ l_c in Ec) and for the absence of symmetries it does not
  have; checked for the correct m = −1 limit against the existing closed forms;
  checked in BigFloat with all constants typed to the working precision; and checked
  for branch and sign errors of acot, atan, and log at every orientation. A formula
  that matches at one aspect ratio and fails at another is wrong, not approximately
  right.

- Require concrete artifacts: formulas, coefficient tables, scripts, and error tables.
  Reject any report without numbers.

- As root, repeatedly synthesize the registry, decide what is proven and what is
  blocked, launch the next round, and audit. Do not stop after the first wave fails.
  Do not stop when one geometry is done. Do not stop when the coplanar geometries
  are done. Do not stop because the perpendicular geometries look like they need a
  two-variable special function; the moments are elementary by theorem, only the
  full kernel is not, and the theorem's proof is itself an algorithm you can run.

## Persistence

This is a hard problem. It will at times look as if a geometry has no tractable closed
form, as if the algebra is unmanageable, or as if the only honest answer is a
one-dimensional integral. That impression is wrong: Section 5.1 of the document proves
that every one of these moments is elementary, the self face has been done for all m
in a few lines, and the m = −1 cases of Ef and Ec were reduced from thirty-term
expressions to three-term ones by hand. Expect the algebra to be long and the
expressions to collapse at the end, as they did in every case so far. Expect
Mathematica-style intermediate results to be unreadable and numerically useless until
regrouped.

Do not return merely because current approaches fail, because sub-agents report that
an integral "has no closed form", or because a CAS times out. Continue launching new
rounds, reopening blocked approaches only with a genuinely new mechanism, and
searching for fresh formulations. Tell every sub-agent you launch, in its own prompt,
that the result is known to exist, that it must not give up, that it must try a
different route when one stalls, and that it must return concrete partial formulas
and the exact point where it is stuck rather than a summary of difficulty.

Return only when the complete solution defined above exists for all ten geometries and
has survived adversarial audit. Do not return a subset of geometries, a reduction, a
recurrence without seeds, a "best effort" summary, or an explanation of why the problem
is difficult. Spend at least eight hours on this before even thinking of returning. If,
after genuinely exhausting every family in the registry and inventing new ones, you
believe a specific geometry is impossible, the only acceptable form of that claim is a
proof that contradicts Theorem 5.1 of the document, which would itself be a
significant finding; expect instead to find the formula.

## Final deliverable

Only at the very end, when the solution is complete and audited, write a LaTeX
document `notes/moments/moments.tex`, compile it to `notes/moments/moments.pdf` with
`pdflatex`, and check the rendered pages. It must contain, for each geometry: the
geometry with a figure in the style of the attached document (reuse
`notes/gen/figs.tex`), the derivation, the final formula or recurrence with its seeds
in a box, the cancellation-free arrangement, the verification tables, and the
comparison with Gila's numerics. It must also contain the registry of approach
families with what worked and what was blocked and why, so that the next reader
learns from the failures as well as the successes, and an honest section on anything
that remains uncertain. Attribute yourself as the author, in the same style as the
attached document. Do not write this document early and update it; write it once,
at the end, from the audited results.
