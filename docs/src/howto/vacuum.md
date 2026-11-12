# Vacuum operators

## Field at the cell centres and its derivatives

A `GlaOprVac` returns the field averaged over each target cell, the coefficient the solvers work with. With `ctr = true` it returns the field at each target cell centre instead, and with `der = n ≥ 1` the derivatives of order `n` of the field there:

```julia
vol = GlaVol((16, 16, 16), (1//32, 1//32, 1//32))
jac = GlaOprVac(vol; der = 1)  # ctr = true is implied
dE = jac * p                   # p is (16, 16, 16, 3), dE is (16, 16, 16, 3, 3)
```

`dE[i, j, k, a, g]` is the derivative ∂^γ E_a at the centre of cell `(i, j, k)`, for the `g`-th of the (n+1)(n+2)/2 orders γ with |γ| = n. They run (n, 0, 0), (n-1, 1, 0), (n-1, 0, 1), (n-2, 2, 0) and so on down to (0, 0, n): ∂x, ∂y, ∂z for n = 1, and ∂xx, ∂xy, ∂xz, ∂yy, ∂yz, ∂zz for n = 2.

`der ≥ 1` needs `ctr = true` because a derivative has no cell average. The field of a polarized cell jumps across the cell's faces, so its derivative holds a delta function on each face, and a target cell that touches the source shares one of those faces.

Orders 1 and 2 are certified. Higher orders run but are not certified, though on the offsets tested the kernels came within half an ulp in Float64 up to n = 4.

An operator with `der ≥ 1` maps 3 values per source cell to 3(n+1)(n+2)/2 per target cell, so it is not square. It has no `\`, solver, scattering operator, adjoint or Hermitian part, and each of these throws. Centre values work on a self volume and on two volumes with equal cells, without `qssApx` or `blc`.

### Derivatives of the total field

An operator with `der = n` differentiates the field radiated by the currents you give it, and nothing else. The total field is the driving field plus the field radiated by the scattered current, ``\vb*{f}_\text{t} = \vb*{f}_\text{d} + \bm{G}_{\circ}\vb*{p}_\text{s}``, so its derivatives at the cell centres are

```math
\partial^\gamma \vb*{f}_\text{t} = \partial^\gamma \vb*{f}_\text{d} + \bm{G}_{\circ}^{(n)}\vb*{p}_\text{s},
\qquad
\vb*{p}_\text{s} = \bm{W}\vb*{p}_\text{d} = \ab(\bbone - \bm{X}\bm{G}_{\circ})^{-1}\bm{X}\vb*{f}_\text{d},
```

where ``\bm{G}_{\circ}^{(n)}`` is the operator built with `der = n`. Solve for ``\vb*{p}_\text{s}`` with the ordinary cell-averaged operator, then apply ``\bm{G}_{\circ}^{(n)}`` to it. Gila does not know the driving field, so ``\partial^\gamma \vb*{f}_\text{d}`` is yours to add, at the same cell centres and in the same order of γ:

```julia
ps = SctOpr(GlaOprVac{Float64}(vol), χ) * (χ .* fd)  # fd is the driving field, (16, 16, 16, 3)
d2ft = d2fd .+ GlaOprVac{Float64}(vol; der = 2) * ps  # d2fd is ∂^γ fd at the centres, (16, 16, 16, 3, 6)
```
