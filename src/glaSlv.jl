"""
    GilaSolvers

This module provides iterative solvers for linear systems of equations in the Gila package.
It includes implementations of GMRES and BiCGStab methods, with support for both CPU and GPU computations.

# Types
- `GlaSlv`: Abstract base type for all solvers
- `GMRESSolver`: Generalized Minimal Residual Method solver
- `BiCGStabSolver`: BiConjugate Gradient Stabilized Method solver
- `MixPrcRfn`: Mixed precision iterative refinement, with a pluggable inner solver

# Functions
- `solve`: Solve a linear system using the specified solver
"""
module GilaSolvers

using LinearAlgebra
using CUDA
using ..GilaTypes
using ..GilaTypes: isgpu, isadjoint

export GMRESSolver, BiCGStabSolver, MixPrcRfn, solve
export MixedPrecisionRefinement

"""
    GMRESSolver

An iterative solver for linear systems of equations that uses the Generalized Minimal
Residual Method (GMRES). This method is particularly effective for non-symmetric
linear systems.

# Fields
- `rstItr::Union{Nothing, Int}`: Number of iterations until restart (default: min(20, length(vec)))
- `maxItr::Union{Nothing, Int}`: Maximum number of iterations (default: max(5000, length(vec)))
- `absTol::Union{Nothing, Real}`: Absolute tolerance for convergence (default: 0)
- `relTol::Union{Nothing, Real}`: Relative tolerance for convergence (default: √ε)
- `preCon`: Left preconditioner, anything with an `ldiv!` method (default: none)
"""
struct GMRESSolver <: GlaSlv
    rstItr::Union{Nothing, Int} # Iterations until restart
    maxItr::Union{Nothing, Int} # Maximum number of iterations
    absTol::Union{Nothing, Real} # Absolute tolerance
    relTol::Union{Nothing, Real} # Relative tolerance
    preCon::Any # Left preconditioner
end

"""
    GMRESSolver(rstItr = nothing, maxItr = nothing, absTol = nothing, relTol = nothing; preCon = nothing)

Create a GMRESSolver. Unset values are resolved against the right hand side at
solve time.

# Returns
- `GMRESSolver`: A new solver instance
"""
GMRESSolver(rstItr = nothing, maxItr = nothing, absTol = nothing, relTol = nothing; preCon = nothing) =
    GMRESSolver(rstItr, maxItr, absTol, relTol, preCon)

# Solver settings with the unset ones filled in from the right hand side
slvPrm(slv::GMRESSolver, vec::AbstractArray) = (
    rstItr = @something(slv.rstItr, min(20, length(vec))),
    maxItr = @something(slv.maxItr, max(5000, length(vec))),
    absTol = @something(slv.absTol, zero(real(eltype(vec)))),
    relTol = @something(slv.relTol, sqrt(eps(real(eltype(vec))))))

"""
    BiCGStabSolver

An iterative solver for linear systems of equations that uses the BiConjugate
Gradient Stabilized Method (BiCGStab). This method is effective for non-symmetric
linear systems and typically requires less memory than GMRES.

# Fields
- `maxItr::Union{Nothing, Int}`: Maximum number of iterations (default: length(vec))
- `absTol::Union{Nothing, Real}`: Absolute tolerance for convergence (default: 0)
- `relTol::Union{Nothing, Real}`: Relative tolerance for convergence (default: √ε)
- `preCon`: Left preconditioner, anything with an `ldiv!` method (default: none)
"""
struct BiCGStabSolver <: GlaSlv
    maxItr::Union{Nothing, Int} # Maximum number of iterations
    absTol::Union{Nothing, Real} # Absolute tolerance
    relTol::Union{Nothing, Real} # Relative tolerance
    preCon::Any # Left preconditioner
end

"""
    BiCGStabSolver(maxItr = nothing, absTol = nothing, relTol = nothing; preCon = nothing)

Create a BiCGStabSolver. Unset values are resolved against the right hand side
at solve time.

# Returns
- `BiCGStabSolver`: A new solver instance
"""
BiCGStabSolver(maxItr = nothing, absTol = nothing, relTol = nothing; preCon = nothing) =
    BiCGStabSolver(maxItr, absTol, relTol, preCon)

# Solver settings with the unset ones filled in from the right hand side
slvPrm(slv::BiCGStabSolver, vec::AbstractVector) = (
    maxItr = @something(slv.maxItr, length(vec)),
    absTol = @something(slv.absTol, zero(real(eltype(vec)))),
    relTol = @something(slv.relTol, sqrt(eps(real(eltype(vec))))))

"""
    solve(opr::AbstractGlaOpr, inp::AbstractVector{T}, slv::BiCGStabSolver) where T

Solve the linear system `opr * out = inp` using the BiCGStab method.

# Arguments
- `opr::AbstractGlaOpr`: The operator in the linear system
- `inp::AbstractVector{T}`: The right-hand side vector
- `slv::BiCGStabSolver`: The solver parameters

# Returns
- `out::AbstractVector{T}`: The solution vector

# Notes
- The solver will iterate until either convergence is achieved or the maximum
  number of iterations is reached
- Convergence is determined by the absolute and relative tolerances specified
  in the solver parameters
"""
function solve(opr::AbstractGlaOpr, inp::AbstractVector{T}, slv::BiCGStabSolver) where T
    (; maxItr, absTol, relTol) = slvPrm(slv, inp)
    preCon = slv.preCon
    out = fill!(similar(inp), zero(T))

    ρPrv = zero(T)
    ω = zero(T)
    α = zero(T)
    # Work buffers, allocated once (zeroed so a β = 0 `mul!` never reads garbage)
    v = fill!(similar(inp), zero(T))
    t = fill!(similar(inp), zero(T))
    res = copyto!(similar(inp), inp) # Residual
    absTol = max(absTol, relTol * norm(res))

    resShd = copy(res) # Residual shadow
    p = copy(res)
    s = similar(res)
    # ldiv! must not clobber p or s, both of which are read again below
    p̂ = isnothing(preCon) ? p : similar(p)
    ŝ = isnothing(preCon) ? s : similar(s)

    for numItr in 1:maxItr
        if norm(res) < absTol
            return out
        end

        ρ = dot(resShd, res)
        if numItr > 1
            β = (ρ / ρPrv) * (α / ω)
            p .= res .+ β .* (p .- ω .* v)
        end
        isnothing(preCon) || ldiv!(p̂, preCon, p)
        mul!(v, opr, p̂)
        α = ρ / dot(resShd, v)
        res .-= α .* v
        s .= res

        if norm(res) < absTol
            out .+= α .* p̂
            return out
        end

        isnothing(preCon) || ldiv!(ŝ, preCon, s)
        mul!(t, opr, ŝ)
        ω = dot(t, s) / dot(t, t)
        # ω = dot(t, res) / dot(t, t)
        out .+= α .* p̂ .+ ω .* ŝ
        res .-= ω .* t
        ρPrv = ρ
    end
    @warn "BiCGStab did not converge after $maxItr iterations."
    return out
end

"""
    solve(opr::AbstractGlaOpr, inp::AbstractArray{T}, slv::GMRESSolver) where T

Solve the linear system `opr * out = inp` using the GMRES method.

# Arguments
- `opr::AbstractGlaOpr`: The operator in the linear system
- `inp::AbstractArray{T}`: The right-hand side vector
- `slv::GMRESSolver`: The solver parameters

# Returns
- `out::AbstractArray{T}`: The solution vector

# Notes
- The solver will iterate until either convergence is achieved or the maximum
  number of iterations is reached
- Convergence is determined by the absolute and relative tolerances specified
  in the solver parameters
- The solver uses restarts to manage memory usage, with the restart frequency
  specified in the solver parameters
"""
function solve(opr::AbstractGlaOpr, inp::AbstractArray{T}, slv::GMRESSolver) where T
    # Algorithm adapted from https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl/blob/0b2f1c5d352069df1bc891750087deda2d14cc9d/src/gmres.jl

    (; rstItr, maxItr, absTol, relTol) = slvPrm(slv, inp)
    preCon = slv.preCon

    basVec = similar(inp, size(opr, 1), 1 + rstItr) # Krylov basis vectors
    hss = zeros(eltype(inp), 1 + rstItr, rstItr) # Hessenberg matrix (always stored on the CPU)
    nllSpc = ones(eltype(inp), 1 + rstItr) # Vector in the nullspace of the Hessenberg matrix (always stored on the CPU)
    out = fill!(similar(inp), zero(T)) # Solution vector
    buf = fill!(similar(out), zero(T)) # Work buffer for the restart residual

    # The first basis vector is b (preconditioned)
    k = 1 # Which restart iteration we are on
    vk = @view basVec[:, k]
    isnothing(preCon) ? copyto!(vk, inp) : ldiv!(vk, preCon, inp) # b - opr * x with x = 0
    β = norm(vk)
    rmul!(vk, inv(β)) # Normalize vk

    # Save the residual
    res = β # Residual
    resAcc = one(real(T)) # Residual accumulator

    absTol = max(absTol, relTol * res)

    for itr in 1:maxItr
        if res <= absTol
            break
        end

        # vₖ₊₁ = precon \ (opr * vₖ)
        vkp1 = @view basVec[:, k + 1]
        mul!(vkp1, opr, vk)
        isnothing(preCon) || ldiv!(preCon, vkp1)

        # Orthogonalize vₖ₊₁ against previous basis vectors
        for j in 1:k
            col = @view basVec[:, j]
            hss[j, k] = dot(col, vkp1)
            axpy!(-hss[j, k], col, vkp1) # vkp1 .-= hessenberg[j, k] .* col
        end

        # Normalize vₖ₊₁
        hss[k + 1, k] = norm(vkp1)
        rmul!(vkp1, inv(hss[k + 1, k]))

        # Compute the residual
        if iszero(hss[k + 1, k])
            res = zero(T)
        end
        nllSpc[k + 1] = -conj(dot(view(nllSpc, 1:k), view(hss, 1:k, k)) / hss[k + 1, k]) # update the nullspace vector
        resAcc += real(abs2(nllSpc[k+1]))
        res = β / sqrt(resAcc)

        # Next iteration
        k += 1
        vk = vkp1

        # At the end of restarts (and/or when we have converged/maxed out the iterations), update our solution vector
        don = (res <= absTol || itr == maxItr) # done or not
        if don || k == 1 + rstItr
            # Solve the least squares problem: Hy = βe₁
            y = fill!(similar(hss, (k,)), zero(T))
            y[1] = β
            lstSqrHss(view(hss, 1:k, 1:k-1), y) # Note: this mutates `hss`, but it's fine because we don't need it anymore
            if out isa CuArray
                y = CuArray(y) # Copy to GPU if needed
            end

            # Update the solution vector
            mul!(out, view(basVec, :, 1:k-1), view(y, 1:k-1), one(T), one(T)) # x .+= basis_vectors[:, 1:k-1] * y

            k = 1

            # If we have not reached max_iter or converged, restart
            if !don
                vk = @view basVec[:, k]
                mul!(buf, opr, out)
                vk .= vec(inp) .- vec(buf)
                isnothing(preCon) || ldiv!(preCon, vk)
                β = norm(vk)
                rmul!(vk, inv(β))
                resAcc = one(real(T))
            end
        end
        if itr == maxItr
            @warn "GMRES failed to converge after $maxItr iterations"
        end
    end
    return out
end

"""
    MixPrcRfn{Tlo}

Mixed precision iterative refinement (GMRES-IR): the residual and the solution
update are formed in the precision of the operator, while the correction to the
solution is solved for on a `Tlo` copy of that operator. One high precision
matrix-vector product per outer step buys a high precision backward error at the
matrix-vector cost of `Tlo`.

The operator passed to `solve` must be of higher precision than `Tlo`, and must
match the precision of the right hand side.

Typical usage: `GlaOpr{Float64}(vol, vol, sus; slv=MixPrcRfn(Float32))`

# Fields
- `innSlv::GlaSlv`: The solver used for the low precision corrections
- `cch::Base.RefValue{Any}`: The `(opr, oprLo)` pair of the last solve, rebuilt when the operator changes
- `maxItr::Union{Nothing, Int}`: Maximum number of outer refinement steps (default: 20)
- `absTol::Union{Nothing, Real}`: Absolute tolerance for convergence (default: 0)
- `relTol::Union{Nothing, Real}`: Relative tolerance for convergence (default: √ε)
"""
struct MixPrcRfn{Tlo<:AbstractFloat} <: GlaSlv
    innSlv::GlaSlv # Inner solver for the low precision corrections
    cch::Base.RefValue{Any} # (opr, Tlo copy of opr) of the last solve
    maxItr::Union{Nothing, Int} # Maximum number of outer refinement steps
    absTol::Union{Nothing, Real} # Absolute tolerance
    relTol::Union{Nothing, Real} # Relative tolerance
end

const MixedPrecisionRefinement = MixPrcRfn

"""
    MixPrcRfn(::Type{Tlo}; innSlv::GlaSlv=GMRESSolver(), maxItr=nothing, absTol=nothing, relTol=nothing)

Create a `MixPrcRfn` refining in the precision `Tlo`. Unset tolerances are
resolved against the right hand side at solve time.

# Arguments
- `Tlo::Type{<:AbstractFloat}`: The precision of the correction solves
- `innSlv::GlaSlv=GMRESSolver()`: The solver for the correction solves
- `maxItr`, `absTol`, `relTol`: Outer loop settings, resolved at solve time when left unset

# Returns
- `MixPrcRfn{Tlo}`: A new solver instance holding no operator copy yet
"""
MixPrcRfn(::Type{Tlo}; innSlv::GlaSlv=GMRESSolver(), maxItr=nothing, absTol=nothing,
    relTol=nothing) where Tlo<:AbstractFloat =
    MixPrcRfn{Tlo}(innSlv, Ref{Any}((nothing, nothing)), maxItr, absTol, relTol)

# Solver settings with the unset ones filled in from the right hand side
slvPrm(slv::MixPrcRfn, vec::AbstractVector) = (
    maxItr = @something(slv.maxItr, 20),
    absTol = @something(slv.absTol, zero(real(eltype(vec)))),
    relTol = @something(slv.relTol, sqrt(eps(real(eltype(vec))))))

"""
    solve(opr::AbstractGlaOpr{Thi}, inp::AbstractVector{Complex{Thi}}, slv::MixPrcRfn{Tlo}) where {Thi, Tlo}

Solve the linear system `opr * out = inp` by iterative refinement, with
corrections computed in the precision `Tlo`.

# Arguments
- `opr::AbstractGlaOpr{Thi}`: The operator in the linear system, of precision `Thi`
- `inp::AbstractVector{Complex{Thi}}`: The right hand side vector
- `slv::MixPrcRfn{Tlo}`: The solver parameters

# Returns
- `out::AbstractVector{Complex{Thi}}`: The solution vector

# Notes
- The `Tlo` copy of the operator is cached and reused across solves with the
  same operator, and rebuilt as soon as a different operator is passed
- Refinement in the precision of the operator is pointless, and throws
"""
function solve(opr::AbstractGlaOpr{Thi}, inp::AbstractVector{Complex{Thi}},
    slv::MixPrcRfn{Tlo}) where {Thi<:AbstractFloat, Tlo<:AbstractFloat}
    if Tlo == Thi
        throw(ArgumentError("Iterative refinement in the precision it refines from does nothing: both the correction solves and the right hand side are $Thi. Either drop MixPrcRfn or raise the precision of the operator and the right hand side."))
    end
    (; maxItr, absTol, relTol) = slvPrm(slv, inp)
    if first(slv.cch[]) !== opr || isgpu(last(slv.cch[])) != isgpu(opr)
        slv.cch[] = (opr, Base.typename(typeof(opr)).wrapper{Tlo}(opr))
    end
    oprLo = last(slv.cch[])
    #= The cached copy has to track every in place change to the operator it was
    built from: adjoint! flips the adjoint mode, setSus! replaces the
    susceptibility, and both leave the copy behind. =#
    isadjoint(oprLo) != isadjoint(opr) && (oprLo = adjoint!(oprLo))
    hasproperty(opr, :sus) && (oprLo.sus = Base.typename(typeof(opr.sus)).wrapper{Tlo}(opr.sus))
    slv.cch[] = (opr, oprLo)

    out = zero(inp) # Solution vector
    res = copy(inp) # Residual
    buf = similar(inp) # Work buffer for the high precision matrix-vector product
    absTol = max(absTol, relTol * norm(inp))

    for _ in 1:maxItr
        nrmRes = norm(res)
        if nrmRes <= absTol
            return out
        end
        # Normalized before narrowing, so the inner right hand side is O(1)
        dirLo = solve(oprLo, Complex{Tlo}.(res ./ nrmRes), slv.innSlv)
        out .+= nrmRes .* dirLo
        mul!(buf, opr, out)
        res .= inp .- buf
    end
    if norm(res) > absTol
        @warn "Iterative refinement did not converge after $maxItr iterations."
    end
    return out
end

"""
    lstSqrHss(hss::AbstractMatrix{T}, inp::AbstractVector{T}) where T

Solve the least squares problem for the Hessenberg matrix in GMRES using
Givens rotations.

# Arguments
- `hss::AbstractMatrix{T}`: The Hessenberg matrix
- `inp::AbstractVector{T}`: The right-hand side vector

# Returns
- `inp::AbstractVector{T}`: The solution vector (mutated in-place)

# Notes
- This function uses Givens rotations to transform the Hessenberg matrix into
  upper triangular form
- The solution is computed by back substitution
"""
function lstSqrHss(hss::AbstractMatrix{T}, inp::AbstractVector{T}) where T
    # Yoinked from https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl/blob/0b2f1c5d352069df1bc891750087deda2d14cc9d/src/hessenberg.jl

    wdt = size(hss, 2) # width

    for i in 1:wdt
        c, s, _ = LinearAlgebra.givensAlgorithm(hss[i, i], hss[i + 1, i])
        hss[i, i] = c * hss[i, i] + s * hss[i + 1, i]
        for j in i+1:wdt
            temp = -conj(s) * hss[i, j] + c * hss[i + 1, j]
            hss[i, j] = c * hss[i, j] + s * hss[i + 1, j]
            hss[i + 1, j] = temp
        end
        tmp = -conj(s) * inp[i] + c * inp[i + 1]
        inp[i] = c * inp[i] + s * inp[i + 1]
        inp[i + 1] = tmp
    end

    U = UpperTriangular(view(hss, 1:wdt, 1:wdt))
    ldiv!(U, view(inp, 1:wdt))

    return inp
end

end # module
