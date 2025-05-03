using LinearAlgebra
using CUDA

Base.eltype(::AbstractGlaOpr) = ComplexF64
Base.size(opr::AbstractGlaOpr) = prod.(glaSze(opr))
Base.size(opr::AbstractGlaOpr, i::Int) = prod(glaSze(opr, i))

function Base.:*(opr::AbstractGlaOpr, inp::AbstractMatrix{ComplexF64})
    out = similar(inp, size(opr, 1), size(inp, 2))
    for i in 1:size(inp, 2)
        out[:, i] .= opr * inp[:, i]
    end
    return out
end

LinearAlgebra.mul!(out::AbstractVector{ComplexF64}, opr::AbstractGlaOpr, inp::AbstractVector{ComplexF64}, α::T, β::T) where T = axpby!(α, opr * inp, β, out)
LinearAlgebra.mul!(out::AbstractArray{ComplexF64, 4}, opr::AbstractGlaOpr, inp::AbstractArray{ComplexF64, 4}, α::T, β::T) where T = axpby!(α, opr * inp, β, out)
LinearAlgebra.mul!(out::AbstractMatrix{ComplexF64}, opr::AbstractGlaOpr, inp::AbstractMatrix{ComplexF64}, α::T, β::T) where T = axpby!(α, opr * inp, β, out)

function Base.:*(opr::GlaOprVac, innVec::AbstractArray{ComplexF64, 4})
    if bckEnd(opr.mem.cmpInf) isa GPUKerOpt && !(innVec isa CuArray)
        @warn "Input array is not a CuArray. Copying data to GPU."
        innVec = CuArray(innVec)
    end
    return egoOpr!(opr.mem, deepcopy(innVec))
end
function Base.:*(opr::GlaOprVac, innVec::AbstractArray{ComplexF64})
    innVecArr = reshape(innVec, glaSze(opr, 2))
    outVec = opr * innVecArr
    if prod(size(innVec)) == prod(glaSze(opr, 1))
        return reshape(outVec, size(innVec))
    elseif ndims(innVec) == 1
        return vec(outVec)
    end
    return reshape(outVec, glaSze(opr, 1))
end

function Base.:*(opr::InvSctOpr, inp::AbstractArray{ComplexF64, 4})
    # Compute the matrix-vector product (I - XG₀) * inp for inp in 4-tensor form
    if isadjoint(opr)
        return inp - (opr.invSctOpr.oprVac * (opr.invSctOpr.sus .* inp))
    end
    return inp - (opr.invSctOpr.sus .* (opr.invSctOpr.oprVac * inp))
end
function Base.:*(opr::InvSctOpr, inp::AbstractVector{ComplexF64})
    # Compute the matrix-vector product (I - XG₀)⁻¹ * inp for inp in vector form
    if isadjoint(opr)
        return inp - vec(opr.oprVac * (opr.sus .* reshape(inp, glaSze(opr, 2))))
    end
    return inp - vec(opr.sus .* (opr.oprVac * reshape(inp, glaSze(opr, 2))))
end

Base.:*(opr::SctOpr, inp::AbstractArray{ComplexF64, 4}) = reshape(solve(opr.invSctOpr, vec(inp), opr.slv), size(inp))
Base.:*(opr::SctOpr, inp::AbstractVector{ComplexF64}) = solve(opr.invSctOpr, inp, opr.slv)

function Base.:*(opr::GlaOpr, inp::AbstractArray{ComplexF64})
    # Compute the matrix-vector product G₀(I - XG₀)⁻¹ * inp
    if isadjoint(opr)
        return opr.sctOpr * (opr.sctOpr.invSctOpr.oprVac * inp)
    end
    return opr.sctOpr.invSctOpr.oprVac * (opr.sctOpr * inp)
end
# function Base.:*(opr::GlaOpr, inp::AbstractVector{ComplexF64})
#     # Compute the matrix-vector product G₀(I - XG₀)⁻¹ * inp for inp in vector form
#     if isadjoint(opr)
#         return opr.sctOpr * (opr.sctOpr.invSctOpr.oprVac * inp)
#     end
#     return opr.sctOpr.invSctOpr.oprVac * (opr.sctOpr * inp)
# end

function adjoint!(opr::GlaOprVac)
    # Mark the adjoint
    opr.mem.cmpInf.adjMod = !opr.mem.cmpInf.adjMod

    # Swap source and target volumes (transpose)
    opr.mem.trgVol, opr.mem.srcVol = opr.mem.srcVol, opr.mem.trgVol
    opr.mem.mixInf = GlaExtInf(opr.mem.trgVol, opr.mem.srcVol)

    # Take the conjugate of the Fourier coefficients (conjugate transpose)
    opr.mem.egoFur = collect(map(arr -> conj.(arr), opr.mem.egoFur))
    return opr
end
Base.adjoint(opr::GlaOprVac) = adjoint!(deepcopy(opr))

function adjoint!(opr::InvSctOpr)
    opr.oprVac = adjoint!(opr.oprVac)
    opr.sus = conj(opr.sus)  # Conjugate the susceptibility
    return opr
end
Base.adjoint(opr::InvSctOpr) = adjoint!(deepcopy(opr))

function adjoint!(opr::SctOpr)
    opr.invSctOpr = adjoint!(opr.invSctOpr)
    return opr
end
Base.adjoint(opr::SctOpr) = adjoint!(deepcopy(opr))

function adjoint!(opr::GlaOpr)
    opr.sctOpr = adjoint!(opr.sctOpr)
    return opr
end
Base.adjoint(opr::GlaOpr) = adjoint!(deepcopy(opr))

"""
    isadjoint(opr::GlaOprVac)

Checks if the operator is the adjoint of the Green's operator.

# Arguments
- `opr::GlaOprVac`: The operator to check.

# Returns
- `true` if the operator is the adjoint, `false` otherwise.
"""
isadjoint(opr::GlaOprVac) = opr.mem.cmpInf.adjMod

"""
    isadjoint(opr::InvSctOpr)

Checks if the inverse scattering operator is the adjoint of its original form.

# Arguments
- `opr::InvSctOpr`: The operator to check.

# Returns
- `true` if the operator is the adjoint, `false` otherwise.
"""
isadjoint(opr::InvSctOpr) = isadjoint(opr.oprVac)

"""
    isadjoint(opr::SctOpr)

Checks if the scattering operator is the adjoint of its original form.

# Arguments
- `opr::SctOpr`: The operator to check.

# Returns
- `true` if the operator is the adjoint, `false` otherwise.
"""
isadjoint(opr::SctOpr) = isadjoint(opr.invSctOpr)

"""
    isadjoint(opr::GlaOpr)

Checks if the full Green's function operator is the adjoint of its original form.

# Arguments
- `opr::GlaOpr`: The operator to check.

# Returns
- `true` if the operator is the adjoint, `false` otherwise.
"""
isadjoint(opr::GlaOpr) = isadjoint(opr.sctOpr)

"""
    isselfoperator(opr::GlaOprVac)

Checks if the operator is a self Green's operator.

# Arguments
- `opr::GlaOprVac`: The operator to check.

# Returns
- `true` if the operator is a self Green's operator, `false` otherwise.
"""
isselfoperator(opr::GlaOprVac) = opr.mem.srcVol == opr.mem.trgVol

"""
    isexternaloperator(opr::GlaOprVac)

Checks if the operator is an external Green's operator.

# Arguments
- `opr::GlaOprVac`: The operator to check.

# Returns
- `true` if the operator is an external Green's operator, `false` otherwise.
"""
isexternaloperator(opr::GlaOprVac) = !isselfoperator(opr)

"""
    glaSze(opr::AbstractGlaOpr)

Returns the size of the input/output arrays for an `AbstractGlaOpr` in tensor form.

# Arguments
- `opr::AbstractGlaOpr`: The operator to check.
"""
glaSze(opr::GlaOprVac) = ((opr.mem.trgVol.cel..., 3), (opr.mem.srcVol.cel..., 3))
glaSze(opr::InvSctOpr) = glaSze(opr.oprVac)
glaSze(opr::SctOpr) = glaSze(opr.invSctOpr)
glaSze(opr::GlaOpr) = glaSze(opr.sctOpr)

"""
    glaSze(opr::AbstractGlaOpr, dim::Int)

Returns the size of the input/output arrays for an `AbstractGlaOpr` in tensor form in a specified dimension.

# Arguments
- `opr::AbstractGlaOpr`: The operator to check.
- `dim::Int`: The index of the dimension to check.
"""
glaSze(opr::AbstractGlaOpr, dim::Int) = glaSze(opr)[dim]

function Base.show(io::IO, opr::GlaOprVac)
    if isadjoint(opr)
        print(io, "Adjoint ")
    end
    if isselfoperator(opr)
        print(io, "Self ")
    else
        print(io, "External ")
    end
    print(io, "G₀ for ")
    if isselfoperator(opr)
        print(io, "a $(eltype(opr)) (" * join(opr.mem.srcVol.cel, "×") * ") volume ")
        print(io, "of size (" * join(opr.mem.srcVol.scl, "×") * ")λ")
    else
        print(io, "$(eltype(opr)) (" * join(opr.mem.srcVol.cel, "×") * ") -> (" * join(opr.mem.trgVol.cel, "×") * ") volumes ")
        print(io, "of sizes (" * join(opr.mem.srcVol.scl, "×") * ")λ -> (" * join(opr.mem.trgVol.scl, "×") * ")λ ")
        print(io, "with separation (" * join(opr.mem.trgVol.org .- opr.mem.srcVol.org, ", ") * ")λ")
    end
end
