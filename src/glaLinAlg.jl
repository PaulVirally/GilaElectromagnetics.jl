using LinearAlgebra
using CUDA

"""
    glaSze(opr::AbstractGlaOpr)

Returns the size of the input/output arrays for an `AbstractGlaOpr` in tensor form.

# Arguments
- `opr::AbstractGlaOpr`: The operator to check.
"""
function glaSze(opr::GlaOprVac)
    if isoverlappingoperator(opr)
        return ((length.(opr.trgMsk)..., 3), (length.(opr.srcMsk)..., 3))
    end
    return ((opr.mem.trgVol.cel..., 3), (opr.mem.srcVol.cel..., 3))
end
glaSze(opr::Union{AsyGlaOprVac, SymGlaOprVac}) = ((opr.mem.trgVol.cel..., 3), (opr.mem.srcVol.cel..., 3))
glaSze(opr::MulRegGlaOprVac) = glaSze.(opr.oprMat)
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
glaSze(opr::MulRegGlaOprVac, dim::Int) = map(x -> x[dim], glaSze(opr))
glaSze(opr::InvSctOpr, dim::Int) = glaSze(opr.oprVac, dim)
glaSze(opr::SctOpr, dim::Int) = glaSze(opr.invSctOpr, dim)
glaSze(opr::GlaOpr, dim::Int) = glaSze(opr.sctOpr, dim)

# Type and size definitions
Base.eltype(::AbstractGlaOpr) = ComplexF64
Base.size(opr::AbstractGlaOpr) = prod.(glaSze(opr))
function Base.size(opr::MulRegGlaOprVac)
    rowSzs = [size(opr.oprMat[i, 1], 1) for i in axes(opr.oprMat, 1)]
    colSzs = [size(opr.oprMat[1, j], 2) for j in axes(opr.oprMat, 2)]
    return (sum(rowSzs), sum(colSzs))
end
Base.size(opr::AbstractGlaOpr, i::Int) = prod(glaSze(opr, i))
Base.size(opr::MulRegGlaOprVac, i::Int) = size(opr)[i]
Base.size(opr::InvSctOpr) = size(opr.oprVac)
Base.size(opr::SctOpr) = size(opr.invSctOpr)
Base.size(opr::GlaOpr) = size(opr.sctOpr)
Base.size(opr::Union{InvSctOpr, SctOpr, GlaOpr}, i::Int) = size(opr)[i]

# Array type definition
Base.similar(opr::AbstractGlaOpr) = arrTyp(opr)(undef, size(opr, 1), size(opr, 2))
function Base.similar(opr::AbstractGlaOpr, ::Type{T}) where T
    AT = arrTyp(opr)
    if AT <: CuArray
        return CuArray{T}(undef, size(opr, 1), size(opr, 2))
    else
        return Array{T}(undef, size(opr, 1), size(opr, 2))
    end
end
function Base.similar(opr::AbstractGlaOpr, dims::Tuple{Vararg{Int}})
    AT = arrTyp(opr)
    if AT <: CuArray
        return CuArray{eltype(opr)}(undef, dims...)
    else
        return Array{eltype(opr)}(undef, dims...)
    end
end
function Base.similar(opr::AbstractGlaOpr, ::Type{T}, dims::Tuple{Vararg{Int}}) where T
    AT = arrTyp(opr)
    if AT <: CuArray
        return CuArray{T}(undef, dims...)
    else
        return Array{T}(undef, dims...)
    end
end

# Indexing functions
Base.IndexStyle(::Type{<:AbstractGlaOpr}) = IndexCartesian()
Base.getindex(opr::AbstractGlaOpr, i::Integer) = getindex(opr, CartesianIndices(opr)[i])
Base.getindex(opr::AbstractGlaOpr, i::CartesianIndex) = getindex(opr, i.I...)
function Base.getindex(opr::AbstractGlaOpr, row::IType, col::IType) where IType <: Union{Integer, AbstractUnitRange{<:Integer}, AbstractVector{<:Integer}, Colon}
    T = eltype(opr)
    numRow, numCol = size(opr)
    row = row === Colon() ? (1:numRow) : row
    col = col === Colon() ? (1:numCol) : col

    if row isa Integer && col isa Integer
        @warn """Scalar indexing is not recommended. Invocation of getindex resulted in scalar
        indexing of an AbstractGlaOpr. This is typically caused by calling an iterating
        implementation of a method. Single element access ($row, $col) does a costly full
        matrix-vector product."""
        e = fill!(arrTyp(opr)(undef, size(opr, 2)), zero(eltype(opr)))
        CUDA.@allowscalar e[col] = one(eltype(opr))
        return CUDA.@allowscalar (opr * e)[row]
    end

    rowInd = row isa Integer ? (row,) : collect(row)
    colInd = col isa Integer ? (col,) : collect(col)

    # Choose the cheapest way to compute the result (adjoint or forward)
    if length(rowInd) <= length(colInd)
        # Forward batched mat–vec
        # build a mini‐identity with 1‑hots in the input slots
        idt = fill!(arrTyp(opr)(undef, numCol, length(colInd)), zero(T))
        for (j, idx) in enumerate(colInd)
            CUDA.@allowscalar idt[idx, j] = one(T)
        end
        out = opr * idt
        return out[rowInd, :]
    end

    # Adjoint batched vec–mat
    # mini‑identity in the output slots
    idt = fill!(arrTyp(opr)(undef, numRow, length(rowInd)), zero(T))
    for (i, idx) in enumerate(rowInd)
        CUDA.@allowscalar idt[idx, i] = one(T)
    end
    adjOpr = adjoint!(opr) # Compute with the adjoint operator
    outDag = adjOpr * idt
    adjoint!(adjOpr) # Restore the memory the caller's operator still points at
    return outDag[colInd, :]'
end
Base.setindex!(::AbstractGlaOpr, _, __...) = throw(ArgumentError("setindex! is not supported for AbstractGlaOpr"))

# Matrix-matrix operation
function Base.:*(opr::AbstractGlaOpr, inp::AbstractMatrix{ComplexF64})
    out = similar(inp, size(opr, 1), size(inp, 2))
    for (outCol, inpCol) in zip(eachcol(out), eachcol(inp))
        outCol .= opr * inpCol
    end
    return out
end

# Generic 5-argument multiplication
LinearAlgebra.mul!(out::AbstractVector{ComplexF64}, opr::AbstractGlaOpr, inp::AbstractVector{ComplexF64}, α::Number, β::Number) = axpby!(α, opr * inp, β, out)
LinearAlgebra.mul!(out::AbstractArray{ComplexF64, 4}, opr::AbstractGlaOpr, inp::AbstractArray{ComplexF64, 4}, α::Number, β::Number) = axpby!(α, opr * inp, β, out)
LinearAlgebra.mul!(out::AbstractMatrix{ComplexF64}, opr::AbstractGlaOpr, inp::AbstractMatrix{ComplexF64}, α::Number, β::Number) = axpby!(α, opr * inp, β, out)

# Matrix-vector operations
function Base.:*(opr::Union{GlaOprVac, AsyGlaOprVac, SymGlaOprVac}, innVec::AbstractArray{ComplexF64, 4})
    if isoverlappingoperator(opr)
        innVecEmb = similar(innVec, opr.mem.srcVol.cel..., 3) # Input array embedded in the input space of the full volume of the overlapping operator
        fill!(innVecEmb, zero(eltype(innVec)))
        innVecEmb[opr.srcMsk..., :] .= innVec # Place the input into the embedded array
        innVec = innVecEmb # Swap out the memory locations
    end
    if isgpu(opr) && !(innVec isa CuArray)
        @warn "Input array is not a CuArray. Copying data to GPU."
        innVec = CuArray(innVec)
    end
    out = egoOpr!(opr.mem, deepcopy(innVec))
    if isoverlappingoperator(opr)
        # Mask out the output
        return out[opr.trgMsk..., :]
    end
    return out
end
function Base.:*(opr::MulRegGlaOprVac, innVec::Vector{<:AbstractArray{ComplexF64, 4}})
    m, n = size(opr.oprMat)
    @assert length(innVec) == n "expected $n source blocks, got $(length(innVec))"
    outVec = [opr.oprMat[i, 1] * innVec[1] for i in 1:m] # j = 1
    for i in 1:m, j in 2:n
        outVec[i] .+= opr.oprMat[i, j] * innVec[j]
    end
    return outVec
end
function Base.:*(opr::Union{GlaOprVac, AsyGlaOprVac, SymGlaOprVac}, innVec::AbstractVector{ComplexF64})
    innVecArr = reshape(innVec, glaSze(opr, 2))
    outVec = opr * innVecArr
    return vec(outVec)
end
function Base.:*(opr::MulRegGlaOprVac, innVec::AbstractVector{ComplexF64})
    rowSzs = [size(opr.oprMat[i, 1], 1) for i in axes(opr.oprMat, 1)]
    colSzs = [size(opr.oprMat[1, j], 2) for j in axes(opr.oprMat, 2)]
    rowOff = cumsum([0; rowSzs]); colOff = cumsum([0; colSzs])
    outVec = fill!(similar(innVec, sum(rowSzs)), zero(eltype(innVec)))
    for i in axes(opr.oprMat, 1)
        outBlk = view(outVec, (rowOff[i]+1):rowOff[i+1])
        for j in axes(opr.oprMat, 2)
            inBlk = view(innVec, (colOff[j]+1):colOff[j+1])
            outBlk .+= opr.oprMat[i, j] * inBlk
        end
    end
    return outVec
end

"""
    *(opr::GlaOprVac, fld::GlaFld)

Apply a vacuum operator to a field over a single region.

The field has to be a field over the source volume of the operator, which for a
`GlaFld` means a tiling of exactly one region equal to that volume. The result is
a field over the target volume, again as a tiling of one region.

A single region has a single cell volume, so the √ΔV normalization of `GlaFld` is
a scalar on each side rather than a diagonal, and it comes out of the operator as
the ratio of the two. That ratio is one whenever the two volumes share a cell
size, which covers every self operator.

# Arguments
- `opr::GlaOprVac`: The operator
- `fld::GlaFld`: The field, which must live on the source volume of `opr`

# Returns
- `GlaFld`: The result, on the target volume of `opr`

# Throws
- `ArgumentError`: If the field is not a field over the source volume of the
  operator, or if the operator takes the masked route
"""
function Base.:*(opr::GlaOprVac, fld::GlaFld)
    srcVol = opr.mem.srcVol
    if isoverlappingoperator(opr)
        throw(ArgumentError("This operator is built on the union of its two volumes and reads its input through a mask, so it does not take a field. Apply it to a plain array of the masked size instead."))
    end
    if nregions(fld.cvol) != 1 || regions(fld.cvol)[1] != srcVol
        throw(ArgumentError("The field does not live on the source volume of the operator, which is a ($(join(srcVol.cel, "×"))) cell volume of ($(join(srcVol.scl, "×")))λ³ cells."))
    end
    outDat = opr * fld.dat
    nrm = sqrt(Float64(prod(opr.mem.trgVol.scl) // prod(srcVol.scl)))
    nrm != 1 && rmul!(outDat, nrm)
    return GlaFld(outDat, GlaCmpVol(opr.mem.trgVol))
end

function Base.:*(opr::InvSctOpr, inp::AbstractArray{ComplexF64, 4})
    # Compute the matrix-vector product (I - XG₀) * inp for inp in 4-tensor form
    if isadjoint(opr)
        return inp - (opr.oprVac * (opr.sus .* inp))
    end
    return inp - (opr.sus .* (opr.oprVac * inp))
end
function Base.:*(opr::InvSctOpr, inp::AbstractVector{ComplexF64})
    # Compute the matrix-vector product (I - XG₀)⁻¹ * inp for inp in vector form
    opr.oprVac isa GlaCmpOprVac && return _cmpSctMul(opr, inp)
    if isadjoint(opr)
        return inp - vec(opr.oprVac * (opr.sus .* reshape(inp, glaSze(opr, 2))))
    end
    return inp - vec(opr.sus .* (opr.oprVac * reshape(inp, glaSze(opr, 2))))
end
Base.:*(opr::SctOpr, inp::AbstractArray{ComplexF64, 4}) = reshape(solve(opr.invSctOpr, vec(inp), opr.slv), size(inp))
Base.:*(opr::SctOpr, inp::AbstractVector{ComplexF64}) = solve(opr.invSctOpr, inp, opr.slv)
function Base.:*(opr::GlaOpr, inp::Union{AbstractVector{ComplexF64}, AbstractArray{ComplexF64, 4}})
    # Compute the matrix-vector product G₀(I - XG₀)⁻¹ * inp
    if isadjoint(opr)
        return opr.sctOpr * (opr.sctOpr.invSctOpr.oprVac * inp)
    end
    return opr.sctOpr.invSctOpr.oprVac * (opr.sctOpr * inp)
end

"""
    adjoint!(opr::AbstractGlaOpr)

Return the adjoint of an operator, reusing its memory.

The call may mutate whatever the argument holds, so the argument must not be used
again: the returned operator is the only valid handle on that memory. A second
call restores the memory, which is what makes `adjoint!(adjoint!(opr))` a valid
operator equal to the original. Code that needs the argument to survive should
call `adjoint` instead, which works on a copy.

Some types rearrange in place and hand the same object back, others return a new
wrapper around the same memory, so the return value always has to be used.

# Arguments
- `opr::AbstractGlaOpr`: The operator to adjoint

# Returns
- The adjoint operator
"""
function adjoint!(opr::GlaOprVac)
    # Mark the adjoint
    opr.mem.cmpInf.adjMod = !opr.mem.cmpInf.adjMod

    # Swap source and target volumes (transpose)
    opr.mem.trgVol, opr.mem.srcVol = opr.mem.srcVol, opr.mem.trgVol
    opr.mem.mixInf = GlaExtInf(opr.mem.trgVol, opr.mem.srcVol)

    # Take the conjugate of the Fourier coefficients (conjugate transpose)
    # Also swap the last two axes because they hold the source and target
    # partition of a cross-scale pair (which must be transposed for the adjoint)
    opr.mem.egoFur = collect(map(arr -> conj.(permutedims(arr, (1, 2, 3, 4, 6, 5))), opr.mem.egoFur))

    # The masks are immutable fields, so the transpose needs a new wrapper
    return GlaOprVac(opr.mem, opr.trgMsk, opr.srcMsk)
end
adjoint!(opr::Union{AsyGlaOprVac, SymGlaOprVac}) = opr # These operators are Hermitian (self-adjoint)
function adjoint!(opr::MulRegGlaOprVac)
    adjMat = similar(opr.oprMat, reverse(size(opr.oprMat))) # New operator matrix with adjoint size
    for i in axes(opr.oprMat, 1)
        for j in axes(opr.oprMat, 2)
            adjMat[j, i] = adjoint!(opr.oprMat[i, j])
        end
    end
    # note that now, opr's entries are all adjoint's or the original entries
    return MulRegGlaOprVac(adjMat)
end
function adjoint!(opr::InvSctOpr)
    opr.oprVac = adjoint!(opr.oprVac)
    opr.sus = conj(opr.sus)  # Conjugate the susceptibility
    return opr
end
function adjoint!(opr::SctOpr)
    opr.invSctOpr = adjoint!(opr.invSctOpr)
    return opr
end
function adjoint!(opr::GlaOpr)
    opr.sctOpr = adjoint!(opr.sctOpr)
    return opr
end
Base.adjoint(opr::AbstractGlaOpr) = adjoint!(deepcopy(opr))

include("glaMatFreExtOps.jl")
