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
glaSze(opr::AsyGlaOprVac) = ((opr.mem.trgVol.cel..., 3), (opr.mem.srcVol.cel..., 3))
glaSze(opr::SymGlaOprVac) = ((opr.mem.trgVol.cel..., 3), (opr.mem.srcVol.cel..., 3))
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
    adjoint!(opr) # Compute with the adjoint operator
    outDag = opr * idt
    adjoint!(opr) # Restore the original operator
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
function Base.:*(opr::GlaOpr, inp::Union{AbstractVector{ComplexF64}, AbstractArray{ComplexF64, 4}})
    # Compute the matrix-vector product G₀(I - XG₀)⁻¹ * inp
    if isadjoint(opr)
        return opr.sctOpr * (opr.sctOpr.invSctOpr.oprVac * inp)
    end
    return opr.sctOpr.invSctOpr.oprVac * (opr.sctOpr * inp)
end
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
