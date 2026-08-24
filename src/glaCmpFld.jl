"""
    GilaFields

This module provides the field type that lives on a composite volume.

# Types
- `GlaCmpFld`: A vector of current density coefficients over a `GlaCmpVol`

# Type Aliases
- `MultiScaleField`: Alias for `GlaCmpFld`

# Functions
- `zerofield`: Allocate a zero field over a composite volume
- `discretize!`: Fill a field from a function of position
- `regionview`: View the block of one region as a 4-tensor
- `eachregion`: Iterate over the region views of a field
- `regrid`: Read a field off as densities on a uniform grid
"""
module GilaFields

using ..GilaVolumes
using LinearAlgebra
using CUDA

import ..GilaVolumes: _lwrEdg, _uprEdg

export GlaCmpFld, MultiScaleField, zerofield, discretize!, regionview, eachregion, regrid

# Start of each region block in the flat buffer, plus the total length at the end
function _dofOff(cvol::GlaCmpVol)
    off = zeros(Int, nregions(cvol) + 1)
    for (idx, reg) in enumerate(regions(cvol))
        off[idx + 1] = off[idx] + 3 * prod(reg.cel)
    end
    return off
end

"""
    GlaCmpFld

A field (in the physics sense, i.e., a vector at each point in space) over a composite volume, stored as a flat buffer.

The buffer holds √ΔV times the current density coefficient of each degree of
freedom, with ΔV the cell volume of the region the degree of freedom belongs to.
In this basis the Euclidean inner product of two fields is their physical L²
pairing, so `dot` and `norm` can be used as-is. `discretize!` and `regrid` convert
between densities and stored coefficients. On a uniform mesh the √ΔV factor is
only one global scalar and the basis is the usual one (up to that scalar).

The layout inside the buffer is the flat degree of freedom layout of
`GlaCmpVol`: region blocks in `regions` order, and inside a block the `vec` of
an array of size `(reg.cel..., 3)`.

# Fields
- `dat::AbstractVector{ComplexF64}`: The flat buffer, a `Vector` on the CPU or a
  `CuVector` on the GPU
- `cvol::GlaCmpVol`: The composite volume the field lives on
- `off::Vector{Int}`: Buffer offset of each region block, with the total length
  as a last entry
"""
struct GlaCmpFld{T<:AbstractVector{ComplexF64}} <: AbstractVector{ComplexF64}
    dat::T
    cvol::GlaCmpVol
    off::Vector{Int}

    function GlaCmpFld(dat::T, cvol::GlaCmpVol) where T<:AbstractVector{ComplexF64}
        off = _dofOff(cvol)
        if length(dat) != off[end]
            throw(ArgumentError("A buffer of length $(length(dat)) does not fit a composite volume with $(off[end]) degrees of freedom."))
        end
        return new{T}(dat, cvol, off)
    end
end

const MultiScaleField = GlaCmpFld

"""
    GlaCmpFld(cvol::GlaCmpVol; useGpu::Bool=false)

Construct the zero field over a composite volume. Same as `zerofield`.

# Arguments
- `cvol::GlaCmpVol`: The composite volume
- `useGpu::Bool=false`: Whether to put the buffer on the GPU

# Returns
- `GlaCmpFld`: A field of zeros with one entry per degree of freedom
"""
GlaCmpFld(cvol::GlaCmpVol; useGpu::Bool=false) = zerofield(cvol; useGpu=useGpu)

Base.zero(fld::GlaCmpFld) = GlaCmpFld(zero(fld.dat), fld.cvol)

# Region blocks only line up if the two fields agree on the tiling
_eqvCvl(cvolA::GlaCmpVol, cvolB::GlaCmpVol) = cvolA === cvolB || cvolA == cvolB

function _chkCvl(fldA::GlaCmpFld, fldB::GlaCmpFld)
    if !_eqvCvl(fldA.cvol, fldB.cvol)
        throw(ArgumentError("The two fields live on different composite volumes. Operations that mix fields need a common tiling."))
    end
    return nothing
end

Base.size(fld::GlaCmpFld) = size(fld.dat)
Base.IndexStyle(::Type{<:GlaCmpFld}) = IndexLinear()
Base.getindex(fld::GlaCmpFld, idx::Int) = fld.dat[idx]
Base.setindex!(fld::GlaCmpFld, val, idx::Int) = setindex!(fld.dat, val, idx)
Base.parent(fld::GlaCmpFld) = fld.dat

Base.similar(fld::GlaCmpFld) = GlaCmpFld(similar(fld.dat), fld.cvol)
Base.similar(fld::GlaCmpFld, ::Type{ComplexF64}) =
    GlaCmpFld(similar(fld.dat), fld.cvol)
# Only ComplexF64 buffers can be wrapped, so any other eltype comes back raw
Base.similar(fld::GlaCmpFld, ::Type{T}) where T = similar(fld.dat, T)
Base.copy(fld::GlaCmpFld) = GlaCmpFld(copy(fld.dat), fld.cvol)

function Base.show(io::IO, fld::GlaCmpFld)
    print(io, "Composite field ($(length(fld)) degrees of freedom, ", isa(fld.dat, CuArray) ? "GPU" : "CPU", ")\n  ", fld.cvol)
end
Base.show(io::IO, ::MIME"text/plain", fld::GlaCmpFld) = show(io, fld)

LinearAlgebra.dot(fldA::GlaCmpFld, fldB::GlaCmpFld) =
    (_chkCvl(fldA, fldB); dot(fldA.dat, fldB.dat))
LinearAlgebra.norm(fld::GlaCmpFld, p::Real=2) = norm(fld.dat, p)

function LinearAlgebra.axpy!(alp, fldX::GlaCmpFld, fldY::GlaCmpFld)
    _chkCvl(fldX, fldY)
    axpy!(alp, fldX.dat, fldY.dat)
    return fldY
end

function LinearAlgebra.axpby!(alp, fldX::GlaCmpFld, bet, fldY::GlaCmpFld)
    _chkCvl(fldX, fldY)
    axpby!(alp, fldX.dat, bet, fldY.dat)
    return fldY
end

LinearAlgebra.rmul!(fld::GlaCmpFld, alp::Number) = (rmul!(fld.dat, alp); fld)
Base.fill!(fld::GlaCmpFld, val) = (fill!(fld.dat, val); fld)

function Base.copyto!(fldDst::GlaCmpFld, fldSrc::GlaCmpFld)
    _chkCvl(fldDst, fldSrc)
    copyto!(fldDst.dat, fldSrc.dat)
    return fldDst
end

Base.BroadcastStyle(::Type{<:GlaCmpFld}) = Broadcast.ArrayStyle{GlaCmpFld}()

#= Replace every field in a broadcast tree by its buffer. Rebuilding with
broadcasted rather than reusing the wrapper style hands the work to the buffer,
so a GPU field goes through a GPU kernel instead of scalar indexing. =#
_bcUnw(arg) = arg
_bcUnw(fld::GlaCmpFld) = fld.dat
_bcUnw(bc::Broadcast.Broadcasted) =
    Broadcast.broadcasted(bc.f, map(_bcUnw, bc.args)...)

# Walk the same tree for the common composite volume, complaining on a mismatch
_bcCvl(cvol, arg) = cvol
_bcCvl(cvol, bc::Broadcast.Broadcasted) = foldl(_bcCvl, bc.args; init=cvol)
function _bcCvl(cvol, fld::GlaCmpFld)
    if !isnothing(cvol) && !_eqvCvl(cvol, fld.cvol)
        throw(ArgumentError("A broadcast mixes fields that live on different composite volumes. Operations that mix fields need a common tiling."))
    end
    return fld.cvol
end

_bcCvl(bc::Broadcast.Broadcasted) = foldl(_bcCvl, bc.args; init=nothing)

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{GlaCmpFld}},
    ::Type{T}) where T
    cvol = _bcCvl(bc)
    dat = similar(Broadcast.instantiate(_bcUnw(bc)), T)
    T === ComplexF64 || return dat
    return GlaCmpFld(dat, cvol)
end

function Base.copy(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{GlaCmpFld}})
    cvol = _bcCvl(bc)
    dat = Broadcast.materialize(_bcUnw(bc))
    eltype(dat) === ComplexF64 || return dat
    return GlaCmpFld(dat, cvol)
end

function Base.copyto!(fld::GlaCmpFld,
    bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{GlaCmpFld}})
    foldl(_bcCvl, bc.args; init=fld.cvol)
    Broadcast.materialize!(fld.dat, _bcUnw(bc))
    return fld
end

"""
    zerofield(cvol::GlaCmpVol; useGpu::Bool=false)

Allocate a zero field over a composite volume.

# Arguments
- `cvol::GlaCmpVol`: The composite volume
- `useGpu::Bool=false`: Whether to put the buffer on the GPU

# Returns
- `GlaCmpFld`: A field of zeros with one entry per degree of freedom
"""
function zerofield(cvol::GlaCmpVol; useGpu::Bool=false)
    len = sum(3 * prod(reg.cel) for reg in regions(cvol))
    dat = useGpu ? CUDA.zeros(ComplexF64, len) : zeros(ComplexF64, len)
    return GlaCmpFld(dat, cvol)
end

"""
    regionview(fld::GlaCmpFld, idx::Integer)

View the block of one region as a 4-tensor.

The result has size `(reg.cel..., 3)` and shares memory with the field, so
writing through it mutates the field. The entries are the stored √ΔV
coefficients, not densities. Dividing the whole view by `sqrt(prod(reg.scl))`
gives the current density, since a region has a single cell size and therefore a
single factor.

The shape and the index order are the ones `GlaOprVac` expects of a source or
target array, so the view can be handed straight to an operator.

# Arguments
- `fld::GlaCmpFld`: The field
- `idx::Integer`: The region index, in the order of `regions(fld.cvol)`

# Returns
- A 4-tensor view of size `(reg.cel..., 3)` into the field buffer
"""
function regionview(fld::GlaCmpFld, idx::Integer)
    reg = regions(fld.cvol)[idx]
    blk = view(fld.dat, (fld.off[idx] + 1):fld.off[idx + 1])
    return reshape(blk, (Int.(reg.cel)..., 3))
end

"""
    eachregion(fld::GlaCmpFld)

Iterate over the region views of a field, in `regions` order.

# Arguments
- `fld::GlaCmpFld`: The field

# Returns
- An iterator of the `regionview` of every region
"""
eachregion(fld::GlaCmpFld) = (regionview(fld, idx) for idx in 1:nregions(fld.cvol))

"""
    discretize!(fld::GlaCmpFld, f)

Fill a field from a function of position.

`f` takes a cell center as an `NTuple{3,Float64}` in wavelengths and returns
something indexable of length three, an `SVector`, a tuple, or a vector, holding
the three components of the current density there. Each value is multiplied by
`sqrt(prod(reg.scl))` on the way into the buffer, which is the conversion from a
density to a stored coefficient. Sampling is setup code, so the values are
always built on the CPU and copied to the buffer at the end.

# Arguments
- `fld::GlaCmpFld`: The field to fill
- `f`: The current density as a function of position

# Returns
- `GlaCmpFld`: The field, filled
"""
function discretize!(fld::GlaCmpFld, f)
    buf = Vector{ComplexF64}(undef, length(fld))
    for (regIdx, reg) in enumerate(regions(fld.cvol))
        celNum = prod(reg.cel)
        sclFac = sqrt(Float64(prod(reg.scl)))
        for (celIdx, celInd) in enumerate(CartesianIndices(Tuple(reg.cel)))
            pos = ntuple(dir -> Float64(reg.grd[dir][celInd[dir]]), 3)
            val = f(pos)
            for dir in 1:3
                buf[fld.off[regIdx] + (dir - 1) * celNum + celIdx] =
                    val[dir] * sclFac
            end
        end
    end
    copyto!(fld.dat, buf)
    return fld
end

"""
    regrid(fld::GlaCmpFld, scl::NTuple{3,Rational}=finest(fld.cvol))

Read a field off as current densities on a uniform grid.

The grid has cells of size `scl` and covers the bounding box of the composite
volume, which the regions tile exactly. Every uniform cell takes the density of
the source cell holding its center, so the result is piecewise constant and the
regions coarser than `scl` come out as blocks of repeated values. Since `scl`
divides every region scale, cell boundaries never cut across a source cell.

The returned array holds densities, not stored coefficients.

# Arguments
- `fld::GlaCmpFld`: The field to read
- `scl::NTuple{3,Rational}=finest(fld.cvol)`: The uniform cell size

# Returns
- `Array{ComplexF64,4}` of size `(cel..., 3)` with `cel` the uniform cell counts

# Throws
- `ArgumentError`: If `scl` is coarser than or incommensurate with the cell size
  of any region, or if it does not evenly divide the bounding box
"""
function regrid(fld::GlaCmpFld, scl::NTuple{3,Rational}=finest(fld.cvol))
    regs = regions(fld.cvol)
    for (idx, reg) in enumerate(regs)
        rat = reg.scl .// scl
        if any(.!isinteger.(rat)) || any(rat .< 1)
            throw(ArgumentError("A resampling scale of $(scl) does not work for region $idx, which has cells of $(reg.scl): the ratio is $(Tuple(rat)). The resampling scale has to divide every region scale a whole number of times."))
        end
    end
    boxLwr = reduce((edgA, edgB) -> min.(edgA, edgB), _lwrEdg.(regs))
    boxUpr = reduce((edgA, edgB) -> max.(edgA, edgB), _uprEdg.(regs))
    celNum = (boxUpr .- boxLwr) .// scl
    if any(.!isinteger.(celNum))
        throw(ArgumentError("A resampling scale of $(scl) does not evenly divide the bounding box, which spans $(Tuple(boxLwr)) to $(Tuple(boxUpr)) for $(Tuple(celNum)) cells."))
    end
    out = zeros(ComplexF64, ntuple(dir -> Int(celNum[dir]), 3)..., 3)
    dat = isa(fld.dat, Array) ? fld.dat : Array(fld.dat)
    for (regIdx, reg) in enumerate(regs)
        rep = ntuple(dir -> Int(reg.scl[dir] // scl[dir]), 3)
        regLwr = _lwrEdg(reg)
        bas = ntuple(dir -> Int((regLwr[dir] - boxLwr[dir]) // scl[dir]), 3)
        celNumReg = prod(reg.cel)
        sclFac = sqrt(Float64(prod(reg.scl)))
        for (celIdx, celInd) in enumerate(CartesianIndices(Tuple(reg.cel)))
            lwr = ntuple(dir -> bas[dir] + (celInd[dir] - 1) * rep[dir], 3)
            rng = ntuple(dir -> (lwr[dir] + 1):(lwr[dir] + rep[dir]), 3)
            for dir in 1:3
                out[rng..., dir] .=
                    dat[fld.off[regIdx] + (dir - 1) * celNumReg + celIdx] / sclFac
            end
        end
    end
    return out
end

end # module
