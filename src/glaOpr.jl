"""
    GilaOperators

This module provides the core operator types for the Gila package, including the vacuum Green function operator,
scattering operators, and their compositions.

# Types
- `AbstractGlaOpr`: Abstract base type for all operators
- `GlaOprVac`: Vacuum Green function operator G₀
- `InvSctOpr`: Inverse scattering operator (I - XG₀)⁻¹
- `SctOpr`: Scattering operator with solver
- `GlaOpr`: Full Green function operator G₀(I - XG₀)⁻¹

# Type Aliases
- `VacuumGreenOperator`: Alias for `GlaOprVac`
- `InverseScatteringOperator`: Alias for `InvSctOpr`
- `ScatteringOperator`: Alias for `SctOpr`
- `GreenOperator`: Alias for `GlaOpr`
"""
module GilaOperators

using ..GilaVolumes
using ..GilaFields
using ..GilaVacuum
using ..GilaTypes
using ..GilaSolvers
using CUDA
using Serialization

import LinearAlgebra: adjoint!

import ..GilaTypes: isgpu, isadjoint
import ..GilaVacuum: useCpu!, useGpu!, egoCmpPos
import ..GilaVolumes: _lwrEdg, _uprEdg, _ovrLap

export GlaOprVac, AsyGlaOprVac, SymGlaOprVac, MulRegGlaOprVac, GlaCmpOprVac, AsyGlaCmpOprVac, SymGlaCmpOprVac, InvSctOpr, SctOpr, GlaOpr
export VacuumGreenOperator, AsymVacuumGreenOperator, SymVacuumGreenOperator, MultiRegionVacuumGreenOperator, CompositeVacuumGreenOperator, AsymCompositeVacuumGreenOperator, SymCompositeVacuumGreenOperator, InverseScatteringOperator, ScatteringOperator, GreenOperator
export isadjoint, isselfoperator, isexternaloperator, isoverlappingoperator, isgpu, isquasistatic, adjoint!, glaSze, slv, asym

"""
    GlaOprVac{T}

Represents the vacuum Green function operator G₀, which describes electromagnetic
interactions in free space. `T` is the real storage precision, so the operator
acts on `Complex{T}` data.

# Fields
- `mem::GlaVacOprMem{T}`: Memory structure containing the operator's data, including
  volume information and Fourier coefficients
- `srcMsk::NTuple{StepRange{Int64, Int64}, 3}`: Tuple of ranges defining the mask for
  the input volume (for overlapping operators only)
- `trgMsk::NTuple{StepRage{Int64, Int64}, 3}`: Tuple of ranges defining the mask for
  the output volume (for overlapping operators only)
"""
struct GlaOprVac{T<:AbstractFloat} <: AbstractGlaVacOpr{T}
    mem::GlaVacOprMem{T}
    srcMsk::NTuple{3, OrdinalRange{Int64, Int64}}
    trgMsk::NTuple{3, OrdinalRange{Int64, Int64}}
end

GlaOprVac(mem::GlaVacOprMem{T}, srcMsk, trgMsk) where T<:AbstractFloat = GlaOprVac{T}(mem, srcMsk, trgMsk)

"""
    MulRegGlaOprVac{T}

Represents the vacuum Green function operator G₀ for multiple disjoint domains, i.e., the source and/or target volumes consist of multiple non-overlapping regions (where the gaps are *not* computed).

# Fields
- oprMat::Matrix{GlaOprVac{T}}: Matrix of vacuum Green function operators for each disjoint region pair
"""
struct MulRegGlaOprVac{T<:AbstractFloat} <: AbstractGlaVacOpr{T}
    oprMat::Matrix{GlaOprVac{T}}
end

MulRegGlaOprVac(oprMat::AbstractMatrix{GlaOprVac{T}}) where T<:AbstractFloat = MulRegGlaOprVac{T}(oprMat)

"""
    AsyGlaOprVac{T}

Represents the anti-Hermitian part of the vacuum Green function operator.

# Fields
- `mem::GlaVacOprMem{T}`: Memory structure containing the operator's data, including
  volume information and Fourier coefficients
"""
struct AsyGlaOprVac{T<:AbstractFloat} <: AbstractGlaVacOpr{T}
    mem::GlaVacOprMem{T}
end

"""
    SymGlaOprVac{T}

Represents the Hermitian part of the vacuum Green function operator.

# Fields
- `mem::GlaVacOprMem{T}`: Memory structure containing the operator's data, including
  volume information and Fourier coefficients
"""
struct SymGlaOprVac{T<:AbstractFloat} <: AbstractGlaVacOpr{T}
    mem::GlaVacOprMem{T}
end

"""
    InvSctOpr{T}

Represents the inverse scattering operator (I - XG₀), where X is the susceptibility
tensor. This operator describes how electromagnetic fields interact with a material
medium.

# Fields
- `oprVac::AbstractGlaVacOpr{T}`: The vacuum Green function operator
- `sus::AbstractArray{Complex{T}}`: The susceptibility (isotropic medium)
  representing the material response. Over a single volume it is a 3-tensor of
  cell values, over a composite volume a flat vector with one entry per degree of
  freedom
"""
mutable struct InvSctOpr{T<:AbstractFloat} <: AbstractGlaOpr{T}
    oprVac::AbstractGlaVacOpr{T}
    sus::AbstractArray{Complex{T}}

    #= A composite operator takes the susceptibility in any of the forms _cmpSus
    accepts and stores it in the flat degree of freedom layout of GlaFld. The
    susceptibility is the one place a precision conversion is allowed. =#
    function InvSctOpr{T}(oprVac::AbstractGlaVacOpr{T}, sus) where T<:AbstractFloat
        oprVac isa GlaCmpOprVac ||
            return new{T}(oprVac, sus isa AbstractArray{Complex{T}} ? sus : Complex{T}.(sus))
        if !isselfoperator(oprVac)
            throw(ArgumentError("An inverse scattering operator needs a self operator, and this composite operator maps between two different tilings."))
        end
        return new{T}(oprVac, _cmpSus(T, oprVac.srcCvl, sus, isgpu(oprVac)))
    end
end

InvSctOpr(oprVac::AbstractGlaVacOpr{T}, sus) where T<:AbstractFloat = InvSctOpr{T}(oprVac, sus)

"""
    SctOpr{T}

Represents the scattering operator (I - XG₀)⁻¹, which includes a solver for
computing the action of the inverse scattering operator.

# Fields
- `invSctOpr::InvSctOpr{T}`: The inverse scattering operator
- `slv::GlaSlv`: The solver to use for solving the linear system
"""
mutable struct SctOpr{T<:AbstractFloat} <: AbstractGlaOpr{T}
    invSctOpr::InvSctOpr{T}
    slv::GlaSlv
end

"""
    GlaOpr{T}

Represents the full Green function operator G₀(I - XG₀)⁻¹, which combines the
vacuum Green function with the scattering operator to describe electromagnetic
interactions in a material medium.

# Fields
- `sctOpr::SctOpr{T}`: The scattering operator
"""
mutable struct GlaOpr{T<:AbstractFloat} <: AbstractGlaOpr{T}
    sctOpr::SctOpr{T}
end

# Type aliases for convenience
const VacuumGreenOperator = GlaOprVac
const AsymVacuumGreenOperator = AsyGlaOprVac
const SymVacuumGreenOperator = SymGlaOprVac
const MultiRegionVacuumGreenOperator = MulRegGlaOprVac
const InverseScatteringOperator = InvSctOpr
const ScatteringOperator = SctOpr
const GreenOperator = GlaOpr

# Returns true if the volumes share interior. Face, edge, and corner contact is
# not overlap: the external construction has contact corrections for it, and the
# union path cannot represent contact between volumes at different cell scales.
function ovrChk(vol1::GlaVol, vol2::GlaVol)
    # Upper/lower edges of the volumes
    lwrEdg1 = first.(vol1.grd) .- (vol1.scl .// 2)
    uprEdg1 = last.(vol1.grd) .+ (vol1.scl .//2)
    lwrEdg2 = first.(vol2.grd) .- (vol2.scl .// 2)
    uprEdg2 = last.(vol2.grd) .+ (vol2.scl .//2)

    # Volume overlap check
    return all(max.(lwrEdg1, lwrEdg2) .< min.(uprEdg1, uprEdg2)) # < not <= to exclude contact
end

# Returns the region where subVol is contained within vol
function mskRng(subVol::GlaVol, vol::GlaVol)
    stpVol = Rational.(step.(vol.grd))
    stpSub = Rational.(step.(subVol.grd))
    stpRat = stpSub .// stpVol
    @assert all(isinteger.(stpRat)) "Volumes must share a common scale grid for masking"
    stpRat = Tuple(numerator.(stpRat))

    off = (first.(subVol.grd) .- first.(vol.grd)) .// stpVol # Offset between volumes
    @assert all(isinteger.(off)) "Sub-volume must align with volume grid for masking"
    off = Tuple(numerator.(off))
    idxBeg = 1 .+ off # 1-based indexing
    idxEnd = idxBeg .+ ((subVol.cel .- 1) .* stpRat)
    return (idxBeg[1]:stpRat[1]:idxEnd[1],
            idxBeg[2]:stpRat[2]:idxEnd[2],
            idxBeg[3]:stpRat[3]:idxEnd[3])
end

"""
    GlaOprVac{T}(trgVol::GlaVol, srcVol::GlaVol; useGpu::Bool=false, prxWrn::Bool=true)
    GlaOprVac(trgVol::GlaVol, srcVol::GlaVol; useGpu::Bool=false, prxWrn::Bool=true)

Construct a vacuum Green function operator for external interactions between different volumes.

The storage precision `T` defaults to `dflPrc`; generation is always done in `Float64` and rounded once.

This constructor creates an external Green function operator that describes electromagnetic interactions between distinct regions in free space. The operator maps sources in the source volume to fields in the target volume, enabling the modeling of coupling effects between different parts of an electromagnetic system. For the computation to work correctly, the source and target volumes must share a common scale grid.

# Arguments
- `trgVol::GlaVol`: The target volume where the field will be computed
- `srcVol::GlaVol`: The source volume containing the sources
- `useGpu::Bool=false`: Whether to use GPU computation. If true, uses GPU acceleration, otherwise uses CPU
- `prxWrn::Bool=true`: Whether to warn when the volumes are closer than a third of a wavelength

# Returns
- `GlaOprVac`: The vacuum Green function operator

"""
function GlaOprVac{T}(trgVol::GlaVol, srcVol::GlaVol;
    useGpu::Bool=false, prxWrn::Bool=true) where T<:AbstractFloat
    innMsk = ntuple(_ -> 0:0, 3)
    outMsk = ntuple(_ -> 0:0, 3)
    if trgVol != srcVol && ovrChk(trgVol, srcVol)
        # Overlap: create the union volume and mask out the input/output regions
        vol = uniVol(trgVol, srcVol)
        innMsk = mskRng(srcVol, vol)
        outMsk = mskRng(trgVol, vol)
        trgVol, srcVol = vol, vol
    end

    # Create the memory structure with appropriate GPU/CPU options
    mem = GlaVacOprMem(useGpu ? GPUKerOpt{T}() : CPUKerOpt{T}(), trgVol, srcVol; prxWrn=prxWrn)
    return GlaOprVac{T}(mem, innMsk, outMsk)
end
GlaOprVac(trgVol::GlaVol, srcVol::GlaVol; useGpu::Bool=false, prxWrn::Bool=true) =
    GlaOprVac{dflPrc}(trgVol, srcVol; useGpu=useGpu, prxWrn=prxWrn)

"""
    GlaOprVac(mem::GlaVacOprMem)

Construct a vacuum Green function operator from a memory structure.

# Arguments
- `mem::GlaVacOprMem`: The memory structure containing the operator's data

# Returns
- `GlaOprVac`: The vacuum Green function operator
"""
function GlaOprVac(mem::GlaVacOprMem{T}) where T<:AbstractFloat
    innMsk = ntuple(_ -> 0:0, 3)
    outMsk = ntuple(_ -> 0:0, 3)
    trgVol, srcVol = mem.trgVol, mem.srcVol
    if trgVol != srcVol && ovrChk(trgVol, srcVol)
        # Overlap: create the union volume and mask out the input/output regions
        vol = uniVol(trgVol, srcVol)
        innMsk = mskRng(srcVol, vol)
        outMsk = mskRng(trgVol, vol)
        trgVol, srcVol = vol, vol
    end
    return GlaOprVac{T}(mem, innMsk, outMsk)
end

"""
    GlaOprVac{T}(vol::GlaVol; useGpu::Bool=false)
    GlaOprVac(vol::GlaVol; useGpu::Bool=false)

Construct a vacuum Green function operator for self-interactions on a single volume.

This constructor creates a self-interaction Green function operator where the source and target volumes are identical. The operator describes electromagnetic interactions within a single volume in free space, making it suitable for modeling self-coupling effects in electromagnetic systems.

# Arguments
- `vol::GlaVol`: The volume to compute the self Green function for
- `useGpu::Bool=false`: Whether to use GPU computation. If true, uses GPU acceleration, otherwise uses CPU

# Returns
- `GlaOprVac`: The vacuum Green function operator

"""
GlaOprVac{T}(vol::GlaVol; useGpu::Bool=false) where T<:AbstractFloat = GlaOprVac{T}(vol, vol; useGpu=useGpu)
GlaOprVac(vol::GlaVol; useGpu::Bool=false) = GlaOprVac{dflPrc}(vol, vol; useGpu=useGpu)

"""
    GlaOprVac{T}(opr::GlaOprVac)

Convert the storage precision of a vacuum Green function operator to `T`, without regenerating its Fourier coefficients.
"""
GlaOprVac{T}(opr::GlaOprVac{T}) where T<:AbstractFloat = opr
GlaOprVac{T}(opr::GlaOprVac) where T<:AbstractFloat =
    GlaOprVac{T}(GlaVacOprMem{T}(opr.mem), opr.srcMsk, opr.trgMsk)

"""
    GlaOprVac(opr::InvSctOpr)

Construct a vacuum Green function operator from an inverse scattering operator.

# Arguments
- `opr::InvSctOpr`: The inverse scattering operator to convert into a vacuum Green function operator

# Returns
- `GlaOprVac`: The vacuum Green function operator
"""
GlaOprVac(opr::InvSctOpr) = opr.oprVac

"""
    GlaOprVac(opr::SctOpr)

Construct a vacuum Green function operator from a scattering operator.

# Arguments
- `opr::SctOpr`: The scattering operator to convert into a vacuum Green function operator

# Returns
- `GlaOprVac`: The vacuum Green function operator
"""
GlaOprVac(opr::SctOpr) = GlaOprVac(opr.invSctOpr)

"""
    GlaOprVac(opr::GlaOpr)

Construct a vacuum Green function operator from a full Green function operator.

# Arguments
- `opr::GlaOpr`: The full Green function operator to convert into a vacuum Green function operator

# Returns
- `GlaOprVac`: The vacuum Green function operator
"""
GlaOprVac(opr::GlaOpr) = GlaOprVac(opr.sctOpr)

"""
    AsyGlaOprVac{T}(vol::GlaVol; useGpu::Bool=false)
    AsyGlaOprVac(vol::GlaVol; useGpu::Bool=false)

Construct the anti-Hermitian part of the vacuum Green function operator for self-interactions on a single volume.

This constructor creates the anti-Hermitian part of the vacuum Green function operator, which describes the radiated components of electromagnetic interactions within a single volume in free space. This operator shows up in the fluctuation-dissipation theorem and is thus related to certain losses in the system.

# Arguments
- `vol::GlaVol`: The volume to compute the anti-Hermitian part of the vacuum Green function for
- `useGpu::Bool=false`: Whether to use GPU computation. If true, uses GPU acceleration, otherwise uses CPU

# Returns
- `AsyGlaOprVac`: The anti-Hermitian part of the vacuum Green function operator
"""
function AsyGlaOprVac{T}(vol::GlaVol; useGpu::Bool=false) where T<:AbstractFloat
    kerOpt = useGpu ? GPUKerOpt{T}() : CPUKerOpt{T}()
    mem = GlaVacOprMem(kerOpt, vol, vol)
    map!(fur -> complex.(imag.(fur)), mem.egoFur) # Take the imaginary part of the Fourier coefficients since Asym commutes with the FFT (to machine epsilon)
    return AsyGlaOprVac{T}(mem)
end
AsyGlaOprVac(vol::GlaVol; useGpu::Bool=false) = AsyGlaOprVac{dflPrc}(vol; useGpu=useGpu)

"""
    AsyGlaOprVac{T}(opr::AsyGlaOprVac)

Convert the storage precision of the operator to `T`, without regenerating its Fourier coefficients.
"""
AsyGlaOprVac{T}(opr::AsyGlaOprVac{T}) where T<:AbstractFloat = opr
AsyGlaOprVac{T}(opr::AsyGlaOprVac) where T<:AbstractFloat = AsyGlaOprVac{T}(GlaVacOprMem{T}(opr.mem))

"""
    AsyGlaOprVac(opr::GlaOprVac)

Construct the anti-Hermitian part of the vacuum Green function operator from a vacuum Green function operator.

# Arguments
- `opr::GlaOprVac`: The vacuum Green function operator to convert into its anti-Hermitian part

# Returns
- `AsyGlaOprVac`: The anti-Hermitian part of the vacuum Green function operator
"""
function AsyGlaOprVac(opr::GlaOprVac{T}) where T<:AbstractFloat
    srcVol, trgVol = opr.mem.srcVol, opr.mem.trgVol
    if srcVol != trgVol
        throw(ArgumentError("AsyGlaOprVac can only be constructed from a GlaOprVac with identical source and target volumes"))
    end
    mem = deepcopy(opr.mem)
    map!(fur -> complex.(imag.(fur)), mem.egoFur) # Take the imaginary part of the Fourier coefficients since Asym commutes with the FFT (to machine epsilon)
    return AsyGlaOprVac{T}(mem)
end

"""
    SymGlaOprVac{T}(vol::GlaVol; useGpu::Bool=false)
    SymGlaOprVac(vol::GlaVol; useGpu::Bool=false)

Construct the Hermitian part of the vacuum Green function operator for self-interactions on a single volume.

This constructor creates the Hermitian part of the vacuum Green function operator

# Arguments
- `vol::GlaVol`: The volume to compute the anti-Hermitian part of the vacuum Green function for
- `useGpu::Bool=false`: Whether to use GPU computation. If true, uses GPU acceleration, otherwise uses CPU

# Returns
- `SymGlaOprVac`: The Hermitian part of the vacuum Green function operator
"""
function SymGlaOprVac{T}(vol::GlaVol; useGpu::Bool=false) where T<:AbstractFloat
    kerOpt = useGpu ? GPUKerOpt{T}() : CPUKerOpt{T}()
    mem = GlaVacOprMem(kerOpt, vol, vol)
    map!(fur -> complex.(real.(fur)), mem.egoFur) # Take the real part of the Fourier coefficients since Sym commutes with the FFT (to machine epsilon)
    return SymGlaOprVac{T}(mem)
end
SymGlaOprVac(vol::GlaVol; useGpu::Bool=false) = SymGlaOprVac{dflPrc}(vol; useGpu=useGpu)

"""
    SymGlaOprVac{T}(opr::SymGlaOprVac)

Convert the storage precision of the operator to `T`, without regenerating its Fourier coefficients.
"""
SymGlaOprVac{T}(opr::SymGlaOprVac{T}) where T<:AbstractFloat = opr
SymGlaOprVac{T}(opr::SymGlaOprVac) where T<:AbstractFloat = SymGlaOprVac{T}(GlaVacOprMem{T}(opr.mem))

"""
    SymGlaOprVac(opr::GlaOprVac)

Construct the Hermitian part of the vacuum Green function operator from a vacuum Green function operator.

# Arguments
- `opr::GlaOprVac`: The vacuum Green function operator to convert into its Hermitian part

# Returns
- `SymGlaOprVac`: The Hermitian part of the vacuum Green function operator
"""
function SymGlaOprVac(opr::GlaOprVac{T}) where T<:AbstractFloat
    srcVol, trgVol = opr.mem.srcVol, opr.mem.trgVol
    if srcVol != trgVol
        throw(ArgumentError("SymGlaOprVac can only be constructed from a GlaOprVac with identical source and target volumes"))
    end
    mem = deepcopy(opr.mem)
    map!(fur -> complex.(real.(fur)), mem.egoFur) # Take the real part of the Fourier coefficients since Asym commutes with the FFT (to machine epsilon)
    return SymGlaOprVac{T}(mem)
end

"""
    MulRegGlaOprVac{T}(trgVols::VT{GlaVol}, srcVols::VT{GlaVol}; useGpu::Bool=false) where VT <: AbstractVector
    MulRegGlaOprVac(trgVols::VT{GlaVol}, srcVols::VT{GlaVol}; useGpu::Bool=false) where VT <: AbstractVector

Construct a vacuum Green function operator for multiple target and source volumes.

This constructor creates a vacuum Green function operator that describes interactions between multiple target and source volumes.

# Arguments
- `trgVols::VT{GlaVol}`: A vector of target volumes
- `srcVols::VT{GlaVol}`: A vector of source volumes
- `useGpu::Bool=false`: Whether to use GPU computation. If true, uses GPU acceleration, otherwise uses CPU

# Returns
- `MulRegGlaOprVac`: The vacuum Green function operator for multiple target and source volumes
"""
function MulRegGlaOprVac{T}(trgVols::VT, srcVols::VT; useGpu::Bool=false) where {T<:AbstractFloat, VT <: AbstractVector{GlaVol}}
    ops = [GlaOprVac{T}(trgVol, srcVol; useGpu=useGpu) for trgVol in trgVols, srcVol in srcVols]
    return MulRegGlaOprVac{T}(ops)
end
MulRegGlaOprVac(trgVols::VT, srcVols::VT; useGpu::Bool=false) where VT <: AbstractVector{GlaVol} =
    MulRegGlaOprVac{dflPrc}(trgVols, srcVols; useGpu=useGpu)

"""
    MulRegGlaOprVac{T}(opr::MulRegGlaOprVac)

Convert the storage precision of every block to `T`, without regenerating any Fourier coefficients.
"""
MulRegGlaOprVac{T}(opr::MulRegGlaOprVac{T}) where T<:AbstractFloat = opr
MulRegGlaOprVac{T}(opr::MulRegGlaOprVac) where T<:AbstractFloat =
    MulRegGlaOprVac{T}(map(blk -> GlaOprVac{T}(blk), opr.oprMat))

"""
    InvSctOpr{T}(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{<:Number}; useGpu::Bool=isa(sus, CuArray))
    InvSctOpr(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{<:Number}; useGpu::Bool=isa(sus, CuArray))

Construct an inverse scattering operator for external interactions between different volumes.

# Arguments
- `trgVol::GlaVol`: The target volume where the field will be computed
- `srcVol::GlaVol`: The source volume containing the sources
- `sus::AbstractArray{<:Number}`: The susceptibility tensor, either as a flat vector or a 3-tensor, converted to `Complex{T}` on construction
- `useGpu::Bool=false`: Whether to use GPU computation. If true, uses GPU acceleration, otherwise uses CPU

# Returns
- `InvSctOpr{T}`: The inverse scattering operator, of storage precision `T` (`dflPrc` when unrequested)

This constructor creates an external inverse scattering operator that describes how electromagnetic fields interact with a material medium between distinct regions. The susceptibility tensor can be provided either as a flat vector (which will be reshaped to match the source volume dimensions) or as a 3-tensor directly. The tensor must match the dimensions of the source volume.
"""
function InvSctOpr{T}(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{<:Number}; useGpu::Bool=isa(sus, CuArray)) where T<:AbstractFloat
    # Create the vacuum operator
    oprVac = GlaOprVac{T}(trgVol, srcVol; useGpu=useGpu)

    # Reshape susceptibility if needed and validate size
    if size(sus) != srcVol.cel
        throw(ArgumentError("Susceptibility tensor dimensions $(size(sus)) do not match volume dimensions $(srcVol.cel)"))
    end
    susTen = rszSus(sus, srcVol.cel)

    return InvSctOpr{T}(oprVac, susTen)
end
InvSctOpr(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{<:Number}; useGpu::Bool=isa(sus, CuArray)) =
    InvSctOpr{dflPrc}(trgVol, srcVol, sus; useGpu=useGpu)

"""
    InvSctOpr{T}(vol::GlaVol, sus::AbstractArray{<:Number}; useGpu::Bool=isa(sus, CuArray))
    InvSctOpr(vol::GlaVol, sus::AbstractArray{<:Number}; useGpu::Bool=isa(sus, CuArray))

Construct an inverse scattering operator for self-interactions on a single volume.

# Arguments
- `vol::GlaVol`: The volume to compute the self-interaction for
- `sus::AbstractArray{<:Number}`: The susceptibility tensor, either as a flat vector or a 3-tensor, converted to `Complex{T}` on construction
- `useGpu::Bool=isa(sus, CuArray)`: Whether to use GPU computation. If true, uses GPU acceleration, otherwise uses CPU

# Returns
- `InvSctOpr{T}`: The inverse scattering operator, of storage precision `T` (`dflPrc` when unrequested)

This constructor creates a self-interaction inverse scattering operator that describes how electromagnetic fields interact with a material medium within a single volume. The susceptibility tensor can be provided either as a flat vector (which will be reshaped to match the volume dimensions) or as a 3-tensor directly. The tensor must match the dimensions of the volume.
"""
InvSctOpr{T}(vol::GlaVol, sus::AbstractArray{<:Number}; useGpu::Bool=isa(sus, CuArray)) where T<:AbstractFloat =
    InvSctOpr{T}(vol, vol, sus; useGpu=useGpu)
InvSctOpr(vol::GlaVol, sus::AbstractArray{<:Number}; useGpu::Bool=isa(sus, CuArray)) =
    InvSctOpr{dflPrc}(vol, vol, sus; useGpu=useGpu)

"""
    InvSctOpr{T}(opr::InvSctOpr)

Convert the storage precision of the operator, and of its susceptibility, to `T`.
"""
InvSctOpr{T}(opr::InvSctOpr{T}) where T<:AbstractFloat = opr
InvSctOpr{T}(opr::InvSctOpr) where T<:AbstractFloat =
    InvSctOpr{T}(Base.typename(typeof(opr.oprVac)).wrapper{T}(opr.oprVac), Complex{T}.(opr.sus))

"""

    InvSctOpr(sctOpr::SctOpr)

Construct an inverse scattering operator from a scattering operator.

# Arguments
- `sctOpr::SctOpr`: The scattering operator to convert into an inverse scattering operator

# Returns
- `InvSctOpr`: The inverse scattering operator
"""
InvSctOpr(opr::SctOpr) = opr.invSctOpr

"""
    InvSctOpr(opr::GlaOpr)

Construct an inverse scattering operator from a full Green function operator.

# Arguments
- `opr::GlaOpr`: The full Green function operator to convert into an inverse scattering operator

# Returns
- `InvSctOpr`: The inverse scattering operator
"""
InvSctOpr(opr::GlaOpr) = InvSctOpr(opr.sctOpr)

# Reshape a flat susceptibility vector into a 3-tensor matching the volume dimensions.
function rszSus(sus::AbstractArray{<:Number}, cel::NTuple{3,Integer})
    if ndims(sus) == 3
        return sus
    elseif ndims(sus) == 1
        if length(sus) != prod(cel)
            throw(ArgumentError("Flat susceptibility vector length ($(length(sus))) does not match volume size ($(prod(cel)))"))
        end
        return reshape(sus, cel)
    else
        throw(ArgumentError("Susceptibility must be either a flat vector or a 3-tensor"))
    end
end

"""
    SctOpr{T}(vol::GlaVol, sus::AbstractArray{<:Number}; useGpu::Bool=isa(sus, CuArray), slv::GlaSlv=BiCGStabSolver())
    SctOpr(vol::GlaVol, sus::AbstractArray{<:Number}; useGpu::Bool=isa(sus, CuArray), slv::GlaSlv=BiCGStabSolver())

Construct a scattering operator for self-interactions on a single volume.

This constructor creates a self-interaction scattering operator that describes how electromagnetic fields interact with a material medium within a single volume. The susceptibility tensor can be provided either as a flat vector (which will be reshaped to match the volume dimensions) or as a 3-tensor directly. The tensor must match the dimensions of the volume.

# Arguments
- `vol::GlaVol`: The volume to compute the self-interaction for
- `sus::AbstractArray{<:Number}`: The susceptibility tensor, either as a flat vector or a 3-tensor, converted to `Complex{T}` on construction
- `useGpu::Bool=isa(sus, CuArray)`: Whether to use GPU computation. If true, uses GPU acceleration, otherwise uses CPU
- `slv::GlaSlv=BiCGStabSolver()`: The solver to use for solving the linear system

# Returns
- `SctOpr`: The scattering operator
"""
SctOpr{T}(vol::GlaVol, sus::AbstractArray{<:Number}; useGpu::Bool=isa(sus, CuArray), slv::GlaSlv=BiCGStabSolver()) where T<:AbstractFloat =
    SctOpr{T}(vol, vol, sus; useGpu=useGpu, slv=slv)
SctOpr(vol::GlaVol, sus::AbstractArray{<:Number}; useGpu::Bool=isa(sus, CuArray), slv::GlaSlv=BiCGStabSolver()) =
    SctOpr{dflPrc}(vol, vol, sus; useGpu=useGpu, slv=slv)

"""
    SctOpr{T}(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{<:Number}; useGpu::Bool=isa(sus, CuArray), slv::GlaSlv=BiCGStabSolver())
    SctOpr(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{<:Number}; useGpu::Bool=isa(sus, CuArray), slv::GlaSlv=BiCGStabSolver())

Construct a scattering operator for external interactions between different volumes.

This constructor creates an external scattering operator that describes how electromagnetic fields interact with a material medium between distinct regions. The susceptibility tensor can be provided either as a flat vector (which will be reshaped to match the source volume dimensions) or as a 3-tensor directly. The tensor must match the dimensions of the source volume.

# Arguments
- `trgVol::GlaVol`: The target volume where the field will be computed
- `srcVol::GlaVol`: The source volume containing the sources
- `sus::AbstractArray{<:Number}`: The susceptibility tensor, either as a flat vector or a 3-tensor, converted to `Complex{T}` on construction
- `useGpu::Bool=isa(sus, CuArray)`: Whether to use GPU computation. If true, uses GPU acceleration, otherwise uses CPU
- `slv::GlaSlv=BiCGStabSolver()`: The solver to use for solving the linear system

# Returns
- `SctOpr`: The scattering operator
"""
function SctOpr{T}(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{<:Number}; useGpu::Bool=isa(sus, CuArray), slv::GlaSlv=BiCGStabSolver()) where T<:AbstractFloat
    invSctOpr = InvSctOpr{T}(trgVol, srcVol, sus; useGpu=useGpu)
    return SctOpr{T}(invSctOpr, slv)
end
SctOpr(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{<:Number}; useGpu::Bool=isa(sus, CuArray), slv::GlaSlv=BiCGStabSolver()) =
    SctOpr{dflPrc}(trgVol, srcVol, sus; useGpu=useGpu, slv=slv)

"""
    SctOpr(opr::GlaOprVac, sus::AbstractArray{<:Number}; slv::GlaSlv=BiCGStabSolver())

Construct a scattering operator from a vacuum Green function operator.

This constructor creates a scattering operator that describes how electromagnetic fields interact with a material medium based on a given vacuum Green function operator. The susceptibility tensor can be provided either as a flat vector (which will be reshaped to match the source volume dimensions) or as a 3-tensor directly. The tensor must match the dimensions of the source volume.

# Arguments
- `opr::GlaOprVac`: The vacuum Green function operator to convert into a scattering operator
- `sus::AbstractArray{<:Number}`: The susceptibility tensor, either as a flat vector or a 3-tensor, converted to `Complex{T}` on construction
- `useGpu::Bool=isa(sus, CuArray)`: Whether to use GPU computation. If true, uses GPU acceleration, otherwise uses CPU
- `slv::GlaSlv=BiCGStabSolver()`: The solver to use for solving the linear system

# Returns
- `SctOpr`: The scattering operator
"""
function SctOpr(opr::GlaOprVac{T}, sus::AbstractArray{<:Number}; slv::GlaSlv=BiCGStabSolver()) where T<:AbstractFloat
    invSctOpr = InvSctOpr{T}(opr, sus)
    return SctOpr{T}(invSctOpr, slv)
end

"""
    SctOpr{T}(opr::SctOpr)

Convert the storage precision of the operator, and of its susceptibility, to `T`. The solver is kept.
"""
SctOpr{T}(opr::SctOpr{T}) where T<:AbstractFloat = opr
SctOpr{T}(opr::SctOpr) where T<:AbstractFloat = SctOpr{T}(InvSctOpr{T}(opr.invSctOpr), opr.slv)

"""
    SctOpr(opr::GlaOpr)

Construct a scattering operator from a full Green function operator.

# Arguments
- `opr::GlaOpr`: The full Green function operator to convert into a scattering operator

# Returns
- `SctOpr`: The scattering operator
"""
SctOpr(opr::GlaOpr) = opr.sctOpr

"""
    GlaOpr{T}(vol::GlaVol, sus::AbstractArray{<:Number}; useGpu::Bool=isa(sus, CuArray), slv::GlaSlv=BiCGStabSolver())
    GlaOpr(vol::GlaVol, sus::AbstractArray{<:Number}; useGpu::Bool=isa(sus, CuArray), slv::GlaSlv=BiCGStabSolver())

Construct a full Green function operator for self-interactions on a single volume.

This constructor creates a self-interaction full Green function operator that combines the vacuum Green function with the scattering operator to describe electromagnetic interactions in a material medium within a single volume. The susceptibility tensor can be provided either as a flat vector (which will be reshaped to match the volume dimensions) or as a 3-tensor directly. The tensor must match the dimensions of the volume.

# Arguments
- `vol::GlaVol`: The volume to compute the self-interaction for
- `sus::AbstractArray{<:Number}`: The susceptibility tensor, either as a flat vector or a 3-tensor, converted to `Complex{T}` on construction
- `useGpu::Bool=isa(sus, CuArray)`: Whether to use GPU computation. If true, uses GPU acceleration, otherwise uses CPU
- `slv::GlaSlv=BiCGStabSolver()`: The solver to use for solving the linear system

# Returns
- `GlaOpr`: The full Green function operator
"""
GlaOpr{T}(vol::GlaVol, sus::AbstractArray{<:Number}; useGpu::Bool=isa(sus, CuArray), slv::GlaSlv=BiCGStabSolver()) where T<:AbstractFloat =
    GlaOpr{T}(vol, vol, sus; useGpu=useGpu, slv=slv)
GlaOpr(vol::GlaVol, sus::AbstractArray{<:Number}; useGpu::Bool=isa(sus, CuArray), slv::GlaSlv=BiCGStabSolver()) =
    GlaOpr{dflPrc}(vol, vol, sus; useGpu=useGpu, slv=slv)

"""
    GlaOpr{T}(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{<:Number}; useGpu::Bool=isa(sus, CuArray), slv::GlaSlv=BiCGStabSolver())
    GlaOpr(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{<:Number}; useGpu::Bool=isa(sus, CuArray), slv::GlaSlv=BiCGStabSolver())

Construct a full Green function operator for external interactions between different volumes.

This constructor creates an external full Green function operator that combines the vacuum Green function with the scattering operator to describe electromagnetic interactions in a material medium between distinct regions. The susceptibility tensor can be provided either as a flat vector (which will be reshaped to match the source volume dimensions) or as a 3-tensor directly. The tensor must match the dimensions of the source volume.

# Arguments
- `trgVol::GlaVol`: The target volume where the field will be computed
- `srcVol::GlaVol`: The source volume containing the sources
- `sus::AbstractArray{<:Number}`: The susceptibility tensor, either as a flat vector or a 3-tensor, converted to `Complex{T}` on construction
- `useGpu::Bool=isa(sus, CuArray)`: Whether to use GPU computation. If true, uses GPU acceleration, otherwise uses CPU
- `slv::GlaSlv=BiCGStabSolver()`: The solver to use for solving the linear system

# Returns
- `GlaOpr`: The full Green function operator
"""
function GlaOpr{T}(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{<:Number}; useGpu::Bool=isa(sus, CuArray), slv::GlaSlv=BiCGStabSolver()) where T<:AbstractFloat
    sctOpr = SctOpr{T}(trgVol, srcVol, sus; useGpu=useGpu, slv=slv)
    return GlaOpr{T}(sctOpr)
end
GlaOpr(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{<:Number}; useGpu::Bool=isa(sus, CuArray), slv::GlaSlv=BiCGStabSolver()) =
    GlaOpr{dflPrc}(trgVol, srcVol, sus; useGpu=useGpu, slv=slv)

"""
    GlaOpr(opr::GlaOprVac, sus::AbstractArray{<:Number}; slv::GlaSlv=BiCGStabSolver())

Construct a full Green function operator from a vacuum Green function operator.

This constructor creates a full Green function operator that combines the vacuum Green function with the scattering operator to describe electromagnetic interactions in a material medium based on a given vacuum Green function operator. The susceptibility tensor can be provided either as a flat vector (which will be reshaped to match the source volume dimensions) or as a 3-tensor directly. The tensor must match the dimensions of the source volume.

# Arguments
- `opr::GlaOprVac`: The vacuum Green function operator to convert into a full Green function operator
- `sus::AbstractArray{<:Number}`: The susceptibility tensor, either as a flat vector or a 3-tensor, converted to `Complex{T}` on construction
- `useGpu::Bool=isa(sus, CuArray)`: Whether to use GPU computation. If true, uses GPU acceleration, otherwise uses CPU
- `slv::GlaSlv=BiCGStabSolver()`: The solver to use for solving the linear system

# Returns
- `GlaOpr`: The full Green function operator
"""
GlaOpr(opr::GlaOprVac, sus::AbstractArray{<:Number}; slv::GlaSlv=BiCGStabSolver()) = GlaOpr(SctOpr(opr, sus; slv=slv))

"""
    GlaOpr{T}(opr::GlaOpr)

Convert the storage precision of the operator, and of its susceptibility, to `T`. The solver is kept.
"""
GlaOpr{T}(opr::GlaOpr{T}) where T<:AbstractFloat = opr
GlaOpr{T}(opr::GlaOpr) where T<:AbstractFloat = GlaOpr{T}(SctOpr{T}(opr.sctOpr))

"""
    GlaOpr(opr::InvSctOpr)

Construct a full Green function operator from an inverse scattering operator.

# Arguments
- `opr::InvSctOpr`: The inverse scattering operator to convert into a full Green function operator

# Returns
- `GlaOpr`: The full Green function operator
"""
GlaOpr(opr::InvSctOpr) = GlaOpr(SctOpr(opr))

function useCpu!(opr::Union{GlaOprVac, AsyGlaOprVac, SymGlaOprVac})
    useCpu!(opr.mem)
    return opr
end

function useGpu!(opr::Union{GlaOprVac, AsyGlaOprVac, SymGlaOprVac})
    useGpu!(opr.mem)
    return opr
end

function useCpu!(opr::MulRegGlaOprVac)
    useCpu!.(opr.oprMat)
    return opr
end

function useGpu!(opr::MulRegGlaOprVac)
    useGpu!.(opr.oprMat)
    return opr
end

function useCpu!(opr::InvSctOpr)
    useCpu!(opr.oprVac)
    opr.sus = Array(opr.sus)
    return opr
end

function useGpu!(opr::InvSctOpr)
    useGpu!(opr.oprVac)
    opr.sus = CuArray(opr.sus)
    return opr
end

function useCpu!(opr::SctOpr)
    useCpu!(opr.invSctOpr)
    return opr
end

function useGpu!(opr::SctOpr)
    useGpu!(opr.invSctOpr)
    return opr
end

function useCpu!(opr::GlaOpr)
    useCpu!(opr.sctOpr)
    return opr
end

function useGpu!(opr::GlaOpr)
    useGpu!(opr.sctOpr)
    return opr
end

GilaVacuum.arrTyp(opr::Union{GlaOprVac, AsyGlaOprVac, SymGlaOprVac}) = arrTyp(opr.mem.cmpInf)
GilaVacuum.arrTyp(opr::MulRegGlaOprVac) = arrTyp(first(opr.oprMat))
GilaVacuum.arrTyp(opr::InvSctOpr) = arrTyp(opr.oprVac)
GilaVacuum.arrTyp(opr::SctOpr) = arrTyp(opr.invSctOpr)
GilaVacuum.arrTyp(opr::GlaOpr) = arrTyp(opr.sctOpr)

"""
    asym(opr::GlaOprVac)

Construct the anti-Hermitian part of a vacuum Green function operator.

# Arguments
- `opr::GlaOprVac`: The vacuum Green function operator to convert into its anti-Hermitian part

# Returns
- `AsyGlaOprVac`: The anti-Hermitian part of the vacuum Green function operator
"""
asym(opr::GlaOprVac) = AsyGlaOprVac(opr)

"""
    sym(opr::GlaOprVac)

Construct the -ermitian part of a vacuum Green function operator.

# Arguments
- `opr::GlaOprVac`: The vacuum Green function operator to convert into its Hermitian part

# Returns
- `SymGlaOprVac`: The Hermitian part of the vacuum Green function operator
"""
sym(opr::GlaOprVac) = SymGlaOprVac(opr)

"""
    isadjoint(opr::AbstractGlaOpr)

Checks if the operator is the adjoint of the Green operator.

# Arguments
- `opr::AbstractGlaOpr`: The operator to check.

# Returns
- `true` if the operator is the adjoint, `false` otherwise.
"""
isadjoint(opr::GlaOprVac) = opr.mem.cmpInf.adjMod
isadjoint(::Union{AsyGlaOprVac, SymGlaOprVac}) = false
isadjoint(opr::MulRegGlaOprVac) = all(isadjoint.(opr.oprMat))
isadjoint(opr::InvSctOpr) = isadjoint(opr.oprVac)
isadjoint(opr::SctOpr) = isadjoint(opr.invSctOpr)
isadjoint(opr::GlaOpr) = isadjoint(opr.sctOpr)

"""
    isquasistatic(opr::AbstractGlaOpr)

Checks if the operator was built from the quasistatic Green function rather than the full one.

# Arguments
- `opr::AbstractGlaOpr`: The operator to check.

# Returns
- `true` if the operator is quasistatic, `false` otherwise.
"""
isquasistatic(opr::Union{GlaOprVac, AsyGlaOprVac, SymGlaOprVac}) = opr.mem.cmpInf.qssApx
isquasistatic(opr::MulRegGlaOprVac) = all(isquasistatic.(opr.oprMat))
isquasistatic(opr::InvSctOpr) = isquasistatic(opr.oprVac)
isquasistatic(opr::SctOpr) = isquasistatic(opr.invSctOpr)
isquasistatic(opr::GlaOpr) = isquasistatic(opr.sctOpr)

"""
    isselfoperator(opr::AbstractGlaOpr)

Checks if the operator is a self Green operator.

# Arguments
- `opr::AbstractGlaOpr`: The operator to check.

# Returns
- `true` if the operator is a self Green operator, `false` otherwise.
"""
isselfoperator(opr::GlaOprVac) = (opr.mem.srcVol == opr.mem.trgVol) && all(==(0:0), opr.srcMsk) && all(==(0:0), opr.trgMsk)
isselfoperator(::Union{AsyGlaOprVac, SymGlaOprVac}) = true # Both are always self operators
isselfoperator(opr::MulRegGlaOprVac) = all(isselfoperator.(opr.oprMat))
isselfoperator(opr::InvSctOpr) = isselfoperator(opr.oprVac)
isselfoperator(opr::SctOpr) = isselfoperator(opr.invSctOpr)
isselfoperator(opr::GlaOpr) = isselfoperator(opr.sctOpr)

"""
    isexternaloperator(opr::GlaOprVac)

Checks if the operator is an external Green operator.

# Arguments
- `opr::GlaOprVac`: The operator to check.

# Returns
- `true` if the operator is an external Green operator, `false` otherwise.
"""
isexternaloperator(opr::GlaOprVac) = opr.mem.srcVol != opr.mem.trgVol && all(==(0:0), opr.srcMsk) && all(==(0:0), opr.trgMsk)
isexternaloperator(::Union{AsyGlaOprVac, SymGlaOprVac}) = false # Both are always self operators
isexternaloperator(opr::MulRegGlaOprVac) = any(isexternaloperator.(opr.oprMat))
isexternaloperator(opr::InvSctOpr) = isexternaloperator(opr.oprVac)
isexternaloperator(opr::SctOpr) = isexternaloperator(opr.invSctOpr)
isexternaloperator(opr::GlaOpr) = isexternaloperator(opr.sctOpr)

"""
    isoverlappingoperator(opr::GlaOprVac)

Checks if the operator is an overlapping Green operator.

# Arguments
- `opr::GlaOprVac`: The operator to check.

# Returns
- `true` if the operator is an overlapping Green operator, `false` otherwise.
"""
isoverlappingoperator(::AbstractGlaVacOpr) = false # Only the masked routes overlap
isoverlappingoperator(opr::GlaOprVac) = !(isselfoperator(opr) || isexternaloperator(opr))
isoverlappingoperator(opr::MulRegGlaOprVac) = any(isoverlappingoperator.(opr.oprMat))
isoverlappingoperator(opr::InvSctOpr) = isoverlappingoperator(opr.oprVac)
isoverlappingoperator(opr::SctOpr) = isoverlappingoperator(opr.invSctOpr)
isoverlappingoperator(opr::GlaOpr) = isoverlappingoperator(opr.sctOpr)

"""
    isgpu(opr::AbstractGlaOpr)

Checks if the operator is using GPU computation.

# Arguments
- `opr::AbstractGlaOpr`: The operator to check.

# Returns
- `true` if the operator is using GPU computation, `false` otherwise.
"""
# cmpInf itself is the GlaKerOpt; bckEnd(cmpInf) is a KernelAbstractions backend,
# which is never a GlaKerOpt
isgpu(opr::Union{GlaOprVac, AsyGlaOprVac, SymGlaOprVac}) = opr.mem.cmpInf isa GPUKerOpt
isgpu(opr::MulRegGlaOprVac) = all(isgpu.(opr.oprMat))
isgpu(opr::InvSctOpr) = isgpu(opr.oprVac)
isgpu(opr::SctOpr) = isgpu(opr.invSctOpr)
isgpu(opr::GlaOpr) = isgpu(opr.sctOpr)

"""
    setSus!(opr::InvSctOpr, sus::AbstractArray{<:Number})

Sets the susceptibility tensor for the inverse scattering operator.

# Arguments
- `opr::InvSctOpr`: The inverse scattering operator to modify.
- `sus::AbstractArray{<:Number}`: The new susceptibility tensor, either as a flat vector or a 3-tensor, converted to the precision of the operator.

# Returns
- The modified operator with the new susceptibility tensor set.
"""
function setSus!(opr::InvSctOpr{T}, sus) where T<:AbstractFloat
    if opr.oprVac isa GlaCmpOprVac
        opr.sus = _cmpSus(T, opr.oprVac.srcCvl, sus, isgpu(opr.oprVac))
        return opr
    end
    # Reshape susceptibility if needed and validate size
    if size(sus) != opr.oprVac.mem.srcVol.cel
        throw(ArgumentError("Susceptibility tensor dimensions $(size(sus)) do not match source volume dimensions $(opr.oprVac.mem.srcVol.cel)"))
    end
    susTen = rszSus(sus, opr.oprVac.mem.srcVol.cel)
    opr.sus = susTen isa AbstractArray{Complex{T}} ? susTen : Complex{T}.(susTen)
    return opr
end

"""
    setSus!(opr::SctOpr, sus::AbstractArray{<:Number})

Sets the susceptibility tensor for the scattering operator.

# Arguments
- `opr::SctOpr`: The scattering operator to modify.
- `sus::AbstractArray{<:Number}`: The new susceptibility tensor, either as a flat vector or a 3-tensor, converted to the precision of the operator.

# Returns
- The modified operator with the new susceptibility tensor set.
"""
function setSus!(opr::SctOpr, sus)
    setSus!(opr.invSctOpr, sus)
    return opr
end

"""
    setSus!(opr::GlaOpr, sus::AbstractArray{<:Number})

Sets the susceptibility tensor for the full Green function operator.

# Arguments
- `opr::GlaOpr`: The full Green function operator to modify.
- `sus::AbstractArray{<:Number}`: The new susceptibility tensor, either as a flat vector or a 3-tensor, converted to the precision of the operator.

# Returns
- The modified operator with the new susceptibility tensor set.
"""
function setSus!(opr::GlaOpr, sus)
    setSus!(opr.sctOpr, sus)
    return opr
end

"""
    slv(opr::AbstractGlaOpr)

Returns the solver associated with the operator.

# Arguments
- `opr::AbstractGlaOpr`: The operator for which to get the solver.

# Returns
- The solver used by the operator, which is always a `GlaSlv` instance.
"""
slv(::AbstractGlaVacOpr) = GilaSolvers.BiCGStabSolver() # Default solver
slv(opr::InvSctOpr) = slv(opr.oprVac)
slv(opr::SctOpr) = opr.slv
slv(opr::GlaOpr) = opr.sctOpr.slv

_strKnd(opr::GlaOprVac) = "G₀"
_strKnd(opr::AsyGlaOprVac) = "Asym(G₀)"
_strKnd(opr::SymGlaOprVac) = "Sym(G₀)"
_strKnd(opr::MulRegGlaOprVac) = "multi-region G₀"
_strKnd(opr::InvSctOpr) = "(I - XG₀)"
_strKnd(opr::SctOpr) = "(I - XG₀)⁻¹"
_strKnd(opr::GlaOpr) = "G₀(I - XG₀)⁻¹"

_srcVol(opr::Union{GlaOprVac, AsyGlaOprVac, SymGlaOprVac}) = opr.mem.srcVol
_srcVol(opr::InvSctOpr) = _srcVol(opr.oprVac)
_srcVol(opr::SctOpr) = _srcVol(opr.invSctOpr)
_srcVol(opr::GlaOpr) = _srcVol(opr.sctOpr)
_trgVol(opr::Union{GlaOprVac, AsyGlaOprVac, SymGlaOprVac}) = opr.mem.trgVol
_trgVol(opr::InvSctOpr) = _trgVol(opr.oprVac)
_trgVol(opr::SctOpr) = _trgVol(opr.invSctOpr)
_trgVol(opr::GlaOpr) = _trgVol(opr.sctOpr)

Base.show(io::IO, opr::AbstractGlaOpr) = _shwOpr(io, opr)

#= A scattering operator over a composite volume has no single source volume to
print, so it borrows the layout of the composite vacuum operator instead. =#
function Base.show(io::IO, opr::Union{InvSctOpr, SctOpr, GlaOpr})
    oprVac = GlaOprVac(opr)
    oprVac isa GlaCmpOprVac || return _shwOpr(io, opr)
    isadjoint(opr) && print(io, "Adjoint ")
    print(io, isgpu(opr) ? "GPU " : "CPU ")
    print(io, "composite ", _strKnd(opr))
    print(io, "\n  $(size(opr, 1)) × $(size(opr, 2)) degrees of freedom")
    print(io, "\n  ", oprVac.srcCvl)
end

function _shwOpr(io::IO, opr::AbstractGlaOpr)
    if isadjoint(opr)
        print(io, "Adjoint ")
    end
    if isselfoperator(opr)
        print(io, "Self ")
    elseif isexternaloperator(opr)
        print(io, "External ")
    else
        print(io, "Overlapping ")
    end
    if isgpu(opr)
        print(io, "GPU ")
    else
        print(io, "CPU ")
    end
    isquasistatic(opr) && print(io, "quasistatic ")
    print(io, _strKnd(opr))
    print(io, " for ")
    if isselfoperator(opr)
        print(io, "a $(eltype(opr)) (" * join(_srcVol(opr).cel, "×") * ") volume ")
        print(io, "of size (" * join(_srcVol(opr).scl, "×") * ")λ³")
    else
        print(io, "$(eltype(opr)) (" * join(glaSze(opr)[2][1:3], "×") * ") -> (" * join(glaSze(opr)[1][1:3], "×") * ") volumes ")
        print(io, "of sizes (" * join(_srcVol(opr).scl, "×") * ")λ³ -> (" * join(_trgVol(opr).scl, "×") * ")λ³")
        if isexternaloperator(opr)
            print(io, " with center separation (" * join(_trgVol(opr).org .- _srcVol(opr).org, ", ") * ")λ")
        end
    end
end
Base.show(io::IO, ::MIME"text/plain", opr::AbstractGlaOpr) = show(io, opr)
function Base.show(io::IO, opr::MulRegGlaOprVac)
    m, n = size(opr.oprMat)
    isadjoint(opr) && print(io, "Adjoint ")
    print(io, isgpu(opr) ? "GPU " : "CPU ")
    print(io, "multi-region G₀ ")
    print(io, "($m target", m == 1 ? "" : "s", " × $n source", n == 1 ? "" : "s", ")")

    trgVols = [_trgVol(opr.oprMat[i, 1]) for i in 1:m]
    srcVols = [_srcVol(opr.oprMat[1, j]) for j in 1:n]

    _fmtVol(v) = "(" * join(v.cel, "×") * ") cells, (" * join(v.scl, "×") * ")λ³"

    println(io)
    print(io, "  targets:")
    for (i, v) in enumerate(trgVols)
        print(io, "\n    [$i] ", _fmtVol(v))
    end
    println(io)
    print(io, "  sources:")
    for (j, v) in enumerate(srcVols)
        print(io, "\n    [$j] ", _fmtVol(v))
    end
end
Base.show(io::IO, ::MIME"text/plain", opr::MulRegGlaOprVac) = show(io, opr)

include("glaLinAlg.jl")
include("glaCmpOpr.jl")

#= An overlapping operator holds the union volume in its memory, so the masks are
the only record of the two sub-volumes and have to be written alongside it. =#
function Serialization.serialize(io::IO, opr::GlaOprVac)
    serialize(io, opr.mem)
    serialize(io, opr.srcMsk)
    serialize(io, opr.trgMsk)
end
function Serialization.deserialize(io::IO, ::Type{<:GlaOprVac})
    mem = deserialize(io, GlaVacOprMem)
    srcMsk = deserialize(io)
    trgMsk = deserialize(io)
    return GlaOprVac(mem, srcMsk, trgMsk)
end
Serialization.serialize(io::IO, opr::AsyGlaOprVac) = serialize(io, opr.mem)
Serialization.deserialize(io::IO, ::Type{<:AsyGlaOprVac}) = AsyGlaOprVac(deserialize(io, GlaVacOprMem))
Serialization.serialize(io::IO, opr::SymGlaOprVac) = serialize(io, opr.mem)
Serialization.deserialize(io::IO, ::Type{<:SymGlaOprVac}) = SymGlaOprVac(deserialize(io, GlaVacOprMem))
Serialization.serialize(io::IO, opr::MulRegGlaOprVac) = serialize(io, opr.oprMat)
# The blocks go through the generic serializer, which rebuilds their FFTW plans
# on load, see vacuum/glaVacOprMem.jl
Serialization.deserialize(io::IO, ::Type{<:MulRegGlaOprVac}) = MulRegGlaOprVac(deserialize(io))
function Serialization.serialize(io::IO, opr::GlaSndOprVac)
    serialize(io, opr.opr)
    serialize(io, opr.trgRat)
    serialize(io, opr.srcRat)
    serialize(io, opr.wgt)
end
function Serialization.deserialize(io::IO, ::Type{<:GlaSndOprVac})
    innOpr = deserialize(io, GlaOprVac)
    trgRat = deserialize(io)
    srcRat = deserialize(io)
    wgt = deserialize(io)
    return GlaSndOprVac(innOpr, trgRat, srcRat, wgt)
end
function Serialization.serialize(io::IO, opr::GlaCmpOprVac)
    serialize(io, opr.trgCvl)
    serialize(io, opr.srcCvl)
    # The blocks go through the generic serializer, as for MulRegGlaOprVac above
    serialize(io, opr.blkMat)
end
function Serialization.deserialize(io::IO, ::Type{<:GlaCmpOprVac})
    trgCvl = deserialize(io)
    srcCvl = deserialize(io)
    blkMat = deserialize(io)
    return GlaCmpOprVac(trgCvl, srcCvl, blkMat)
end
#= The written blocks already carry the Fourier coefficients of the part, so the
raw constructor is the one to read them back with. =#
Serialization.serialize(io::IO, opr::AsyGlaCmpOprVac) = serialize(io, opr.opr)
Serialization.deserialize(io::IO, ::Type{<:AsyGlaCmpOprVac}) =
    AsyGlaCmpOprVac(deserialize(io, GlaCmpOprVac), Val(:raw))
Serialization.serialize(io::IO, opr::SymGlaCmpOprVac) = serialize(io, opr.opr)
Serialization.deserialize(io::IO, ::Type{<:SymGlaCmpOprVac}) =
    SymGlaCmpOprVac(deserialize(io, GlaCmpOprVac), Val(:raw))
function Serialization.serialize(io::IO, opr::InvSctOpr)
    #= The vacuum operator is written by whichever method its runtime type
    selects, so its type leads the payload and picks the reader. =#
    serialize(io, typeof(opr.oprVac))
    serialize(io, opr.oprVac)
    sus = opr.sus
    if sus isa CuArray
        sus = Array(sus) # Convert to CPU array for serialization
    end
    serialize(io, sus)
end
function Serialization.deserialize(io::IO, ::Type{<:InvSctOpr})
    vacTyp = deserialize(io)
    oprVac = deserialize(io, vacTyp)
    sus = deserialize(io)
    return InvSctOpr(oprVac, sus)
end
function Serialization.serialize(io::IO, opr::SctOpr)
    serialize(io, opr.invSctOpr)
    serialize(io, opr.slv)
end
function Serialization.deserialize(io::IO, ::Type{<:SctOpr})
    invSctOpr = deserialize(io, InvSctOpr)
    slv = deserialize(io)
    return SctOpr(invSctOpr, slv)
end
Serialization.serialize(io::IO, opr::GlaOpr) = serialize(io, opr.sctOpr)
Serialization.deserialize(io::IO, ::Type{<:GlaOpr}) = GlaOpr(deserialize(io, SctOpr))

end # module
