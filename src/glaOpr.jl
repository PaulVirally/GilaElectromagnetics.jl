"""
    GilaOperators

This module provides the core operator types for the Gila package, including the vacuum Green's function operator,
scattering operators, and their compositions.

# Types
- `AbstractGlaOpr`: Abstract base type for all operators
- `GlaOprVac`: Vacuum Green's function operator G₀
- `InvSctOpr`: Inverse scattering operator (I - XG₀)⁻¹
- `SctOpr`: Scattering operator with solver
- `GlaOpr`: Full Green's function operator G₀(I - XG₀)⁻¹

# Type Aliases
- `VacuumGreensOperator`: Alias for `GlaOprVac`
- `InverseScatteringOperator`: Alias for `InvSctOpr`
- `ScatteringOperator`: Alias for `SctOpr`
- `GreensOperator`: Alias for `GlaOpr`
"""
module GilaOperators

using ..GilaVolumes
using ..GilaVacuum
using ..GilaTypes
using ..GilaSolvers

export GlaOprVac, InvSctOpr, SctOpr, GlaOpr
export VacuumGreensOperator, InverseScatteringOperator, ScatteringOperator, GreensOperator
export isadjoint, isselfoperator, isexternaloperator, adjoint!, glaSze

"""
    GlaOprVac

Represents the vacuum Green's function operator G₀, which describes electromagnetic
interactions in free space.

# Fields
- `mem::GlaVacOprMem`: Memory structure containing the operator's data, including
  volume information and Fourier coefficients
"""
struct GlaOprVac <: AbstractGlaOpr
    mem::GlaVacOprMem
end

"""
    InvSctOpr

Represents the inverse scattering operator (I - XG₀)⁻¹, where X is the susceptibility
tensor. This operator describes how electromagnetic fields interact with a material
medium.

# Fields
- `oprVac::GlaOprVac`: The vacuum Green's function operator
- `sus::AbstractArray{ComplexF64, 3}`: The susceptibility tensor (isotropic medium)
  representing the material response
"""
mutable struct InvSctOpr <: AbstractGlaOpr
    oprVac::GlaOprVac
    sus::AbstractArray{ComplexF64, 3}
end

"""
    SctOpr

Represents the scattering operator (I - XG₀)⁻¹, which includes a solver for
computing the action of the inverse scattering operator.

# Fields
- `invSctOpr::InvSctOpr`: The inverse scattering operator
- `slv::GlaSlv`: The solver to use for solving the linear system
"""
mutable struct SctOpr <: AbstractGlaOpr
    invSctOpr::InvSctOpr
    slv::GlaSlv
end

"""
    GlaOpr

Represents the full Green's function operator G₀(I - XG₀)⁻¹, which combines the
vacuum Green's function with the scattering operator to describe electromagnetic
interactions in a material medium.

# Fields
- `sctOpr::SctOpr`: The scattering operator
"""
mutable struct GlaOpr <: AbstractGlaOpr
    sctOpr::SctOpr
end

# Type aliases for convenience
const VacuumGreensOperator = GlaOprVac
const InverseScatteringOperator = InvSctOpr
const ScatteringOperator = SctOpr
const GreensOperator = GlaOpr

"""
    GlaOprVac(trgVol::GlaVol, srcVol::GlaVol; useGpu::Bool=false)

Construct a vacuum Green's function operator for external interactions between different volumes.

This constructor creates an external Green's function operator that describes electromagnetic interactions between distinct regions in free space. The operator maps sources in the source volume to fields in the target volume, enabling the modeling of coupling effects between different parts of an electromagnetic system. For the computation to work correctly, the source and target volumes must share a common scale grid.

# Arguments
- `trgVol::GlaVol`: The target volume where the field will be computed
- `srcVol::GlaVol`: The source volume containing the sources
- `useGpu::Bool=false`: Whether to use GPU computation. If true, uses GPU acceleration, otherwise uses CPU

# Returns
- `GlaOprVac`: The vacuum Green's function operator

"""
function GlaOprVac(trgVol::GlaVol, srcVol::GlaVol; useGpu::Bool=false)
    # Create the memory structure with appropriate GPU/CPU options
    mem = GlaVacOprMem(useGpu ? GPUKerOpt() : CPUKerOpt(), trgVol, srcVol)
    return GlaOprVac(mem)
end

"""
    GlaOprVac(vol::GlaVol; useGpu::Bool=false)

Construct a vacuum Green's function operator for self-interactions on a single volume.

This constructor creates a self-interaction Green's function operator where the source and target volumes are identical. The operator describes electromagnetic interactions within a single volume in free space, making it suitable for modeling self-coupling effects in electromagnetic systems.

# Arguments
- `vol::GlaVol`: The volume to compute the self Green's function for
- `useGpu::Bool=false`: Whether to use GPU computation. If true, uses GPU acceleration, otherwise uses CPU

# Returns
- `GlaOprVac`: The vacuum Green's function operator

"""
GlaOprVac(vol::GlaVol; useGpu::Bool=false) = GlaOprVac(vol, vol; useGpu=useGpu)

"""
    InvSctOpr(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{ComplexF64}; useGpu::Bool=false)

Construct an inverse scattering operator for external interactions between different volumes.

# Arguments
- `trgVol::GlaVol`: The target volume where the field will be computed
- `srcVol::GlaVol`: The source volume containing the sources
- `sus::AbstractArray{ComplexF64}`: The susceptibility tensor, either as a flat vector or a 3-tensor
- `useGpu::Bool=false`: Whether to use GPU computation

# Returns
- `InvSctOpr`: The inverse scattering operator

This constructor creates an external inverse scattering operator that describes how electromagnetic fields interact with a material medium between distinct regions. The susceptibility tensor can be provided either as a flat vector (which will be reshaped to match the source volume dimensions) or as a 3-tensor directly. The tensor must match the dimensions of the source volume.
"""
function InvSctOpr(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{ComplexF64}; useGpu::Bool=false)
    # Create the vacuum operator
    oprVac = GlaOprVac(trgVol, srcVol; useGpu=useGpu)
    
    # Reshape susceptibility if needed and validate size
    susTen = rszSus(sus, srcVol.cel)
    # Make sure sus has the right size
    if size(sus) != srcVol.cel
        throw(ArgumentError("Susceptibility tensor dimensions $(size(sus)) do not match volume dimensions $cel"))
    end
    
    return InvSctOpr(oprVac, susTen)
end

"""
    InvSctOpr(vol::GlaVol, sus::AbstractArray{ComplexF64}; useGpu::Bool=false)

Construct an inverse scattering operator for self-interactions on a single volume.

# Arguments
- `vol::GlaVol`: The volume to compute the self-interaction for
- `sus::AbstractArray{ComplexF64}`: The susceptibility tensor, either as a flat vector or a 3-tensor
- `useGpu::Bool=false`: Whether to use GPU computation

# Returns
- `InvSctOpr`: The inverse scattering operator

This constructor creates a self-interaction inverse scattering operator that describes how electromagnetic fields interact with a material medium within a single volume. The susceptibility tensor can be provided either as a flat vector (which will be reshaped to match the volume dimensions) or as a 3-tensor directly. The tensor must match the dimensions of the volume.
"""
InvSctOpr(vol::GlaVol, sus::AbstractArray{ComplexF64}; useGpu::Bool=false) = InvSctOpr(vol, vol, sus; useGpu=useGpu)

# Reshape a flat susceptibility vector into a 3-tensor matching the volume dimensions.
function rszSus(sus::AbstractArray{ComplexF64}, cel::NTuple{3,Integer})
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
    SctOpr(vol::GlaVol, sus::AbstractArray{ComplexF64}; useGpu::Bool=false, slv::GlaSlv=BiCGStabSolver())

Construct a scattering operator for self-interactions on a single volume.

This constructor creates a self-interaction scattering operator that describes how electromagnetic fields interact with a material medium within a single volume. The susceptibility tensor can be provided either as a flat vector (which will be reshaped to match the volume dimensions) or as a 3-tensor directly. The tensor must match the dimensions of the volume.

# Arguments
- `vol::GlaVol`: The volume to compute the self-interaction for
- `sus::AbstractArray{ComplexF64}`: The susceptibility tensor, either as a flat vector or a 3-tensor
- `useGpu::Bool=false`: Whether to use GPU computation
- `slv::GlaSlv=BiCGStabSolver()`: The solver to use for solving the linear system

# Returns
- `SctOpr`: The scattering operator
"""
SctOpr(vol::GlaVol, sus::AbstractArray{ComplexF64}; useGpu::Bool=false, slv::GlaSlv=BiCGStabSolver()) = 
    SctOpr(vol, vol, sus; useGpu=useGpu, slv=slv)

"""
    SctOpr(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{ComplexF64}; useGpu::Bool=false, slv::GlaSlv=BiCGStabSolver())

Construct a scattering operator for external interactions between different volumes.

This constructor creates an external scattering operator that describes how electromagnetic fields interact with a material medium between distinct regions. The susceptibility tensor can be provided either as a flat vector (which will be reshaped to match the source volume dimensions) or as a 3-tensor directly. The tensor must match the dimensions of the source volume.

# Arguments
- `trgVol::GlaVol`: The target volume where the field will be computed
- `srcVol::GlaVol`: The source volume containing the sources
- `sus::AbstractArray{ComplexF64}`: The susceptibility tensor, either as a flat vector or a 3-tensor
- `useGpu::Bool=false`: Whether to use GPU computation
- `slv::GlaSlv=BiCGStabSolver()`: The solver to use for solving the linear system

# Returns
- `SctOpr`: The scattering operator
"""
function SctOpr(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{ComplexF64}; useGpu::Bool=false, slv::GlaSlv=BiCGStabSolver())
    invSctOpr = InvSctOpr(trgVol, srcVol, sus; useGpu=useGpu)
    return SctOpr(invSctOpr, slv)
end

"""
    GlaOpr(vol::GlaVol, sus::AbstractArray{ComplexF64}; useGpu::Bool=false, slv::GlaSlv=BiCGStabSolver())

Construct a full Green's function operator for self-interactions on a single volume.

This constructor creates a self-interaction full Green's function operator that combines the vacuum Green's function with the scattering operator to describe electromagnetic interactions in a material medium within a single volume. The susceptibility tensor can be provided either as a flat vector (which will be reshaped to match the volume dimensions) or as a 3-tensor directly. The tensor must match the dimensions of the volume.

# Arguments
- `vol::GlaVol`: The volume to compute the self-interaction for
- `sus::AbstractArray{ComplexF64}`: The susceptibility tensor, either as a flat vector or a 3-tensor
- `useGpu::Bool=false`: Whether to use GPU computation
- `slv::GlaSlv=BiCGStabSolver()`: The solver to use for solving the linear system

# Returns
- `GlaOpr`: The full Green's function operator
"""
GlaOpr(vol::GlaVol, sus::AbstractArray{ComplexF64}; useGpu::Bool=false, slv::GlaSlv=BiCGStabSolver()) = 
    GlaOpr(vol, vol, sus; useGpu=useGpu, slv=slv)

"""
    GlaOpr(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{ComplexF64}; useGpu::Bool=false, slv::GlaSlv=BiCGStabSolver())

Construct a full Green's function operator for external interactions between different volumes.

This constructor creates an external full Green's function operator that combines the vacuum Green's function with the scattering operator to describe electromagnetic interactions in a material medium between distinct regions. The susceptibility tensor can be provided either as a flat vector (which will be reshaped to match the source volume dimensions) or as a 3-tensor directly. The tensor must match the dimensions of the source volume.

# Arguments
- `trgVol::GlaVol`: The target volume where the field will be computed
- `srcVol::GlaVol`: The source volume containing the sources
- `sus::AbstractArray{ComplexF64}`: The susceptibility tensor, either as a flat vector or a 3-tensor
- `useGpu::Bool=false`: Whether to use GPU computation
- `slv::GlaSlv=BiCGStabSolver()`: The solver to use for solving the linear system

# Returns
- `GlaOpr`: The full Green's function operator
"""
function GlaOpr(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{ComplexF64}; useGpu::Bool=false, slv::GlaSlv=BiCGStabSolver())
    sctOpr = SctOpr(trgVol, srcVol, sus; useGpu=useGpu, slv=slv)
    return GlaOpr(sctOpr)
end

include("glaLinAlg.jl")

end # module