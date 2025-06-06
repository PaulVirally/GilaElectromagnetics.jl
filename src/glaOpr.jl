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
using CUDA

import ..GilaVacuum: useCpu!, useGpu!

export GlaOprVac, InvSctOpr, SctOpr, GlaOpr
export VacuumGreensOperator, InverseScatteringOperator, ScatteringOperator, GreensOperator
export isadjoint, isselfoperator, isexternaloperator, adjoint!, glaSze, slv

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
    GlaOprVac(opr::InvSctOpr)

Construct a vacuum Green's function operator from an inverse scattering operator.

# Arguments
- `opr::InvSctOpr`: The inverse scattering operator to convert into a vacuum Green's function operator

# Returns
- `GlaOprVac`: The vacuum Green's function operator
"""
GlaOprVac(opr::InvSctOpr) = opr.oprVac

"""
    GlaOprVac(opr::SctOpr)

Construct a vacuum Green's function operator from a scattering operator.

# Arguments
- `opr::SctOpr`: The scattering operator to convert into a vacuum Green's function operator

# Returns
- `GlaOprVac`: The vacuum Green's function operator
"""
GlaOprVac(opr::SctOpr) = GlaOprVac(opr.invSctOpr)

"""
    GlaOprVac(opr::GlaOpr)

Construct a vacuum Green's function operator from a full Green's function operator.

# Arguments
- `opr::GlaOpr`: The full Green's function operator to convert into a vacuum Green's function operator

# Returns
- `GlaOprVac`: The vacuum Green's function operator
"""
GlaOprVac(opr::GlaOpr) = GlaOprVac(opr.sctOpr)

"""
    InvSctOpr(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{ComplexF64}; useGpu::Bool=isa(sus, CuArray))

Construct an inverse scattering operator for external interactions between different volumes.

# Arguments
- `trgVol::GlaVol`: The target volume where the field will be computed
- `srcVol::GlaVol`: The source volume containing the sources
- `sus::AbstractArray{ComplexF64}`: The susceptibility tensor, either as a flat vector or a 3-tensor
- `useGpu::Bool=false`: Whether to use GPU computation. If true, uses GPU acceleration, otherwise uses CPU

# Returns
- `InvSctOpr`: The inverse scattering operator

This constructor creates an external inverse scattering operator that describes how electromagnetic fields interact with a material medium between distinct regions. The susceptibility tensor can be provided either as a flat vector (which will be reshaped to match the source volume dimensions) or as a 3-tensor directly. The tensor must match the dimensions of the source volume.
"""
function InvSctOpr(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{ComplexF64}; useGpu::Bool=isa(sus, CuArray))
    # Create the vacuum operator
    oprVac = GlaOprVac(trgVol, srcVol; useGpu=useGpu)
    
    # Reshape susceptibility if needed and validate size
    if size(sus) != srcVol.cel
        throw(ArgumentError("Susceptibility tensor dimensions $(size(sus)) do not match volume dimensions $(srcVol.cel)"))
    end
    susTen = rszSus(sus, srcVol.cel)
    
    return InvSctOpr(oprVac, susTen)
end

"""
    InvSctOpr(vol::GlaVol, sus::AbstractArray{ComplexF64}; useGpu::Bool=isa(sus, CuArray))

Construct an inverse scattering operator for self-interactions on a single volume.

# Arguments
- `vol::GlaVol`: The volume to compute the self-interaction for
- `sus::AbstractArray{ComplexF64}`: The susceptibility tensor, either as a flat vector or a 3-tensor
- `useGpu::Bool=isa(sus, CuArray)`: Whether to use GPU computation. If true, uses GPU acceleration, otherwise uses CPU

# Returns
- `InvSctOpr`: The inverse scattering operator

This constructor creates a self-interaction inverse scattering operator that describes how electromagnetic fields interact with a material medium within a single volume. The susceptibility tensor can be provided either as a flat vector (which will be reshaped to match the volume dimensions) or as a 3-tensor directly. The tensor must match the dimensions of the volume.
"""
InvSctOpr(vol::GlaVol, sus::AbstractArray{ComplexF64}; useGpu::Bool=isa(sus, CuArray)) = InvSctOpr(vol, vol, sus; useGpu=useGpu)

"""

    InvSctOpr(sctOpr::SctOpr)

Construct an inverse scattering operator from a scattering operator.

# Arguments
- `sctOpr::SctOpr`: The scattering operator to convert into an inverse scattering operator

# Returns
- `InvSctOpr`: The inverse scattering operator
"""
InvSctOpr(opr::SctOpr) = opr.invSctOpt

"""
    InvSctOpr(opr::GlaOpr)

Construct an inverse scattering operator from a full Green's function operator.

# Arguments
- `opr::GlaOpr`: The full Green's function operator to convert into an inverse scattering operator

# Returns
- `InvSctOpr`: The inverse scattering operator
"""
InvSctOpr(opr::GlaOpr) = InvSctOpr(opr.sctOpr)

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
    SctOpr(vol::GlaVol, sus::AbstractArray{ComplexF64}; useGpu::Bool=isa(sus, CuArray), slv::GlaSlv=BiCGStabSolver())

Construct a scattering operator for self-interactions on a single volume.

This constructor creates a self-interaction scattering operator that describes how electromagnetic fields interact with a material medium within a single volume. The susceptibility tensor can be provided either as a flat vector (which will be reshaped to match the volume dimensions) or as a 3-tensor directly. The tensor must match the dimensions of the volume.

# Arguments
- `vol::GlaVol`: The volume to compute the self-interaction for
- `sus::AbstractArray{ComplexF64}`: The susceptibility tensor, either as a flat vector or a 3-tensor
- `useGpu::Bool=isa(sus, CuArray)`: Whether to use GPU computation. If true, uses GPU acceleration, otherwise uses CPU
- `slv::GlaSlv=BiCGStabSolver()`: The solver to use for solving the linear system

# Returns
- `SctOpr`: The scattering operator
"""
SctOpr(vol::GlaVol, sus::AbstractArray{ComplexF64}; useGpu::Bool=isa(sus, CuArray), slv::GlaSlv=BiCGStabSolver()) = 
    SctOpr(vol, vol, sus; useGpu=useGpu, slv=slv)

"""
    SctOpr(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{ComplexF64}; useGpu::Bool=isa(sus, CuArray), slv::GlaSlv=BiCGStabSolver())

Construct a scattering operator for external interactions between different volumes.

This constructor creates an external scattering operator that describes how electromagnetic fields interact with a material medium between distinct regions. The susceptibility tensor can be provided either as a flat vector (which will be reshaped to match the source volume dimensions) or as a 3-tensor directly. The tensor must match the dimensions of the source volume.

# Arguments
- `trgVol::GlaVol`: The target volume where the field will be computed
- `srcVol::GlaVol`: The source volume containing the sources
- `sus::AbstractArray{ComplexF64}`: The susceptibility tensor, either as a flat vector or a 3-tensor
- `useGpu::Bool=isa(sus, CuArray)`: Whether to use GPU computation. If true, uses GPU acceleration, otherwise uses CPU
- `slv::GlaSlv=BiCGStabSolver()`: The solver to use for solving the linear system

# Returns
- `SctOpr`: The scattering operator
"""
function SctOpr(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{ComplexF64}; useGpu::Bool=isa(sus, CuArray), slv::GlaSlv=BiCGStabSolver())
    invSctOpr = InvSctOpr(trgVol, srcVol, sus; useGpu=useGpu)
    return SctOpr(invSctOpr, slv)
end

"""
    SctOpr(opr::GlaOprVac, sus::AbstractArray{ComplexF64}; useGpu::Bool=isa(sus, CuArray), slv::GlaSlv=BiCGStabSolver())

Construct a scattering operator from a vacuum Green's function operator.

This constructor creates a scattering operator that describes how electromagnetic fields interact with a material medium based on a given vacuum Green's function operator. The susceptibility tensor can be provided either as a flat vector (which will be reshaped to match the source volume dimensions) or as a 3-tensor directly. The tensor must match the dimensions of the source volume.

# Arguments
- `opr::GlaOprVac`: The vacuum Green's function operator to convert into a scattering operator
- `sus::AbstractArray{ComplexF64}`: The susceptibility tensor, either as a flat vector or a 3-tensor
- `useGpu::Bool=isa(sus, CuArray)`: Whether to use GPU computation. If true, uses GPU acceleration, otherwise uses CPU
- `slv::GlaSlv=BiCGStabSolver()`: The solver to use for solving the linear system

# Returns
- `SctOpr`: The scattering operator
"""
function SctOpr(opr::GlaOprVac, sus::AbstractArray{ComplexF64}; slv::GlaSlv=BiCGStabSolver())
    invSctOpr = InvSctOpr(opr, sus)
    return SctOpr(invSctOpr, slv)
end

"""
    SctOpr(opr::GlaOpr)

Construct a scattering operator from a full Green's function operator.

# Arguments
- `opr::GlaOpr`: The full Green's function operator to convert into a scattering operator

# Returns
- `SctOpr`: The scattering operator
"""
SctOpr(opr::GlaOpr) = opr.sctOpr

"""
    GlaOpr(vol::GlaVol, sus::AbstractArray{ComplexF64}; useGpu::Bool=isa(sus, CuArray), slv::GlaSlv=BiCGStabSolver())

Construct a full Green's function operator for self-interactions on a single volume.

This constructor creates a self-interaction full Green's function operator that combines the vacuum Green's function with the scattering operator to describe electromagnetic interactions in a material medium within a single volume. The susceptibility tensor can be provided either as a flat vector (which will be reshaped to match the volume dimensions) or as a 3-tensor directly. The tensor must match the dimensions of the volume.

# Arguments
- `vol::GlaVol`: The volume to compute the self-interaction for
- `sus::AbstractArray{ComplexF64}`: The susceptibility tensor, either as a flat vector or a 3-tensor
- `useGpu::Bool=isa(sus, CuArray)`: Whether to use GPU computation. If true, uses GPU acceleration, otherwise uses CPU
- `slv::GlaSlv=BiCGStabSolver()`: The solver to use for solving the linear system

# Returns
- `GlaOpr`: The full Green's function operator
"""
GlaOpr(vol::GlaVol, sus::AbstractArray{ComplexF64}; useGpu::Bool=isa(sus, CuArray), slv::GlaSlv=BiCGStabSolver()) = 
    GlaOpr(vol, vol, sus; useGpu=useGpu, slv=slv)

"""
    GlaOpr(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{ComplexF64}; useGpu::Bool=isa(sus, CuArray), slv::GlaSlv=BiCGStabSolver())

Construct a full Green's function operator for external interactions between different volumes.

This constructor creates an external full Green's function operator that combines the vacuum Green's function with the scattering operator to describe electromagnetic interactions in a material medium between distinct regions. The susceptibility tensor can be provided either as a flat vector (which will be reshaped to match the source volume dimensions) or as a 3-tensor directly. The tensor must match the dimensions of the source volume.

# Arguments
- `trgVol::GlaVol`: The target volume where the field will be computed
- `srcVol::GlaVol`: The source volume containing the sources
- `sus::AbstractArray{ComplexF64}`: The susceptibility tensor, either as a flat vector or a 3-tensor
- `useGpu::Bool=isa(sus, CuArray)`: Whether to use GPU computation. If true, uses GPU acceleration, otherwise uses CPU
- `slv::GlaSlv=BiCGStabSolver()`: The solver to use for solving the linear system

# Returns
- `GlaOpr`: The full Green's function operator
"""
function GlaOpr(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{ComplexF64}; useGpu::Bool=isa(sus, CuArray), slv::GlaSlv=BiCGStabSolver())
    sctOpr = SctOpr(trgVol, srcVol, sus; useGpu=useGpu, slv=slv)
    return GlaOpr(sctOpr)
end

"""
    GlaOpr(opr::GlaOprVac, sus::AbstractArray{ComplexF64}; useGpu::Bool=isa(sus, CuArray), slv::GlaSlv=BiCGStabSolver())

Construct a full Green's function operator from a vacuum Green's function operator.

This constructor creates a full Green's function operator that combines the vacuum Green's function with the scattering operator to describe electromagnetic interactions in a material medium based on a given vacuum Green's function operator. The susceptibility tensor can be provided either as a flat vector (which will be reshaped to match the source volume dimensions) or as a 3-tensor directly. The tensor must match the dimensions of the source volume.

# Arguments
- `opr::GlaOprVac`: The vacuum Green's function operator to convert into a full Green's function operator
- `sus::AbstractArray{ComplexF64}`: The susceptibility tensor, either as a flat vector or a 3-tensor
- `useGpu::Bool=isa(sus, CuArray)`: Whether to use GPU computation. If true, uses GPU acceleration, otherwise uses CPU
- `slv::GlaSlv=BiCGStabSolver()`: The solver to use for solving the linear system

# Returns
- `GlaOpr`: The full Green's function operator
"""
GlaOpr(opr::GlaOprVac, sus::AbstractArray{ComplexF64}; slv::GlaSlv=BiCGStabSolver()) = GlaOpr(SctOpr(opr, sus; slv=slv))

"""
    GlaOpr(opr::InvSctOpr)

Construct a full Green's function operator from an inverse scattering operator.

# Arguments
- `opr::InvSctOpr`: The inverse scattering operator to convert into a full Green's function operator

# Returns
- `GlaOpr`: The full Green's function operator
"""
GlaOpr(opr::InvSctOpr) = GlaOpr(SctOpr(opr))

function useCpu!(opr::GlaOprVac)
    useCpu!(opr.mem)
    return opr
end

function useGpu!(opr::GlaOprVac)
    useGpu!(opr.mem)
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

GilaVacuum.arrTyp(opr::GlaOprVac) = arrTyp(opr.mem.cmpInf)
GilaVacuum.arrTyp(opr::InvSctOpr) = arrTyp(opr.oprVac)
GilaVacuum.arrTyp(opr::SctOpr) = arrTyp(opr.invSctOpr)
GilaVacuum.arrTyp(opr::GlaOpr) = arrTyp(opr.sctOpr)

"""
    isadjoint(opr::AbstractGlaOpr)

Checks if the operator is the adjoint of the Green's operator.

# Arguments
- `opr::AbstractGlaOpr`: The operator to check.

# Returns
- `true` if the operator is the adjoint, `false` otherwise.
"""
isadjoint(opr::GlaOprVac) = opr.mem.cmpInf.adjMod
isadjoint(opr::InvSctOpr) = isadjoint(opr.oprVac)
isadjoint(opr::SctOpr) = isadjoint(opr.invSctOpr)
isadjoint(opr::GlaOpr) = isadjoint(opr.sctOpr)

"""
    isselfoperator(opr::AbstractGlaOpr)

Checks if the operator is a self Green's operator.

# Arguments
- `opr::AbstractGlaOpr`: The operator to check.

# Returns
- `true` if the operator is a self Green's operator, `false` otherwise.
"""
isselfoperator(opr::GlaOprVac) = opr.mem.srcVol == opr.mem.trgVol
isselfoperator(opr::InvSctOpr) = isselfoperator(opr.oprVac)
isselfoperator(opr::SctOpr) = isselfoperator(opr.invSctOpr)
isselfoperator(opr::GlaOpr) = isselfoperator(opr.sctOpr)

"""
    isexternaloperator(opr::GlaOprVac)

Checks if the operator is an external Green's operator.

# Arguments
- `opr::GlaOprVac`: The operator to check.

# Returns
- `true` if the operator is an external Green's operator, `false` otherwise.
"""
isexternaloperator(opr::GlaOprVac) = !isselfoperator(opr)
isexternaloperator(opr::InvSctOpr) = isexternaloperator(opr.oprVac)
isexternaloperator(opr::SctOpr) = isexternaloperator(opr.invSctOpr)
isexternaloperator(opr::GlaOpr) = isexternaloperator(opr.sctOpr)

"""
    setSus!(opr::InvSctOpr, sus::AbstractArray{ComplexF64})

Sets the susceptibility tensor for the inverse scattering operator.

# Arguments
- `opr::InvSctOpr`: The inverse scattering operator to modify.
- `sus::AbstractArray{ComplexF64}`: The new susceptibility tensor, either as a flat vector or a 3-tensor.

# Returns
- The modified operator with the new susceptibility tensor set.
"""
setSus!(opr::InvSctOpr, sus::AbstractArray{ComplexF64}) = begin
    # Reshape susceptibility if needed and validate size
    if size(sus) != opr.oprVac.mem.srcVol.cel
        throw(ArgumentError("Susceptibility tensor dimensions $(size(sus)) do not match source volume dimensions $(opr.oprVac.mem.srcVol.cel)"))
    end
    opr.sus = rszSus(sus, opr.oprVac.mem.srcVol.cel)
    return opr
end

"""
    setSus!(opr::SctOpr, sus::AbstractArray{ComplexF64})

Sets the susceptibility tensor for the scattering operator.

# Arguments
- `opr::SctOpr`: The scattering operator to modify.
- `sus::AbstractArray{ComplexF64}`: The new susceptibility tensor, either as a flat vector or a 3-tensor.

# Returns
- The modified operator with the new susceptibility tensor set.
"""
function setSus!(opr::SctOpr, sus::AbstractArray{ComplexF64})
    setSus!(opr.invSctOpr, sus)
    return opr
end

"""
    setSus!(opr::GlaOpr, sus::AbstractArray{ComplexF64})

Sets the susceptibility tensor for the full Green's function operator.

# Arguments
- `opr::GlaOpr`: The full Green's function operator to modify.
- `sus::AbstractArray{ComplexF64}`: The new susceptibility tensor, either as a flat vector or a 3-tensor.

# Returns
- The modified operator with the new susceptibility tensor set.
"""
function setSus!(opr::GlaOpr, sus::AbstractArray{ComplexF64})
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
slv(::GlaOprVac) = GilaSolvers.BiCGStabSolver() # Default solver
slv(opr::InvSctOpr) = slv(opr.oprVac)
slv(opr::SctOpr) = opr.slv
slv(opr::GlaOpr) = opr.sctOpr.slv

_strKnd(opr::GlaOprVac) = "G₀"
_strKnd(opr::InvSctOpr) = "(I - XG₀)"
_strKnd(opr::SctOpr) = "(I - XG₀)⁻¹"
_strKnd(opr::GlaOpr) = "G₀(I - XG₀)⁻¹"

_srcVol(opr::GlaOprVac) = opr.mem.srcVol
_srcVol(opr::InvSctOpr) = _srcVol(opr.oprVac)
_srcVol(opr::SctOpr) = _srcVol(opr.invSctOpr)
_srcVol(opr::GlaOpr) = _srcVol(opr.sctOpr)
_trgVol(opr::GlaOprVac) = opr.mem.trgVol
_trgVol(opr::InvSctOpr) = _trgVol(opr.oprVac)
_trgVol(opr::SctOpr) = _trgVol(opr.invSctOpr)
_trgVol(opr::GlaOpr) = _trgVol(opr.sctOpr)

function Base.show(io::IO, opr::AbstractGlaOpr)
    if isadjoint(opr)
        print(io, "Adjoint ")
    end
    if isselfoperator(opr)
        print(io, "Self ")
    else
        print(io, "External ")
    end
    print(io, _strKnd(opr))
    print(io, " for ")
    if isselfoperator(opr)
        print(io, "a $(eltype(opr)) (" * join(_srcVol(opr).cel, "×") * ") volume ")
        print(io, "of size (" * join(_srcVol(opr).scl, "×") * ")λ")
    else
        print(io, "$(eltype(opr)) (" * join(_srcVol(opr).cel, "×") * ") -> (" * join(_trgVol(opr).cel, "×") * ") volumes ")
        print(io, "of sizes (" * join(_srcVol(opr).scl, "×") * ")λ -> (" * join(_trgVol(opr).scl, "×") * ")λ ")
        print(io, "with separation (" * join(_trgVol(opr).org .- _srcVol(opr).org, ", ") * ")λ")
    end
end
Base.show(io::IO, ::MIME"text/plain", opr::AbstractGlaOpr) = show(io, opr)

include("glaLinAlg.jl")

end # module
