using KernelAbstractions
using CUDA
using Serialization
using ..GilaTypes

"""
    GlaKerOpt{T<:AbstractFloat}

Abstract type for computational information that determines how the Green function
operator is computed. `T` is the real storage precision of the operator data
(`Complex{T}`); generation is always performed in `Float64`. Concrete implementations
include `CPUKerOpt` for CPU computation and `GPUKerOpt` for GPU computation.
"""
abstract type GlaKerOpt{T<:AbstractFloat} end

"""
    CPUKerOpt{T} <: GlaKerOpt{T}

Options for CPU computation of the Green function operator.

`CPUKerOpt` determines the accuracy of the Green function computation for cells in contact through its integration order. Higher values give better accuracy but require more computation. The phase factor allows for complex frequencies, which is useful for modeling dispersive media or for numerical stability.

# Fields
- `frqPhz::Number`: Multiplicative scaling factor allowing for complex frequencies
- `intOrd::Integer`: Gauss-Legendre integration order for cells in contact
- `adjMod::Bool`: Adjoint mode flag (true if adjoint mode is enabled)
- `bckEnd::CPU`: Backend for CPU computation
"""
mutable struct CPUKerOpt{T<:AbstractFloat} <: GlaKerOpt{T}
    frqPhz::Number
    intOrd::Integer
    adjMod::Bool
    bckEnd::CPU
end

"""
    CPUKerOpt{T}()

Construct a CPUKerOpt of storage precision `T` with default values.

The default integration order of 48 keeps the singular corrections near
5e-11. This is affordable because the weak integrals fold their redundant terms.

# Returns
- `CPUKerOpt{T}`: A new CPU kernel options object with:
  - Phase factor of 1.0 + 0.0im
  - Integration order of 48
  - Adjoint mode disabled
  - Default CPU backend
"""
CPUKerOpt{T}() where T<:AbstractFloat = CPUKerOpt{T}(1.0+0.0im, 48, false, CPU())

"""
    CPUKerOpt(frqPhz, intOrd, adjMod, bckEnd)
    CPUKerOpt()

Construct a CPUKerOpt of the default storage precision `dfltPrc`.
"""
CPUKerOpt(frqPhz::Number, intOrd::Integer, adjMod::Bool, bckEnd::CPU) = CPUKerOpt{dfltPrc}(frqPhz, intOrd, adjMod, bckEnd)
CPUKerOpt() = CPUKerOpt{dfltPrc}()

"""
    CPUKerOpt{T}(opt::CPUKerOpt)

Re-type a CPUKerOpt to storage precision `T`, keeping all field values.
"""
CPUKerOpt{T}(opt::CPUKerOpt) where T<:AbstractFloat = CPUKerOpt{T}(opt.frqPhz, opt.intOrd, opt.adjMod, opt.bckEnd)

"""
    frqPhz(opt::CPUKerOpt) -> Number

Get the multiplicative scaling factor allowing for complex frequencies.

# Arguments
- `opt::CPUKerOpt`: The CPU kernel options object

# Returns
- `Number`: The phase factor
"""
frqPhz(opt::CPUKerOpt) = opt.frqPhz

"""
    intOrd(opt::CPUKerOpt)

Get the integration order from a CPU kernel options object.

# Arguments
- `opt::CPUKerOpt`: The CPU kernel options object

# Returns
- `Integer`: The integration order
"""
intOrd(opt::CPUKerOpt) = opt.intOrd

"""
    adjMod(opt::CPUKerOpt)

Get the adjoint mode flag from a CPU kernel options object.

# Arguments
- `opt::CPUKerOpt`: The CPU kernel options object

# Returns
- `Bool`: True if adjoint mode is enabled, false otherwise
"""
adjMod(opt::CPUKerOpt) = opt.adjMod

"""
    bckEnd(opt::CPUKerOpt)

Get the CPU backend from a CPU kernel options object.

# Arguments
- `opt::CPUKerOpt`: The CPU kernel options object

# Returns
- `CPU`: The CPU backend
"""
bckEnd(opt::CPUKerOpt) = opt.bckEnd

"""
    arrTyp(opt::CPUKerOpt{T})

Get the array type from a CPU kernel options object.

# Arguments
- `opt::CPUKerOpt{T}`: The CPU kernel options object

# Returns
- `Type`: Array{Complex{T}}
"""
arrTyp(::CPUKerOpt{T}) where T<:AbstractFloat = Array{Complex{T}}

"""
    useCpu(opt::CPUKerOpt)

Does nothing. This function is a placeholder for consistency with the GPU version.

# Arguments
- `opt::CPUKerOpt`: The CPU kernel options object

# Returns
- `CPUKerOpt`: The same CPU kernel options object
"""
useCpu(opt::CPUKerOpt) = opt

"""
    useGpu(opt::CPUKerOpt)

Switch to GPU computation for the given CPU kernel options object.

Creates a new `GPUKerOpt` object with the same phase factor and integration order as the original `CPUKerOpt`, but with default thread and block counts for GPU computation. The adjoint mode flag is also preserved.

# Arguments
- `opt::CPUKerOpt`: The CPU kernel options object

# Returns
- `GPUKerOpt`: A new GPU kernel options object with:
  - Phase factor of `opt.frqPhz`
  - Integration order of `opt.intOrd`
  - Default thread and block counts for GPU computation
  - Adjoint mode flag from `opt`
  - Default CUDA backend
"""
useGpu(opt::CPUKerOpt{T}) where T<:AbstractFloat = GPUKerOpt{T}(opt.frqPhz, opt.intOrd, (128, 2, 1), (1, 128, 256), opt.adjMod, CUDABackend())

"""
    GPUKerOpt{T} <: GlaKerOpt{T}

Options for GPU computation of the Green function operator.

`GPUKerOpt` determines the parallelization strategy for GPU computation through its thread and block counts. The integration order determines the accuracy of the Green function computation for cells in contact. Higher values give better accuracy but require more computation. The phase factor allows for complex frequencies, which is useful for modeling dispersive media or for numerical stability.

# Fields
- `frqPhz::Number`: Multiplicative scaling factor allowing for complex frequencies
- `intOrd::Integer`: Gauss-Legendre integration order for cells in contact
- `numTrd::NTuple{3, Integer}`: Number of threads to use when running GPU kernels
- `numBlk::NTuple{3, Integer}`: Number of thread blocks to use when running GPU kernels
- `adjMod::Bool`: Adjoint mode flag (true if adjoint mode is enabled)
- `bckEnd::GPU`: Backend for GPU computation
"""
mutable struct GPUKerOpt{T<:AbstractFloat} <: GlaKerOpt{T}
    frqPhz::Number
    intOrd::Integer
    numTrd::NTuple{3, Integer}
    numBlk::NTuple{3, Integer}
    adjMod::Bool
    bckEnd::GPU
end

"""
    GPUKerOpt{T}()

Construct a GPUKerOpt of storage precision `T` with default values.

The default thread and block counts are chosen to provide good performance on
most NVIDIA GPUs. The default integration order of 48 keeps the singular
corrections near 5e-11. This is affordable because the weak integrals fold their
redundant terms.

# Returns
- `GPUKerOpt{T}`: A new GPU kernel options object with:
  - Phase factor of 1.0 + 0.0im
  - Integration order of 48
  - 128 threads per block
  - 256 blocks
  - Adjoint mode disabled
  - Default CUDA backend
"""
GPUKerOpt{T}() where T<:AbstractFloat = GPUKerOpt{T}(1.0+0.0im, 48, (128, 2, 1), (1, 128, 256), false, CUDABackend())

"""
    GPUKerOpt(frqPhz, intOrd, numTrd, numBlk, adjMod, bckEnd)
    GPUKerOpt()

Construct a GPUKerOpt of the default storage precision `dfltPrc`.
"""
GPUKerOpt(frqPhz::Number, intOrd::Integer, numTrd::NTuple{3,Integer}, numBlk::NTuple{3,Integer}, adjMod::Bool, bckEnd::GPU) = GPUKerOpt{dfltPrc}(frqPhz, intOrd, numTrd, numBlk, adjMod, bckEnd)
GPUKerOpt() = GPUKerOpt{dfltPrc}()

"""
    GPUKerOpt{T}(opt::GPUKerOpt)

Re-type a GPUKerOpt to storage precision `T`, keeping all field values.
"""
GPUKerOpt{T}(opt::GPUKerOpt) where T<:AbstractFloat = GPUKerOpt{T}(opt.frqPhz, opt.intOrd, opt.numTrd, opt.numBlk, opt.adjMod, opt.bckEnd)

"""
    frqPhz(opt::GPUKerOpt)

Get the phase factor from a GPU kernel options object.

# Arguments
- `opt::GPUKerOpt`: The GPU kernel options object

# Returns
- `Number`: The phase factor
"""
frqPhz(opt::GPUKerOpt) = opt.frqPhz

"""
    intOrd(opt::GPUKerOpt)

Get the integration order from a GPU kernel options object.

# Arguments
- `opt::GPUKerOpt`: The GPU kernel options object

# Returns
- `Integer`: The integration order
"""
intOrd(opt::GPUKerOpt) = opt.intOrd

"""
    adjMod(opt::GPUKerOpt)

Get the number of threads to use when running GPU kernels.

The thread count determines the parallelization strategy for GPU computation. Higher values may improve performance but require more GPU resources.
"""
numTrd(opt::GPUKerOpt) = opt.numTrd

"""
    numBlk(opt::GPUKerOpt) -> NTuple{3, Integer}

Get the number of thread blocks to use when running GPU kernels.

The block count determines the parallelization strategy for GPU computation. Higher values may improve performance but require more GPU resources.
"""
numBlk(opt::GPUKerOpt) = opt.numBlk

"""
    adjMod(opt::GPUKerOpt) -> Bool

# Arguments
- `opt::GPUKerOpt`: The GPU kernel options object

# Returns
- `Bool`: True if adjoint mode is enabled, false otherwise
"""
adjMod(opt::GPUKerOpt) = opt.adjMod

"""
    bckEnd(opt::GPUKerOpt)

Get the GPU backend from a GPU kernel options object.

# Arguments
- `opt::GPUKerOpt`: The GPU kernel options object

# Returns
- `GPU`: The GPU backend
"""
bckEnd(opt::GPUKerOpt) = opt.bckEnd

"""
    arrTyp(opt::GPUKerOpt{T})

Get the array type from a GPU kernel options object.

# Arguments
- `opt::GPUKerOpt{T}`: The GPU kernel options object

# Returns
- `Type`: CuArray{Complex{T}}
"""
arrTyp(::GPUKerOpt{T}) where T<:AbstractFloat = CuArray{Complex{T}}


"""
    useCpu(opt::GPUKerOpt)

Switch to CPU computation for the given GPU kernel options object.

Creates a new `CPUKerOpt` object with the same phase factor and integration order as the original `GPUKerOpt`, but with default values for CPU computation. The thread and block counts are ignored in this case. The adjoint mode flag is also preserved.

# Arguments
- `opt::GPUKerOpt`: The GPU kernel options object

# Returns
- `CPUKerOpt`: A new CPU kernel options object with:
  - Phase factor of `opt.frqPhz`
  - Integration order of `opt.intOrd`
  - Adjoint mode flag from `opt`
  - Default CPU backend
"""
useCpu(opt::GPUKerOpt{T}) where T<:AbstractFloat = CPUKerOpt{T}(opt.frqPhz, opt.intOrd, opt.adjMod, CPU())

"""
    useGpu(opt::GPUKerOpt)

Does nothing. This function is a placeholder for consistency with the CPU version.

# Arguments
- `opt::GPUKerOpt`: The GPU kernel options object

# Returns
- `GPUKerOpt`: The same GPU kernel options object
"""
useGpu(opt::GPUKerOpt) = opt

# Add serialization support for GlaKerOpt
function Serialization.serialize(io::IO, opt::CPUKerOpt{T}) where T<:AbstractFloat
    serialize(io, T)
    serialize(io, opt.frqPhz)
    serialize(io, opt.intOrd)
    serialize(io, opt.adjMod)
end

function Serialization.deserialize(io::IO, ::Type{<:CPUKerOpt})
    prc = deserialize(io)
    frqPhz = deserialize(io)
    intOrd = deserialize(io)
    adjMod = deserialize(io)
    return CPUKerOpt{prc}(frqPhz, intOrd, adjMod, CPU())
end

function Serialization.serialize(io::IO, opt::GPUKerOpt{T}) where T<:AbstractFloat
    serialize(io, T)
    serialize(io, opt.frqPhz)
    serialize(io, opt.intOrd)
    serialize(io, opt.numTrd)
    serialize(io, opt.numBlk)
    serialize(io, opt.adjMod)
end

function Serialization.deserialize(io::IO, ::Type{<:GPUKerOpt})
    prc = deserialize(io)
    frqPhz = deserialize(io)
    intOrd = deserialize(io)
    numTrd = deserialize(io)
    numBlk = deserialize(io)
    adjMod = deserialize(io)
    return GPUKerOpt{prc}(frqPhz, intOrd, numTrd, numBlk, adjMod, CUDABackend())
end
