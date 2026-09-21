using KernelAbstractions
using CUDA
using ..GilaTypes

"""
    GlaKerOpt{T<:AbstractFloat}

Abstract type for computational information that determines how the Green function
operator is computed. `T` is the real storage precision of the operator data
(`Complex{T}`); `genPrc` (a field of the concrete kernel options) is the generation
precision. Concrete implementations include `CPUKerOpt` for CPU computation and
`GPUKerOpt` for GPU computation.
"""
abstract type GlaKerOpt{T<:AbstractFloat} end

# the options are mutable, so the constructor's genPrc guard has to hold here too
function Base.setproperty!(opt::GlaKerOpt, fld::Symbol, val)
    fld === :genPrc && chkGenPrc(val)
    return setfield!(opt, fld, convert(fieldtype(typeof(opt), fld), val))
end

# genPrc = Float32 overflows the h_l recursion near l ≈ 27 and returns NaN (D14)
chkGenPrc(genPrc::Type{<:AbstractFloat}) = genPrc === Float32 &&
    throw(ArgumentError("genPrc = Float32 would return NaN: the contact h_l recursion overflows near l ≈ 27 (D14). Use Float64 (default) or Double64."))

"""
    CPUKerOpt{T} <: GlaKerOpt{T}

Options for CPU computation of the Green function operator.

`CPUKerOpt` determines the precision of Green function generation through `genPrc`.
The phase factor allows for complex frequencies, which is useful for modeling
dispersive media or for numerical stability.

# Fields
- `frqPhz::ComplexF64`: Multiplicative scaling factor allowing for complex frequencies
- `genPrc::Type{<:AbstractFloat}`: Generation precision (default `Float64`; `Float32` throws, it would return NaN)
- `qssApx::Bool`: Quasistatic approximation flag (true to generate the quasistatic kernel)
- `adjMod::Bool`: Adjoint mode flag (true if adjoint mode is enabled)
- `bckEnd::CPU`: Backend for CPU computation
"""
mutable struct CPUKerOpt{T<:AbstractFloat} <: GlaKerOpt{T}
    frqPhz::ComplexF64
    genPrc::Type{<:AbstractFloat}
    qssApx::Bool
    adjMod::Bool
    bckEnd::CPU
    function CPUKerOpt{T}(frqPhz::Number, genPrc::Type{<:AbstractFloat}, qssApx::Bool, adjMod::Bool, bckEnd::CPU) where T<:AbstractFloat
        chkGenPrc(genPrc)
        return new{T}(frqPhz, genPrc, qssApx, adjMod, bckEnd)
    end
end

"""
    CPUKerOpt{T}()

Construct a CPUKerOpt of storage precision `T` with default values.

# Returns
- `CPUKerOpt{T}`: A new CPU kernel options object with:
  - Phase factor of 1.0 + 0.0im
  - Generation precision of `Float64`
  - Quasistatic approximation disabled
  - Adjoint mode disabled
  - Default CPU backend
"""
CPUKerOpt{T}() where T<:AbstractFloat = CPUKerOpt{T}(1.0+0.0im, Float64, false, false, CPU())

"""
    CPUKerOpt(frqPhz, genPrc, qssApx, adjMod, bckEnd)
    CPUKerOpt()

Construct a CPUKerOpt of the default storage precision `dflPrc`.
"""
CPUKerOpt(frqPhz::Number, genPrc::Type{<:AbstractFloat}, qssApx::Bool, adjMod::Bool, bckEnd::CPU) = CPUKerOpt{dflPrc}(frqPhz, genPrc, qssApx, adjMod, bckEnd)
CPUKerOpt() = CPUKerOpt{dflPrc}()

"""
    CPUKerOpt{T}(opt::CPUKerOpt)

Re-type a CPUKerOpt to storage precision `T`, keeping all field values.
"""
CPUKerOpt{T}(opt::CPUKerOpt) where T<:AbstractFloat = CPUKerOpt{T}(opt.frqPhz, opt.genPrc, opt.qssApx, opt.adjMod, opt.bckEnd)

"""
    frqPhz(opt::CPUKerOpt) -> ComplexF64

Get the multiplicative scaling factor allowing for complex frequencies.

# Arguments
- `opt::CPUKerOpt`: The CPU kernel options object

# Returns
- `ComplexF64`: The phase factor
"""
frqPhz(opt::CPUKerOpt) = opt.frqPhz

"""
    genPrc(opt::CPUKerOpt)

Get the generation precision from a CPU kernel options object.

# Arguments
- `opt::CPUKerOpt`: The CPU kernel options object

# Returns
- `Type{<:AbstractFloat}`: The generation precision
"""
genPrc(opt::CPUKerOpt) = opt.genPrc

"""
    qssApx(opt::CPUKerOpt)

Get the quasistatic approximation flag from a CPU kernel options object.

# Arguments
- `opt::CPUKerOpt`: The CPU kernel options object

# Returns
- `Bool`: True if the quasistatic kernel is generated, false otherwise
"""
qssApx(opt::CPUKerOpt) = opt.qssApx

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

Internal, not exported.

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

Creates a new `GPUKerOpt` object with the same phase factor and generation precision as the original `CPUKerOpt`, but with default thread and block counts for GPU computation. The adjoint mode flag is also preserved.

# Arguments
- `opt::CPUKerOpt`: The CPU kernel options object

# Returns
- `GPUKerOpt`: A new GPU kernel options object with:
  - Phase factor of `opt.frqPhz`
  - Generation precision of `opt.genPrc`
  - Default thread and block counts for GPU computation
  - Adjoint mode flag from `opt`
  - Default CUDA backend
"""
useGpu(opt::CPUKerOpt{T}) where T<:AbstractFloat = GPUKerOpt{T}(opt.frqPhz, opt.genPrc, opt.qssApx, (128, 2, 1), (1, 128, 256), opt.adjMod, CUDABackend())

"""
    GPUKerOpt{T} <: GlaKerOpt{T}

Options for GPU computation of the Green function operator.

`GPUKerOpt` determines the parallelization strategy for GPU computation through its thread and block counts, and the precision of Green function generation through `genPrc`. The phase factor allows for complex frequencies, which is useful for modeling dispersive media or for numerical stability.

# Fields
- `frqPhz::ComplexF64`: Multiplicative scaling factor allowing for complex frequencies
- `genPrc::Type{<:AbstractFloat}`: Generation precision (default `Float64`; `Float32` throws, it would return NaN)
- `qssApx::Bool`: Quasistatic approximation flag (true to generate the quasistatic kernel)
- `numTrd::NTuple{3, Int}`: Number of threads to use when running GPU kernels
- `numBlk::NTuple{3, Int}`: Number of thread blocks to use when running GPU kernels
- `adjMod::Bool`: Adjoint mode flag (true if adjoint mode is enabled)
- `bckEnd::GPU`: Backend for GPU computation
"""
mutable struct GPUKerOpt{T<:AbstractFloat} <: GlaKerOpt{T}
    frqPhz::ComplexF64
    genPrc::Type{<:AbstractFloat}
    qssApx::Bool
    numTrd::NTuple{3, Int}
    numBlk::NTuple{3, Int}
    adjMod::Bool
    bckEnd::GPU
    function GPUKerOpt{T}(frqPhz::Number, genPrc::Type{<:AbstractFloat}, qssApx::Bool, numTrd::NTuple{3,Integer}, numBlk::NTuple{3,Integer}, adjMod::Bool, bckEnd::GPU) where T<:AbstractFloat
        chkGenPrc(genPrc)
        return new{T}(frqPhz, genPrc, qssApx, numTrd, numBlk, adjMod, bckEnd)
    end
end

"""
    GPUKerOpt{T}()

Construct a GPUKerOpt of storage precision `T` with default values.

The default thread and block counts are chosen to provide good performance on
most NVIDIA GPUs.

# Returns
- `GPUKerOpt{T}`: A new GPU kernel options object with:
  - Phase factor of 1.0 + 0.0im
  - Generation precision of `Float64`
  - Quasistatic approximation disabled
  - 128 threads per block
  - 256 blocks
  - Adjoint mode disabled
  - Default CUDA backend
"""
GPUKerOpt{T}() where T<:AbstractFloat = GPUKerOpt{T}(1.0+0.0im, Float64, false, (128, 2, 1), (1, 128, 256), false, CUDABackend())

"""
    GPUKerOpt(frqPhz, genPrc, qssApx, numTrd, numBlk, adjMod, bckEnd)
    GPUKerOpt()

Construct a GPUKerOpt of the default storage precision `dflPrc`.
"""
GPUKerOpt(frqPhz::Number, genPrc::Type{<:AbstractFloat}, qssApx::Bool, numTrd::NTuple{3,Integer}, numBlk::NTuple{3,Integer}, adjMod::Bool, bckEnd::GPU) = GPUKerOpt{dflPrc}(frqPhz, genPrc, qssApx, numTrd, numBlk, adjMod, bckEnd)
GPUKerOpt() = GPUKerOpt{dflPrc}()

"""
    GPUKerOpt{T}(opt::GPUKerOpt)

Re-type a GPUKerOpt to storage precision `T`, keeping all field values.
"""
GPUKerOpt{T}(opt::GPUKerOpt) where T<:AbstractFloat = GPUKerOpt{T}(opt.frqPhz, opt.genPrc, opt.qssApx, opt.numTrd, opt.numBlk, opt.adjMod, opt.bckEnd)

"""
    frqPhz(opt::GPUKerOpt)

Get the phase factor from a GPU kernel options object.

# Arguments
- `opt::GPUKerOpt`: The GPU kernel options object

# Returns
- `ComplexF64`: The phase factor
"""
frqPhz(opt::GPUKerOpt) = opt.frqPhz

"""
    genPrc(opt::GPUKerOpt)

Get the generation precision from a GPU kernel options object.

# Arguments
- `opt::GPUKerOpt`: The GPU kernel options object

# Returns
- `Type{<:AbstractFloat}`: The generation precision
"""
genPrc(opt::GPUKerOpt) = opt.genPrc

"""
    qssApx(opt::GPUKerOpt)

Get the quasistatic approximation flag from a GPU kernel options object.

# Arguments
- `opt::GPUKerOpt`: The GPU kernel options object

# Returns
- `Bool`: True if the quasistatic kernel is generated, false otherwise
"""
qssApx(opt::GPUKerOpt) = opt.qssApx

"""
    adjMod(opt::GPUKerOpt)

Get the number of threads to use when running GPU kernels.

The thread count determines the parallelization strategy for GPU computation. Higher values may improve performance but require more GPU resources.
"""
numTrd(opt::GPUKerOpt) = opt.numTrd

"""
    numBlk(opt::GPUKerOpt) -> NTuple{3, Int}

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

Internal, not exported.

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

Creates a new `CPUKerOpt` object with the same phase factor and generation precision as the original `GPUKerOpt`, but with default values for CPU computation. The thread and block counts are ignored in this case. The adjoint mode flag is also preserved.

# Arguments
- `opt::GPUKerOpt`: The GPU kernel options object

# Returns
- `CPUKerOpt`: A new CPU kernel options object with:
  - Phase factor of `opt.frqPhz`
  - Generation precision of `opt.genPrc`
  - Adjoint mode flag from `opt`
  - Default CPU backend
"""
useCpu(opt::GPUKerOpt{T}) where T<:AbstractFloat = CPUKerOpt{T}(opt.frqPhz, opt.genPrc, opt.qssApx, opt.adjMod, CPU())

"""
    useGpu(opt::GPUKerOpt)

Does nothing. This function is a placeholder for consistency with the CPU version.

# Arguments
- `opt::GPUKerOpt`: The GPU kernel options object

# Returns
- `GPUKerOpt`: The same GPU kernel options object
"""
useGpu(opt::GPUKerOpt) = opt
