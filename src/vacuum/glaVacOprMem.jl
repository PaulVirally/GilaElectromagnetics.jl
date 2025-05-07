using Base.Threads
using AbstractFFTs
using FFTW
using CUDA
using LinearAlgebra
using Serialization
using ..GilaVolumes

"""
    GlaVacOprMem

Memory structure for the vacuum Green's function operator. This structure holds all the memory needed for computing the vacuum Green's function operator. The structure is designed to minimize memory allocation during computation. The Fourier transform plans are used to efficiently compute the Green's function integrals. The phase information is used to handle the splitting of Fourier transforms.

# Fields
- `cmpInf::GlaKerOpt`: Computation information, settings and kernel options, see `GlaKerOpt`
- `trgVol::GlaVol`: Target volume of Green function
- `srcVol::GlaVol`: Source volume of Green function
- `mixInf::GlaExtInf`: Information for matching source and target grids, see `GlaExtInf`
- `dimInf::NTuple{3,Integer}`: Dimension information for Green function volumes, host side
- `egoFur::AbstractVector{<:AbstractArray{ComplexF64}}`: Unique Fourier transform data for circulant Green function
- `fftPlnFwd::AbstractVector{<:AbstractFFTs.Plan}`: Forward Fourier transform plans
- `fftPlnRev::AbstractVector{<:AbstractFFTs.Plan}`: Reverse Fourier transform plans
- `adjFftPlnFwd::AbstractVector{<:AbstractFFTs.Plan}`: Forward Fourier transform plans for adjoint
- `adjFftPlnRev::AbstractVector{<:AbstractFFTs.Plan}`: Reverse Fourier transform plans for adjoint
- `phzInf::AbstractVector{<:AbstractArray{ComplexF64}}`: Phase vector for splitting Fourier transforms

# Notes
- This structure holds all the memory needed for computing the vacuum Green's function operator
- The structure is designed to minimize memory allocation during computation
- The Fourier transform plans are used to efficiently compute the Green's function integrals
- The phase information is used to handle the splitting of Fourier transforms
"""
mutable struct GlaVacOprMem
    cmpInf::GlaKerOpt
    trgVol::GlaVol
    srcVol::GlaVol
    mixInf::GlaExtInf
    dimInf::NTuple{3,Integer} 
    egoFur::AbstractVector{<:AbstractArray{ComplexF64}}
    fftPlnFwd::AbstractVector{<:AbstractFFTs.Plan}
    fftPlnRev::AbstractVector{<:AbstractFFTs.Plan}
    adjFftPlnFwd::AbstractVector{<:AbstractFFTs.Plan}
    adjFftPlnRev::AbstractVector{<:AbstractFFTs.Plan}
    phzInf::AbstractVector{<:AbstractArray{ComplexF64}}
end
#=
If intConTest.jl was failed the default intOrd used in the simplified constructor
may not be sufficient to insure that all integral values are properly converged.
It may be prudent to create the associated GlaVacOprMem with higher order. 
=#

"""
    GlaVacOprMem(cmpInf::GlaKerOpt, egoFur::AbstractVector{<:AbstractArray{ComplexF64}}, trgVol::GlaVol, srcVol::GlaVol=trgVol)

Prepare memory for Green's function operator. When called with a single GlaVol, 
or identical source and target volumes, yields the self construction. 

# Arguments
- `cmpInf::GlaKerOpt`: Computation information, settings and kernel options, see `GlaKerOpt`.
- `egoFur::AbstractVector{<:AbstractArray{ComplexF64}}`: Unique Fourier transform data for Green's function.
- `trgVol::GlaVol`: Target volume or self volume definition.
- `srcVol::Union{Nothing,GlaVol}=nothing`: Source volume for external construction. Nothing will generate the self construction.

# Returns
- `GlaVacOprMem`: The memory structure for the Green's function operator.
"""
function GlaVacOprMem(cmpInf::GlaKerOpt, egoFur::AbstractVector{<:AbstractArray{ComplexF64}}, trgVol::GlaVol, srcVol::GlaVol=trgVol)
    mixInf = genEveExtInf(trgVol, srcVol)
    # branching depth of multiplication
    lvl = 3
    # number of multiplication branches     
    eoDim = 2^lvl
    # verify that egoFur contains only numeric values
    for eoItr ∈ eachindex(1:eoDim)
        if !all(isfinite.(egoFur[eoItr]))
            throw(ArgumentError("Fourier information contains non-numeric values."))
        end
    end
    return glaOprPrp(egoFur, trgVol, srcVol, mixInf, cmpInf)
end

include("glaVacOprMemGen.jl") # For genEgoCrc!

"""
    GlaVacOprMem(cmpInf::GlaKerOpt, trgVol::GlaVol, srcVol::GlaVol=trgVol)

Prepare memory for Green's function operator. Automatically computes the Fourier transform data.

# Arguments
- `cmpInf::GlaKerOpt`: Computation information, settings and kernel options, see `GlaKerOpt`.
- `trgVol::GlaVol`: Target volume or self volume definition.
- `srcVol::Union{Nothing,GlaVol}=nothing`: Source volume for external construction. Nothing will generate the self construction.

# Returns
- `GlaVacOprMem`: The memory structure for the Green's function operator.
"""
function GlaVacOprMem(cmpInf::GlaKerOpt, trgVol::GlaVol, srcVol::GlaVol=trgVol)
    mixInf = genEveExtInf(trgVol, srcVol)

    # total cells in circulant
    totCelCrc = mixInf.trgCel .+ mixInf.srcCel
    # total number of target and source partitions
    totParTrg = prod(mixInf.trgDiv)
    totParSrc = prod(mixInf.srcDiv)

    # memory for circulant green function vector
    egoCrc = Array{ComplexF64}(undef, 3, 3, totCelCrc..., totParSrc, totParTrg) # FIXME: Should this be a CuArray if we have access to a GPU?
    genEgoCrc!(egoCrc, trgVol, srcVol, mixInf, cmpInf)
    # verify that egoCrc contains numeric values
    if !all(isfinite.(egoCrc))
        throw(ArgumentError("Computed circulant contains non-numeric values."))
    end
    # Fourier transform of circulant green function
    egoFurPrp = Array{eltype(egoCrc)}(undef, totCelCrc..., 6, totParSrc, totParTrg)
    # plan Fourier transform
    fftCrcOut = plan_fft(egoCrc[1,1,:,:,:,1,1], (1, 2, 3))
    # Fourier transform of the green function, making use of real space 
    # symmetry under transposition--entries are xx, yy, zz, xy, xz, yz
    for trgItr ∈ eachindex(1:totParTrg), srcItr ∈ eachindex(1:totParSrc), colItr ∈ eachindex(1:3), rowItr ∈ eachindex(1:colItr)
        # vector direction moved to outer volume index---largest stride
        egoFurPrp[:, :, :, blkEgoItr(3 * (colItr - 1) + rowItr), srcItr, trgItr] = fftCrcOut * egoCrc[rowItr, colItr, :, :, :, srcItr, trgItr]
    end
    # verify integrity of Fourier transform data
    if !all(isfinite.(egoFurPrp))
        throw(ArgumentError("Fourier transform of circulant contains non-numeric values."))
    end
    # number of unique green function blocks
    ddDim = 6
    # number of unique elements in each cartesian index for a branch
    truInf = Array{Int}(undef, 3)
    for dirItr ∈ eachindex(1:3)
        # row and column entries are symmetric or anti-symmetric
        if mixInf.trgCel[dirItr] == mixInf.srcCel[dirItr] && all(mixInf.srcDiv .== 1) && all(mixInf.trgDiv .== 1) && trgVol.org[dirItr] == srcVol.org[dirItr]
            # store only necessary information
            truInf[dirItr] = max(Integer(ceil(mixInf.trgCel[dirItr] / 2)) + iseven(mixInf.trgCel[dirItr]), 2)
            continue
        end
        # genVolEve enforces that number of cells is even 
        truInf[dirItr] = totCelCrc[dirItr] ÷ 2
    end
    # branching depth of multiplication
    lvl = 3
    # number of multiplication branches     
    eoDim = 2^lvl
    # final Fourier coefficients for a given branch
    egoFur = Array{arrTyp(cmpInf)}(undef, eoDim)
    # intermediate storage
    egoFurInt = Array{ComplexF64}(undef, max.(div.(totCelCrc, 2), (2,2,2))..., 
        ddDim, totParSrc, totParTrg)
    # only one one eighth of the green function is unique 
    for eoItr ∈ 0:(eoDim - 1)
        # odd / even branch extraction
        # egoFur[eoItr + 1] = Array{ComplexF64}(undef, truInf..., ddDim, totParSrc, totParTrg)
        egoFur[eoItr + 1] = arrTyp(cmpInf)(undef, truInf..., ddDim, totParSrc, totParTrg)
        # first division is along smallest stride -> largest binary division
        egoFurInt .= ComplexF64.(egoFurPrp[(1 + 
            mod(div(eoItr, 4), 2)):2:(end - 1 + mod(div(eoItr, 4), 2)), 
            (1 + mod(div(eoItr, 2), 2)):2:(end - 1 + mod(div(eoItr, 2), 2)),
            (1 + mod(eoItr, 2)):2:(end - 1 + mod(eoItr, 2)),:,:,:])
        itr = CartesianIndices(egoFur[eoItr + 1])
        copyto!(egoFur[eoItr + 1], itr, egoFurInt, itr)
    end
    return GlaVacOprMem(cmpInf, egoFur, trgVol, srcVol)
end

# Create Fourier transform plans
function fftPlnGen(fwdSze::NTuple, revSze::NTuple, dir::Int, ::CPUKerOpt)
    # Fourier transform planning area
    fftWrkFwd = Array{ComplexF64}(undef, fwdSze...)
    fftWrkRev = Array{ComplexF64}(undef, revSze...)
    # create Fourier transform plans
    fftPlnFwd = plan_fft!(fftWrkFwd, [dir]; flags=FFTW.MEASURE)
    fftPlnRev = plan_ifft!(fftWrkRev, [dir]; flags=FFTW.MEASURE)
    adjFftPlnFwd = plan_fft!(fftWrkRev, [dir]; flags=FFTW.MEASURE)
    adjFftPlnRev = plan_ifft!(fftWrkFwd, [dir]; flags=FFTW.MEASURE)
    return fftPlnFwd, fftPlnRev, adjFftPlnFwd, adjFftPlnRev
end

function fftPlnGen(fwdSze::NTuple, revSze::NTuple, dir::Int, ::GPUKerOpt)
    # Fourier transform planning area
    fftWrkFwdDev = CuArray{ComplexF64}(undef, fwdSze...)
    fftWrkRevDev = CuArray{ComplexF64}(undef, revSze...)
    # create Fourier transform plans
    fftPlnFwdDev = plan_fft!(fftWrkFwdDev, [dir])
    fftPlnRevDev = plan_ifft!(fftWrkRevDev, [dir])
    adjFftPlnFwdDev = plan_fft!(fftWrkRevDev, [dir])
    adjFftPlnRevDev = plan_ifft!(fftWrkFwdDev, [dir])
    return fftPlnFwdDev, fftPlnRevDev, adjFftPlnFwdDev, adjFftPlnRevDev
end

# Memory preparation sub-protocol
function glaOprPrp(egoFur::AbstractVector{<:AbstractArray{ComplexF64}}, trgVol::GlaVol, srcVol::GlaVol, mixInf::GlaExtInf, cmpInf::GlaKerOpt)
    # number of embedding levels---dimensionality of ambient space
    lvls = 3
    # operator dimensions---unique vector information does not typically 
    # match operator size for distinct source and target volumes
    # sum of source and target volumes being divisible by 2 is guaranteed by 
    # genVolEve in GlaVacOprMem
    brnSze = div.(mixInf.trgCel .+ mixInf.srcCel, 2)
    phzInf = Array{arrTyp(cmpInf)}(undef, lvls)
    # Fourier transform plans
    fftPlnFwd = Array{AbstractFFTs.Plan}(undef, lvls)
    fftPlnRev = Array{AbstractFFTs.Plan}(undef, lvls)
    adjFftPlnFwd = Array{AbstractFFTs.Plan}(undef, lvls)
    adjFftPlnRev = Array{AbstractFFTs.Plan}(undef, lvls)

    # initialize Fourier transform plans
    for dir ∈ eachindex(1:lvls)
        # size of vector changes throughout application for external Green 
        vecSzeFwd = ntuple(x -> x <= dir ? brnSze[x] : mixInf.srcCel[x], 3)
        vecSzeRev = ntuple(x -> x > dir ? mixInf.trgCel[x] : brnSze[x], 3)
        fwdSze = (vecSzeFwd..., lvls, prod(mixInf.srcDiv))
        revSze = (vecSzeRev..., lvls, prod(mixInf.trgDiv))
        fftPlnFwd[dir], fftPlnRev[dir], adjFftPlnFwd[dir], adjFftPlnRev[dir] = fftPlnGen(fwdSze, revSze, dir, cmpInf)
    end
    # phase transformations (internal for block Toeplitz transformations)
    for itr ∈ eachindex(1:lvls)
        # allows calculation odd coefficient numbers
        phzInfHst = ComplexF64.([cispi(-k / brnSze[itr]) for k ∈ 0:(brnSze[itr] - 1)])
        phzInf[itr] = similar(first(egoFur), brnSze[itr])
        copyto!(phzInf[itr], phzInfHst)
    end
    return GlaVacOprMem(cmpInf, trgVol, srcVol, mixInf, brnSze, egoFur, fftPlnFwd, fftPlnRev, adjFftPlnFwd, adjFftPlnRev, phzInf)
end

# Block index for a given Cartesian index.
@inline function blkEgoItr(crtInd::Integer)
    if crtInd == 1
        return 1
    elseif crtInd == 2 || crtInd == 4
        return 4
    elseif crtInd == 5 
        return 2    
    elseif crtInd == 7 || crtInd == 3
        return 5
    elseif crtInd == 8 || crtInd == 6
        return 6
    elseif crtInd == 9 
        return 3
    end
    throw(ArgumentError("Improper use case, there are only nine blocks."))
end

isadjoint(vacOprMem::GlaVacOprMem) = adjMod(vacOprMem.cmpInf)

function useCpu!(mem::GlaVacOprMem)
    if mem.cmpInf isa CPUKerOpt
        return
    end

    # Convert to CPU
    mem.egoFur = collect(map(Array, mem.egoFur))
    mem.cmpInf = useCpu(mem.cmpInf)
    fwdPln = Array{AbstractFFTs.Plan}(undef, 3)
    revPln = Array{AbstractFFTs.Plan}(undef, 3)
    ajdFwdPln = Array{AbstractFFTs.Plan}(undef, 3)
    ajdRevPln = Array{AbstractFFTs.Plan}(undef, 3)
    for dir in 1:3
        pln = fftPlnGen(size(mem.fftPlnFwd[dir]), size(mem.fftPlnRev[dir]), dir, CPUKerOpt())
        fwdPln[dir] = pln[1]
        revPln[dir] = pln[2]
        ajdFwdPln[dir] = pln[3]
        ajdRevPln[dir] = pln[4]
    end
    mem.fftPlnFwd = fwdPln
    mem.fftPlnRev = revPln
    mem.adjFftPlnFwd = ajdFwdPln
    mem.adjFftPlnRev = ajdRevPln
    mem.phzInf = collect(map(Array, mem.phzInf))

    return mem
end

function useGpu!(mem::GlaVacOprMem)
    if mem.cmpInf isa GPUKerOpt
        return
    end

    # Convert to GPU
    mem.egoFur = collect(map(CuArray, mem.egoFur))
    mem.cmpInf = useGpu(mem.cmpInf)
    fwdPln = Array{AbstractFFTs.Plan}(undef, 3)
    revPln = Array{AbstractFFTs.Plan}(undef, 3)
    ajdFwdPln = Array{AbstractFFTs.Plan}(undef, 3)
    ajdRevPln = Array{AbstractFFTs.Plan}(undef, 3)
    for dir in 1:3
        pln = fftPlnGen(size(mem.fftPlnFwd[dir]), size(mem.fftPlnRev[dir]), dir, GPUKerOpt())
        fwdPln[dir] = pln[1]
        revPln[dir] = pln[2]
        ajdFwdPln[dir] = pln[3]
        ajdRevPln[dir] = pln[4]
    end
    mem.fftPlnFwd = fwdPln
    mem.fftPlnRev = revPln
    mem.adjFftPlnFwd = ajdFwdPln
    mem.adjFftPlnRev = ajdRevPln
    mem.phzInf = collect(map(CuArray, mem.phzInf))

    return mem
end

# Add serialization support for GlaVacOprMem
function Serialization.serialize(io::IO, mem::GlaVacOprMem)
    wasGpu = false
    egoFur = mem.egoFur
    if mem.cmpInf isa GPUKerOpt
        wasGpu = true
        useCpu!(mem) # Convert to CPU for serialization
        egoFur = collect(map(Array, mem.egoFur))
    end
    serialize(io, egoFur)
    serialize(io, mem.cmpInf)
    serialize(io, mem.trgVol)
    serialize(io, mem.srcVol)
    serialize(io, mem.mixInf)

    if wasGpu
        useGpu!(mem) # Convert back to GPU after serialization
    end
end

function Serialization.deserialize(io::IO, ::Type{GlaVacOprMem})
    egoFur = deserialize(io)
    cmpInf = deserialize(io, CPUKerOpt)
    trgVol = deserialize(io)
    srcVol = deserialize(io)
    mixInf = deserialize(io)
     
    # Reconstruct the full operator
    return glaOprPrp(egoFur, trgVol, srcVol, mixInf, cmpInf)
end
