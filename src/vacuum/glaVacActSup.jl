using ..GilaVolumes

#=
restructure input vector to match partitioned Green function data
=#
function genPrt!(actVec::AbstractArray{ComplexF64, 4}, cmpInf::GlaKerOpt, mixInf::GlaExtInf, parNum::Integer)
    maxItr = mixInf.srcCel
    orgVec = similar(actVec, maxItr..., 3, parNum)
    ker = genPrtKer!(bckEnd(cmpInf))
    for parItr in 1:parNum, dirItr in 1:3
        stp = mixInf.srcDiv
        off = mixInf.srcPar[parItr]
        ker(stp, off, dirItr, parItr, actVec, orgVec; ndrange=maxItr)
    end
    memCln!(actVec)
    return orgVec
end
@kernel function genPrtKer!(stp::NTuple{3,Integer}, off::NTuple{3,Integer}, dirItr::Integer, parItr::Integer, actVec::AbstractArray{ComplexF64,4}, orgVec::AbstractArray{ComplexF64,5})
    # get the global linear index
    itr = @index(Global)

    # pull out the 3D dimensions of orgVec
    dims = size(orgVec)
    maxX, maxY = dims[1], dims[2]
    # (we don't actually need maxZ explicitly for the conversion)

    # convert to zero‑based and then to 3D coords
    idx = itr - 1
    itrX = (idx % maxX) + 1
    itrY = ((idx ÷ maxX) % maxY) + 1
    itrZ = (idx ÷ (maxX * maxY)) + 1

    @inbounds begin
        srcX = (itrX - 1) * stp[1] + off[1] + 1
        srcY = (itrY - 1) * stp[2] + off[2] + 1
        srcZ = (itrZ - 1) * stp[3] + off[3] + 1
        orgVec[itrX, itrY, itrZ, dirItr, parItr] = actVec[srcX, srcY, srcZ, dirItr]
    end
end

#=
split a branch so that even and odd Fourier coefficients are independent
=#
function sptBrn!(prgVec::AbstractArray{ComplexF64, 5}, sptDir::Integer, phzVec::AbstractVector{ComplexF64}, parNum::Integer, orgVec::AbstractArray{ComplexF64, 5}, cmpInf::GlaKerOpt)
    vecSze = size(orgVec)[1:3]
    ker = sptKer!(bckEnd(cmpInf))
    for parItr in 1:parNum
        ker(sptDir, phzVec, parItr, orgVec, prgVec; ndrange=vecSze)
    end
    return nothing
end
@kernel function sptKer!(sptDir::Integer, phzVec::AbstractVector{ComplexF64}, parItr::Integer, orgVec::AbstractArray{ComplexF64, 5}, prgVec::AbstractArray{ComplexF64, 5})
    # linear index
    itr = @index(Global)

    # dimensions
    dims = size(orgVec)
    maxX, maxY = dims[1], dims[2]

    # convert to 3D indices
    idx  = itr - 1
    itrX = (idx % maxX) + 1
    itrY = ((idx ÷ maxX) % maxY) + 1
    itrZ = (idx ÷ (maxX * maxY)) + 1

    @inbounds begin
        # pick the correct phase coefficient
        phz = sptDir == 1 ? phzVec[itrX] :
              sptDir == 2 ? phzVec[itrY] :
                            phzVec[itrZ]

        # apply it to all three Fourier‐branch slots
        prgVec[itrX, itrY, itrZ, 1, parItr] = phz * orgVec[itrX, itrY, itrZ, 1, parItr]
        prgVec[itrX, itrY, itrZ, 2, parItr] = phz * orgVec[itrX, itrY, itrZ, 2, parItr]
        prgVec[itrX, itrY, itrZ, 3, parItr] = phz * orgVec[itrX, itrY, itrZ, 3, parItr]
    end
end
#=
split branches for external Green function
=#
function sptBrn!(prgVecEve::AbstractArray{ComplexF64,5}, prgVecOdd::AbstractArray{ComplexF64,5}, dirSpt::Integer, phzVec::AbstractVector{ComplexF64}, mixInf::GlaExtInf, parNum::Integer, orgVec::AbstractArray{ComplexF64,5}, cmpInf::GlaKerOpt)
    brnSze = div.(mixInf.trgCel .+ mixInf.srcCel, 2)
    curSze = size(orgVec)[1:3]
    prgSze = ntuple(x -> x==dirSpt ? brnSze[x] : curSze[x], 3)

    ovrRng  = ntuple(i-> i==dirSpt ?
                     (1 : mixInf.srcCel[i]-brnSze[i]) :
                     (1 : prgSze[i]), 3)
    stdRng  = ntuple(i-> i==dirSpt ?
                     (last(ovrRng[i])+1 : min(mixInf.srcCel[i],brnSze[i])) :
                     (1 : prgSze[i]), 3)
    ovrOff  = ntuple(i-> i==dirSpt ? brnSze[dirSpt] : 0, 3)

    # Zero the progeny vectors
    fill!(prgVecEve, 0.0+0.0im)
    fill!(prgVecOdd, 0.0+0.0im)

    # Launch one kernel per (par,dir) over the entire prgSze
    ker = sptBrnExt!(bckEnd(cmpInf))
    for parItr in 1:parNum, dirItr in 1:3
      ker(ovrRng, stdRng, ovrOff, dirSpt, dirItr, parItr,
          phzVec, orgVec, prgVecEve, prgVecOdd;
          ndrange=prgSze)
    end
    return nothing
end
@kernel function sptBrnExt!(ovrRng::NTuple{3,UnitRange{Int}}, stdRng::NTuple{3,UnitRange{Int}}, ovrOff::NTuple{3,Int}, dirSpt::Int, dirItr::Int, parItr::Int, phzVec::AbstractVector{ComplexF64}, orgVec::AbstractArray{ComplexF64,5}, eveVec::AbstractArray{ComplexF64,5}, oddVec::AbstractArray{ComplexF64,5})
    # linear global index
    itr = @index(Global)

    # dims for 3D conversion
    dims = size(eveVec)
    maxX, maxY = dims[1], dims[2]

    # convert to 3D coords
    idx = itr - 1
    gX = (idx % maxX) + 1
    gY = ((idx ÷ maxX) % maxY) + 1
    gZ = (idx ÷ (maxX * maxY)) + 1

    @inbounds begin
        # pick the phase factor
        phz = dirSpt == 1 ? phzVec[gX] :
              dirSpt == 2 ? phzVec[gY] :
                            phzVec[gZ]

        # is it in the spill‑over region?
        if (gX in ovrRng[1]) && (gY in ovrRng[2]) && (gZ in ovrRng[3])
            # compute the other index
            sX = gX + ovrOff[1]
            sY = gY + ovrOff[2]
            sZ = gZ + ovrOff[3]

            # even
            eveVec[gX,gY,gZ,dirItr,parItr] =
            orgVec[gX,gY,gZ,dirItr,parItr] +
            orgVec[sX,sY,sZ,dirItr,parItr]

            # odd
            oddVec[gX,gY,gZ,dirItr,parItr] =
            phz * (orgVec[gX,gY,gZ,dirItr,parItr] -
                    orgVec[sX,sY,sZ,dirItr,parItr])

        # else if it's in the standard region
        elseif (gX in stdRng[1]) && (gY in stdRng[2]) && (gZ in stdRng[3])
            eveVec[gX,gY,gZ,dirItr,parItr] =
            orgVec[gX,gY,gZ,dirItr,parItr]

            oddVec[gX,gY,gZ,dirItr,parItr] =
            phz * orgVec[gX,gY,gZ,dirItr,parItr]
        # else: skip (they're already zero)
        end
    end
end
#=
merge two branches, eliminating unused coefficients
=#
function mrgBrn!(orgVec::AbstractArray{ComplexF64,5}, mrgDir::Integer, parNumTrg::Integer, phzVec::AbstractVector{ComplexF64}, prgVec::AbstractArray{ComplexF64,5}, cmpInf::GlaKerOpt)
    # launch one fused kernel per (partition, direction)
    vecSze = size(orgVec)[1:3]
    ker = mrgKer!(bckEnd(cmpInf))

    for prtItr in 1:parNumTrg, dirItr in 1:3
        ker(mrgDir, phzVec, dirItr, prtItr, orgVec, prgVec;
            ndrange = vecSze)
    end

    # free the progeny vector on GPU
    memCln!(prgVec)
    return nothing
end
@kernel function mrgKer!(mrgDir::Integer, phzVec::AbstractVector{ComplexF64}, dirItr::Integer, prtItr::Integer, orgVec::AbstractArray{ComplexF64,5}, prgVec::AbstractArray{ComplexF64,5})
    # linear index
    itr = @index(Global)

    # dims for conversion
    dims = size(orgVec)
    maxX, maxY = dims[1], dims[2]
    idx = itr - 1

    # decompose into 3D coords
    itrX = (idx % maxX) + 1
    itrY = ((idx ÷ maxX) % maxY) + 1
    itrZ = (idx ÷ (maxX * maxY)) + 1

    @inbounds begin
        # select phase factor
        ph = mrgDir == 1 ? phzVec[itrX] :
             mrgDir == 2 ? phzVec[itrY] :
                           phzVec[itrZ]

        # merge branches in place
        orgVec[itrX, itrY, itrZ, dirItr, prtItr] =
          0.5 * (
            orgVec[itrX, itrY, itrZ, dirItr, prtItr] +
            conj(ph) * prgVec[itrX, itrY, itrZ, dirItr, prtItr]
          )
    end
end
#=
generalized host merge allowing for different output size
=#
function mrgBrn!(mrgVec::AbstractArray{ComplexF64, 5}, mrgDir::Integer, phzVec::AbstractVector{ComplexF64}, eveVec::AbstractArray{ComplexF64, 5}, oddVec::AbstractArray{ComplexF64, 5})
	# size of current and merged vectors
	mrgSze = size(mrgVec)
	curSze = size(eveVec)
	# range for top half of merge vector
	topRng = ntuple(x -> x == mrgDir ? (1:min(mrgSze[x], curSze[x])) : 
		(1:mrgSze[x]), 3)
	# range for bottom half of merge vector
	botRng = ntuple(x -> x == mrgDir ? 
		(1:(mrgSze[x] - getproperty(topRng[x], :stop))) : (1:mrgSze[x]), 3)
	# offset for merge vector
	offSet = ntuple(x -> x == mrgDir ? 
		(getproperty(botRng[x], :stop) > 0 ? 
			getproperty(topRng[x], :stop) : 0) : 0, 5)
	# merge top range
    itr = CartesianIndices((topRng..., 1:3, 1:mrgSze[5]))
    phzItr = map(itr -> itr[mrgDir], itr)
    mrgVec[itr] .= 0.5 * (eveVec[itr] + conj(phzVec[phzItr]) .* oddVec[itr])
	# check if merge vector is filled
    itr = CartesianIndices((botRng..., 1:3, 1:mrgSze[5]))
    phzItr = map(itr -> itr[mrgDir], itr)
    mrgItr = map(itr -> itr + CartesianIndex(offSet), itr)
    @. mrgVec[mrgItr] = 0.5 * (eveVec[itr] - conj(phzVec[phzItr]) * oddVec[itr])
    memCln!(eveVec, oddVec)
	return nothing
end
function mrgBrn!(mrgVec::AbstractArray{ComplexF64,5}, mrgDir::Integer, parNumTrg::Integer, phzVec::AbstractVector{ComplexF64}, eveVec::AbstractArray{ComplexF64,5}, oddVec::AbstractArray{ComplexF64,5}, cmpInf::GlaKerOpt)
    # 3D sizes
    mrgSze3 = size(mrgVec)[1:3]
    curSze3 = size(eveVec)[1:3]

    # how much of the top comes from eve/odd
    topStop = min(mrgSze3[mrgDir], curSze3[mrgDir])
    offSet  = topStop

    # kernel launcher
    ker = mrgKerGnl!(bckEnd(cmpInf))

    # launch one kernel per (partition, direction)
    for prtItr in 1:parNumTrg, dirItr in 1:3
        ker(topStop, offSet,
            mrgDir, dirItr, prtItr,
            phzVec, eveVec, oddVec, mrgVec;
            ndrange = mrgSze3)
    end

    # free intermediate buffers
    memCln!(eveVec)
    memCln!(oddVec)
    return nothing
end
@kernel function mrgKerGnl!(topStop::Integer, offSet::Integer, mrgDir::Integer, dirItr::Integer, prtItr::Integer, phzVec::AbstractVector{ComplexF64}, eveVec::AbstractArray{ComplexF64,5}, oddVec::AbstractArray{ComplexF64,5}, mrgVec::AbstractArray{ComplexF64,5})
    # linear index of this work‑item
    itr = @index(Global)

    # 3D dims for the merged vector
    dims = size(mrgVec)
    maxX, maxY = dims[1], dims[2]

    # convert to 3D coords
    idx  = itr - 1
    itrX = (idx % maxX) + 1
    itrY = ((idx ÷ maxX) % maxY) + 1
    itrZ = (idx ÷ (maxX * maxY)) + 1

    # which coordinate along the merge axis?
    gCoord = mrgDir == 1 ? itrX :
             mrgDir == 2 ? itrY : itrZ

    # map global→input coords (for accessing eve/odd)
    inX = mrgDir == 1 ?
          (gCoord <= topStop ? itrX : itrX - offSet) : itrX
    inY = mrgDir == 2 ?
          (gCoord <= topStop ? itrY : itrY - offSet) : itrY
    inZ = mrgDir == 3 ?
          (gCoord <= topStop ? itrZ : itrZ - offSet) : itrZ

    # select the phase factor
    ph = mrgDir == 1 ? phzVec[inX] :
         mrgDir == 2 ? phzVec[inY] :
                       phzVec[inZ]

    @inbounds begin
        if gCoord <= topStop
            # top half merge
            mrgVec[itrX, itrY, itrZ, dirItr, prtItr] =
              0.5 * (
                eveVec[inX, inY, inZ, dirItr, prtItr] +
                conj(ph) * oddVec[inX, inY, inZ, dirItr, prtItr]
              )
        else
            # bottom half merge
            mrgVec[itrX, itrY, itrZ, dirItr, prtItr] =
              0.5 * (
                eveVec[inX, inY, inZ, dirItr, prtItr] -
                conj(ph) * oddVec[inX, inY, inZ, dirItr, prtItr]
              )
        end
    end
end
#=
merge partitions and return output vector 
=#
function mrgPrt!(mixInf::GlaExtInf, cmpInf::GlaKerOpt, parNum::Integer, prtVec::AbstractArray{ComplexF64, 5})
    @warn "This function should not be called"
    # restructure partitioned vector
    mrgVec = similar(prtVec, (mixInf.trgDiv .* mixInf.trgCel)..., 3)
    sncGpu(cmpInf)
    # perform partitioning
    ker = mrgPrtKer!(bckEnd(cmpInf))
    for parItr ∈ eachindex(1:parNum)
        for dirItr ∈ eachindex(1:3)
            ker(mixInf.trgDiv, mixInf.trgPar[parItr].I, dirItr, paritr, mrgVec, prtVec, ndrange=-1)
        end
    end
    # release active vector to reduce memory pressure
    memCln!(prtVec)
    return mrgVec
end
@kernel function mrgPrtKer!(stp::NTuple{3, Integer}, off::NTuple{3, Integer}, dirItr::Integer, parItr::Integer, mrgVec::AbstractArray{ComplexF64, 4}, prtVec::AbstractArray{ComplexF64, 5})
    # Linear index
    itr = @index(Global)

    # Convert linear to 3D indices
    idX = (itr - 1) % max[1] + 1
    idY = ((itr - 1) ÷ max[1]) % max[2] + 1
    idZ = (itr - 1) ÷ (max[1] * max[2]) + 1

    maxX, maxY, maxZ = size(prtVec)[1:3]
    stpX, stpY, stpZ = stp
    offX, offY, offZ = off

    @inbounds for itrZ = idZ:strZ:maxZ, itrY = idY:strY:maxY, itrX = idX:strX:maxX
        mrgVec[(itrX - 1) * stpX + offX + 1, 
            (itrY - 1) * stpY + offY + 1, 
            (itrZ - 1) * stpZ + offZ + 1, 
            dirItr] = prtVec[itrX, itrY, itrZ, dirItr, parItr]
    end
end

#=
binary branch indexing
=#
@inline function nxtBrnId(maxLvl::Integer, lvl::Integer, bId::Integer)::Integer
	return bId + ^(2, maxLvl - (lvl + 1))
end

# Calls CUDA.synchronize on GPU, and noop on CPU
sncGpu(::GPUKerOpt) = CUDA.synchronize(CUDA.stream())
sncGpu(::CPUKerOpt) = nothing

function memCln!(vec::CuArray)
    CUDA.synchronize(CUDA.stream())
    CUDA.unsafe_free!(vec)
    CUDA.synchronize(CUDA.stream())
    return nothing
end
function memCln!(vec1::CuArray, vec2::CuArray)
    CUDA.synchronize(CUDA.stream())
    CUDA.unsafe_free!(vec1)
    CUDA.unsafe_free!(vec2)
    CUDA.synchronize(CUDA.stream())
    return nothing
end
memCln!(::AbstractArray) = nothing
memCln!(::AbstractArray, ::AbstractArray) = nothing
