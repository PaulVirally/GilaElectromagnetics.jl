using KernelAbstractions

"""
    egoOpr!(egoMem::GlaVacOprMem, actVec::AbstractArray{<:Complex})::AbstractArray{<:Complex}

Apply the Green operator to a `(srcCel..., 3)` current density of eltype
`Complex{T}`, returning the `(trgCel..., 3)` field, or the `(trgCel..., 3nγ)`
derivatives of an operator of order `der ≥ 1`. Overwrites `actVec`.
"""
function egoOpr!(egoMem::GlaVacOprMem, actVec::AbstractArray{<:Complex})
    # srcCel counts the cells of one source partition, so the full input size is
    # the per-partition count times the number of partitions in each direction
    srcCelTot = egoMem.mixInf.srcCel .* egoMem.mixInf.srcDiv
    if (srcCelTot..., 3) != size(actVec)
        throw(ArgumentError("Size of input vector does not match size of source volume. Required data layout is (srcCelX, srcCelY, srcCelZ, 3)."))
    end
    return egoBrn!(egoMem, 0, 0, actVec)
end

# Support functions for operator action
include("glaVacActSup.jl")

function egoBrn!(egoMem::GlaVacOprMem, lvl::Integer, bId::Integer, actVec::AbstractArray{<:Complex})
    brnSze = egoMem.brnSze
    # number of source partitions
    parNumSrc = prod(egoMem.mixInf.srcDiv)
    # number of target partitions
    parNumTrg = prod(egoMem.mixInf.trgDiv)
    # if necessary, reshape source vector into partitions
    if lvl == 0 && sum(egoMem.mixInf.srcCel .!= size(actVec)[1:3]) > 0 
        # reshape current vector 
        # orgVec and actVec share the same underlying data
        orgVec = genPrt!(actVec, egoMem.mixInf)
    elseif lvl == 0
        orgVec = reshape(actVec, egoMem.mixInf.srcCel..., 3, parNumSrc)
    else
        # partitions have been created, simply rename memory
        orgVec = actVec
    end
    # forward FFT; the plans account for a vector that changes size
    if lvl > 0
        egoMem.fftPlnFwd[lvl] * orgVec
    end
    # split branch
    if lvl < 3
        # check if size of current active vector dimension is compatible 
        if size(orgVec)[lvl + 1] == brnSze[lvl + 1]
            prgVec = similar(orgVec)
            # split branch, includes phase operation
            sptBrn!(prgVec, lvl + 1, egoMem.phzInf[lvl+1], orgVec)
            # rename vectors for consistent convention with general branching
            prgVecEve = orgVec 
            prgVecOdd = prgVec
        # size of progeny vectors must be changed
        else
            # current and progeny vector sizes
            curSze = size(orgVec)
            prgSze = ntuple(x -> x == (lvl + 1) ? brnSze[x] : curSze[x], 3)
            # allocate progeny vectors
            prgVecEve = similar(orgVec, prgSze..., 3, parNumSrc)
            prgVecOdd = similar(orgVec, prgSze..., 3, parNumSrc)
            # memory previously associated with orgVec is freed internally
            sptBrn!(prgVecEve, prgVecOdd, lvl + 1, egoMem.phzInf[lvl+1],
                egoMem.mixInf, orgVec)
        end
    end
    # branch until depth of block structure
    if lvl < 3
        # the even branch, then the odd one. Sequential: tasks sharing an FFTW plan segfault on its scratch space
        eveVec = egoBrn!(egoMem, lvl + 1, bId, prgVecEve)
        oddVec = egoBrn!(egoMem, lvl + 1, bId + 2^(2 - lvl), prgVecOdd)
        # check if size of vector must change on merge
        if size(eveVec)[lvl + 1] == egoMem.mixInf.trgCel[lvl + 1]
            # merge branches, includes phase operation
			mrgBrn!(eveVec, lvl + 1, egoMem.phzInf[lvl + 1], oddVec)
            # rename eveVec to match return name used elsewhere
            retVec = eveVec
        # merge and resize
        else
            # size of current vector
            curSze = size(eveVec)
            # size of cartesian indices of merged vector
            mrgSze = ntuple(x -> x == (lvl + 1) ? egoMem.mixInf.trgCel[x] : 
                curSze[x], 3)
            # allocate memory for merge
            retVec = similar(eveVec, mrgSze..., size(eveVec, 4), parNumTrg)
            # merge branches, even and odd vectors are freed
			mrgBrn!(retVec, lvl + 1, egoMem.phzInf[lvl + 1], eveVec, oddVec)
        end
    else
        # 3 outputs per kernel of the 6 components each egoFur holds
        retVec = similar(orgVec, brnSze..., size(first(egoMem.egoFur), 4) ÷ 2, parNumTrg)
        fill!(retVec, zero(eltype(retVec)))
        # multiply by Toeplitz vector, sequentially: every source partition adds into the same target slice
        for trgItr in 1:parNumTrg 
            for srcItr in 1:parNumSrc 
                mulBrn!(egoMem, bId, selectdim(retVec, 5, trgItr),
                view(egoMem.egoFur[bId + 1], :, :, :, :, srcItr, trgItr), 
                selectdim(orgVec, 5, srcItr))
            end
        end
        # release orgVec ahead of the GC, a no-op on the CPU
        finalize(orgVec)
    end
    # inverse FFT
    lvl > 0 && egoMem.fftPlnRev[lvl] * retVec
    # zero level return
    if lvl == 0
        # check if there are partitions that need to be merged
        if parNumTrg == 1
            return reshape(retVec, egoMem.mixInf.trgCel..., :)
        else
            return mrgPrt!(egoMem.mixInf, retVec)
        end
    end
    return retVec
end

#= The stored Green function range, the vector range and the reflection sign of
one direction of one octant. Only half the coefficients are stored in a direction
where the branch is shorter than the source and target together. =#
function _octRng(mixInf::GlaExtInf, brnSze, egoSze, dir::Int, brnOdd::Bool, rev::Bool)
    if egoSze[dir] == brnSze[dir]
        rng = rev ? (1:0) : (1:brnSze[dir])
        return rng, rng, 1f0
    end
    trgCel, srcCel = mixInf.trgCel[dir], mixInf.srcCel[dir]
    if brnOdd
        begInd, endInd = 1, rev ? fld(srcCel, 2) : cld(trgCel, 2)
    else
        begInd, endInd = rev ? (2, fld(srcCel, 2) + 1 - iseven(trgCel)) : (1, cld(trgCel, 2) + iseven(trgCel))
    end
    vecRng = rev ? (brnSze[dir]:-1:(brnSze[dir] - endInd + begInd)) : (1:(1 + endInd - begInd))
    return begInd:endInd, vecRng, rev ? -1f0 : 1f0
end

# The Hadamard product of a branch with its Green function, forward and reverse in each direction
function mulBrn!(egoMem::GlaVacOprMem, bId::Integer, prdVec::AbstractArray{<:Complex, 4}, vecMod::AbstractArray{<:Complex, 4}, orgVec::AbstractArray{<:Complex, 4})
    egoSze = size(vecMod)[1:3]
    krn = mulKer!(get_backend(prdVec))
    gms = drvGam(egoMem.cmpInf.der)
    for oct in 0:7
        rngs = ntuple(dir -> _octRng(egoMem.mixInf, egoMem.brnSze, egoSze, dir, isodd(bId >> (3 - dir)), isodd(oct >> (dir - 1))), 3)
        modRng, vecRng, sgn = getindex.(rngs, 1), getindex.(rngs, 2), getindex.(rngs, 3)
        any(isempty, modRng) && continue
        # kernel g: outputs 3g - 2 to 3g, components 6g - 5 to 6g, flipped by a derivative along a reflected axis
        for (g, gam) in enumerate(gms)
            krn(view(prdVec, vecRng..., (3g - 2):3g), view(vecMod, modRng..., (6g - 5):6g), view(orgVec, vecRng..., :),
                (sgn[1] * sgn[2], sgn[1] * sgn[3], sgn[2] * sgn[3]), prod(sgn .^ gam); ndrange = length.(modRng))
        end
    end
    return nothing
end
@kernel function mulKer!(prdVec::AbstractArray{<:Complex, 4}, vecMod::AbstractArray{<:Complex, 4}, orgVec::AbstractArray{<:Complex, 4}, dirSym::NTuple{3, Float32}, sgn::Float32)
    itrX, itrY, itrZ = @index(Global, NTuple)

    # vecMod follows ii, jj, kk, ij, ik, jk storage format
    # i vector direction
    @inbounds prdVec[itrX, itrY, itrZ, 1] += sgn * (
                    vecMod[itrX, itrY, itrZ, 1] * orgVec[itrX, itrY, itrZ, 1] +
        dirSym[1] * vecMod[itrX, itrY, itrZ, 4] * orgVec[itrX, itrY, itrZ, 2] +
        dirSym[2] * vecMod[itrX, itrY, itrZ, 5] * orgVec[itrX, itrY, itrZ, 3])
    # j vector direction
    @inbounds prdVec[itrX, itrY, itrZ, 2] += sgn * (
        dirSym[1] * vecMod[itrX, itrY, itrZ, 4] * orgVec[itrX, itrY, itrZ, 1] +
                    vecMod[itrX, itrY, itrZ, 2] * orgVec[itrX, itrY, itrZ, 2] +
        dirSym[3] * vecMod[itrX, itrY, itrZ, 6] * orgVec[itrX, itrY, itrZ, 3])
    # k vector direction
    @inbounds prdVec[itrX, itrY, itrZ, 3] += sgn * (
        dirSym[2] * vecMod[itrX, itrY, itrZ, 5] * orgVec[itrX, itrY, itrZ, 1] +
        dirSym[3] * vecMod[itrX, itrY, itrZ, 6] * orgVec[itrX, itrY, itrZ, 2] +
                    vecMod[itrX, itrY, itrZ, 3] * orgVec[itrX, itrY, itrZ, 3])
end
