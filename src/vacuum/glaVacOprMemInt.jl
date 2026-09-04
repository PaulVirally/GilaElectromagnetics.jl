#=
Conventions for the values returned by the weak functions. Small letters 
correspond to normal face directions; capital letters correspond to grid 
increment directions. 

Self    xx  yy  zz
        1   2   3


Edge    xxY xxZ yyX yyZ zzX zzY xy xz yz
        1   2   3   4   5   6   7  8  9     


Vertex  xx  yy  zz  xy  xz  yz 
        1   2   3   4   5   6

GilaWInt contains all necessary support functions for numerical integration of 
the electromagnetic Green function. This code is translated from DIRECTFN_E by 
Athanasios Polimeridis, and is distributed under the GNU LGPL.

Author: Sean Molesky

Reference: Polimeridis AG, Vipiana F, Mosig JR, Wilton DR. 
DIRECTFN: Fully numerical algorithms for high precision computation of singular 
integrals in Galerkin SIE methods. 
IEEE Transactions on Antennas and Propagation. 2013; 61(6):3112-22.

In what follows the word weak is used in reference to the fact that the scalar 
Green function surface integral is weakly singular: the integrand exhibits a 
singularity proportional to the inverse of the separation distance. The letters 
S, E and V refer, respectively, to integration over self-adjacent triangles, 
edge-adjacent triangles, and vertex-adjacent triangles. 

The article cited above contains useful error comparison plots for the number 
evaluation points considered. 
=#
using LinearAlgebra
using StaticArrays
#=
Returns the scalar (Helmholtz) Green function. The separation dstMag is assumed 
to be scaled by wavelength. 
=#
@inline function sclEgo_(dstMag::Number, frqPhz::Number)
    # cispi reduces the argument exactly, as in sclEgoN_ below
    return cispi(2 * dstMag * frqPhz) / (4 * π * dstMag * frqPhz^2)
end
function sclEgo(dstMag::Number, frqPhz::ComplexF64)
    if imag(frqPhz) == zero(real(typeof(frqPhz)))
        return sclEgo_(dstMag, real(frqPhz))
    end
    return sclEgo_(dstMag, frqPhz)
end
sclEgo(dstMag::Number, frqPhz::Number) = sclEgo_(dstMag, frqPhz)
#=
Returns the scalar (Helmholtz) Green function with the singularity removed. The 
separation distance dstMag is assumed to be scaled by the wavelength. The 
function is used in the included glaIntSup.jl code to improve the convergence of
all weakly singular integrals.
=#
@inline function sclEgoN_(dstMag::Number, frqPhz::Number)
    # Computes g = [exp(im z) - 1] / (4π dstMag frqPhz^2) with z = 2π dstMag frqPhz.
    # Note that expressing g in terms of a sinc improves numerical stability for
    # small z and for complex frqPhz. For very large imaginary parts, the sinc
    # stays within a few ulp of a 256 bit BigFloat reference.
    xPrd = dstMag * frqPhz
    return im * cispi(xPrd) * sinc(xPrd) / (2 * frqPhz)
end
function sclEgoN(dstMag::Number, frqPhz::ComplexF64)
    if imag(frqPhz) == zero(real(typeof(frqPhz)))
        return sclEgoN_(dstMag, real(frqPhz))
    end
    return sclEgoN_(dstMag, frqPhz)
end
sclEgoN(dstMag::Number, frqPhz::Number) = sclEgoN_(dstMag, frqPhz)
#=
Returns the three dimensional Euclidean norm of a vector. 
=#
@inline function dstMag(v1::T, v2::T, v3::T) where T <: AbstractFloat
    return sqrt(v1^2 + v2^2 + v3^2)
end
#=
Head function for integration over coincident square panels. The scl vector 
contains the characteristic lengths of a cuboid voxel relative to the 
wavelength. glQud1 is an array of Gauss-Legendre quadrature weights and 
positions. The cmpInf parameter determines the level of precision used for 
integral calculations. Namely, cmpInf.intOrd is used internally in all 
weakly singular integral computations. 
=#
function wekS(scl::NTuple{3,Number}, glQud1::AbstractMatrix{<:AbstractFloat}, 
    cmpInf::GlaKerOpt)

    # grdPts = Array{Float64}(undef, 3, 18)
    grdPts = MMatrix{3, 18, Float64}(undef)
    # weak self integrals for the three characteristic faces of a cuboid
    # dir = 1 -> xy face (z-nrm)   dir = 2 -> xz face (y-nrm)
    # dir = 3 -> yz face (x-nrm)
    #= For a cubic cell the three directions see the same grid points, so one
    evaluation is bitwise the value of all three. =#
    if cubScl(scl)
        val = wekSDir(1, scl, grdPts, glQud1, cmpInf)
        return [val + rSrfSlf(Float64(scl[2]), Float64(scl[3]), cmpInf);
        val + rSrfSlf(Float64(scl[1]), Float64(scl[3]), cmpInf);
        val + rSrfSlf(Float64(scl[1]), Float64(scl[2]), cmpInf)]
    end
    return [wekSDir(3, scl, grdPts, glQud1, cmpInf) +
    rSrfSlf(Float64(scl[2]), Float64(scl[3]), cmpInf);
    wekSDir(2, scl, grdPts, glQud1, cmpInf) +
    rSrfSlf(Float64(scl[1]), Float64(scl[3]), cmpInf);
    wekSDir(1, scl, grdPts, glQud1, cmpInf) +
    rSrfSlf(Float64(scl[1]), Float64(scl[2]), cmpInf)]
end
#=
The three characteristic faces of a cell coincide when its scales are equal. We
check this equality over Rational types to avoid floating point errors. The
panel integrands depend only on the separation |r - r'|, so every panel integral
is invariant under rigid motions of the pair, and a cubic cell presents grid
points related by the coordinate permutation in all three directions. The three
faces of a cubic cell have identical panel geometries, so we only need one
evaluation for all three.
=#
@inline cubScl(scl::NTuple{3,Number}) = scl[1] == scl[2] && scl[2] == scl[3]
#=
Weak self-integral of a particular face. The four self terms of the two diagonal
splits integrate congruent right triangles. The 180 degree rotation about the
face centre maps one triangle of a split onto the other, a reflection in a face
axis maps one split onto the other. Thus, only one evaluation is needed for all
four. On a square face the reflection in the other diagonal (note: needs to be
a square) maps the edge pair of one split onto the edge pair of the other, so
the second split's edge terms is the same as the first's. Both identities rest
on the kernel depending only on |r - r'| (see cubScl). They must be re-derived
before using a kernel that is not invariant under reflections in the cell axes,
such as an anisotropic background or an oblique lattice.
=#
function wekSDir(dir::Integer, scl::NTuple{3,Number},
    grdPts::AbstractMatrix{<:AbstractFloat}, glQud1::AbstractMatrix{<:AbstractFloat},
    cmpInf::GlaKerOpt)

    wekGrdPts!(dir, scl, grdPts)
    sSlf = wekSInt(hcat(grdPts[:,1], grdPts[:,2], grdPts[:,5]), glQud1, cmpInf)
    edgA = wekEInt(hcat(grdPts[:,1], grdPts[:,2], grdPts[:,5],
        grdPts[:,1], grdPts[:,5], grdPts[:,4]), glQud1, cmpInf)
    edgB = wekEInt(hcat(grdPts[:,1], grdPts[:,5], grdPts[:,4],
        grdPts[:,1], grdPts[:,2], grdPts[:,5]), glQud1, cmpInf)
    # square face: the two in-plane scales, compared as the Rationals they are
    if scl[mod1(5 - dir, 3)] == scl[mod1(6 - dir, 3)]
        return ((sSlf + sSlf) + edgA) + edgB
    end
    edgC = wekEInt(hcat(grdPts[:,4], grdPts[:,1], grdPts[:,2],
        grdPts[:,4], grdPts[:,2], grdPts[:,5]), glQud1, cmpInf)
    edgD = wekEInt(hcat(grdPts[:,4], grdPts[:,2], grdPts[:,5],
        grdPts[:,4], grdPts[:,1], grdPts[:,2]), glQud1, cmpInf)
    return ((((sSlf + sSlf) + edgA) + edgB) +
    (((sSlf + sSlf) + edgC) + edgD)) / 2.0
end
#=
Head function for integration over edge adjacent square panels. See wekS for 
input parameter descriptions. 
=#
function wekE(scl::NTuple{3,Number}, glQud1::AbstractMatrix{<:AbstractFloat}, 
    cmpInf::GlaKerOpt)
    
    grdPts = Array{Float64,2}(undef, 3, 18)
    # a cubic cell sees the same grid points in all three directions
    cubMod = cubScl(scl)
    # labels are panelDir-panelDir-gridIncrement
    vals = valsCub = wekEDir(3, scl, grdPts, glQud1, cmpInf)
    # lower case letters reference the normal directions of the rectangles
    # upper case letter reference the increasing axis direction when necessary 
    # first set
    xxY = vals[1] + rSrfEdgFlt(Float64(scl[3]), Float64(scl[2]), cmpInf)
    xxZ = vals[3] + rSrfEdgFlt(Float64(scl[2]), Float64(scl[3]), cmpInf)
    xyA = vals[2] + rSrfEdgCrn(Float64(scl[3]), Float64(scl[2]), 
        Float64(scl[1]), cmpInf)
    xzA = vals[4] + rSrfEdgCrn(Float64(scl[2]), Float64(scl[3]), 
        Float64(scl[1]), cmpInf)
    vals = cubMod ? valsCub : wekEDir(2, scl, grdPts, glQud1, cmpInf)
    # second set
    yyZ = vals[1] + rSrfEdgFlt(Float64(scl[1]), Float64(scl[3]), cmpInf)
    yyX = vals[3] + rSrfEdgFlt(Float64(scl[3]), Float64(scl[1]), cmpInf)
    yzA = vals[2] + rSrfEdgCrn(Float64(scl[1]), Float64(scl[3]), 
        Float64(scl[2]), cmpInf)
    xyB = vals[4] + rSrfEdgCrn(Float64(scl[3]), Float64(scl[2]), 
        Float64(scl[1]), cmpInf)
    vals = cubMod ? valsCub : wekEDir(1, scl, grdPts, glQud1, cmpInf)
    # third set
    zzX = vals[1] + rSrfEdgFlt(Float64(scl[2]), Float64(scl[1]), cmpInf)
    zzY = vals[3] + rSrfEdgFlt(Float64(scl[1]), Float64(scl[2]), cmpInf)
    xzB = vals[2] + rSrfEdgCrn(Float64(scl[2]), Float64(scl[3]), 
        Float64(scl[1]), cmpInf)
    yzB = vals[4] + rSrfEdgCrn(Float64(scl[1]), Float64(scl[3]), 
        Float64(scl[2]), cmpInf)
    return @SVector [xxY; xxZ; yyX; yyZ; zzX; zzY; (xyA + xyB) / 2.0; (xzA + xzB) / 2.0;
    (yzA + yzB) / 2.0]
end
#= 
Weak edge integrals for a given face as specified by dir.
    dir = 1 -> z face -> [y-edge (++ gridX): zz(x), xz(x);
                          x-edge (++ gridY) zz(y) yz(y)]

    dir = 2 -> y face -> [x-edge (++ gridZ): yy(z), yz(z); 
                          z-edge (++ gridX) yy(x) xy(x)]

    dir = 3 -> x face -> [z-edge (++ gridY): xx(y), xy(y); 
                          y-edge (++ gridZ) xx(z) xz(z)]
=#
function wekEDir(dir::Integer, scl::NTuple{3,Number}, 
    grdPts::AbstractMatrix{<:AbstractFloat}, glQud1::AbstractMatrix{<:AbstractFloat}, 
    cmpInf::GlaKerOpt)

    wekGrdPts!(dir, scl, grdPts) 
    return @SVector [wekEInt(hcat(grdPts[:,1], grdPts[:,2], grdPts[:,5],
    grdPts[:,2], grdPts[:,3], grdPts[:,5]), glQud1, cmpInf) +
    wekVInt(true, hcat(grdPts[:,1], grdPts[:,2], grdPts[:,5], 
    grdPts[:,3], grdPts[:,6], grdPts[:,5]), glQud1, cmpInf) +
    wekVInt(true, hcat(grdPts[:,1], grdPts[:,5], grdPts[:,4],
    grdPts[:,2], grdPts[:,3], grdPts[:,5]), glQud1, cmpInf) +
    wekVInt(true, hcat(grdPts[:,1], grdPts[:,5], grdPts[:,4], 
    grdPts[:,3], grdPts[:,6], grdPts[:,5]), glQud1, cmpInf);
    wekEInt(hcat(grdPts[:,1], grdPts[:,2], grdPts[:,5],
    grdPts[:,2], grdPts[:,11], grdPts[:,5]), glQud1, cmpInf) +
    wekVInt(true, hcat(grdPts[:,1], grdPts[:,2], grdPts[:,5], 
    grdPts[:,11], grdPts[:,14], grdPts[:,5]), glQud1, cmpInf) +
    wekVInt(true, hcat(grdPts[:,1], grdPts[:,5], grdPts[:,4],
    grdPts[:,2], grdPts[:,11], grdPts[:,5]), glQud1, cmpInf) +
    wekVInt(true, hcat(grdPts[:,1], grdPts[:,5], grdPts[:,4], 
    grdPts[:,11], grdPts[:,14], grdPts[:,5]), glQud1, cmpInf);
    wekVInt(true, hcat(grdPts[:,1], grdPts[:,2], grdPts[:,5], 
    grdPts[:,4], grdPts[:,5], grdPts[:,7]), glQud1, cmpInf) +
    wekVInt(true, hcat(grdPts[:,1], grdPts[:,2], grdPts[:,5], 
    grdPts[:,5], grdPts[:,8], grdPts[:,7]), glQud1, cmpInf) +
    wekEInt(hcat(grdPts[:,1], grdPts[:,5], grdPts[:,4], 
    grdPts[:,4], grdPts[:,5], grdPts[:,7]), glQud1, cmpInf) +
    wekVInt(true, hcat(grdPts[:,1], grdPts[:,5], grdPts[:,4], 
    grdPts[:,5], grdPts[:,8], grdPts[:,7]), glQud1, cmpInf);
    wekVInt(true, hcat(grdPts[:,1], grdPts[:,2], grdPts[:,5], 
    grdPts[:,4], grdPts[:,5], grdPts[:,13]), glQud1, cmpInf) +
    wekVInt(true, hcat(grdPts[:,1], grdPts[:,2], grdPts[:,5], 
    grdPts[:,5], grdPts[:,14], grdPts[:,13]), glQud1, cmpInf) +
    wekEInt(hcat(grdPts[:,1], grdPts[:,5], grdPts[:,4], 
    grdPts[:,4], grdPts[:,5], grdPts[:,13]), glQud1, cmpInf) +
    wekVInt(true, hcat(grdPts[:,1], grdPts[:,5], grdPts[:,4], 
    grdPts[:,5], grdPts[:,14], grdPts[:,13]), glQud1, cmpInf)]
end
#=
Head function returning integral values for the Ego function over vertex 
adjacent square panels. See wekS for input parameter descriptions. 
=#
function wekV(scl::NTuple{3,Number}, glQud1::AbstractMatrix{<:AbstractFloat}, 
    cmpInf::GlaKerOpt)

    # grdPts = Array{Float64,2}(undef,3,18)
    grdPts = MMatrix{3, 18, Float64}(undef)
    # a cubic cell sees the same grid points in all three directions
    cubMod = cubScl(scl)
    # vertex integrals for x-normal face
    vals = valsCub = wekVDir(3, scl, grdPts, glQud1, cmpInf)
    xxO = vals[1]
    xyA = vals[2]
    xzA = vals[3]
    # vertex integrals for y-normal face
    vals = cubMod ? valsCub : wekVDir(2, scl, grdPts, glQud1, cmpInf)
    yyO = vals[1]
    yzA = vals[2]
    xyB = vals[3]
    # vertex integrals for z-normal face
    vals = cubMod ? valsCub : wekVDir(1, scl, grdPts, glQud1, cmpInf)
    zzO = vals[1]
    xzB = vals[2]
    yzB = vals[3]
    return @SVector [xxO; yyO; zzO; (xyA + xyB) / 2.0; (xzA + xzB) / 2.0; 
    (yzA + yzB) / 2.0]
end
#= 
Weak edge integrals for a given face as specified by dir.
    dir = 1 -> z face -> [zz zx zy]
    dir = 2 -> y face -> [yy yz yx]
    dir = 3 -> x face -> [xx xy xz]
=#
function wekVDir(dir::Integer, scl::NTuple{3,Number}, 
    grdPts::AbstractMatrix{<:AbstractFloat}, glQud1::AbstractMatrix{<:AbstractFloat}, 
    cmpInf::GlaKerOpt)

    wekGrdPts!(dir, scl, grdPts) 
    return @SVector [wekVInt(false, hcat(grdPts[:,1], grdPts[:,2], grdPts[:,5], 
    grdPts[:,5], grdPts[:,6], grdPts[:,9]), glQud1, cmpInf) +
    wekVInt(false, hcat(grdPts[:,1], grdPts[:,2], grdPts[:,5], 
    grdPts[:,5], grdPts[:,9], grdPts[:,8]), glQud1, cmpInf) +
    wekVInt(false, hcat(grdPts[:,1], grdPts[:,5], grdPts[:,4], 
    grdPts[:,5], grdPts[:,6], grdPts[:,9]), glQud1, cmpInf) +
    wekVInt(false, hcat(grdPts[:,1], grdPts[:,5], grdPts[:,4], 
    grdPts[:,5], grdPts[:,9], grdPts[:,8]), glQud1, cmpInf);
    wekVInt(false, hcat(grdPts[:,1], grdPts[:,2], grdPts[:,5], 
    grdPts[:,5], grdPts[:,17], grdPts[:,14]), glQud1, cmpInf) +
    wekVInt(false, hcat(grdPts[:,1], grdPts[:,2], grdPts[:,5], 
    grdPts[:,5], grdPts[:,8], grdPts[:,17]), glQud1, cmpInf) +
    wekVInt(false, hcat(grdPts[:,1], grdPts[:,5], grdPts[:,4], 
    grdPts[:,5], grdPts[:,17], grdPts[:,14]), glQud1, cmpInf) +
    wekVInt(false, hcat(grdPts[:,1], grdPts[:,5], grdPts[:,4], 
    grdPts[:,5], grdPts[:,8], grdPts[:,17]), glQud1, cmpInf);
    wekVInt(false, hcat(grdPts[:,1], grdPts[:,2], grdPts[:,5], 
    grdPts[:,5], grdPts[:,15], grdPts[:,14]), glQud1, cmpInf) +
    wekVInt(false, hcat(grdPts[:,1], grdPts[:,2], grdPts[:,5], 
    grdPts[:,5], grdPts[:,6], grdPts[:,15]), glQud1, cmpInf) +
    wekVInt(false, hcat(grdPts[:,1], grdPts[:,5], grdPts[:,4], 
    grdPts[:,5], grdPts[:,15], grdPts[:,14]), glQud1, cmpInf) +
    wekVInt(false, hcat(grdPts[:,1], grdPts[:,5], grdPts[:,4], 
    grdPts[:,5], grdPts[:,6], grdPts[:,15]), glQud1, cmpInf)]
end
#=
Generate all unique pairs of cube faces. Row 1 is the target face and row 2 the
source face, the convention srfSum! assembles by: the tensor component (a, b)
sums the face pairs whose row 1 normal is a and row 2 normal is b.
=#
function facPar()
    fPairs = Array{Int,2}(undef, 2, 36)
    for i ∈ 1:6, j ∈ 1:6
        k = (i - 1) * 6 + j
        fPairs[1, k] = i
        fPairs[2, k] = j
    end
    return SMatrix{2, 36, Int}(fPairs)
end
#=
Determine scaling factors for surface integrals.
=#
function srfScl(sclT::NTuple{3,Number}, sclS::NTuple{3,Number})

    srcScl = 1.0
    trgScl = 1.0
    srfScl = Array{Float64,1}(undef, 36)
    
    for srcFId ∈ 1 : 6
        if srcFId == 1 || srcFId == 2           srcScl = sclS[2] * sclS[3]
        elseif srcFId == 3 || srcFId == 4       srcScl = sclS[1] * sclS[3]
        else                                    srcScl = sclS[1] * sclS[2]
        end
        #= The face pair key is (target face - 1) * 6 + source face, see the use
        of fPairs in cubVecAltAdp, so the source area belongs to the second
        index. Writing it transposed scales tensor component (a, b) by the ratio
        of the per-direction scale ratios, which is only visible when
        sclS ./ sclT varies with direction. =#
        for trgFId ∈ 1 : 6
            if trgFId == 1 || trgFId == 2       trgScl = sclT[1]
            elseif trgFId == 3 || trgFId == 4   trgScl = sclT[2]
            else                                trgScl = sclT[3]
            end
            srfScl[(trgFId - 1) * 6 + srcFId] = Float64(srcScl / trgScl)
        end
    end
    return SVector{36}(srfScl)
end
#=
Generate array of cuboid faces based from a characteristic size, l[]. 
L and U reference relative positions on the corresponding normal axis.
Points are number in a counter-clockwise convention when viewing the 
face from the exterior of the cube. 
=#
function cubFac(size::NTuple{3,Number})
    
    yzL = hcat([-size[1], -size[2], -size[3]], [-size[1], size[2], -size[3]], 
        [-size[1], size[2], size[3]], [-size[1], -size[2], size[3]]) ./ 2
    yzU = hcat([size[1], -size[2], -size[3]], [size[1], -size[2], size[3]], 
        [size[1], size[2], size[3]], [size[1], size[2], -size[3]]) ./ 2
    xzL = hcat([-size[1], -size[2], -size[3]], [-size[1], -size[2], size[3]], 
        [size[1], -size[2], size[3]], [size[1], -size[2], -size[3]]) ./ 2
    xzU = hcat([-size[1], size[2], -size[3]], [size[1], size[2], -size[3]], 
        [size[1], size[2], size[3]], [-size[1], size[2], size[3]]) ./ 2
    xyL = hcat([-size[1], -size[2], -size[3]], [size[1], -size[2], -size[3]], 
        [size[1], size[2], -size[3]], [-size[1], size[2], -size[3]]) ./ 2
    xyU = hcat([-size[1], -size[2], size[3]], [-size[1], size[2], size[3]], 
        [size[1], size[2], size[3]], [size[1], -size[2], size[3]]) ./ 2
    return SArray{Tuple{3,4,6}}(cat(yzL, yzU, xzL, xzU, xyL, xyU, dims = 3))
end
#=
Determine a directional component, set by dir, of the separation vector for a 
pair points, as determined by ord, which may take on values between zero and 
one. The first pair of entries are coordinates in the source surface, the 
second pair of entries are coordinates in the target surface. 
=#
@inline function cubVecAltAdp(dir::Integer, ordVec::AbstractVector{<:AbstractFloat}, 
    fp::Integer, trgFaces::AbstractArray{<:AbstractFloat,3}, 
    srcFaces::AbstractArray{<:AbstractFloat,3}, fPairs::AbstractMatrix{<:Integer})

    trgFId, srcFId = fPairs[:, fp]
    pst = trgFaces[dir, 1, trgFId] +
        ordVec[3] * (trgFaces[dir, 2, trgFId] - trgFaces[dir, 1, trgFId]) +
        ordVec[4] * (trgFaces[dir, 4, trgFId] - trgFaces[dir, 1, trgFId])

    ngt = srcFaces[dir, 1, srcFId] +
        ordVec[1] * (srcFaces[dir, 2, srcFId] - srcFaces[dir, 1, srcFId]) +
        ordVec[2] * (srcFaces[dir, 4, srcFId] - srcFaces[dir, 1, srcFId])

    return pst - ngt
end
#=
Create grid point system for calculation for calculation of weakly singular 
integrals. 
=#
function wekGrdPts!(dir::Integer, scl::NTuple{3,Number}, 
    grdPts::AbstractMatrix{<:AbstractFloat})

    if dir == 1
        # standard orientation
        gridX = Float64(scl[1])
        gridY = Float64(scl[2])
        gridZ = Float64(scl[3])
    elseif dir == 2
        # single coordinate rotation
        gridX = Float64(scl[3])
        gridY = Float64(scl[1])
        gridZ = Float64(scl[2])
    elseif dir == 3
        # double coordinate rotation
        gridX = Float64(scl[2]) 
        gridY = Float64(scl[3])
        gridZ = Float64(scl[1])
    else
        error("Invalid direction selection.")
    end
    grdPts[:,1] .=  @SVector [0.0;             0.0;            0.0]
    grdPts[:,2] .=  @SVector [gridX;           0.0;            0.0]
    grdPts[:,3] .=  @SVector [2.0 * gridX;     0.0;            0.0]
    grdPts[:,4] .=  @SVector [0.0;             gridY;          0.0]
    grdPts[:,5] .=  @SVector [gridX;           gridY;          0.0]
    grdPts[:,6] .=  @SVector [2.0 * gridX;     gridY;          0.0]
    grdPts[:,7] .=  @SVector [0.0;             2.0 * gridY;    0.0]
    grdPts[:,8] .=  @SVector [gridX;           2.0 * gridY;    0.0]
    grdPts[:,9] .=  @SVector [2.0 * gridX;     2.0 * gridY;    0.0]
    grdPts[:,10] .= @SVector [0.0;            0.0;            gridZ]
    grdPts[:,11] .= @SVector [gridX;          0.0;            gridZ]
    grdPts[:,12] .= @SVector [2.0 * gridX;    0.0;            gridZ]
    grdPts[:,13] .= @SVector [0.0;            gridY;          gridZ]
    grdPts[:,14] .= @SVector [gridX;          gridY;          gridZ]
    grdPts[:,15] .= @SVector [2.0 * gridX;    gridY;          gridZ]
    grdPts[:,16] .= @SVector [0.0;            2.0 * gridY;    gridZ]
    grdPts[:,17] .= @SVector [gridX;          2.0 * gridY;    gridZ]
    grdPts[:,18] .= @SVector [2.0 * gridX;    2.0 * gridY;    gridZ]
    return nothing
end
#=
The code contained in glaIntSup evaluates the integrands called by the wekS, 
wekE, and wekV head functions using a series of variable transformations 
and analytic integral evaluations---reducing the four dimensional surface 
integrals performed for ``standard'' cells to chains of one dimensional 
integrals. No comments are included in this low level code, which is simply a 
julia translation of DIRECTFN_E by Athanasios Polimeridis with added support for
multi-threading. For a complete description of the steps being performed see 
the article cited above and references included therein. 
=#
include("glaVacOprMemIntSup.jl")
#=
Direct evaluation of 1 / (4 * π * dstMag * frqPhz^2) integral for a square panel
with itself. la and lb are the edge lengths. Reducing the four dimensional
integral to the difference variables (u, v), with triangular densities
(la - u)(lb - v), gives 4 * [la * lb * M00 - lb * M10 - la * M01 + M11], the
moments M_ij of 1 / sqrt(u^2 + v^2) over the la by lb rectangle. Using
h * (la^2 + lb^2) = h^3 the asinh moments collapse to three terms. The cubic
difference is evaluated as p^3 - p^2 * (h^2 + h * q + q^2) / (h + q) so that no
digits are lost for slender panels.
=#
@inline function rSrfSlf(la::AbstractFloat, lb::AbstractFloat,
    cmpInf::GlaKerOpt)::ComplexF64

    prtMin, prtMax = minmax(la, lb)
    hypLen = hypot(prtMin, prtMax)
    # la^3 + lb^3 - hypLen^3, cancellation-free
    cubDif = prtMin^3 - prtMin^2 *
    (hypLen^2 + hypLen * prtMax + prtMax^2) / (hypLen + prtMax)
    return (cubDif + 3 * la^2 * lb * asinh(lb / la) +
    3 * la * lb^2 * asinh(la / lb)) / (6 * π * frqPhz(cmpInf)^2)
end
#=
Direct evaluation of 1 / (4 * π * dstMag * frqPhz^2) integral for a pair of
cornered edge panels. la, lb, and lc are the edge lengths, and la is assumed to
be common to both panels. Integrating by parts along the shared edge gives
int_0^la N0(u, lb, lc) du, N0 being the Newtonian potential of the box at its
corner, which evaluates to the form below. The lb^3 and lc^3 asinh terms cancel
against the leading part of the asinh(lc / hypAB) and asinh(lb / hypAC) terms
for la much smaller than lb and lc, so they are grouped as log1p differences;
each x - sqrt(x^2 + y^2) is likewise written as -y^2 / (x + sqrt(x^2 + y^2)).
The form is symmetric under lb <-> lc, both orderings being called by wekE.
=#
@inline function rSrfEdgCrn(la::AbstractFloat, lb::AbstractFloat,
    lc::AbstractFloat, cmpInf::GlaKerOpt)::ComplexF64

    hypAll = sqrt(la^2 + lb^2 + lc^2)
    hypAB = hypot(la, lb)
    hypAC = hypot(la, lc)
    hypBC = hypot(lb, lc)
    # asinh(lc / lb) - asinh(lc / hypAB) and its lb <-> lc mirror
    difB = log1p(la^2 * (lc / (hypAB + lb) +
    lc^2 / (hypAB * hypBC + lb * hypAll)) / (lb * (lc + hypAll)))
    difC = log1p(la^2 * (lb / (hypAC + lc) +
    lb^2 / (hypAC * hypBC + lc * hypAll)) / (lc * (lb + hypAll)))
    return ((lb^3 / 6) * difB + (lc^3 / 6) * difC +
    (la^2 * lb / 2) * asinh(lc / hypAB) +
    (la^2 * lc / 2) * asinh(lb / hypAC) +
    la * lb * lc * asinh(la / hypBC) -
    (la^3 / 6) * atan(lb * lc / (la * hypAll)) -
    (la * lb^2 / 2) * atan(la * lc / (lb * hypAll)) -
    (la * lc^2 / 2) * atan(la * lb / (lc * hypAll)) -
    (la^2 * lb * lc / 3) / (hypBC + hypAll)) / (2 * π * frqPhz(cmpInf)^2)
end
#=
Direct evaluation of 1 / (4 * π * dstMag * frqPhz^2) integral for a pair of flat
edge panels. la and lb are the edge lengths, and lb is assumed to be
``doubled''. In the difference variables the pair reduces to the moments M_ij of
1 / sqrt(u^2 + v^2) used by rSrfSlf, giving
la * M01(la, lb) - M11(la, lb) + g(2 * lb) - g(lb), with
g(q) = 2 * lb * la * M00(la, q) - 2 * lb * M10(la, q) - la * M01(la, q) +
M11(la, q). The polynomial part collapses to
2 * hypA^3 - hypB^3 - la^3 + 6 * lb^3, which cancels to O(la * lb^2) for slender
panels and to O(la^3) for short shared edges; it is rewritten as a manifestly
negative product, and the two asinh differences as log1p of positive ratios.
=#
@inline function rSrfEdgFlt(la::AbstractFloat, lb::AbstractFloat,
    cmpInf::GlaKerOpt)::ComplexF64

    hypA = hypot(la, lb)
    hypB = hypot(la, 2 * lb)
    hypSum = hypB + 2 * hypA
    # asinh(2 * lb / la) - asinh(lb / la)
    difA = log1p((lb + 3 * lb^2 / (hypA + hypB)) / (lb + hypA))
    # 2 * asinh(la / (2 * lb)) - asinh(la / lb)
    difB = log1p(la^3 * (la / (hypA + lb)^2 + 1 / (hypB + 2 * lb)) /
    (2 * lb * (la + hypA)))
    # 2 * hypA^3 - hypB^3 - la^3 + 6 * lb^3, cancellation-free
    plyPrt = -2 * la^3 * lb^2 *
    (3 * la / (hypSum * (hypA + lb) * (hypB + 2 * lb)) +
    (1 + 3 * la / hypSum) / ((hypA + la) * (hypB + la)))
    return (plyPrt + 6 * la^2 * lb * difA + 6 * la * lb^2 * difB) /
    (12 * π * frqPhz(cmpInf)^2)
end
