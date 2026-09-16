#=
Accuracy study of the Float32 storage precision and of MixPrcRfn. Usage:

    julia --project=. -t auto benchmark/prcAcc.jl [--big]

    --big : add the (16,16,16) cells, which cost minutes of quadrature each

For every volume size and susceptibility it prints the truncation error of the
fp32 operator against the fp64 one it was rounded from, and then, for the three
routes through an (I - XG₀) solve, the error of the solution against the fp64
reference, the residual measured with the fp64 operator, and the matrix-vector
count. The three routes are fp64 -> fp64 (the reference), fp64 -> fp32 iterative
refinement, and fp32 -> fp32 (the control, which shows the floor refinement is
there to beat). Runs on the GPU as well when one is available.

This script prints, it asserts nothing, and CI does not run it.
=#
include("bmkEnv.jl")

using CUDA
using GilaElectromagnetics
using LinearAlgebra
using Printf
using Random

import GilaElectromagnetics: isgpu, isadjoint, glaSze
import GilaElectromagnetics.GilaOperators: mulAct!
import GilaElectromagnetics.GilaVacuum: arrTyp

const bigRun = "--big" ∈ ARGS
for arg ∈ ARGS
    arg == "--big" || error("unknown argument: $arg (expected --big)")
end

const celScl = (1//32, 1//32, 1//32)
const celLst = bigRun ? [(4,4,4), (8,8,8), (16,16,16)] : [(4,4,4), (8,8,8)]
const susLst = [0.5 + 0.05im, 3.0 + 0.1im, 10.0 + 1.0im]
const slvTgt = 1e-10 # Requested relative tolerance, tight enough to expose the fp32 floor
const slvItr = 2000 # A stagnating fp32 solve must not run to the default 5000
const mvcNum = 5 # Random vectors behind the matrix-free operator error

#= Counts the matrix-vector products of the operator it wraps. The precision
conversion carries a fresh counter, so a MixPrcRfn built on a wrapped operator
leaves its own inner count in the cached copy, last(slv.cch[]). =#
struct CntOpr{T<:AbstractFloat} <: AbstractGlaOpr{T}
    opr::AbstractGlaOpr{T}
    cnt::Base.RefValue{Int}
end

CntOpr(opr::AbstractGlaOpr{T}) where T<:AbstractFloat = CntOpr{T}(opr, Ref(0))
CntOpr{Tlo}(cnt::CntOpr) where Tlo<:AbstractFloat =
    CntOpr(Base.typename(typeof(cnt.opr)).wrapper{Tlo}(cnt.opr))

function mulAct!(cnt::CntOpr{T}, act::AbstractVector{Complex{T}}) where T<:AbstractFloat
    cnt.cnt[] += 1
    return mulAct!(cnt.opr, act)
end
Base.size(cnt::CntOpr) = size(cnt.opr)
glaSze(cnt::CntOpr) = glaSze(cnt.opr)
arrTyp(cnt::CntOpr) = arrTyp(cnt.opr)
isgpu(cnt::CntOpr) = isgpu(cnt.opr)
isadjoint(cnt::CntOpr) = isadjoint(cnt.opr)

gmrSlv() = GMRESSolver(nothing, slvItr, nothing, slvTgt)
vecErr(act, ref) = norm(ComplexF64.(act) .- ComplexF64.(ref)) / norm(ComplexF64.(ref))

# Dense form, only affordable for the smallest volume
function dnsMat(opr::AbstractGlaOpr)
    n = size(opr, 2)
    mat = zeros(eltype(opr), size(opr, 1), n)
    for i in 1:n
        v = zeros(eltype(opr), n)
        v[i] = one(eltype(opr))
        mat[:, i] .= opr * v
    end
    return mat
end

# Truncation error of the fp32 operator: spectral for a small volume, and the
# worst of a few random matrix-vector products otherwise
function oprErr(opr32, opr64, dns::Bool)
    dns && return opnorm(ComplexF64.(dnsMat(opr32)) .- dnsMat(opr64)) / opnorm(dnsMat(opr64))
    n = size(opr64, 2)
    return maximum(1:mvcNum) do _
        v64 = randn(ComplexF64, n)
        v64 ./= norm(v64)
        v32 = ComplexF32.(v64)
        isgpu(opr64) && ((v64, v32) = (CuArray(v64), CuArray(v32)))
        vecErr(opr32 * v32, opr64 * v64)
    end
end

function runCel(cel, sus, useGpu::Bool)
    vol = GlaVol(cel, celScl, (0//1, 0//1, 0//1))
    g64 = GlaOprVac{Float64}(vol; useGpu=useGpu)
    sus64 = fill(ComplexF64(sus), cel...)
    sus32 = ComplexF32.(sus64)
    useGpu && ((sus64, sus32) = (CuArray(sus64), CuArray(sus32)))
    inv64 = InvSctOpr(g64, sus64)
    inv32 = InvSctOpr(GlaOprVac{Float32}(g64), sus32)

    b64 = randn(ComplexF64, size(inv64, 2))
    b64 ./= norm(b64)
    b32 = ComplexF32.(b64)
    useGpu && ((b64, b32) = (CuArray(b64), CuArray(b32)))
    res64(x) = norm(inv64 * ComplexF64.(x) .- b64) / norm(b64)

    err = oprErr(inv32, inv64, prod(cel) <= 64 && !useGpu)

    cntRef = CntOpr(inv64)
    solRef = solve(cntRef, copy(b64), gmrSlv())

    cntIr = CntOpr(inv64)
    rfn = MixPrcRfn(Float32; relTol=slvTgt)
    solIr = solve(cntIr, copy(b64), rfn)

    cnt32 = CntOpr(inv32)
    sol32 = solve(cnt32, copy(b32), gmrSlv())

    @printf("%-9s %-14s %9.2e  %-12s %9.2e %10.2e  %s\n", join(cel, "x"),
        string(sus), err, "fp64->fp64", 0.0, res64(solRef), "$(cntRef.cnt[]) mv")
    @printf("%-9s %-14s %9s  %-12s %9.2e %10.2e  %s\n", "", "", "", "fp64->fp32",
        vecErr(solIr, solRef), res64(solIr),
        "$(cntIr.cnt[]) outer / $(last(rfn.cch[]).cnt[]) inner fp32 mv")
    @printf("%-9s %-14s %9s  %-12s %9.2e %10.2e  %s\n", "", "", "", "fp32->fp32",
        vecErr(sol32, solRef), res64(sol32), "$(cnt32.cnt[]) mv")
    return nothing
end

Random.seed!(0x67696c61)
for useGpu in (false, true)
    useGpu && !CUDA.functional() && continue
    println("\n", useGpu ? "GPU" : "CPU", ", requested relative tolerance $slvTgt")
    @printf("%-9s %-14s %9s  %-12s %9s %10s  %s\n", "cells", "sus", "oprErr",
        "route", "solErr", "res64", "matvecs")
    for cel in celLst, sus in susLst
        runCel(cel, sus, useGpu)
    end
end
