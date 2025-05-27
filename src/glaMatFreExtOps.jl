# This file defines the necessary constructors to be able to interface with
# other linear operator packages, including `SciMLOperators.jl`,
# `LinearOperators.jl`, and `LinearMaps.jl`.

import SciMLOperators: FunctionOperator
using LinearOperators
using LinearMaps

invMul!(w, opr::AbstractGlaOpr, v, α, β) = axpby!(α, solve(opr, v, slv(opr)), β, w)
invMul!(w, opr::SctOpr, v, α, β) = axpby!(α, opr.invSctOpr * v, β, w)
function invMul!(w, opr::GlaOpr, v, α, β)
    actG0Inv = solve(opr.sctOpr.invSctOpr.oprVac, v, slv(opr.sctOpr.invSctOpr))
    return axpby!(α, opr.sctOpr.invSctOpr.oprVac * actG0Inv, β, w)
end

function invMulAdj!(w, opr::AbstractGlaOpr, v, α, β)
    adjoint!(opr) # Compute with the adjoint operator
    out = invMul!(w, opr, v, α, β) # Inverse adjoint matrix-vector product
    adjoint!(opr) # Restore the original operator
    return out
end
function invMulAdj!(w, opr::SctOpr, v, α, β)
    adjoint!(opr.invSctOpr) # Compute with the adjoint operator
    out = invMul!(w, opr.invSctOpr, v, α, β)
    adjoint!(opr.invSctOpr) # Restore the original operator
    return out
end
function invMulAdj!(w, opr::GlaOpr, v, α, β)
    adjoint!(opr.sctOpr.invSctOpr) # Compute with the adjoint operator
    actWInvDag = opr.sctOpr.invSctOpr * v
    adjoint!(opr.sctOpr.invSctOpr) # Restore the original operator

    adjoint!(opr.sctOpr.invSctOpr.oprVac) # Compute with the adjoint operator
    out = axpby!(α, opr.sctOpr.invSctOpr.oprVac * actWInvDag, β, w)
    adjoint!(opr.sctOpr.invSctOpr.oprVac) # Restore the original operator
    return out
end

function FunctionOperator(opr::AbstractGlaOpr)
    T = eltype(opr)
    m, n = size(opr)
    inp = fill!(arrTyp(opr)(undef, n), zero(T)) # Input prototype
    out = fill!(arrTyp(opr)(undef, m), zero(T)) # Output prototype
    fwd!(w, v, _u, _p, _t) = mul!(w, opr, v, one(T), zero(T)) # Matrix-vector product
    adj!(w, v, _u, _p, _t) = begin # Adjoint matrix-vector product
        adjoint!(opr) # Compute with the adjoint operator
        out = mul!(w, opr, v, one(T), zero(T))
        adjoint!(opr) # Restore the original operator
        return out
    end
    invFwd!(w, v, _u, _p, _t) = invMul!(w, opr, v, one(T), zero(T)) # Inverse matrix-vector product
    invAdj!(w, v, _u, _p, _t) = invMulAdj!(w, opr, v, one(T), zero(T)) # Inverse adjoint matrix-vector product
    return FunctionOperator(fwd!, inp, out; op_adjoint=adj!, op_inverse=invFwd!, op_adjoint_inverse=invAdj!, isconstant=true, islinear=true)
end

function LinearOperator(opr::AbstractGlaOpr)
    T = eltype(opr)
    m, n = size(opr)
    fwd!(w, v) = mul!(w, opr, v, one(T), zero(T)) # Matrix-vector product
    adj!(w, v) = begin # Adjoint matrix-vector product
        adjoint!(opr) # Compute with the adjoint operator
        out = mul!(w, opr, v, one(T), zero(T))
        adjoint!(opr) # Restore the original operator
        return out
    end
    is_symmetric = false
    is_hermitian = false
    return LinearOperator(T, m, n, is_symmetric, is_hermitian, fwd!, nothing, adj!; S=arrTyp(opr))
end

function LinearMap(opr::AbstractGlaOpr)
    T = eltype(opr)
    m, n = size(opr)
    fwd!(w, v) = mul!(w, opr, v, one(T), zero(T)) # Matrix-vector product
    adj!(w, v) = begin # Adjoint matrix-vector product
        adjoint!(opr) # Compute with the adjoint operator
        out = mul!(w, opr, v, one(T), zero(T))
        adjoint!(opr) # Restore the original operator
        return out
    end
    return LinearMap{T}(fwd!, adj!, m, n; ismutating=true)
end
