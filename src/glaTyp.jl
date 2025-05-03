"""
    GilaTypes

This module provides the core abstract types and interfaces used across the Gila package.
It helps break circular dependencies between modules.
"""
module GilaTypes

using LinearAlgebra

export GlaSlv, AbstractGlaOpr

"""
    GlaSlv

Abstract base type for all solvers in the Gila package. All concrete solver types
must subtype this type.
"""
abstract type GlaSlv end

"""
    AbstractGlaOpr

Abstract base type for all operators in the Gila package. All concrete operator types
must subtype this type and implement the AbstractMatrix interface.
"""
abstract type AbstractGlaOpr end

# Make AbstractGlaOpr compatible with AbstractMatrix
Base.promote_rule(::Type{<:AbstractGlaOpr}, ::Type{<:AbstractMatrix}) = AbstractMatrix
Base.promote_rule(::Type{<:AbstractMatrix}, ::Type{<:AbstractGlaOpr}) = AbstractMatrix

end # module 