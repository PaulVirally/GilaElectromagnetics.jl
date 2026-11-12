using KernelAbstractions
using KernelAbstractions: supports_float64

"""
    GlaKerOpt{T<:AbstractFloat, B<:KernelAbstractions.Backend}

Options that determine how the Green function operator is computed. `T` is the
real storage precision of the operator data (`Complex{T}`), generated in
Float64 and rounded once, and `dev` the KernelAbstractions backend the operator
computes on.

# Fields
- `frqPhz::ComplexF64`: Multiplicative scaling factor allowing for complex frequencies
- `qssApx::Bool`: Quasistatic approximation flag (true to generate the quasistatic kernel)
- `der::Int`: Order of the field derivatives returned, 0 for the field itself
- `ctr::Bool`: Values at the target cell centres (true) or averaged over the target cells (false)
- `adjMod::Bool`: Adjoint mode flag (true if adjoint mode is enabled)
- `hrmPrt::Union{Nothing,Symbol}`: `:sym` or `:asy` keeps only the entry by entry real or imaginary part of the kernel, which for a self operator is its Hermitian part `(G₀ + G₀†) / 2` or `(G₀ - G₀†) / 2i`; `nothing` is the full kernel
- `dev::B`: Backend the operator computes on
- `blc::Union{Nothing,NamedTuple}`: Bloch lattice `(ax, P, θ)` of a periodic unit cell: periodic axes, periods in cells, one Bloch phase per axis (radians, or an exact twist θ/2π as a `Rational`), a composite block's with the periods `L` in λ for `P`; `nothing` is vacuum
- `blcSet`: The `BlcSet` of the current frequency, kept for `setBlc!`
- `kapCch`: A composite operator's κ cache of the current frequency, kept for `setBlc!` (first block only)

`Float64` storage on a backend without double precision throws an `ArgumentError`.
"""
mutable struct GlaKerOpt{T<:AbstractFloat, B<:KernelAbstractions.Backend}
    frqPhz::ComplexF64
    qssApx::Bool
    der::Int
    ctr::Bool
    adjMod::Bool
    hrmPrt::Union{Nothing,Symbol}
    dev::B
    blc::Union{Nothing,NamedTuple}
    blcSet::Union{Nothing,BlcSet}
    kapCch::Union{Nothing,Bloch.KapCache}
    function GlaKerOpt{T}(dev::B = CPU(); frqPhz::Number = 1.0+0.0im,
        qssApx::Bool = false, der::Int = 0, ctr::Bool = der > 0, adjMod::Bool = false, hrmPrt::Union{Nothing,Symbol} = nothing, blc = nothing, blcSet = nothing, kapCch = nothing) where {T<:AbstractFloat, B<:KernelAbstractions.Backend}
        hrmPrt ∈ (nothing, :sym, :asy) || throw(ArgumentError("hrmPrt is :sym, :asy or nothing, got $(repr(hrmPrt))."))
        T === Float64 && !supports_float64(dev) && throw(ArgumentError("$(nameof(B)) has no Float64. Build the operator in Float32."))
        qssApx && blc !== nothing && throw(ArgumentError("The quasistatic kernel has no Bloch lattice."))
        der >= 0 || throw(ArgumentError("der is a derivative order, 0 or more, got $der."))
        der == 0 || ctr || throw(ArgumentError("Field derivatives (der = $der) exist only at the cell centres, ctr = true: the average of a derivative over a cell is ill-defined where cells touch."))
        ctr && (qssApx || blc !== nothing) && throw(ArgumentError("Values at the cell centres (ctr = true) have no quasistatic kernel and no Bloch lattice."))
        return new{T, B}(frqPhz, qssApx, der, ctr, adjMod, hrmPrt, dev, blc, blcSet, kapCch)
    end
end

"""
    GlaKerOpt{T}(dev::KernelAbstractions.Backend=CPU(); frqPhz=1.0+0.0im, qssApx=false, der=0, ctr=der>0, adjMod=false, hrmPrt=nothing, blc=nothing, blcSet=nothing, kapCch=nothing)
    GlaKerOpt(dev::KernelAbstractions.Backend=CPU(); kwargs...)

Options on `dev`, by default a phase factor of 1, the full
kernel and not only a Hermitian part of it, not adjoint and no Bloch lattice. The storage precision `T` defaults to `dflPrc`.
"""
GlaKerOpt(dev::KernelAbstractions.Backend=CPU(); kw...) = GlaKerOpt{dflPrc}(dev; kw...)

"""
    GlaKerOpt{T}(opt::GlaKerOpt; dev::KernelAbstractions.Backend=opt.dev)

Copy of `opt` with storage precision `T`, on `dev`.
"""
GlaKerOpt{T}(opt::GlaKerOpt; dev::KernelAbstractions.Backend=opt.dev) where T<:AbstractFloat =
    GlaKerOpt{T}(dev; opt.frqPhz, opt.qssApx, opt.der, opt.ctr, opt.adjMod, opt.hrmPrt, opt.blc, opt.blcSet, opt.kapCch)

"""
    devMem(dev::KernelAbstractions.Backend)

Bytes free on `dev` for the Fourier stage of operator creation, `typemax(Int)`
where the backend does not say. `GilaCUDAExt` answers for CUDA.
"""
devMem(::KernelAbstractions.Backend) = typemax(Int)
