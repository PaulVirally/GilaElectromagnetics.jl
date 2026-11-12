# Point-observer derivative kernels of the cell potential, near route. With the source cell (edges
# scl) centred at 0 and the observer at R, psi(R) = int_cell g(R - y) dy, g = e^{ikr}/(4 pi f^2 r),
# k = 2 pi f, and K^gam_ab(R) = d^gam (d_a d_b + k^2 delta_ab) psi(R). psi is the k-series of the
# moments M_m(R) = int_cell |y - R|^m dy. Each differentiated axis of D^al M_m collapses onto the faces:
# three such axes leave corner values, two the segment integrals powSeg, one the rectangle integrals
# psiBox, and a lone axis of order 3 or more goes through Laplace.

unv(a::Int) = ntuple(i -> Int(i == a), 3)

# the multi-indices of the derivatives of order der, in the order of the kernels the routes fill
drvGam(der::Int) = [(a, b, der - a - b) for a in der:-1:0 for b in (der - a):-1:0]

# d^bet r^m as the terms (c, gam, n) of sum c u^gam r^n
function drvTrm(m::Int, bet::NTuple{3,Int})
    trm = [(1, (0, 0, 0), m)]
    for a in 1:3, _ in 1:bet[a]
        nxt = empty(trm)
        for (c, gam, n) in trm
            gam[a] > 0 && push!(nxt, (c * gam[a], gam .- unv(a), n))
            n != 0 && push!(nxt, (c * n, gam .+ unv(a), n - 2))
        end
        trm = nxt
    end
    return trm
end

# powSeg(n, l, h, d) for odd n >= nLo and every n >= 0 up to nHi, upwards from the n = -1, 0 seeds
function segLad(nLo::Int, nHi::Int, l::T, h::T, d::T) where {T}
    Sl = sqrt(l^2 + d^2); Sh = sqrt(h^2 + d^2)
    J = Dict(n => powSeg(n, l, h, d) for n in [nLo:2:-1; 0])
    for n in 1:nHi
        J[n] = (h * Sh^n - l * Sl^n + n * d^2 * J[n - 2]) / (n + 1)
    end
    return J
end

# psiBox(n, t, a1, a2, b1, b2) at entry n + 2 for n = -1:nHi, upwards from the n = -1, 0 seeds
function psiLad(nHi::Int, t::T, a1::T, a2::T, b1::T, b2::T) where {T}
    Ja1, Ja2 = (segLad(-1, nHi, a1, a2, sqrt(t^2 + b^2)) for b in (b1, b2))
    Jb1, Jb2 = (segLad(-1, nHi, b1, b2, sqrt(t^2 + a^2)) for a in (a1, a2))
    P = [psiBox(-1, t, a1, a2, b1, b2); (a2 - a1) * (b2 - b1); zeros(T, nHi)]
    for n in 1:nHi
        P[n + 2] = (a2 * Jb2[n] - a1 * Jb1[n] + b2 * Ja2[n] - b1 * Ja1[n] + n * t^2 * P[n]) / (n + 2)
    end
    return P
end

# powSeg and psiBox take nonnegative ranges: split [l, h] at 0 and reflect, the integrand being even
spl(l, h) = h <= 0 ? ((-h, -l),) : l >= 0 ? ((l, h),) : ((zero(l), -l), (zero(l), h))

# D^al M_m at entry m + 2 for m = -1:N, the cell [lo, hi] relative to the observer; mem holds the al
# already done, at any length
function drvMom!(mem::Dict, al::NTuple{3,Int}, lo::NTuple{3,T}, hi::NTuple{3,T}, N::Int) where {T}
    v = get(mem, al, nothing)
    !isnothing(v) && length(v) >= N + 2 && return v
    S = filter(a -> al[a] > 0, (1, 2, 3))
    if isempty(S)                                   # Δ r^(m+2) = (m+2)(m+3) r^m
        w = sum(drvMom!(mem, 2 .* unv(a), lo, hi, N + 2) for a in 1:3)
        v = [w[m + 4] / ((m + 2) * (m + 3)) for m in -1:N]
    elseif length(S) == 1 && al[S[1]] >= 3          # Δ r^m = m(m+1) r^(m-2); the δ of m = -1 dies under d^bet
        a = S[1]; b, c = filter(!=(a), (1, 2, 3)); bet = al .- 2 .* unv(a)
        vb = drvMom!(mem, bet .+ 2 .* unv(b), lo, hi, N)
        vc = drvMom!(mem, bet .+ 2 .* unv(c), lo, hi, N)
        v0 = drvMom!(mem, bet, lo, hi, N)
        v = [(m < 1 ? zero(T) : m * (m + 1) * v0[m]) - vb[m + 2] - vc[m + 2] for m in -1:N]
    else
        fac = map(Iterators.product((((hi[a], 1), (lo[a], -1)) for a in S)...)) do ends
            u = collect(zero.(lo)); sg = 1
            for ((e, s), a) in zip(ends, S); u[a] = e; sg *= s; end
            if length(S) == 3
                r = sqrt(sum(abs2, u))
                w = [sum(c * prod(u .^ gam) * r^n for (c, gam, n) in drvTrm(m, al .- 1); init = zero(T)) for m in -1:N]
            elseif length(S) == 2
                a, b = S; c = 6 - a - b; bet = Base.setindex(al .- 1, 0, c)
                w = sum(spl(lo[c], hi[c])) do (l, h)
                    J = segLad(-1 - 2 * sum(bet), N, l, h, hypot(u[a], u[b]))
                    [sum(cf * u[a]^gam[a] * u[b]^gam[b] * J[n] for (cf, gam, n) in drvTrm(m, bet); init = zero(T)) for m in -1:N]
                end
            else                                    # al[a] is 1 (psi_m) or 2 (d_a r^m = m t r^(m-2))
                a = S[1]; b, c = filter(!=(a), (1, 2, 3)); t = u[a]
                pcs = [(p..., q...) for p in spl(lo[b], hi[b]) for q in spl(lo[c], hi[c])]
                P = sum(psiLad(N, t, p...) for p in pcs)
                w = al[a] == 1 ? P : [-sum(solAng(t, p...) for p in pcs); zero(T); [m * t * P[m] for m in 1:N]]
            end
            return sg .* w
        end
        # summed over one collapsed axis at a time, so face pairs that cancel by parity cancel exactly
        v = only(foldl((A, d) -> sum(A; dims = d), 1:length(S); init = fac))
        isodd(sum(al)) && (v .= .-v)
    end
    return mem[al] = v
end

# the moments carry no frequency: one table per (shape, separation), extended in length and in al as
# asked; never evicted
const drvMem = Dict{Tuple{NTuple{3,QI},NTuple{3,QI}},Dict{NTuple{3,Int},Vector{Double64}}}()
const drvLck = ReentrantLock()

# egoDrv[:, :, j, i] = K^gam_j at the separation Ds[i] (target centre minus source centre), gam_j = drvGam(der)[j]
function drvNer!(egoDrv::AbstractArray{<:Complex,4}, Ds::AbstractVector{NTuple{3,QI}}, scl::NTuple{3,QI}, frq, der::Int)
    all(D -> all(abs.(D) .!= scl ./ 2), Ds) || throw(ArgumentError("drvNer!: the observer must not lie on a face plane of the source cell, |separation| = edge/2 along some axis."))
    F = ComplexDF64(frq)
    kSq = (2 * Double64(π) * F)^2
    @threads for i in eachindex(Ds)
        key = (scl, Ds[i])
        mem = lock(() -> copy(get(drvMem, key, valtype(drvMem)())), drvLck)
        N = serOrd(scl ./ 2, abs.(Ds[i]), frq)              # reach |D| + scl/2, the far corner
        lo = Double64.(.-Ds[i] .- scl ./ 2); hi = Double64.(.-Ds[i] .+ scl ./ 2)
        ser(al) = momSer(view(drvMom!(mem, al, lo, hi, N), 1:N + 2), F)
        for (j, gam) in enumerate(drvGam(der))
            kg = kSq * ser(gam)
            for a in 1:3, b in 1:3
                egoDrv[a, b, j, i] = ser(gam .+ unv(a) .+ unv(b)) + (a == b ? kg : 0)
            end
        end
        lock(() -> drvMem[key] = mem, drvLck)
    end
    return egoDrv
end
