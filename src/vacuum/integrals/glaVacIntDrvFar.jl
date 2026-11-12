# Point-observer derivative kernels of one cell, far route.
# K^g_ab(R) = d^g (da db + dab k^2) psi(R), psi(R) = int_cell g(R - y) dy, g = e^{ikr}/(4 pi f^2 r), k = 2 pi f,
# the cell centred at 0, R the observer, |g| = der.  For |R| > rho, half the cell diagonal, the
# addition theorem about the cell centre gives
#   K^g_ab = (i k/f^2) sum_lm (-1)^l h_l(k|R|) Y_lm(Rhat) int_cell [d^(g+ea+eb) + dab k^2 d^g] j_l(k|d|) Y_lm(dhat) dd ,
# the derivatives on the regular factor, a polynomial in d: the table holds the exact moments
# int_cell d^al [|d|^{l+2n} Y_lm] (momAcc with the flat weight), and frequency enters by j_l's series.
# Only l = der mod 2 survives, and for each al only the (l, m) whose axis parities are those of al.

# Bound D.  With u_l = (j_l(k|d|) Y_lm(dhat))_m, d_a u_l = k (P_a u_{l-1} + Q_a u_{l+1}) for constant
# matrices, and Schur's lemma gives sum_a |P_a x + Q_a y|^2 = l/(2l-1) |x|^2 + (l+1)/(2l+3) |y|^2 for
# every x, y (farMomO's gradient identity is the order-1 case).  Iterated, at every order p
#   sum_m |grad^p u_lm|^2 = |k|^{2p} sum_j W^p_lj (2j+1)/(4 pi) |j_j(k|d|)|^2 ,
#   W^0_lj = dlj ,  W^p_l = l/(2l-1) W^{p-1}_{l-1} + (l+1)/(2l+3) W^{p-1}_{l+1} ,
# so the bound is proven for every der.  Cauchy-Schwarz over m, Minkowski over the cell and
# |j_j(z)| <= |z|^j e^{|z|^2/(4j+6)}/(2j+1)!! give, for every entry and p = der + 2,
#   |K - K^(L)| <= (|k|/|f|^2) sum_{l > L, l = der mod 2} hb(l, k|R|) sqrt((2l+1)/4pi)
#                  sum_j (sqrt W^p_lj + sqrt W^{p-2}_lj) sqrt((2j+1)/4pi) |k|^{p+j} e^{(|k| rho)^2/(4j+6)} M_j/(2j+1)!! ,
# M_j = int_cell |d|^j.  The table stops j_l's series at n = N: t^{-l/2} u_l(k sqrt t) is entire in t,
# so Cauchy on |t| = tau bounds that tail by eps(N) = tau^{p-N-1} e^{(tau-1)(|k| rho)^2/6}/(1-1/tau)
# times the bound's own l term, and the whole n-truncation by eps(N) times the full bound sum.

# t[i]: the bound's l = ls[i] term without hb, scaled by (2l+1)!!/|k|^l as hnkBndS is
function farDrvBnd(lTop::Int, der::Int, s::NTuple{3,Float64}, k::ComplexF64, f::ComplexF64)
    p = der + 2; ak = abs(k); x = (ak * norm(s) / 2)^2
    sch(q, l) = q == 0 ? [1.0] : (w = zeros(2q + 1);
        l > 0 && (w[1:(2q - 1)] .+= l / (2l - 1) .* sch(q - 1, l - 1));
        w[3:end] .+= (l + 1) / (2l + 3) .* sch(q - 1, l + 1); w)
    hM = div(lTop + p + 1, 2) + 1
    mj = radMom(2hM, ntuple(d -> [fltMom(2i, 0.0, s[d] / 2) for i in 0:hM], 3))
    ls = collect(isodd(der):2:lTop)
    t = map(ls) do l
        w = sqrt.(sch(p, l)); w[3:(end - 2)] .+= sqrt.(sch(p - 2, l))
        sm = 0.0
        for (i, j) in enumerate((l - p):(l + p))
            (j < 0 || w[i] == 0) && continue
            dfr = j > l ? prod(ak / (2q + 1) for q in (l + 1):j) : prod(Float64(2q + 1) / ak for q in (j + 1):l; init = 1.0)
            sm += w[i] * sqrt((2j + 1) / (4pi)) * exp(x / (4j + 6)) * radQ(mj, j) * dfr
        end
        ak^(p + 1) / abs2(f) * sqrt((2l + 1) / (4pi)) * sm
    end
    ls, t
end

# per al of [orders der + 2; der], the (l, m) columns and int_cell d^al [|d|^{l+2n} Y_lm], n = 0..N
struct FarDrvTab
    L::Int
    N::Int
    lm::Vector{Vector{Int}}
    lv::Vector{Vector{Int}}
    A::Vector{Matrix{BigFloat}}
end

function farDrvTab(sQ::NTuple{3,QI}, als::Vector{NTuple{3,Int}}, L::Int, N::Int)
    setprecision(BigFloat, BLDPRC) do
        dM = L + 2N; nA = div(dM, 2) + 2
        W = ntuple(d -> momTrp([wgtA(n, one(BigFloat), 0, BigFloat(sQ[d]) / 2) for n in 0:dM], maximum(sum, als)), 3)
        lm = [Int[] for _ in als]; lv = [Int[] for _ in als]; cl = [Vector{BigFloat}[] for _ in als]
        P = bigMat(nA); U = bigMat(nA); V = bigMat(nA); tmp = BigInt(0); rg = momReg(); ztl = zroTol(BLDPRC)
        for l in isodd(sum(als[1])):2:L, mm in -l:l
            snf = mm < 0; m = abs(mm); p1, p2 = prtXY(m, snf)
            hit = [i for (i, al) in enumerate(als) if iseven(al[1] - p1) && iseven(al[2] - p2) && iseven(al[3] - l + p1 + p2)]
            hrmPly!(P, U, V, l, m, snf, p1, p2, tmp)
            nc = nrmCof(l, m, BigFloat)
            c = [zeros(BigFloat, N + 1) for _ in hit]
            for n in 0:N
                for (q, i) in enumerate(hit)
                    c[q][n + 1] = momAcc(P, W, als[i] .+ 1, l + 2n, p1, p2, ztl, rg) * nc
                end
                n < N && (mulRsq!(V, P); (P, V) = (V, P))
            end
            for (q, i) in enumerate(hit)
                all(iszero, c[q]) && continue
                push!(lm[i], ylmInd(l, mm)); push!(lv[i], l); push!(cl[i], c[q])
            end
        end
        FarDrvTab(L, N, lm, lv, [reduce(hcat, c; init = zeros(BigFloat, N + 1, 0)) for c in cl])
    end
end

# never evicted, keyed (cell, der); rebuilt whole when an offset asks for more L or N
const FARDRV = Dict{Tuple{NTuple{3,QI},Int},FarDrvTab}()
const FARDRVLK = ReentrantLock()

# Fill egoDrv[:, :, ig, i] with K^g at the observer Ds[i] (target centre minus source centre), g = drvGam(der)[ig].
# Offsets where bound D cannot meet tol times the a priori scale within LMAX are left NaN.  A Double64
# egoDrv rounded to Float64 lands within 1 ulp of the tensor max; a Float64 one within ~80.  The
# default tol is one eps, not FARTOL: at the band edge the scale is 5x the max, and FARTOL leaves 4 ulp.
function drvFar!(egoDrv::AbstractArray{<:Complex,4}, Ds::AbstractVector{NTuple{3,QI}}, scl::NTuple{3,QI}, frq, der::Int; tol = eps(Float64), disk::Bool = false)
    T = real(eltype(egoDrv))
    gms = drvGam(der); als = [drvGam(der + 2); gms]; nG = length(gms); nA2 = length(als) - nG
    fill!(egoDrv, NaN)
    # selection in Float64, as farSet's
    s6 = Float64.(scl); f6 = ComplexF64(frq); k6 = 2pi * f6; vt = prod(s6)
    r6 = [norm(Float64.(D)) for D in Ds]
    est6(r) = est(r, k6, f6, vt) * (abs(k6) + (der + 1) / r)^der
    ls, t = farDrvBnd(LMAX + LEXT, der, s6, k6, f6)
    tt = ntuple(_ -> t, 6)
    thr = whlThr(ls, tt, k6, est6, Float64(tol), norm(s6) / 2, maximum(r6; init = 0.0), LMAX)
    Ls = [(L = whlLvl(thr, r); L < 0 ? L : L + isodd(der)) for r in r6]
    ok = findall(>=(0), Ls)
    isempty(ok) && return egoDrv
    # the n-cut, certified at both ends of the radius range as cutVec's
    ful(r) = bndCum(ls, tt, k6, r)[1] + t[1] * hnkBndS(ls[1], k6 * r, abs(k6))[ls[1] + 1]
    amp = maximum(r -> ful(r) / est6(r), extrema(r6[ok]))
    x = (abs(k6) * norm(s6) / 2)^2; p = der + 2
    epn(N) = (tau = max(2.0, 6 * (N + 1 - p) / x); tau^(p - N - 1) * exp((tau - 1) * x / 6) / (1 - 1 / tau))
    N = findfirst(N -> epn(N) * amp <= tol / NBUD, 0:NCAP)
    N === nothing && error("drvFar!: the k-series of j_l is not certified within N = $NCAP at |k| rho = ", sqrt(x))
    N -= 1; L = maximum(Ls[ok])
    tb = lock(FARDRVLK) do
        t0 = get(FARDRV, (scl, der), nothing)
        fn = disk ? joinpath(tabDir(), string("drvfar_", join((string(x.num, "_", x.den) for x in scl), "-"), "_d", der, ".jls")) : ""
        t0 === nothing && disk && isfile(fn) && (t0 = deserialize(fn))
        if t0 === nothing || t0.L < L || t0.N < N
            t0 = farDrvTab(scl, als, max(L, t0 === nothing ? 0 : t0.L), max(N, t0 === nothing ? 0 : t0.N))
            disk && (tmp = "$fn.$(getpid())"; serialize(tmp, t0); mv(tmp, fn; force = true))
        end
        FARDRV[(scl, der)] = t0
    end
    # contract the series in BigFloat; the k^2 of the dab term goes into the order-der tables
    f = Complex{T}(frq); k = 2 * T(pi) * f; E = imag(frq) == 0 ? T : Complex{T}
    Q = setprecision(BigFloat, BLDPRC) do
        kB = 2 * BigFloat(pi) * Complex{BigFloat}(frq); cb = Vector{Complex{BigFloat}}(undef, tb.N + 1)
        map(eachindex(als)) do i
            map(eachindex(tb.lv[i])) do j
                l = tb.lv[i][j]; serCof!(cb, l, tb.N, kB)
                q = (-1)^l * sum(cb[n] * tb.A[i][n, j] for n in 1:(tb.N + 1)) * (i > nA2 ? kB^2 : 1)
                E <: Real ? E(real(q)) : E(q)
            end
        end
    end
    pos = Dict(al => i for (i, al) in enumerate(als))
    ix = [pos[gms[g] .+ ntuple(d -> (d == a) + (d == b), 3)] for a in 1:3, b in 1:3, g in 1:nG]
    prf = prfFrq(f, k)
    wss = [(FarWrk(L, T), zeros(Complex{T}, length(als))) for _ in 1:Threads.maxthreadid()]
    Threads.@threads :static for i in ok
        ws, S = wss[Threads.threadid()]
        R = T.(Ds[i]); rr, e = phsSed(R, f)
        shmFil!(ws, R[1] / rr, R[2] / rr, R[3] / rr, Ls[i]); hnkFil!(ws, rr, e, k, Ls[i])
        for q in eachindex(als)
            S[q] = accOne(ws.h, ws.Y, tb.lm[q], tb.lv[q], Q[q], Ls[i])
        end
        for g in 1:nG, b in 1:3, a in 1:3
            egoDrv[a, b, g, i] = prf * (S[ix[a, b, g]] + (a == b ? S[nA2 + g] : 0))
        end
    end
    egoDrv
end
