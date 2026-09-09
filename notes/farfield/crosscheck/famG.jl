"Derivative-free Taylor coefficients of g(R+δ)=e^{ik|R+δ|}/(4π f²|R+δ|); family (g)."

const Idx3 = NTuple{3,Int}

@inline sub(a::Idx3, i::Int) = (a[1] - (i == 1), a[2] - (i == 2), a[3] - (i == 3))
@inline add(a::Idx3, i::Int) = (a[1] + (i == 1), a[2] + (i == 2), a[3] + (i == 3))

"Coefficient at multi-index (i,j,k), zero outside the stored simplex."
@inline function gt(A::Array{C,3}, i::Int, j::Int, k::Int) where C
	(i < 0 || j < 0 || k < 0) && return zero(C)
	p = size(A, 1) - 1
	(i > p || j > p || k > p) && return zero(C)
	@inbounds A[i+1, j+1, k+1]
end
@inline gt(A::Array{C,3}, a::Idx3) where C = gt(A, a[1], a[2], a[3])

"Axis of the largest component of a multi-index (the best-conditioned pivot)."
@inline piv(g::Idx3) = g[1] >= g[2] ? (g[1] >= g[3] ? 1 : 3) : (g[2] >= g[3] ? 2 : 3)

"Route (i): Taylor coefficients a of G(δ)=g(R+δ) and b of S(δ)=|R+δ|G(δ) to total order p."
function taylorSys(R::NTuple{3,T}, f::Complex{T}, p::Int) where T<:AbstractFloat
	C = Complex{T}
	k = 2 * T(pi) * f
	rho2 = R[1]^2 + R[2]^2 + R[3]^2
	rho = sqrt(rho2)
	a = zeros(C, p + 1, p + 1, p + 1)
	b = zeros(C, p + 1, p + 1, p + 1)
	b[1, 1, 1] = exp(im * k * rho) / (4 * T(pi) * f^2)
	a[1, 1, 1] = b[1, 1, 1] / rho
	for n in 0:p-1, g1 in 0:n+1, g2 in 0:n+1-g1
		g3 = n + 1 - g1 - g2
		g = (g1, g2, g3)
		j = piv(g)
		al = sub(g, j)
		aj = al[j]
		am = sub(al, j)
		acc = R[j] * (im * k * gt(b, al) - gt(a, al)) + (im * k * gt(b, am) - gt(a, am))
		acc -= 2 * R[j] * aj * gt(a, al) + (aj - 1) * gt(a, am)
		for i in 1:3
			i == j && continue
			acc -= 2 * R[i] * (aj + 1) * gt(a, add(sub(al, i), j))
			acc -= (aj + 1) * gt(a, add(sub(sub(al, i), i), j))
		end
		a[g1+1, g2+1, g3+1] = acc / (rho2 * (aj + 1))
		b[g1+1, g2+1, g3+1] = im * k * (R[j] * gt(a, al) + gt(a, am)) / (aj + 1)
	end
	a, b
end

"Sum of |terms| of the route-(i) recurrence over its levels, the amplification diagnostic."
function taylorSysAmp(R::NTuple{3,T}, f::Complex{T}, p::Int) where T<:AbstractFloat
	a, b = taylorSys(R, f, p)
	amp = zeros(T, p + 1)
	k = 2 * T(pi) * f
	rho2 = R[1]^2 + R[2]^2 + R[3]^2
	for n in 0:p-1, g1 in 0:n+1, g2 in 0:n+1-g1
		g3 = n + 1 - g1 - g2
		g = (g1, g2, g3)
		j = piv(g)
		al = sub(g, j)
		aj = al[j]
		am = sub(al, j)
		s = abs(R[j] * (im * k * gt(b, al) - gt(a, al))) + abs(im * k * gt(b, am) - gt(a, am))
		s += abs(2 * R[j] * aj * gt(a, al)) + abs((aj - 1) * gt(a, am))
		for i in 1:3
			i == j && continue
			s += abs(2 * R[i] * (aj + 1) * gt(a, add(sub(al, i), j)))
			s += abs((aj + 1) * gt(a, add(sub(sub(al, i), i), j)))
		end
		r = abs(a[g1+1, g2+1, g3+1] * rho2 * (aj + 1))
		amp[n+2] = max(amp[n+2], r == 0 ? zero(T) : s / r)
	end
	amp
end

"Route (ii): Taylor coefficients of φ(t)=g(R+tu) and ψ=|R+tu|φ to order p, u need not be a unit vector."
function taylor1D(R::NTuple{3,T}, u::NTuple{3,T}, f::Complex{T}, p::Int) where T<:AbstractFloat
	C = Complex{T}
	k = 2 * T(pi) * f
	A = R[1]^2 + R[2]^2 + R[3]^2
	B = 2 * (R[1] * u[1] + R[2] * u[2] + R[3] * u[3])
	D = u[1]^2 + u[2]^2 + u[3]^2
	rho = sqrt(A)
	pc = zeros(C, p + 1)
	qc = zeros(C, p + 1)
	qc[1] = exp(im * k * rho) / (4 * T(pi) * f^2)
	pc[1] = qc[1] / rho
	for n in 0:p-1
		pn = pc[n+1]
		qn = qc[n+1]
		pm = n >= 1 ? pc[n] : zero(C)
		qm = n >= 1 ? qc[n] : zero(C)
		acc = (B / 2) * (im * k * qn - pn) + D * (im * k * qm - pm) - B * n * pn - D * (n - 1) * pm
		pc[n+2] = acc / (A * (n + 1))
		qc[n+2] = im * k * ((B / 2) * pn + D * pm) / (n + 1)
	end
	pc, qc
end

"Route (i-b): march the Helmholtz recurrence in δ₃ from Cauchy data a[:,:,1:2], overwriting a."
function helmMarch!(a::Array{C,3}, f::Complex{T}, p::Int) where {T,C}
	k2 = (2 * T(pi) * f)^2
	for m in 0:p-2, i in 0:p-m-2, j in 0:p-m-2-i
		# α = (i,j,m); solve for c_{α+2e₃}
		t = k2 * gt(a, (i, j, m))
		t += (i + 2) * (i + 1) * gt(a, (i + 2, j, m))
		t += (j + 2) * (j + 1) * gt(a, (i, j + 2, m))
		a[i+1, j+1, m+3] = -t / ((m + 2) * (m + 1))
	end
	a
end

"Exact weight moment ∫_{-s}^{s}(s-|t|)tⁿ dt."
@inline mom1(n::Int, s::T) where T = isodd(n) ? zero(T) : 2 * s^(n + 2) / T((n + 1) * (n + 2))

"Far tensor (1/V)∫_D w(δ)[∂_a∂_b + δ_ab k²]g(R+δ)dδ from Taylor coefficients a, integrand truncated at degree q."
function farTen(a::Array{C,3}, s::NTuple{3,T}, f::Complex{T}, q::Int) where {T,C}
	k = 2 * T(pi) * f
	V = s[1] * s[2] * s[3]
	M = zeros(C, 3, 3)
	mt = T[mom1(n, s[i]) for n in 0:q, i in 1:3]
	for b1 in 0:2:q, b2 in 0:2:q-b1, b3 in 0:2:q-b1-b2
		w = mt[b1+1, 1] * mt[b2+1, 2] * mt[b3+1, 3]
		be = (b1, b2, b3)
		ab = gt(a, be)
		for A in 1:3, B in 1:3
			g = add(add(be, A), B)
			c = (be[A] + 1 + (A == B)) * (be[B] + 1)
			M[A, B] += w * c * gt(a, g)
			A == B && (M[A, B] += w * k^2 * ab)
		end
	end
	M ./ V
end

"Term-magnitude sum of farTen, entrywise, for the cancellation diagnostic."
function farTenAbs(a::Array{C,3}, s::NTuple{3,T}, f::Complex{T}, q::Int) where {T,C}
	k = 2 * T(pi) * f
	V = s[1] * s[2] * s[3]
	M = zeros(T, 3, 3)
	mt = T[mom1(n, s[i]) for n in 0:q, i in 1:3]
	for b1 in 0:2:q, b2 in 0:2:q-b1, b3 in 0:2:q-b1-b2
		w = mt[b1+1, 1] * mt[b2+1, 2] * mt[b3+1, 3]
		be = (b1, b2, b3)
		ab = gt(a, be)
		for A in 1:3, B in 1:3
			g = add(add(be, A), B)
			c = (be[A] + 1 + (A == B)) * (be[B] + 1)
			M[A, B] += abs(w * c * gt(a, g))
			A == B && (M[A, B] += abs(w * k^2 * ab))
		end
	end
	M ./ abs(V)
end

# ---------- route (iii): minimal multivariate truncated-Taylor arithmetic ----------

"Indices of the truncated simplex up to total order p, level-major."
function simplex(p::Int)
	v = Idx3[]
	for n in 0:p, i in 0:n, j in 0:n-i
		push!(v, (i, j, n - i - j))
	end
	v
end

"Truncated product of two simplex-supported series."
function tmul(x::Array{C,3}, y::Array{C,3}, p::Int) where C
	z = zeros(C, p + 1, p + 1, p + 1)
	for n in 0:p, i in 0:n, j in 0:n-i
		k = n - i - j
		acc = zero(C)
		for a1 in 0:i, a2 in 0:j, a3 in 0:k
			acc += x[a1+1, a2+1, a3+1] * y[i-a1+1, j-a2+1, k-a3+1]
		end
		z[i+1, j+1, k+1] = acc
	end
	z
end

"Truncated sqrt: r with r²=P, P a stored series with P[0]≠0."
function tsqrt(P::Array{C,3}, p::Int) where C
	r = zeros(C, p + 1, p + 1, p + 1)
	r[1, 1, 1] = sqrt(P[1, 1, 1])
	for n in 1:p, i in 0:n, j in 0:n-i
		k = n - i - j
		acc = P[i+1, j+1, k+1]
		for a1 in 0:i, a2 in 0:j, a3 in 0:k
			(a1 == 0 && a2 == 0 && a3 == 0) && continue
			(a1 == i && a2 == j && a3 == k) && continue
			acc -= r[a1+1, a2+1, a3+1] * r[i-a1+1, j-a2+1, k-a3+1]
		end
		r[i+1, j+1, k+1] = acc / (2 * r[1, 1, 1])
	end
	r
end

"Truncated exp of a series V by dE = E dV along the largest axis."
function texp(V::Array{C,3}, p::Int) where C
	E = zeros(C, p + 1, p + 1, p + 1)
	E[1, 1, 1] = exp(V[1, 1, 1])
	for n in 1:p, i in 0:n, j in 0:n-i
		k = n - i - j
		g = (i, j, k)
		d = piv(g)
		al = sub(g, d)
		acc = zero(C)
		for a1 in 0:al[1], a2 in 0:al[2], a3 in 0:al[3]
			be = (a1, a2, a3)
			bd = add(be, d)
			acc += E[al[1]-a1+1, al[2]-a2+1, al[3]-a3+1] * (be[d] + 1) * V[bd[1]+1, bd[2]+1, bd[3]+1]
		end
		E[i+1, j+1, k+1] = acc / g[d]
	end
	E
end

"Truncated quotient N/D with D[0]≠0."
function tdiv(N::Array{C,3}, D::Array{C,3}, p::Int) where C
	Q = zeros(C, p + 1, p + 1, p + 1)
	for n in 0:p, i in 0:n, j in 0:n-i
		k = n - i - j
		acc = N[i+1, j+1, k+1]
		for a1 in 0:i, a2 in 0:j, a3 in 0:k
			(a1 == 0 && a2 == 0 && a3 == 0) && continue
			acc -= D[a1+1, a2+1, a3+1] * Q[i-a1+1, j-a2+1, k-a3+1]
		end
		Q[i+1, j+1, k+1] = acc / D[1, 1, 1]
	end
	Q
end

"Route (iii): Taylor coefficients of g(R+δ) by truncated Taylor arithmetic on e^{ik√P}/(4πf²√P)."
function taylorAD(R::NTuple{3,T}, f::Complex{T}, p::Int) where T<:AbstractFloat
	C = Complex{T}
	k = 2 * T(pi) * f
	P = zeros(C, p + 1, p + 1, p + 1)
	P[1, 1, 1] = R[1]^2 + R[2]^2 + R[3]^2
	p >= 1 && (P[2, 1, 1] = 2 * R[1]; P[1, 2, 1] = 2 * R[2]; P[1, 1, 2] = 2 * R[3])
	p >= 2 && (P[3, 1, 1] = one(C); P[1, 3, 1] = one(C); P[1, 1, 3] = one(C))
	r = tsqrt(P, p)
	V = (im * k) .* r
	E = texp(V, p)
	E ./= (4 * T(pi) * f^2)
	tdiv(E, r, p)
end

# ---------- route (iv): Gegenbauer / Legendre structure ----------

"Monomial coefficients of P_l: P_l(x)=Σ_j lgc(l)[j+1] x^{l-2j}."
function lgc(l::Int, ::Type{T}) where T
	c = zeros(T, div(l, 2) + 1)
	for j in 0:div(l, 2)
		c[j+1] = (isodd(j) ? -one(T) : one(T)) * T(factorial(big(2l - 2j))) /
				 (T(2)^l * T(factorial(big(j))) * T(factorial(big(l - j))) * T(factorial(big(l - 2j))))
	end
	c
end

"Spherical Hankel h_l^{(1)}(z) for l=0..L by upward recurrence."
function hank(z::C, L::Int) where C
	h = zeros(C, L + 1)
	h[1] = -im * exp(im * z) / z
	L >= 1 && (h[2] = (-1 / z - im / z^2) * exp(im * z))
	for l in 1:L-1
		h[l+2] = (2l + 1) / z * h[l+1] - h[l]
	end
	h
end

"∫_D w(δ) |δ|^{2j} (n·δ)^{m} dδ / V, exactly, from the weight moments."
function legMom(n::NTuple{3,T}, m::Int, j::Int, s::NTuple{3,T}) where T
	V = s[1] * s[2] * s[3]
	acc = zero(T)
	# |δ|^{2j} = Σ_{p+q+r=j} j!/(p!q!r!) δ1^{2p} δ2^{2q} δ3^{2r}
	for p in 0:j, q in 0:j-p
		r = j - p - q
		mc = T(factorial(big(j))) / T(factorial(big(p)) * factorial(big(q)) * factorial(big(r)))
		for u in 0:m, v in 0:m-u
			w = m - u - v
			nc = T(factorial(big(m))) / T(factorial(big(u)) * factorial(big(v)) * factorial(big(w)))
			nc *= n[1]^u * n[2]^v * n[3]^w
			acc += mc * nc * mom1(2p + u, s[1]) * mom1(2q + v, s[2]) * mom1(2r + w, s[3])
		end
	end
	acc / V
end

"Double factorial n!! (n<=0 gives 1)."
@inline dfac(n::Int, ::Type{T}) where T = n <= 0 ? one(T) : T(prod(big(n):big(-2):big(1)))

"Factorial table 0!..N! as T."
facTab(N::Int, ::Type{T}) where T = T[T(factorial(big(i))) for i in 0:N]

"Route (iv): Taylor coefficients of g(R+δ) from the addition theorem, and Σ|terms| per coefficient."
function taylorLeg(R::NTuple{3,T}, f::Complex{T}, p::Int) where T<:AbstractFloat
	C = Complex{T}
	k = 2 * T(pi) * f
	rho = sqrt(R[1]^2 + R[2]^2 + R[3]^2)
	h = hank(k * rho, p)
	ft = facTab(2p + 2, T)
	a = zeros(C, p + 1, p + 1, p + 1)
	ab = zeros(T, p + 1, p + 1, p + 1)
	pre0 = im * k / (4 * T(pi) * f^2)
	for n in 0:p, l in n:-2:0
		m = div(n - l, 2)
		pl = pre0 * (2l + 1) * (isodd(l) ? -one(T) : one(T)) * h[l+1] *
			 (isodd(m) ? -one(C) : one(C)) * k^n / (T(2)^m * ft[m+1] * dfac(l + n + 1, T))
		cl = lgc(l, T)
		for j in 0:div(l, 2)
			cj = pl * cl[j+1] / rho^(l - 2j)
			M = m + j
			L = l - 2j
			for p1 in 0:M, p2 in 0:M-p1
				p3 = M - p1 - p2
				mc = ft[M+1] / (ft[p1+1] * ft[p2+1] * ft[p3+1])
				for q1 in 0:L, q2 in 0:L-q1
					q3 = L - q1 - q2
					v = cj * mc * ft[L+1] / (ft[q1+1] * ft[q2+1] * ft[q3+1]) *
						R[1]^q1 * R[2]^q2 * R[3]^q3
					i1 = 2p1 + q1; i2 = 2p2 + q2; i3 = 2p3 + q3
					a[i1+1, i2+1, i3+1] += v
					ab[i1+1, i2+1, i3+1] += abs(v)
				end
			end
		end
	end
	a, ab
end

"∫_{S²} u₁^a u₂^b u₃^c dΩ, exact."
@inline function sphMom(a::Int, b::Int, c::Int, ::Type{T}) where T
	(isodd(a) || isodd(b) || isodd(c)) && return zero(T)
	4 * T(pi) * dfac(a - 1, T) * dfac(b - 1, T) * dfac(c - 1, T) / dfac(a + b + c + 1, T)
end

"Multi-indices of total degree n, and the L²(S²) Gram matrix of the degree-n monomials."
function sphGram(n::Int, ::Type{T}) where T
	ix = [(i, j, n - i - j) for i in 0:n for j in 0:n-i]
	G = T[sphMom(x[1] + y[1], x[2] + y[2], x[3] + y[3], T) for x in ix, y in ix]
	G, ix
end

"M points of the Fibonacci spiral on S²."
function fibDir(M::Int, ::Type{T}) where T
	g = (sqrt(T(5)) - 1) / 2
	[begin
		z = 1 - T(2m + 1) / M
		r = sqrt(max(zero(T), 1 - z^2))
		ph = 2 * T(pi) * g * m
		(r * cos(ph), r * sin(ph), z)
	end for m in 0:M-1]
end

"Homogeneous degree-n part of a stored series evaluated at u."
function homEval(a::Array{C,3}, u::NTuple{3,T}, n::Int) where {T,C}
	s = zero(C)
	for i in 0:n, j in 0:n-i
		s += a[i+1, j+1, n-i-j+1] * u[1]^i * u[2]^j * u[3]^(n - i - j)
	end
	s
end
