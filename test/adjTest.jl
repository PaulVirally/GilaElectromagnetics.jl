using Test

Random.seed!(0xdeadbeef)

slfOprHst = GlaOpr((8, 8, 8), (1//32, 1//32, 1//32); setTyp=ComplexF32, useGpu=false)
# extOprHst = GlaOpr((6, 4, 2), (1//32, 1//32, 1//32), (0//1, 0//1, 0//1), (8, 2, 10), (1//64, 1//64, 1//64), (147//3, 0//1, 0//1); setTyp=ComplexF64, useGpu=false)
extOprHst = GlaOpr((6, 4, 2), (1//32, 1//32, 1//32), (0//1, 0//1, 0//1), (8, 2, 10), (1//32, 1//32, 1//32), (147//3, 0//1, 0//1); setTyp=ComplexF32, useGpu=false)
if CUDA.functional()
	slfOprDev = GlaOpr((8, 8, 8), (1//32, 1//32, 1//32); setTyp=ComplexF32, useGpu=true)
	extOprHst = GlaOpr((6, 4, 2), (1//32, 1//32, 1//32), (0//1, 0//1, 0//1), (8, 2, 10), (1//64, 1//64, 1//64), (147//3, 0//1, 0//1); setTyp=ComplexF64, useGpu=true)
end

function tstSlf(G::GlaOpr, Gdag::GlaOpr; num_tests::Int=100)
	@assert isadjoint(G) != isadjoint(Gdag) "Exactly one of the operators needs to be an adjoint"
	@assert isselfoperator(G) "G needs to be a self operator"
	@assert isselfoperator(Gdag) "Gdag needs to be a self operator"
	@assert glaSze(G) == glaSze(Gdag) "Both operators need to have the same size"
	@assert eltype(G) == eltype(Gdag) "Both operators need to have the same element type"
	isGpu = G.mem.cmpInf.devMod
	@testset "Statistical adjoint self operator test for size $(glaSze(G, 2))" begin
		for _ in 1:num_tests
			if isGpu
				x = CUDA.rand(eltype(G), size(G, 2))
			else
				x = rand(eltype(G), size(G, 2))
			end
			@test Gdag * x ≈ conj.(G * conj.(x))
			@test conj.(Gdag * x) ≈ G * conj.(x)
		end
	end
end

function dnsMat(G::GlaOpr)
	isGpu = G.mem.cmpInf.devMod
	mat = zeros(eltype(G), size(G, 1), size(G, 2))
	if isGpu
		mat = CUDA.zeros(eltype(G), size(G, 1), size(G, 2))
	end
	for i in 1:size(G, 2)
		v = zeros(eltype(G), size(G, 2))
		v[i] = one(eltype(G))
		if isGpu
			v = CuArray(v)
		end
		mat[:, i] .= G * v
	end
	return mat
end

function tstMat(G::GlaOpr, Gdag::GlaOpr)
	mat = dnsMat(G)
	adjMat = dnsMat(Gdag)
	@testset "Dense adjoint" begin
		@test mat' ≈ adjMat
		@test mat ≈ adjMat'
	end
end

Gops = [slfOprHst, extOprHst]
if CUDA.functional()
	append!(Gops, [slfOprDev, extOprHst])
end

for G in Gops
	Gdag = adjoint(G)
	if isselfoperator(G)
		tstSlf(G, Gdag)
	end
	tstMat(G, Gdag)
end
