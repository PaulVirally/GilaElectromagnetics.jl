using Test

Random.seed!(0xdeadbeef)

slfOprHst = GlaOpr(oprSlfHst)
extOprHst = GlaOpr(oprExtHst)
mrgOprHst = GlaOpr(oprMrgHst)
if CUDA.functional()
	extOprDev = GlaOpr(oprExtDev)
	mrgOprDev = GlaOpr(oprMrgDev)
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
			adjOut = Gdag * x
			fkeAdjOut = conj.(G * conj.(x))
			@test adjOut ≈ fkeAdjOut
		end
	end
end

Gops = [slfOprHst, extOprHst, mrgOprHst]
if CUDA.functional()
	append!(Gops, [extOprDev, mrgOprDev])
end

for G in Gops
	if isselfoperator(G)
		Gdag = adjoint(G)
		tstSlf(G, Gdag)
	else
		# TODO: External operator tests
	end
end
