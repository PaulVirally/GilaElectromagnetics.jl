using GilaElectromagnetics, Serialization
const GV = GilaElectromagnetics.GilaVacuum
e = zeros(ComplexF64, 3, 3, 8, 8, 8)
GV.farBlk!(e, (3//64,5//64,7//64), 1.0 + 0.1im)
serialize(joinpath(@__DIR__, "thr_$(Threads.nthreads()).bin"), e)
println("wrote ", Threads.nthreads())
