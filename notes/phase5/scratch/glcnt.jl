# Does the thin-axis Gauss-Legendre rule of section 4.4 ever fire on the contact path?  The
# generic method is shadowed by a counting method per concrete type, which forwards to it.
using GilaElectromagnetics, DoubleFloats
const GV = GilaElectromagnetics.GilaVacuum
const GLC = Ref(0)
pr(x...) = (println(x...); flush(stdout))
for D in (Float64, Double64, BigFloat)
    @eval GV function gauLeg(n::Int, ::Type{$D})
        Main.GLC[] += 1
        invoke(gauLeg, Tuple{Int,Type{T}} where {T}, n, $D)
    end
end

vols = (("self coarse cube", GlaVol((3,3,3), (1//8,1//8,1//8), (0//1,0//1,0//1))),)
for (lbl, v) in vols
    GLC[] = 0
    t = @elapsed GlaVacOprMem(CPUKerOpt{Float64}(), v)
    pr(rpad(lbl, 22), " gauLeg calls ", GLC[], "  ", round(t; digits = 2), " s")
end
# and one separated equal-cell external pair, where route 3 is reached and the rule should fire
let a = GlaVol((3,3,3), (1//32,1//32,1//32), (0//1,0//1,0//1)),
    b = GlaVol((3,3,3), (1//32,1//32,1//32), (1//4,0//1,0//1))
    GLC[] = 0
    t = @elapsed GlaVacOprMem(CPUKerOpt{Float64}(), b, a)
    pr(rpad("external separated", 22), " gauLeg calls ", GLC[], "  ", round(t; digits = 2), " s")
end
pr("ALLDONE")
