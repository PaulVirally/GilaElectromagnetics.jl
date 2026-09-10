using GilaElectromagnetics
import GilaElectromagnetics.GilaVolumes: sepGrd, GlaVol

# does Gila's own machinery ever reject a genuinely misaligned equal-cell
# external pair, or does it silently build one?
scl = (1//16, 1//16, 1//16)
volA = GlaVol((4,4,4), scl, (0//1, 0//1, 0//1))
# origin offset by 1/3 of a cell in x -- not an integer multiple of scl, and
# not constructible by any refine()/composite snapping path
volB = GlaVol((4,4,4), scl, (1//1 + 1//48, 0//1, 0//1))
println("volA org=", volA.org, " volB org=", volB.org)
println("Δorg/s = ", (volB.org .- volA.org) ./ scl)

try
    opr = GlaVacOprMem(CPUKerOpt{Float64}(), volB, volA)
    println("GlaVacOprMem SUCCEEDED for the misaligned pair.")
    grd0 = sepGrd(volB, volA, 0)
    println("sepGrd(volB,volA,0) = ", grd0)
    println("start ./ s = ", getproperty.(grd0, :start) ./ scl,
        "  isinteger = ", isinteger.(getproperty.(grd0, :start) ./ scl))
catch e
    println("THREW: ", sprint(showerror, e))
end
