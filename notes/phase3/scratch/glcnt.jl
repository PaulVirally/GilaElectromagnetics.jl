using GilaElectromagnetics
const GVM = GilaElectromagnetics.GilaVacuum
volFin = GlaVol((4,4,4), (1//32,1//32,1//32), (0//1,0//1,0//1))
volCrs = GlaVol((2,2,2), (1//16,1//16,1//16), (0//1,0//1,0//1))
volSep = GlaVol((4,4,4), (1//32,1//32,1//32), (1//2,0//1,0//1))
volTch = GlaVol((4,4,4), (1//32,1//32,1//32), (4//32,0//1,0//1))
volSld = GlaVol((6,6,12), (1//32,1//32,1//512), (0//1,0//1,0//1))
for (lbl, f) in (("self fine", () -> GlaVacOprMem(CPUKerOpt{Float64}(), volFin)),
                 ("self slender", () -> GlaVacOprMem(CPUKerOpt{Float64}(), volSld)),
                 ("equal separated", () -> GlaVacOprMem(CPUKerOpt{Float64}(), volSep, volFin)),
                 ("cross-scale separated", () -> GlaVacOprMem(CPUKerOpt{Float64}(), volCrs, volSep)),
                 ("cross-scale touching", () -> GlaVacOprMem(CPUKerOpt{Float64}(), volCrs, volTch)))
    GVM.GLCNT[] = 0
    t = @elapsed f()
    println(rpad(lbl, 24), " gauLeg calls ", GVM.GLCNT[], "  ", round(t; digits = 2), " s")
end
