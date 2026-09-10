using GilaElectromagnetics, Printf
const GVF = GilaElectromagnetics.GilaVacuum
volCrs = GlaVol((2,2,2), (1//16,1//16,1//16), (0//1,0//1,0//1))
volFin = GlaVol((4,4,4), (1//32,1//32,1//32), (1//2,0//1,0//1))
t1 = @elapsed GlaVacOprMem(CPUKerOpt{Float64}(), volCrs, volFin)
n1 = length(GVF.FRQCX)
t2 = @elapsed GlaVacOprMem(CPUKerOpt{Float64}(), volCrs, volFin)
@printf("8 source partitions: first build %.2f s, %d FrqSetX cached; rebuild %.2f s, %d cached\n",
    t1, n1, t2, length(GVF.FRQCX))
