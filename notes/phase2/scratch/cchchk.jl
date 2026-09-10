# the shape-table disk cache: off by default, on through the GlaVacOprMem keyword
using GilaElectromagnetics, Scratch
const GVF = GilaElectromagnetics.GilaVacuum
opt = CPUKerOpt{Float64}()
vol = GlaVol((3,3,3), (1//24,1//24,1//24), (0//1,0//1,0//1))
println("cold, cache on:  ", @elapsed GlaVacOprMem(opt, vol; shpCch = true), " s")
println("dir: ", GVF.TABDIR[])
println("files: ", [(f, filesize(joinpath(GVF.TABDIR[], f))) for f in readdir(GVF.TABDIR[])])
empty!(GVF.SHPC); empty!(GVF.FRQC)
println("warm, cache on:  ", @elapsed GlaVacOprMem(opt, vol; shpCch = true), " s")
empty!(GVF.SHPC); empty!(GVF.FRQC)
println("warm, cache off: ", @elapsed GlaVacOprMem(opt, vol), " s")
