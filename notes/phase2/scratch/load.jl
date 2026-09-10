t0 = time()
using GilaElectromagnetics
println("gila loaded ", round(time()-t0, digits=1), " s")
const GV = GilaElectromagnetics.GilaVacuum
const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
module FarOld
include(joinpath(normpath(joinpath(@__DIR__, "..", "..", "..")), "notes", "farfield", "farfield.jl"))
end
println("old loaded")
module FarNew
const parFac = Main.GV.parFac
const parMom = Main.GV.parMom
const boxFace = Main.GV.boxFace
const FACES = Main.GV.FACES
const srfSum! = Main.GV.srfSum!
include(joinpath(normpath(joinpath(@__DIR__, "..", "..", "..")), "notes", "phase2", "scratch", "far_v1.jl"))
end
println("new loaded")
