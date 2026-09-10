const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const NF = joinpath(ROOT, "notes", "farfield")
using GilaElectromagnetics
const GV = GilaElectromagnetics.GilaVacuum

macro mkmod(name, file)
    quote
        @eval module $(esc(name))
            const parFac = $(GV).parFac
            const parMom = $(GV).parMom
            const boxFace = $(GV).boxFace
            const FACES = $(GV).FACES
            const srfSum! = $(GV).srfSum!
            include($(esc(file)))
        end
    end
end
