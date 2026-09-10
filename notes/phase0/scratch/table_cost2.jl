include("/Users/pvirally/.julia/dev/GilaElectromagnetics/notes/farfield/farfield.jl")

const SCRATCHDIR = "/Users/pvirally/.julia/dev/GilaElectromagnetics/notes/phase0/scratch/coldtab2"
rm(SCRATCHDIR; force=true, recursive=true)
mkpath(SCRATCHDIR)
dirsize(d) = isdir(d) ? sum(filesize(joinpath(d,f)) for f in readdir(d); init=0) : 0

const C32 = (QI(1,32), QI(1,32), QI(1,32))
const f1 = ComplexF64(1.0)
g = QI(1,32)

println("=== unequal pairs, forced to production-realistic Lw (lDef=54), cold then warm ===")
for (nm, sS) in (("r8", (8g,8g,8g)), ("r16", (16g,16g,16g)))
    d = joinpath(SCRATCHDIR, nm)
    mkpath(d)
    GC.gc()
    t0 = time(); fx = farSetupX(C32, sS, f1; disk = true, dir = d, lDef = 54); tCold = time() - t0
    sz = dirsize(d)
    GC.gc()
    t0 = time(); fx2 = farSetupX(C32, sS, f1; disk = true, dir = d, lDef = 54); tWarm = time() - t0
    println(nm, "  sT=", C32, " sS=", sS, "  cold=", round(tCold,digits=3), " s   warm=", round(tWarm,digits=3),
        " s   file size=", round(sz/1e6,digits=2), " MB   Lw=", fx.Lw)
end
