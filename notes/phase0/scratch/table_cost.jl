include("/Users/pvirally/.julia/dev/GilaElectromagnetics/notes/farfield/farfield.jl")

const SCRATCHDIR = "/Users/pvirally/.julia/dev/GilaElectromagnetics/notes/phase0/scratch/coldtab"
rm(SCRATCHDIR; force=true, recursive=true)
mkpath(SCRATCHDIR)

dirsize(d) = isdir(d) ? sum(filesize(joinpath(d,f)) for f in readdir(d); init=0) : 0

const C32 = (QI(1,32), QI(1,32), QI(1,32))
const C8  = (QI(1,8), QI(1,8), QI(1,8))
const C4  = (QI(1,4), QI(1,4), QI(1,4))
const SL  = (QI(1,32), QI(1,32), QI(1,512))
const f1 = ComplexF64(1.0)

println("=== equal-cell production shapes, cold build then warm read, L = LMAX default (56) ===")
for (nm, s) in (("c32", C32), ("c8", C8), ("c4", C4), ("sl", SL))
    d = joinpath(SCRATCHDIR, nm)
    mkpath(d)
    GC.gc()
    t0 = time(); fs = farSetup(s, f1; disk = true, dir = d); tCold = time() - t0
    sz = dirsize(d)
    GC.gc()
    t0 = time(); fs2 = farSetup(s, f1; disk = true, dir = d); tWarm = time() - t0
    println(nm, "  s=", s, "  cold=", round(tCold,digits=3), " s   warm=", round(tWarm,digits=3),
        " s   file size=", round(sz/1e6,digits=2), " MB   Lw=", fs.Lw)
end

println()
println("=== two unequal pairs (ratio 8, ratio 16 vs g), cold then warm ===")
g = QI(1,32)
for (nm, sS) in (("r8", (8g,8g,8g)), ("r16", (16g,16g,16g)))
    d = joinpath(SCRATCHDIR, nm)
    mkpath(d)
    GC.gc()
    t0 = time(); fx = farSetupX(C32, sS, f1; disk = true, dir = d); tCold = time() - t0
    sz = dirsize(d)
    GC.gc()
    t0 = time(); fx2 = farSetupX(C32, sS, f1; disk = true, dir = d); tWarm = time() - t0
    println(nm, "  sT=", C32, " sS=", sS, "  cold=", round(tCold,digits=3), " s   warm=", round(tWarm,digits=3),
        " s   file size=", round(sz/1e6,digits=2), " MB   Lw=", fx.Lw)
end

println()
println("=== build time vs L, C32 shape (cold, fresh dir each time) ===")
for L in (32, 40, 48, 56, 64, 72)
    d = joinpath(SCRATCHDIR, "c32_L$(L)")
    mkpath(d)
    GC.gc()
    t0 = time(); fs = farSetup(C32, f1; disk = true, dir = d, L = L, lDef = L); tCold = time() - t0
    sz = dirsize(d)
    println("L=", L, "  cold=", round(tCold,digits=3), " s   file size=", round(sz/1e6,digits=2), " MB")
end
