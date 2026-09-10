include("/Users/pvirally/.julia/dev/GilaElectromagnetics/notes/farfield/farfield.jl")

const C32 = (QI(1,32), QI(1,32), QI(1,32))
const f1 = ComplexF64(1.0)
g = QI(1,32)

println("=== farSetup (equal-cell, C32) time vs nBlk, disk off (pure compute) ===")
for nBlk in (8, 16, 32, 64, 128, 256, 512)
    GC.gc()
    t0 = time(); fs = farSetup(C32, f1; disk = false, nBlk = nBlk); tt = time() - t0
    println("nBlk=", nBlk, "  time=", round(tt,digits=3), " s   rHi=", round(fs.rHi,digits=2),
        "  nMx=", fs.nMx, "  Lw=", fs.Lw)
end

println()
println("=== farSetupX (16g pair) time vs nBlk, disk off ===")
sS = (16g,16g,16g)
for nBlk in (8, 16, 32, 64, 128, 256, 512)
    GC.gc()
    t0 = time(); fx = farSetupX(C32, sS, f1; disk = false, nBlk = nBlk, lDef = 54); tt = time() - t0
    println("nBlk=", nBlk, "  time=", round(tt,digits=3), " s   rHi=", round(fx.rHi,digits=2), "  Lw=", fx.Lw)
end
