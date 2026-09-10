include("/Users/pvirally/.julia/dev/GilaElectromagnetics/notes/farfield/farfield.jl")

const GQ = (QI(1,32), QI(1,32), QI(1,32))
const f1 = ComplexF64(1.0)
g = GQ[1]
sS = (16g, 16g, 16g)

# isolate pure shape-table + threshold build cost vs L, disk cache off so every
# call is a genuine cold build (no contamination from earlier scratch runs)
for Ltry in (56, 58, 60, 64, 72, 80)
    GC.gc()
    t0 = time()
    fx = farSetupX(GQ, sS, f1; offs = (), L = Ltry, lDef = Ltry, disk = false)
    tt = time() - t0
    println("L = ", Ltry, "  cold table build (no offs, disk off) = ", round(tt,digits=3), " s  Lw = ", fx.Lw,
        "  maxrss = ", round(Sys.maxrss()/1e9,digits=2), " GB")
end
