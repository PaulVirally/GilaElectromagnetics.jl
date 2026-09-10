include("/Users/pvirally/.julia/dev/GilaElectromagnetics/notes/farfield/farfield.jl")

const GQ = (QI(1,32), QI(1,32), QI(1,32))
const f1 = ComplexF64(1.0)
g = GQ[1]
sS = (16g, 16g, 16g)
b  = ntuple(d -> (GQ[d] + sS[d]) // 2, 3)

Rs = [(QI(2k1 + 1)//64, QI(2k2 + 1)//64, QI(2k3 + 1)//64) for k3 in -56:55 for k2 in -56:55 for k1 in 8:119]
xTouch(R, b) = all(abs(R[d]) <= b[d] for d in 1:3)
filter!(R -> !xTouch(R, b), Rs)
println("offsets: ", length(Rs))

for Ltry in (56, 60, 64)
    GC.gc()
    t0 = time()
    fx = farSetupX(GQ, sS, f1; offs = Rs, L = Ltry, lDef = Ltry)
    tt = time() - t0
    println("L = ", Ltry, "  farSetupX (full block, cold) = ", round(tt,digits=2), " s  Lw(stored) = ", fx.Lw,
        "  maxrss = ", round(Sys.maxrss()/1e9,digits=2), " GB")
end
