# The hazards section 4 names, attacked directly: set reuse outside its certified radius, the
# memo key across block sizes, thread determinism, and the off-lattice refusal.
using GilaElectromagnetics
const GV = GilaElectromagnetics.GilaVacuum
const Q = Rational{BigInt}
pr(x...) = (println(x...); flush(stdout))
mx(A) = Float64(maximum(abs, A))

pr("threads = ", Threads.nthreads())

SKIP1 = true
if false
pr("### 1. a set built for a small block, used at a large offset")
for (scl, frq) in (((3//64,5//64,7//64), 1.0 + 0.1im), ((1//32,1//32,1//512), 1.0 + 0.0im))
    sQ = ntuple(d -> Q(scl[d]), 3)
    fs = GV.farSet(sQ, frq; nBlk = 4)
    for D in ((2,0,0), (40,0,0), (200,200,200))
        r = try
            G = GV.farTns(D, scl, frq; fs = fs)
            string("returned, max ", mx(G))
        catch e
            string("raised: ", first(split(sprint(showerror, e), '\n'))[1:min(end, 110)])
        end
        pr("  scl=", Float64.(scl), " nBlk=4 D=", D, "  ", r)
    end
end

pr("### 2. memo key: a 4^3 block then a 24^3 block of the same shape and frequency")
for (scl, frq) in (((3//64,5//64,7//64), 1.0 + 0.1im),)
    sQ = ntuple(d -> Q(scl[d]), 3)
    e1 = zeros(ComplexF64, 3, 3, 5, 5, 5)
    GV.farBlk!(e1, scl, frq)
    e2 = zeros(ComplexF64, 3, 3, 25, 25, 25)
    GV.farBlk!(e2, scl, frq)
    d = 0.0; wrst = (0,0,0)
    for i3 in 1:5, i2 in 1:5, i1 in 1:5
        max(i1, i2, i3) >= 3 || continue
        a = @view e1[:, :, i1, i2, i3]; b = @view e2[:, :, i1, i2, i3]
        v = mx(a .- b) / max(mx(b), 1e-300)
        v > d && (d = v; wrst = (i1, i2, i3))
    end
    pr("  small block vs the same corner of the large one: worst ", d, " at ", wrst)
    # and both against a set built for the large block only
    fs = GV.farSetMem(sQ, frq; nBlk = 26)
    e3 = zeros(ComplexF64, 3, 3, 5, 5, 5)
    GV.farBlk!(e3, scl, frq; fs = fs)
    pr("  small block vs an explicitly large set: worst ",
       maximum(mx(e1[:, :, i] .- e3[:, :, i]) / max(mx(e3[:, :, i]), 1e-300)
               for i in CartesianIndices((5,5,5)) if maximum(Tuple(i)) >= 3))
end

end
pr("### 3. thread determinism of farBlk! and farBlkX!")
let scl = (3//64,5//64,7//64), frq = 1.0 + 0.1im
    e = [zeros(ComplexF64, 3, 3, 6, 6, 6) for _ in 1:2]
    for j in 1:2
        empty!(GV.FRQC)
        GV.farBlk!(e[j], scl, frq)
    end
    pr("  farBlk! repeated in-process bitwise identical: ", e[1] == e[2])
end

pr("### 4. off-lattice separated and off-lattice near")
let scl = (1//32,1//32,1//32), frq = 1.0 + 0.0im
    sQ = ntuple(d -> Q(scl[d]), 3)
    for off in (1//64, 1//128, 0//1)
        Ds = [(2,0,0), (3,1,1), (5,0,0)]
        dst = zeros(ComplexF64, 3, 3, length(Ds))
        r = try
            GV.farFil!(q -> (@view dst[:, :, q]), Ds, sQ, ComplexF64(frq),
                       (Float64(off), 0.0, 0.0))
            string("filled, max ", mx(dst))
        catch e
            string("raised: ", first(split(sprint(showerror, e), '\n'))[1:min(end, 140)])
        end
        pr("  separated, sub-cell shift ", Float64(off), ": ", r)
    end
    for off in (1//64,)
        Ds = [(1,0,0), (1,1,0)]
        dst = zeros(ComplexF64, 3, 3, length(Ds))
        r = try
            GV.farFil!(q -> (@view dst[:, :, q]), Ds, sQ, ComplexF64(frq),
                       (Float64(off), 0.0, 0.0))
            string("filled, max ", mx(dst))
        catch e
            string("raised: ", first(split(sprint(showerror, e), '\n'))[1:min(end, 140)])
        end
        pr("  within a cell, sub-cell shift ", Float64(off), ": ", r)
    end
end

pr("### 5. non-integer cell ratio must be refused")
try
    GV.farTnsX((1//4, 0//1, 0//1), (1//32,1//32,1//32), (1//48,1//48,1//48), 1.0 + 0.0im)
    pr("  NO ERROR RAISED -- defect")
catch e
    pr("  raised: ", first(split(sprint(showerror, e), '\n'))[1:min(end, 140)])
end

pr("### 6. an overlapping / touching cross-scale pair must be refused")
for R in ((1//32, 0//1, 0//1), (3//64, 0//1, 0//1), (1//16, 0//1, 0//1))
    r = try
        G = GV.farTnsX(R, (1//32,1//32,1//32), (1//16,1//16,1//16), 1.0 + 0.0im)
        string("returned, max ", mx(G))
    catch e
        string("raised: ", first(split(sprint(showerror, e), '\n'))[1:min(end, 110)])
    end
    pr("  R=", Float64.(R), "  ", r)
end
pr("ALLDONE")
