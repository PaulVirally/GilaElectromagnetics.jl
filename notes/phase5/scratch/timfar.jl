using GilaElectromagnetics
const GV = GilaElectromagnetics.GilaVacuum
pr(x...) = (println(x...); flush(stdout))
pr("threads = ", Threads.nthreads(), "  ", strip(read(`uptime`, String)))
scl = (1//32, 1//32, 1//32); frq = 1.0 + 0.0im
sQ = ntuple(d -> Rational{BigInt}(scl[d]), 3)
for n in (32, 64, 128)
    e = zeros(ComplexF64, 3, 3, n, n, n)
    t0 = time(); GV.farBlk!(e, scl, frq); t1 = time()
    t2 = time(); GV.farBlk!(e, scl, frq); t3 = time()
    pr("  far fill ", n, "^3  first ", round(t1 - t0, digits = 3),
       " s   with the set held ", round(t3 - t2, digits = 3), " s")
    e = nothing; GC.gc()
end
# farSet cost alone, for the section 4.1 memo claim
for s in ((1//32,1//32,1//32), (3//64,5//64,7//64), (1//8,1//8,1//8))
    q = ntuple(d -> Rational{BigInt}(s[d]), 3)
    t0 = time(); GV.farSet(q, frq; nBlk = 128); pr("  farSet ", Float64.(s), " nBlk=128  ",
                                                   round(time() - t0, digits = 3), " s")
end
pr("ALLDONE")
