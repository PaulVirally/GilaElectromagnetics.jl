using GilaElectromagnetics
const GV = GilaElectromagnetics.GilaVacuum
pr(x...) = (println(x...); flush(stdout))
pr("threads = ", Threads.nthreads(), "  ", strip(read(`uptime`, String)))
for (lbl, scl) in (("lambda/32 cube", (1//32,1//32,1//32)), ("slender 1:1:16", (1//32,1//32,1//512)),
                   ("coprime aniso", (3//64,5//64,7//64)), ("lambda/8 cube", (1//8,1//8,1//8)),
                   ("lambda/4 cube", (1//4,1//4,1//4)))
    v = GlaVol((4,4,4), scl, (0//1,0//1,0//1))
    o = CPUKerOpt{Float64}()
    e = zeros(ComplexF64, 3, 3, 5, 5, 5)
    t0 = time(); GV.cntBlk!(e, v, o); t1 = time()
    t2 = time(); GV.cntBlk!(e, v, o); t3 = time()
    pr("  ", rpad(lbl, 16), " cold ", round(t1 - t0, digits = 2), " s   memoized ",
       round(t3 - t2, digits = 3), " s")
end
pr("ALLDONE")
