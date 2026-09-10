# Build timings.  Runs against whichever GilaElectromagnetics the active project resolves to, so
# the same file times HEAD in a worktree and the live tree here.
using GilaElectromagnetics
pr(x...) = (println(x...); flush(stdout))
opt(frq) = (o = CPUKerOpt{Float64}(); o.frqPhz = frq; o)

pr("threads = ", Threads.nthreads(), "  loadavg = ", read(`uptime`, String) |> strip)
const CASES = [((32,32,32), (1//32,1//32,1//32)), ((64,64,64), (1//32,1//32,1//32)),
               ((128,128,128), (1//32,1//32,1//32)), ((32,32,32), (1//8,1//8,1//8)),
               ((32,32,32), (1//4,1//4,1//4))]
sel = length(ARGS) >= 1 ? parse.(Int, split(ARGS[1], ',')) : collect(1:length(CASES))
# warm the code paths on a tiny volume first
GlaVacOprMem(opt(1.0 + 0.0im), GlaVol((2,2,2), (1//32,1//32,1//32), (0//1,0//1,0//1)))
for i in sel
    cel, scl = CASES[i]
    t0 = time()
    m = GlaVacOprMem(opt(1.0 + 0.0im), GlaVol(cel, scl, (0//1,0//1,0//1)))
    t1 = time()
    pr("  ", cel, " ", Float64.(scl), "  cold build ", round(t1 - t0, digits = 3), " s")
    t2 = time()
    GlaVacOprMem(opt(1.0 + 0.0im), GlaVol(cel, scl, (0//1,0//1,0//1)))
    pr("  ", cel, " ", Float64.(scl), "  warm build ", round(time() - t2, digits = 3), " s")
    m = nothing; GC.gc()
end
pr("ALLDONE")
