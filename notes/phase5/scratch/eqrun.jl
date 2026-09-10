include(joinpath(@__DIR__, "eqatk.jl"))
pr("threads = ", Threads.nthreads())
cases = [
    (:one,    :f1,  NTuple{3,Int}[(2,0,0), (0,0,2), (2,2,2), (3,1,0)]),
    (:one,    :f37, NTuple{3,Int}[(2,0,0), (2,2,2)]),
    (:half,   :fA,  NTuple{3,Int}[(2,0,0), (2,2,2)]),
]
for (sh, fr, ex) in cases
    try
        run(sh, fr; extra = ex, rmax = 24)
    catch e
        pr("  CASE FAILED ", sh, " ", fr, " : ", sprint(showerror, e)[1:min(end, 400)])
    end
end
pr("ALLDONE")
