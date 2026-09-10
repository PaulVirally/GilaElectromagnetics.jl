include(joinpath(@__DIR__, "..", "..", "moments", "moments.jl"))
function probe()
    a=b=c=1.0
    pA,pB = canonPanels("Vp", a,b,c)
    f = 2.0+2.0im
    t0=time_ns()
    mMax=-1
    rel = 1.0
    while mMax < 150
        mMax += 1
        _, rel = momentSeries(pA,pB,f,mMax)
        rel < 1e-16 && break
    end
    println("mMax=",mMax," rel=",rel," time=",(time_ns()-t0)/1e9,"s")
end
probe()
probe()
