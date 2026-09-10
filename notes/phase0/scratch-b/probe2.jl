include(joinpath(@__DIR__, "..", "..", "moments", "moments.jl"))
function probe(mMax)
    a=b=c=1.0
    pA,pB = canonPanels("Vp", a,b,c)
    f = 2.0+2.0im
    t0=time_ns()
    v, rel = momentSeries(pA,pB,f,mMax)
    println("mMax=",mMax," rel=",rel," time=",(time_ns()-t0)/1e9,"s")
end
probe(20)
probe(20)
probe(40)
probe(60)
probe(80)
