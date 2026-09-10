include(joinpath(@__DIR__, "..", "..", "moments", "moments.jl"))
function probe(mMax)
    setprecision(BigFloat, 256) do
        a=b=c=BigFloat(1)
        pA,pB = canonPanels("Vp", a,b,c)
        f = Complex{BigFloat}(2.0,2.0)
        t0=time_ns()
        v, rel = momentSeries(pA,pB,f,mMax)
        println("BigFloat mMax=",mMax," rel=",Float64(rel)," time=",(time_ns()-t0)/1e9,"s")
    end
end
probe(40); probe(40); probe(80); probe(120)
