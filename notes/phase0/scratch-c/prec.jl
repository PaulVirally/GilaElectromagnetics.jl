using Printf
@printf("nthreads = %d\n", Threads.nthreads())
@printf("main, before      : %d\n", precision(BigFloat))
setprecision(BigFloat, 128)
@printf("main, after set   : %d\n", precision(BigFloat))
p = zeros(Int, Threads.maxthreadid())
Threads.@threads for i in 1:Threads.nthreads()
    p[Threads.threadid()] = precision(BigFloat)
end
@printf("inside @threads   : %s\n", string(sort(unique(p))))
r = fetch(Threads.@spawn precision(BigFloat))
@printf("inside @spawn     : %d\n", r)
q = Ref(0)
setprecision(BigFloat, 200) do
    q[] = fetch(Threads.@spawn precision(BigFloat))
end
@printf("spawn inside do-block (200): %d\n", q[])
