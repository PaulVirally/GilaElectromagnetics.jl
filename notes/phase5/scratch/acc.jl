# Section 7 acceptance checks that are not timings.
using GilaElectromagnetics, LinearAlgebra
const GV = GilaElectromagnetics.GilaVacuum
pr(x...) = (println(x...); flush(stdout))
mx(A) = Float64(maximum(abs, A))

pr("### genPrc = Float32 must raise")
try
    CPUKerOpt{Float64}(1.0 + 0.0im, Float32, false, GilaElectromagnetics.GilaVacuum.CPU())
    pr("  constructor: NO ERROR RAISED -- defect")
catch e
    pr("  constructor raised: ", first(split(sprint(showerror, e), '\n')))
end
try
    o = CPUKerOpt{Float64}(); o.genPrc = Float32
    m = GlaVacOprMem(o, GlaVol((2,2,2), (1//32,1//32,1//32), (0//1,0//1,0//1)))
    pr("  field assignment: NO ERROR RAISED; finite egoFur: ",
       all(all(isfinite, real.(a)) && all(isfinite, imag.(a)) for a in m.egoFur))
catch e
    pr("  field assignment raised: ", first(split(sprint(showerror, e), '\n')))
end

pr("### fp32 storage is the narrowed fp64 build")
for (cel, scl, frq) in (((4,4,4), (1//32,1//32,1//32), 1.0 + 0.1im),
                        ((3,4,5), (3//64,5//64,7//64), 0.5 + 2.0im),
                        ((4,4,8), (1//32,1//32,1//512), 1.0 + 0.0im))
    o6 = CPUKerOpt{Float64}(); o6.frqPhz = ComplexF64(frq)
    o3 = CPUKerOpt{Float32}(); o3.frqPhz = ComplexF32(frq)
    v = GlaVol(cel, scl, (0//1,0//1,0//1))
    m6 = GlaVacOprMem(o6, v); m3 = GlaVacOprMem(o3, v)
    ok = all(ComplexF32.(m6.egoFur[i]) == m3.egoFur[i] for i in eachindex(m6.egoFur))
    pr("  cel=", cel, " scl=", Float64.(scl), " f=", frq, "  bitwise equal: ", ok)
end

pr("### asym(G0) PSD")
dnsSlf(mem) = begin
    n = prod(mem.srcVol.cel) * 3
    m = zeros(ComplexF64, n, n)
    for i in 1:n
        x = zeros(ComplexF64, mem.srcVol.cel..., 3); x[i] = 1
        m[:, i] .= vec(egoOpr!(mem, x))
    end
    m
end
for (cel, scl) in (((6,6,6), (1//32,1//32,1//32)), ((6,6,12), (1//32,1//32,1//512)),
                   ((4,4,4), (3//64,5//64,7//64)), ((4,4,4), (1//8,1//24,1//48)),
                   ((3,3,6), (1//6,1//96,1//96)))
    for frq in (1.0 + 0.0im, 1.0 + 0.1im)
        o = CPUKerOpt{Float64}(); o.frqPhz = frq
        m = dnsSlf(GlaVacOprMem(o, GlaVol(cel, scl, (0//1,0//1,0//1))))
        ev = eigvals(Hermitian((m - adjoint(m)) / 2im))
        pr("  cel=", cel, " scl=", Float64.(scl), " f=", frq,
           "  lamMin = ", minimum(ev), "  lamMin/lamMax = ", minimum(ev) / maximum(ev))
    end
end
pr("ALLDONE")
