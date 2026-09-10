using GilaElectromagnetics, LinearAlgebra
const GV = GilaElectromagnetics.GilaVacuum
pr(x...) = (println(x...); flush(stdout))
mx(A) = Float64(maximum(abs, A))

pr("### fp32 storage is the narrowed fp64 build (same ComplexF64 frequency in both)")
for (cel, scl, frq) in (((4,4,4), (1//32,1//32,1//32), 1.0 + 0.0im),
                        ((4,4,4), (1//32,1//32,1//32), 1.0 + 0.1im),
                        ((3,4,5), (3//64,5//64,7//64), 0.5 + 2.0im),
                        ((4,4,8), (1//32,1//32,1//512), 1.0 + 0.0im))
    o6 = CPUKerOpt{Float64}(); o6.frqPhz = frq
    o3 = CPUKerOpt{Float32}(); o3.frqPhz = frq
    v = GlaVol(cel, scl, (0//1,0//1,0//1))
    m6 = GlaVacOprMem(o6, v); m3 = GlaVacOprMem(o3, v)
    ok = all(ComplexF32.(a) == b for (a, b) in zip(m6.egoFur, m3.egoFur))
    wd = maximum(mx(ComplexF32.(a) .- b) / max(mx(b), 1e-30) for (a, b) in zip(m6.egoFur, m3.egoFur))
    pr("  cel=", cel, " scl=", Float64.(scl), " f=", frq, "  bitwise: ", ok, "  worst rel: ", wd)
end

pr("### the Float32 refusal can be bypassed by assigning the field; does the build then break?")
for cel in ((2,2,2), (8,8,8))
    o = CPUKerOpt{Float64}(); o.genPrc = Float32
    try
        m = GlaVacOprMem(o, GlaVol(cel, (1//32,1//32,1//32), (0//1,0//1,0//1)))
        nn = sum(count(!isfinite, real.(a)) + count(!isfinite, imag.(a)) for a in m.egoFur)
        o6 = CPUKerOpt{Float64}()
        m6 = GlaVacOprMem(o6, GlaVol(cel, (1//32,1//32,1//32), (0//1,0//1,0//1)))
        pr("  cel=", cel, "  built; non-finite entries: ", nn,
           "  worst rel vs the fp64 build: ",
           maximum(mx(a .- b) / max(mx(b), 1e-30) for (a, b) in zip(m.egoFur, m6.egoFur)))
    catch e
        pr("  cel=", cel, "  raised: ", first(split(sprint(showerror, e), '\n'))[1:min(end, 140)])
    end
end
pr("ALLDONE")
