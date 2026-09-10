# Where does a coarse grid stop disagreeing with a 2x refined one, and what sets
# that distance? Sweeps body extent along the gap, lateral extent, cube size,
# anisotropic cells and complex frequency.
using GilaElectromagnetics, LinearAlgebra, Printf

function dns(mem)
    T = eltype(first(mem.egoFur))
    nn = prod(mem.srcVol.cel) * 3
    mm = prod(mem.trgVol.cel) * 3
    A = zeros(T, mm, nn)
    for i in 1:nn
        v = zeros(T, mem.srcVol.cel..., 3)
        v[i] = one(T)
        A[:, i] .= vec(egoOpr!(mem, v))
    end
    A
end

function sigs(cel, scl, xOrg, frq, kTop)
    opt = CPUKerOpt{Float64}()
    opt.frqPhz = frq
    volT = GlaVol(cel, scl, (0//1, 0//1, 0//1))
    volS = GlaVol(cel, scl, (xOrg, 0//1, 0//1))
    A = dns(GlaVacOprMem(opt, volT, volS))
    s = svdvals(A)[1:kTop]
    A = nothing
    GC.gc()
    s
end

# interpolate the gap at which ratio-to-floor falls through thr
function xOver(ds, rs, thr)
    i = findlast(>=(thr), rs)
    i === nothing && return 0.0
    i == length(rs) && return NaN
    t = (log(thr) - log(rs[i])) / (log(rs[i+1]) - log(rs[i]))
    ds[i] + t * (ds[i+1] - ds[i])
end

function run(tag, cel, scl, jMax, frq; kTop = 20)
    celF = 2 .* cel
    sclF = scl .// 2
    ext = cel[1] * scl[1]
    ds = Float64[]
    r1 = Float64[]
    r5 = Float64[]
    for j in 1:jMax
        xOrg = ext + j * scl[1]
        sC = sigs(cel, scl, xOrg, frq, kTop)
        sF = sigs(celF, sclF, xOrg, frq, kTop)
        rel = abs.(sC .- sF) ./ sF
        push!(ds, float(j * scl[1]))
        push!(r1, rel[1])
        push!(r5, maximum(rel[1:5]))
    end
    f1, f5 = r1[end], r5[end]
    @printf("%-22s cel=%-10s scl=%-22s frq=%-10s ext=%.4f\n",
            tag, string(cel), string(float.(scl)), string(frq), float(ext))
    @printf("    floor sig1=%.3e top5=%.3e | xover(1.05) sig1 d=%.4f (%.2f cel) top5 d=%.4f (%.2f cel)\n",
            f1, f5, xOver(ds, r1 ./ f1, 1.05), xOver(ds, r1 ./ f1, 1.05) / float(scl[1]),
            xOver(ds, r5 ./ f5, 1.05), xOver(ds, r5 ./ f5, 1.05) / float(scl[1]))
    print("    d/lam :"); for d in ds; @printf(" %7.4f", d); end; println()
    print("    s1/flr:"); for r in r1 ./ f1; @printf(" %7.3f", r); end; println()
    print("    s5/flr:"); for r in r5 ./ f5; @printf(" %7.3f", r); end; println()
    flush(stdout)
end

const C = 1//16
const S16 = (C, C, C)

function main()
    println("### A. body extent ALONG the gap (lateral 4x4, cells lambda/16)")
    for t in (1, 2, 4, 6, 8)
        run("thick=$t", (t, 4, 4), S16, 12, 1.0 + 0.0im)
    end

    println("\n### B. lateral extent (thickness 4 along gap, cells lambda/16)")
    for L in (2, 4, 6)
        run("lat=$L", (4, L, L), S16, 12, 1.0 + 0.0im)
    end

    println("\n### C. cubes (cells lambda/16)")
    for n in (2, 4, 6)
        run("cube=$n", (n, n, n), S16, 12, 1.0 + 0.0im)
    end

    println("\n### D. same bodies one octave finer (cells lambda/32)")
    run("cube=4 fine", (4, 4, 4), (1//32, 1//32, 1//32), 16, 1.0 + 0.0im)
    run("cube=8 fine", (8, 8, 8), (1//32, 1//32, 1//32), 16, 1.0 + 0.0im)

    println("\n### E. anisotropic cells")
    run("aniso gap-fine", (8, 2, 2), (1//32, 1//8, 1//8), 16, 1.0 + 0.0im)
    run("aniso gap-coarse", (2, 8, 8), (1//8, 1//32, 1//32), 8, 1.0 + 0.0im)
    run("aniso 3-way", (2, 4, 8), (1//16, 1//32, 1//64), 12, 1.0 + 0.0im)

    println("\n### F. complex frequency (cube 4, cells lambda/16)")
    for f in (1.0 + 0.0im, 1.0 + 0.05im, 1.0 + 0.1im, 1.0 + 0.2im, 1.0 + 0.5im, 1.0 + 1.0im)
        run("frq=$f", (4, 4, 4), S16, 12, f)
    end
end

main()
