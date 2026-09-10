include("/Users/pvirally/.julia/dev/GilaElectromagnetics/notes/farfield/farfield.jl")

# Isolate the rounding cost of boxAcc's plain Float64 running sum: take the exact
# same Float64-valued terms the library generates (h_l * Y_lm * Q_lm), and compare
# (a) the naive left-to-right Float64 accumulation the library performs against
# (b) a compensated (double-double, using the file's own twoSum/twoPrd/ddAdd) sum
# of the identical terms, against (c) the exact sum of those same Float64 values
# computed in BigFloat and rounded back -- the ground truth for the accumulation
# step alone (term generation error, e.g. in h_l/Y_lm, is a separate and already
# measured question and is deliberately not in scope here).

function ddSumComplex(terms::Vector{ComplexF64})
    accR = (0.0, 0.0); accI = (0.0, 0.0)
    for t in terms
        accR = ddAdd(accR, (real(t), 0.0))
        accI = ddAdd(accI, (imag(t), 0.0))
    end
    Complex(accR[1] + accR[2], accI[1] + accI[2])
end

function bigSumComplex(terms::Vector{ComplexF64})
    setprecision(BigFloat, 256) do
        sR = sum(BigFloat(real(t)) for t in terms)
        sI = sum(BigFloat(imag(t)) for t in terms)
        Complex(Float64(sR), Float64(sI))
    end
end

function termsA1(fs, ws, R6, L)
    a1 = boxAcc # placeholder, unused
    rr, e = phsSeed(R6, fs.frq)
    shrm!(ws, R6[1]/rr, R6[2]/rr, R6[3]/rr, L)
    hFill!(ws, rr, e, fs.k, L)
    Y = ws.Y; h = ws.h; lm = fs.whl.dLm; lv = fs.whl.dLv; Qd = fs.whl.Qd
    terms = ComplexF64[]
    naive = zero(ComplexF64)
    for j in eachindex(lm)
        lv[j] > L && break
        term = h[lv[j] + 1] * (Y[lm[j]] * Qd[1, j])
        push!(terms, term)
        naive += term
    end
    (terms, naive)
end

function report(tag, sQ, f, D)
    fs = farSetup(sQ, ComplexF64(f); disk = false)
    s6 = ntuple(d -> Float64(sQ[d]), 3)
    R6 = ntuple(d -> Float64(D[d]) * s6[d], 3)
    rr = sqrt(sum(R6[i]^2 for i in 1:3))
    L = boundL(fs, rr)
    if L < 0
        println(tag, " D=", D, ": route 1 not available (L<0), skipping")
        return
    end
    ws = FarWs(fs.Lw, Float64)
    terms, naive = termsA1(fs, ws, R6, L)
    amp = sum(abs, terms) / abs(naive)
    dd = ddSumComplex(terms)
    big = bigSumComplex(terms)
    errNaive = abs(naive - big)
    errDD = abs(dd - big)
    println(tag, "  D=", D, "  L=", L, "  nterms=", length(terms), "  |sum|=", abs(naive),
        "  l-sum amp Σ|term|/|sum| = ", amp)
    println("    naive Float64 vs BigFloat-exact-of-same-terms: abs err = ", errNaive,
        "  (", errNaive/eps(abs(big)), " ulp)")
    println("    compensated (dd) vs BigFloat-exact:            abs err = ", errDD,
        "  (", errDD/eps(abs(big)), " ulp)")
    println("    naive relative to |sum|: ", errNaive/abs(big), "   dd relative: ", errDD/abs(big))
end

const C32 = (QI(1,32), QI(1,32), QI(1,32))
const SL  = (QI(1,32), QI(1,32), QI(1,512))

println("=== cubic cell, near-boundary offsets (moderate amp) ===")
report("c32", C32, 1.0, (2,2,0))
report("c32", C32, 2.0+0.2im, (2,2,0))
report("c32", C32, 1.0, (2,0,0))
report("c32", C32, 1.0, (3,0,0))

println()
println("=== slender cell (1/32,1/32,1/512), short-axis offsets near rho=1 (worst amp) ===")
for n in (24, 25, 28, 32, 40, 64, 128)
    report("sl", SL, 1.0, (0,0,n))
end
report("sl", SL, 1.0+1.0im, (0,0,32))
report("sl", SL, 2.0+0.2im, (0,0,32))

println()
println("=== slender cell, offsets along the long (x) axis, near threshold ===")
for n in (2,3,4,6,8,12,16,24,32)
    report("sl-x", SL, 1.0, (n,0,0))
end
println()
println("=== slender cell, diagonal offsets ===")
for n in (2,3,4,8,16)
    report("sl-xy", SL, 1.0, (n,n,0))
end

println()
println("=== extreme aspect-1020 flat cell (1/32,1/32,1/32640), probing for high amp ===")
XS = (QI(1,32), QI(1,32), QI(1,32640))
for D in [(2,0,0),(3,0,0),(4,0,0),(2,2,0),(3,3,0),(4,4,0),(2,1,0),(3,1,0),(4,1,0),(8,0,0),(8,8,0),(1,0,0),(1,1,0)]
    report("xs", XS, 2.0+0.2im, D)
end
