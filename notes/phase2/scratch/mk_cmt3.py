src = open("far_v2_cmt.jl").read()
def rep(old, new, n=1):
    global src
    c = src.count(old); assert c == n, f"count {c} != {n} for:\n{old[:160]}"
    src = src.replace(old, new)

rep('''"A proven lower bound on max_ab |T_ab(R)|; 0 when the trace argument does not apply."
estLo(fs::FrqSet, rr::Real) =
    lowVal(fs.wLs, fs.e0, fs.eT, ComplexF64(fs.k), Float64(rr))
# the scale the selector divides by: est by default, else the proven lower bound
estScl(fs::FrqSet, rr::Real) = fs.scl === :low ? estLo(fs, rr) :
    est(Float64(rr), ComplexF64(fs.k), ComplexF64(fs.frq), Float64(prod(fs.s)))''',
'''# the scale the selector divides by: est by default, else the proven lower bound
estScl(fs::FrqSet, rr::Real) =
    fs.scl === :low ? lowVal(fs.wLs, fs.e0, fs.eT, ComplexF64(fs.k), Float64(rr)) :
    est(Float64(rr), ComplexF64(fs.k), ComplexF64(fs.frq), Float64(prod(fs.s)))''')
rep("    e0::Float64                     # leading factor of the proven lower bound estLo",
    "    e0::Float64                     # leading factor of the proven lower bound lowVal")

rep('''qTok(x::QI) = string(numerator(x), " ", denominator(x))
# a cache line is "r1n r1d r2n r2d r3n r3d N Lambda re im ..." (26 tokens), one file per ordered pair''',
    '''# a cache line is "r1n r1d r2n r2d r3n r3d N Lambda re im ..." (26 tokens), one file per ordered pair''')
rep('''                    print(io, qTok(RQ[1]), " ", qTok(RQ[2]), " ", qTok(RQ[3]), " ", N, " ", amp)''',
    '''                    for d in 1:3; print(io, numerator(RQ[d]), " ", denominator(RQ[d]), " "); end
                    print(io, N, " ", amp)''')
open("far_v2_cmt.jl", "w").write(src)
print("ok", len(src.splitlines()))
