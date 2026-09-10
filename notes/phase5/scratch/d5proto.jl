using GilaElectromagnetics
const GVM = GilaElectromagnetics.GilaVacuum
mMx = 10
shp = ((1/32,1/32,1/32), (1/32,1/32,1/512), (1/4,1/4,1/4), (1/32,1/16,1/8))
off = ((0,0,0),(1,0,0),(1,1,0),(1,1,1))
for lams in (((2.0, 0.5, 0.125), true), ((3.0, 1/10), false))
    lam, dyad = lams
    cnt = 0; bad = 0; wst = 0.0
    t = @elapsed for s in shp, D in off, F in 1:6, Fp in 1:6
        v = GVM.parMom(GVM.parFac(D, F, Fp, s)..., mMx)
        for l in lam
            w = GVM.parMom(GVM.parFac(D, F, Fp, l .* s)..., mMx)
            for m in -1:mMx
                cnt += 1
                r = l^(m+4) * v[m+2]
                dyad && (w[m+2] === r || (bad += 1))
                iszero(w[m+2]) || (wst = max(wst, abs(w[m+2] - r)/abs(w[m+2])))
            end
        end
    end
    println("dyadic=", dyad, "  cnt=", cnt, " bad=", bad, " worst rel=", wst, "  ", t, " s")
end
