using GilaElectromagnetics
import GilaElectromagnetics.GilaVolumes: sepGrd, GlaVol, GlaCmpVol, refine, regions, nregions

const cmpScl16 = (1//16, 1//16, 1//16)
const cmpScl32 = (1//32, 1//32, 1//32)
const cmpOrg0 = (0//1, 0//1, 0//1)

function report(name, trg::GlaVol, src::GlaVol)
    println("=== ", name, " ===")
    println("trg: cel=", trg.cel, " scl=", trg.scl, " org=", trg.org)
    println("src: cel=", src.cel, " scl=", src.scl, " org=", src.org)
    if trg.scl != src.scl
        println("  DIFFERENT SCALE -- not equal-cell, skip")
        return nothing
    end
    s = trg.scl
    # replicate the constant offset sepGrd bakes in: grid-start difference
    grdTrg = sepGrd(trg, src, 0)
    grdSrc = sepGrd(trg, src, 1)
    strTrg = getproperty.(trg.grd, :start) .- getproperty.(src.grd, :start)
    println("  grid-start diff (trg.grd.start - src.grd.start) = ", strTrg)
    ratio = strTrg ./ s
    println("  ratio / s = ", ratio, "   (integer? ", isinteger.(ratio), ")")
    println("  sepGrd(trg,src,0) = ", grdTrg)
    println("  sepGrd(trg,src,1) = ", grdSrc)
    return strTrg, s, ratio
end

# --- extSlfTest.jl pair ---
volSrc1 = GlaVol((8,8,8), (1//32,1//32,1//32), (0//1,0//1,0//1))
volTrg1 = GlaVol((8,8,8), (1//32,1//32,1//32), (1//1,1//1,1//1))
report("extSlfTest self/ext pair", volTrg1, volSrc1)

# --- crsSclTest.jl same-scale touching pair (contact, not separated -- included for completeness) ---
stdOrg = (0//1,0//1,0//1)
_xsSclFin = (1//32,1//32,1//32)
volSrcT = GlaVol((4,4,4), _xsSclFin, stdOrg)
volTrgT = GlaVol((4,4,4), _xsSclFin, (4//32,0//1,0//1))
report("crsSclTest same-scale touching", volTrgT, volSrcT)

# --- cmpOprTest.jl geometry ---
mnyVol = GlaVol((4,4,4), cmpScl16, cmpOrg0)
mnyCvl = refine(GlaCmpVol(mnyVol), ((-1//16,0//1,0//1), (1//8,1//4,1//4)))
println("\nmnyCvl regions:")
for (i,r) in enumerate(regions(mnyCvl))
    println("  region ", i, ": cel=", r.cel, " scl=", r.scl, " org=", r.org)
end

srcCvlBody = GlaCmpVol(GlaVol((2,2,2), cmpScl16, (1//2,0//1,0//1)))
# region 2 of mnyCvl is the coarse (1/16) region -- same-scale external pair
report("cmpOprTest 'between two bodies': mny coarse region vs srcCvl",
    regions(mnyCvl)[2], regions(srcCvlBody)[1])

triCvl = refine(GlaCmpVol(GlaVol((6,4,4), cmpScl16, cmpOrg0)),
    (cmpOrg0, (1//8,1//4,1//4)))
println("\ntriCvl regions:")
for (i,r) in enumerate(regions(triCvl))
    println("  region ", i, ": cel=", r.cel, " scl=", r.scl, " org=", r.org)
end
# regions 2 and 3 should be the two coarse regions, apart in x
if nregions(triCvl) >= 3
    report("cmpOprTest triCvl: coarse region 2 vs coarse region 3",
        regions(triCvl)[2], regions(triCvl)[3])
end
