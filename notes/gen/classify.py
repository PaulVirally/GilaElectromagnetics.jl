# Classify the 36 face pairs of a source cell at the origin and a target cell at offset D (in cells).
import itertools
from fractions import Fraction as Fr
AX = "xyz"
FACES = [(0,-1),(0,1),(1,-1),(1,1),(2,-1),(2,1)]   # yzL yzU xzL xzU xyL xyU
NAME = ["yzL","yzU","xzL","xzU","xyL","xyU"]
def box(c, sig, off):  # returns list of 3 intervals (lo,hi) for face (c,sig) of a unit cell at offset off
    iv = []
    for d in range(3):
        if d == c: v = Fr(off[d]) + Fr(sig,2); iv.append((v,v))
        else: iv.append((Fr(off[d])-Fr(1,2), Fr(off[d])+Fr(1,2)))
    return iv
def inter(a,b): 
    lo, hi = max(a[0],b[0]), min(a[1],b[1])
    return None if lo>hi else (lo,hi)
def classify(F, Fp, D):
    (c,s),(cp,sp) = F, Fp
    A, B = box(c,s,D), box(cp,sp,(0,0,0))
    I = [inter(A[d],B[d]) for d in range(3)]
    if any(i is None for i in I):
        touching = False
    else:
        touching = True
        dims = [i[1]-i[0] for i in I]
        ndeg = sum(1 for x in dims if x==0)
    if c == cp:
        if A[c][0] == B[c][0]:   # coplanar
            if not touching: return "coplanar gap (does not occur)"
            if ndeg == 1: return "S"   # coincident (only the normal coordinate is degenerate)
            if ndeg == 2: return "Ef"  # flat edge
            return "Vc"                # coplanar vertex (ndeg == 3)
        dist = abs(A[c][0]-B[c][0]); shift = sum(1 for d in range(3) if d!=c and A[d][0]!=B[d][0])
        return f"P{int(dist)}" + ("" if shift==0 else "s")   # parallel, offset 1 or 2 cells, s = laterally shifted
    else:
        if touching:
            if ndeg == 2: return "Ec"  # perpendicular sharing an edge (segment intersection)
            return "Vp"                # perpendicular sharing a vertex
        return "Xg"                    # perpendicular, not meeting
for D in [(0,0,0),(1,0,0),(1,1,0),(1,1,1)]:
    print("offset", D)
    print("      " + " ".join(f"{n:>4}" for n in NAME))
    for i,F in enumerate(FACES):
        print(f"{NAME[i]:>5} " + " ".join(f"{classify(F,Fp,D):>4}" for Fp in FACES))
    from collections import Counter
    print(Counter(classify(F,Fp,D) for F in FACES for Fp in FACES))
