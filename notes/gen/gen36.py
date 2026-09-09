# Emit LaTeX for the 36 explicit face-pair integrals with a general cell offset Delta.
FACES = [(0,-1),(0,1),(1,-1),(1,1),(2,-1),(2,1)]
NAME = ["yzL","yzU","xzL","xzU","xyL","xyU"]
def lim(d): return r"\int_{-s_%d/2}^{s_%d/2}" % (d+1,d+1)
def term(d, F, Fp):
    (c,s),(cp,sp) = F, Fp
    sgn = lambda v: "+" if v>0 else "-"
    tx = r"\Delta_%d s_%d" % (d+1,d+1)
    if d == c and d == cp:          # both fixed on this axis
        if s == sp: return "(%s)^2" % tx
        return r"(%s %s s_%d)^2" % (tx, sgn(s-sp), d+1)   # (s - sp)/2 * s_d = +-s_d
    if d == c:                       # target fixed, source varies
        return r"(%s %s \tfrac{s_%d}{2} - \eta_%d)^2" % (tx, sgn(s), d+1, d+1)
    if d == cp:                      # source fixed, target varies
        return r"(%s + \xi_%d %s \tfrac{s_%d}{2})^2" % (tx, d+1, sgn(-sp), d+1)
    return r"(%s + \xi_%d - \eta_%d)^2" % (tx, d+1, d+1)
def integral(i, j):
    F, Fp = FACES[i], FACES[j]
    tv = [d for d in range(3) if d != F[0]]; sv = [d for d in range(3) if d != Fp[0]]
    ints = "".join(lim(d) for d in tv) + "".join(lim(d) for d in sv)
    arg = " + ".join(term(d,F,Fp) for d in range(3))
    diffs = "".join(r"\,\dd\eta_%d" % (d+1) for d in reversed(sv)) + "".join(r"\,\dd\xi_%d" % (d+1) for d in reversed(tv))
    return r"I_{\mathrm{%s},\mathrm{%s}} &= %s g\Bigl(\sqrt{%s}\Bigr)%s" % (NAME[i], NAME[j], ints, arg, diffs)
# tensor components: (a,b) uses target faces with axis a? For diagonal aa: faces with axis c != a on both, same axis; off-diagonal ab: target axis a, source axis b.
comp = {}
for a in range(3):
    for b in range(3):
        L = []
        for i,F in enumerate(FACES):
            for j,Fp in enumerate(FACES):
                if a == b:
                    if F[0] == Fp[0] and F[0] != a: L.append((i,j, F[1]*Fp[1]))
                else:
                    if F[0] == a and Fp[0] == b: L.append((i,j, -F[1]*Fp[1]))
        comp[(a,b)] = L
AX = "xyz"
out = []
for (a,b),L in comp.items():
    out.append(r"\paragraph{Component $%s%s$}" % (AX[a],AX[b]))
    sgnsum = " ".join(("+" if s>0 else "-") + r" I_{\mathrm{%s},\mathrm{%s}}" % (NAME[i],NAME[j]) for (i,j,s) in L)
    pref = r"\frac{1}{k^2 V_t}" if a==b else r"\frac{1}{k^2 V_t}"
    out.append(r"\[ (\mathbf{G}_0)_{%s%s} = %s\bigl[\, %s \,\bigr] %s \]" % (AX[a],AX[b], pref, sgnsum.lstrip("+ ").replace("- I","- I"), r"- \frac{\delta_{ts}}{k^2}" if a==b else ""))
    out.append(r"\begin{align*}")
    out.append(r" \\".join(integral(i,j) for (i,j,s) in L))
    out.append(r"\end{align*}")
open("integrals36.tex","w").write("\n".join(out))
print(len(out), "lines;", sum(len(L) for L in comp.values()), "integrals")
