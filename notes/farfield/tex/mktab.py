#!/usr/bin/env python3
"""Turn the fixed-width tables of notes/farfield/tables into LaTeX bodies under tex/tab/.

Every number in farfield.tex that comes from a table file comes through here, so the
document cannot drift from the tables verify.jl and bench.jl produce.  Run from
notes/farfield/:  python3 tex/mktab.py
"""
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(HERE, "..", "tables")
OUT = os.path.join(HERE, "tab")
os.makedirs(OUT, exist_ok=True)


def esc(t):
    """A table cell: numbers in math mode, everything else escaped text."""
    t = t.strip()
    if t in ("", "-"):
        return "--"
    if t.startswith("$") or t.startswith("\\") or t in ("(i)", "(ii)", "(iii)"):
        return t          # already formatted by the caller
    m = re.fullmatch(r"([+-]?[0-9.]+)e([+-][0-9]+)", t)
    if m:
        man, ex = m.group(1), int(m.group(2))
        return r"$%s{\times}10^{%d}$" % (man, ex)
    if re.fullmatch(r"[+-]?[0-9.]+", t):
        return "$%s$" % t
    t = t.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
    t = t.replace("(", "$(").replace(")", ")$") if t.startswith("(") else t
    return r"\texttt{%s}" % t if re.search(r"[a-zA-Z]", t) else t


def rows(name, hdr, keep=None, cols=None):
    """Read tables/<name>.txt, return the rows after the header line starting with hdr."""
    out, on = [], False
    for ln in open(os.path.join(TAB, name + ".txt")):
        if not on:
            if ln.startswith(hdr):
                on = True
            continue
        if not ln.strip() or ln.startswith("#"):
            break
        fl = ln.split()
        if keep and not keep(fl):
            continue
        out.append([fl[c] for c in cols] if cols else fl)
    return out


def emit(fn, colspec, head, body, long=False, cap=None):
    with open(os.path.join(OUT, fn), "w") as f:
        env = "longtable" if long else "tabular"
        f.write("\\begin{%s}{%s}\n\\toprule\n" % (env, colspec))
        f.write(" & ".join(head) + " \\\\\n\\midrule\n")
        if long:
            f.write("\\endfirsthead\n\\toprule\n" + " & ".join(head) +
                    " \\\\\n\\midrule\n\\endhead\n\\bottomrule\n\\endfoot\n")
        for r in body:
            f.write(" & ".join(esc(c) for c in r) + " \\\\\n")
        if not long:
            f.write("\\bottomrule\n")
        f.write("\\end{%s}\n" % env)
    print("wrote", fn, len(body), "rows")


# ---- (e) term counts from the bounds, f = 1 only ---------------------------
b = rows("e_terms", "shp ", keep=lambda c: c[1] == "1.0" and c[2] == "+" and c[3] == "0.0im",
         cols=[0, 4, 5, 6, 7, 8, 9, 10, 11, 12])
b = [r[:-1] + [{"1": "(i)", "2": "(ii)", "3": "(iii)"}[r[-1]]] for r in b]
emit("e_terms.tex", "llrrrrrrrc",
     ["shape", "dir", "$n$", r"$\rho$", r"$|k\mathbf{R}|$", "$L_{\\rm wb}$", "cost",
      "$L_{\\rm oct}$", "cost", "route"], b, long=True)

# ---- (e) routing counts ----------------------------------------------------
b = rows("e_route", "shp ", cols=None)
def frq(a, b_, c):
    return "$%s%s%s$" % (a, b_, c.replace("im", r"\ii"))
b = [[r[0], frq(r[1], r[2], r[3]),
      "$" + (r[4] + r[5] + r[6]).strip("()").replace(",", r"\!\times\!") + "$",
      r[7], r[8], r[9], r[10]] for r in b]
emit("e_route.tex", "llrrrrr",
     ["shape", "$f$", "block", "(i)", "(ii)", "(iii)", "setup s"], b)

# ---- (d) the summary block of the reference comparison ---------------------
b = rows("d_ref", "shp  f           n     eMx")
b = [[r[0], "$%s + %s\\ii$" % tuple(r[1].split(","))] + r[2:] for r in b]
emit("d_ref.tex", "llrrrrr",
     ["shape", "$f$", "rows", "$e_{\\max}$", "per entry", "Re", "Im"], b)

# ---- (e) digits lost, f = 1 ------------------------------------------------
b = rows("e_digits", "shp ", keep=lambda c: c[1] == "1.0" and c[3] == "0.0im",
         cols=[0, 4, 5, 6, 7, 8, 9, 10])
emit("e_digits.tex", "lrrrrrrr",
     ["shape", "$\\mathbf{n}$", r"$\rho$", r"$|k\mathbf{R}|$", "route", "$L$",
      "rel.\\ err", "digits lost"], b, long=True)

# ---- (i) cost per offset against L, c32 f = 1 ------------------------------
b = rows("i_cost", "shp ", keep=lambda c: c[0] == "c32" and c[3] == "0.0im",
         cols=[0, 4, 5, 6, 7])
emit("i_cost.tex", "lrrrr", ["shape", "$L$", "terms", "ns/offset", "ns/term"], b)
