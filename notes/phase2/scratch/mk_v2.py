import re, sys
src = open("far_v1.jl").read()
pairs = [l.split() for l in open("renames.txt") if l.strip()]
for a, b in pairs:
    if a.endswith("!"):
        pat = r"(?<![A-Za-z0-9_])" + re.escape(a)
    else:
        pat = r"(?<![A-Za-z0-9_])" + re.escape(a) + r"(?![A-Za-z0-9_!])"
    src, n = re.subn(pat, b, src)
    if n == 0:
        print("WARN: no hit for", a)
open("far_v2_raw.jl", "w").write(src)
# report any name still off the three-letter rule
names = set()
for m in re.finditer(r"^(?:@inline\s+)?(?:function\s+)?([a-zA-Z_][A-Za-z0-9_]*!?)\s*\(", src, re.M):
    names.add(m.group(1))
for m in re.finditer(r"^const\s+([A-Za-z_][A-Za-z0-9_]*)", src, re.M):
    names.add(m.group(1))
def ok(n):
    s = n.rstrip("!")
    # strip trailing single-capital markers
    while len(s) > 3 and s[-1].isupper() and (len(s) < 2 or not s[-2].isupper()):
        s = s[:-1]
    while len(s) > 3 and s[-1].isupper():
        s = s[:-1]
    return len(s) % 3 == 0
bad = sorted(n for n in names if not n.isupper() and not ok(n))
print("names off the rule:", bad)
print("all names:", len(names))
