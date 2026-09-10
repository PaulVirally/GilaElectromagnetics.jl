# refill — the missing 220-bit slender short-axis volume references

Task: D = (0,0,3), (0,0,6), (0,0,32), (0,0,64), s = (1/32,1/32,1/512), f = 1 and f = 1+0.1i,
via refquad.jl's graded Gauss-Legendre volume rule (notes/farfield/refquad.jl), cached as
kind `tns:refQuad` in notes/farfield/refcache/reftensors.txt (the record format verify.jl's
`loadRef`/`refOf` read). Work dir: SCRATCH/work/refill/ (run.jl driver, farcheck.jl crosscheck,
conv_r.txt/conv_c.txt convergence logs). Backup of reftensors.txt before any writes:
SCRATCH/work/refill/reftensors.txt.bak (2093199 bytes, taken before this session appended anything).

(report filled in incrementally below as each stage completes)

## 1. Cache confirmation

`diff SCRATCH/work/refill/reftensors.txt.bak notes/farfield/refcache/reftensors.txt` gives exactly
8 added lines, all appended at the end (`.bak` has 1060 lines, the live file has 1068). Every added
line has the form

    tns:refQuad|D=0,0,n|s=1//32,1//32,1//512|f=<re>;<im>|p=220|n=<ord>|<9 complex entries, re;im>

for (n, ord) in {(3,28), (6,24), (32,20), (64,16)} at both f=1.0;0.0 and f=1.0;0.1 — 4 offsets x
2 frequencies = 8 records, matching run.jl's `rqPut` output and mkrefcache.jl/refquad.jl's
documented format exactly (kind `tns:refQuad`, p=220 bits, s the slender cell). No stray writes
elsewhere: `rqPut`'s `RQDIR` is `@__DIR__` of refquad.jl = notes/farfield/, so it only ever touches
notes/farfield/refcache/reftensors.txt (confirmed by reading refquad.jl; the scratch/refcache/
directory is a separate, untouched file — the reference agent's own raw cache, copied verbatim by
mkrefcache.jl, not the live deliverable).

## 2. Per-tensor order and convergence (from conv_r.txt / conv_c.txt)

f = 1 (conv_r.txt):

| D        | ord used | relchange (ord vs ord-4) | max entry            |
|----------|----------|---------------------------|-----------------------|
| (0,0,3)  | 28       | 5.016e-37 (28 vs 24)       | 0.04384183898995512   |
| (0,0,6)  | 24       | 8.941e-32 (24 vs 20)       | 0.023155382459166284  |
| (0,0,32) | 20       | 9.323e-38 (20 vs 16)       | 0.0010830373897131082 |
| (0,0,64) | 16       | 2.397e-34 (16 vs 12)       | 0.0001875255136891093 |

f = 1+0.1i (conv_c.txt):

| D        | ord used | relchange (ord vs ord-4) | max entry             |
|----------|----------|---------------------------|------------------------|
| (0,0,3)  | 28       | 5.017e-37 (28 vs 24)       | 0.04339789292745008    |
| (0,0,6)  | 24       | 8.945e-32 (24 vs 20)       | 0.022916603098889148   |
| (0,0,32) | 20       | 9.388e-38 (20 vs 16)       | 0.0010649163810164777  |
| (0,0,64) | 16       | 2.475e-34 (16 vs 12)       | 0.0001798589383862741  |

All four stopped well inside the run.jl break criterion (relchange < 1e-30), so none hit the
1500 s per-tensor BUDGET; per-order costs at f=1 ranged from 34-39 s (ord 12) to 438 s (ord 28,
the (0,0,3) case, the most expensive because it is the closest, most singular-adjacent offset).

## 3. Cross-check against farTensor (notes/farfield/farfield.jl), f = 1

Script: SCRATCH/work/refill/farcheck2.jl (a parametrised copy of the previous instance's
farcheck.jl, extended to take n and the freq tag as args and to print farRoute's (kind,L,Lc)).
TABDIR resolved to scratch/work/unify/cache (found `*_p<TABPRC>.txt` shape tables there).

- D=(0,0,32): route = 2 (octants, route ii), L absent, Lc = [33,33,33,34,33,34,34,34].
  eMx (farTensor vs refQuad) = 9.174235584771969e-16, worst per-entry pEn = 2.172543459428697e-15.
  max|Gr| = 0.0010830373897131082. Both well under the 1e-14 bar.
- D=(0,0,6): route = 3 (k-series, route iii), all Lc = -1 (every octant needs the k-series band).
  eMx = 5.842195560051354e-17, pEn = 5.842195560051354e-17 (only the diagonal entries clear the
  1e-8*max threshold; off-diagonals are ~0 for an axis offset). max|Gr| = 0.023155382459166284.

Both agree with the freshly cached refQuad reference to 1e-14 or (much) better; the task's guess of
route (i) or (ii) for (0,0,32) resolves to (ii), and (0,0,6) is confirmed route (iii) as stated.

## 4. verify.jl parts (c) and (d)

Run from notes/farfield/, `JULIA_NUM_THREADS=1 julia --startup-file=no --project=$ENV verify.jl c`
and `... verify.jl d`. Only one other Julia process was ever running concurrently during these
runs (an unrelated audit2 sub-agent job, `ps aux` confirmed at each check); load average during
the two runs was 2.6-3.2 (1-, 5-, 15-min figures all in that band, `uptime` sampled immediately
before/after each run). Timings: part c real 7.57 s (its own internal timer: "part c: 2.8 s"),
part d real 17.43 s (internal timer: "part d: 15.3 s").

**Part (c) — no change for the slender ax3 rows.** They still print `no cached reference` for
sep = 3, 6, 32, 64 (the exact four offsets just cached):

    ax3   2    9    1.57e-02   30   425.0    11  7.89e-01  1.60e+00  11  5.80e-02
    ax3   3    no cached reference
    ax3   4    7    6.40e-04   30   590.0    11  3.68e-02  7.51e-02  11  3.46e-02
    ax3   6    no cached reference
    ax3   8    5    2.83e-05   30   920.0    11  1.79e-03  3.70e-03  11  1.63e-02
    ax3   16   5    7.17e-08   30   1850.0   11  2.97e-06  6.35e-06  11  5.19e-03
    ax3   32   no cached reference
    ax3   64   no cached reference

This is not a bug in the caching, it is a kind mismatch: `partC` (verify.jl:365,382-383) looks up
the `"pairs"` kind only (36 raw face-pair values, needed to compute its `pairErr`/`amp` columns per
face pair), and refquad.jl's `refQuad` only ever caches the assembled 9-entry `tns:refQuad` tensor
— there is no 36-value face-pair decomposition to give it. So part (c)'s ax3 rows for n=3,6,32,64
are unchanged by this session's work at both frequencies (f=1 and f=1+0.1i checked, same 4 blanks
each). Filling these in would need `refPairs(D,s,f)` from ref.jl (its own docstring: "13.7 s per
cubic offset at 220 bits" — the slender cell would cost more; nobody has run it for these offsets).

**Part (d) — the 8 new rows now carry numbers.** `refOf` (verify.jl:101-112) falls back to
`tns:refQuad` when no `pairs`/`vol`/`famDQ` record exists, so all 8 of the newly cached (D,s,f)
triples now appear (previously they were silently skipped: `refOf` returned `nothing` and the row
was never printed, verify.jl:449 `Gr === nothing && continue`):

    shp  D             f           rho     kR       rt  L    eMx       pEn       eRe       eIm       grd  src
    sl   (0,0,3)       1.0,0.0     7.5498  0.037    3   -1   2.23e-17  4.55e-17  4.55e-17  3.42e-16  yes  refQuad
    sl   (0,0,6)       1.0,0.0     3.7749  0.074    3   -1   5.84e-17  5.84e-17  5.84e-17  3.95e-17  yes  refQuad
    sl   (0,0,32)      1.0,0.0     0.7078  0.393    2   34   9.17e-16  2.17e-15  2.18e-15  2.96e-16  yes  refQuad
    sl   (0,0,64)      1.0,0.0     0.3539  0.785    1   30   1.22e-16  1.52e-16  1.33e-16  2.71e-16  yes  refQuad
    sl   (0,0,3)       1.0,0.1     7.5498  0.037    3   -1   3.34e-17  6.80e-17  6.71e-17  8.60e-17  yes  refQuad
    sl   (0,0,6)       1.0,0.1     3.7749  0.074    3   -1   4.21e-17  4.21e-17  4.28e-17  5.67e-17  yes  refQuad
    sl   (0,0,32)      1.0,0.1     0.7078  0.395    2   34   8.12e-16  1.88e-15  1.95e-15  4.35e-17  yes  refQuad
    sl   (0,0,64)      1.0,0.1     0.3539  0.789    1   30   1.40e-16  3.67e-16  3.83e-16  1.23e-15  yes  refQuad

All 8 land at route rt: 3 (k-series) for n=3,6; rt=2 (octants) for n=32; rt=1 (whole box) for
n=64 — farTensor's routing at the slender aspect ratio pushes to k-series only for the two closest
offsets. eMx and pEn are all <= 9.17e-16, i.e. the new method agrees with the fresh 220-bit
reference at every one of these offsets to far better than the 1e-14 publishability bar, at both
frequencies. Full logs: SCRATCH/work/refill/verify_c.log, SCRATCH/work/refill/verify_d.log.
Cross-check script: SCRATCH/work/refill/farcheck2.jl.
