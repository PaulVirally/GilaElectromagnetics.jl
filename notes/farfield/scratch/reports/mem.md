# mem -- D8 memory fix (root, 2026-09-07 17:50)

Cause (merge S4.2): hrmPly! and mulRsq! allocated a fresh BigInt per coefficient operation,
2.4-5.8 MB per (l,m) column, ~15 GB of GMP garbage over an L = 50 octant table; GC forcing and
heap hints did not return it.

Fix: `bigMat(nA)` builds matrices of distinct BigInt objects (`zeros(BigInt, n, n)` aliases a
single zero); `bigZero!` clears in place; `mulRsq!` uses MPZ.set!/add!; `hrmPly!` uses MPZ.set!
for the azimuthal seed and MPZ.mul!/add! through one scratch BigInt passed from tabWhl/tabOct.
momAcc was already in place (merge). Comment at the momAcc block updated.

Checks: work/root/mem/bitid.jl loads the pre-fix file as module Old and the new one as New and
compares every GeoBox field of 24 tables: 0 mismatches. m5_tab.jl cold builds (files in
work/root/mem/m5_*_after.txt): peak RSS 0.66-0.74 GB, build+setup 14.3-18.3 s, cache byte counts
identical to merge's. m0_smoke.jl unchanged.

Not done: the twelve audit2 shapes were not rebuilt (same code path; stated as inferred in
uncertain.tex). legCof/azmCof still allocate O(l^2) small BigInts per column (<1 MB/column).
