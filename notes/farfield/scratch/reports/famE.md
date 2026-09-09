# famE (families e, f) -- root's digest of the report lost in the 13:10 reboot
Scripts (specchk.jl etc.) were lost; the numbers below were read from the agent's summary.

## (e) spectral / Ewald: NOT VIABLE at 1e-13
Spectrum, derived and validated: T^_ab(xi) = V_t prod_i sinc^2(xi_i s_i/2) (delta_ab k^2 - xi_a xi_b)
/ (f^2 (|xi|^2 - k^2)). 1D check 1.9e-15 at M = 128; 3D by shifted Poisson summation on a lossy
lattice: 3.44e-13 max relative difference to the real-space volume form; the real-space volume
evaluator matched Gila's egoSrfFxd!+srfSum! with ratio exactly 1 (spread 2-4e-13 = Gila's error)
at 4 offsets x 2 frequencies (so the volume form has no extra factor or sign).
(1) Aliasing: tensor multiplier -> O(1), decay only from prod sinc^2, tail C/M with C ~ 0.04:
M ~ 4e11 per axis for 1e-13. Ewald damping e^{-(xi^2-k^2)/(4 eta^2)}: alias error at Float64 zero
for M = 2 at eta s <= 2.5, M = 8 at eta s = 10 (rule M >= 1.74 eta s). Aliasing is not the obstacle.
(2) The pole at |xi| = k, fatal and structural: the inverse DFT of a periodized spectrum is
sum_j T(R_{n+jP}), the P-periodization of a kernel that does not decay; trapezoid error
e^{-Im k P s}. Measured 1D pipeline (block n = 0..128, lambda/32): real f gives NaN/O(1) at every
P; f = 1+0.1i needs P = 1024 (32 lambda) for 2.4e-8 and P ~ 1651 for 1e-13; padding rule
(P - N) s > 4.76/Im f wavelengths. At N = 128: Im f = 0.1 -> P = 2048 (768 GB), Im f = 0.01 ->
P = 16384, Im f = 0 -> infinite. Reciprocal spacing pi/(N s) = 3.14/1.57/0.785 at N = 32/64/128,
2/4/8 grid points between 0 and k = 6.28; pole 0.2-0.8 spacings off the real axis at Im k = 0.63.
Standard fixes (complex frequency, artificial loss, contour deformation, principal value + residue)
priced: none delivers the finite block at 1e-13; a C^infty-windowed truncated kernel has no closed
form transform. (3) Near correction: eta s = 1.25 -> 6.65 cells (1231 offsets) at 125 alias terms.
Cost even if fixed: ~1e11 flops at N = 128 vs 5e9 for a 300-flop direct expansion. Right tool for a
periodic (Bloch) operator, not for Gila's finite free-space block.

## (f) interpolation: 9.6x at best, only above ~2200 flops/offset
Analyticity radius rho_an(R) = min_{delta in D} |R + delta| = the cell gap, in every coordinate;
on-axis the nearest singularity is real, Bernstein parameter rho_B = x0 + sqrt(x0^2 - 1),
x0 = (c - 1)/a in cells. Predicted vs measured Chebyshev degree for 1e-13 (lambda/32, radial):
(16,16) 23/18, (32,32) 25/18, (64,32) 21/20, (64,64) 30/26 -> bound honest, pessimistic by 2-7.
cell      n0  H   q   frac=((q+1)/H)^3  8 cInt  break-even X*
lambda/32 32  32  18  0.209             1775    2.24e3 flops/offset
lambda/32 64  64  26  0.075             2073    2.24e3
lambda/8  64  64  46  0.396             5129    8.49e3
lambda/4  32  32  46  3.17              10436   never
slender-z 64  64  20  0.035             1495    1.55e3
128^3 block at lambda/32: direct fraction 0.104 -> 9.6x. Equispaced sub-lattice nodes never reach
1e-13 (best 8.1e-12 at H = 32, then Runge divergence to 6.8e-10; 5e3 at lambda/8 H = 64):
Chebyshev nodes are mandatory, so the direct method must be callable off-lattice. Nothing for the
2-8 cell band. If the direct method reaches a few hundred flops/offset, (f) is a net loss (~7x).
