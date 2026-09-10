# geometry (items 5, 7, 8) -- root's digest of the report lost in the 13:10 reboot
(The agent's scripts rho_tables.jl, moments_check.jl, error_target.jl and CSVs were lost; every
number below was read by the root from the report before the crash.)

## Item 5: rho tables (cubic cells, scale-independent)
class      n    rho       rho2      rho4      rho8
axis       2   0.866025  0.522233  0.333333  0.190117
axis       3   0.577350  0.333333  0.190117  0.101535
axis       4   0.433013  0.242536  0.132453  0.069171
axis       8   0.216506  0.114960  0.059655  0.030378
axis      16   0.108253  0.055815  0.028387  0.014313
axis      32   0.054127  0.027486  0.013856  0.006956
axis      64   0.027063  0.013637  0.006846  0.003430
facediag   2   0.612372  0.397360  0.242536  0.135665
facediag   8   0.153093  0.081559  0.042220  0.021485
bodydiag   2   0.500000  0.333333  0.200000  0.111111
bodydiag   8   0.125000  0.066667  0.034483  0.017544
(n,1,0)    2   0.774597  0.522233  0.333333  0.190117
(n,2,1)    2   0.577350  0.397360  0.242536  0.135665
rho_m = sqrt(sum (s_i/m)^2) / min_j |R + c_j| over all m^3 sub-box centres.
Slender (1/32,1/32,1/512): rho<1 first at n=2 (rho<0.5 at n=3) for every offset class except the
short axis (0,0,n): rho<1 at n=23, rho<0.5 at n=46. Directional: for (0,0,n) only
rho_1D(axis 3) = s3/|R| = 1/n shrinks (0.031 at n=32); rho_2D excluding either long axis stays
~2 at n=8, ~0.5 at n=32.
Offset counts in the egoToe octant (n_i in 0..N-1) with rho > 0.5/0.3/0.2/0.1: 37/141/436/3096,
identical for N = 32, 64, 128 (the band is a ball of fixed radius: n1^2+n2^2+n3^2 < 3/b^2).

## Item 7: exact moments
mu_n(s) = 2 s^{n+2}/((n+1)(n+2)) (even n), 0 (odd n); proof: 2 int_0^s (s-t) t^n dt.
Volume even moments int_D w |R+delta|^{2j} by two exact routes agree, e.g. D=(2,0,0), s=1/32:
j=1: 9/2199023255552, j=2: 691/33776997205278720, j=3: 82279/726340547902313594880;
D=(3,1,0): 21/2199023255552, 3511/33776997205278720, 866881/726340547902313594880;
D=(2,2,2): 25/2199023255552, 4931/33776997205278720, 285155/145268109580462718976.
Divergence identity in exact rationals for D in {(2,0,0),(3,1,0),(2,2,2)}, j = 1,2,3, all exact:
int_Vt int_Vs d_a d_b F(x-y) = - sum_{c(F)=a, c(F')=b} sigma_F sigma_F' int_F int_F' F,
diagonal (d_a d_a - lap)F = - sum_{b!=a} d_b d_b F. Hand check D=(2,0,0), j=1: (d_a d_a - lap)|x-y|^2
= -4, times mu_0^3 = s^6: -4/32^6 = -1/268435456 = assembled value.
moments.jl even moments agree with the exact polynomial integrator to 154 digits (512 bits).
pairMoments(..., Rational{BigInt}) fails at moments.jl:442 (sqrt of Rational); BigFloat panels work.

## Item 8: error target
dfltPrc = Float32 (src/glaTyp.jl:19); GMRES/BiCGStab/MixPrcRfn default relTol = sqrt(eps(T)):
1.49e-8 (Float64), 3.45e-4 (Float32). FFT roundoff N=256 doubled grid: 5.3e-15 (Float64),
2.9e-6 (Float32). Anti-Hermitian floor (Sec 2.4 of integrals.tex, 6^3 block, real f):
lambda_min = -5e-15 vs lambda_max 0.077 -> 6.5e-14 relative; a 1e-13 kernel error is 1.5x above it.
Far tensor magnitude at lambda/32, f=1, axis offsets: n=2: 2.08e-2, n=8: 5.77e-4, n=32: 9.44e-5,
n=128: 2.39e-5 (max|G| R flat ~9.5e-5 from 1 lambda on); self term O(1) (identity -1/f^2).
Conclusion: "1e-13 relative to the largest entry" is 4 orders looser than 1e-13 for far entries;
per-entry relative accuracy in Float64 is the honest target; Float32 storage and the default
solver tolerance discard 6-8 digits regardless. egoSrfFxd! at n=1 returns Inf (touching cells go
through the contact path, as expected).
