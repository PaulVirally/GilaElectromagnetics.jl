# bound2: a sharper proven l-truncation bound for the local expansion

Work dir `SCRATCH/work/bound2/`, SCRATCH = `/Users/pvirally/.julia/dev/GilaElectromagnetics/notes/farfield/scratch`.
Deliverables: `work/bound2/bound2.jl` (standalone, generic in `T`, same call signature as
`remBnd` of `work/theory/chkRem_fns.jl`) and `work/bound2/bounds2.tex` (LaTeX fragment in the
style of `bounds.tex`; compiles clean inside a copy of `work/theory/test.tex`, see S8).

**Headline.** Two exact identities replace the two loose steps of Theorem (c).  Over 116
BigFloat rows (4 shapes, offsets 2..64 cells, f = 1, 1+0.1i, 1+1i, L = 8,16,24,32, no
violation) the bound-to-truth ratio falls from **1.35e3 .. 1.73e11 (median 1.53e5)** to
**8.79 .. 2.93e3 (median 49.1)**, a median gain of **3233x**, and the selected L falls by
1.00-1.50x (median 1.21), i.e. **1.00-2.25x (median 1.47x) in the O(L^2) per-offset cost**.
On an octant sub-box (45 rows) the ratio falls from 130 .. 1.13e5 (median 2.58e3) to
5.51 .. 163 (median 26.2), a median gain of 96x, and L falls by 1.11-1.27x.

---

## 1. A structural finding first: the old bound bounds a different truncation

With `psi_lm(R) = h_l(k|R|) Y_lm(Rhat)` and `phi_lm(d) = j_l(k|d|) Y_lm(dhat)`, shell l is

    Sig_l^sng = sum_m [(da db + dab k^2) psi_lm](R) * int_D w phi_lm            (bounds.tex Thm (c))
    Sig_l^reg = sum_m psi_lm(R) * int_D w (da db + dab k^2) phi_lm              (what farfield.jl sums)

These are NOT equal term by term: the first differentiates `S_l(R,d)` in R, the second in d,
and `S_l` is not a function of `R+d`.  Both series converge to `T_ab`.  The library's tables
`Ad/Ao/Bd` (`expansion.tex` eq. (exp:tabs), `unify.md` S1.2) carry the derivatives on the
delta side, so **the library truncates the second while Theorem (c) bounds the first**.

Measured (`chkA.jl`, 116 rows): `max_ab|T - T^(L),reg| / max_ab|T - T^(L),sng)|` has median
**17.5** (range 0.043 .. 7.4e3), and the L needed for 1e-13*est is a median one shell larger
for the regular organization (`Lmeas_sng/Lmeas_reg` median 0.909, range 0.75 .. 1.0).  The
mechanism is visible in the first shell (`chkD.jl`, `out_chkD.txt`), `sum_{l<=L} shell / T[1,1]`:

    case             L=0 regular   L=0 singular
    c32 (4,0,0)        0.162          1.0014
    c32 (16,0,0)       0.998          1.0000
    c4  (4,0,0)        2.061          0.9964

The singular organization front-loads: `(da db + dab k^2) h_0(k|R|)` already carries the whole
`3 RaRb/R^5` dyadic, so shell 0 is already the answer to 1.4e-3 at rho = 0.433, whereas the
regular shell 0 is
isotropic in Rhat and the anisotropy has to be built from l = 2 upwards.

Nothing computed so far is wrong -- the old bound is loose by 1e3..1e11, which covers a factor
17.5 many times over -- but a bound used to *select* L should bound the sum that is truncated.
Theorem A below does; Theorem C is the sharpened version of Theorem (c) for the other
organization, for use wherever the singular form is the one summed.

---

## 2. The two replacements (both exact identities, both verified)

### Lemma 1 (m-sums).  For any solution f of the spherical Bessel equation of order l,

    sum_m Y_lm(u)^2 = (2l+1)/(4 pi)                                          [addition theorem]
    sum_m |grad(f_l(kr) Y_lm(rhat))|^2 = (|k|^2/(4 pi)) G_f(l),
        G_f(l) = l |f_{l-1}(kr)|^2 + (l+1) |f_{l+1}(kr)|^2 .

Proof: the gradient formula `grad(f_l Y_lm) = k[sqrt(l/(2l+1)) f_{l-1} Y_{l,l-1,m}
- sqrt((l+1)/(2l+1)) f_{l+1} Y_{l,l+1,m}]` plus
`sum_m Y_{J l' m} . conj(Y_{J l'' m}) = delta_{l'l''} (2J+1)/(4 pi)`, which follows because the
tensor `sum_m Y_{Jl'm} (x) conj(Y_{Jl''m})` is rotation-covariant, hence its trace is constant
on the sphere and equals its own spherical average.  Measured (`chk0.jl`, 400 bit, central
differences at 1e-30): **ratio true/formula = 1.000000 in all 48 rows** (f = j and h, four
points, l in {0,1,2,5,12,20}).

### Lemma 2 (Hessian in l^2 over m) -- the replacement of the constant 17.

    sqrt( sum_m sum_{a,b} |da db (f_l(kr) Y_lm)|^2 )
        <= Theta_f(l,r) := (|k|^2/sqrt(4 pi)) [ sqrt(l G_f(l-1)/(2l-1))
                                              + sqrt((l+1) G_f(l+1)/(2l+3)) ] ,

involving only `f_{l-2}, f_l, f_{l+2}`, with equality at l = 0.

Proof: apply the gradient formula once, split the two channels by Minkowski in l^2(m,a,b), and
observe that `m -> ((Y_{l,l-1,m})_b)_b` is an equivariant map from D^l into
`D^1 (x) D^{l-1} = D^{l-2}+D^{l-1}+D^l`, in which D^l occurs once.  Schur's lemma makes its
matrix A a coisometry (`A A^dag = I` after the trace normalisation `sum_m int |Y_{l,l-1,m}|^2
= 2l+1`), `A^dag A` the projector onto that D^l component, and the partial trace of that
projector over the vector index is `((2l+1)/(2l-1)) I`.  Hence
`sum_{m,b} |grad(f_{l-1} (Y_{l,l-1,m})_b)|^2 = ((2l+1)/(2l-1)) (|k|^2/4pi) G_f(l-1)`, and
`a_l^2 (2l+1)/(2l-1) = l/(2l-1)`, `b_l^2 (2l+1)/(2l+3) = (l+1)/(2l+3)`.

Measured (`chk0.jl`, 400 bit, no violation in 48 rows), true/bound:

    |k|r     f    l=0     l=1     l=2     l=5     l=12    l=20
    0.785    h    1.000   0.958   0.984   0.996   0.9991  0.9997
    0.621    h    1.000   0.974   0.990   0.997   0.9994  0.9998
    7.234    h    1.000   0.722   0.717   0.731   0.920   0.970
    9.425    h    1.000   0.719   0.712   0.710   0.849   0.949
    0.785    j    1.000   0.708   0.945   0.993   0.9989  0.9996
    9.425    j    1.000   0.712   0.808   0.736   0.835   0.942

**Lemma 2 is loose by at most 1.41 and usually by less than 1.01, against the measured
32x-34000x of the constant 17** (`chk17.jl`).  The l-dependence is also removed: the old
bound carried an extra `sqrt(2l+5)` from `max_m |Y_lm|` times `sum_m |Y_lm|`; the joint
Cauchy-Schwarz `|sum_m x_m y_m| <= ||x||_2 ||y||_2` carries none, so the improvement grows
like `1/(34 sqrt(2l))` in the term ratio.

### Lemma 3 (integration by parts onto the triangle weight).  Extend
`w = prod (s_i - |d_i|)_+` by zero.  Then `w_i'' = delta_{-s_i} - 2 delta_0 + delta_{s_i}`
distributionally and `w = 0` on `dD`, so for every C^2 function phi and a != b (c the third axis)

    int_D w da db phi  =  int_D sgn(d_a) sgn(d_b) (s_c - |d_c|) phi                      (i)
    int_D w da da phi  =  int_{Q_a} (s_b-|d_b|)(s_c-|d_c|) [ phi|_{d_a=s_a} + phi|_{d_a=-s_a}
                                                             - 2 phi|_{d_a=0} ]           (ii)

Measured (`chk0.jl`): `max|direct - ibp| / max|direct|` = **2.5e-50 .. 2.0e-44** (quadrature
limited) for (1/32)^3 and (1/32,1/32,1/512), all six (a,b), l <= 8.

Why (i)-(ii) matter: on the singular side two derivatives cost `hb(l+2)/hb(l) ~ (2l+1)(2l+3)/
|kR|^2`; on the regular side, after (i)-(ii), they cost `V^{ab}_0 / W_0 = 4/(s_a s_b)`.  The
regular side wins whenever `l > |R|/s = sqrt3/rho`, which is the entire range in which the
expansion is used (`L ~ 13/log10(1/(0.7 rho))`).

---

## 3. Theorem A (whole box, the organization the library truncates)

    |T_ab - T^(L)_ab| <= (|k|/(4 pi |f|^2 V_t)) sum_{l > L, l even} (2l+1) |h_l(k|R|)|
                          * |k|^l e^{(|k| r_d)^2/(4l+6)} / (2l+1)!! * V^{ab}_l ,

with the exact positive geometry moments (l = 2p, lam_n(s) = 2 s^{n+1}/(n+1), mu_n as usual)

    V^{ab}_l = int_D (s_c-|d_c|) |d|^l dd
             = sum_{i+j+q=p} p!/(i!j!q!) lam_2i(s_a) lam_2j(s_b) mu_2q(s_c)          (a != b)
    V^{aa}_l = 2 sum_{i+j+q=p} p!/(i!j!q!) s_a^{2i} mu_2j(s_b) mu_2q(s_c)
             + 2 sum_{j+q=p} p!/(j!q!) mu_2j(s_b) mu_2q(s_c)  +  |k|^2 W_l .

Proof: `da db g(R+d) = d_{d_a} d_{d_b} g(R+d)`; Cauchy-Schwarz over m against
`sum_m |psi_lm(R)|^2 = ((2l+1)/4pi)|h_l|^2` (exact); Lemma 3 to move the derivatives onto w;
Minkowski's integral inequality with `||phi_lm(d)||_{l2(m)} = sqrt((2l+1)/4pi) |j_l(k|d|)|`
(exact, Lemma 1); then `|j_l(kx)| <= (|k|x)^l e^{(|k| r_d)^2/(4l+6)}/(2l+1)!!` pointwise, which
turns each positive measure into a multinomial moment.  Odd l is omitted because the triangle
weight kills it.

### 3.1 Ratios (script `chkA.jl`; raw `out_chkA_main.txt`, `out_chkA_far.txt`)

Truth = `max_ab |T - T^(L)|` with T from an independent BigFloat volume Gauss-Legendre rule on
the 8 octants (order 22) and the shell data from a 24-point positive-octant rule with exact
parity factors; the two agree at L = 32 to 1e-18 .. 1e-57 relative.  116 rows, **no violation**
(the four rows flagged in the 320-bit pass were the central-difference floor of the *singular*
truth, and are clean in the 512-bit rerun with step 1e-38, `out_chkA_far.txt`).

    bound / truth                     min       median     max
    Theorem A   (regular)             8.79      49.1       2.93e3
    Theorem C   (singular, S5)        1.49      140        792
    Theorem (c) (in use today)        1.35e3    1.53e5     1.73e11
    gain (c)/A                        31.3      3.23e3     4.61e9
    gain (c)/C                        162       661        7.21e8

    per L (29 cases each):  L= 8  A median 20.6 [8.79, 144]     (c) median 3.76e4
                            L=16  A median 88.6 [27.5, 445]     (c) median 1.48e5
                            L=24  A median 28.7 [14.5, 83.3]    (c) median 2.40e5
                            L=32  A median 96.6 [9.30, 2.93e3]  (c) median 2.01e5

Representative rows (f = 1 unless stated):

    shape  n           rho     kR      L   truth(reg)  Thm A       A/tru   Thm(c)      (c)/tru(sng)
    c32    (4,0,0)     0.433   0.785    8  2.816e-8    2.923e-7    10.4    5.13e-5     3.17e4
    c32    (4,0,0)     0.433   0.785   16  2.854e-13   4.618e-11   162     1.39e-8     5.50e4
    c32    (4,0,0)     0.433   0.785   24  4.755e-16   1.289e-14   27.1    5.36e-12    2.21e5
    c32    (2,0,0)     0.866   0.393    8  4.532e-5    9.347e-4    20.6    5.09e-1     3.41e4
    c32    (3,1,0)     0.548   0.621   16  1.489e-10   4.375e-9    29.4    1.82e-6     1.57e5
    c32    (8,8,8)     0.125   2.721   16  7.950e-23   2.521e-21   31.7    3.60e-19    8.51e5
    c32    (64,0,0)    0.027  12.566   24  7.595e-48   1.690e-46   22.3    1.79e-42    1.74e9
    c8     (8,8,8)     0.125  10.883   24  3.528e-30   1.011e-28   28.7    8.99e-24    2.11e8
    c4     (4,0,0)     0.433   6.283   16  3.274e-13   5.424e-11   166     1.33e-7     4.69e5
    sl     (4,0,0)     0.354   0.785   16  4.403e-14   4.577e-12   104     9.63e-11    5.93e4
    sl     (0,0,32)    0.708   0.393   16  6.672e-8    3.321e-6    49.8    2.17e-4     8.78e3

The rows that overlap `theory.md` S3(c) reproduce it exactly: at c32 (4,0,0), L = 8/16/24 the
singular truth is 1.620e-9 / 2.527e-13 / 2.425e-17 and eq (bnd:rem) is 5.13e-5 / 1.39e-8 /
5.36e-12, ratios 3.17e4 / 5.50e4 / 2.21e5 -- the same digits as the theory table -- and at
sl (4,0,0), L = 16, 1.624e-15 against 9.63e-11, ratio 5.93e4.  My reimplementation of the old
bound is therefore the same object.

The largest A/truth values are dips of the *truth*, not looseness: the true remainder is
non-monotone in L (the l-sum alternates with the phase of `h_l`), and no monotone majorant
follows a dip.  At the L the bound actually selects the ratio is 9-100.

### 3.2 The L it selects (tol = 1e-13 * est(R))

`Lmeas` = smallest even L with `max_ab|T - T^(L),reg| <= 1e-13 est`; `-1` = not reached by L = 32.

    shape f        n           rho     kR      Lmeas Lmeas Lold  Lnew  Lnew  Lnew(hb)
                                               REG   SNG   (c)   ThmA  ThmC
    c32  1        (2,0,0)      0.866   0.393    -1    -1    -1    126   134   128
    c32  1        (3,0,0)      0.577   0.589    32    30    50     40    40    40
    c32  1        (4,0,0)      0.433   0.785    24    22    34     28    28    28
    c32  1        (3,1,0)      0.548   0.621    32    30    46     36    38    38
    c32  1        (8,0,0)      0.217   1.571    16    14    20     16    16    18
    c32  1        (8,8,8)      0.125   2.721    12    10    14     12    12    14
    c32  1        (16,0,0)     0.108   3.142    10     8    14     12    10    14
    c32  1        (32,0,0)     0.054   6.283     8     6    12     10     8    10
    c32  1        (64,0,0)     0.027  12.566     8     6    10      8     8    10
    c32  1        (2,2,0)      0.612   0.555    -1    -1    56     44    44    44
    c32  1+0.1i   (4,0,0)      0.433   0.789    24    22    34     28    28    28
    c32  1+1i     (4,0,0)      0.433   1.111    26    24    34     28    28    28
    c32  1+1i     (16,0,0)     0.108   4.443    12    10    14     12    12    14
    c32  1+0.1i   (3,1,0)      0.548   0.624    32    30    46     36    38    38
    c8   1        (4,0,0)      0.433   3.142    22    20    36     26    26    30
    c8   1        (8,8,8)      0.125  10.883    12    10    16     12    12    16
    c8   1+0.1i   (3,1,0)      0.548   2.496    28    26    48     36    36    38
    c4   1        (4,0,0)      0.433   6.283    22    20    38     26    26    32
    c4   1        (8,0,0)      0.217  12.566    14    14    24     16    16    22
    c4   1        (32,0,0)     0.054  50.265    14    12    18     16    14    16
    c4   1+1i     (4,0,0)      0.433   8.886    26    24    40     32    32    34
    c4   1        (8,8,8)      0.125  21.766    14    12    20     16    14    18
    sl   1        (4,0,0)      0.354   0.785    22    20    30     26    24    28
    sl   1        (8,0,0)      0.177   1.571    14    12    18     16    14    18
    sl   1        (2,2,0)      0.501   0.555    -1    32    46     38    36    38
    sl   1        (0,0,64)     0.354   0.785    24    22    30     26    24    28
    sl   1+0.1i   (4,0,0)      0.354   0.789    22    20    30     26    24    28
    sl   1+1i     (16,0,0)     0.089   4.443    12    10    14     14    12    14
    sl   1        (0,0,32)     0.708   0.393    -1    -1    90     70    72    72

Over the 25 cases where all three are defined:

    Lold/Lnew        1.00 .. 1.50   median 1.21   ->  cost (L^2)  1.00 .. 2.25  median 1.47
    Lnew/Lmeas       1.00 .. 1.29   median 1.14
    Lold/Lmeas       1.17 .. 1.73   median 1.40

At (2,0,0) the old bound gives **no certificate at all** below L = 136 while Theorem A
certifies L = 126.  (`bands.jl` of the theory agent reports Lwb = 174 there with a longer
tail cap; the two are consistent -- the tail is summed further there.)

Comparison with the measured L of famB (`famB.md` S3.3, L(1e-13) column, cube 1/32):
famB measures 32 (3,0,0), 26 (4,0,0), 18 (6,0,0), 16 (8,0,0), 12 (16,0,0), 10 (32,0,0),
8 (64,0,0) against my Lmeas 32 / 24 / -- / 16 / 10 / 8 / 8 (famB's target is 1e-13 relative to
|T_ab|, mine 1e-13*est, and est/max|T| is 1.3-3.8, hence the one-shell differences).
Theorem A over-selects those by 8, 4, --, 0, 2, 2, 0 shells; Theorem (c) by 18, 8, --, 4, 4, 4, 2.

---

## 4. Theorem B (one affine-weight sub-box)

On a sub-box `D_j = prod [-h_i,h_i]` with `u(d') = prod (al_i + bt_i d'_i) >= 0` and local
centre `V = R + c_j`, the weight no longer vanishes on the boundary, so the 1D identities

    int u X'  = u^+ X(h) - u^- X(-h) - bt int X ,
    int u X'' = u^+ X'(h) - u^- X'(-h) - bt (X(h) - X(-h))

leave face and edge terms, and on the diagonal one *first* derivative of phi, which Lemma 1
controls exactly.  With `jhat(q) = |k|^q e^{(|k| r_j)^2/(4q+6)}/(2q+1)!!` and `N_q[.]` the
`|d'|^q` moment of the indicated product measure (odd q by Cauchy-Schwarz),

    |T^(j)_ab - T^(j),L_ab| <= (|k|/(4 pi |f|^2 V_t)) sum_{l > L} (2l+1) |h_l(k|V|)| Xi^{ab}_l ,
    Xi^{ab}_l = jhat(l) N_l[nu_a (x) nu_b (x) u_c]                                    (a != b)
    Xi^{aa}_l = sqrt(l/(2l+1)) jhat(l-1) |k| N_{l-1}[pi_a (x) u_b (x) u_c]
              + sqrt((l+1)/(2l+1)) jhat(l+1) |k| N_{l+1}[pi_a (x) u_b (x) u_c]
              + jhat(l) N_l[bet_a (x) u_b (x) u_c] + |k|^2 jhat(l) N_l[u_a (x) u_b (x) u_c]
    nu_i  = u_i^+ delta_{h_i} + u_i^- delta_{-h_i} + |bt_i| dt ,
    pi_i  = u_i^+ delta_{h_i} + u_i^- delta_{-h_i} ,   bet_i = |bt_i|(delta_{h_i}+delta_{-h_i}) .

ALL l contribute (subdivision destroys the parity), and for the octant split `u_i^+ = 0`,
`u_i^- = s_i`, so only the inner face carries a derivative.

The identity was verified against direct differentiation of phi under the integral
(`chk0b.jl`): `max|direct - ibp|/max|direct|` = **5.6e-48 .. 1.2e-43**, (1/32)^3 octant
(1,1,1) and slender octant (-1,1,-1), all six (a,b), l <= 6.

### Ratios and L (script `chkB.jl`, raw `out_chkB_main.txt`; truth = BigFloat Gauss rule on the
sub-box, order 24, self-consistent to 8e-36 .. 5e-40; the shell sums reproduce it to 3.8e-16
.. 2.4e-23 at L = 30).  45 rows, no violation.

    bound/truth    Theorem B: 5.51 .. 163 (median 26.2)    eq (bnd:sub) today: 130 .. 1.13e5
                                                            (median 2.58e3);  gain median 96x

    shape n         octant     rho_j   |kV|    L    truth      Thm B      B/tru  old/tru
    c32   (2,0,0)   (+,+,+)    0.333   0.510   12  9.510e-10  1.096e-8   11.5    738
    c32   (2,0,0)   (+,+,+)    0.333   0.510   24  3.960e-16  9.461e-15  23.9    2310
    c32   (2,0,0)   (-,+,+)    0.522   0.326   12  2.077e-7   7.843e-6   37.8    5070
    c32   (2,0,0)   (-,+,+)    0.522   0.326   30  5.003e-13  2.381e-11  47.6    10800
    c32   (2,1,1)   (+,+,+)    0.264   0.644   24  2.438e-18  2.037e-17   8.36   577
    c4    (2,0,0)   (-,+,+)    0.522   2.605   18  1.741e-9   1.155e-7   66.4    92800
    c4    (2,1,1)   (+,+,+)    0.264   7.283   24  1.207e-18  1.052e-17   8.71   1640
    sl    (2,0,0)   (-,+,+)    0.448   0.311   18  5.471e-11  3.854e-9   70.5    3020
    sl    (0,0,8)   (+,+,-)    0.834   0.167   30  5.240e-6   1.696e-4   32.4    11800

    sub-box L (tol = 1e-13 est/8):     Lmeas  Thm B  old
    c32 (2,0,0) (+,+,+)  rho_j 0.333    24     27    31
    c32 (2,1,1) (+,+,+)  rho_j 0.264    21     23    26
    c32 (4,0,0) (+,+,+)  rho_j 0.190    17     19    21
    c4  (2,1,1) (+,+,+)  rho_j 0.264    22     24    28
    c32 (2,0,0) (-,+,+)  rho_j 0.522    -1     46    55
    c4  (2,0,0) (-,+,+)  rho_j 0.522    -1     44    56
    sl  (2,0,0) (-,+,+)  rho_j 0.448    -1     40    47
    sl  (0,0,8) (+,+,-)  rho_j 0.834    -1     -1    -1   (neither certifies; truth 5.2e-6 at L=30)

Over-selection `Lnew/Lmeas` = 1.08-1.13 against `Lold/Lmeas` = 1.24-1.29 (famBsub reported
1.2-1.3 for the old sub-box bound: confirmed).  L saving 1.11-1.27x, i.e. 1.25-1.60x in cost.

My evaluation of the OLD sub-box bound reproduces `theory.md` S4's octant table exactly: for
c32 (2,0,0) it gives 31 for the four near octants (rho_j = 0.333) and 55 for the four far ones
(rho_j = 0.522), which is theory's "31,31,31,31,55,55,55,55".  Theorem B gives 27 and 46 for
the same two octants, so the whole octant set at 2 cells is certified below L = 48 -- i.e.
below famBsub's cap, which `theory.md` S5.2 records as unresolved against theory's 55-56.  I do
not resolve *why* famBsub's evaluation of the old formula gave 48; I only note that with
Theorem B the cap of 48 is enough and the inconsistency stops mattering.

---

## 5. Theorem C (the singular organization, with 17 removed)

    |T_ab - T^(L),sng_ab| <= (|k|/(|f|^2 V_t)) sum_{l>L, even}
        [ Theta_h(l,|R|) + dab |k|^2 sqrt((2l+1)/4pi) |h_l| ] * sqrt((2l+1)/4pi)
        * |k|^l e^{(|k| r_d)^2/(4l+6)}/(2l+1)!! * W_l

with `Theta_h` of Lemma 2.  Term by term the ratio to eq (bnd:rem) is
`~ 1/(34 sqrt(2l) * 1.35)`: 1/260 at l = 16, 1/370 at l = 32.  Measured over the same 116
rows: bound/truth 1.49 .. 792 (median 140), against 1.35e3 .. 1.73e11 (median 1.53e5) for the
bound it replaces -- a median gain of **661x**.  Same structure, same inputs, no new geometry
data: `W_l` and `hb` (or `|h_l|`) as before, only the derivative factor changes from
`17 |k|^2 hb(l+2) sqrt((2l+5)/4pi) * (2l+1)/sqrt(4pi)` to `Theta_h(l) * sqrt((2l+1)/4pi)`.
**This is a drop-in replacement for `remBnd` if the singular organization is what is summed.**

---

## 6. The other two routes asked for

**Route (2), the Cauchy estimate along R + t d.**  Does not apply.  The l-truncation is not a
truncation of the Taylor series in t: shell l contains the degrees l, l+2, l+4, ... and degree
q contains the shells q, q-2, ..., so neither tail majorises the other.  A Cauchy estimate on
`|t| = beta/rho_d` bounds the *degree-graded* remainder of family (a)/(g), where famA already
measured it 2.0-7.7x pessimistic in the selected degree, and it is silent for
`1/sqrt2 <= rho < 1` (the complex ball touches the null cone at `|R|/sqrt2`, Remark
(bnd:two)).  It is not a route to a sharper l-tail bound and I did not pursue it further.

**Route (3), computable Bessel magnitudes instead of majorants.**  Adopted for `h_l` (default),
rejected as negligible for `j_l`.  Measured at L = 32 over 29 cases, tail with `hb(l,kR)`
divided by tail with `|h_l(kR)|`:

    case          kR      hb/|h|      case          kR      hb/|h|
    c32 (2,0,0)   0.393    1.478      c32 (32,0,0)   6.283   297.1
    c32 (4,0,0)   0.785    2.173      c4  (4,0,0)    6.283   298.7
    c32 (8,0,0)   1.571    4.637      c8  (8,8,8)   10.883   9.10e3
    c32 (8,8,8)   2.721   13.6        c32 (64,0,0)  12.566   2.71e4
    c8  (4,0,0)   3.142   20.0        c4  (8,0,0)   12.566   2.72e4
    c32 (16,0,0)  3.142   19.97       c4  (8,8,8)   21.766   2.28e6
    c4  (4,0,0)f=1+1i 8.886 7.609     c4  (32,0,0)  50.265   7.14e4

i.e. `hb` blows up once `|kR|` approaches the l of the tail (l ~ L+2 = 34 here), non-monotonically
(the c4 (8,8,8) row at kR = 21.8 is worse than the c4 (32,0,0) row at kR = 50.3, because the
ratio depends on the phase of h_l as well as on kR/l).  But because L is set by rho a constant
factor moves it by only a few shells: **using `hb` costs a median 8% (at most 38%) in the selected L**
(column `Lnew(hb)` of S3.2).  Both variants are in `bound2.jl` (`hm = :ex` / `:bd`); `:bd` is
fully certified, `:ex` relies on the upward `h_l` recurrence, measured accurate to 3e-15 for
l <= 40 (famB) but not proven.
Replacing the exponential majorant of `j_l` by the termwise-absolute series
`sum_n |k|^{l+2n} V_{l+2n} / (2^n n! (2l+2n+1)!!)` gains **0.002% to 2.2%** and never changes
the selected L: the `j_l` majorant is not where the looseness is (Theorem (a) is 1.0006-1.5
loose in the regime used, as the theory report already measured).

---

## 7. The deliverable

`work/bound2/bound2.jl`, standalone, generic in `T`, every constant typed:

    boundL(s::NTuple{3,T}, R::AbstractVector{T}, f::Complex{T}, L::Int;
           lMax::Int = 140, hm::Symbol = :ex) -> T          # max over the 9 entries
    boundLent(same args) -> 3x3 Matrix{T}                    # per (a,b)
    boundLsub(al, bt, h::NTuple{3,T}, V, Vt::T, f, L; lMax, hm) -> T

`boundL` has the call signature of `remBnd(s, R, f, L; lMax)` in `work/theory/chkRem_fns.jl`
and returns the same kind of object (an absolute bound on `max_ab |T_ab - T^(L)_ab|`), so it
drops into the band-selection code unchanged.  Internals: `hScl` carries
`|h_l| |k|^l/(2l+1)!!` by the scaled upward recurrence
`hs_{l+1} = (2l+1)|k|/((2l+3)z) hs_l - |k|^2/((2l+1)(2l+3)) hs_{l-1}`, which keeps every
intermediate near `1/((2l+1)|R|^l)` so nothing overflows in Float64 for `l <= 160`,
`|R| >= 1/64`; `hMaj` is the same quantity with the proven majorant, accumulated term by term
with the scaling folded in (no cancellation).  The tail is summed to `lMax` with a geometric
continuation at the ratio attained there, and returns `Inf` if that ratio is >= 1 -- the guard
`theory.md` S5.8 asks for.

There is also a cached form, because `V^{ab}_l` depends on the cell shape only once the
`|k|^2 W_l` piece is split off:

    farMom(s, lMax)                       -> (fc, wr, s, lMax), once per shape
    boundL(mom, R, f, L; hm)              -> T, per offset, O(lMax) work

Verification of the deliverable (`chkC.jl`, `out_chkC.txt`):

    Float64 vs BigFloat(256), 108 whole-box evaluations (12 offsets x 3 f x 3 L, 4 shapes):
        worst relative difference 3.38e-15
    Float64 vs BigFloat(256), 108 octant sub-box evaluations:  worst 3.74e-15
    boundL(mom, ...) vs boundL(s, ...) over the same 108:      0.0 (bit identical)
    boundL Float64 at c32 (4,0,0), L = 16, f = 1: 4.618376431262725e-11 against the 320-bit
        check-script value 4.618376431262723e-11 -- the scaled h_l recurrence of bound2.jl and
        the closed finite Hankel sum of the check scripts agree to the last bit.

Timings (single thread, machine NOT quiet -- other agents' jobs were running, so treat these as
upper bounds and re-measure on a quiet machine):

    farMom(s, lMax = 140)              31.4 ms   once per cell shape
    boundL(mom, R, f, L)               3.8 us (L = 8), 3.9 us (16), 9.7 us (32)   per offset
    boundL(s, R, f, L)  (no table)     0.99-1.09 ms   (the O(lMax^3) moment build dominates)
    boundLsub(al, bt, h, V, Vt, f, L)  7.6-7.9 ms     (no table; the odd-l Cauchy-Schwarz
                                       moments are rebuilt per l -- an obvious 100x saving is
                                       available there and was not done)

---

## 8. LaTeX

`work/bound2/bounds2.tex`, 6 pages, same macros as `bounds.tex` (`\dd`, `\ii`, `\code`, the
theorem environments, `booktabs`, `enumitem`), cross-referencing the theory fragment's labels.
Compile test: `work/bound2/tex/` holds a copy of the theory agent's five fragments plus
`test.tex` with `\input{bounds2.tex}` added before `join.tex`; three `pdflatex` passes:

    exit status 0,  errors 0,  undefined references 0,  Overfull \hbox 0,  Underfull 0
    Output written on test.pdf (34 pages)

---

## 9. What is still loose, honestly

The remaining factor of 9-100 in Theorem A, in order of size:

1. **Cauchy-Schwarz over m.**  Equality needs `M_lm ~ Y_lm(Rhat)`.  For the whole box only one
   parity class per entry is non-zero, so between 1 and `sqrt(#surviving m)` is given away and
   no argument recovers it without reading the geometry table.  This is now the dominant term.
2. **Minkowski under the delta integral**, which discards the cancellation of `Y_lm` over the
   box.  The same loss the `W_l`/`V_l` moments already minimise for the radial part.
3. **Absolute values across l**: the true l-sum has `sum_l |t_l|/|T| <= 8.4` at lambda/32 and
   <= 67 at lambda/4 (famB), so up to 1.8 digits are unavoidable in any termwise bound.
4. **Lemma 2 (1.00-1.41) and the `j_l` majorant (1.0006-1.5)**: now negligible.

Not proven anywhere here: any rounding-error statement.  `hm = :ex` relies on the measured
accuracy of the `h_l` upward recurrence; `hm = :bd` is proven but costs a median 8% in L.
The non-monotonicity of the true remainder means bound/truth at a *fixed* L can reach 2.9e3 at
an accidental dip; the number that matters, the selected L, over-selects by 1.00-1.29 (median
1.14).

Not done: the (2,0,0) whole-box certificate is L = 126, which is above the library's
`Lmax = 56`, so those twelve offsets still go to the octant split -- with Theorem B they need
L_j <= 46 rather than <= 56.  The slender needle `(0,0,n)`, n <= 22, still has rho > 1 and no
bound of this family can certify it; the octant of `sl (0,0,8)` has rho_j = 0.834 and neither
bound certifies it below L = 116.

---

## 10. Files

    work/bound2/bnd2.jl        kernel: j_l, h_l, hb, Y_lm, Gauss-Legendre, moments, Lemmas 1-2
    work/bound2/shell.jl       exact shell data of both organizations + reference tensor
    work/bound2/bounds.jl      the old bound and the two new ones as term arrays (BigFloat)
    work/bound2/bound2.jl      THE DELIVERABLE (standalone, generic in T)
    work/bound2/bounds2.tex    THE LaTeX FRAGMENT
    work/bound2/chk0.jl        Lemmas 1, 2, 3  -> out_chk0.txt
    work/bound2/chk0b.jl       the sub-box IBP identity -> out_chk0b.txt
    work/bound2/chkA.jl        whole box end to end -> out_chkA_main.txt, out_chkA_far.txt
    work/bound2/chkB.jl        sub-box end to end -> out_chkB_main.txt
    work/bound2/chkB_fns.jl    shared by chkB/chk0b
    work/bound2/chkC.jl        the deliverable, Float64 vs BigFloat, timings -> out_chkC.txt
    work/bound2/chkD.jl        first-shell content of the two organizations -> out_chkD.txt
    work/bound2/refcache.txt   cached 320-bit reference tensors (key shape@offset@f@order)
    work/bound2/tex/           compile test of bounds2.tex inside the theory test document

Run as `JULIA_NUM_THREADS=1 julia --startup-file=no --project=SCRATCH/env <script>`;
`SMOKE=1` shrinks every script to one case, `FAR=1 PREC=512 HEXP=38` is the far-offset rerun.
