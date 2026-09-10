# Coplanar self-panel moments: closed form, cancellation-free evaluation, and a closed-form Gila contact integral

Files: `mom7.jl` (the three implementations), `t1c.jl` (verification), `t3.jl`/`t3b.jl`
(cancellation), `t5.jl` (recurrence), `t4.jl` (wekS comparison).

## 1. The general formula

With `h = sqrt(p^2 + q^2)` and

$$\mathrm{mom}(m)=\int_0^p\!\!\int_0^q (p-u)(q-v)\,(u^2+v^2)^{m+\frac12}\,dv\,du ,$$

$$\boxed{\;
\mathrm{mom}(m)=B_m\Big[p^2q^2\sum_{j=1}^{m+1}\frac{(2j-2)!!}{(2j-1)!!}\,h^{2j-1}\big(p^{2(m+1-j)}+q^{2(m+1-j)}\big)
+p\,q^{2m+4}\operatorname{asinh}\!\tfrac pq+q\,p^{2m+4}\operatorname{asinh}\!\tfrac qp\Big]
-\frac{h^{2m+5}-p^{2m+5}-q^{2m+5}}{(2m+3)(2m+4)(2m+5)}\;}$$

$$B_m=\frac{(2m+1)!!}{(2m+2)!!\,(2m+3)(2m+4)} .$$

**It is valid for every integer $m\ge-1$** (empty sum at $m=-1$), so the same
expression reproduces the known $1/r$ self-panel form
$\tfrac12\big(p^2q\operatorname{asinh}\frac qp+pq^2\operatorname{asinh}\frac pq\big)-\tfrac16(h^3-p^3-q^3)$.
The $m=-1$ term $M_{11}=(h^3-p^3-q^3)/3$ is literally the $m=-1$ member of the
family $(h^{2m+5}-p^{2m+5}-q^{2m+5})$.

### Derivation

Polar coordinates about the corner, $u=\rho\cos\theta$, $v=\rho\sin\theta$. The radial
integral is exact:
$$\mathrm{mom}(m)=\int_0^{\pi/2}\Big[\frac{pq\,R^{N}}{N}-\frac{(p\sin\theta+q\cos\theta)R^{N+1}}{N+1}+\frac{\cos\theta\sin\theta\,R^{N+2}}{N+2}\Big]d\theta,\qquad N=2m+3,$$
with $R=p\sec\theta$ on $[0,\theta_0]$, $R=q\csc\theta$ on $[\theta_0,\pi/2]$,
$\theta_0=\arctan(q/p)$. On the first wedge the two $\sin\theta\sec^{N+1}\theta$
contributions combine into $-\frac{p^{N+2}}{(N+1)(N+2)}\int_0^{\theta_0}\sin\theta\sec^{N+1}\theta\,d\theta
=-\frac{p^2h^{N}-p^{N+2}}{N(N+1)(N+2)}$ (elementary, $d(\sec^N)/d\theta=N\sec^N\theta\tan\theta$),
and the two $\sec^N\theta$ contributions combine into $\frac{q\,p^{N+1}}{N(N+1)}S_N$ with
$S_N=\int_0^{\theta_0}\sec^N\theta\,d\theta$. The second wedge is the $p\leftrightarrow q$
mirror. The standard reduction $\int\sec^N=\frac{\sec^{N-2}\theta\tan\theta}{N-1}+\frac{N-2}{N-1}\int\sec^{N-2}$
with $\sec\theta_0=h/p$, $\tan\theta_0=q/p$, $S_1=\operatorname{asinh}(q/p)$ gives the
double-factorial coefficients above; the $\operatorname{asinh}$ coefficient is
$c_N=(N-2)!!/(N-1)!!$, whence $B_m=c_N/(N(N+1))$.

### Two-term recurrence (verified, `t5.jl`)

Let $X_m(p,q)=q\,p^{2m+4}S_{2m+3}$ be the single-wedge object. Then

$$X_{-1}=p^2q\operatorname{asinh}(q/p),\qquad
X_m=\frac{p^2q^2h^{2m+1}}{2m+2}+\frac{2m+1}{2m+2}\,p^2X_{m-1},$$
$$\mathrm{mom}(m)=\frac{X_m(p,q)+X_m(q,p)}{(2m+3)(2m+4)}-\frac{h^{2m+5}-p^{2m+5}-q^{2m+5}}{(2m+3)(2m+4)(2m+5)} .$$

This generates all $m$ at $O(1)$ cost each and agrees with the closed form to
$7\times10^{-66}$ in 220-bit arithmetic for $m=-1\ldots10$ at aspect ratios $1$ to $10^6$.

### Coefficient table

| $m$ | $B_m$ | $(2j-2)!!/(2j-1)!!$, $j=1..m{+}1$ | $1/[(2m{+}3)(2m{+}4)(2m{+}5)]$ |
|---|---|---|---|
| 0 | 1/24 | 1 | 1/60 |
| 1 | 1/80 | 1, 2/3 | 1/210 |
| 2 | 5/896 | 1, 2/3, 8/15 | 1/504 |
| 3 | 7/2304 | 1, 2/3, 8/15, 16/35 | 1/990 |
| 4 | 21/11264 | 1, 2/3, 8/15, 16/35, 128/315 | 1/1716 |
| 5 | 33/26624 | +256/693 | 1/2730 |
| 6 | 143/163840 | +1024/3003 | 1/4080 |

$B_m/B_{m-1}=(2m+1)^2/[(2m+3)(2m+4)]$.

## 2. Verification

Reference implementations: `momPol` (polar form, radial integral exact, angular by
BigFloat Gauss-Legendre) and `pairMom` from `../series/mom.jl` (independent 4D
evaluator; $I_{2m+1}=4\,\mathrm{mom}(m)$ for the self pair). All in 220-bit BigFloat.

| test | result |
|---|---|
| `momGen` vs `momPol` (ord 160), aspect $1,2,6,\tfrac14$, $m=0..10$ | max rel $2.5\times10^{-65}$ |
| `momGen` vs Mathematica output, $m=0..6$, aspect $1..6$ | max rel $1.4\times10^{-65}$ |
| `momGen` vs Mathematica output, $m=0..6$, aspect $10^{3},10^{6}$ | max rel $6.1\times10^{-55}$ (this is the Mathematica form's own cancellation at 66 digits, not a discrepancy) |
| `momGen` vs `pairMom`, $m=0..10$, aspect $1,2,6,\tfrac14,10^{3}$ | max rel $1.3\times10^{-40}$; the evaluator's own order-refinement residual is the same $1.3\times10^{-40}$, i.e. the formula is at the evaluator's noise floor |
| `momGen(-1)` vs the $1/r$ closed form $pqM_{00}-qM_{10}-pM_{01}+M_{11}$ | max rel $6\times10^{-56}$ |
| recurrence vs closed form, $m=-1..10$ | max rel $7\times10^{-66}$ |

`momPol` cannot be used at extreme aspect: at $p/q=10^{-3}$ the $\sec^{2m+5}$ peak near
$\theta_0$ needs adaptive grading, so Gauss-Legendre at order 200 is off by $10^{-2}$.
That was the only "failure" seen and it is the reference's, not the formula's.

## 3. Cancellation

As $p\to0$, $\mathrm{mom}(m)\to \dfrac{p^2q^{2m+3}}{2(2m+2)(2m+3)}$, i.e. $O(p^2q^{2m+3})$.
Every group in the boxed formula is already $O(p^2q^{2m+3})$ **except**
$h^{2m+5}-q^{2m+5}$, which is a difference of two $O(q^{2m+5})$ numbers: that alone
costs $2\log_{10}(q/p)$ digits. The fix is to write it as a manifestly positive product,
$n=2m+5$:

$$h^{n}-b^{n}=\frac{a^{2}}{h+b}\sum_{i=0}^{n-1}h^{i}b^{\,n-1-i},\qquad (a,b)=\mathrm{minmax}(p,q),$$

then subtract $a^{n}$ (negligible). Everything else — the double-factorial sum, both
$\operatorname{asinh}$ terms — is a sum of positive quantities, so no further regrouping is
needed; `asinh(a/b)` is already accurate for small argument. This is `momSaf` in `mom7.jl`.

Digits lost in Float64 relative to 400-bit truth, worst over $m$ ($m\le6$ for the
Mathematica column, $m\le10$ for the others), $q=1$:

| $p/q$ | Mathematica as printed | boxed form as written | regrouped (`momSaf`) |
|---|---|---|---|
| 1e-8 | 14.7 | 15.6 | 0.5 |
| 1e-6 | 10.6 | 11.6 | 0.1 |
| 1e-4 | 6.2 | 7.3 | 0.2 |
| 1e-2 | 2.9 | 3.9 | 0.5 |
| 1e-1 | 1.0 | 1.8 | 0.4 |
| 1 | 0.1 | 0.6 | 0.6 |
| 1e1 | 1.1 | 1.5 | 0.3 |
| 1e2 | 3.2 | 3.1 | 0.4 |
| 1e4 | 7.3 | 7.8 | 0.7 |
| 1e6 | 11.0 | 10.9 | 0.3 |
| 1e8 | $\infty$ (m=0,1,2 lose all significance) | 15.6 | 0.5 |

Target met: $\le0.7$ digits over $p/q\in[10^{-8},10^{8}]$. The Mathematica forms for
$m=0,1,2$ are additionally poisoned by their log groupings — e.g. the $m=1$ combination
$-15\operatorname{asinh}\frac pq+5\log q-12\log(h-p)+7\log(p+h)$ is identically
$4\operatorname{asinh}(p/q)$, so at $p/q=10^{-8}$ four $O(1)$ logs must cancel to
$4\times10^{-8}$. Likewise the $m=2$ term $11\operatorname{asinh}\frac pq-7\log q+\log\frac{(p-h)^8}{p+h}$
is exactly $2\operatorname{asinh}(p/q)$.

## 4. Closed-form Gila contact integral vs `wekS`

Series $S=\frac{1}{4\pi f^{2}}\sum_n\frac{(2\pi i f)^n}{n!}I_{n-1}$ with
$I_{-1}=4\,\mathrm{mom}(-1)$, $I_{2m+1}=4\,\mathrm{mom}(m)$ from the boxed formula, and
$I_{2j}=4\sum_{k=0}^{j}\binom jk\frac{p^{2k+2}}{(2k+1)(2k+2)}\frac{q^{2(j-k)+2}}{(2j-2k+1)(2j-2k+2)}$
(exact polynomial). $f=1$, `intOrd = 64`.

| scl | face $(p,q)$ | `wekS` | closed form | rel | re rel | im rel |
|---|---|---|---|---|---|---|
| $(1/32)^3$ | $(0.03125,0.03125)$ | 7.196101733912074e-6 + 4.7581696358419294e-7i | 7.1961017339062345e-6 + 4.7581696358419357e-7i | 8.1e-13 | 8.1e-13 | 1.3e-15 |
| $(1/8)^3$ | $(0.125,0.125)$ | 4.377010421867182e-4 + 1.1795936256720547e-4i | 4.3770104218073664e-4 + 1.179593625672056e-4i | 1.3e-11 | 1.4e-11 | 1.1e-15 |
| $(1/32,1/32,1/512)$ | $(0.03125,0.001953125)$ | 7.55717633823222e-8 + 1.860644154511704e-9i | 7.557176338212843e-8 + 1.8606441545117055e-9i | 2.6e-12 | 2.6e-12 | 7.8e-16 |
| $(1/32,1/32,1/512)$ | $(0.03125,0.03125)$ | 7.196101733912074e-6 + 4.7581696358419294e-7i | as row 1 | 8.1e-13 | 8.1e-13 | 1.3e-15 |

The series itself is not the limiting factor: Float64 vs 300-bit BigFloat evaluation of
the same series agrees to $1.2\times10^{-16}$, and truncation at $n=40$ vs $n=60$ is
bitwise identical. So the whole discrepancy is `wekS`'s DIRECTFN quadrature, matching the
earlier intOrd-64 error estimate of $\sim10^{-11}$.

Note the split: the imaginary part agrees to $10^{-15}$, the real part only to
$10^{-11..13}$. The imaginary part of $S$ is carried by the *even* moments (exact
polynomials, and Gila's regular quadrature handles them well); the real part contains the
$1/r$ term (which Gila gets exactly from `rSrfSlf`) plus the odd moments $I_1,I_3,\dots$,
whose $r^{2m+1}$ integrands are the ones `wekSInt`/`wekEInt` resolve worst. The closed
form therefore pins down exactly where Gila's contact-integral error lives.

## 5. Surprises

1. **The formula is uniform in $m\ge-1$.** The $1/r$ case is not a special case needing
   its own derivation: $M_{11}=(h^3-p^3-q^3)/3$ is the $m=-1$ member of
   $(h^{2m+5}-p^{2m+5}-q^{2m+5})$, and $B_{-1}=1/2$.
2. **The $\operatorname{asinh}$ exponent is $2m+4$, not $2m+3$**, and the polynomial part
   contains only *odd* powers of $h$ ($h^{2j-1}$, $j=1..m+1$, and $h^{2m+5}$) — the
   $p^{2m+3}h$ / $q^{2m+3}h$ shape guessed in the task brief is what Mathematica's
   $\frac{1}{h}\times(\text{even polynomial})$ presentation hides.
3. **Mathematica's printed forms are not just ugly, they are numerically wrong in Float64**
   for slender panels: the $m=0..2$ outputs carry log groupings that are algebraically
   equal to a single small $\operatorname{asinh}$, so they lose *all* significance at
   $p/q=10^{-8}$ while the $m\ge3$ outputs merely lose 15 digits.
4. **The only cancellation in the whole formula is one term.** Once $h^n-q^n$ is written as
   a product, nothing else needs care — the double-factorial sum and both asinh terms are
   sign-definite and individually of the correct asymptotic size.
5. `pairMom`'s own refinement residual at aspect 2 ($1.3\times10^{-40}$) is larger than at
   aspect $10^{-3}$; the closed form now supersedes it as the reference for this geometry.
