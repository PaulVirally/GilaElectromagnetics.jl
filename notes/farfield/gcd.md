# Non-integer cell ratios: the rational-gcd extension

`farSetupX` refuses a pair whose edges are not integer multiples per axis, matching Gila's own
`GlaExtInf` rule. Route 1 (the whole trapezoid box) needs nothing for such a pair; route 2 needs only
the *rational* gcd cell instead of `min(sT, sS)`. Verified in the cross-scale round on a patched copy
(`gcdexp.jl`): `(3/64, g, g)` against `g` and `g` against `(1/48)^3`, route 1 to 1.0e-15 and route 2
(6 and 216 pieces) to 5.4e-16 against the 220-bit reference, with the swap identity at 2.8e-16.

Three hunks, to be applied **by hand**:

1. `farSetupX`: drop the `all(isinteger, ...)` assertion and replace
   `gQ = ntuple(d -> min(sTQ[d], sSQ[d]), 3)` by `gQ = ntuple(d -> gcd(sTQ[d], sSQ[d]), 3)`.
2. `farTensorX`: the same `gQ` line.
3. `farBlockX!`: the same `gQ` line.

`Rational` `gcd` is exact, so `nT = Int(sTQ / gQ)` then succeeds and defect C1 (an `InexactError`
raised before the clear message) becomes moot.

**Do not** reconstruct this from a diff against `scratch/xwork/xverify/farfield_gcd.jl`: that copy was
taken before the D13 and D15 fixes, so a mechanical diff also reverts the `[rNc, rHi]` guards, which
are the certification contract (plan §4.1).
