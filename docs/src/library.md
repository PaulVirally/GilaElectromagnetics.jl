# Library

The following is the exhaustive list of the API available to users, accompanied by explanations.

## Module Index

```@index
Modules = [GilaElectromagnetics, GilaElectromagnetics.GilaTypes, GilaElectromagnetics.GilaVolumes, GilaElectromagnetics.GilaFields, GilaElectromagnetics.GilaVacuum, GilaElectromagnetics.GilaSolvers, GilaElectromagnetics.GilaOperators]
Order   = [:constant, :type, :function, :macro]
```
## Detailed API

!!! note "Storage precision"
    Operators and fields carry a real type parameter `T<:AbstractFloat` (`Float32`
    or `Float64`); the data itself is `Complex{T}`. Unparameterized constructors
    default to `T = dfltPrc` (`Float32`); request another precision with the type
    parameter, e.g. `GlaOprVac{Float64}(vol)`. See [Precision](usage.md#precision)
    for details.

```@autodocs
Modules = [GilaElectromagnetics, GilaElectromagnetics.GilaTypes, GilaElectromagnetics.GilaVolumes, GilaElectromagnetics.GilaFields, GilaElectromagnetics.GilaVacuum, GilaElectromagnetics.GilaSolvers, GilaElectromagnetics.GilaOperators]
Order   = [:constant, :type, :function, :macro]
Private = false
```
