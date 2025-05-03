module GilaVacuum
"""
    GilaVacuum

This module provides memory structures and operations for the vacuum Green's function operator in Gila.
"""

# Holds computational information (CPU vs GPU, number of threads, etc.) for the Green's function
include("glaVacCmp.jl")
export GlaKerOpt, CPUKerOpt, GPUKerOpt, frqPhz, intOrd, adjMod, bckEnd

# Defines GlaVacOprMem, the struct holding the info to compute the Green's function
include("glaVacOprMem.jl")
export GlaVacOprMem

include("glaVacAct.jl")
export egoOpr!

end
