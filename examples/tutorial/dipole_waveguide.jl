using GilaElectromagnetics
using LinearAlgebra
include("plot_utils.jl")
using .PlotUtils

cells = (64, 32, 32)
scale = (1//32, 1//32, 1//32)

susceptibility = zeros(ComplexF64, cells)
for y in 12:20, z in 12:20, x in 1:cells[1]
    susceptibility[x,y,z] = 1.5 + 0im
end

dipole = zeros(ComplexF64, prod(cells))
center = (32, 16, 16)
index = (center[3]-1)*cells[1]*cells[2] + (center[2]-1)*cells[1] + center[1]
dipole[index] = 1.0

g0 = VacuumGreensOperator(cells, scale)
W = ScatteringOperator(cells, susceptibility)

j_total = dipole + W * dipole
field_total = (im/1) .* (g0 * j_total)
field_total = reshape(field_total, cells)

fig = plot_intensity_slice(field_total, :y, 16, title="Waveguide slice")
save("waveguide.png", fig)
