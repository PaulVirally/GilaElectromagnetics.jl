using GilaElectromagnetics
using LinearAlgebra
using StaticArrays
include("plot_utils.jl")
using .PlotUtils

cells = (32, 32, 32)
scale = (1//32, 1//32, 1//32)

radius = 10
susceptibility = fill(0.0 + 0im, cells)
for x in 1:cells[1], y in 1:cells[2], z in 1:cells[3]
    cx = x - cells[1]÷2
    cy = y - cells[2]÷2
    cz = z - cells[3]÷2
    if cx^2 + cy^2 + cz^2 <= radius^2
        susceptibility[x,y,z] = -3.1 + 1.8im
    end
end

kvec = (0.0, 1.0, 0.0)
knorm = norm(SVector(kvec))

G0 = VacuumGreensOperator(cells, scale)
W = ScatteringOperator(cells, susceptibility)

function plane_wave_field(k, x, y, z)
    return exp(im * (k[1]*x + k[2]*y + k[3]*z))
end

E_in = [plane_wave_field(kvec, x-1, y-1, z-1) for x in 1:cells[1], y in 1:cells[2], z in 1:cells[3]]
J_in = -im*knorm .* susceptibility .* E_in
J_gen = W * reshape(J_in, :, 1)
E_scat = im/knorm .* (G0 * J_gen)
E_tot = reshape(E_scat, cells) + E_in

fig = plot_intensity_slice(E_tot, :z, 16, title="Intensity slice")
save("sphere_scat.png", fig)
