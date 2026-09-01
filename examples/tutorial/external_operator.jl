using GilaElectromagnetics

source_cells = (8, 8, 8)
source_scale = (1//32, 1//32, 1//32)
source_origin = (0//1, 0//1, 0//1)

target_origin = (1//1, 0//1, 0//1)
target_cells = source_cells

target_volume = GlaVol(target_cells, source_scale, target_origin)
source_volume = GlaVol(source_cells, source_scale, source_origin)

Gext = VacuumGreensOperator(target_volume, source_volume)

println("External operator size: ", size(Gext))
