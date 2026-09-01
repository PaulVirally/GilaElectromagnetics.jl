using GilaElectromagnetics

# Domain definition
cells = (8, 8, 8)
scale = (1//32, 1//32, 1//32)

G = VacuumGreensOperator(cells, scale)

sources = rand(eltype(G), size(G, 2))
fields = G * sources

println("Field vector size: ", size(fields))
