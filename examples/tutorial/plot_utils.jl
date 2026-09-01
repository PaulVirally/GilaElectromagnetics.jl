module PlotUtils

using GLMakie

export plot_real_slice, plot_intensity_slice

function slice2d(field, axis::Symbol, idx::Int, comp::Int=1)
    if ndims(field) == 4
        if axis == :x
            return field[idx, :, :, comp]'
        elseif axis == :y
            return field[:, idx, :, comp]'
        elseif axis == :z
            return field[:, :, idx, comp]'
        else
            error("axis must be :x, :y or :z")
        end
    else
        if axis == :x
            return field[idx, :, :]'
        elseif axis == :y
            return field[:, idx, :]'
        elseif axis == :z
            return field[:, :, idx]'
        else
            error("axis must be :x, :y or :z")
        end
    end
end

function plot_real_slice(field, axis::Symbol, idx::Int; component::Int=1, title="")
    data = slice2d(field, axis, idx, component)
    fig = Figure()
    heatmap(fig[1,1], real(data))
    fig[1,1].title = title
    fig
end

function plot_intensity_slice(field, axis::Symbol, idx::Int; title="")
    intensity = sum(abs2, field, dims=4)
    data = slice2d(intensity, axis, idx)
    fig = Figure()
    heatmap(fig[1,1], real(data))
    fig[1,1].title = title
    fig
end

end
