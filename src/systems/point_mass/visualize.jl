function visualize_system!(env::PointMass)
    X = env.x
    O = env.o
    r = env.r

    fig = Figure()
    ax = Axis(fig[1,1], aspect=DataAspect(), limits=(-10.,10.,-10.,10.))
    robot_position_observable = Observable(SVector{2, Float64}(X...))

    scatter!(ax, [O[1]], [O[2]]; marker=:circle, markersize=r, color=:red, markerspace=SceneSpace)
    scatter!(ax, robot_position_observable; marker=:circle, markersize=0.5, color=:black, markerspace=SceneSpace)

    tail = CircularBuffer{SVector{2, Float64}}(10000)
    fill!(tail, SVector{2, Float64}(X...))
    tail = Observable(tail)
    c = to_color(:purple)
    tailcol = [RGBA(c.r, c.g, c.b, (i/300)^2) for i in 1:10000]
    lines!(ax, tail; linewidth = 3, color = tailcol)
    
    env.obs_x = robot_position_observable
    env.obs_tail = tail

    hidedecorations!(ax)
    display(fig)
    
    return ax, fig
end