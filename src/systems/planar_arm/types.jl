mutable struct PlanarArm 
    θ::Vector{Float64}
    θ̇ ::Vector{Float64}
    o::Vector{Vector{Float64}}
    r::Vector{Float64}
    l1::Float64
    l2::Float64
    l3::Float64
    Δt::Float64
    show_contacts::Bool 
    g::Vector{Float64}
    k::Float64
    λ::Float64
    lb::Vector{Float64}
    ub::Vector{Float64}
    link_observables 
    joint_observables 
    obstacle_observables
    task_maps
    
    function PlanarArm(robot_position, robot_velocity, obstacle_positions, obstacle_radii, goal_position)
        task_maps = [:attractor, :repeller, :joint_lower_limit, :joint_upper_limit]
        k = 0.5
        λ = 0.7
        lb = [0.0, 0.0, 0.0]
        ub = [π, π, π]
        new(robot_position, robot_velocity, obstacle_positions, obstacle_radii, 5.0,5.0,5.0, 10E-4, false, goal_position, k, λ, lb, ub, nothing, nothing, nothing, task_maps)
    end
end