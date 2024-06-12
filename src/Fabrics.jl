module Fabrics

using GLMakie
using DataStructures: CircularBuffer
using ForwardDiff
using LinearAlgebra
using StaticArrays
using Colors 

include("/media/ducthan/376b23a1-5a02-4960-b3ca-24b2fcef8f89/4_REMANIAN_MANIFOLD/RMP_JULIA/Fabrics.jl/src/systems/point_mass/types.jl")
include("/media/ducthan/376b23a1-5a02-4960-b3ca-24b2fcef8f89/4_REMANIAN_MANIFOLD/RMP_JULIA/Fabrics.jl/src/systems/point_mass/visualize.jl")
include("/media/ducthan/376b23a1-5a02-4960-b3ca-24b2fcef8f89/4_REMANIAN_MANIFOLD/RMP_JULIA/Fabrics.jl/src/systems/point_mass/step.jl")
include("/media/ducthan/376b23a1-5a02-4960-b3ca-24b2fcef8f89/4_REMANIAN_MANIFOLD/RMP_JULIA/Fabrics.jl/src/systems/point_mass/fabric.jl")

include("/media/ducthan/376b23a1-5a02-4960-b3ca-24b2fcef8f89/4_REMANIAN_MANIFOLD/RMP_JULIA/Fabrics.jl/src/systems/planar_arm/types.jl")
include("/media/ducthan/376b23a1-5a02-4960-b3ca-24b2fcef8f89/4_REMANIAN_MANIFOLD/RMP_JULIA/Fabrics.jl/src/systems/planar_arm/visualize.jl")
include("/media/ducthan/376b23a1-5a02-4960-b3ca-24b2fcef8f89/4_REMANIAN_MANIFOLD/RMP_JULIA/Fabrics.jl/src/systems/planar_arm/step.jl")
include("/media/ducthan/376b23a1-5a02-4960-b3ca-24b2fcef8f89/4_REMANIAN_MANIFOLD/RMP_JULIA/Fabrics.jl/src/systems/planar_arm/fabric.jl")

include("/media/ducthan/376b23a1-5a02-4960-b3ca-24b2fcef8f89/4_REMANIAN_MANIFOLD/RMP_JULIA/Fabrics.jl/src/systems/multi_planar_arm/types.jl")
include("/media/ducthan/376b23a1-5a02-4960-b3ca-24b2fcef8f89/4_REMANIAN_MANIFOLD/RMP_JULIA/Fabrics.jl/src/systems/multi_planar_arm/visualize.jl")
include("/media/ducthan/376b23a1-5a02-4960-b3ca-24b2fcef8f89/4_REMANIAN_MANIFOLD/RMP_JULIA/Fabrics.jl/src/systems/multi_planar_arm/step.jl")
include("/media/ducthan/376b23a1-5a02-4960-b3ca-24b2fcef8f89/4_REMANIAN_MANIFOLD/RMP_JULIA/Fabrics.jl/src/systems/multi_planar_arm/fabric.jl")


include("/media/ducthan/376b23a1-5a02-4960-b3ca-24b2fcef8f89/4_REMANIAN_MANIFOLD/RMP_JULIA/Fabrics.jl/src/systems/picklerick/types.jl")
include("/media/ducthan/376b23a1-5a02-4960-b3ca-24b2fcef8f89/4_REMANIAN_MANIFOLD/RMP_JULIA/Fabrics.jl/src/systems/picklerick/visualize.jl")
include("/media/ducthan/376b23a1-5a02-4960-b3ca-24b2fcef8f89/4_REMANIAN_MANIFOLD/RMP_JULIA/Fabrics.jl/src/systems/picklerick/step.jl")
include("/media/ducthan/376b23a1-5a02-4960-b3ca-24b2fcef8f89/4_REMANIAN_MANIFOLD/RMP_JULIA/Fabrics.jl/src/systems/picklerick/fabric.jl") 

export visualize_system!,
       step!,
       jvp,
       compute_COM

export PointMass,
       pointmass_fabric_solve,
       move_obstacles!

export PlanarArm,
       link_poses,
       planararm_fabric_solve

export MultiPlanarArm,
       multiplanar_arm_fabric_solve

export PickleRick,
       picklerick_fabric_solve 

end
