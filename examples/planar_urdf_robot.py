from copy import deepcopy
import pdb
import gymnasium as gym
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.urdf_common.helpers import add_shape as urdf_env_add_shape
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor
import os

from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.obstacles.sphere_obstacle import SphereObstacle

import numpy as np
import os
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner

# TODO hardcoding the indices for subgoal_1 is undesired
import sys
import subprocess
import os
import torch
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.utils import torus
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)

# Set the device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

def add_sample_torus_point_cloud(env):
    points = sample_points_from_meshes(meshes=torus(3, 10, 50, 80, device),
                                       num_samples=100)
    px, py, pz = points.clone().detach().cpu().squeeze().unbind(1)
    for ix, x in enumerate(np.array(px)):
        for iy, y in enumerate(np.array(py)):
            for iz, z in enumerate(np.array(pz)):
                color = [np.random.random(), np.random.random(), np.random.random(), 0.5]
                pt = SphereObstacle(name=f"pt{ix}_{iy}_{iz}", content_dict={
                    "type": "sphere",
                    "geometry": {"position": [0.1 * float(x), 0.1 * float(y), 0.1 * float(z) + 3.0], "radius": 0.01},
                    "rgba": color
                })
                env.add_obstacle(pt)

def add_torus(env, rings, sides):
    from math import cos, pi, sin
    r = 3
    R = 10

    thetas = np.linspace(0.0, 2*pi, num=rings)
    phis = np.linspace(0.0, 2 * pi, num=sides)

    for i, phi in enumerate(phis):
        # phi ranges from 0 to 2 pi
        color = [float(phi / 2*pi), float(np.random.random()), float(np.random.random()), 0.5]
        for j, theta in enumerate(thetas):
            # theta ranges from 0 to 2 pi
            x = (R + r * cos(theta)) * cos(phi)
            y = (R + r * cos(theta)) * sin(phi)
            z = r * sin(theta)
            # This vertex has index i * sides + j
            pt = SphereObstacle(name=f"pt{i}_{j}", content_dict={
                "type": "sphere",
                "geometry": {"position": [0.1 * float(x), 0.1 * float(y), 0.1 * float(z) + 3.0], "radius": 0.05},
                "rgba": color
            })
            env.add_obstacle(pt)


pt_cnt = 0
def add_toroidal_point(env, theta, phi):
    global pt_cnt
    pt_cnt+=1
    from math import cos, pi, sin
    r = 3
    R = 10

    color = [float(np.random.random()), float(np.random.random()), float(np.random.random()), 0.5]
    x = (R + r * cos(theta)) * cos(phi)
    y = (R + r * cos(theta)) * sin(phi)
    z = r * sin(theta)

    # This vertex has index i * sides + j
    urdf_env_add_shape(
        shape_type="sphere",
        size=[0.02],
        color=color,
        position=(0.1 * float(x), 0.1 * float(y), 0.1 * float(z) + 2.0)
    )

# https://en.wikipedia.org/wiki/Spherical_coordinate_system
# theta: zenith/polar/inclination/normal angle
# phi: azimuthal angle
def add_spherical_point(env, theta, phi, a=1 , b=1, c=1):
    global pt_cnt
    pt_cnt+=1
    from math import cos, pi, sin, sqrt
    R = 10

    color = [float(np.random.random()), float(np.random.random()), float(np.random.random()), 0.5]
    x = (R * sin(theta) * cos(phi))/sqrt(a)
    y = (R * sin(theta) * sin(phi))/sqrt(b)
    z = (R * cos(phi))/sqrt(c)

    # This vertex has index i * sides + j
    urdf_env_add_shape(
        shape_type="sphere",
        size=[0.02],
        color=color,
        position=(0.1 * float(x), 0.1 * float(y), 0.1 * float(z) + 3.0)
    )

def initalize_environment(render=True):
    """
    Initializes the simulation environment.

    Adds obstacles and goal visualizaion to the environment based and
    steps the simulation once.
    """
    robots = [
        GenericUrdfReacher(urdf="planar_urdf_2_joints.urdf", mode="acc"),
    ]
    env: UrdfEnv  = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    full_sensor = FullSensor(
            goal_mask=["position", "weight"],
            obstacle_mask=["position", "size"],
            variance=0.0,
    )
    # Definition of the obstacle.
    static_obst_dict = {
        "type": "sphere",
        "geometry": {"position": [0.0, -0.9, 0.3], "radius": 0.1},
    }
    obst1 = SphereObstacle(name="staticObst", content_dict=static_obst_dict)
    static_obst_dict = {
        "type": "sphere",
        "geometry": {"position": [-0.0, 1.2, 1.4], "radius": 0.3},
    }
    obst2 = SphereObstacle(name="staticObst", content_dict=static_obst_dict)
    # Definition of the goal.
    goal_dict = {
        "subgoal0": {
            "weight": 1.0,
            "is_primary_goal": True,
            "indices": [1, 2],
            "parent_link": "panda_link0",
            "child_link": "panda_link4",
            "desired_position": [1.2, 1.4],
            "epsilon": 0.05,
            "type": "staticSubGoal",
        },
    }
    visualize_goal_dict = deepcopy(goal_dict)
    visualize_goal_dict['subgoal0']['indices'] = [0] + goal_dict['subgoal0']['indices']
    visualize_goal_dict['subgoal0']['desired_position'] = [0.0] + goal_dict['subgoal0']['desired_position']
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    vis_goal = GoalComposition(name="goal", content_dict=visualize_goal_dict)
    obstacles = (obst1, obst2)
    env.reset()

    # Add sensor
    env.add_sensor(full_sensor, [0])
    for obst in obstacles:
        env.add_obstacle(obst)
    for sub_goal in vis_goal.sub_goals():
        env.add_goal(sub_goal)
    env.set_spaces()
    return (env, goal)


def set_planner(goal: GoalComposition, degrees_of_freedom: int = 2):
    """
    Initializes the fabric planner for the panda robot.

    This function defines the forward kinematics for collision avoidance,
    and goal reaching. These components are fed into the fabrics planner.

    In the top section of this function, an example for optional reconfiguration
    can be found. Commented by default.

    Params
    ----------
    goal: StaticSubGoal
        The goal to the motion planning problem.
    degrees_of_freedom: int
        Degrees of freedom of the robot (default = 7)
    """

    ## Optional reconfiguration of the planner
    # base_inertia = 0.03
    # attractor_potential = "20 * ca.norm_2(x)**4"
    # damper = {
    #     "alpha_b": 0.5,
    #     "alpha_eta": 0.5,
    #     "alpha_shift": 0.5,
    #     "beta_distant": 0.01,
    #     "beta_close": 6.5,
    #     "radius_shift": 0.1,
    # }
    # planner = ParameterizedFabricPlanner(
    #     degrees_of_freedom,
    #     robot_type,
    #     base_inertia=base_inertia,
    #     attractor_potential=attractor_potential,
    #     damper=damper,
    # )
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    with open(absolute_path + "/planar_urdf_2_joints.urdf", "r") as file:
        urdf = file.read()
    forward_kinematics = GenericURDFFk(
        urdf,
        root_link="panda_link0",
        end_links=["panda_link4"]
    )
    planner = ParameterizedFabricPlanner(
        degrees_of_freedom,
        forward_kinematics=forward_kinematics
    )
    q = planner.variables.position_variable()
    collision_links = ['panda_link1', 'panda_link4']
    self_collision_pairs = {}
    panda_limits = [
            [-2.8973, 2.8973],
            [-1.7628, 1.7628],
        ]
    # The planner hides all the logic behind the function set_components.
    planner.set_components(
        collision_links=collision_links,
        self_collision_pairs=self_collision_pairs,
        goal=goal,
        number_obstacles=2,
        #limits=panda_limits,
    )
    planner.concretize()
    return planner


def run_panda_example(n_steps=5000, render=True):
    (env, goal) = initalize_environment(render)
    planner = set_planner(goal)
    action = np.zeros(env.n())
    ob, *_ = env.step(action)


    for _ in range(n_steps):
        ob_robot = ob['robot_0']
        action = planner.compute_action(
            q=ob_robot["joint_state"]["position"],
            qdot=ob_robot["joint_state"]["velocity"],
            x_goal_0=goal.sub_goals()[0].position(),
            weight_goal_0=goal.sub_goals()[0].weight(),
            x_obst_0=ob_robot['FullSensor']['obstacles'][2]['position'],
            radius_obst_0=ob_robot['FullSensor']['obstacles'][2]['size'],
            x_obst_1=ob_robot['FullSensor']['obstacles'][3]['position'],
            radius_obst_1=ob_robot['FullSensor']['obstacles'][3]['size'],
            radius_body_panda_link1=0.2,
            radius_body_panda_link4=0.2,
        )
        q = ob_robot["joint_state"]["position"]
        ob, *_ = env.step(action)
        add_toroidal_point(env, q[0], q[1])
        add_spherical_point(env, q[0], q[1])
    env.close()
    return {}


if __name__ == "__main__":
    res = run_panda_example(n_steps=5000)
