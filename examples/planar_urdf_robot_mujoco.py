from copy import deepcopy
import pdb
import gymnasium as gym

from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
from robotmodels.utils.robotmodel import RobotModel, LocalRobotModel

from urdfenvs.generic_mujoco.generic_mujoco_env import GenericMujocoEnv
from urdfenvs.generic_mujoco.generic_mujoco_robot import GenericMujocoRobot

from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.urdf_common.helpers import add_shape as urdf_env_add_shape
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor
import os

from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.obstacles.sphere_obstacle import SphereObstacle

import numpy as np
import os
from typing_extensions import List
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

def add_mujoco_shape(env, name: str, position: List[float], size: float,
                     color: List[float] = None, shape_type: str = "sphere"):
    if not color:
        color = [0, 1, 0, 0.3]
    geom_values = {
        "name": name,
        "type": shape_type,
        "rgba": " ".join([str(c) for c in color]),
        "pos": " ".join([str(pos) for pos in position]),
        "size": str(size),
    }
    env._model_dm.worldbody.add("site", **geom_values)

def add_sample_torus_point_cloud(env):
    points = sample_points_from_meshes(meshes=torus(3, 10, 50, 80, device),
                                       num_samples=100)
    px, py, pz = points.clone().detach().cpu().squeeze().unbind(1)
    for ix, x in enumerate(np.array(px)):
        for iy, y in enumerate(np.array(py)):
            for iz, z in enumerate(np.array(pz)):
                add_mujoco_shape(env, name=f"pnt{ix}_{iy}_{iz}",
                                 position=[0.1 * float(x), 0.1 * float(y), 0.1 * float(z) + 3.0],
                                 size=0.01, color=[np.random.random(), np.random.random(), np.random.random(), 0.5])


def add_torus(env, rings, sides):
    from math import cos, pi, sin
    r = 3
    R = 10

    thetas = np.linspace(0.0, 2 * pi, num=rings)
    phis = np.linspace(0.0, 2 * pi, num=sides)

    for i, phi in enumerate(phis):
        # phi ranges from 0 to 2 pi
        for j, theta in enumerate(thetas):
            # theta ranges from 0 to 2 pi
            x = (R + r * cos(theta)) * cos(phi)
            y = (R + r * cos(theta)) * sin(phi)
            z = r * sin(theta)
            # This vertex has index i * sides + j
            add_mujoco_shape(env, name=f"pnt{i}_{j}",
                             position=[0.1 * float(x), 0.1 * float(y), 0.1 * float(z) + 3.0],
                             size=0.05, color=[float(phi / 2 * pi), float(np.random.random()), float(np.random.random()), 0.5])


pt_cnt = 0


def add_toroidal_point(env, theta, phi):
    global pt_cnt
    pt_cnt += 1
    from math import cos, pi, sin
    r = 3
    R = 10

    x = (R + r * cos(theta)) * cos(phi)
    y = (R + r * cos(theta)) * sin(phi)
    z = r * sin(theta)

    # This vertex has index i * sides + j
    add_mujoco_shape(env, name=f"pnt{pt_cnt}",
                     position=[0.1 * float(x), 0.1 * float(y), 0.1 * float(z) + 2.0],
                     size=0.02, color=[float(np.random.random()), float(np.random.random()), float(np.random.random()), 0.5])


# https://en.wikipedia.org/wiki/Spherical_coordinate_system
# theta: zenith/polar/inclination/normal angle
# phi: azimuthal angle
def add_spherical_point(env, theta, phi, a=1, b=1, c=1):
    global pt_cnt
    pt_cnt += 1
    from math import cos, pi, sin, sqrt
    R = 10

    color = [float(np.random.random()), float(np.random.random()), float(np.random.random()), 0.5]
    x = (R * sin(theta) * cos(phi)) / sqrt(a)
    y = (R * sin(theta) * sin(phi)) / sqrt(b)
    z = (R * cos(phi)) / sqrt(c)

    # This vertex has index i * sides + j
    add_mujoco_shape(env, name=f"pnt{pt_cnt}",
                     position=[0.1 * float(x), 0.1 * float(y), 0.1 * float(z) + 3.0],
                     size=0.02,
                     color=[float(np.random.random()), float(np.random.random()), float(np.random.random()), 0.5])


CONTROL_MODE = "vel"


def initalize_environment(render=True):
    """
    Initializes the simulation environment.

    Adds obstacles and goal visualizaion to the environment based and
    steps the simulation once.
    """
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    xml_file = f"{absolute_path}/planar_2dof.xml"
    robots = [
        GenericMujocoRobot(xml_file=xml_file, mode=CONTROL_MODE),
    ]
    home_config = np.array([0, 0])

    # OBSTACLES
    static_obst_dict1 = {
        "type": "sphere",
        "geometry": {"position": [0.0, -0.9, 0.3], "radius": 0.1},
    }
    obst1 = SphereObstacle(name="staticObst1", content_dict=static_obst_dict1)
    static_obst_dict2 = {
        "type": "sphere",
        "geometry": {"position": [-0.0, 1.2, 1.4], "radius": 0.3},
    }
    obst2 = SphereObstacle(name="staticObst2", content_dict=static_obst_dict2)
    obstacles = [obst1, obst2]

    # GOAL
    goal_dict = {
        "subgoal0": {
            "weight": 1.0,
            "is_primary_goal": True,
            "indices": [1, 2],
            "parent_link": "link0",
            "child_link": "link4",
            "desired_position": [1.0, 1.2],
            "epsilon": 0.1,
            "type": "staticSubGoal",
        },
    }
    goal = GoalComposition(name="goal", content_dict=goal_dict)

    # Add sensor
    full_sensor = FullSensor(
        goal_mask=["position", "weight"],
        obstacle_mask=["position", "size"],
        variance=0.0,
        physics_engine_name="mujoco"
    )
    env = GenericMujocoEnv(robots=robots, obstacles=obstacles, goals=goal.sub_goals(),
                           sensors=[full_sensor],
                           width=2048,
                           height=960,
                           render=render)
    env.reset(pos=home_config)
    return (env, goal)


def set_planner(goal: GoalComposition, dt: float):
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
    degrees_of_freedom = 2
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    with open(absolute_path + "/planar_urdf_2_joints.urdf", "r") as file:
        urdf = file.read()
    forward_kinematics = GenericURDFFk(
        urdf,
        root_link="link0",
        end_links=["link4"]
    )
    planner = ParameterizedFabricPlanner(
        dof=degrees_of_freedom,
        forward_kinematics=forward_kinematics
    )
    q = planner.variables.position_variable()
    collision_links = ['link1', 'link4']
    self_collision_pairs = {}
    # The planner hides all the logic behind the function set_components.
    planner.set_components(
        collision_links=collision_links,
        self_collision_pairs=self_collision_pairs,
        goal=goal,
        number_obstacles=2,
        #limits=limits,
    )
    planner.concretize(mode=CONTROL_MODE, time_step=dt)
    return planner


def run_example(n_steps=5000, render=True):
    (env, goal) = initalize_environment(render)
    planner = set_planner(goal, env.dt)
    action = np.zeros(2)
    ob, *_ = env.step(action)

    for _ in range(n_steps):
        robot = ob['robot_0']
        joint_state = robot["joint_state"]
        goal_0 = goal.sub_goals()[0]
        obstacles = robot['FullSensor']['obstacles']
        obstacle_pos = lambda i: obstacles[i]['position']
        obstacle_size = lambda i: obstacles[i]['size']
        action = planner.compute_action(
            q=joint_state["position"],
            qdot=joint_state["velocity"],
            x_goal_0=goal_0.position(),
            weight_goal_0=goal_0.weight(),
            x_obst_0=obstacle_pos(0),
            radius_obst_0=obstacle_size(0),
            x_obst_1=obstacle_pos(1),
            radius_obst_1=obstacle_size(1),
            #radius_body_link1=0.2,
            radius_body_link4=0.2,
        )
        q = joint_state["position"]
        ob, *_ = env.step(action)
        add_toroidal_point(env, q[0], q[1])
        add_spherical_point(env, q[0], q[1])
    env.close()
    return {}


if __name__ == "__main__":
    res = run_example(n_steps=10000)
