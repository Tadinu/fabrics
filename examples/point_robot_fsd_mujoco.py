from copy import deepcopy
import os
import shutil
import logging
import time
import gymnasium as gym
import numpy as np
from typing_extensions import List, Union
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
from forwardkinematics.xmlFks.generic_xml_fk import GenericXMLFk
from robotmodels.utils.robotmodel import RobotModel, LocalRobotModel

from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.generic_mujoco.generic_mujoco_env import GenericMujocoEnv
from urdfenvs.generic_mujoco.generic_mujoco_robot import GenericMujocoRobot
from urdfenvs.sensors.full_sensor import FullSensor

from urdfenvs.sensors.free_space_decomposition import FreeSpaceDecompositionSensor

from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.obstacles.box_obstacle import BoxObstacle
from mpscenes.goals.goal_composition import GoalComposition

from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner
logging.basicConfig(level=logging.INFO)

NUMBER_OF_RAYS = 10
NUMBER_OF_CONSTRAINTS = 1
constraint_0 = np.array([0, 0, 1, 0.0])

ROBOTTYPE = 'pointRobot'
ROBOTMODEL = 'pointRobot'
CONTROL_MODE = 'vel'

urdf_model = RobotModel(ROBOTMODEL)
urdf_file = urdf_model.get_urdf_path()

robot_model = LocalRobotModel(ROBOTTYPE, ROBOTMODEL)
xml_file = robot_model.get_xml_path()

def get_goal_fsd():
    goal_dict = {
        "subgoal0": {
            "weight": 5,
            "is_primary_goal": True,
            "indices": [0, 1],
            "parent_link": 'world',
            "child_link": 'base_link',
            "desired_position": [3.5, 0.5],
            "epsilon": 0.1,
            "type": "staticSubGoal"
        }
    }
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    return goal

def get_obstacles_fsd():
    static_obst_dict = {
        "type": "sphere",
        "geometry": {"position": [2.0, -0.5, 0.15], "radius": 0.8},
    }
    obst1 = SphereObstacle(name="staticObst1", content_dict=static_obst_dict)
    static_obst_dict = {
        "type": "sphere",
        "geometry": {"position": [0.0, 1.0, 0.15], "radius": 0.3},
    }
    obst2 = SphereObstacle(name="staticObst2", content_dict=static_obst_dict)
    static_obst_dict = {
        "type": "sphere",
        "geometry": {"position": [3.0, 1.2, 0.15], "radius": 0.3},
    }
    obst3 = SphereObstacle(name="staticObst3", content_dict=static_obst_dict)
    static_obst_dict = {
        "type": "box",
        "geometry": {"position": [2.0, 1.7, 0.15], "width": 0.3, "length": 0.2, "height": 0.3},
    }
    obst4 = BoxObstacle(name="staticObst4", content_dict=static_obst_dict)
    return [obst1, obst2, obst3, obst4]


def initalize_environment(render: bool = True):
    """
    Initializes the simulation environment.

    Adds an obstacle and goal visualizaion to the environment and
    steps the simulation once.
   j
    Params
    ----------
    render
        Boolean toggle to set rendering on (True) or off (False).
    """
    #if os.path.exists(ROBOTTYPE):
    #    shutil.rmtree(ROBOTTYPE)
    #robot_model = RobotModel(ROBOTTYPE, ROBOTMODEL)
    #robot_model.copy_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), ROBOTTYPE))
    #del robot_model
    robots = [
        GenericMujocoRobot(
            xml_file=xml_file,
            mode=CONTROL_MODE,
        ),
    ]

    # Set the initial position and velocity of the point mass.
    pos0 = np.array([-2.0, 0.5, 0.0])
    vel0 = np.array([0.1, 0.0, 0.0])
    full_sensor = FullSensor(
        goal_mask=["position", "weight"],
        obstacle_mask=["position", "size"],
        variance=0.0,
        physics_engine_name="mujoco"
    )
    fsd_sensor = FreeSpaceDecompositionSensor(
            'lidar_sensor_link',
            max_radius=5,
            plotting_interval=100,
            nb_rays=NUMBER_OF_RAYS,
            number_constraints=NUMBER_OF_CONSTRAINTS,
            physics_engine_name="mujoco"
    )
    # Definition of the obstacle.
    obstacles = get_obstacles_fsd()
    # Definition of the goal.
    goal = get_goal_fsd()
    env = GenericMujocoEnv(robots=robots, obstacles=obstacles, goals=goal.sub_goals(),
                           sensors=[full_sensor], # fsd_sensor
                           width=1024,
                           height=960,
                           render=render)
    env.reset(pos=pos0, vel=vel0)
    return env, goal, obstacles


def set_planner(goal: GoalComposition, dt: float):
    """
    Initializes the fabric planner for the point robot.

    This function defines the forward kinematics for collision avoidance,
    and goal reaching. These components are fed into the fabrics planner.

    In the top section of this function, an example for optional reconfiguration
    can be found. Commented by default.

    Params
    ----------
    goal: StaticSubGoal
        The goal to the motion planning problem.
    """
    collision_geometry = "-2.0 / (x ** 1) * xdot ** 2"
    collision_finsler = "1.0/(x**2) * (1 - ca.heaviside(xdot))* xdot**2"
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    with open(absolute_path + "/point_robot.urdf", "r", encoding="utf-8") as file:
        urdf = file.read()
    collision_links = ["base_link"]
    degrees_of_freedom = 3
    forward_kinematics = GenericURDFFk(
        urdf,
        root_link="world",
        end_links=["base_link"],
    )
    planner = ParameterizedFabricPlanner(
        degrees_of_freedom,
        forward_kinematics,
        geometry_plane_constraint=collision_geometry,
        finsler_plane_constraint=collision_finsler
    )
    collision_links = ["base_link"]
    # The planner hides all the logic behind the function set_components.
    planner.set_components(
        collision_links=collision_links,
        goal=goal,
        number_obstacles=4,
        number_plane_constraints=NUMBER_OF_CONSTRAINTS,
    )
    planner.concretize(mode=CONTROL_MODE, time_step=dt)
    return planner


def run_point_robot(n_steps=10000, render: bool = True):
    """
    Set the gym environment, the planner and run point robot example.
    The initial zero action step is needed to initialize the sensor in the
    urdf environment.

    Params
    ----------
    n_steps
        Total number of simulation steps.
    render
        Boolean toggle to set rendering on (True) or off (False).
    """
    (env, goal, obstacles) = initalize_environment(render)
    #env.reconfigure_camera(5, 0, 270.1, [0, 0, 0])
    planner = set_planner(goal, env.dt)

    action = np.array([0.0, 0.0, 0.0])
    observation, *_ = env.step(action)

    for _ in range(n_steps):
        t0 = time.perf_counter()
        # Calculate action with the fabric planner, slice the states to drop Z-axis [3] information.
        robot = observation['robot_0']
        full_sensor = robot['FullSensor']
        goals = full_sensor['goals']
        goal_pos = lambda i: goals[i]['position'][0:2]
        goal_weight = lambda i: goals[i]['weight']
        obst = full_sensor['obstacles']
        obstacle_size = lambda i: obst[i]['size'][0]
        obstacle_pos = lambda i: obst[i]['position']

        arguments = dict(
            constraint_0=constraint_0,
            q=robot["joint_state"]["position"],
            qdot=robot["joint_state"]["velocity"],
            x_goal_0=goal_pos(0),
            weight_goal_0=goal_weight(0),
            radius_obst_0=obstacle_size(0),
            radius_obst_1=obstacle_size(1),
            radius_obst_2=obstacle_size(2),
            radius_obst_3=obstacle_size(3),
            x_obst_0=obstacle_pos(0),
            x_obst_1=obstacle_pos(1),
            x_obst_2=obstacle_pos(2),
            x_obst_3=obstacle_pos(3),
            radius_body_base_link=np.array([0.35]),
        )
        #for i in range(NUMBER_OF_CONSTRAINTS):
        #    arguments[f"constraint_{i}"] = ob_robot["FreeSpaceDecompSensor"][f"constraint_{i}"]
        action = planner.compute_action(**arguments)
        ob, *_, = env.step(action)
        t1 = time.perf_counter()
        #print(t1-t0)
    return {}

if __name__ == "__main__":
    res = run_point_robot(n_steps=10000)
