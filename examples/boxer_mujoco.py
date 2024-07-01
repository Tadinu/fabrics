from copy import deepcopy
import os
import shutil
import logging
import gymnasium as gym
import numpy as np
from typing_extensions import List

from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
from robotmodels.utils.robotmodel import RobotModel, LocalRobotModel

from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.generic_mujoco.generic_mujoco_env import GenericMujocoEnv
from urdfenvs.generic_mujoco.generic_mujoco_robot import GenericMujocoRobot
from urdfenvs.sensors.full_sensor import FullSensor

from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.goals.goal_composition import GoalComposition

from fabrics.planner.non_holonomic_parameterized_planner import NonHolonomicParameterizedFabricPlanner
from generic_diffdrive_mujoco_robot import GenericDiffDriveMujocoRobot

logging.basicConfig(level=logging.INFO)
"""
Fabrics example for the boxer robot.
"""

ROBOTTYPE = 'boxer'
ROBOTMODEL = 'boxer'

boxer_model = RobotModel(ROBOTMODEL)
urdf_file = boxer_model.get_urdf_path()


def initalize_environment(render: bool = True):
    """
    Initializes the simulation environment.

    Adds an obstacle and goal visualizaion to the environment and
    steps the simulation once.
    
    Params
    ----------
    render
        Boolean toggle to set rendering on (True) or off (False).
    """
    #if os.path.exists(ROBOTTYPE):
        #shutil.rmtree(ROBOTTYPE)
    #robot_model = RobotModel(ROBOTTYPE, ROBOTMODEL)
    #robot_model.copy_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), ROBOTTYPE))
    #del robot_model

    robot_model = LocalRobotModel(ROBOTTYPE, ROBOTMODEL)

    xml_file = robot_model.get_xml_path()
    robots = [
        GenericDiffDriveMujocoRobot(
            xml_file=xml_file,
            urdf_file=urdf_file,
            mode="acc",
            actuated_wheels=["wheel_right_joint", "wheel_left_joint"],
            castor_wheels=["rotacastor_right_joint", "rotacastor_left_joint"],
            wheel_radius=0.08,
            wheel_distance=0.494,
            spawn_rotation=0.0,
            facing_direction='-y',
        ),
    ]

    full_sensor = FullSensor(
        goal_mask=["position", "weight"],
        obstacle_mask=['position', 'size'],
        variance=0.0,
        physics_engine_name="mujoco"
    )

    # OBSTACLES
    static_obst_dict = {
        "type": "sphere",
        "geometry": {"position": [-0.3, -4.0, 0.0], "radius": 1.0},
    }
    obst1 = SphereObstacle(name="staticObst1", content_dict=static_obst_dict)
    obstacles = [obst1] # Add additional obstacles here.

    # GOALS
    goal_dict = {
        "subgoal0": {
            "weight": 1.0,
            "is_primary_goal": True,
            "indices": [0, 1],
            "parent_link" : 'origin',
            "child_link" : 'ee_link',
            "desired_position": [0.0, -6.0],
            "epsilon" : 0.1,
            "type": "staticSubGoal"
        }
    }
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    env = GenericMujocoEnv(robots=robots, obstacles=obstacles, goals=goal.sub_goals()[0:1],
                           sensors=[full_sensor],
                           render=render)
    pos0 = np.array([-0.0, 0.0, 0.0, 0.0])
    vel0 = np.array([1.0, 0.0, 0.0, 0.0])
    env.reset(pos=pos0, vel=vel0)
    return env, goal


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
    degrees_of_freedom = 3
    # Optional reconfiguration of the planner with collision_geometry/finsler, remove for defaults.
    collision_geometry = "-2.0 / (x ** 2) * xdot ** 2"
    collision_finsler = "1.0/(x**2) * (1 - ca.heaviside(xdot))* xdot**2"
    with open(urdf_file, "r", encoding="utf-8") as file:
        urdf = file.read()
    forward_kinematics = GenericURDFFk(
        urdf,
        root_link="base_link",
        end_links=["ee_link"],
        base_type="diffdrive",
    )
    planner = NonHolonomicParameterizedFabricPlanner(
        degrees_of_freedom,
        forward_kinematics,
        collision_geometry=collision_geometry,
        collision_finsler=collision_finsler,
        l_offset="0.1/ca.norm_2(xdot)",
    )
    collision_links = ["ee_link"]
    self_collision_links = {}
    # The planner hides all the logic behind the function set_components.
    planner.set_components(
        collision_links=collision_links,
        goal=goal,
        number_obstacles=1,
    )
    planner.concretize(time_step=dt)
    return planner


def run_boxer_example(n_steps=10000, render=True):
    """
    Set the gym environment, the planner and run point robot example.
    
    Params
    ----------
    n_steps
        Total number of simulation steps.
    render
        Boolean toggle to set rendering on (True) or off (False).
    """
    (env, goal) = initalize_environment(render)
    planner = set_planner(goal, env.dt)
    action = np.zeros(4)
    ob, *_ = env.step(action)
    #env.reconfigure_camera(3.000001907348633, -90.00001525878906, -94.20011138916016, (0.15715950727462769, -2.938774585723877, -0.02000000700354576))

    for _ in range(n_steps):
        ob_robot = ob['robot_0']
        qudot = np.array([
            ob_robot['joint_state']['velocity'][0],
            ob_robot['joint_state']['velocity'][1],
            ob_robot['joint_state']['velocity'][2]
        ])
        arguments = dict(
            q=ob_robot["joint_state"]["position"][0:3],
            qdot=ob_robot["joint_state"]["velocity"][0:3],
            qudot=qudot[0:2],
            x_goal_0=ob_robot['FullSensor']['goals'][0]['position'][0:2],
            weight_goal_0=ob_robot['FullSensor']['goals'][0]['weight'],
            m_rot=0.2,
            m_base_x=1.5,
            m_base_y=1.5,
            m_arm=1.0,
            x_obst_0=ob_robot['FullSensor']['obstacles'][0]['position'],
            radius_obst_0=ob_robot['FullSensor']['obstacles'][0]['size'],
            radius_body_ee_link=0.5,
        )
        action = planner.compute_action(**arguments)
        action = np.append(action, 0.1)
        action = np.append(action, 0.1)
        ob, *_, = env.step(action)
    env.close()
    return {}


if __name__ == "__main__":
    res = run_boxer_example(n_steps=10000, render=True)



