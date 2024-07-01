import logging
import os
from copy import deepcopy
import gymnasium as gym
import numpy as np

from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
from robotmodels.utils.robotmodel import RobotModel, LocalRobotModel

from urdfenvs.sensors.full_sensor import FullSensor
from urdfenvs.generic_mujoco.generic_mujoco_env import GenericMujocoEnv

from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.goals.goal_composition import GoalComposition

from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner
from generic_diffdrive_mujoco_robot import GenericDiffDriveMujocoRobot


logging.basicConfig(level=logging.ERROR)

root_link = 'torso_lift_link'
arm = 'left'
child_link = f'gripper_{arm}_grasping_frame'

ROBOTTYPE = 'tiago'
ROBOTMODEL = 'tiago'

tiago_model = RobotModel(ROBOTMODEL)
urdf_file = tiago_model.get_urdf_path()


def initalize_environment(render: bool = True):
    """
    Initializes the simulation environment.

    Adds obstacles and goal visualizaion to the environment based and
    steps the simulation once.
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
            castor_wheels=[
                "caster_front_right_2_joint",
                "caster_front_left_2_joint",
                "caster_back_right_2_joint",
                "caster_back_left_2_joint",
            ],
            not_actuated_joints=[
                "suspension_right_joint",
                "suspension_left_joint",
            ],
            wheel_radius=0.1,
            wheel_distance=0.4044,
            spawn_offset=np.array([-0.1764081, 0.0, 0.1]),
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
        "geometry": {"position": [0.8, 0.1, 0.7], "radius": 0.1},
    }
    obst = SphereObstacle(name="staticObst", content_dict=static_obst_dict)
    desired_position = deepcopy(obst.position().tolist())
    desired_position[2] -= 0.99
    desired_position[0] += 0.242
    obst_transformed_dict = deepcopy(static_obst_dict)
    obst_transformed_dict['geometry']['position'] = desired_position
    obst_transformed = SphereObstacle(name="obst", content_dict=obst_transformed_dict)
    obstacles = [obst_transformed]

    # GOALS
    high_goal_limits = {
        'left': [0.4, 1.0, 1.3],
        'right': [0.4, -1.0, 1.3]
    }
    low_goal_limits = {
        'left': [-0.4, 0.5, 0.4],
        'right': [-0.4, -0.5, 0.4]
    }
    goal_dict = {
        "subgoal0": {
            "weight": 1.0,
            "is_primary_goal": True,
            "indices": [0, 1, 2],
            "parent_link": root_link,
            "child_link": child_link,
            "desired_position": [0.3, -0.05, 0.8],
            "high": high_goal_limits[arm],
            "low": low_goal_limits[arm],
            "epsilon": 0.05,
            "type": "staticSubGoal",
        },
    }
    # Transform goal and obst into world frame
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    desired_position = deepcopy(goal.primary_goal().position().tolist())
    desired_position[2] -= 0.99
    desired_position[0] += 0.242
    goal_transformed_dict = deepcopy(goal_dict)
    goal_transformed_dict['subgoal0']['desired_position'] = desired_position
    goal_transformed = GoalComposition(name="goal", content_dict=goal_transformed_dict)
    env = GenericMujocoEnv(robots=robots, obstacles=obstacles, goals=goal_transformed.sub_goals()[0:1],
                           sensors=[full_sensor],
                           render=render)

    pos0 = np.zeros(20)
    pos0[0] = 0.0
    # base
    # pos0[0:3] = np.array([0.0, 1.0, -1.0])
    # torso
    pos0[3] = 0.1
    env.reset(pos=pos0)
    return env, goal


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
    """

    degrees_of_freedom: int = 7
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    with open(absolute_path + "/tiago_dual_fk.urdf", "r", encoding="utf-8") as file:
        urdf = file.read()
    forward_kinematics = GenericURDFFk(
        urdf,
        root_link='torso_lift_link',
        end_links=[f'gripper_{arm}_link'],
    )
    planner = ParameterizedFabricPlanner(
        degrees_of_freedom,
        forward_kinematics,
    )
    q = planner.variables.position_variable()
    collision_links = [f'arm_{arm}_{i}_link' for i in [3, 4, 5, 6, 7]]
    # The planner hides all the logic behind the function set_components.
    planner.set_components(
        collision_links=collision_links,
        goal=goal,
        number_obstacles=1,
    )
    planner.concretize(time_step=dt)
    return planner


def run_tiago_example(n_steps=5000, render=True):
    (env, goal) = initalize_environment(render)
    action = np.zeros(23)
    ob, *_ = env.step(action)
    planner = set_planner(goal=goal, dt=env.dt)

    # Initializing actions and joint states
    augmented_action = np.zeros(23)
    q = np.zeros(7)
    qdot = np.zeros(7)
    body_arguments = {}
    for i in [3, 4, 5, 6, 7]:
        body_arguments[f'radius_body_arm_{arm}_{i}_link'] =np.array([0.1])

    for _ in range(n_steps):
        ob_robot = ob['robot_0']
        if arm == 'left':
            q = ob_robot['joint_state']['position'][6:13]
            qdot = ob_robot['joint_state']['velocity'][6:13]
        elif arm == 'right':
            q = ob_robot['joint_state']['position'][13:20]
            qdot = ob_robot['joint_state']['velocity'][13:20]
        action = planner.compute_action(
            q=q,
            qdot=qdot,
            x_goal_0=ob_robot['FullSensor']['goals'][3]['position'],
            weight_goal_0=ob_robot['FullSensor']['goals'][3]['weight'],
            x_obst_0=ob_robot['FullSensor']['obstacles'][2]['position'],
            radius_obst_0=ob_robot['FullSensor']['obstacles'][2]['size'],
            **body_arguments
        )
        if arm == 'left':
            augmented_action[5:12] = action
        elif arm == 'right':
            augmented_action[12:19] = action
        ob, *_ = env.step(augmented_action)
    env.close()
    return {}


if __name__ == "__main__":
    res = run_tiago_example(n_steps=5000)
