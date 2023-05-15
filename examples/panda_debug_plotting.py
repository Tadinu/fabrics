import pdb
import gym
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor
import os

from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.obstacles.sphere_obstacle import SphereObstacle

import numpy as np
import os
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# TODO hardcoding the indices for subgoal_1 is undesired


def initalize_environment(render=True):
    """
    Initializes the simulation environment.

    Adds obstacles and goal visualizaion to the environment based and
    steps the simulation once.
    """
    robots = [
        GenericUrdfReacher(urdf="panda.urdf", mode="acc"),
    ]
    env: UrdfEnv  = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    full_sensor = FullSensor(goal_mask=["position"], obstacle_mask=["position", "radius"])
    # Definition of the obstacle.
    static_obst_dict = {
        "type": "sphere",
        "geometry": {"position": [0.5, -0.3, 0.3], "radius": 0.1},
    }
    obst1 = SphereObstacle(name="staticObst", content_dict=static_obst_dict)
    static_obst_dict = {
        "type": "sphere",
        "geometry": {"position": [-0.7, 0.0, 0.5], "radius": 0.1},
    }
    obst2 = SphereObstacle(name="staticObst", content_dict=static_obst_dict)
    # Definition of the goal.
    goal_dict = {
        "subgoal0": {
            "weight": 1.0,
            "is_primary_goal": True,
            "indices": [0, 1, 2],
            "parent_link": "panda_link0",
            "child_link": "panda_hand",
            "desired_position": [0.1, -0.6, 0.4],
            "epsilon": 0.05,
            "type": "staticSubGoal",
        },
        "subgoal1": {
            "weight": 5.0,
            "is_primary_goal": False,
            "indices": [0, 1, 2],
            "parent_link": "panda_link7",
            "child_link": "panda_hand",
            "desired_position": [0.1, 0.0, 0.0],
            "epsilon": 0.05,
            "type": "staticSubGoal",
        }
    }
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    obstacles = (obst1, obst2)
    env.reset()
    env.add_sensor(full_sensor, [0])
    for obst in obstacles:
        env.add_obstacle(obst)
    for sub_goal in goal.sub_goals():
        env.add_goal(sub_goal)
    return (env, goal)


def set_planner(goal: GoalComposition, degrees_of_freedom: int = 7):
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

    robot_type = 'panda'

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
    with open(absolute_path + "/panda_for_fk.urdf", "r") as file:
        urdf = file.read()
    planner = ParameterizedFabricPlanner(
        degrees_of_freedom,
        'panda',
        urdf=urdf,
        root_link='panda_link0',
        end_link='panda_link9',
    )
    q = planner.variables.position_variable()
    collision_links = ['panda_link9', 'panda_link3', 'panda_link4']
    panda_limits = [
            [-2.8973, 2.8973],
            [-1.7628, 1.7628],
            [-2.8973, 2.8973],
            [-3.0718, -0.0698],
            [-2.8973, 2.8973],
            [-0.0175, 3.7525],
            [-2.8973, 2.8973]
        ]
    # The planner hides all the logic behind the function set_components.
    planner.set_components(
        collision_links=collision_links,
        goal=goal,
        number_obstacles=1,
        limits=panda_limits,
    )
    planner.concretize()
    return planner


def run_panda_example(n_steps=5000, render=True):
    """ This example illustrates how to analyze the specified geometries knowing the current joint positions and velocities.
    There are two options to get the geometries:
     -  Option 1: pull the geometry to configuration space, concretize. To later input (q, qdot) for evaluate()
     -  Option 2: get the unpulled geometry: To later input (x and xdot) in evaluate()
    h_num is just the inverted value of x_ddot, since xddot + h = 0.
    """

    nr_obst = 2
    r_body_panda_links = [np.array([0.02]), np.array([0.02]), np.array([0.02])]

    (env, goal) = initalize_environment(render)
    planner = set_planner(goal)
    action = np.zeros(7)
    ob, *_ = env.step(action)

    # specify the list of geometry leaf names that you would like to observe:
    all_leaf_names = list(planner.leaves.keys())
    all_leaf_names = all_leaf_names[0:-2]   #IF THIS IS UNCOMMENTED THAT LEADS TO THE BUG: "Initialization failed since variables [weight_goal_0] are free"
    leaves = planner.get_leaves(leaf_names=all_leaf_names)

    # dictionary for plotting
    q_ddot_geometries = {} #joint space
    x_ddot_geometries = {} #task space
    for leaf_name in all_leaf_names:
        q_ddot_geometries[leaf_name] = []
        x_ddot_geometries[leaf_name] = []

    # set goal
    planner.set_goal_component(goal=goal)

    pulled_geometry_list = []
    unpulled_geometry_list = []
    mapping_list = []
    for leave in leaves:
        # Option 1: pull the geometry to configuration space, concretize. To later input (q, qdot) for evaluate():
        pulled_geometry = leave._geo.pull(leave._forward_map)
        pulled_geometry.concretize()
        pulled_geometry_list.append(pulled_geometry)

        # Option 2: get the unpulled geometry: To later input (x and xdot) in evaluate()
        mapping = leave.map()
        mapping.concretize()
        mapping_list.append(mapping)
        unpulled_geometry=leave._geo  #weight_goal_0=goal.sub_goals()[0].weight() must be inputted somewhere.
        unpulled_geometry.concretize()
        unpulled_geometry_list.append(unpulled_geometry)

    # list of possible input keys: To Do for Saray
    x_obsts = [ob['robot_0']['FullSensor']['obstacles'][i][0] for i in range(nr_obst)]
    r_obsts = [ob['robot_0']['FullSensor']['obstacles'][i][1] for i in range(nr_obst)]

    actions = []
    for _ in range(n_steps):
        ob_robot = ob['robot_0']

        # define variables
        q_num = ob_robot["joint_state"]["position"]
        qdot_num = ob_robot["joint_state"]["velocity"]

        for i in range(len(pulled_geometry_list)):
            #Option 1:
            [xddot_opt1, h_opt1] = pulled_geometry_list[i].evaluate(q=q_num, qdot=qdot_num,
                                                                    radius_body_panda_link3=r_body_panda_links[0],
                                                                    radius_body_panda_link4=r_body_panda_links[1],
                                                                    radius_body_panda_link9=r_body_panda_links[2],
                                                                    radius_obsts=r_obsts,
                                                                    x_obsts=x_obsts,
                                                                    x_goal_0=ob_robot['FullSensor']['goals'][0][0],
                                                                    weight_goal_0=goal.sub_goals()[0].weight(),
                                                                    x_goal_1=ob_robot['FullSensor']['goals'][1][0],
                                                                    weight_goal_1=goal.sub_goals()[1].weight(),
                                                                    )
            leaf_name = all_leaf_names[i]
            q_ddot_geometries[leaf_name].append(xddot_opt1)

            #Option 2:
            [x_num, J, Jdot] = mapping_list[i].forward(q=q_num, qdot=qdot_num,
                                                       radius_body_panda_link3=r_body_panda_links[0],
                                                       radius_body_panda_link4=r_body_panda_links[1],
                                                       radius_body_panda_link9=r_body_panda_links[2],
                                                       radius_obsts=r_obsts,
                                                       x_obsts=x_obsts,
                                                       x_goal_0=ob_robot['FullSensor']['goals'][0][0],
                                                       weight_goal_0=goal.sub_goals()[0].weight(),
                                                       x_goal_1=ob_robot['FullSensor']['goals'][1][0],
                                                       weight_goal_1=goal.sub_goals()[1].weight(),
                                                       )
            xdot_num = J @ qdot_num
            # pos_argument = unpulled_geometry_list[i]._vars.position_variable()  #WOULD LIKE TO USE THESE AS VARIABLE NAMES, BUT DON'T KNOW HOW!
            # vel_argument = unpulled_geometry_list[i]._vars.velocity_variable()
            [x_ddot_opt2, h_opt2] = unpulled_geometry_list[i].evaluate(x_obst_0_panda_link9_leaf=x_num,
                                                                       x_obst_0_panda_link3_leaf=x_num,
                                                                       x_obst_0_panda_link4_leaf=x_num,  #QUESTION, WHY NO X_OBST_1??
                                                                       x_limit_joint_0_0_leaf=x_num,
                                                                       x_limit_joint_0_1_leaf=x_num,
                                                                       x_limit_joint_1_0_leaf=x_num,
                                                                       x_limit_joint_1_1_leaf=x_num,
                                                                       x_limit_joint_2_0_leaf=x_num,
                                                                       x_limit_joint_2_1_leaf=x_num,
                                                                       x_limit_joint_3_0_leaf=x_num,
                                                                       x_limit_joint_3_1_leaf=x_num,
                                                                       x_limit_joint_4_0_leaf=x_num,
                                                                       x_limit_joint_4_1_leaf=x_num,
                                                                       x_limit_joint_5_0_leaf=x_num,
                                                                       x_limit_joint_5_1_leaf=x_num,
                                                                       x_limit_joint_6_0_leaf=x_num,
                                                                       x_limit_joint_6_1_leaf=x_num,
                                                                       xdot_obst_0_panda_link9_leaf=xdot_num,
                                                                       xdot_obst_0_panda_link3_leaf=xdot_num,
                                                                       xdot_obst_0_panda_link4_leaf=xdot_num,
                                                                       xdot_limit_joint_0_0_leaf=xdot_num,
                                                                       xdot_limit_joint_0_1_leaf=xdot_num,
                                                                       xdot_limit_joint_1_0_leaf=xdot_num,
                                                                       xdot_limit_joint_1_1_leaf=xdot_num,
                                                                       xdot_limit_joint_2_0_leaf=xdot_num,
                                                                       xdot_limit_joint_2_1_leaf=xdot_num,
                                                                       xdot_limit_joint_3_0_leaf=xdot_num,
                                                                       xdot_limit_joint_3_1_leaf=xdot_num,
                                                                       xdot_limit_joint_4_0_leaf=xdot_num,
                                                                       xdot_limit_joint_4_1_leaf=xdot_num,
                                                                       xdot_limit_joint_5_0_leaf=xdot_num,
                                                                       xdot_limit_joint_5_1_leaf=xdot_num,
                                                                       xdot_limit_joint_6_0_leaf=xdot_num,
                                                                       xdot_limit_joint_6_1_leaf=xdot_num,
                                                                       )
            x_ddot_geometries[leaf_name].append(x_ddot_opt2)

        action = planner.compute_action(
            q=q_num,
            qdot=qdot_num,
            x_goal_0=ob_robot['FullSensor']['goals'][0][0],
            weight_goal_0=goal.sub_goals()[0].weight(),
            x_goal_1=ob_robot['FullSensor']['goals'][1][0],
            weight_goal_1=goal.sub_goals()[1].weight(),
            x_obsts=x_obsts,
            radius_obsts=r_obsts,
            radius_body_panda_link3=r_body_panda_links[0],
            radius_body_panda_link4=r_body_panda_links[1],
            radius_body_panda_link9=r_body_panda_links[2],
        )
        ob, *_ = env.step(action)
        actions.append(action)
    return q_ddot_geometries, x_ddot_geometries, actions

def plot_different_geometries(n_steps, qddot_geometries, xddot_geometries, actions):
    time_x = np.linspace(0, n_steps * 0.01, n_steps)
    keys_geometries = list(xddot_geometries.keys())

    #------ Plot geometries in configuration space ------#
    fig = plt.figure(figsize=(20, 15))
    plt.clf()
    gs = GridSpec(1, 4, figure=fig)
    colorscheme = ['r', 'b', 'm', 'g', 'k', 'y', 'deeppink', 'coral', 'cornflowerblue', 'cyan', 'darkred', 'gray', 'violet', 'orange']

    # Plot obstacle geometries
    obst_keys = [key for key in keys_geometries if key.startswith("obst")]
    fig.add_subplot(gs[0, 0])
    for i, obst_key in enumerate(obst_keys):
        obst_plt = plt.plot(time_x, qddot_geometries[obst_key], colorscheme[i])
    plt.xlabel('time [s]')
    plt.ylabel('$\ddot{q}$')
    plt.title("Obstacle geometries")
    plt.grid()

    # Plot limit geometries
    limit_keys = [key for key in keys_geometries if key.startswith("limit")]
    fig.add_subplot(gs[0, 1])
    for i, limit_key in enumerate(limit_keys):
        limit_plt = plt.plot(time_x, qddot_geometries[limit_key], colorscheme[i])
    plt.xlabel('time [s]')
    plt.ylabel('$\ddot{q}$')
    plt.title("Limit geometries")
    plt.grid()

    # Plot goal geometries
    goal_keys = [key for key in keys_geometries if key.startswith("limit")]  #should  be goal!!
    fig.add_subplot(gs[0, 2])
    # for i, goal_key in enumerate(goal_keys):
    #     goal_plt = plt.plot(time_x, qddot_geometries[goal_key], colorscheme[i])
    plt.xlabel('time [s]')
    plt.ylabel('$\ddot{q}$')
    plt.title("Goal geometries [To Do!]")
    plt.grid()

    # Plot total action
    fig.add_subplot(gs[0, 3])
    act_plt = plt.plot(time_x, actions)
    plt.xlabel('time [s]')
    plt.ylabel('$\ddot{x}$')
    plt.title("Total action")
    plt.grid()

    fig.suptitle("Geometries in configuration space", fontsize=50)
    plt.show()

    # -------- Plot geometries in task space --------- #
    fig = plt.figure(figsize=(20, 15))
    plt.clf()
    gs = GridSpec(1, 3, figure=fig)
    colorscheme = ['r', 'b', 'm', 'g', 'k', 'y', 'deeppink', 'coral', 'cornflowerblue', 'cyan', 'darkred', 'gray', 'violet', 'orange']

    # Plot obstacle geometries
    obst_keys = [key for key in keys_geometries if key.startswith("obst")]
    fig.add_subplot(gs[0, 0])
    for i, obst_key in enumerate(obst_keys):
        obst_plt = plt.plot(time_x, xddot_geometries[obst_key], colorscheme[i])
    plt.xlabel('time [s]')
    plt.ylabel('$\ddot{q}$')
    plt.title("Obstacle geometries")
    plt.grid()

    # Plot limit geometries
    limit_keys = [key for key in keys_geometries if key.startswith("limit")]
    fig.add_subplot(gs[0, 1])
    for i, limit_key in enumerate(limit_keys):
        limit_plt = plt.plot(time_x, xddot_geometries[limit_key], colorscheme[i])
    plt.xlabel('time [s]')
    plt.ylabel('$\ddot{q}$')
    plt.title("Limit geometries")
    plt.grid()

    # Plot goal geometries
    goal_keys = [key for key in keys_geometries if key.startswith("limit")]  #should  be goal!!
    fig.add_subplot(gs[0, 2])
    # for i, goal_key in enumerate(goal_keys):
    #     goal_plt = plt.plot(time_x, xddot_geometries[goal_key], colorscheme[i])
    plt.xlabel('time [s]')
    plt.ylabel('$\ddot{q}$')
    plt.title("Goal geometries [To Do!]")
    plt.grid()

    fig.suptitle("Geometries in task space", fontsize=50)
    plt.show()

if __name__ == "__main__":
    N_steps = 1000
    [q_ddot_geometries, x_ddot_geometries, actions] = run_panda_example(n_steps=N_steps)
    plot_different_geometries(n_steps=N_steps, qddot_geometries=q_ddot_geometries, xddot_geometries=x_ddot_geometries, actions=actions)
