import numpy as np
from typing_extensions import List
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.generic_mujoco.generic_mujoco_robot import GenericMujocoRobot


class GenericDiffDriveMujocoRobot(GenericMujocoRobot, GenericDiffDriveRobot):
    def __init__(self, xml_file: str, urdf_file: str, mode: str,
                 actuated_wheels: List[str],
                 castor_wheels: List[str],
                 wheel_radius: float,
                 wheel_distance: float,
                 spawn_offset: np.ndarray = np.array([0.0, 0.0, 0.15]),
                 spawn_rotation: float = 0.0,
                 facing_direction: str = 'x',
                 not_actuated_joints: List[str] = None):
        GenericMujocoRobot.__init__(self, xml_file, mode)
        GenericDiffDriveRobot.__init__(self, urdf=urdf_file, mode=mode, actuated_wheels=actuated_wheels,
                                       castor_wheels=castor_wheels,
                                       wheel_radius=wheel_radius,
                                       wheel_distance=wheel_distance,
                                       spawn_offset=spawn_offset,
                                       spawn_rotation=spawn_rotation,
                                       facing_direction=facing_direction,
                                       not_actuated_joints=not_actuated_joints)
    pass