from __future__ import annotations

from typing import Any, Dict, List, Tuple

import chex
import jax.random
import mujoco
from jax import numpy as jnp
from jax._src.scipy.spatial.transform import Rotation
from moojoco.moojoco.environment.mjx_env import MJXEnvState, MJXObservable
from .fab_mjx_base_env import FabMjxBaseEnv, FabEnvBaseConfig


class FabMjxEnv(FabMjxBaseEnv):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        xml_path: str,
        configuration: FabEnvBaseConfig,
    ) -> None:
        self._robot_base_body_name = "link0"
        super().__init__(xml_path=xml_path, configuration=configuration)
        self._mjx_model, self._mjx_data = self._initialize_mjx_model_and_data()

    def _create_observables(self, joint_subname: str) -> List[MJXObservable]:
        base_observables = self.get_observation(
            mj_model=self.frozen_mj_model, mj_data=self.frozen_mj_data,
            joint_subname=joint_subname
        )
        return base_observables

    def get_observation(self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData, joint_subname: str)\
            -> List[MJXObservable]:
        joints = [
            mj_model.joint(joint_id)
            for joint_id in range(mj_model.njnt)
            if joint_subname in mj_model.joint(joint_id).name
        ]
        joints_qpos_adr = jnp.array([joint.qposadr[0] for joint in joints])
        joint_dof_adr = jnp.array([joint.dofadr[0] for joint in joints])
        joint_range = jnp.array([joint.range for joint in joints]).T

        joint_position_observable = MJXObservable(
            name="joint_position",
            low=joint_range[0],
            high=joint_range[1],
            retriever=lambda state: state.mjx_data.qpos[joints_qpos_adr],
        )

        joint_velocity_observable = MJXObservable(
            name="joint_velocity",
            low=-jnp.inf * jnp.ones(len(joints)),
            high=jnp.inf * jnp.ones(len(joints)),
            retriever=lambda state: state.mjx_data.qvel[joint_dof_adr],
        )

        joint_actuator_force_observable = MJXObservable(
            name="joint_actuator_force",
            low=-jnp.inf * jnp.ones(len(joint_dof_adr)),
            high=jnp.inf * jnp.ones(len(joint_dof_adr)),
            retriever=lambda state: state.mjx_data.qfrc_actuator[joint_dof_adr].flatten(),
        )

        low_actuator_force_limit = jnp.array(
            [limits[0] for limits in mj_model.actuator_forcerange]
        )
        high_actuator_force_limit = jnp.array(
            [limits[1] for limits in mj_model.actuator_forcerange]
        )

        def calculate_actuator_force(state: MJXEnvState) -> jnp.ndarray:
            l = state.mjx_data.actuator_length
            v = state.mjx_data.actuator_velocity
            gain = state.mjx_model.actuator_gainprm[:, 0]
            b0 = state.mjx_model.actuator_biasprm[:, 0]
            b1 = state.mjx_model.actuator_biasprm[:, 1]
            b2 = state.mjx_model.actuator_biasprm[:, 2]
            ctrl = state.mjx_data.ctrl

            return jnp.clip(
                gain * ctrl + b0 + b1 * l + b2 * v,
                low_actuator_force_limit,
                high_actuator_force_limit,
            )

        actuator_force_observable = MJXObservable(
            name="actuator_force",
            low=low_actuator_force_limit,
            high=high_actuator_force_limit,
            retriever=calculate_actuator_force,
        )

        body_geom_ids = jnp.array(
            [
                geom_id
                for geom_id in range(mj_model.ngeom)
                if "link" in mj_model.geom(geom_id).name
            ]
        )

        external_contact_geom_ids = jnp.array(
            [
                mj_model.geom("floor").id
            ]
        )

        def get_body_contacts(state: MJXEnvState) -> jnp.ndarray:
            contact_data = state.mjx_data.contact
            contacts = contact_data.dist <= 0

            def solve_contact(geom_id: int) -> jnp.ndarray:
                return (
                        jnp.sum(contacts * jnp.any(geom_id == contact_data.geom, axis=-1)) > 0
                ).astype(int)

            return jax.vmap(solve_contact)(body_geom_ids)

        touch_observable = MJXObservable(
            name="body_contact",
            low=jnp.zeros(len(body_geom_ids)),
            high=jnp.ones(len(body_geom_ids)),
            retriever=get_body_contacts,
        )

        robot_base_id = mj_model.body(self._robot_base_body_name).id
        position_observable = MJXObservable(
            name="base_position",
            low=-jnp.inf * jnp.ones(3),
            high=jnp.inf * jnp.ones(3),
            retriever=lambda state: state.mjx_data.xpos[robot_base_id],
        )

        # framequat
        rotation_observable = MJXObservable(
            name="base_rotation",
            low=-jnp.pi * jnp.ones(3),
            high=jnp.pi * jnp.ones(3),
            retriever=lambda state: Rotation.from_quat(
                quat=state.mjx_data.xquat[robot_base_id]
            ).as_euler(seq="xyz"),
        )

        # framelinvel
        basejoint_adr = mj_model.joint(
            "joint1"
        ).dofadr[0]
        linvel_observable = MJXObservable(
            name="base_linear_velocity",
            low=-jnp.inf * jnp.ones(3),
            high=jnp.inf * jnp.ones(3),
            retriever=lambda state: state.mjx_data.qvel[
                                    basejoint_adr: basejoint_adr + 3
                                    ],
        )
        # frameangvel
        angvel_observable = MJXObservable(
            name="base_angular_velocity",
            low=-jnp.inf * jnp.ones(3),
            high=jnp.inf * jnp.ones(3),
            retriever=lambda state: state.mjx_data.qvel[
                                    basejoint_adr + 3: basejoint_adr + 6
                                    ],
        )

        return [
            joint_position_observable,
            joint_velocity_observable,
            joint_actuator_force_observable,
            actuator_force_observable,
            touch_observable,
            position_observable,
            rotation_observable,
            linvel_observable,
            angvel_observable,
        ]

    @staticmethod
    def _get_time(state: MJXEnvState) -> MJXEnvState:
        return state.mjx_data.time

    @staticmethod
    def _get_xy_distance_from_origin(state: MJXEnvState) -> float:
        disk_body_id = state.mj_model.body("BrittleStarMorphology/central_disk").id
        xy_disk_position = state.mjx_data.xpos[disk_body_id][:2]
        return jnp.linalg.norm(xy_disk_position)

    def _get_mj_models_and_datas_to_render(
        self, state: MJXEnvState
    ) -> Tuple[List[mujoco.MjModel], List[mujoco.MjData]]:
        mj_models, mj_datas = super()._get_mj_models_and_datas_to_render(state=state)
        if self.environment_configuration.color_contacts:
            self._color_body_contacts(
                mj_models=mj_models, contact_bools=state.observations["body_contact"]
            )
        return mj_models, mj_datas

    def _get_target_position(
        self, rng: jnp.ndarray, target_position: jnp.ndarray
    ) -> jnp.ndarray:
        def return_given_target_position() -> jnp.ndarray:
            return target_position

        def return_random_target_position() -> jnp.ndarray:
            angle = jax.random.uniform(key=rng, shape=(), minval=0, maxval=jnp.pi * 2)
            radius = self.environment_configuration.target_distance
            random_position = jnp.array(
                [radius * jnp.cos(angle), radius * jnp.sin(angle), 0.05]
            )
            return random_position

        return jax.lax.cond(
            jnp.any(jnp.isnan(target_position)),
            return_random_target_position,
            return_given_target_position,
        )

    def reset(self, rng: chex.PRNGKey, *args, **kwargs) -> MJXEnvState:
        (mj_model, mj_data), (mjx_model, mjx_data) = self._prepare_reset()

        rng, qpos_rng, qvel_rng = jax.random.split(key=rng, num=3)

        robot_base_body_id = mj_model.body(self._robot_base_body_name).id

        # Reset robot base position
        robot_base_pos = jnp.array([0.0, 0.0, 0.0])
        mjx_model = mjx_model.replace(
            body_pos=mjx_model.body_pos.at[robot_base_body_id].set(robot_base_pos)
        )

        # Add noise to initial qpos and qvel of segment joints
        qpos = jnp.copy(mjx_model.qpos0)
        qvel = jnp.zeros(mjx_model.nv)

        joint_qpos_adrs = self._get_joints_qpos_adrs(mj_model=mj_model, joint_subname="joint")
        joint_qvel_adrs = self._get_joints_qvel_adrs(mj_model=mj_model, joint_subname="joint")
        num_joints = len(joint_qpos_adrs)

        qpos = qpos.at[joint_qpos_adrs].set(
            qpos[joint_qpos_adrs]
            + jax.random.uniform(
                key=qpos_rng,
                shape=(num_joints,),
                minval=-self.environment_configuration.joint_randomization_noise_scale,
                maxval=self.environment_configuration.joint_randomization_noise_scale,
            )
        )
        qvel = qvel.at[joint_qvel_adrs].set(
            jax.random.uniform(
                key=qvel_rng,
                shape=(num_joints,),
                minval=-self.environment_configuration.joint_randomization_noise_scale,
                maxval=self.environment_configuration.joint_randomization_noise_scale,
            )
        )

        mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)

        state = self._finish_reset(
            models_and_datas=((mj_model, mj_data), (mjx_model, mjx_data)), rng=rng
        )
        return state
