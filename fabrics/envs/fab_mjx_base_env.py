from __future__ import annotations

import abc
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import chex
import gymnasium
import mujoco
import numpy as np
from flax import struct
from gymnasium.core import RenderFrame

import jax
import jax.numpy as jnp

from mujoco import mjx

import moojoco.moojoco.environment.mjx_spaces as mjx_spaces
from moojoco.moojoco.environment.renderer import MujocoRenderer
from moojoco.moojoco.environment.base import (BaseEnvironment, BaseEnvState, MuJoCoEnvironmentConfiguration,
                                              BaseObservable, SpaceType, BoxSpaceType, DictSpaceType, ModelType, DataType)

from moojoco.moojoco.environment.mjx_env import MJXEnvState, MJXObservable, mjx_get_model

rgba_green = np.array([214, 174, 114, 190]) / 255.0
rgba_red = np.array([183, 86, 89, 210]) / 255.0

rgba_sand = np.array([225, 191, 146, 255]) / 255.0

class FabEnvBaseConfig(MuJoCoEnvironmentConfiguration):
    def __init__(
        self,
        solver_iterations: int = 1,
        solver_ls_iterations: int = 5,
        *args,
        **kwargs,
    ):
        super().__init__(
            disable_eulerdamp=True,
            solver_iterations=solver_iterations,
            solver_ls_iterations=solver_ls_iterations,
            *args,
            **kwargs,
        )
        self.joint_randomization_noise_scale = 0.0
        self.color_contacts = True

# Ref: moojoco - BaseMuJoCoEnvironment
class FabMujocoBaseEnv(BaseEnvironment, abc.ABC):
    metadata = {"render_modes": []}

    def __init__(
        self,
        xml_path: str,
        configuration: MuJoCoEnvironmentConfiguration,
    ) -> None:
        super().__init__(configuration=configuration)
        self._xml_path = xml_path
        self._renderers: Dict[int, mujoco.Renderer] = dict()

        self._mj_model, self._mj_data = self._initialize_mj_model_and_data()

        assert (
            configuration.render_mode is None
            or self.environment_configuration.render_mode
            in self.metadata["render_modes"]
        ), (
            f"Unsupported render mode: '{self.environment_configuration.render_mode}'. Must be one "
            f"of {self.metadata['render_modes']}"
        )

        self._observables: Optional[List[BaseObservable]] = self._create_observables("joint")
        self._action_space: Optional[BoxSpaceType] = self._create_action_space()
        self._observation_space: Optional[DictSpaceType] = (
            self._create_observation_space()
        )

    def _color_body_contacts(
        self, mj_models: List[mujoco.MjModel], contact_bools: chex.Array
    ) -> None:
        for i, mj_model in enumerate(mj_models):
            if len(contact_bools.shape) > 1:
                contacts = contact_bools[i]
            else:
                contacts = contact_bools

            for body_geom_id, contact in zip(
                self._body_geom_ids, contacts
            ):
                if contact:
                    mj_model.geom(body_geom_id).rgba = rgba_red
                else:
                    mj_model.geom(body_geom_id).rgba = rgba_green

    @property
    def frozen_mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def frozen_mj_data(self) -> mujoco.MjData:
        return self._mj_data

    @property
    def actuators(self) -> List[str]:
        return [
            self.frozen_mj_model.actuator(i).name
            for i in range(self.frozen_mj_model.nu)
        ]

    @property
    def action_space(self) -> SpaceType:
        return self._action_space

    @property
    def observation_space(self) -> SpaceType:
        return self._observation_space

    def _initialize_mj_model_and_data(self) -> Tuple[mujoco.MjModel, mujoco.MjData]:
        mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        mj_data = mujoco.MjData(mj_model)

        mj_model.vis.global_.offheight = self.environment_configuration.render_size[0]
        mj_model.vis.global_.offwidth = self.environment_configuration.render_size[1]
        mj_model.opt.timestep = self.environment_configuration.physics_timestep
        if self.environment_configuration.solver_iterations:
            mj_model.opt.iterations = self.environment_configuration.solver_iterations
        if self.environment_configuration.solver_ls_iterations:
            mj_model.opt.ls_iterations = (
                self.environment_configuration.solver_ls_iterations
            )
        if self.environment_configuration.disable_eulerdamp:
            mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP

        return mj_model, mj_data

    def step(
        self, state: BaseEnvState, action: chex.Array, *args, **kwargs
    ) -> BaseEnvState:
        previous_state = state
        state = self._update_simulation(state=state, ctrl=action)

        state = self._update_observations(state=state)
        state = self._update_reward(state=state, previous_state=previous_state)
        state = self._update_terminated(state=state)
        state = self._update_truncated(state=state)
        state = self._update_info(state=state)

        return state

    def get_renderer(
        self,
        identifier: int,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        state: BaseEnvState,
    ) -> Union[MujocoRenderer, mujoco.Renderer]:
        if identifier not in self._renderers:
            if self.environment_configuration.render_mode == "human":
                renderer = MujocoRenderer(
                    model=model, data=data, default_cam_config=None
                )
            else:
                renderer = mujoco.Renderer(
                    model=model,
                    height=self.environment_configuration.render_size[0],
                    width=self.environment_configuration.render_size[1],
                )
            self._renderers[identifier] = renderer
        return self._renderers[identifier]

    def get_renderer_context(self, renderer: Union[MujocoRenderer, mujoco.Renderer]) -> mujoco.MjrContext:
        try:
            # noinspection PyProtectedMember
            context = renderer._get_viewer(
                render_mode=self.environment_configuration.render_mode
            ).con
        except AttributeError:
            # noinspection PyProtectedMember
            context = renderer._mjr_context
        return context

    @abc.abstractmethod
    def _get_mj_models_and_datas_to_render(
        self, state: BaseEnvState
    ) -> Tuple[List[mujoco.MjModel], List[mujoco.MjData]]:
        raise NotImplementedError

    def render(self, state: BaseEnvState) -> List[RenderFrame] | None:
        camera_ids = self.environment_configuration.camera_ids or [-1]

        mj_models, mj_datas = self._get_mj_models_and_datas_to_render(state=state)

        frames = []
        for i, (model, data) in enumerate(zip(mj_models, mj_datas)):
            mujoco.mj_forward(m=model, d=data)
            renderer = self.get_renderer(
                identifier=i, model=model, data=data, state=state
            )

            if self.environment_configuration.render_mode == "human":
                viewer = renderer._get_viewer("human")
                viewer.model = model
                viewer.data = data

                return renderer.render(render_mode="human", camera_id=camera_ids[0])
            else:
                for camera_id in camera_ids:
                    renderer._model = model
                    renderer.update_scene(data=data, camera=camera_id)
                    frame = renderer.render()[:, :, ::-1]
                    frames.append(frame)

        return frames

    def _close_renderers(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()

    def close(self) -> None:
        self._close_renderers()
        del self._mj_model
        del self._mj_data
        del self._observation_space
        del self._action_space
        del self._observables

    @abc.abstractmethod
    def _create_observation_space(self) -> DictSpaceType:
        raise NotImplementedError

    @abc.abstractmethod
    def _update_observations(self, state: BaseEnvState) -> BaseEnvState:
        raise NotImplementedError

    @abc.abstractmethod
    def _create_action_space(self) -> BoxSpaceType:
        raise NotImplementedError

    @abc.abstractmethod
    def _update_simulation(self, state: BaseEnvState, ctrl: chex.Array) -> BaseEnvState:
        raise NotImplementedError

    @abc.abstractmethod
    def _prepare_reset(self) -> Tuple[Tuple[ModelType, DataType], ...]:
        raise NotImplementedError

    @abc.abstractmethod
    def _finish_reset(
        self,
        models_and_datas: Tuple[Tuple[ModelType, DataType], ...],
        rng: np.random.RandomState | chex.PRNGKey,
    ) -> BaseEnvState:
        raise NotImplementedError

    @abc.abstractmethod
    def reset(
        self, rng: np.random.RandomState | chex.PRNGKey, *args, **kwargs
    ) -> BaseEnvState:
        raise NotImplementedError

    @abc.abstractmethod
    def _create_observables(self) -> List[BaseObservable]:
        raise NotImplementedError

    @abc.abstractmethod
    def _update_reward(
        self, state: BaseEnvState, previous_state: BaseEnvState
    ) -> BaseEnvState:
        raise NotImplementedError

    @abc.abstractmethod
    def _update_terminated(self, state: BaseEnvState) -> BaseEnvState:
        raise NotImplementedError

    @abc.abstractmethod
    def _update_truncated(self, state: BaseEnvState) -> BaseEnvState:
        raise NotImplementedError

    @abc.abstractmethod
    def _update_info(self, state: BaseEnvState) -> BaseEnvState:
        raise NotImplementedError

# Ref: moojoco - MJXEnv
class FabMjxBaseEnv(FabMujocoBaseEnv):
    def __init__(
        self,
        xml_path: str,
        configuration: FabEnvBaseConfig
    ) -> None:
        super().__init__(xml_path=xml_path, configuration=configuration)
        self._body_geom_ids = None
        self._joints_qpos_adrs = None
        self._joints_qvel_adrs = None
        self._cache_references(mj_model=self.frozen_mj_model)

    def _cache_references(self, mj_model: mujoco.MjModel) -> None:
        self._get_body_geom_ids(mj_model=mj_model)
        self._get_joints_qpos_adrs(mj_model=mj_model, joint_subname="joint")
        self._get_joints_qvel_adrs(mj_model=mj_model, joint_subname="joint")

    def _get_joints_qpos_adrs(self, mj_model: mujoco.MjModel, joint_subname: str) -> jnp.ndarray:
        if self._joints_qpos_adrs is None:
            self._joints_qpos_adrs = jnp.array(
                [
                    mj_model.joint(joint_id).qposadr[0]
                    for joint_id in range(mj_model.njnt)
                    if joint_subname in mj_model.joint(joint_id).name
                ]
            )
        return self._joints_qpos_adrs

    def _get_joints_qvel_adrs(self, mj_model: mujoco.MjModel, joint_subname: str) -> jnp.ndarray:
        if self._joints_qvel_adrs is None:
            self._joints_qvel_adrs = jnp.array(
                [
                    mj_model.joint(joint_id).dofadr[0]
                    for joint_id in range(mj_model.njnt)
                    if joint_subname in mj_model.joint(joint_id).name
                ]
            )
        return self._joints_qvel_adrs

    def _get_body_geom_ids(self, mj_model: mujoco.MjModel) -> jnp.ndarray:
        if self._body_geom_ids is None:
            self._body_geom_ids = jnp.array(
                [
                    geom_id
                    for geom_id in range(mj_model.ngeom)
                    if "link" in mj_model.geom(geom_id).name
                ]
            )
        return self._body_geom_ids

    @property
    def frozen_mjx_model(self) -> mjx.Model:
        return self._mjx_model

    @property
    def frozen_mjx_data(self) -> mjx.Data:
        return self._mjx_data

    def _initialize_mjx_model_and_data(self) -> Tuple[mjx.Model, mjx.Data]:
        return mjx.put_model(m=self.frozen_mj_model), mjx.put_data(
            m=self.frozen_mj_model, d=self.frozen_mj_data
        )

    @staticmethod
    def _get_batch_size(state: MJXEnvState) -> int:
        try:
            return state.reward.shape[0]
        except IndexError:
            return 1
        except AttributeError:
            return 1

    def _get_mj_models_and_datas_to_render(
        self, state: MJXEnvState
    ) -> Tuple[List[mujoco.MjModel], List[mujoco.MjData]]:
        num_models = (
            1
            if self.environment_configuration.render_mode == "human"
            else self._get_batch_size(state=state)
        )
        mj_models = mjx_get_model(
            mj_model=state.mj_model, mjx_model=state.mjx_model, n_mj_models=num_models
        )
        mj_datas = mjx.get_data(m=mj_models[0], d=state.mjx_data)
        if not isinstance(mj_datas, list):
            mj_datas = [mj_datas]
        return mj_models, mj_datas

    @property
    def observables(self) -> List[MJXObservable]:
        return self._observables

    def _create_observation_space(self) -> mjx_spaces.Dict:
        observation_space = dict()
        for observable in self.observables:
            observation_space[observable.name] = mjx_spaces.Box(
                low=observable.low, high=observable.high, shape=observable.shape
            )
        return mjx_spaces.Dict(observation_space)

    def _create_action_space(self) -> mjx_spaces.Box:
        bounds = jnp.array(
            self.frozen_mj_model.actuator_ctrlrange.copy().astype(np.float32)
        )
        low, high = bounds.T
        action_space = mjx_spaces.Box(
            low=low, high=high, shape=low.shape, dtype=jnp.float32
        )
        return action_space

    def _update_observations(self, state: MJXEnvState) -> MJXEnvState:
        observations = jax.tree_util.tree_map(
            lambda observable: (observable.name, observable(state=state)),
            self.observables,
        )
        # noinspection PyUnresolvedReferences
        return state.replace(observations=dict(observations))

    def _update_simulation(self, state: MJXEnvState, ctrl: jnp.ndarray) -> MJXEnvState:
        mjx_data = state.mjx_data

        def _simulation_step(_data, _) -> Tuple[mjx.Data, None]:
            _data = _data.replace(ctrl=ctrl)
            return mjx.step(state.mjx_model, _data), None

        mjx_data, _ = jax.lax.scan(
            _simulation_step,
            mjx_data,
            (),
            self.environment_configuration.num_physics_steps_per_control_step,
        )
        # noinspection PyUnresolvedReferences
        return state.replace(mjx_data=mjx_data)

    def close(self) -> None:
        super().close()
        del self._mjx_model
        del self._mjx_data

    def _prepare_reset(
        self,
    ) -> Tuple[Tuple[mujoco.MjModel, mujoco.MjData], Tuple[mjx.Model, mjx.Data]]:
        mj_model = self.frozen_mj_model
        mj_data = self.frozen_mj_data

        mjx_model = self.frozen_mjx_model
        mjx_data = self.frozen_mjx_data
        return (mj_model, mj_data), (mjx_model, mjx_data)

    def _finish_reset(
        self,
        models_and_datas: Tuple[
            Tuple[mujoco.MjModel, mujoco.MjData], Tuple[mjx.Model, mjx.Data]
        ],
        rng: np.random.RandomState,
    ) -> MJXEnvState:
        (mj_model, mj_data), (mjx_model, mjx_data) = models_and_datas
        mjx_data = mjx.forward(m=mjx_model, d=mjx_data)
        state = MJXEnvState(
            mj_model=mj_model,
            mj_data=mj_data,
            mjx_model=mjx_model,
            mjx_data=mjx_data,
            observations={},
            reward=0,
            terminated=False,
            truncated=False,
            info={},
            rng=rng,
        )
        state = self._update_observations(state=state)
        state = self._update_info(state=state)
        return state

    def _update_reward(
        self, state: BaseEnvState, previous_state: BaseEnvState
    ) -> BaseEnvState:
        return state.replace(reward=0)

    def _update_info(self, state: BaseEnvState) -> BaseEnvState:
        info = {
            "time": self._get_time(state=state)
        }
        return state.replace(info=info)

    def _update_terminated(self, state: BaseEnvState) -> BaseEnvState:
        return state.replace(terminated=0)

    def _update_truncated(self, state: BaseEnvState) -> BaseEnvState:
        truncated = (
            self._get_time(state=state) > self.environment_configuration.simulation_time
        )
        return state.replace(truncated=truncated)