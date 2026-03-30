from typing import List, Any, Type, Optional, Union, Sequence

import gym
import gymnasium as gymnasium
import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn, VecEnvIndices

from mbt_gym.gym.TradingEnvironment import TradingEnvironment


def _to_gymnasium_space(space):
    """Convert gym spaces to gymnasium spaces for SB3 2.x compatibility."""
    if isinstance(space, gym.spaces.Box):
        return gymnasium.spaces.Box(
            low=space.low,
            high=space.high,
            shape=space.shape,
            dtype=space.dtype,
        )
    elif isinstance(space, gym.spaces.Discrete):
        return gymnasium.spaces.Discrete(n=space.n)
    elif isinstance(space, gym.spaces.MultiDiscrete):
        return gymnasium.spaces.MultiDiscrete(nvec=space.nvec)
    elif isinstance(space, gym.spaces.MultiBinary):
        return gymnasium.spaces.MultiBinary(n=space.n)
    return space


class StableBaselinesTradingEnvironment(VecEnv):
    def __init__(
        self,
        trading_env: TradingEnvironment,
        store_terminal_observation_info: bool = True,
    ):
        self.env = trading_env
        self.store_terminal_observation_info = store_terminal_observation_info
        self.actions: np.ndarray = self.env.action_space.sample()
        # Required for stable_baselines3 compatibility
        self.render_mode = None
        self.metadata = {"render_modes": []}
        # Convert spaces to gymnasium spaces for SB3 2.x compatibility
        self.observation_space = _to_gymnasium_space(self.env.observation_space)
        self.action_space = _to_gymnasium_space(self.env.action_space)
        super().__init__(self.env.num_trajectories, self.observation_space, self.action_space)

    def reset(self) -> VecEnvObs:
        return self.env.reset()

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, dones, infos = self.env.step(self.actions)
        if dones.min():
            if self.store_terminal_observation_info:
                infos = infos.copy()
                for count, info in enumerate(infos):
                    # save final observation where user can get it, then automatically reset (an SB3 convention).
                    info["terminal_observation"] = obs[count, :]
            obs = self.env.reset()
        return obs, rewards, dones, infos

    def close(self) -> None:
        pass

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        if indices is None:
            indices = range(self.num_envs)
        return [getattr(self.env, attr_name, None) for _ in indices]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        if indices is None:
            indices = range(self.num_envs)
        for i in indices:
            setattr(self.env, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        if indices is None:
            indices = range(self.num_envs)
        return [getattr(self.env, method_name)(*method_args, **method_kwargs) for _ in indices]

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        return [False for _ in range(self.env.num_trajectories)]

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        return self.env.seed(seed)

    def get_images(self) -> Sequence[np.ndarray]:
        pass

    @property
    def num_trajectories(self):
        return self.env.num_trajectories

    @property
    def n_steps(self):
        return self.env.n_steps
