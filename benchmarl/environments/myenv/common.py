from typing import Callable, Dict, List, Optional

import numpy as np

from benchmarl.environments.common import Task, TaskClass

from benchmarl.utils import DEVICE_TYPING

from gymnasium import spaces
from pettingzoo import ParallelEnv

from torchrl.data import CompositeSpec
from torchrl.envs import EnvBase, PettingZooWrapper
import gymnasium as gym
import highway_env

class MyCustomEnv2(ParallelEnv):
    """
    Multi-agent version of my single agent class.
    """

    metadata = {"render_modes": ["human"], "name": "myclass_v0"}

    def __init__(self, num_envs=2):
        super(MyCustomEnv2, self).__init__()
        self.state = None
        self.t = 0
        self.env = gym.make('highway-cacc', render_mode='human')
        # self.env = gym.make('highway-cacc')
        self.episode_count = 0
        num_agents = 4
        self.possible_agents = ["veh_" + str(r + 1) for r in range(num_agents)]

        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.render_mode = 'human'        # 'human'None

    def render(self):
        return self.env.render()

    def observation_space(self, agent):
        state_low = np.full(37, -np.inf, dtype=np.float32)
        state_high = np.full(37, np.inf, dtype=np.float32)

        return spaces.Dict({
            # 原始的物理观测
            "observation": spaces.Box(
                low=state_low,
                high=state_high,
                shape=(37,),
                dtype=np.float32,
            ),
            # 新增的动作屏蔽掩码 (5个离散动作)
            "action_mask": spaces.Box(
                low=0,
                high=1,
                shape=(5,),
                dtype=np.int8,
            )
        })

    def action_space(self, agent):
        return spaces.Discrete(5)

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, options=None):
        self.t = 0
        self.agents = self.possible_agents[:]

        joint_obs, info = self.env.reset(seed=seed, options=options)
        mask = info.get("action_mask", np.ones((len(self.agents), 5), dtype=np.int8))

        # 将观测值和掩码打包成字典
        observations = {
            agent: {
                "observation": joint_obs[i].astype(np.float32),
                "action_mask": mask[i].astype(np.int8)
            } for i, agent in enumerate(self.agents)
        }

        infos = {agent: {} for agent in self.agents}
        self.state = observations
        self.episode_count += 1
        # print(f"\n=========================================")
        # print(f"🎬 正在开始第 {self.episode_count} 局测试...")
        # print(f"=========================================")

        return observations, infos

    def step(self, actions):
        self.t += 1
        action_tuple = tuple(actions[agent] for agent in self.agents)
        # action_names = {0: "LEFT(左转)", 1: "IDLE(保持)", 2: "RIGHT(右转)", 3: "FASTER(加速)", 4: "SLOWER(减速)"}
        # print(f"\n--- 第 {self.t} 步动作输出 ---")
        # for i, agent in enumerate(self.agents):
        #     # 兼容 Tensor 和 int
        #     act_val = int(action_tuple[i].item()) if hasattr(action_tuple[i], 'item') else int(action_tuple[i])
        #     act_str = action_names.get(act_val, f"未知({act_val})")
        #     print(f"{agent}: {act_str}")
        joint_obs, reward, terminated, truncated, info = self.env.step(action_tuple)

        # 获取环境传出来的 mask
        mask = info.get("action_mask", np.ones((len(self.agents), 5), dtype=np.int8))

        # 将观测值和掩码打包成字典
        observations = {
            agent: {
                "observation": joint_obs[i].astype(np.float32),
                "action_mask": mask[i].astype(np.int8)
            } for i, agent in enumerate(self.agents)
        }

        rewards = {agent: float(reward) for agent in self.agents}
        truncated = terminated or truncated
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}

        self.state = observations
        infos = {agent: {} for agent in self.agents}

        return observations, rewards, terminations, truncations, infos

class MyenvTask(Task):
    MY_TASK = "my_task"

    @staticmethod
    def associated_class():
        return MyEnvClass

class MyEnvClass(TaskClass):

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:

        return lambda: PettingZooWrapper(
            MyCustomEnv2(),
            categorical_actions=True,
            device=device,
            seed=seed,
            return_state=False,
            **self.config,
        )

    def supports_continuous_actions(self) -> bool:
        return False

    def supports_discrete_actions(self) -> bool:
        return True

    def has_state(self) -> bool:
        return False

    def has_render(self, env: EnvBase) -> bool:
        return True

    def max_steps(self, env: EnvBase) -> int:
        return 50

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        return {"veh":list(env.agents)}

    def state_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        return None

    def action_mask_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        observation_spec = env.observation_spec.clone()
        mask_spec = CompositeSpec()

        for group in self.group_map(env):
            # 如果底层 wrapper 成功解析了 action_mask，把它提取出来作为掩码的 spec
            if "action_mask" in observation_spec[group].keys():
                mask_spec[group] = CompositeSpec(
                    {"action_mask": observation_spec[group]["action_mask"]}
                )

        return mask_spec if not mask_spec.is_empty() else None

    def observation_spec(self, env: EnvBase) -> CompositeSpec:
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_obs_spec = observation_spec[group]
            for key in list(group_obs_spec.keys()):
                if key != "observation":
                    del group_obs_spec[key]
        if "state" in observation_spec.keys():
            del observation_spec["state"]
        return observation_spec

    def info_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_obs_spec = observation_spec[group]
            for key in list(group_obs_spec.keys()):
                if key != "info":
                    del group_obs_spec[key]
        if "state" in observation_spec.keys():
            del observation_spec["state"]
        return observation_spec

    def action_spec(self, env: EnvBase) -> CompositeSpec:
        return env.full_action_spec

    @staticmethod
    def env_name() -> str:
        return "myenv"
