#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
"""课程学习入口：从上一阶段的 checkpoint 只加载网络权重（warm-start）。

当前阶段作为一个全新的实验运行（独立的名字 / 输出目录 / wandb run），
只是用上一阶段学到的策略/价值网络初始化，帧数、优化器、collector 全部从头开始。

用法（PowerShell，在 D:\\code\\BenchMARL-main 目录下）：
    python benchmarl/run_curriculum.py task=myenv/stage2 experiment.max_n_frames=2000000

切换阶段时，只需要改下面这个 WARMSTART_FROM 路径即可。
"""

import warnings

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from benchmarl.hydra_config import load_experiment_from_hydra

warnings.filterwarnings("ignore", message="size_average and reduce args will be deprecated")

# ============================================================
# 【在这里填上一阶段的最终 checkpoint 路径】
# 切到 stage2 时填 stage1 的 checkpoint，切到 stage3 时填 stage2 的。
# ============================================================
WARMSTART_FROM = r"outputs\2026-09-08\21-58-37\mappo_stage1_mlp__8a805c68_26_09_08-21_58_37\checkpoints\checkpoint_2640000.pt"


@hydra.main(version_base=None, config_path="conf", config_name="config")
def hydra_experiment(cfg: DictConfig) -> None:
    task_name = HydraConfig.get().runtime.choices.task
    algorithm_name = HydraConfig.get().runtime.choices.algorithm

    print(f"\nAlgorithm: {algorithm_name}, Task: {task_name}")
    print("\nLoaded config:\n")
    print(OmegaConf.to_yaml(cfg))

    experiment = load_experiment_from_hydra(cfg, task_name=task_name)

    # 只把上一阶段的网络权重加载进来，不恢复帧数/优化器/collector/wandb
    if WARMSTART_FROM:
        ckpt = torch.load(WARMSTART_FROM, map_location="cpu")
        for group in experiment.group_map.keys():
            experiment.losses[group].load_state_dict(ckpt[f"loss_{group}"])
        print(f"\n[WarmStart] 已从 {WARMSTART_FROM} 加载网络权重\n")

    experiment.run()


if __name__ == "__main__":
    hydra_experiment()
