# CACC 注意力策略的课程学习

三个阶段共用同一结构：自车查询注意力（默认 32 维、4 个头）+ 共享策略 MLP，critic 保持 MLP。
阶段切换只迁移各组的策略/价值网络状态；优化器、采样器、经验缓存和训练计数重新开始。
奖励函数和三个阶段的场景定义保持原样。本次未实现阶段 4。

## 连续训练阶段 1 → 2 → 3

在 highwaymacc 仓库根目录、激活开发环境后运行：

```bash
python benchmarl/run_curriculum.py "curriculum.stages=[1,2,3]" experiment.max_n_frames=2000000
```

每个阶段分别训练指定的 200 万帧，再迁移到下一阶段；这是按预算切换，不是按成功率自动晋级。
默认实验配置需要 CUDA 和 wandb。CPU/本地日志训练时追加：

```text
experiment.sampling_device=cpu experiment.train_device=cpu experiment.buffer_device=cpu experiment.loggers=[csv]
```

无窗口环境设置 SDL_VIDEODRIVER=dummy、SDL_AUDIODRIVER=dummy。
不需要在线日志时无需 wandb 密钥。云环境可先执行：

```bash
source /workspace/.cloud-setup/activate.sh
```

## 手动切换，分别设置每阶段预算

阶段 1 从头训练：

```bash
python benchmarl/run_curriculum.py task=myenv/stage1 experiment.max_n_frames=2000000
```

查看输出目录中的 `curriculum_manifest.json`，找到阶段 1 的 `weights` 路径，再运行：

```bash
python benchmarl/run_curriculum.py task=myenv/stage2 "curriculum.warmstart_from=/path/to/stage1/curriculum_weights.pt" experiment.max_n_frames=1000000
```

阶段 3 同理，必须指定阶段 2 导出的权重。也可以从阶段 2 连续训练到阶段 3：

```bash
python benchmarl/run_curriculum.py task=myenv/stage2 "curriculum.stages=[2,3]" "curriculum.warmstart_from=/path/to/stage1/curriculum_weights.pt"
```

默认输出在 Hydra 本次运行目录的 stage1、stage2、stage3 子目录中，每个实验有自己的日志和权重。
可以通过 `experiment.save_folder=/path/to/new/run` 指定一个新的输出根目录。
`curriculum_manifest.json` 记录每个已完成阶段的输入权重、输出权重、帧数和实验目录。
已有清单的目录不会覆盖；续训下一阶段时使用新的运行目录。

## 兼容性与恢复

- 后续阶段必须接续前一阶段，不能直接从阶段 1 跳到阶段 3。
- 各阶段保持相同 actor、critic、算法设置、共享参数选项和观测格式。注意力头数变化也会拒绝加载。
- 阶段 1 没有 HDV 时，其观测槽位仍保留并通过 padding mask 屏蔽。
- `curriculum_weights.pt` 使用 `weights_only=True` 加载，不包含完整采样器/缓存或优化器状态。
- 旧 MLP 模型不能直接迁移到注意力模型；旧版完整 checkpoint 也不能直接充当本入口的迁移文件。
- 同一阶段的完整实验恢复仍使用现有 resume 入口及普通 checkpoint；不要同时设置
  `experiment.restore_file` 和课程迁移。迁移文件不能用于完整实验恢复。
- 课程入口单独保存每阶段的 Hydra 配置和 task 选择，避免评估时误用阶段 1 的场景。
- 当前不做自动达标判断。建议检查编队成功率、碰撞率和恢复时间后再决定训练预算或手动升级。

## 最小验证

```bash
python benchmarl/run_curriculum.py "curriculum.stages=[1,2,3]" experiment.sampling_device=cpu experiment.train_device=cpu experiment.buffer_device=cpu "experiment.loggers=[csv]" experiment.evaluation=false experiment.render=false experiment.max_n_frames=20 experiment.on_policy_collected_frames_per_batch=20 experiment.on_policy_minibatch_size=10 experiment.on_policy_n_minibatch_iters=1 experiment.checkpoint_interval=0 experiment.checkpoint_at_end=false experiment.create_json=false
```

此命令只验证数据采集、参数更新、导出和跨阶段加载，不代表学会编队。
奖励漏洞及固定车道偏好的分析仍然适用，增加注意力或迁移权重不会自动解决它们。
