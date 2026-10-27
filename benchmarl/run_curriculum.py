"""CACC curriculum training with explicit, network-only stage transitions."""

import json
import copy
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from benchmarl.hydra_config import load_experiment_from_hydra


def validate_stages(stages, warmstart_from=None):
    stages = list(stages)
    if not stages or any(type(s) is not int or s not in (1, 2, 3) for s in stages):
        raise ValueError("Curriculum stages must be integers from 1 to 3")
    if any(b != a + 1 for a, b in zip(stages, stages[1:])):
        raise ValueError("Curriculum stages must be consecutive and increasing")
    if stages[0] > 1 and not warmstart_from:
        raise ValueError("Stage 2/3 requires curriculum.warmstart_from from the previous stage")
    return stages


def network_signature(cfg):
    # Include architecture choices beyond tensor shapes (e.g. attention head count).
    return OmegaConf.to_container(OmegaConf.create({
        'algorithm': OmegaConf.to_container(cfg.algorithm, resolve=True),
        'model': OmegaConf.to_container(cfg.model, resolve=True),
        'critic_model': OmegaConf.to_container(cfg.critic_model, resolve=True),
        'share_policy_params': cfg.experiment.share_policy_params,
    }), resolve=True)


def load_stage_weights(experiment, path, stage, signature, *, evaluation=False):
    """Load network state only; never restore counters, optimizer, or collector."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format') != 'benchmarl-curriculum-v1':
        raise ValueError("Use curriculum_weights.pt exported by this curriculum entry point; "
                         "legacy/full checkpoints are not supported for stage transfer")
    source_stage = stage if evaluation else stage - 1
    if checkpoint.get('stage') != source_stage:
        raise ValueError(f"Stage {stage} requires weights from stage {source_stage}")
    if checkpoint.get('signature') != signature:
        raise ValueError("Network configuration differs from the previous stage; "
                         "keep actor, critic, algorithm and sharing settings identical")
    expected_groups = {f'loss_{group}' for group in experiment.losses}
    weights = checkpoint['weights']
    if set(weights) != expected_groups:
        raise ValueError("Checkpoint agent groups do not match this experiment")
    # Validate every group before mutating any parameters.
    for group, loss in experiment.losses.items():
        current, incoming = loss.state_dict(), weights[f'loss_{group}']
        if current.keys() != incoming.keys() or any(
            (not isinstance(incoming[key], torch.Tensor)
             or current[key].shape != incoming[key].shape)
            if isinstance(current[key], torch.Tensor)
            else current[key] != incoming[key]
            for key in current
        ):
            raise ValueError(f"Incompatible network weights for group {group}")
    for group, loss in experiment.losses.items():
        loss.load_state_dict(weights[f'loss_{group}'], strict=True)
    if not experiment.config.collect_with_grad:
        experiment.collector.update_policy_weights_()
    print(f'[Curriculum] Loaded stage {source_stage} networks from {path}; '
          f'frames={experiment.total_frames}, iterations={experiment.n_iters_performed}', flush=True)


def save_stage_weights(experiment, stage, signature):
    path = Path(experiment.folder_name) / 'curriculum_weights.pt'
    payload = {
        'format': 'benchmarl-curriculum-v1', 'stage': stage,
        'signature': signature,
        'weights': {f'loss_{group}': {
            key: value.detach().cpu().clone() if isinstance(value, torch.Tensor)
            else copy.deepcopy(value) for key, value in loss.state_dict().items()
        } for group, loss in experiment.losses.items()},
    }
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def run_curriculum(cfg, selected_task, output_dir):
    if selected_task not in ('myenv/stage1', 'myenv/stage2', 'myenv/stage3'):
        raise ValueError('This curriculum entry point supports myenv/stage1, stage2 and stage3')
    if cfg.experiment.restore_file:
        raise ValueError('Do not combine experiment.restore_file with curriculum warm-start; '
                         'use the resume entry point for a same-stage resume')
    selected_stage = int(selected_task[-1])
    if cfg.task.env_args.curriculum_stage != selected_stage:
        raise ValueError('task choice and task.env_args.curriculum_stage disagree')
    source = cfg.curriculum.warmstart_from
    source = Path(to_absolute_path(source)).resolve() if source else None
    stages = validate_stages(
        cfg.curriculum.stages if cfg.curriculum.stages is not None else [selected_stage],
        source,
    )
    if stages[0] != selected_stage:
        raise ValueError('The first curriculum stage must match the selected task')
    if source and not source.is_file():
        raise FileNotFoundError(f'Curriculum weights not found: {source}')
    root = Path(to_absolute_path(cfg.experiment.save_folder or output_dir)).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / 'curriculum_manifest.json'
    if manifest_path.exists():
        raise FileExistsError(f'Use a new output directory; existing curriculum manifest: {manifest_path}')
    signature = network_signature(cfg)
    manifest = {'stages': stages, 'runs': []}
    for stage in stages:
        # Preserve structured algorithm/experiment dataclasses for the loader.
        stage_cfg = cfg.copy()
        stage_cfg.task.env_args.curriculum_stage = stage
        stage_cfg.curriculum.stages = [stage]
        stage_cfg.curriculum.warmstart_from = str(source) if source else None
        stage_dir = root / f'stage{stage}'
        stage_dir.mkdir(parents=True, exist_ok=True)
        stage_cfg.experiment.save_folder = str(stage_dir)
        print(f'[Curriculum] Starting stage {stage}', flush=True)
        experiment = load_experiment_from_hydra(stage_cfg, task_name=f'myenv/stage{stage}')
        # Store stage-specific metadata so evaluation does not use stage 1's task.
        metadata = Path(experiment.folder_name) / '.hydra'
        metadata.mkdir()
        OmegaConf.save(stage_cfg, metadata / 'config.yaml')
        original_overrides = Path(output_dir) / '.hydra' / 'overrides.yaml'
        overrides = list(OmegaConf.load(original_overrides)) if original_overrides.exists() else []
        overrides = [value for value in overrides if not value.startswith('task=')]
        overrides.append(f'task=myenv/stage{stage}')
        OmegaConf.save(OmegaConf.create(overrides), metadata / 'overrides.yaml')
        try:
            if source:
                load_stage_weights(experiment, source, stage, signature)
        except Exception:
            experiment.close()
            raise
        # Experiment.run() closes its services on completion or training errors.
        experiment.run()
        weights_path = save_stage_weights(experiment, stage, signature)
        manifest['runs'].append({
            'stage': stage, 'warmstart_from': str(source) if source else None,
            'weights': str(weights_path), 'output_dir': str(experiment.folder_name),
            'frames': experiment.total_frames,
        })
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        source = weights_path
        print(f'[Curriculum] Stage {stage} complete: {weights_path}', flush=True)
    return manifest


@hydra.main(version_base=None, config_path='conf', config_name='config')
def hydra_experiment(cfg: DictConfig):
    runtime = HydraConfig.get()
    run_curriculum(cfg, runtime.runtime.choices.task, runtime.runtime.output_dir)


if __name__ == '__main__':
    hydra_experiment()
