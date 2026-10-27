from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

from benchmarl.experiment import logger as logger_module


def make_logger(monkeypatch, tmp_path, name, **extra):
    calls = []
    monkeypatch.setattr(logger_module, 'get_logger', lambda **kwargs: calls.append(kwargs))
    folder = tmp_path / 'stage1' / name
    folder.mkdir(parents=True, exist_ok=True)
    result = logger_module.Logger(
        experiment_name=name, folder_name=str(folder),
        experiment_config=SimpleNamespace(create_json=True, loggers=['csv', 'wandb']),
        algorithm_name='mappo', environment_name='myenv', task_name='stage1',
        model_name='entityattention', group_map={'veh': ['a']}, seed=0,
        project_name='benchmarl', wandb_extra_kwargs=extra,
    )
    return result, calls


def test_short_paths_preserve_display_name_and_explicit_settings(monkeypatch, tmp_path):
    name = 'mappo_stage1_entityattention__890eab2a_26_10_27-22_31_28'
    result, calls = make_logger(monkeypatch, tmp_path, name)
    csv, wandb = calls
    assert csv['logger_name'] == str(tmp_path / 'stage1' / name)
    assert wandb['logger_name'] == str(tmp_path / 'stage1')
    assert wandb['experiment_name'] == name
    run_id = wandb['wandb_kwargs']['id']
    assert len(run_id) == 16
    assert result.json_writer.path.name == 'results.json'
    # Reconstruct the user's Windows upload path with the shortened components.
    path = PureWindowsPath('D:/code/BenchMARL-main/outputs/2026-10-27/22-31-27/stage1')
    path = path / 'wandb' / f'run-20261027_223133-{run_id}' / 'files' / 'results.json.tmp~wandb'
    assert len(str(path)) < 240
    _, same = make_logger(monkeypatch, tmp_path, name)
    assert same[1]['wandb_kwargs']['id'] == run_id
    _, next_stage = make_logger(monkeypatch, tmp_path, name.replace('stage1', 'stage2'))
    assert next_stage[1]['wandb_kwargs']['id'] != run_id
    override = tmp_path / 'custom'
    _, custom = make_logger(monkeypatch, tmp_path, name, id='custom-id', save_dir=str(override))
    assert custom[1]['wandb_kwargs']['id'] == 'custom-id'
    assert custom[1]['logger_name'] == str(override)
    assert 'save_dir' not in custom[1]['wandb_kwargs']
    assert override.is_dir()
