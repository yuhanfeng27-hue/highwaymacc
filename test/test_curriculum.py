from types import SimpleNamespace

import pytest
import torch
from torch import nn

from benchmarl.run_curriculum import load_stage_weights, save_stage_weights, validate_stages


@pytest.mark.parametrize('stages,source', [([1, 2, 3], None), ([2, 3], 'weights.pt'), ([1], None)])
def test_valid_stages(stages, source):
    assert validate_stages(stages, source) == stages


@pytest.mark.parametrize('stages,source', [([], None), ([1, 3], None), ([2], None), ([3, 2], 'x'), ([4], 'x'), ([True], None)])
def test_invalid_stages(stages, source):
    with pytest.raises(ValueError):
        validate_stages(stages, source)


def experiment(tmp_path):
    return SimpleNamespace(
        folder_name=tmp_path, losses={'veh': nn.Linear(5, 3)},
        config=SimpleNamespace(collect_with_grad=True),
        total_frames=0, n_iters_performed=0,
    )


def test_transfer_weights_and_reset_training_state(tmp_path):
    source = experiment(tmp_path)
    signature = {'model': {'embed_dim': 32, 'num_heads': 4}}
    path = save_stage_weights(source, 1, signature)
    target = experiment(tmp_path)
    assert not torch.equal(source.losses['veh'].weight, target.losses['veh'].weight)
    optimizer = torch.optim.Adam(target.losses['veh'].parameters())
    load_stage_weights(target, path, 2, signature)
    for key, value in source.losses['veh'].state_dict().items():
        torch.testing.assert_close(value, target.losses['veh'].state_dict()[key])
    assert not optimizer.state
    assert target.total_frames == target.n_iters_performed == 0


def test_evaluation_loads_same_stage(tmp_path):
    source, target = experiment(tmp_path), experiment(tmp_path)
    path = save_stage_weights(source, 2, {})
    load_stage_weights(target, path, 2, {}, evaluation=True)
    torch.testing.assert_close(source.losses['veh'].weight, target.losses['veh'].weight)


@pytest.mark.parametrize('failure', ['stage', 'signature', 'shape', 'legacy'])
def test_reject_incompatible_transfer_before_mutation(tmp_path, failure):
    source, target = experiment(tmp_path), experiment(tmp_path)
    path = save_stage_weights(source, 1, {'heads': 4})
    checkpoint = torch.load(path, weights_only=True)
    if failure == 'stage':
        checkpoint['stage'] = 2
    elif failure == 'signature':
        checkpoint['signature'] = {'heads': 8}
    elif failure == 'shape':
        checkpoint['weights']['loss_veh']['weight'] = torch.zeros(1)
    else:
        checkpoint.pop('format')
    torch.save(checkpoint, path)
    before = target.losses['veh'].weight.detach().clone()
    with pytest.raises(ValueError):
        load_stage_weights(target, path, 2, {'heads': 4})
    torch.testing.assert_close(before, target.losses['veh'].weight)
