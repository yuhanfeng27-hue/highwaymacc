import pytest
import torch

from benchmarl.models.entityattention import VehicleAttentionEncoder


@pytest.mark.parametrize('entities', [1, 8, 12])
@pytest.mark.parametrize('batch_shape', [(), (4,), (2, 4)])
def test_encoder_shapes_and_padding(entities, batch_shape):
    torch.manual_seed(0)
    encoder = VehicleAttentionEncoder()
    x = torch.randn(*batch_shape, entities, 5)
    x[..., 0, 0] = 1
    if entities > 1:
        x[..., -1, :] = 0
    y = encoder(x.flatten(-2))
    assert y.shape == (*batch_shape, 32)
    assert torch.isfinite(y).all()
    if entities > 1:
        changed = x.clone()
        changed[..., -1, 1:] = 1000
        torch.testing.assert_close(y, encoder(changed.flatten(-2)))
    # Appending a padded slot must not change the representation.
    padded = torch.cat((x, torch.zeros_like(x[..., :1, :])), dim=-2)
    torch.testing.assert_close(y, encoder(padded.flatten(-2)), atol=1e-6, rtol=1e-5)
    y.square().sum().backward()
    grad = encoder.attention.in_proj_weight.grad
    assert torch.isfinite(grad).all()
    if entities > 1:
        assert grad.abs().sum() > 0


def test_entity_permutation_invariance_and_agent_isolation():
    torch.manual_seed(7)
    encoder = VehicleAttentionEncoder()
    x = torch.randn(2, 4, 8, 5)
    x[..., 0] = 1
    y = encoder(x.flatten(-2))
    permuted = x[..., [0, 4, 2, 7, 1, 6, 3, 5], :]
    torch.testing.assert_close(y, encoder(permuted.flatten(-2)), atol=1e-6, rtol=1e-5)
    changed = x.clone()
    changed[:, 1:] += 20
    torch.testing.assert_close(y[:, 0], encoder(changed.flatten(-2))[:, 0])


def test_invalid_head_dimensions():
    with pytest.raises(ValueError, match='divisible'):
        VehicleAttentionEncoder(embed_dim=31, num_heads=4)
