import sys
from pathlib import Path

import pytest
import torch


UNET_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UNET_DIR))

from models.CSANet import CSANet, CrossSliceAttention  # noqa: E402


@pytest.fixture(scope="module")
def small_model():
    model = CSANet(
        num_classes=3,
        in_channels=3,
        encoder_weights=None,
        image_size=32,
        transformer_layers=1,
        transformer_heads=4,
        hidden_size=64,
        mlp_dim=128,
        dropout=0.0,
    )
    return model.eval()


def test_neighbor_indices_duplicate_center_at_boundaries():
    previous, following = CSANet._neighbor_indices(4, torch.device("cpu"))
    assert previous.tolist() == [0, 0, 1, 2]
    assert following.tolist() == [1, 2, 3, 3]

    previous, following = CSANet._neighbor_indices(1, torch.device("cpu"))
    assert previous.tolist() == [0]
    assert following.tolist() == [0]


@pytest.mark.parametrize("length", [1, 2, 4])
def test_all_slices_are_predicted(small_model, length):
    images = torch.randn(1, length, 3, 32, 32)
    with torch.inference_mode():
        logits = small_model(images)
    assert logits.shape == (1, length, 3, 32, 32)
    assert torch.isfinite(logits).all()


def test_cross_slice_attention_is_registered_and_differentiable():
    attention = CrossSliceAttention(channels=16, num_heads=4).train()
    # Le zero-init non-local bloque volontairement Q/K/V au tout premier pas.
    # Une contribution non nulle verifie ensuite toute la branche de gradient.
    with torch.no_grad():
        attention.output[1].weight.fill_(1.0)

    center = torch.randn(2, 16, 3, 3, requires_grad=True)
    neighbor = torch.randn(2, 16, 3, 3, requires_grad=True)
    attention(center, neighbor).square().mean().backward()

    assert center.grad is not None
    assert neighbor.grad is not None
    assert all(parameter.grad is not None for parameter in attention.parameters())


def test_state_dict_round_trip(small_model):
    state = small_model.state_dict()
    result = small_model.load_state_dict(state)
    assert not result.missing_keys
    assert not result.unexpected_keys
