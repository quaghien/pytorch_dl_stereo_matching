import torch

from dl_stereo_matching_pytorch.models import build_cost_volume, create_model, get_network_spec, soft_target_cross_entropy


def test_model_output_shapes() -> None:
    model19 = create_model("win19_dep9", in_channels=1)
    logits19 = model19(torch.randn(2, 1, 19, 19), torch.randn(2, 1, 19, 23))
    assert logits19.shape == (2, 5)

    model37 = create_model("win37_dep9", in_channels=3)
    logits37 = model37(torch.randn(2, 3, 37, 37), torch.randn(2, 3, 37, 43))
    assert logits37.shape == (2, 7)


def test_soft_target_cross_entropy_is_finite() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    targets = torch.tensor([[0.05, 0.2, 0.75]], dtype=torch.float32)
    loss = soft_target_cross_entropy(logits, targets)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_network_receptive_fields() -> None:
    assert get_network_spec("win19_dep9").receptive_field == 19
    assert get_network_spec("win37_dep9").receptive_field == 37


def test_cost_volume_alignment() -> None:
    left = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    right = torch.tensor([[[[10.0, 20.0, 30.0, 40.0]]]])
    volume = build_cost_volume(left, right, disp_range=3)

    assert volume.shape == (1, 3, 1, 4)
    assert torch.equal(volume[0, 0, 0], torch.tensor([10.0, 40.0, 90.0, 160.0]))
    assert torch.equal(volume[0, 1, 0], torch.tensor([0.0, 20.0, 60.0, 120.0]))
    assert torch.equal(volume[0, 2, 0], torch.tensor([0.0, 0.0, 30.0, 80.0]))
