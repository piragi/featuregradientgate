import numpy as np
import torch

from featuregating.experiments.saco import (
    _measure_bin_drops,
    _target_class_confidences,
)


class _TwoClassModel(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = inputs.flatten(start_dim=1)[:, 0]
        return torch.stack((values, -values), dim=1)


def test_target_class_confidences_do_not_follow_perturbed_argmax():
    model = _TwoClassModel()
    batch = torch.tensor([[[[-1.0]]], [[[-2.0]]]])

    confidences = _target_class_confidences(
        model,
        batch,
        target_class_idx=0,
        device=torch.device("cpu"),
    )

    expected = torch.softmax(model(batch), dim=1)[:, 0].numpy()
    np.testing.assert_allclose(confidences, expected)
    assert np.all(confidences < 0.5)


def test_measure_bin_drops_uses_original_explained_class():
    model = _TwoClassModel()
    perturbed = [torch.tensor([[[-1.0]]]), torch.tensor([[[-2.0]]])]

    drops = _measure_bin_drops(
        perturbed,
        original_confidence=0.9,
        target_class_idx=0,
        model=model,
        device=torch.device("cpu"),
    )

    batch = torch.stack(perturbed)
    expected = 0.9 - torch.softmax(model(batch), dim=1)[:, 0].numpy()
    np.testing.assert_allclose(drops, expected)
    assert drops[1] > drops[0] > 0
