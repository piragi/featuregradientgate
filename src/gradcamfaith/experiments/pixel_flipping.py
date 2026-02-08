"""
Patch-based pixel flipping faithfulness metric following Bach et al. (2015).

Progressively perturbs the most important patches based on attribution scores,
measures prediction degradation, and returns AUC scores.
"""

import math

import numpy as np

from gradcamfaith.experiments.faithfulness import (
    apply_baseline_perturbation,
    create_patch_mask,
    predict_on_batch,
)


class PatchPixelFlipping:
    """
    Standalone patch-based pixel flipping implementation.

    Progressively perturbs the most important patches and measures prediction
    degradation.  Returns AUC scores measuring explanation faithfulness.
    """

    def __init__(self, n_patches=196, features_in_step=1, perturb_baseline="mean"):
        self.n_patches = n_patches
        self.patch_size = 32 if n_patches == 49 else 16
        self.features_in_step = features_in_step
        self.perturb_baseline = perturb_baseline

    def __call__(self, model, x_batch, y_batch, a_batch, device=None, batch_size=256):
        x_batch = np.asarray(x_batch)
        y_batch = np.asarray(y_batch)
        a_batch = np.asarray(a_batch)

        scores = []
        for start in range(0, len(x_batch), batch_size):
            end = min(start + batch_size, len(x_batch))
            scores.extend(
                self.evaluate_batch(
                    model=model,
                    x_batch=x_batch[start:end],
                    y_batch=y_batch[start:end],
                    a_batch=a_batch[start:end],
                    device=device,
                )
            )
        return scores

    def evaluate_batch(self, model, x_batch, y_batch, a_batch, device=None):
        """
        Progressively perturb most-important patches and measure prediction
        degradation.  Returns list of AUC scores.
        """
        batch_size = a_batch.shape[0]
        if a_batch.shape[1] != self.n_patches:
            raise ValueError(f"Expected {self.n_patches} patches, got {a_batch.shape[1]}")

        # Sort patches by attribution importance (descending)
        patch_indices_sorted = np.argsort(-a_batch, axis=1)

        n_perturbations = math.ceil(self.n_patches / self.features_in_step)
        predictions = []
        x_perturbed = x_batch.copy()

        # Initial predictions (baseline for the curve)
        predictions.append(predict_on_batch(model, x_batch, y_batch, device, use_softmax=True))

        # Progressive perturbation
        for step in range(n_perturbations):
            start_idx = step * self.features_in_step
            end_idx = min(start_idx + self.features_in_step, self.n_patches)
            patches_to_perturb = patch_indices_sorted[:, start_idx:end_idx]

            for batch_idx in range(batch_size):
                mask = create_patch_mask(
                    patches_to_perturb[batch_idx], (x_batch.shape[2], x_batch.shape[3]),
                    self.n_patches, self.patch_size
                )
                x_perturbed[batch_idx] = apply_baseline_perturbation(
                    x_perturbed[batch_idx], mask, self.perturb_baseline
                )

            predictions.append(predict_on_batch(model, x_perturbed, y_batch, device, use_softmax=True))

        # Calculate AUC for each sample
        predictions_array = np.stack(predictions, axis=1)
        return [float(np.trapezoid(predictions_array[i], dx=1)) for i in range(batch_size)]
