"""
Faithfulness correlation metric for attribution evaluation.

Measures Spearman correlation between attribution scores and prediction changes
when random subsets of patches are perturbed.  Higher correlation = more faithful.
"""

import numpy as np

from featuregating.experiments.faithfulness import (
    _BatchedFaithfulnessMetric,
    apply_baseline_perturbation,
    create_patch_mask,
    patch_size_for_n_patches,
    predict_on_batch,
)


class FaithfulnessCorrelation(_BatchedFaithfulnessMetric):
    """
    Standalone patch-based faithfulness correlation implementation.

    Perturbs random patch subsets, measures prediction change, and computes
    Spearman correlation with the attribution sum of those patches.
    """

    def __init__(self, n_patches=196, subset_size=20, nr_runs=50, perturb_baseline="mean"):
        self.n_patches = n_patches
        self.patch_size = patch_size_for_n_patches(n_patches)
        self.subset_size = min(subset_size, n_patches)
        self.nr_runs = nr_runs
        self.perturb_baseline = perturb_baseline

    def evaluate_batch(self, model, x_batch, y_batch, a_batch, device=None):
        """
        Compute faithfulness correlation for a batch.

        For each of ``nr_runs`` random patch subsets, perturb those patches and
        measure the prediction delta.  Then compute Spearman correlation between
        the attribution sums and prediction deltas across runs.
        """
        batch_size = a_batch.shape[0]
        if a_batch.shape[1] != self.n_patches:
            raise ValueError(f"Expected {self.n_patches} patches, got {a_batch.shape[1]}")

        # Original predictions (logits to avoid softmax saturation)
        y_pred = predict_on_batch(model, x_batch, y_batch, device, use_softmax=False)

        # Pre-generate all random patch choices
        all_patch_choices = np.stack([
            np.random.choice(self.n_patches, self.subset_size, replace=False)
            for _ in range(batch_size * self.nr_runs)
        ]).reshape(self.nr_runs, batch_size, self.subset_size)

        # Pre-compute attribution sums
        att_sums = np.array([
            a_batch[np.arange(batch_size)[:, None], all_patch_choices[run_idx]].sum(axis=1)
            for run_idx in range(self.nr_runs)
        ]).T  # (batch_size, nr_runs)

        # Compute prediction deltas for each run
        pred_deltas = []
        for run_idx in range(self.nr_runs):
            patch_choices = all_patch_choices[run_idx]
            x_perturbed = x_batch.copy()

            for batch_idx in range(batch_size):
                mask = create_patch_mask(
                    patch_choices[batch_idx], (x_batch.shape[2], x_batch.shape[3]),
                    self.n_patches, self.patch_size
                )
                x_perturbed[batch_idx] = apply_baseline_perturbation(
                    x_perturbed[batch_idx], mask, self.perturb_baseline
                )

            y_pred_perturb = predict_on_batch(model, x_perturbed, y_batch, device, use_softmax=False)
            pred_deltas.append(y_pred - y_pred_perturb)

        pred_deltas = np.stack(pred_deltas, axis=1)  # (batch_size, nr_runs)

        return self._compute_spearman_correlation(att_sums, pred_deltas).tolist()

    @staticmethod
    def _compute_spearman_correlation(a, b):
        """Compute Spearman correlation for batched 2-D arrays."""
        from scipy.stats import spearmanr

        correlations = np.zeros(a.shape[0])
        for i in range(a.shape[0]):
            if np.std(a[i]) < 1e-10 or np.std(b[i]) < 1e-10:
                correlations[i] = 0.0
            else:
                corr, _ = spearmanr(a[i], b[i])
                correlations[i] = corr if not np.isnan(corr) else 0.0
        return correlations
