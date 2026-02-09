"""
Model adapters for the experiments layer.

Wraps model objects so they conform to the standard inference interface
(model(input) → logits) expected by faithfulness and SaCo metrics.
"""

import torch

from gradcamfaith.models.clip_classifier import CLIPClassifier


class CLIPModelWrapper(torch.nn.Module):
    """
    Wrapper that makes a CLIP classifier behave like a standard model for inference.

    This is needed because the attribution binning code expects to call model(input)
    directly, but CLIP needs the full classifier with text embeddings.
    """

    def __init__(self, clip_classifier: CLIPClassifier):
        super().__init__()
        self.clip_classifier = clip_classifier
        self.training = False  # Always in eval mode for attribution

    def eval(self):
        """Set to evaluation mode (no-op, always in eval)."""
        self.training = False
        return self

    def train(self, mode: bool = True):
        """Prevent training mode."""
        if mode:
            raise ValueError("CLIPModelWrapper should not be used in training mode")
        self.training = False
        return self

    def to(self, device):
        # move underlying models if they exist
        if hasattr(self.clip_classifier, "vision_model"):
            self.clip_classifier.vision_model.to(device)
        if hasattr(self.clip_classifier, "text_model") and self.clip_classifier.text_model is not None:
            self.clip_classifier.text_model.to(device)
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass that returns logits."""
        result = self.clip_classifier.forward(images, requires_grad=False)
        return result["logits"]

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass that returns logits."""
        return self.forward(images)

    @property
    def cfg(self):
        """Access to model configuration through vision model."""
        if hasattr(self.clip_classifier.vision_model, 'cfg'):
            return self.clip_classifier.vision_model.cfg
        return None
