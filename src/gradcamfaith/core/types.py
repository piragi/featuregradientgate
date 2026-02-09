from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ClassificationPrediction:
    predicted_class_label: str
    predicted_class_idx: int
    confidence: float
    probabilities: List[float]




@dataclass
class AttributionDataBundle:
    positive_attribution: np.ndarray
    raw_attribution: np.ndarray

    def __eq__(self, other):
        if not isinstance(other, AttributionDataBundle):
            return NotImplemented

        return (
            np.array_equal(self.positive_attribution, other.positive_attribution) and
            np.array_equal(self.raw_attribution, other.raw_attribution)
        )


@dataclass
class AttributionOutputPaths:
    attribution_path: Path
    raw_attribution_path: Path


@dataclass
class ClassificationResult:
    image_path: Path
    prediction: ClassificationPrediction
    true_label: Optional[str] = None  # The ground truth label
    attribution_paths: Optional[AttributionOutputPaths] = None
    # Cache for efficiency - avoids reloading/retransforming for faithfulness
    _cached_tensor: Optional[np.ndarray] = None  # Preprocessed image tensor (C, H, W) as numpy
    _cached_raw_attribution: Optional[np.ndarray] = None  # Raw attribution array

    def to_dict_for_cache(self) -> Dict[str, Any]:
        data = asdict(self)
        data['image_path'] = str(data['image_path'])
        if data.get('attribution_paths'
                    ) and isinstance(data['attribution_paths'], dict):  # It will be a dict after asdict
            attrs = data['attribution_paths']
            for k_attr, v_attr in attrs.items():
                if isinstance(v_attr, Path):  # Should always be Path as per AttributionOutputPaths types
                    attrs[k_attr] = str(v_attr)
        # Don't serialize cached tensors/attributions to disk (too large)
        data.pop('_cached_tensor', None)
        data.pop('_cached_raw_attribution', None)
        return data

    @classmethod
    def from_dict_for_cache(cls, data: Dict[str, Any]) -> 'ClassificationResult':
        data['image_path'] = Path(data['image_path'])

        # Explicitly reconstruct nested dataclasses
        if 'prediction' in data and isinstance(data['prediction'], dict):
            data['prediction'] = ClassificationPrediction(**data['prediction'])

        if 'attribution_paths' in data and data['attribution_paths'] is not None:
            if isinstance(data['attribution_paths'], dict):
                attrs_data = data['attribution_paths']
                # Convert string paths back to Path objects before constructing AttributionOutputPaths
                converted_attrs_data = {k: Path(v) if isinstance(v, str) else v for k, v in attrs_data.items()}
                data['attribution_paths'] = AttributionOutputPaths(**converted_attrs_data)
            else:
                raise ValueError("attribution_paths in cache is not a dict or None")
        elif 'attribution_paths' not in data:  # if key is missing, treat as None
            data['attribution_paths'] = None

        return cls(**data)
