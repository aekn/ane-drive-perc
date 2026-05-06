"""D-FINE detection model.

Vendored from https://github.com/ArgoHA/D-FINE-seg (Apache-2.0), which itself
is derived from https://github.com/Peterande/D-FINE (Apache-2.0). See
THIRD_PARTY_LICENSES/D-FINE-seg-LICENSE for the original license text.
"""

from adp.model.dfine.dfine import DFINE, build_loss, build_model, build_optimizer
from adp.model.dfine.utils import ensure_pretrained, load_tuning_state

__all__ = [
    "DFINE",
    "build_model",
    "build_loss",
    "build_optimizer",
    "ensure_pretrained",
    "load_tuning_state",
]
