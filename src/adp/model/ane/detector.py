import torch
import torch.nn as nn

from adp.model.ane.backbone import FastViTBackbone
from adp.model.ane.decoder_full import ANEDecoder
from adp.model.ane.encoder import ANEHybridEncoder


__all__ = ["ANEDetector"]


class ANEDetector(nn.Module):
    def __init__(
        self,
        backbone: FastViTBackbone,
        encoder: ANEHybridEncoder,
        decoder: ANEDecoder,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        x: torch.Tensor,
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict:
        feats = self.backbone(x)  # list of 3 (B, C_i, H_i, W_i)
        feats = self.encoder(feats)  # list of 3 (B, embed_dim, H_i, W_i)
        return self.decoder(feats, targets)

    def deploy(self) -> "ANEDetector":
        """Reparameterize all submodules for ANE export."""
        self.eval()
        self.backbone.deploy()
        for module in self.modules():
            for fn_name in ("convert_to_deploy", "switch_to_deploy", "fuse"):
                fn = getattr(module, fn_name, None)
                if callable(fn):
                    fn()
                    break
        return self
