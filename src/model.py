"""
Multi-modal (SAR + SAR-footprint + Optical + Optical-footprint)
late-fusion damage classifier for the QuickQuakeBuildings dataset.

Architecture follows the QQB paper (Sun et al., 2024): four independent
ResNet18 encoders (one per modality), each producing a 128-d embedding,
concatenated or gated and passed through a small MLP head to a single logit
(binary: damaged vs intact). The SAR branch can start from SAR-HUB
(TerraSAR-X-pretrained) weights instead of ImageNet, since SAR pixel
statistics are very different from RGB photos and ImageNet init transfers
poorly to them.
"""
import torch
import torch.nn as nn
from torchvision import models

import config

CLASSES = config.CLASSES


def _resnet18_branch(pretrained_weights_path=None, imagenet=True, out_dim=128):
    if pretrained_weights_path:
        net = models.resnet18(weights=None)
        ckpt = torch.load(pretrained_weights_path, map_location="cpu")
        ckpt.pop("fc.weight", None)
        ckpt.pop("fc.bias", None)
        net.load_state_dict(ckpt, strict=False)
    else:
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if imagenet else None)
    net.fc = nn.Linear(512, out_dim)
    return net


class MultiModalDamageClassifier(nn.Module):
    """mode='all': fuse all 4 modalities. mode='sar': SAR+SARftp only.
    mode='opt': optical+optftp only. Branches are keyed in a ModuleDict so
    gradcam.py can hook any single branch's layer4 by name.
    fusion='concat' (default) or 'gated' (learned per-branch scalar gate)."""

    def __init__(self, mode="all", sar_pretrain=None, embed_dim=128, fusion="concat"):
        super().__init__()
        self.mode = mode
        self.fusion = fusion
        self.embed_dim = embed_dim
        self.last_gates = None

        branches = {}
        if mode in ("all", "sar"):
            branches["sar"] = _resnet18_branch(sar_pretrain, imagenet=False, out_dim=embed_dim)
            branches["sarftp"] = _resnet18_branch(None, imagenet=True, out_dim=embed_dim)
        if mode in ("all", "opt"):
            branches["opt"] = _resnet18_branch(None, imagenet=True, out_dim=embed_dim)
            branches["optftp"] = _resnet18_branch(None, imagenet=True, out_dim=embed_dim)
        self.branches = nn.ModuleDict(branches)

        n_modalities = len(branches)
        if self.fusion == "gated":
            self.gate = nn.Linear(embed_dim * n_modalities, n_modalities)
            head_in = embed_dim
        else:
            head_in = embed_dim * n_modalities

        self.head = nn.Sequential(
            nn.Linear(head_in, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, **inputs):
        # inputs: whichever of sar/sarftp/opt/optftp this mode needs
        embeds = [self.branches[k](inputs[k]) for k in self.branches]
        if self.fusion == "gated":
            concat_embeds = torch.cat(embeds, dim=1)
            gates = torch.softmax(self.gate(concat_embeds), dim=1)
            self.last_gates = gates.detach()
            stacked_embeds = torch.stack(embeds, dim=1)  # (batch, n_branches, embed_dim)
            fused = (gates.unsqueeze(-1) * stacked_embeds).sum(dim=1)  # (batch, embed_dim)
            return self.head(fused).squeeze(1)
        else:
            combined = torch.cat(embeds, dim=1)
            self.last_gates = None
            return self.head(combined).squeeze(1)  # raw logit, shape (batch,)
