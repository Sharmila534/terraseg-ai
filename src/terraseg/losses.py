import torch
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6): super().__init__(); self.smooth=smooth
    def forward(self, logits, targets):
        probs=torch.sigmoid(logits).flatten(1); targets=targets.flatten(1)
        dice=(2*(probs*targets).sum(1)+self.smooth)/(probs.sum(1)+targets.sum(1)+self.smooth)
        return 1-dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self): super().__init__(); self.bce=nn.BCEWithLogitsLoss(); self.dice=DiceLoss()
    def forward(self, logits, targets): return self.bce(logits,targets)+self.dice(logits,targets)


def build_loss(name, positive_weight=15.0, device="cpu"):
    if name == "bce": return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([positive_weight],device=device))
    if name == "dice": return DiceLoss()
    if name == "bce_dice": return BCEDiceLoss()
    raise ValueError(f"Unknown loss: {name}")
