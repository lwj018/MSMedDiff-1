import torch.nn as nn
import torch.nn.functional as F

import torch


class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, pred, target):
        # 计算 MAE 损失
        mae_loss = torch.mean(torch.abs(pred - target))
        return mae_loss


def dice_loss(input, target, smooth=1e-5):
    input = torch.sigmoid(input)
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    dice=1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
    return max(0,dice)

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1., gamma=2., reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Clamp BCE_loss to avoid very large negative values which can lead to NaNs or Infs in exp
        BCE_loss = torch.clamp(BCE_loss, min=-100)  # Adjust the value based on your specific needs

        # Calculate pt with clamping for stability
        pt = torch.exp(-BCE_loss)
        pt = torch.clamp(pt, min=1e-6, max=1 - 1e-6)  # Clamp pt to prevent it from becoming too close to 0 or 1

        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


class CombinedLoss(torch.nn.Module):
    def __init__(self, mae_weight=0.8, dice_weight=0.5, focal_weight=0.2):
        super(CombinedLoss, self).__init__()
        self.mae_weight = mae_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_criterion = dice_loss
        self.focal_criterion = FocalLoss()
        self.maeloss=MAELoss()
    def forward(self, inputs, targets):
        mae_loss = self.maeloss(inputs, targets)

        # # 渐进式增加Dice和Focal Loss的权重
        # dice_weight = min(self.dice_weight * ((current_epoch - 200) / 600), self.dice_weight)
        # focal_weight = min(self.focal_weight * ((current_epoch - 200) / 600), self.focal_weight)

        #dice_loss_val = self.dice_criterion(inputs, targets)*self.dice_weight
        focal_loss_val = self.focal_criterion(inputs, targets)*self.focal_weight

        loss = (mae_loss+focal_loss_val)
        return loss
