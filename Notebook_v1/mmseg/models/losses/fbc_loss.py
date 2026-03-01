# mmseg/models/losses/sim_contrast_loss.py
#https://github.com/cys1102/FBA-Net/blob/main/code/utils/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from mmseg.models.losses.utils import weight_reduce_loss
from typing import List, Tuple

import os
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import numpy as np
from sklearn.decomposition import PCA



@MODELS.register_module()
class FBContrastLoss(nn.Module):
    def __init__(self, in_dim=256, temperature=0.07, loss_name='loss_fb_contrast'):
        super(FBContrastLoss, self).__init__()
        self.temperature = temperature
        self._loss_name = loss_name

    def extract_fg_bg_all_upsample(
        self, feat: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        不采样，直接对前景和背景区域做平均池化（mean pooling）。

        feat: [B, C, H, W]
        mask: [B, H, W] or [B, 1, H, W]
        return: fg_feat: [B, C], bg_feat: [B, C]
        """
        B, C, H, W = feat.shape

        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # [B, 1, H, W]

        feat = F.interpolate(feat, size=mask.shape[2:], mode='bilinear', align_corners=False)
        mask = mask.float()  # [B, 1, H, W]

        fg_mask = (mask > 0).float()  # 前景: 1, 背景: 0
        bg_mask = 1.0 - fg_mask       # 背景: 1, 前景: 0

        # 防止除以 0
        eps = 1e-6
        fg_feat = (feat * fg_mask).sum(dim=[2, 3]) / (fg_mask.sum(dim=[2, 3]) + eps)  # [B, C]
        bg_feat = (feat * bg_mask).sum(dim=[2, 3]) / (bg_mask.sum(dim=[2, 3]) + eps)  # [B, C]

        # # 可选：MLP投影
        # fg_mlp = self.f_mlp(fg_feat)  # [B, dim]
        # bg_mlp = self.f_mlp(bg_feat)  # [B, dim]

        return fg_feat, bg_feat

    def forward(self, support_feats, support_labels,query_label, query_logits, query_feat):
        # support：用真掩码提取前/背景特征
        fg_s, bg_s = self.extract_fg_bg_all_upsample(support_feats, support_labels)
        fg_q, bg_q = self.extract_fg_bg_all_upsample(query_feat, query_label)

        # 计算对比损失
        return self.contrastive_loss(fg_s, bg_s, fg_q, bg_q)

    def contrastive_loss(self, fg1, fg2, bg1, bg2):
        # Normalize all features
        fg1, fg2, bg1, bg2 = map(lambda x: F.normalize(x, dim=1), [fg1, fg2, bg1, bg2])
        B, C = fg1.shape

        # # Reshape to [B, 1, C] to match图中形式
        fg1 = fg1.unsqueeze(1)  # [B, 1, C]
        fg2 = fg2.unsqueeze(1)
        bg1 = bg1.unsqueeze(1)
        bg2 = bg2.unsqueeze(1)
        # bg1 as anchor, 正: bg2，负: fg1
        group1 = torch.cat([bg2, fg1], dim=1)
        # # fg2 as anchor, 正: fg1，负: bg2
        group2 = torch.cat([fg1, bg2], dim=1)
        # fg1 as anchor,
        
        # # Build candidate groups [B, 3, C] with positive always at index 0
        # group1 = torch.cat([bg2, fg1, fg2], dim=1)  # bg1 as anchor
        # group2 = torch.cat([fg2, bg1, bg2], dim=1)  # fg1 as anchor
        # group3 = torch.cat([bg1, fg1, fg2], dim=1)  # bg2 as anchor
        # group4 = torch.cat([fg1, bg1, bg2], dim=1)  #fg2 as anchor

        # # # 第一组：bg1 vs [bg2, fg1, fg2]
        sim1 = F.cosine_similarity(bg1, group1)/ self.temperature  # [B, 3]
        label1 = torch.zeros(B, dtype=torch.long, device=sim1.device)  # index 0 is pos (bg2)
        loss1 = F.cross_entropy(sim1, label1)

        # 第二组：fg1 vs [fg2, bg1, bg2]
        sim2 = F.cosine_similarity(fg2, group2)/ self.temperature  # [B, 3]
        label2 = torch.zeros(B, dtype=torch.long, device=sim2.device)  # index 0 is pos (fg2)
        loss2 = F.cross_entropy(sim2, label2)

        # # 第三组：bg2 vs [bg1, fg1, fg2]
        # sim3 = F.cosine_similarity(bg2, group3) / self.temperature  # [B, 3]
        # label3 = torch.zeros(B, dtype=torch.long, device=sim3.device)  # 正例是 index 0 (bg1)
        # loss3 = F.cross_entropy(sim3, label3)

        # # 第四组：fg2 vs [fg1, bg1, bg2]
        # sim4 = F.cosine_similarity(fg2, group4) / self.temperature  # [B, 3]
        # label4 = torch.zeros(B, dtype=torch.long, device=sim4.device)  # 正例是 index 0 (fg1)
        # loss4 = F.cross_entropy(sim4, label4)


        # 合并
        loss = (loss1+loss2)/ 2
        return 0.1*loss

    @property
    def loss_name(self):
        return self._loss_name


