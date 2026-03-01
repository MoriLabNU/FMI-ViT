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

# @MODELS.register_module()
# class ContraModule(nn.Module):
#     def __init__(self, channel, dim=128):
#         super(ContraModule, self).__init__()

#         self.activation_head = nn.Conv2d(channel, channel, 3, padding=1, stride=2, bias=False)
#         self.f_mlp = nn.Sequential(nn.Linear(7*7, dim), nn.ReLU())
#         self.b_mlp = nn.Sequential(nn.Linear(7*7, dim), nn.ReLU())

#     def forward(self, x):
#         # x: feature maps (output of U-Net)
#         ccam = torch.sigmoid(self.activation_head(x))
#         N, C, H, W = ccam.size()

#         ccam_ = ccam.reshape(N, C, H * W)  # [N, C, H*W]
#         fg_feats = ccam_ / (H * W)  # [N, C, H*W]
#         bg_feats = (1 - ccam_) / (H * W)  # [N, C, H*W]

#         fg_feats = self.f_mlp(fg_feats)
#         bg_feats = self.b_mlp(bg_feats)
#         return fg_feats.reshape(x.size(0), -1), bg_feats.reshape(x.size(0), -1)


@MODELS.register_module()
class FBContrastLoss(nn.Module):
    def __init__(self, in_dim=256, temperature=0.07, loss_name='loss_fb_contrast'):
        super(FBContrastLoss, self).__init__()
        self.temperature = temperature
        self._loss_name = loss_name
        #self.contra_module = ContraModule(channel=256, dim=128)
        # self.f_mlp = nn.Sequential(nn.Linear(128*256, 256), nn.ReLU())
        # self.b_mlp = nn.Sequential(nn.Linear(128*256, 256), nn.ReLU())


    def save_debug_image(self, tensor, save_path, normalize=True):
        """
        tensor: [1, H, W] or [H, W] or [C, H, W]
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)  # [H, W]
        if tensor.ndim == 3 and tensor.shape[0] == 3:
            img = TF.to_pil_image(tensor)
        else:
            arr = tensor.detach().cpu().numpy()
            if normalize:
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
            plt.imsave(save_path, arr, cmap='gray')

    def save_feature_pca_rgb(self,feat: torch.Tensor, save_path):
        """
        将 [C, H, W] 的特征图做 PCA 降维为单通道后，使用 colormap 可视化并保存为 RGB 图。

        Args:
            feat: torch.Tensor, shape [C, H, W]
            save_path: str, 保存路径
            cmap: str, matplotlib 的 colormap 名称，如 'viridis', 'plasma', 'inferno' 等
        """
        import numpy as np
        from sklearn.decomposition import PCA
        from matplotlib import cm
        from PIL import Image
        import os

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        feat = feat.detach().cpu().numpy()  # [C, H, W]
        C, H, W = feat.shape
        feat = feat.reshape(C, -1).T  # [H*W, C]

        # PCA 降维为单通道
        pca = PCA(n_components=1)
        feat_pca = pca.fit_transform(feat)  # [H*W, 1]
        feat_pca = feat_pca.reshape(H, W)

        # 归一化到 [0, 1]
        feat_pca -= feat_pca.min()
        feat_pca /= (feat_pca.max() + 1e-6)

        feat_pca = 1.0 - feat_pca

        # 使用 colormap 映射为 RGB
        cmap='viridis'
        colormap = cm.get_cmap(cmap)
        feat_rgb = colormap(feat_pca)[:, :, :3]  # [H, W, 3] 去掉 alpha 通道
        feat_rgb = (feat_rgb * 255).astype(np.uint8)

        img = Image.fromarray(feat_rgb)
        img.save(save_path)
    def extract_fg_bg_topk(self,
        feat: torch.Tensor,
        fg_weight: torch.Tensor,
        bg_weight: torch.Tensor,
        topk: float = 0.2
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从前景和背景的 soft mask 中提取 top-k 最前景和最背景区域用于对比。

        Args:
            feat: Tensor [B, C, H, W] - 特征图
            fg_weight: Tensor [B, 1, H, W] - 前景概率图
            bg_weight: Tensor [B, 1, H, W] - 背景概率图
            topk: float, 0~1 - 取前景/背景 top-k 比例区域

        Returns:
            fg_feat: Tensor [B, C] - 每个 batch 样本的前景区域平均特征
            bg_feat: Tensor [B, C] - 每个 batch 样本的背景区域平均特征
        """
        B, C, H, W = feat.shape
        N = H * W
        feat_flat = feat.view(B, C, -1)          # [B, C, N]
        fg_weight_flat = fg_weight.view(B, -1)   # [B, N]
        bg_weight_flat = bg_weight.view(B, -1)   # [B, N]

        k = int(N * topk)

        # 取 top-k 前景位置
        fg_topk_vals, fg_topk_idxs = torch.topk(fg_weight_flat, k, dim=1)
        # 取 top-k 背景位置
        bg_topk_vals, bg_topk_idxs = torch.topk(bg_weight_flat, k, dim=1)

        # 按索引提取对应特征，并在特征维度上求平均
        fg_feat = torch.stack([
            feat_flat[b, :, fg_topk_idxs[b]].mean(dim=1) for b in range(B)
        ], dim=0)  # [B, C]

        bg_feat = torch.stack([
            feat_flat[b, :, bg_topk_idxs[b]].mean(dim=1) for b in range(B)
        ], dim=0)  # [B, C]

        return fg_feat, bg_feat
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
    # def generate_mask_from_prototype(self, query_feat: torch.Tensor, fg_prototype: torch.Tensor) -> torch.Tensor:
    #     """
    #     通过与前景 prototype 的余弦相似度，生成伪前景 soft mask。

    #     Args:
    #         query_feat: [B, C, H, W]
    #         fg_prototype: [B, C]

    #     Returns:
    #         pseudo_mask: [B, 1, H, W]，相似度图
    #     """
    #     B, C, H, W = query_feat.shape
    #     query_flat = query_feat.view(B, C, -1)                  # [B, C, HW]
    #     query_flat = F.normalize(query_flat, dim=1)             # normalize query feat

    #     fg_proto = F.normalize(fg_prototype, dim=1).unsqueeze(2)  # [B, C, 1]
    #     sim = torch.bmm(fg_proto.transpose(1, 2), query_flat)   # [B, 1, HW]
    #     sim = sim.view(B, 1, H, W)                              # [B, 1, H, W]

    #     # Normalize to [0, 1]
    #     sim = (sim - sim.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]) / \
    #         (sim.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-6)

    #     return sim  # 越大越像前景

    def forward(self, support_feats, support_labels,query_label, query_logits, query_feat):
        # support：用真掩码提取前/背景特征
        fg_s, bg_s = self.extract_fg_bg_all_upsample(support_feats, support_labels)
        fg_q, bg_q = self.extract_fg_bg_all_upsample(query_feat, query_label)


        # f_prob = self.generate_mask_from_prototype(query_feat, fg_s)
        # b_prob = self.generate_mask_from_prototype(query_feat, bg_s)

        # fg_q, bg_q = self.extract_fg_bg_topk(query_feat, f_prob, b_prob)

        # mask = query_logits.max(1)[1]  # [B, H, W]
        # fg_q, bg_q = self.extract_fg_bg_all_upsample(query_feat, mask)

        # #===== 保存可视化（带 resize） =====
        # save_dir = "./debug_vis"
        # os.makedirs(save_dir, exist_ok=True)

        # # # 原始支持标签（GT）
        # self.save_debug_image(support_labels[0], f"{save_dir}/support_gt_mask.png")
        # self.save_debug_image(query_label[0], f"{save_dir}/query_gt_mask.png")

        # # #mask = mask.unsqueeze(1)  # [B, 1, H, W]
        # # f_prob = F.interpolate(f_prob, size=support_labels.shape[-2:], mode='bilinear', align_corners=False)
        # # #mask = mask.squeeze(1)    # 回到 [B, H, W]，如果后续只需要这个格式
        # # self.save_debug_image(f_prob[0], f"{save_dir}/query_mask.png")

        # # # 将 support_feats[0] 上采样后做 PCA RGB 可视化
        # feat_s = F.interpolate(support_feats, size=support_labels.shape[-2:], mode='bilinear', align_corners=False)[0]
        # self.save_feature_pca_rgb(feat_s, f"{save_dir}/support_feat_pca_rgb.png")

        # # query_feat 同理
        # feat_q = F.interpolate(query_feat, size=support_labels.shape[-2:], mode='bilinear', align_corners=False)[0]
        # self.save_feature_pca_rgb(feat_q, f"{save_dir}/query_feat_pca_rgb.png")

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


