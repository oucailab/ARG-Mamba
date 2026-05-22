import torch
import torch.nn as nn
import torch.nn.functional as F

class OhemCrossEntropyLoss(nn.Module):
    """
    OHEM (Online Hard Example Mining) Cross Entropy Loss
    """
    def __init__(self, ignore_index=-1, thresh=0.7, min_kept=100000, weight=None):
        super(OhemCrossEntropyLoss, self).__init__()
        self.thresh = thresh
        self.min_kept = min_kept
        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index,
            reduction='none'  # 关键：计算每个像素的损失，而不是求平均
        )

    def forward(self, score, target):
        # 1. 计算每个像素的原始损失
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(h, w), mode='bilinear', align_corners=False)
        
        loss = self.criterion(score, target)
        
        # 2. 根据阈值筛选困难样本
        #    首先计算每个像素属于其正确类别的概率
        pred = F.softmax(score, dim=1)
        # --- 修改这里：在 gather 之前处理 ignore_index ---
        # 创建一个 target 的副本，用于 gather 操作
        tmp_target = target.clone()
        # 将副本中的 ignore_index 替换为 0（一个安全的索引）
        tmp_target[tmp_target == self.ignore_index] = 0
        # 使用处理过的 tmp_target 进行 gather
        pred = pred.gather(1, tmp_target.unsqueeze(1)).squeeze(1)
        # --- 修改结束 ---
        #    创建一个掩码，标记所有未被忽略的像素
        mask = target.contiguous().view(-1) != self.ignore_index
        
        #    根据阈值筛选出困难样本 (概率低于阈值)
        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_index] = 0
        pred = pred.contiguous().view(-1)
        
        #    将损失和概率展平
        loss = loss.contiguous().view(-1)
        
        #    筛选出有效的、且预测概率低于阈值的困难样本
        hard_mask = (pred < self.thresh) & mask
        
        # 3. 确保至少保留 min_kept 个样本
        if hard_mask.sum() < self.min_kept:
            # 如果困难样本太少，则选择损失最高的 min_kept 个样本
            loss_sorted, _ = loss[mask].sort(descending=True)
            min_kept_loss = loss_sorted[min(self.min_kept, len(loss_sorted)-1)]
            hard_mask = (loss >= min_kept_loss) & mask

        # 4. 只对困难样本计算最终损失
        final_loss = loss[hard_mask].mean()
        
        return final_loss
