import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F


class HingeLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(HingeLoss, self).__init__()
        self.margin = margin

    def forward(self, reshaped_logits, labels):
        # 分数过 softmax
        reshaped_logits = F.softmax(reshaped_logits, dim=1)  # [batch, 5]
        # 得到 label 的 onehot
        oh_labels = F.one_hot(labels,num_classes=5).squeeze(1)  # [batch, 5]
        mask = oh_labels.bool()
        # pdb.set_trace()
        # 利用 mask 选出正例 
        logits_p = reshaped_logits.masked_select(mask)
        logits_p = logits_p.reshape(-1, 1)  # [batch, 1]
        logits_p = logits_p.repeat(1, 5)                   # [batch,1] -> [batch,5]
        logits_n = reshaped_logits
        logits_pn = logits_p - logits_n         # pos - neg
        oh_labels = oh_labels * 2 - 1           # 0, 1  -> -1, 1

        Loss = nn.HingeEmbeddingLoss(self.margin)
        return Loss(logits_pn, oh_labels)

