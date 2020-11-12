import torch
import torch.nn as nn
import torch.nn.functional as F


class HingeLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(HingeLoss, self).__init__()
        self.margin = margin

    def forward(self, reshaped_logits, labels):
        # logits 过 softmax
        reshaped_logits = F.softmax(reshaped_logits, dim=1)
        # onehot 编码的labels  [batch, 5]
        oh_labels = F.one_hot(labels).squeeze(1)
        mask = oh_labels.bool()
        # 利用 mask 选出正例 [batch, 1]
        logits_p = reshaped_logits.masked_select(mask).reshape(-1, 1)
        logits_p.repeat(1, 5)                   # [batch,1] -> [batch,5]
        logits_n = reshaped_logits
        logits_pn = logits_p - logits_n         # pos - neg
        oh_labels = oh_labels * 2 - 1           # 0, 1  -> -1, 1

        Loss = nn.HingeEmbeddingLoss(self.margin)
        return Loss(logits_pn, oh_labels*2-1)

