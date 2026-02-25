import torch
import torch.nn as nn

class SCCLoss(nn.Module):
    def __init__(self, args):
        super(SCCLoss, self).__init__()
        # 初始化L1Loss作为SCC损失
        self.l1_loss = nn.L1Loss()

    def forward(self, sample, output):
        # 获取输出中的特征变量
        pred_depth = output["pred_depth"]
        reference_depth = output["reference_depth"]
        
        # 计算SCC损失
        scc_loss = self.l1_loss(pred_depth, reference_depth)
        return scc_loss
