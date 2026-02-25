import torch
import torch.nn as nn

class NoiseLoss(nn.Module):
    def __init__(self, args):
        super(NoiseLoss, self).__init__()
        # 初始化MSELoss作为噪声损失
        self.l2_loss = nn.MSELoss()

    def forward(self, sample, output):
        # 获取输出中的噪声相关变量
        noise_output = output["noise_output"]
        e = output["e"]
        
        # 计算噪声损失
        noise_loss = self.l2_loss(noise_output, e)
        return noise_loss
