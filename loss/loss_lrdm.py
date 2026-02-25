from . import BaseLoss
import torch

class Loss_LRDM(BaseLoss):
    def __init__(self, args):
        super(Loss_LRDM, self).__init__(args)

        # 记录loss名称
        self.loss_name = []
        for k, _ in self.loss_dict.items():
            self.loss_name.append(k)

    def compute(self, sample, output):
        loss_val = []

        for loss_type in self.loss_dict:
            loss = self.loss_dict[loss_type]
            loss_func = loss['func']

            if loss_func is None:
                continue

            # 计算对应的Noise Loss和SCC Loss
            if loss_type == "Noise":
                noise_loss = loss_func(sample, output)
                loss_val.append(noise_loss)
            elif loss_type == "SCC":
                scc_loss = loss_func(sample, output)
                loss_val.append(scc_loss)
            else:
                raise NotImplementedError(f"Loss type {loss_type} is not implemented.")

        loss_val = torch.stack(loss_val)

        # 计算总loss
        loss_sum = torch.sum(loss_val, dim=0, keepdim=True)

        # 格式化输出
        loss_val = torch.cat((loss_val, loss_sum))
        loss_val = torch.unsqueeze(loss_val, dim=0).detach()

        return loss_sum, loss_val
