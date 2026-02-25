from . import BaseLoss
import torch

class Loss_All(BaseLoss):
    def __init__(self, args):
        super(Loss_All, self).__init__(args)

        self.loss_name = []
        self.t_valid = 0.0001

        for k, _ in self.loss_dict.items():
            self.loss_name.append(k)

        # self.grad_loss_weight = args.grad_loss_weight
        self.intermediate_loss_weight = args.intermediate_loss_weight

    def compute(self, sample, output):
        loss_val = []

        for loss_type in self.loss_dict:
            loss = self.loss_dict[loss_type]
            loss_func = loss['func']

            if loss_func is None:
                continue

            loss_tmp = 0.0
            if loss_type in ['SeqL1', 'SeqL2']:
                all_depth_pred = output['pred_inter']
                gt_depth = sample['gt']
                loss_tmp += loss_func(all_depth_pred, gt_depth) * 1.0

                if self.intermediate_loss_weight > 0.0:
                    all_depth_pred_init = output['depth_predictions_up_initial']
                    loss_tmp += loss_func(all_depth_pred_init, gt_depth) * self.intermediate_loss_weight

            elif loss_type == 'SeqGradL1':
                all_grad_pred = output['log_depth_grad_inter']
                loss_tmp += loss_func(all_grad_pred, gt_depth) * 1.0

            elif loss_type == 'Noise':
                noise_loss = loss_func(sample, output)
                loss_tmp += noise_loss

            elif loss_type == 'SCC':
                scc_loss = loss_func(sample, output)
                loss_tmp += scc_loss

            else:
                raise NotImplementedError(f"Loss type {loss_type} is not implemented.")

            loss_tmp = loss['weight'] * loss_tmp
            loss_val.append(loss_tmp)

        loss_val = torch.stack(loss_val)

        loss_sum = torch.sum(loss_val, dim=0, keepdim=True)


        loss_val = torch.cat((loss_val, loss_sum))
        loss_val = torch.unsqueeze(loss_val, dim=0).detach()

        return loss_sum, loss_val
