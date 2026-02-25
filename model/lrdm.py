import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import utils
from model.unet import DiffusionUNet

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

class LRDM_Depth(nn.Module):
    def __init__(self, args, device='cuda'):
        super(LRDM_Depth, self).__init__()

        self.args = args
        self.device = device

        self.Unet = DiffusionUNet(args)

        self.conv1x1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)

        betas = get_beta_schedule(
            beta_schedule=args.beta_schedule,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            num_diffusion_timesteps=args.num_diffusion_timesteps,
        )

        self.betas = torch.from_numpy(betas).float()
        self.num_timesteps = self.betas.shape[0]

    @staticmethod
    def compute_alpha(beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def sample_training(self, low_condition_norm, b, eta=0.):
        skip = self.args.num_diffusion_timesteps // self.args.num_sampling_timesteps
        seq = range(0, self.args.num_diffusion_timesteps, skip)
        n, c, h, w = low_condition_norm.shape
        seq_next = [-1] + list(seq[:-1])
        x = torch.randn(n, c, h, w, device=self.device) 
        xs = [x]

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = self.compute_alpha(b, t.long())
            at_next = self.compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)

            # print("xt: ", xt.shape)
            et = self.Unet(torch.cat([low_condition_norm, xt], dim=1), t)
            # print("et:", et.shape)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to(x.device))

        return xs[-1]

    def forward(self, delta_log_depth, net, log_depth_lrdm):
        data_dict = {}

        # β parameter of the diffusion model
        b = self.betas.to(delta_log_depth.device)
        d2 = torch.quantile(delta_log_depth, 0.02)
        d98 = torch.quantile(delta_log_depth, 0.98)

        depth_feature = net

        depth_feature = self.conv1x1(depth_feature)

        depth_feature = utils.data_transform_depth(depth_feature)

        # diffusion step size t
        t = torch.randint(low=0, high=self.num_timesteps, size=(depth_feature.shape[0] // 2 + 1,)).to(self.device)
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:depth_feature.shape[0]].to(delta_log_depth.device)
        a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)

        # random noise
        e = torch.randn_like(depth_feature)

        delta_depth = delta_log_depth
        delta_depth = utils.data_transform_depth(delta_log_depth)

        x = delta_depth * a.sqrt() + e * (1.0 - a).sqrt()

        # noise predicter
        x = torch.nn.functional.interpolate(x, size=(64,64) , mode="nearest")
        depth_feature = torch.nn.functional.interpolate(depth_feature, size=(64,64) , mode="nearest")
        noise_output = self.Unet(torch.cat([depth_feature, x], dim=1), t.float())

        pred_depth = self.sample_training(depth_feature, b)
        pred_depth = torch.nn.functional.interpolate(pred_depth, size=(57,76) , mode="nearest")

        noise_output = torch.nn.functional.interpolate(noise_output, size=(57,76) , mode="nearest")
        pred_depth = utils.data_transform_depth_inverse(pred_depth, d2, d98)
        reference_depth = log_depth_lrdm

        data_dict["noise_output"] = noise_output
        data_dict["e"] = e
        data_dict["pred_depth"] = pred_depth
        data_dict["reference_depth"] = reference_depth

        return data_dict
