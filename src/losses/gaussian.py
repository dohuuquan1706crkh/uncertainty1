import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
import numpy as np
import math

class GenGaussLoss(nn.Module):
    def __init__(
        self, reduction='mean',
        alpha_eps = 1e-4, beta_eps=1e-4,
        resi_min = 1e-4, resi_max=1e3
    ) -> None:
        super(GenGaussLoss, self).__init__()
        self.reduction = reduction
        self.alpha_eps = alpha_eps
        self.beta_eps = beta_eps
        self.resi_min = resi_min
        self.resi_max = resi_max
	
    def forward(
        self, 
        mean: Tensor,
        one_over_alpha: Tensor,
        beta: Tensor,
        target: Tensor
    ):
        one_over_alpha1 = one_over_alpha + self.alpha_eps
        beta1 = beta + self.beta_eps

        resi = torch.abs(mean - target)
        # resi = torch.pow(resi*one_over_alpha1, beta1).clamp(min=self.resi_min, max=self.resi_max)
        resi = (resi*one_over_alpha1*beta1).clamp(min=self.resi_min, max=self.resi_max)
        # resi = resi*one_over_alpha1*beta1

        ## check if resi has nans
        if torch.sum(resi != resi) > 0:
            print('resi has nans!!')
            return None

        # one_over_alpha1 towards too small -> log_one_over_alpha towards too big negative num.
        # NOTE: all varialbles are 2D tensors
        log_one_over_alpha = torch.log(one_over_alpha1)
        log_beta = torch.log(beta1)
        lgamma_beta = torch.lgamma(torch.pow(beta1, -1))

        if torch.sum(log_one_over_alpha != log_one_over_alpha) > 0:
            print('log_one_over_alpha has nan')
        if torch.sum(lgamma_beta != lgamma_beta) > 0:
            print('lgamma_beta has nan')
        if torch.sum(log_beta != log_beta) > 0:
            print('log_beta has nan')
		
        l = resi - log_one_over_alpha + lgamma_beta - log_beta

        if self.reduction == 'mean':
            return torch.abs(l.mean())
        elif self.reduction == 'sum':
            return l.sum()
        else:
            print('Reduction not supported')
            return None

class NormGaussLoss(nn.Module):
    def __init__(
        self,
        logvar_eps=1e-4,
        resi_min=1e-4,
        resi_max=1e3,
        reduction='mean',
    ) -> None:
        super(NormGaussLoss, self).__init__()
        self.reduction = reduction
        self.logvar_eps = logvar_eps
        self.resi_min = resi_min
        self.resi_max = resi_max
    def forward(
        self, 
        mean: Tensor,
        target: Tensor,
        one_over_sigma: Tensor,  ## logvar
    ):
        # square_sigma = sigma + self.sigma_eps
        one_over_sigma = one_over_sigma + self.logvar_eps
        resi = torch.abs(mean - target)
        # print(f"mean: {mean.shape} | one_over_sigma: {one_over_sigma.shape}")
        resi = torch.pow(resi*one_over_sigma, 2).clamp(min=self.resi_min, max=self.resi_max)
        if torch.sum(resi != resi) > 0:
            print('resi has nans!!')
            return None
        log_sigma_square = torch.log(torch.pow(1/one_over_sigma,2))
        # log_sigma = torch.log(1/one_over_sigma)

        if torch.sum(log_sigma_square != log_sigma_square) > 0:
            print('log_sigma_square has nan')
        # l = 0.5 * (resi + log_sigma_square + math.log(2 * math.pi))
        l = 0.5 * (resi + log_sigma_square)
        if self.reduction == 'mean':
            return l.mean()
        elif self.reduction == 'sum':
            return l.sum()
        else:
            print('Reduction not supported')
            return None