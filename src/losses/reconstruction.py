import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
import numpy as np
import math
from src.losses.gaussian import GenGaussLoss, NormGaussLoss

# from src.losses.gaussian import GenGaussLoss, NormGaussLoss
def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device
def loglikelihood_loss(y, alpha, device=None):
    if not device:
        device = get_device()
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)

    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood

def kl_divergence(alpha, num_classes, device=None):
    if not device:
        device = get_device()
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    kl=torch.zeros([alpha.shape[0],1],device=device)
    for i in range(alpha.shape[0]):
      first_term = (
          torch.lgamma(sum_alpha[i])
          - torch.lgamma(alpha[i].unsqueeze(0)).sum(dim=1, keepdim=True)
          + torch.lgamma(ones).sum(dim=1, keepdim=True)
          - torch.lgamma(ones.sum(dim=1, keepdim=True))
      )
      second_term = (
          (alpha[i] - ones)
          .mul(torch.digamma(alpha[i]) - torch.digamma(sum_alpha[i]))
          .sum(dim=1, keepdim=True)
      )
      kl[i]= first_term + second_term
    return kl

class RecLoss(nn.Module):
    def __init__(
        self,
        reduction='mean',
        alpha_eps = 1e-4,
        beta_eps=1e-4,
        resi_min = 1e-4,
        resi_max=1e3,
        gamma=0.1,
        unc_mode:str="bayescap"
    ) -> None:
        super(RecLoss, self).__init__()
        self.reduction = reduction
        self.alpha_eps = alpha_eps
        self.beta_eps = beta_eps
        self.resi_min = resi_min
        self.resi_max = resi_max
        self.gamma = gamma
        self.unc_mode = unc_mode
        self.L_l1 = nn.L1Loss(reduction=self.reduction)
    def forward(
        # self,
        # mean: Tensor,   ## before sm batch x C x size
        # yhat: Tensor, ## yhat
        # mask: Tensor, ## mask i.e y
        # gauss_params
        # ):
        # # out_mu, yhat, mask, gauss_params
        # y_predict=mean/torch.sum(mean,dim=1).unsqueeze(1)
        # y_hat=yhat.permute(0, 2,3, 1)        
        # y_hat=y_hat.reshape(-1, y_hat.shape[3])
        # y_predict=y_predict.permute(0, 2,3, 1)
        # y_predict=y_predict.reshape(-1, y_predict.shape[3])
        # y=mask.permute(0, 2,3, 1)  
        # y=y.reshape(-1, y.shape[3])
        # ## mean and yhat is before softmax
        # l1 = self.L_l1(y, y_hat)
        # l2 = loglikelihood_loss(y,y_predict)
        # kl_alpha = (y_predict - 1) * (1 - y) + 1
        # l3 = kl_divergence(kl_alpha, y_predict.shape[1])
        # l = l1 + l2 +l3
        # return torch.sum(l)
        self,
        mean: Tensor,   ## before sm
        yhat: Tensor, ## yhat
        mask: Tensor, ## mask
        gauss_params
    ):  # out_mu, yhat, mask, gauss_params
        
        ## mean and yhat is before softmax
        l1 = self.L_l1(mean, yhat)

        ##target1 is the base model output for identity mapping
        ##target2 is the ground truth for the GenGauss loss
        
        ## need to convert from onehot [B, 3, W, H] -> [B, 1, W, H]

        mean = mean.softmax(dim=1)
        # mean = torch.argmax(mean, dim=1)
        # mean = torch.unsqueeze(mean, dim=1)

        # yhat = torch.argmax(yhat, dim=1)
        # yhat = torch.unsqueeze(yhat, dim=1)
        # yhat = yhat.float()

        # mask = torch.argmax(mask, dim=1)
        # mask = torch.unsqueeze(mask, dim=1)
        # mask = mask.float()

        if self.unc_mode == "bayescap":
            one_over_alpha, beta = gauss_params
            l2 = self.L_Gauss(mean, one_over_alpha, beta, mask)
        elif self.unc_mode in ["bnn", "vae", "teacher-student"]:
            sigma = gauss_params
            l2 = self.L_Gauss(mean, mask, sigma)
        else:
            raise(f"{self.unc_mode} doesnt implement")

        # l = self.gamma*l1 + (1-self.gamma)*l2
        # l = 2*self.gamma*l1 + self.gamma*l2
        l = l1 + self.gamma*l2
        # l = (1+torch.exp(-l2))*l1 + (self.gamma+torch.exp(-l1))*l2
        # l = (1+torch.exp(-l2))*l1 + torch.exp(-l1)*l2
        # l = l1 + torch.exp(l2)*l2
        return l
if __name__ == "__main__":
    x1 = torch.randn(4,3,5,5)
    x2 = torch.rand(4,3,5,5)
    x3 = torch.rand(4,3,5,5)
    x4 = torch.randn(4,3,5,5)
    L =  RecLoss(alpha_eps=1e-4, beta_eps=1e-4, resi_min=1e-4, resi_max=1e0, gamma=1., unc_mode="bnn")
    print(L(x1, x2, x3, x4))
