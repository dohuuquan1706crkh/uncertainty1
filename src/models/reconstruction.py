import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from collections import OrderedDict

from metrics import dice
from utils import norm

class Reconstruction(pl.LightningModule):

    def __init__(self, model, loss_fn, config):
        super().__init__()
        self.config = config
        self.model = model
        self.loss_fn = loss_fn
        ## https://github.com/Lightning-AI/lightning/pull/16520
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.batch_loss = []
        
        if len(self.config.DATA.CLASSES) == 1:
            self.dice_loss_fn = smp.losses.DiceLoss(
                mode=smp.losses.BINARY_MODE,
                from_logits=True,
            )
        else:
            self.dice_loss_fn = smp.losses.DiceLoss(
                mode=smp.losses.MULTICLASS_MODE,
                from_logits=True,
            )

    def forward(self, yhat, x=None):
        if self.config.USE_CONTEXT:
            pred = self.model(yhat, x)
        else:
            pred = self.model(yhat)
        return pred

    def shared_step(self, batch, stage):
        
        image = batch["image"]
        mask = batch["mask"]    # [B, C, W, H] onehot
        yhat = batch["yhat"]

        ## scaling for stable training
        # yhat = norm(yhat)   ## batchnorm
        mask[mask==1] = 0.9 ## 3channel -> [0.1, 0.1, 0.8]
        mask[mask==0] = 0.1

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        # Using the logits of yhat
        if self.config.UNC_MODE == "bayescap":
            (out_mu, out_alpha, out_beta)= self.forward(yhat)
            gauss_params = out_alpha, out_beta
        elif self.config.UNC_MODE == "vae":
            # out_mu, out_logvar = self.forward(
            #     yhat=yhat,
            #     x=image if self.config.USE_CONTEXT else None
            # ) ## assuming out_mu is regression after softmax
            (out_mu, out_logvar) = self.forward(
                yhat=yhat,
                x=image if self.config.USE_CONTEXT else None
            )
            gauss_params = out_logvar
        else:
            (out_mu, out_logvar) = self.forward(
                yhat=yhat,
                x=image if self.config.USE_CONTEXT else None
            ) ## assuming out_mu is regression after softmax
            gauss_params = out_logvar

        # if len(self.config.DATA.CLASSES) == 1:
        #     out_mu = out_mu.sigmoid()
        # else:
        #     out_mu = out_mu.softmax(dim=1)
        
        loss = self.loss_fn(
            out_mu, yhat, mask, gauss_params
        )
        # if self.config.UNC_MODE == "vae":
        #     loss = loss + clip_loss

        mask = torch.argmax(mask, dim=1)
        mask = mask.long()
        
        dice_loss = self.dice_loss_fn(
            out_mu, mask
        )
        loss = loss + 2*dice_loss
     
        ## categorize
        out_mu = torch.argmax(out_mu, dim=1)
        out_mu = out_mu.long()

        batch_score = []
        for i in range(out_mu.shape[0]):
            score = dice(
                pred=out_mu[i].cpu().numpy(),
                target=mask[i].cpu().numpy(),
                labels=list(range(len(self.config.DATA.CLASSES))),
                exclude_bg=self.config.IGNORE_BG
            )
            batch_score.append(score)

        return {
            "loss": loss,
            "dice": batch_score
        }

    def shared_epoch_end(self, outputs, stage):

        # losses = torch.tensor([x["loss"] for x in outputs])
        # dataset_dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        dataset_dice = []
        for x in outputs:
            dataset_dice.extend(x["dice"])
        all_dice = np.array(dataset_dice)

        metrics = {
            f"{stage}_dice": all_dice.mean(),
            # f"{stage}_loss": losses.mean().item(),
        }
        print(f"\nMetrics: {metrics}")
        self.log_dict(metrics, prog_bar=False)

    def training_step(self, batch, batch_idx):
        pred = self.shared_step(batch, "train")
        self.training_step_outputs.append(pred)
        self.batch_loss.append(pred["loss"].item())
        return pred

    def on_train_epoch_end(self):
        return self.shared_epoch_end(self.training_step_outputs, "train")

    def validation_step(self, batch, batch_idx):
        pred = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(pred)
        return pred

    def on_validation_epoch_end(self):
        return self.shared_epoch_end(self.validation_step_outputs, "valid")

    def test_step(self, batch, batch_idx):
        pred = self.shared_step(batch, "test")
        self.test_step_outputs.append(pred)
        return pred

    def on_test_epoch_end(self):
        return self.shared_epoch_end(self.test_step_outputs, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)