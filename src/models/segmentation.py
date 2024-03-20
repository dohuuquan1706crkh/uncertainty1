import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from collections import OrderedDict

from utils import norm
from models.enet import ENet
from metrics import dice

class Segmentation(pl.LightningModule):

    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        if arch.lower() == "enet":
            self.model = ENet(
                input_channel=in_channels,
                num_classes=out_classes
            )
        elif arch.lower() == "fpn":
            self.model = smp.create_model(
                arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
            )
            
        # preprocessing parameteres for image
        # params = smp.encoders.get_preprocessing_params(encoder_name)
        # self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        # self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        # for image segmentation dice loss could be the best first choice
        self.out_classes = out_classes
        if out_classes == 1:
            self.loss_fn = smp.losses.DiceLoss(
                mode=smp.losses.BINARY_MODE,
                from_logits=True,
            )
        else:
            self.loss_fn = smp.losses.DiceLoss(
                mode=smp.losses.MULTICLASS_MODE,
                from_logits=True,
            )

        ## https://github.com/Lightning-AI/lightning/pull/16520
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.batch_loss = []

    def forward(self, image):
        # normalize image here
        # image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        
        image = batch["image"]
        mask = batch["mask"]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32, 
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of 
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have 
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        # print(h, w)
        assert h % 32 == 0 and w % 32 == 0

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)
        
        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        mask = torch.argmax(mask, dim=1)
        # mask = torch.unsqueeze(mask, dim=1)
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then 
        # apply thresholding
        if self.out_classes == 1:
            prob_mask = logits_mask.sigmoid()
            pred_mask = (prob_mask > 0.5).float()
        else:
            prob_mask = logits_mask.softmax(dim=1)
            pred_mask = torch.argmax(prob_mask, dim=1)  ## original mask (torch.Size([1, 320, 320]), tensor([0, 1, 2]))
        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        
        # ignore background
        pred_mask = pred_mask.long()
        mask = mask.long()
        # if self.config.IGNORE_BG:
        #     pred_mask -= 1
        #     mask -= 1
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask,
            mask,
            mode="binary" if self.out_classes == 1 else "multiclass",
            # ignore_index=-1 if self.config.IGNORE_BG else None,
            num_classes=None if self.out_classes == 1 else self.out_classes - 1
        )
        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image 
        # and then compute mean over these scores
        # per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        
        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset 
        # with "empty" images (images without target class) a large gap could be observed. 
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        # dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        dataset_dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            # f"{stage}_per_image_iou": per_image_iou.item(),
            f"{stage}_dataset_dice": dataset_dice.item(),
            # f"{stage}_dataset_iou": dataset_iou.item(),
        }
        print(f"\nMetrics: {metrics}")
        self.log_dict(metrics, prog_bar=False)

    def training_step(self, batch, batch_idx):
        pred = self.shared_step(batch, "train")
        self.batch_loss.append(pred["loss"].item())
        self.training_step_outputs.append(pred)
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
    
    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict