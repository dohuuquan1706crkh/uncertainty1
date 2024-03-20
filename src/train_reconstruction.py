import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
import os
import pytorch_lightning as pl
from pprint import pprint
import argparse
from utils import MyProgressBar, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import random
import shutil
from datetime import datetime, date
from tqdm import tqdm
import numpy as np
import json

from transforms import (
    get_training_augmentation,
    get_preprocessing,
    get_validation_augmentation
)

from dataset_seed import Dataset, ReconsDataset

from load_cfg import load_config

from models.bayesian import BayesCap, BNN, GaussCapContext
from models.vae import VAE, ConvAutoencoder
from models.custom_fpn import FPN

from models.segmentation import Segmentation
from models.reconstruction import Reconstruction

from losses.reconstruction import RecLoss
from utils import seed_everything, load_checkpoint
from metrics import (
    get_uncertainty_map,
    get_error_map,
    get_confidence,
    get_accuracy,
    get_ece,
    get_mutual_information,
    get_correlation
)
def save_checkpoint(model, optimizer, epoch, loss, save_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, save_path)

def load_checkpoint(model, optimizer, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss
def main(config, run_mode:str="train"):
    print(f"🥝 Start Running Reconstruction Network in {run_mode} mode")
    seed_everything(seed=config.SEED_NUM)
    ## random seed 
    ## DEBUG: change the setting
    total_ids=list(range(1,501))
    random.shuffle(total_ids)
    train_ids = total_ids[:400]
    valid_ids = total_ids[400:450]
    test_ids = total_ids[450:]
    print(f"🥝 Verify seed num: {config.SEED_NUM} \n -Train: {train_ids[:5]} \n -Valid: {valid_ids[:5]} \n -Test: {test_ids[:5]}")

    # Segmentation Component
    seg_model = Segmentation(
        arch=config.MODEL.ARCH,
        encoder_name=config.MODEL.ENCODER,
        in_channels=1 if config.DATA.SPACE=="gray" else 3,  
        out_classes=len(config.DATA.CLASSES),
    )

    ckpt_dir = os.path.join(
        config.SAVE_DIR, f"seed_{config.SEED_NUM}"
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_files = os.listdir(ckpt_dir)
    for f in ckpt_files:
        if f.startswith("frozen") and config.MODEL.FROZEN_CKPT == "":
            config.MODEL.FROZEN_CKPT = os.path.join(ckpt_dir, f)
        elif f.startswith(f"recons_{config.EXP_NAME}") and config.MODEL.RECONS_CKPT == "":
            if run_mode == "train":
                print(f"🧨 There is an existing ckpt...")
                os.remove(os.path.join(ckpt_dir, f))
                print(f"🧨 Removed {f}")
            else:
                config.MODEL.RECONS_CKPT = os.path.join(ckpt_dir, f)
    print(f"Loading the frozen checkpoint: {config.MODEL.FROZEN_CKPT}")
    ckpt = torch.load(config.MODEL.FROZEN_CKPT, map_location="cpu")
    seg_model.load_state_dict(ckpt["state_dict"]) ## lightning state_dict, naive model

    ## init the preprocessing and augmentation module
    if config.DATA.SPACE == "rgb":
        preprocessing_fn = smp.encoders.get_preprocessing_fn(
            config.MODEL.ENCODER, config.MODEL.ENCODER_WEIGHTS
        )
    ## gray image needs a custom preprocessing_fn
    else:
        preprocessing_fn = None
    print(f"preprocessing_fn: {preprocessing_fn}")
    train_dataset = ReconsDataset(
        config.DATA.ROOT, 
        train_ids,
        seg_model,
        config.DEVICE,
        augmentation=get_training_augmentation(config.DATA.SPACE), 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=config.DATA.CLASSES,
        input_space=config.DATA.SPACE
    )

    valid_dataset = ReconsDataset(
        config.DATA.ROOT, 
        valid_ids,
        seg_model,
        config.DEVICE,
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=config.DATA.CLASSES,
        input_space=config.DATA.SPACE
    )

    test_dataset = ReconsDataset(
        config.DATA.ROOT, 
        test_ids,
        seg_model,
        config.DEVICE,
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=config.DATA.CLASSES,
        input_space=config.DATA.SPACE
    )

    dataloader = {
        "train": DataLoader(
            train_dataset,
            batch_size=config.DATA.BATCH_SIZE,
            num_workers=config.DATA.NUM_WORKERS,
            shuffle=True
        ),
        "valid": DataLoader(
            valid_dataset,
            batch_size=config.DATA.BATCH_SIZE,
            num_workers=config.DATA.NUM_WORKERS,
            shuffle=False
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=config.DATA.NUM_WORKERS,
            shuffle=False
        )
    }

    # Reconstruction Component
    rec_loss = RecLoss(
            alpha_eps=1e-5,
            beta_eps=1e-2,
            resi_min=1e-3,
            resi_max=1e4,
            gamma=config.GAMMA,
            unc_mode=config.UNC_MODE
        )
    rec_backbone = None
    if config.UNC_MODE == "bayescap":
        rec_backbone =  BayesCap(
            in_channels=len(config.DATA.CLASSES), out_channels=len(config.DATA.CLASSES)
        )
    elif config.UNC_MODE == "bnn":
        if config.USE_CONTEXT:
            rec_backbone = GaussCapContext(
                fusion_mode=config.EXP_NAME,   ## ["channel_att", "spatial_att", "cbam_att", "conv", "entrance"]
                yhat_channels=len(config.DATA.CLASSES),
                img_channels=1 if config.DATA.SPACE=="gray" else 3,
                out_channels=len(config.DATA.CLASSES)
            )
        else:
            rec_backbone = BNN(
                in_channels=len(config.DATA.CLASSES),
                out_channels=len(config.DATA.CLASSES)
            )
    elif config.UNC_MODE == "vae":
        rec_backbone = ConvAutoencoder(
            yhat_channels=len(config.DATA.CLASSES),
            out_channels=len(config.DATA.CLASSES),
            img_channels=1 if config.USE_CONTEXT else None,
            fusion_mode=config.EXP_NAME
        )
    elif config.UNC_MODE == "teacher-student":
        rec_backbone = FPN(
                encoder_name=config.MODEL.ENCODER,
                in_channels=len(config.DATA.CLASSES),
                classes=len(config.DATA.CLASSES),
                img_channels=1 if config.USE_CONTEXT else None,
                fusion_mode=config.EXP_NAME
            )
        print(f"Loading teacher checkpoint...")
        rec_backbone.load_state_dict(ckpt["state_dict"], strict=False)
        for param in rec_backbone.parameters():
            param.requires_grad = False
        if config.EXP_NAME in ["channel_att", "spatial_att", "cbam_att", "conv"]:
            for param in rec_backbone.img_encoder.parameters():
                param.requires_grad = True
        for param in rec_backbone.decoder.parameters():
            param.requires_grad = True
        for param in rec_backbone.segmentation_head.parameters():
            param.requires_grad = True
        for param in rec_backbone.uncertainty_head.parameters():
            param.requires_grad = True
    else:
        raise(f"Donot implement the style: {config.UNC_MODE}")
    
    if run_mode == "train":
        print(f"🥝 Start Training Reconstruction Network in {run_mode} mode")
        rec_model = Reconstruction(
                rec_backbone,   ## inference -> use only rec_backbone
                rec_loss,
                config
            )
        
        # checkpoint_callback = ModelCheckpoint(
        #     monitor='valid_dice',
        #     dirpath=ckpt_dir,
        #     filename=f"recons_{config.EXP_NAME}_" + "{epoch}-{valid_dice:.2f}",
        #     mode='max',
        #     save_top_k=1
        # )
        trainer = pl.Trainer(
            check_val_every_n_epoch=config.EVAL_FREQ,       ## config.EVAL_FREQ,
            accelerator=config.DEVICE,
            devices=config.GPU_ID, ## 1
            max_epochs=config.NUM_EPOCHS,
            callbacks=[MyProgressBar()]
            # callbacks=[MyProgressBar(), checkpoint_callback]
        )

        trainer.fit(
            rec_model,
            train_dataloaders=dataloader["train"], 
            val_dataloaders=dataloader["valid"],
            ckpt_path = config.MODEL.RESUME_CKPT if config.MODEL.RESUME_CKPT != "" else None
        )
        valid_metrics = trainer.validate(
            rec_model, dataloaders=dataloader["valid"], verbose=False
        )
        pprint(valid_metrics)

        test_metrics = trainer.test(
            rec_model, dataloaders=dataloader["test"], verbose=False
        )
        pprint(test_metrics)

        save_checkpoint(
            rec_model.model,
            rec_model.optimizer,
            epoch,
            loss,
            os.path.join(ckpt_dir, f"recons_{config.EXP_NAME}_epoch_{epoch}.pt")
        )

        
        # print(f'Saved checkpoint @ {checkpoint_callback.best_model_path}')
        print(f'Saved checkpoint ')
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f"🥝 Start Testing Reconstruction Network in {run_mode} mode")
        print(f"🥝 Loading the reconstruction checkpoint: {config.MODEL.RECONS_CKPT}")
        rec_backbone = load_checkpoint(bmodel=rec_backbone, ckpt_path=config.MODEL.RECONS_CKPT)
        rec_backbone.to(device)
        unc_metrics = {}
        with torch.no_grad():
            for phase in ["valid", "test"]:       # ["train", "valid", "test"]
                list_ece = []
                list_mi = []
                error_sums = []
                # support to compute the correlation
                uncertainties = []
                accuracies = []
                unc_metrics[phase] = {}
                for batch in tqdm(dataloader[phase], total=len(dataloader[phase])):
                    image = batch["image"]
                    yhat = batch["yhat"]    ## (B, 1, 320, 320)
                    mask = batch["mask"]    ## (B, 1, 320, 320)
                    bs = image.shape[0]
                    yhat = yhat.to(device)
                    image = image.to(device)
                    ## Open comment here
                    if config.UNC_MODE == "bayescap":
                        (out_mu, out_alpha, out_beta)= rec_backbone(yhat)
                        out_gauss = out_alpha, out_beta
                        out_mu = out_mu.detach().cpu()
                    elif config.UNC_MODE in ["vae", "teacher-student", "bnn"]:
                        out_mu, out_logvar = rec_backbone(
                            yhat=yhat,
                            img=image if config.USE_CONTEXT else None
                        ) ## assuming out_mu is regression after softmax
                        out_gauss = out_logvar.detach().cpu()
                        out_mu = out_mu.detach().cpu()
                        out_gauss = out_logvar.detach().cpu()
                        out_mu = out_mu.detach().cpu()
                    else:
                        print(f"Donot infer using Bayesian model")
                    #### stop here
                    yhat = yhat.detach().cpu()
                    
                    for i in range(bs):
                        try:
                            ## evaluate uncertainty in the sample-level
                            if config.UNC_MODE in ["bayescap", "bnn", "vae", "teacher-student"]:
                                ## 1. Directly predict uncertainty-map from variance
                                err_map, cor_map = get_error_map(out_mu[i], mask[i])
                                unc_map = get_uncertainty_map(out_gauss[i], config.UNC_MODE)
                                unc, conf = get_confidence(unc_map, out_mu[i])
                                acc = get_accuracy(out_mu[i], mask[i], ignore_bg=config.IGNORE_BG)
                                ece = get_ece(
                                    confidences=1-unc_map, 
                                    accuracies=cor_map,
                                    pred_mask=out_mu[i]
                                )
                            else:
                                # 2. Using entropy
                                err_map, cor_map = get_error_map(yhat[i], mask[i])
                                unc_map = torch.tensor(scipy.stats.entropy(yhat[i].softmax(dim=0), axis=0))
                                unc_map[unc_map>1]=1.
                                unc, conf = get_confidence(unc_map, yhat[i])
                                acc = get_accuracy(yhat[i], mask[i], ignore_bg=config.IGNORE_BG)
                                ece = get_ece(
                                    confidences=1-unc_map, 
                                    accuracies=cor_map,
                                    pred_mask=yhat[i]
                                )

                            uncertainties.append(unc)
                            accuracies.append(acc)
                            
                            mi = get_mutual_information(err_map, unc_map)
                            list_ece.append(ece)
                            list_mi.append(mi)
                            error_sums.append(err_map.sum())
                        except Exception as e:
                            print(f"Error because: {e}")
                unc_metrics[phase]["ece"] = np.mean(list_ece)
                unc_metrics[phase]["mi"] = np.mean(list_mi)
                unc_metrics[phase]["w_mi"] = np.average(list_mi, weights=error_sums)

                # unc_metrics[phase]["ece"] = batch_ece/total_sample
                # unc_metrics[phase]["mi"] = batch_mi/total_sample

                uncertainties = np.array(uncertainties)
                accuracies=np.array(accuracies)

                unc_metrics[phase]["corr"] = get_correlation(
                    uncertainties=uncertainties[np.isfinite(uncertainties)],
                    accuracies=accuracies[np.isfinite(uncertainties)]
                )
        d = date.today().strftime("%m_%d_%Y")
        h = datetime.now().strftime("%H_%M_%S").split("_")
        h_offset = int(datetime.now().strftime("%H_%M_%S").split("_")[0]) + 7
        h[0] = str(h_offset)
        h = d+"_"+"_".join(h)

        save_result_path = os.path.join(
            ckpt_dir,
            f'results_{config.EXP_NAME}_{h}.json'
        )
        with open(save_result_path, 'w') as fp:
            json.dump(unc_metrics, fp)
        ## save cfg
        save_cfg_path = os.path.join(
            ckpt_dir,
            f'configs_{h}.json'    
        )
        with open(save_cfg_path, 'w') as fp:
            json.dump(dict(config), fp)
        print(f"unc_metrics: {unc_metrics}")
        if config.MODEL.RECONS_CKPT != "":
            rec_backbone, _, _, _ = load_checkpoint(
                rec_backbone, None, config.MODEL.RECONS_CKPT
            )
        else:
            print("No checkpoint specified for testing!")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-t', '--template-cfg', default='./configs/tmpl.yaml', help='template configurations')
    parser.add_argument('-e', '--exp-cfg', default='./configs/local.yaml', help='experiment configurations')
    parser.add_argument('-s', '--seed', default=1, help='seed number')
    parser.add_argument('-g', '--gpuid', default=1, help='gpu id')
    parser.add_argument('--exp-name', default="gausscap", help='seed number')
    parser.add_argument('-r', '--run-mode', default="train", help='train/test mode')

    args = parser.parse_args()
    config = load_config(args.template_cfg, args.exp_cfg)
    config.SEED_NUM = int(args.seed)
    config.GPU_ID = int(args.gpuid)
    config.EXP_NAME = args.exp_name
    main(config, args.run_mode)