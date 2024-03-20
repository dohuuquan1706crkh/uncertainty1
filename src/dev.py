from load_cfg import load_config
import segmentation_models_pytorch as smp
import torch
import numpy as np
from torch.utils.data import DataLoader
import os
# from dataset import Dataset, ReconsDataset
from dataset_seed import Dataset, ReconsDataset
from transforms import get_training_augmentation, get_preprocessing, get_validation_augmentation
from engine import Segmentation, Reconstruction
from models.bayesian import BayesCap, BNN, GaussCapContext
from models.vae import VAE
from metrics import (
    get_uncertainty_map,
    get_error_map,
    get_confidence,
    get_accuracy,
    get_ece,
    get_mutual_information,
    get_correlation
)
from collections import OrderedDict
from utils import seed_everything, load_checkpoint, visualize
from tqdm import tqdm
import scipy
import argparse
import random
import json

def main(config):
    seed_everything(seed=config.SEED_NUM)
    ## random seed
    total_ids=list(range(1,501))
    random.shuffle(total_ids)
    train_ids = total_ids[:400]
    valid_ids = total_ids[400:450]
    test_ids = total_ids[450:]
    print(f"Verify seed num: {config.SEED_NUM} \n -Train: {train_ids[:5]} \n -Valid: {valid_ids[:5]} \n -Test: {test_ids[:5]}")
    ckpt_dir = os.path.join(
        config.SAVE_DIR, f"seed_{config.SEED_NUM}"
    )
    ckpt_files = os.listdir(ckpt_dir)
    for f in ckpt_files:
        if f.startswith("frozen"):
            config.MODEL.FROZEN_CKPT = os.path.join(ckpt_dir, f)
        elif f.startswith(f"recons_{config.EXP_NAME}"):
            config.MODEL.RECONS_CKPT = os.path.join(ckpt_dir, f)
    
    # load frozen model
    segmodel = Segmentation(
        arch=config.MODEL.ARCH,
        encoder_name=config.MODEL.ENCODER,
        in_channels=1 if config.DATA.SPACE=="gray" else 3,  
        out_classes=len(config.DATA.CLASSES),   # 1 is binary
    )
    
    print(f"Loading the frozen checkpoint: {config.MODEL.FROZEN_CKPT}")
    frozen_ckpt = torch.load(config.MODEL.FROZEN_CKPT, map_location="cpu")
    segmodel.load_state_dict(frozen_ckpt["state_dict"])

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
        segmodel,
        config.DEVICE,
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=config.DATA.CLASSES,
        input_space=config.DATA.SPACE
    )

    valid_dataset = ReconsDataset(
        config.DATA.ROOT, 
        valid_ids,
        segmodel,
        config.DEVICE,
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=config.DATA.CLASSES,
        input_space=config.DATA.SPACE
    )

    test_dataset = ReconsDataset(
        config.DATA.ROOT, 
        test_ids,
        segmodel,
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
            shuffle=False
        ),
        "valid": DataLoader(
            valid_dataset,
            batch_size=1,
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
    bnnmodel = None
    if config.UNC_MODE == "bayescap":
        bnnmodel =  BayesCap(
            in_channels=len(config.DATA.CLASSES), out_channels=len(config.DATA.CLASSES)
        )
    elif config.UNC_MODE == "bnn":
        if config.USE_CONTEXT:
            bnnmodel = GaussCapContext(
                fusion_mode=config.EXP_NAME,
                yhat_channels=len(config.DATA.CLASSES),
                img_channels=3,
                out_channels=len(config.DATA.CLASSES)
            )
        else:
            bnnmodel = BNN(
                in_channels=len(config.DATA.CLASSES),
                out_channels=len(config.DATA.CLASSES)
            )
    elif config.UNC_MODE == "vae":
        bnnmodel = VAE(
            image_size=config.DATA.IMG_SIZE,
            input_channels=len(config.DATA.CLASSES),
            out_channels=len(config.DATA.CLASSES),
            embedding_dim=512,
            shape_before_flattening=[64, 16, 16],
            use_context=True
        )
    # else:
    #     raise(f"Donot implement the style: {config.UNC_MODE}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if bnnmodel:
        if config.MODEL.RECONS_CKPT == "":
            config.MODEL.RECONS_CKPT = os.path.join(
                config.SAVE_DIR,
                f"seed_{config.SEED_NUM}_"+config.EXP_NAME+".ckpt"
            )
        print(f"Loading the reconstruction checkpoint: {config.MODEL.RECONS_CKPT}")
        bnnmodel = load_checkpoint(bmodel=bnnmodel, ckpt_path=config.MODEL.RECONS_CKPT)
        bnnmodel.to(device)
    ## run testing
    unc_metrics = {}

    ## save cfg
    with open(os.path.join(config.SAVE_DIR, f'configs_seed_{config.SEED_NUM}.json'), 'w') as fp:
        json.dump(dict(config), fp)

    with torch.no_grad():
        for phase in ["valid", "test"]:       # ["train", "valid", "test"]
            total_sample = 0
            batch_ece = 0
            batch_mi = 0
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
                with torch.no_grad():
                    if config.UNC_MODE == "bayescap":
                        (out_mu, out_alpha, out_beta)= bnnmodel(yhat)
                        out_gauss = out_alpha, out_beta
                        out_mu = out_mu.detach().cpu()

                    elif config.UNC_MODE == "vae":
                        out_mu, out_logvar, z, z_context = bnnmodel(
                            yhat=yhat,
                            x=image if config.USE_CONTEXT else None
                        ) ## assuming out_mu is regression after softmax
                        out_mu = out_mu.detach().cpu()

                        out_gauss = out_logvar.detach().cpu()
                    elif config.UNC_MODE == "bnn":
                        if config.USE_CONTEXT:
                            (out_mu, out_logvar) = bnnmodel(
                                yhat=yhat,
                                img=image
                            ) ## assuming out_mu is regression after softmax
                        else:
                            (out_mu, out_logvar) = bnnmodel(yhat)
                        out_gauss = out_logvar.detach().cpu()
                        out_mu = out_mu.detach().cpu()

                #### stop here
                yhat = yhat.detach().cpu()
                for i in range(bs):
                    # try:
                    if config.UNC_MODE in ["bayescap", "bnn", "vae"]:
                        ## 1. Directly predict uncertainty-map from variance
                        err_map, cor_map = get_error_map(out_mu[i], mask[i])
                        unc_map = get_uncertainty_map(out_gauss[i], config.UNC_MODE)
                        unc, conf = get_confidence(unc_map, out_mu[i])
                        acc = get_accuracy(out_mu[i], mask[i], ignore_bg=config.IGNORE_BG)
                        pred_seg = torch.argmax(out_mu[i].detach(), dim=0)
                        ece = get_ece(
                            confidences=1-unc_map, 
                            accuracies=cor_map,
                            pred_mask=out_mu[i]
                        )
                    else:
                        # 2. Using entropy
                        err_map, cor_map = get_error_map(yhat[i], mask[i])
                        unc_map = torch.tensor(scipy.stats.entropy(yhat[i].softmax(dim=0), axis=0))
                        unc, conf = get_confidence(unc_map, yhat[i])
                        acc = get_accuracy(yhat[i], mask[i], ignore_bg=config.IGNORE_BG)
                        pred_seg = torch.argmax(yhat[i].detach(), dim=0)
                        ece = get_ece(
                            confidences=1-unc_map, 
                            accuracies=cor_map,
                            pred_mask=yhat[i]
                        )

                    uncertainties.append(unc)
                    accuracies.append(acc)
                    ece = get_ece(1-unc_map, cor_map)
                    mi = get_mutual_information(err_map, unc_map)
                    batch_ece += ece
                    batch_mi += mi
                    total_sample += 1
                    visualize([pred_seg, err_map, unc_map], f"test_{total_sample}")
                    if total_sample == 2:
                        break
                    # except Exception as e:
                    #     print(f"Error because: {e}")
                
                
            unc_metrics[phase]["ece"] = batch_ece/total_sample
            unc_metrics[phase]["mi"] = batch_mi/total_sample

            uncertainties = np.array(uncertainties)
            accuracies=np.array(accuracies)

            unc_metrics[phase]["corr"] = get_correlation(
                uncertainties=uncertainties[np.isfinite(uncertainties)],
                accuracies=accuracies[np.isfinite(uncertainties)]
            )
    with open(os.path.join(config.SAVE_DIR, f'results_seed_{config.SEED_NUM}.json'), 'w') as fp:
        json.dump(unc_metrics, fp)
    print(f"unc_metrics: {unc_metrics}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-t', '--template-cfg', default='./configs/tmpl.yaml', help='template configurations')
    parser.add_argument('-e', '--exp-cfg', default='./configs/local.yaml', help='experiment configurations')
    parser.add_argument('-s', '--seed', default=1, help='seed number')
    args = parser.parse_args()
    config = load_config(args.template_cfg, args.exp_cfg)
    config.SEED_NUM = int(args.seed)
    main(config)