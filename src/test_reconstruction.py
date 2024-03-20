from load_cfg import load_config
import segmentation_models_pytorch as smp
import torch
import numpy as np
from torch.utils.data import DataLoader
import os
# from dataset import Dataset, ReconsDataset
from dataset_seed import Dataset, ReconsDataset
from transforms import (
    get_training_augmentation,
    get_preprocessing,
    get_validation_augmentation,
)
from engine import Segmentation, Reconstruction
from models.bayesian import BayesCap, BNN, GaussCapContext
from models.vae import VAE, ConvAutoencoder
from models.custom_fpn import FPN

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
from utils import seed_everything, load_checkpoint
from tqdm import tqdm
import scipy
import argparse
import random
import json
from datetime import datetime, date

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
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_files = os.listdir(ckpt_dir)
    for f in ckpt_files:
        if f.startswith("frozen") and config.MODEL.FROZEN_CKPT == "":
            config.MODEL.FROZEN_CKPT = os.path.join(ckpt_dir, f)
        elif f.startswith(f"recons_{config.EXP_NAME}") and config.MODEL.RECONS_CKPT == "":
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
                img_channels=1 if config.DATA.SPACE=="gray" else 3,
                out_channels=len(config.DATA.CLASSES)
            )
        else:
            bnnmodel = BNN(
                in_channels=len(config.DATA.CLASSES),
                out_channels=len(config.DATA.CLASSES)
            )
    elif config.UNC_MODE == "vae":
        bnnmodel = ConvAutoencoder(
            yhat_channels=len(config.DATA.CLASSES),
            out_channels=len(config.DATA.CLASSES),
            img_channels=1 if config.USE_CONTEXT else None,
            fusion_mode=config.EXP_NAME
        )
    elif config.UNC_MODE == "teacher-student":
        bnnmodel = FPN(
                encoder_name=config.MODEL.ENCODER,
                in_channels=len(config.DATA.CLASSES),
                classes=len(config.DATA.CLASSES),
                img_channels=1 if config.USE_CONTEXT else None,
                fusion_mode=config.EXP_NAME
            )
    else:
        print("Do not use the Bayes theory")
        # raise(f"Donot implement the style: {config.UNC_MODE}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if bnnmodel:
        print(f"Loading the reconstruction checkpoint: {config.MODEL.RECONS_CKPT}")
        bnnmodel = load_checkpoint(bmodel=bnnmodel, ckpt_path=config.MODEL.RECONS_CKPT)
        bnnmodel.to(device)
    ## run testing
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
                with torch.no_grad():
                    if config.UNC_MODE == "bayescap":
                        (out_mu, out_alpha, out_beta)= bnnmodel(yhat)
                        out_gauss = out_alpha, out_beta
                        out_mu = out_mu.detach().cpu()
                    elif config.UNC_MODE in ["vae", "teacher-student", "bnn"]:
                        out_mu, out_logvar = bnnmodel(
                            yhat=yhat,
                            img=image if config.USE_CONTEXT else None
                        ) ## assuming out_mu is regression after softmax
                        out_gauss = out_logvar.detach().cpu()
                        out_mu = out_mu.detach().cpu()
                    # elif config.UNC_MODE == "bnn":
                    #     if config.USE_CONTEXT:
                    #         (out_mu, out_logvar) = bnnmodel(
                    #             yhat=yhat,
                    #             img=image
                    #         ) ## assuming out_mu is regression after softmax
                    #     else:
                    #         (out_mu, out_logvar) = bnnmodel(yhat)
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
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-t', '--template-cfg', default='./configs/tmpl.yaml', help='template configurations')
    parser.add_argument('-e', '--exp-cfg', default='./configs/local.yaml', help='experiment configurations')
    parser.add_argument('-s', '--seed', default=1, help='seed number')
    parser.add_argument('--exp-name', default="gausscap", help='seed number')

    args = parser.parse_args()
    config = load_config(args.template_cfg, args.exp_cfg)
    config.SEED_NUM = int(args.seed)
    config.EXP_NAME = args.exp_name
    main(config)