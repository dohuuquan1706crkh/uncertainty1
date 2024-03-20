from load_cfg import load_config
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
import os
import pytorch_lightning as pl
from pprint import pprint

# from dataset import Dataset
from dataset_seed import Dataset
from transforms import (
    get_training_augmentation,
    get_preprocessing,
    get_validation_augmentation,
    normalize_sample
)
from src.models.segmentation import Segmentation
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import MyProgressBar, seed_everything
import random
import shutil

def main(config):
    seed_everything(seed=config.SEED_NUM)
    ## random seed
    total_ids=list(range(1,501))
    random.shuffle(total_ids)
    train_ids = total_ids[:400]
    valid_ids = total_ids[400:450]
    test_ids = total_ids[450:]
    print(f"Verify seed num: {config.SEED_NUM} \n -Train: {train_ids[:5]} \n -Valid: {valid_ids[:5]} \n -Test: {test_ids[:5]}")
    ## init the preprocessing and augmentation module
    if config.DATA.SPACE == "rgb":
        preprocessing_fn = smp.encoders.get_preprocessing_fn(
            config.MODEL.ENCODER, config.MODEL.ENCODER_WEIGHTS
        )
    ## gray image needs a custom preprocessing_fn
    else:
        preprocessing_fn = None
    
    print(f"preprocessing_fn: {preprocessing_fn}")
    train_dataset = Dataset(
        config.DATA.ROOT, 
        train_ids,
        augmentation=get_training_augmentation(config.DATA.SPACE), 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=config.DATA.CLASSES,
        input_space=config.DATA.SPACE
    )

    valid_dataset = Dataset(
        config.DATA.ROOT, 
        valid_ids, 
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=config.DATA.CLASSES,
        input_space=config.DATA.SPACE
    )

    test_dataset = Dataset(
        config.DATA.ROOT, 
        test_ids, 
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
            batch_size=config.DATA.BATCH_SIZE,
            num_workers=config.DATA.NUM_WORKERS,
            shuffle=False
        )
    }

    model = Segmentation(
        arch=config.MODEL.ARCH,
        encoder_name=config.MODEL.ENCODER,
        in_channels=1 if config.DATA.SPACE=="gray" else 3,  
        out_classes=len(config.DATA.CLASSES),   # 1 is binary
    )
    
    ckpt_dirpath = os.path.join(
        config.SAVE_DIR, f"seed_{config.SEED_NUM}"
    )
    # if os.path.isdir(ckpt_dirpath):
    #     shutil.rmtree(ckpt_dirpath)
    os.makedirs(ckpt_dirpath, exist_ok=True)
    print(f"Created {ckpt_dirpath} saved directory...")
    checkpoint_callback = ModelCheckpoint(
        monitor='valid_dataset_dice',
        dirpath=ckpt_dirpath,
        filename=f"frozen_{config.EXP_NAME}_" + "{epoch}-{valid_dataset_dice:.2f}",
        mode='max'
    )
    ckpt_files = os.listdir(ckpt_dirpath)
    for f in ckpt_files:
        if f.startswith("frozen"):
            print("The frozen model is existed...")
            return
    
    print("Start training the frozen model...")
    trainer = pl.Trainer(
        check_val_every_n_epoch=config.EVAL_FREQ,
        accelerator=config.DEVICE,
        devices=config.GPU_ID, ## 1
        max_epochs=config.NUM_EPOCHS,
        callbacks=[MyProgressBar(), checkpoint_callback]
    )

    trainer.fit(
        model, 
        train_dataloaders=dataloader["train"], 
        val_dataloaders=dataloader["valid"],
    )
    valid_metrics = trainer.validate(
        model, dataloaders=dataloader["valid"], verbose=False
    )
    pprint(valid_metrics)
    # run test dataset
    test_metrics = trainer.test(
        model,
        dataloaders=dataloader["test"],
        verbose=False,
        ckpt_path="best"
    )
    pprint(test_metrics)
    print(f'Saved checkpoint @ {checkpoint_callback.best_model_path}')
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-t', '--template-cfg', default='./configs/tmpl.yaml', help='template configurations')
    parser.add_argument('-e', '--exp-cfg', default='./configs/local.yaml', help='experiment configurations')
    parser.add_argument('-s', '--seed', default=1, help='seed number')
    parser.add_argument('-g', '--gpuid', default=1, help='gpu id')
    parser.add_argument('--exp-name', default="gausscap", help='seed number')

    args = parser.parse_args()
    config = load_config(args.template_cfg, args.exp_cfg)
    config.SEED_NUM = int(args.seed)
    config.GPU_ID = int(args.gpuid)
    config.EXP_NAME = args.exp_name    
    main(config)