from torch.utils.data import Dataset as BaseDataset
import cv2
import numpy as np
import os
from utils import visualize
from transforms import (
    get_training_augmentation,
    get_validation_augmentation,
    get_preprocessing,
    normalize_sample
)
import torch
# import torch.nn.functional as F

VIEWS = ["2CH", "4CH"]
INSTANTS = ["ED", "ES"]
img_pattern = "{patient_name}_{view}_{instant}.jpg"
# mask_pattern = "{patient_name}_{view}_{instant}_gt.nii.gz"

class Dataset(BaseDataset):
    """CAMUS Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)

        CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 
               'tree', 'signsymbol', 'fence', 'car', 
               'pedestrian', 'bicyclist', 'unlabelled']
    
    """
    
    CLASSES = ['background', 'myo', 'lv', 'atrium']
    
    def __init__(
            self,
            root_dir: str,
            patient_ids: list,
            classes=None, 
            augmentation=None, 
            preprocessing=None,
            input_space: str = "rgb"
    ):
        self.input_space = input_space
        self.ids = []
        for _id in patient_ids:
            for view in VIEWS:
                for instant in INSTANTS:
                    img_name=img_pattern.format(
                        patient_name=f"patient{_id:04d}", view=view, instant=instant
                    )
                    self.ids.append(img_name)
        images_dir = os.path.join(root_dir,"img")
        masks_dir = os.path.join(root_dir,"mask")
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        if self.input_space == "gray":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  ## COLOR_BGR2RGB
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  ## COLOR_BGR2RGB
        mask = cv2.imread(self.masks_fps[i])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # extract certain classes from mask (e.g. cars)
        mask_unique = np.unique(mask).tolist()
        ignore_mask_ids = list(set(mask_unique) - set(self.class_values))
        for rm_id in ignore_mask_ids:
            mask[mask==rm_id] = 0
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float') ## onehot
        ## convert onehot to original
        # mask = np.argmax(mask, axis=-1)
        # apply augmentations
        sample = {"image": image, "mask": mask}
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        if self.input_space == "gray":
            image = normalize_sample(image)
        # apply preprocessing
        # print(f"After normalizing: {image.min(), image.max()}")
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            # image, mask = sample['image'], sample['mask']
        return sample
        
    def __len__(self):
        return len(self.ids)
    
class ReconsDataset(BaseDataset):
    """CAMUS Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)

        CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 
               'tree', 'signsymbol', 'fence', 'car', 
               'pedestrian', 'bicyclist', 'unlabelled']
    
    """
    
    CLASSES = ['background', 'myo', 'lv', 'atrium']
    
    def __init__(
            self, 
            root_dir: str,
            patient_ids: list,
            segmodel,
            device="cpu",
            classes=None, 
            augmentation=None, 
            preprocessing=None,
            input_space: str = "rgb"
    ):
        self.input_space = input_space
        self.ids = []
        for _id in patient_ids:
            for view in VIEWS:
                for instant in INSTANTS:
                    img_name=img_pattern.format(
                        patient_name=f"patient{_id:04d}", view=view, instant=instant
                    )
                    self.ids.append(img_name)
        
        images_dir = os.path.join(root_dir,"img")
        masks_dir = os.path.join(root_dir,"mask")
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.segmodel = segmodel
        self.segmodel.model.eval()
        # self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # self.segmodel.to(self.device)

    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        if self.input_space == "gray":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  ## COLOR_BGR2RGB
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  ## COLOR_BGR2RGB        
        mask = cv2.imread(self.masks_fps[i])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # extract certain classes from mask (e.g. cars)
        mask_unique = np.unique(mask).tolist()
        ignore_mask_ids = list(set(mask_unique) - set(self.class_values))
        for rm_id in ignore_mask_ids:
            mask[mask==rm_id] = 0
        # print(np.unique(mask))
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float') ## onehot
        # print(np.unique(mask), mask.shape)
        # apply augmentations

        sample = {"image": image, "mask": mask}
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        if self.input_space == "gray":
            image = normalize_sample(image)
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            # image, mask = sample['image'], sample['mask']
    
        # pass the sample into segmentation model
        with torch.no_grad():
            img = torch.tensor(sample["image"])
            # img = img.to(self.device)
            img = torch.unsqueeze(img, 0)
            logits = self.segmodel.model(img)
            logits = logits.detach().cpu()
        
        # Using the logits of yhat
        if len(self.class_values) == 1:
            pr_masks = logits.sigmoid()
            pr_masks = (pr_masks > 0.5).float()
        else:
            # pr_masks = logits.softmax(dim=1) ## [B, C, W, H]
            pr_masks = logits.detach() ## [B, C, W, H]
        # from utils import visualize
        # visualize([mask, torch.argmax(pr_masks, dim=1)[0].numpy()])
        pr_mask = pr_masks[0].cpu()
        sample["yhat"] = pr_mask.float()
        return sample
    
    def __len__(self):
        return len(self.ids)

if __name__ == "__main__":
    DATA_DIR = "../dataset/camus"
    patient_ids = list(range(1,10))
    dataset = Dataset(
        DATA_DIR, patient_ids, classes=['lv']
    )

    ## test augmentation
    augmented_dataset = Dataset(
        DATA_DIR,
        patient_ids, 
        augmentation=get_training_augmentation(), 
        classes=['lv'],
    )

    # same image with different random transforms
    for i in range(3):
        sample = dataset[1]
        image, mask = sample["image"], sample["mask"]
        print(f"image: {image.max(), image.min()} | mask: {mask.min(), mask.max()}")