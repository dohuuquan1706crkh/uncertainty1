import torch
import torch.nn.functional as F
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.filters import threshold_multiotsu
import os
import random
from collections import OrderedDict

def resize_aspect_ratio(img, size, interp=cv2.INTER_LINEAR):
    """
    resize min edge to target size, keeping aspect ratio
    """
    if len(img.shape) == 2:
        h,w = img.shape
    elif len(img.shape) == 3:
        h,w,_ = img.shape
    else:
        return None
    if h > w:
        new_w = size
        new_h = h*new_w//w
    else:
        new_h = size
        new_w = w*new_h//h
    return cv2.resize(img, (new_w, new_h), interpolation=interp)

def create_tensor_from_img(img):
    img = np.transpose(img, (0, 3, 1, 2))
    img = img / 255
    imagenet_mean = np.asarray([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1])
    imagenet_std = np.asarray([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1])
    img = (img-imagenet_mean)/imagenet_std
    img = torch.FloatTensor(img)
    return img

def min_edge_crop(img, position="center"):
    """
    crop image base on min size
    :param img: image to be cropped
    :param position: where to crop the image
    :return: cropped image
    """
    assert position in ['center', 'left', 'right'], "position must either be: left, center or right"

    h, w = img.shape[:2]

    if h == w:
        return img

    min_edge = min(h, w)
    if h > min_edge:
        if position == "left":
            img = img[:min_edge]
        elif position == "center":
            d = (h - min_edge) // 2
            img = img[d:-d] if d != 0 else img

            if h % 2 != 0:
                img = img[1:]
        else:
            img = img[-min_edge:]

    if w > min_edge:
        if position == "left":
            img = img[:, :min_edge]
        elif position == "center":
            d = (w - min_edge) // 2
            img = img[:, d:-d] if d != 0 else img

            if w % 2 != 0:
                img = img[:, 1:]
        else:
            img = img[:, -min_edge:]

    assert img.shape[0] == img.shape[1], f"height and width must be the same, currently {img.shape[:2]}"
    return img


def read_image(img_path, target_size = 224):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = resize_aspect_ratio(img, target_size)
    img = min_edge_crop(img, 'center')
    rs_img = img
    img = np.transpose(img, (2, 0, 1))
    img = img / 255
    imagenet_mean = np.asarray([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1])
    imagenet_std = np.asarray([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1])
    img = (img-imagenet_mean)/imagenet_std
    img = torch.FloatTensor(img)
    return rs_img, img

def read_mask(img_path, target_size = 224):
    mask = cv2.imread(img_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = resize_aspect_ratio(mask, target_size)
    mask = cv2.threshold(mask, 128, 1, cv2.THRESH_BINARY)[1]
    # print(np.unique(mask, return_counts=True))
    # mask[mask==255]=1
    # print(mask.max(), mask.min())

    mask = min_edge_crop(mask, 'center')
    mask = torch.FloatTensor(mask)
    return mask

def overlay(image, mask, color = (255, 0, 0), alpha = 0.5, resize=None):
    """Combines image and its segmentation mask into a single image.
    https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay

    Params:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray

    """
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined, colored_mask

# helper function for data visualization
def visualize(images, save_name=None):
    """
    PLot images in one row.
    """
    n = 4 # len(images)
    out_mu, err_map, unc_map = images
    err_map = err_map.data.numpy()
    unc_map = unc_map.data.numpy()
    unc_map_bin = unc_map
    thresholds = threshold_multiotsu(unc_map_bin, classes = 3)
    ## upper_thr = thresholds[1], lower_thr = thresholds[0]
    # unc_map_bin[unc_map_bin<thresholds[1]] = 0
    unc_map_bin = np.digitize(unc_map_bin, bins=thresholds)
    unc_map_bin[unc_map_bin == 1] = 0
    unc_map_bin[unc_map_bin == 2] = 1
    
    plt.figure(figsize=(10, 5))
    for i, img in enumerate([out_mu, err_map, unc_map, unc_map_bin]):
    # for i, img in enumerate([err_map, unc_map]):
        plt.subplot(1, n, i + 1)
        plt.imshow(img)
        plt.axis('off')
    if save_name:
        plt.savefig(f'./outputs/{save_name}.png')

from pytorch_lightning.callbacks import TQDMProgressBar
class MyProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)
        items["train_loss"] = np.mean(pl_module.batch_loss or float("nan"))
        return items

def norm(data, min: int=0.1, max: int=0.9):
    scaled_data = []
    assert len(data.shape)==4
    for x in data:
        new_x = (max-min)*(x-x.min())/(x.max()-x.min()) + min
        scaled_data.append(new_x)
    return torch.stack(scaled_data, dim=0)

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def load_checkpoint(bmodel, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    fix_ckpt = OrderedDict()
    for key, value in ckpt["state_dict"].items():
        new_key = ".".join(key.split(".")[1:])
        fix_ckpt[new_key] = value
    bmodel.load_state_dict(fix_ckpt)
    bmodel.eval();
    return bmodel