import numpy as np
import torch
import segmentation_models_pytorch as smp
from scipy.stats import pearsonr
from skimage.filters import threshold_multiotsu
from sklearn.metrics import mutual_info_score

"""
Reference:
https://towardsdatascience.com/expected-calibration-error-ece-a-step-by-step-visual-explanation-with-python-code-c3e9aa12937d
"""
from typing import List

import numpy as np

def dc(result, reference):
    r"""
    Dice coefficient
    
    Computes the Dice coefficient (also known as Sorensen index) between the binary
    objects in two images.
    
    The metric is defined as
    
    .. math::
        
        DC=\frac{2|A\cap B|}{|A|+|B|}
        
    , where :math:`A` is the first and :math:`B` the second set of samples (here: binary objects).
    
    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    
    Returns
    -------
    dc : float
        The Dice coefficient between the object(s) in ```result``` and the
        object(s) in ```reference```. It ranges from 0 (no overlap) to 1 (perfect overlap).
        
    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    result = np.atleast_1d(result.astype(bool))
    reference = np.atleast_1d(reference.astype(bool))
    
    intersection = np.count_nonzero(result & reference)
    
    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)
    
    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0
    
    return dc

def dice(pred: np.ndarray, target: np.ndarray, labels: list, exclude_bg: bool):
    """Compute dice for one sample.

    Args:
        pred: prediction array in categorical form (H, W)
        target: target array in categorical form (H, W)

    Returns:
        mean dice
    """
    bg_idx = 0
    dices = []
    if len(labels) > 2:
        for label in labels:
            if exclude_bg and label == bg_idx:
                pass
            else:
                pred_mask, gt_mask = np.isin(pred, label), np.isin(target, label)
                dices.append(dc(pred_mask, gt_mask))
        return np.array(dices).mean()
    else:
        return dc(pred.squeeze(), target.squeeze())

def binary_converter(probs):
    """Converts a binary probability vector into a matrix."""
    return np.array([[1-p, p] for p in probs])

def get_uncertainty_map(
        out_gauss,
        mode
):
    if mode == "bayescap":
        # compute output variance
        out_alpha, out_beta = out_gauss
        a_map = (1/(out_alpha.squeeze() + 1e-5)).data
        b_map = out_beta.squeeze().data
        out_var = (a_map**2)*(torch.exp(torch.lgamma(3/(b_map + 1e-3)))/torch.exp(torch.lgamma(1/(b_map + 1e-3))))
        # # normalize and clip
        # hist, bin_edges = np.histogram(out_var.data.numpy().flatten(), bins=bins)
        # # TODO: consider min max by bin_edges
        # unc_map = torch.clamp(
        #     out_var,
        #     min=bin_edges[-2],
        #     max=bin_edges[-1]
        # )
        # unc_map -= unc_map.min()
        # unc_map /= unc_map.max()
        # thres = (unc_map.min() + unc_map.max())/2
        # unc_map_bin = (unc_map>thres).float()
    else:
        ## one over sigma
        # out_gauss = out_gauss.squeeze() + 1e-4
        out_gauss = out_gauss.squeeze()
        out_var = torch.pow(1/out_gauss, 2)
        ## norm
        # out_var_min = out_var.min()
        # out_var_max = out_var.max()
        # out_var = (out_var - out_var_min)/(out_var_max - out_var_min)
        ## clamp
        out_var[out_var>1.]=1.
    return out_var

def get_error_map(
        out_mu: torch.Tensor,   # [C, W, H]
        mask: torch.Tensor,     # [C, W, H]
):
    out_mu = torch.argmax(out_mu, dim=0)    # [B, W, H]
    mask = torch.argmax(mask, dim=0)
    error_map = torch.tensor(1 * ~np.equal(out_mu.numpy(), mask.numpy()))
    corre_map = 1 - error_map
    return error_map.squeeze(), corre_map.squeeze()

def get_confidence(
    unc_map: torch.Tensor,
    pred: torch.Tensor,     ## out_mu before
):
    mask = torch.argmax(pred, dim=0)
    # unc_map = torch.where((mask!=0), unc_map, 0)
    mask = (mask!=0).float()    # foreground pixels
    # unc_map = unc_map.data.numpy()
    # thresholds = threshold_multiotsu(unc_map, classes = 3)
    # unc_map[unc_map<thresholds[1]] = 0
    # # unc_map[unc_map!=0] = 1
    # unc_map = torch.tensor(unc_map)
    # unc_map = (unc_map!=0).float()
    unc = unc_map.sum() / mask.sum()
    conf = 1 - unc
    return unc.item(), conf.item()

def get_accuracy(
    pred: torch.Tensor, 
    mask: torch.Tensor,
    mode: str = "multiclass", # binary
    num_classes: int = 3,
    ignore_bg: bool = False
):
    """
    dice score --> correlation
    """
    pred = torch.argmax(pred, dim=0)
    mask = torch.argmax(mask, dim=0)
    pred = pred.long()
    mask = mask.long()
    score = dice(
        pred=pred.numpy(),
        target=mask.numpy(),
        labels=list(range(num_classes)),
        exclude_bg=ignore_bg
    )
    return score

def get_correlation(
    uncertainties: np.ndarray,
    accuracies: np.ndarray,
):
    """
    dataset-based
    """
    corr, _  = pearsonr(uncertainties, accuracies)
    return abs(corr)

def get_mutual_information(error: torch.Tensor, uncertainty: torch.Tensor):
    """Computes mutual information between error and uncertainty.

    Args:
        error: numpy binary array indicating error.
        uncertainty: numpy float array indicating uncertainty.

    Returns:
        mutual_information
    """
    error = error.data.numpy()
    uncertainty = uncertainty.data.numpy()

    hist_2d, x_edges, y_edges = np.histogram2d(error.ravel(), uncertainty.ravel())

    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)  # marginal for x over y
    py = np.sum(pxy, axis=0)  # marginal for y over x
    px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0  # Only non-zero pxy values contribute to the sum
    # mi_1 = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
    # mi_2 = mutual_info_score(None, None, contingency=hist_2d)
    # print(f"mi_1: {mi_1} | mi_2: {mi_2}")
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def get_ece(
        confidences: torch.Tensor,
        accuracies: torch.Tensor,
        pred_mask: torch.Tensor,
        M: int=20
):
    ## remove bg
    pred_mask = torch.argmax(pred_mask, 0)
    not_bg = pred_mask != 0
    confidences = confidences[not_bg].flatten()
    accuracies = accuracies[not_bg].flatten()

    confidences = confidences.data.numpy()
    accuracies = accuracies.data.numpy()
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = np.greater(confidences, bin_lower) * np.less(confidences, bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece.item()


def calc_MI(x, y, bins=10):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

# if __name__ == "__main__":
    # samples = np.array([0.22, 0.64, 0.92, 0.42, 0.51, 0.15, 0.70, 0.37, 0.83])
    # true_labels = np.array([0,1,0,0,0,1,1,0,1])
    # e = get_ece(samples, true_labels, 3)
    # c = get_correlation(np.random.randn(100), np.random.randn(100))
    # print(c)
    # a = torch.randn(10,10)
    # get_mutual_information(a, a)
    # print(m)



