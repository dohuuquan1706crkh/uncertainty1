from typing import Any, Dict, Tuple, List, Sequence
from numbers import Real
import numpy as np
import SimpleITK as sitk
import os
import cv2
from tqdm import tqdm
import random
import argparse
from numbers import Number
from pathlib import Path
from typing import Tuple

import numpy as np
import SimpleITK
from PIL import Image

def sitk_load(filepath) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Loads an image using SimpleITK and returns the image and its metadata.

    Args:
        filepath: Path to the image.

    Returns:
        - ([N], H, W), Image array.
        - Collection of metadata.
    """
    # Load image and save info
    image = sitk.ReadImage(str(filepath))
    info = {"origin": image.GetOrigin(), "spacing": image.GetSpacing(), "direction": image.GetDirection()}

    # Extract numpy array from the SimpleITK image object
    im_array = np.squeeze(sitk.GetArrayFromImage(image))

    return im_array, info

def load_mhd(filepath: Path) -> Tuple[np.ndarray, Tuple[Tuple[Number, ...], ...]]:
    """Loads a mhd image and returns the image and its metadata.

    Args:
        filepath: Path to the image.

    Returns:
        - ([N], H, W), Image array.
        - Collection of metadata.
    """
    # load image and save info
    image = SimpleITK.ReadImage(str(filepath))
    info = (image.GetSize(), image.GetOrigin(), image.GetSpacing(), image.GetDirection())

    # create numpy array from the .mhd file and corresponding image
    im_array = np.squeeze(SimpleITK.GetArrayFromImage(image))

    return im_array, info

def remove_labels(segmentation: np.ndarray, labels_to_remove: Sequence[int], fill_label: int = 0) -> np.ndarray:
    """Removes labels from the segmentation map, reassigning the affected pixels to `fill_label`.

    Args:
        segmentation: ([N], H, W, [1|C]), Segmentation map from which to remove labels.
        labels_to_remove: Labels to remove.
        fill_label: Label to assign to the pixels currently assigned to the labels to remove.

    Returns:
        ([N], H, W, [1]), Categorical segmentation map with the specified labels removed.
    """
    seg = segmentation.copy()
    if seg.max() == 1 and seg.shape[-1] > 1:  # If the segmentation map is in one-hot format
        for label_to_remove in labels_to_remove:
            seg[..., fill_label] += seg[..., label_to_remove]
        seg = np.delete(seg, labels_to_remove, axis=-1)
    else:  # the segmentation map is categorical
        seg[np.isin(seg, labels_to_remove)] = fill_label
    return seg

def resize_image(image: np.ndarray, size: Tuple[int, int], resample=Image.Resampling.BILINEAR) -> np.ndarray:
    """Resizes the image to the specified dimensions.

    Args:
        image: Input image to process. Must be in a format supported by PIL.
        size: Width and height dimensions of the processed image to output.
        resample: Resampling filter to use.

    Returns:
        Input image resized to the specified dimensions.
    """
    resized_image = np.array(Image.fromarray(image).resize(size, resample=resample))
    return resized_image

def _get_sequence_data(root: str, patient_id: str, view: str) -> Tuple[List[np.ndarray], List[np.ndarray], List[Real]]:
    """Fetches additional reference segmentations, interpolated between ED and ES instants.

    Args:
        patient_id: Patient id formatted to match the identifiers in the mhd files' names.
        view: View for which to fetch the patient's data.

    Returns:
        - Sequence of ultrasound images acquired over a cardiac cycle.
        - Segmentation masks associated with the sequence of ultrasound images.
        - Metadata concerning the sequence.
    """
    patient_folder = os.path.join(root, patient_id)
    sequence_fn_template = f"{patient_id}_{view}_half_sequence{{}}.nii.gz"

    # Open interpolated segmentations
    data_x, data_y = [], []
    sequence, info = load_mhd(
        os.path.join(patient_folder, sequence_fn_template.format(""))
    )
    sequence_gt, _ = load_mhd(
        os.path.join(patient_folder, sequence_fn_template.format("_gt"))
    )

    for image, segmentation in zip(sequence, sequence_gt):  # For every instant in the sequence
        data_x.append(image)
        data_y.append(segmentation)

    info = [item for sublist in info for item in sublist]  # Flatten info

    return data_x, data_y, info

def _get_view_data(root: str, patient_id: str, view: str) -> Tuple[np.ndarray, np.ndarray, List[Real], Dict[str, int]]:
    """Fetches the data for a specific view of a patient.

    If ``self.use_sequence`` is ``True``, augments the dataset with sequence between the ED and ES instants.
    Otherwise, returns the view data as is.

    Args:
        patient_id: Patient ID formatted to match the identifiers in the mhd files' names.
        view: View for which to fetch the patient's data.

    Returns:
        - Sequence of ultrasound images acquired over a cardiac cycle.
        - Segmentation masks associated with the sequence of ultrasound images.
        - Metadata concerning the sequence.
        - Mapping between clinically important instants and the index where they appear in the sequence.
    """
    info_filename_format = "Info_{view}.cfg"
    sequence_type_instants = ['ED', 'ES']
    view_info_fn = os.path.join(
        root, patient_id, info_filename_format.format(view=view)
    )

    # Determine the index of segmented instants in sequence
    instants = {}
    with open(str(view_info_fn), "r") as view_info_file:
        view_info = {(pair := line.split(": "))[0]: pair[1] for line in view_info_file.read().splitlines()}
    for instant in sequence_type_instants:
        # For [ED,ES], read the frame number from the corresponding field in the info file
        # The ED_E is always the last frame, so populate this info from the total number of frames instead
        instants[instant] = int(view_info[instant if instant != "ED_E" else "NbFrame"]) - 1

    # Get data for the whole sequence ranging from ED to ES
    sequence, sequence_gt, info = _get_sequence_data(root, patient_id, view)

    # Ensure ED comes before ES (swap when ES->ED)
    if (ed_idx := instants["ED"]) > (es_idx := instants["ES"]):
        print(
            f"The image and reference sequence for '{patient_id}_{view}' were reversed because the metadata file "
            f"indicates that ED originally came after ES in the frames: {instants}."
        )
        sequence, sequence_gt = list(reversed(sequence)), list(reversed(sequence_gt))
        instants["ED"], instants["ES"] = es_idx, ed_idx

    # Include all or only some instants from the input and reference data according to the parameters
    data_x, data_y = [], []
    # if self.flags[CamusTags.full_sequence]:
    #     data_x, data_y = sequence, sequence_gt
    # else:
    for instant in instants:
        data_x.append(sequence[instants[instant]])
        data_y.append(sequence_gt[instants[instant]])

    # Update indices of clinically important instants to match the new slicing of the sequences
    instants = {instant_key: idx for idx, instant_key in enumerate(instants)}

    # Add channel dimension
    return np.array(data_x), np.array(data_y), info, instants

def _write_patient_img(root: str, patient_id: str, target_image_size: tuple=(256,256)) -> None:
    """Writes the raw image data of a patient to a image.

    Args:
        patient_group: HDF5 patient group for which to fetch and save the data.
    """

    available_views = ["2CH", "4CH"]
    labels_to_remove = [3]
    for view in available_views:
        # The order of the instants within a view dataset is chronological: ED -> ES -> ED
        data_x, data_y, info_view, instants = _get_view_data(root, patient_id, view)

        data_y = remove_labels(data_y, labels_to_remove, fill_label=0)

        # if self.flags[CamusTags.registered]:
        #     registering_parameters, data_y_proc, data_x_proc = self.registering_transformer.register_batch(
        #         data_y, data_x
        #     )
        # else:
        data_x_proc = np.array([resize_image(x, target_image_size, resample=Image.Resampling.BILINEAR) for x in data_x])
        data_y_proc = np.array([resize_image(y, target_image_size) for y in data_y])

        # # Write image and groundtruth data
        # patient_view_group = patient_group.create_group(view)
        # patient_view_group.create_dataset(
        #     name=CamusTags.img_proc, data=data_x_proc[..., np.newaxis], **img_save_options
        # )
        # patient_view_group.create_dataset(name=CamusTags.gt, data=data_y, **seg_save_options)
        # patient_view_group.create_dataset(name=CamusTags.gt_proc, data=data_y_proc, **seg_save_options)

        # # Write metadata useful for providing instants or full sequences
        # patient_view_group.attrs[CamusTags.info] = info_view
        # patient_view_group.attrs[CamusTags.instants] = list(instants)
        # patient_view_group.attrs.update(instants)

        # # Write metadata concerning the registering applied
        # if self.flags[CamusTags.registered]:
        #     patient_view_group.attrs.update(registering_parameters)
    return data_x_proc, data_y_proc

def main(dataset: str, src_dir: str, dst_dir: str):
    # os.makedirs(dst_dir, exist_ok=True)
    # if dataset.lower() == "camus":
    #     """
    #     _IMG_DIR = "../dataset/database_nifti"
    #     0: background
    #     1: myo
    #     2: lv
    #     3: atrium
    #     """
    #     VIEWS = ["2CH", "4CH"]
    #     INSTANTS = ["ED", "ES"]
    #     img_pattern = "{patient_name}_{view}_{instant}.nii.gz"
    #     mask_pattern = "{patient_name}_{view}_{instant}_gt.nii.gz"
        
    #     total_ids=list(range(1,501))
    #     ## img, mask folder
    #     img_dir = os.path.join(dst_dir, "img")
    #     mask_dir = os.path.join(dst_dir, "mask")
    #     os.makedirs(img_dir, exist_ok=True)
    #     os.makedirs(mask_dir, exist_ok=True)
    #     for _id in tqdm(total_ids):
    #         patient_name = f"patient{_id:04d}"
            
    #         for view in VIEWS:
    #             for instant in INSTANTS:
    #                 img_name=img_pattern.format(patient_name=patient_name, view=view, instant=instant)
    #                 mask_name=mask_pattern.format(patient_name=patient_name, view=view, instant=instant)
    #                 img, _ = sitk_load(os.path.join(src_dir, patient_name, img_name))
    #                 mask, _ = sitk_load(os.path.join(src_dir, patient_name, mask_name))
    #                 cv2.imwrite(
    #                     os.path.join(img_dir, img_name.replace("nii.gz", "jpg")),
    #                     img
    #                 )
    #                 cv2.imwrite(
    #                     os.path.join(mask_dir, mask_name.replace("_gt.nii.gz", ".jpg")),
    #                     mask
    #                 )
    # else:
    #     raise(f"Dataset name: {dataset} is not implemented")

    ## crisp code
    _id = 1
    data_x_proc, data_y_proc = _write_patient_img(
        root=src_dir,
        patient_id=f"patient{_id:04d}"
    )
    print(f"data_x_proc: {data_x_proc.shape}, data_y_proc: {data_y_proc.shape}")
    print(f"data_x_proc: {data_x_proc.min(), data_x_proc.max()}, data_y_proc: {data_y_proc.min(), data_y_proc.max()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-ds', '--dataset-name', default='camus', help='the dataset name')
    parser.add_argument('-s', '--src-dir', help='path to the downloaded dataset')
    parser.add_argument('-d', '--dst-dir', default='./dataset', help='path to the destination dataset after preparing')
    args = parser.parse_args()

    main(
        dataset=args.dataset_name,
        src_dir=args.src_dir,
        dst_dir=args.dst_dir
    )