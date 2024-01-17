import json
import os
from typing import Dict, Hashable, Mapping, Tuple

import monai
import numpy as np
import torch
from easydict import EasyDict
from monai.utils import ensure_tuple_rep


class ConvertToMultiChannelClassesd(monai.transforms.MapTransform):
    """
    TC WT ET
    Dictionary-based wrapper of :py:class:`monai.transforms.ConvertToMultiChannelBasedOnBratsClasses`.
    Convert labels to multi channels based on brats18 classes:
    label 1 is the necrotic and non-enhancing tumor core
    label 2 is the peritumoral edema
    label 4 is the GD-enhancing tumor
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """

    backend = [monai.utils.TransformBackends.TORCH, monai.utils.TransformBackends.NUMPY]

    def __init__(
        self,
        keys: monai.config.KeysCollection,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)

    def converter(self, img: monai.config.NdarrayOrTensor):

        result = [img == 1, img == 2]
        # result = [img == 1]

        return (
            torch.stack(result, dim=0)
            if isinstance(img, torch.Tensor)
            else np.stack(result, axis=0)
        )

    def __call__(
        self, data: Mapping[Hashable, monai.config.NdarrayOrTensor]
    ) -> Dict[Hashable, monai.config.NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


def load_dataset_images(root):
    images_list = []

    for i in range (0, 267):
        img = root + "image/image_" + str(i).zfill(3) + ".nii.gz"
        seg_img = root + "label/label_" + str(i).zfill(3) + ".nii.gz"
        images_list.append(
            {"image": img, "label": seg_img}
        )
    '''
    for i in range(0, 40):
        img = "J:\\Dataset\\TEE-Labeling\\_TTE_images\\image\\nii_gz_128\\image_" + str(i).zfill(3) + ".nii.gz"
        seg_img = "J:\\Dataset\\TEE-Labeling\\_TTE_images\\label\\nii_gz_128\\label_" + str(i).zfill(3) + ".nii.gz"
        images_list.append(
            {"image": img, "label": seg_img}
        )

    for i in range(0, 15):
        img = root + "image/1_label_" + str(i).zfill(3) + "_nm.nii.gz"
        seg_img = root + "label/1_label_" + str(i).zfill(3) + ".nii.gz"
        images_list.append(
            {"image": img, "label": seg_img}
        )
        '''
    return images_list

# Saparate train images to load > get_dataloader_sap()
def load_dataset_train_images_sap(root):
    images_list = []

    for i in range (0, 221):
        img = root + "train/image/image_" + str(i).zfill(3) + ".nii.gz"
        seg_img = root + "train/label/label_" + str(i).zfill(3) + ".nii.gz"
        images_list.append(
            {"image": img, "label": seg_img}
        )

    return images_list

# Saparate test images to load > get_dataloader_sap()
def load_dataset_test_images_sap(root):
    images_list = []

    for i in range(0, 46):
        img = root + "test/image/image_" + str(i).zfill(3) + ".nii.gz"
        seg_img = root + "test/label/label_" + str(i).zfill(3) + ".nii.gz"
        images_list.append(
            {"image": img, "label": seg_img}
        )

    return images_list


def get_transforms(
    config: EasyDict,
) -> Tuple[monai.transforms.Compose, monai.transforms.Compose]:
    train_transform = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=["image", "label"]),
            monai.transforms.EnsureChannelFirstd(keys="image"),
            monai.transforms.EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelClassesd(keys=["label"]),
            monai.transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            monai.transforms.SpatialPadD(
                keys=["image", "label"],
                spatial_size=(255, 255, config.trainer.image_size),
                method="symmetric",
                mode="constant",
            ),
            monai.transforms.Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            monai.transforms.CenterSpatialCropD(
                keys=["image", "label"],
                roi_size=ensure_tuple_rep(config.trainer.image_size, 3),
            ),
            monai.transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                num_samples=2,
                spatial_size=ensure_tuple_rep(config.trainer.image_size, 3),
                pos=1,
                neg=1,
                image_key="image",
                image_threshold=0,
            ),
            monai.transforms.RandFlipd(
                keys=["image", "label"], prob=0.5, spatial_axis=0
            ),
            monai.transforms.RandFlipd(
                keys=["image", "label"], prob=0.5, spatial_axis=1
            ),
            monai.transforms.RandFlipd(
                keys=["image", "label"], prob=0.5, spatial_axis=2
            ),
            monai.transforms.NormalizeIntensityd(
                keys="image", nonzero=True, channel_wise=True
            ),
            monai.transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            monai.transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            monai.transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=["image", "label"]),
            monai.transforms.EnsureChannelFirstd(keys="image"),
            monai.transforms.EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelClassesd(keys="label"),
            monai.transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            monai.transforms.Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            monai.transforms.NormalizeIntensityd(
                keys="image", nonzero=True, channel_wise=True
            ),
        ]
    )
    return train_transform, val_transform


def get_dataloader(
    config: EasyDict
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:

    train_images = load_dataset_images(config.data_root)

    train_transform, val_transform = get_transforms(config)

    train_dataset = monai.data.Dataset(
        data=train_images[: int(len(train_images) * config.trainer.train_ratio)],
        transform=train_transform,
    )
    val_dataset = monai.data.Dataset(
        data=train_images[int(len(train_images) * config.trainer.train_ratio) :],
        transform=val_transform,
    )

    train_loader = monai.data.DataLoader(
        train_dataset,
        num_workers=config.trainer.num_workers,
        batch_size=config.trainer.batch_size,
        shuffle=True,
    )

    batch_size = config.trainer.batch_size

    val_loader = monai.data.DataLoader(
        val_dataset,
        num_workers=config.trainer.num_workers,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader


def get_dataloader_val_only(
    config: EasyDict
) -> [torch.utils.data.DataLoader]:

    train_images = load_dataset_images(config.data_root)

    _, val_transform = get_transforms(config)

    val_dataset = monai.data.Dataset(
        data=train_images,
        transform=val_transform,
    )

    batch_size = config.trainer.batch_size

    val_loader = monai.data.DataLoader(
        val_dataset,
        num_workers=config.trainer.num_workers,
        batch_size=batch_size,
        shuffle=False,
    )

    return val_loader

# Saparate train images and test images in different directory
def get_dataloader_sap(
    config: EasyDict
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:

    train_images = load_dataset_train_images_sap(config.data_root)
    test_images = load_dataset_test_images_sap(config.data_root)

    train_transform, val_transform = get_transforms(config)

    train_dataset = monai.data.Dataset(
        data=train_images,
        transform=train_transform,
    )
    val_dataset = monai.data.Dataset(
        data=test_images,
        transform=val_transform,
    )

    train_loader = monai.data.DataLoader(
        train_dataset,
        num_workers=config.trainer.num_workers,
        batch_size=config.trainer.batch_size,
        shuffle=True,
    )

    batch_size = config.trainer.batch_size

    val_loader = monai.data.DataLoader(
        val_dataset,
        num_workers=config.trainer.num_workers,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader