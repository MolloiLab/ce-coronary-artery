import marimo

__generated_with = "0.4.10"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __():
    from monai.utils import first, set_determinism
    from monai.transforms import (
        AsDiscrete,
        AsDiscreted,
        EnsureChannelFirstd,
        Compose,
        CropForegroundd,
        LoadImaged,
        Orientationd,
        RandCropByPosNegLabeld,
        SaveImaged,
        ScaleIntensityRanged,
        Spacingd,
        Invertd,
    )
    return (
        AsDiscrete,
        AsDiscreted,
        Compose,
        CropForegroundd,
        EnsureChannelFirstd,
        Invertd,
        LoadImaged,
        Orientationd,
        RandCropByPosNegLabeld,
        SaveImaged,
        ScaleIntensityRanged,
        Spacingd,
        first,
        set_determinism,
    )


@app.cell
def __():
    from monai.handlers.utils import from_engine
    from monai.networks.nets import UNet
    from monai.networks.layers import Norm
    from monai.metrics import DiceMetric
    from monai.losses import DiceLoss
    from monai.inferers import sliding_window_inference
    from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
    from monai.config import print_config
    from monai.apps import download_and_extract
    return (
        CacheDataset,
        DataLoader,
        Dataset,
        DiceLoss,
        DiceMetric,
        Norm,
        UNet,
        decollate_batch,
        download_and_extract,
        from_engine,
        print_config,
        sliding_window_inference,
    )


@app.cell
def __():
    import torch
    import matplotlib.pyplot as plt
    import tempfile
    import shutil
    import os
    import glob
    return glob, os, plt, shutil, tempfile, torch


@app.cell
def __(print_config):
    print_config()
    return


@app.cell
def __():
    # Download Dataset
    return


@app.cell
def __(os):
    # Cleaning and organizing ImageCAS dataset

    root_dir = "/dfs7/symolloi-lab/imageCAS"
    images = []
    labels = []
    for filename in os.listdir(root_dir):
        # Construct full file path
        filepath = os.path.join(root_dir, filename)
        for f in os.listdir(filepath):
            if f.startswith('img'):
                images.append( os.path.join(filepath, f))
            else:
                labels.append(os.path.join(filepath, f))

    data_set = zip(images, labels)

    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(images, labels)]

    print(data_dicts)
    return (
        data_dicts,
        data_set,
        f,
        filename,
        filepath,
        images,
        labels,
        root_dir,
    )


@app.cell
def __(data_dicts):
    print(len(data_dicts))
    train_files, val_files = data_dicts[:-9], data_dicts[-9:]
    return train_files, val_files


@app.cell
def __(set_determinism):
    # Set deterministic training for reproducibility
    set_determinism(seed=0)
    return


@app.cell
def __(
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
):
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            # user can also add other random transforms
            # RandAffined(
            #     keys=['image', 'label'],
            #     mode=('bilinear', 'nearest'),
            #     prob=1.0, spatial_size=(96, 96, 96),
            #     rotate_range=(0, 0, np.pi/15),
            #     scale_range=(0.1, 0.1, 0.1)),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ]
    )
    return train_transforms, val_transforms


@app.cell
def __(DataLoader, Dataset, first, plt, val_files, val_transforms):
    check_ds = Dataset(data=val_files, transform=val_transforms)
    check_loader = DataLoader(check_ds, batch_size=1)
    check_data = first(check_loader)
    image, label = (check_data["image"][0][0], check_data["label"][0][0])
    print(f"image shape: {image.shape}, label shape: {label.shape}")
    # plot the slice [:, :, 80]
    plt.figure("check", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("image")
    plt.imshow(image[:, :, 40], cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("label")
    plt.imshow(label[:, :, 20])
    plt.show()
    return check_data, check_ds, check_loader, image, label


if __name__ == "__main__":
    app.run()
