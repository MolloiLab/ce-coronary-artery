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
        RandSpatialCropd,
        SaveImaged,
        ScaleIntensityRanged,
        Spacingd,
        Invertd,
        CenterSpatialCropd,

        ResizeWithPadOrCropd
    )
    return (
        AsDiscrete,
        AsDiscreted,
        CenterSpatialCropd,
        Compose,
        CropForegroundd,
        EnsureChannelFirstd,
        Invertd,
        LoadImaged,
        Orientationd,
        RandCropByPosNegLabeld,
        RandSpatialCropd,
        ResizeWithPadOrCropd,
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
    from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch, pad_list_data_collate
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
        pad_list_data_collate,
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


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r'''
        # Downloading and organizing the ImageCAS dataset
        '''
    )
    return


@app.cell
def __(os):
    # Cleaning and organizing ImageCAS dataset

    root_dir = "/dfs7/symolloi-lab/imageCAS"
    global_images = []
    global_labels = []
    for filename in os.listdir(root_dir):
        # Construct full file path
        filepath = os.path.join(root_dir, filename)
        for f in os.listdir(filepath):
            if f.startswith('img'):
                global_images.append( os.path.join(filepath, f))
            else:
                global_labels.append(os.path.join(filepath, f))

    data_set = zip(global_images, global_labels)

    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(global_images, global_labels)]

    print(data_dicts)
    return (
        data_dicts,
        data_set,
        f,
        filename,
        filepath,
        global_images,
        global_labels,
        root_dir,
    )


@app.cell
def __(data_dicts):
    print(len(data_dicts))
    train_files, val_files = data_dicts[:-975], data_dicts[-975:]
    print(len(train_files))
    print(len(val_files))
    return train_files, val_files


@app.cell
def __(set_determinism):
    # Set deterministic training for reproducibility
    set_determinism(seed=0)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r'''
        # Setting up transforms for Training & Validation
        '''
    )
    return


@app.cell
def __():
    spatial_size = [224, 224, 112]
    return spatial_size,


@app.cell
def __(
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    ResizeWithPadOrCropd,
    ScaleIntensityRanged,
    spatial_size,
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
            # CenterSpatialCropd(keys=["image", "label"], roi_size=image_size),
            # CropForegroundd(keys=["image", "label"], source_key="label"),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size = spatial_size),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
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
            # CenterSpatialCropd(keys=["image", "label"], roi_size=image_size),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size = spatial_size),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
        ]
    )
    return train_transforms, val_transforms


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"# Checking transforms in  DataLoader")
    return


@app.cell
def __(DataLoader, Dataset, first, val_files, val_transforms):
    check_ds = Dataset(data=val_files, transform=val_transforms)
    check_loader = DataLoader(check_ds, batch_size=2)
    check_data = first(check_loader)
    image, label = (check_data["image"][0][0], check_data["label"][0][0])
    return check_data, check_ds, check_loader, image, label


@app.cell
def __(image, mo):
    z = mo.ui.slider(start=0, stop=image.shape[2])
    z
    return z,


@app.cell
def __(image, label, plt, z):
    print(f"image shape: {image.shape}, label shape: {label.shape}")
    # plot the slice [:, :, 80]
    plt.figure("check", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("image")
    plt.imshow(image[:, :, z.value], cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("label")
    plt.imshow(label[:, :, z.value])
    plt.show()
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r'''
        # Define CacheDataset and DataLoader for training and validation
        '''
    )
    return


@app.cell
def __():
    991/4
    return


@app.cell
def __(
    CacheDataset,
    DataLoader,
    Dataset,
    pad_list_data_collate,
    train_files,
    train_transforms,
    val_files,
    val_transforms,
):
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
    # train_ds = Dataset(data=train_files, transform=train_transforms)

    # use batch_size=2 to load images and use RandCropByPosNegLabeld
    # to generate 2 x 4 images for network training
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,
                              num_workers=1, collate_fn=pad_list_data_collate)

    # val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)
    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=1)

    # debug_loader = DataLoader(train_ds, batch_size=1, num_workers=0)
    # for i, bruh in enumerate(debug_loader):
    #     print({key: value.shape for key, value in bruh.items()})
    #     if i > 10:  # Limit the number of batches to inspect
    #         break
    return train_ds, train_loader, val_ds, val_loader


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"# Create Model, Loss, Optimizer (Dice Loss)")
    return


@app.cell
def __(DiceLoss, DiceMetric, Norm, UNet, torch):
    # standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda:0")
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    return device, dice_metric, loss_function, model, optimizer


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r'''
        # Execute PyTorch training process
        '''
    )
    return


@app.cell
def __(
    AsDiscrete,
    Compose,
    decollate_batch,
    device,
    dice_metric,
    loss_function,
    model,
    optimizer,
    os,
    root_dir,
    sliding_window_inference,
    torch,
    train_ds,
    train_loader,
    val_loader,
):
    max_epochs = 100
    val_interval = 101
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    roi_size = (160, 160, 160)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)

                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()

                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )
    return (
        batch_data,
        best_metric,
        best_metric_epoch,
        epoch,
        epoch_loss,
        epoch_loss_values,
        inputs,
        labels,
        loss,
        max_epochs,
        metric,
        metric_values,
        outputs,
        post_label,
        post_pred,
        roi_size,
        step,
        sw_batch_size,
        val_data,
        val_inputs,
        val_interval,
        val_labels,
        val_outputs,
    )


@app.cell
def __(best_metric, best_metric_epoch):
    print(f"train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")
    return


@app.cell
def __(epoch_loss_values, metric_values, plt, val_interval):
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.show()
    return x, y


if __name__ == "__main__":
    app.run()
