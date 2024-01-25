import os
import sys
from datetime import datetime
from typing import Dict

import monai
import torch
import yaml
from accelerate import Accelerator
from easydict import EasyDict
from monai.utils import ensure_tuple_rep
from objprint import objstr
from timm.optim import optim_factory

from src import utils
from src.loader import get_dataloader, get_dataloader_val_only, get_dataloader_sap, get_dataloader_sap_val_only
from src.optimizer import LinearWarmupCosineAnnealingLR
from src.SlimUNETR.SlimUNETR import SlimUNETR
from src.utils import Logger, load_pretrain_model

from skimage.transform import resize

best_acc = 0
best_class = []


def warm_up(
    model: torch.nn.Module,
    loss_functions: Dict[str, torch.nn.modules.loss._Loss],
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    accelerator: Accelerator,
    epoch: int,
    step: int,
):
    # warm_up
    model.train()
    accelerator.print(f"Start Warn Up!")
    for i, image_batch in enumerate(train_loader):
        logits = model(image_batch["image"])
        total_loss = 0
        log = ""
        for name in loss_functions:
            alpth = 1
            loss = loss_functions[name](logits, image_batch["label"])
            accelerator.log({"Train/" + name: float(loss)}, step=step)
            total_loss += alpth * loss
        accelerator.backward(total_loss)
        optimizer.step()
        optimizer.zero_grad()
        accelerator.log(
            {
                "Train/Total Loss": float(total_loss),
            },
            step=step,
        )
        step += 1
    scheduler.step(epoch)
    accelerator.print(f"Warn Up Over!")
    return step


@torch.no_grad()
def val_one_epoch(
    model: torch.nn.Module,
    inference: monai.inferers.Inferer,
    val_loader: torch.utils.data.DataLoader,
    metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
    step: int,
    post_trans: monai.transforms.Compose,
    accelerator: Accelerator,
    upsample: bool
):
    # inference
    model.eval()
    dice_acc = 0
    dice_class = []
    hd95_acc = 0
    hd95_class = []
    for i, image_batch in enumerate(val_loader):
        logits = inference(image_batch["image"], model)
        val_outputs = [post_trans(i) for i in logits]
        val_outputs_ups = []

        # Upsample to 256
        if upsample:
            for val_output in val_outputs:
                lab_res_stores = torch.zeros(val_output.shape[0], 256, 256, 256)
                # print(lab_res_stores.shape)
                # print(val_output.__len__())

                for j in range(0, val_output.shape[0]):

                    vo_np = val_output[j, :, :, :].detach().cpu().numpy()
                    lab_reshape = resize(vo_np, (256, 256, 256),
                                           mode='edge',
                                           anti_aliasing=False,
                                           anti_aliasing_sigma=None,
                                           preserve_range=True,
                                           order=0)
                    # print(lab_reshape.shape)

                    lab_res_store = torch.zeros([256, 256, 256])
                    lab_res_store[lab_reshape > 1e-5] = 1
                    lab_res_stores[j] = lab_res_store

                lab_res_stores = lab_res_stores.cuda()
                val_outputs_ups.append(lab_res_stores)

        for metric_name in metrics:
            if upsample:
                metrics[metric_name](y_pred=val_outputs_ups, y=image_batch["label"])
            else:
                metrics[metric_name](y_pred=val_outputs, y=image_batch["label"])

        accelerator.print(f"dice: {metrics['dice_metric'].get_buffer()}")
        accelerator.print(f"[{i + 1}/{len(val_loader)}] Validation Loading", flush=True)

        step += 1
    metric = {}

    for metric_name in metrics:
        batch_acc = metrics[metric_name].aggregate()
        if accelerator.num_processes > 1:
            batch_acc = (
                    accelerator.reduce(batch_acc.to(accelerator.device))
                    / accelerator.num_processes
            )
        metrics[metric_name].reset()
        if metric_name == "dice_metric":
            metric.update(
                {
                    f"Val/mean {metric_name}": float(batch_acc.mean()),
                    f"Val/label1 {metric_name}": float(batch_acc[0]),
                    f"Val/label2 {metric_name}": float(batch_acc[1]),
                }
            )
            dice_acc = torch.Tensor([metric["Val/mean dice_metric"]]).to(
                accelerator.device
            )
            dice_class = batch_acc
        else:
            metric.update(
                {
                    f"Val/mean {metric_name}": float(batch_acc.mean()),
                    f"Val/label1 {metric_name}": float(batch_acc[0]),
                    f"Val/label2 {metric_name}": float(batch_acc[1]),
                }
            )
            hd95_acc = torch.Tensor([metric["Val/mean hd95_metric"]]).to(
                accelerator.device
            )
            hd95_class = batch_acc

    return dice_acc, dice_class, hd95_acc, hd95_class

if __name__ == "__main__":

    device_num = 2
    torch.cuda.set_device(device_num)

    # load yml
    config = EasyDict(
        yaml.load(open("config.yml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
    )

    utils.same_seeds(50)
    #logging_dir = os.getcwd() + "/logs/" + str(datetime.now())
    logging_dir = os.path.join(f'{config.work_dir}','inference','logs',f'{config.finetune.checkpoint}',f'{str(datetime.now())}')
    accelerator = Accelerator(cpu=False)
    Logger(logging_dir if accelerator.is_local_main_process else None)
    accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])
    accelerator.print(objstr(config))

    accelerator.print("load model...")
    '''
    model = SlimUNETR(
        in_channels=1,
        out_channels=2,
        embed_dim=96,
        embedding_dim=64,
        channels=(24, 48, 60),
        blocks=(1, 2, 3, 2),
        heads=(1, 2, 4, 4),
        r=(4, 2, 2, 1),
        dropout=0.3)
    '''
    model = SlimUNETR(**config.finetune.slim_unetr)
    image_size = config.trainer.image_size

    accelerator.print("load dataset...")
    _, val_loader = get_dataloader(config)
    # _, val_loader = get_dataloader_sap(config)
    # val_loader = get_dataloader_val_only(config)
    # val_loader = get_dataloader_sap_val_only(config)

    inference = monai.inferers.SlidingWindowInferer(
        roi_size=ensure_tuple_rep(image_size, 3),
        overlap=0.5,
        sw_device=accelerator.device,
        device=accelerator.device,
    )
    loss_functions = {
        "focal_loss": monai.losses.FocalLoss(to_onehot_y=False),
        "dice_loss": monai.losses.DiceLoss(
            smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=True
        ),
    }
    metrics = {
        "dice_metric": monai.metrics.DiceMetric(
            include_background=True,
            reduction=monai.utils.MetricReduction.MEAN_BATCH,
            get_not_nans=False,
        ),
        "hd95_metric": monai.metrics.HausdorffDistanceMetric(
            percentile=95,
            include_background=True,
            reduction=monai.utils.MetricReduction.MEAN_BATCH,
            get_not_nans=False,
        ),
    }
    post_trans = monai.transforms.Compose(
        [
            monai.transforms.Activations(sigmoid=True),
            monai.transforms.AsDiscrete(threshold=0.5),
        ]
    )

    step = 0
    best_eopch = -1
    val_step = 0

    # load pre-train model
    model = load_pretrain_model(
        # f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/best/pytorch_model.bin",
        #f"/data/jionkim/Slim_UNETR/model_store/{config.finetune.checkpoint}/best/pytorch_model.bin",
        os.path.join(f'{config.work_dir}',f'{config.inference.pathseg}','model_store',f'{config.finetune.checkpoint}',f'{("epoch_"+f"{config.inference.epoch:05d}") if isinstance(config.inference.epoch, int) else "best"}','pytorch_model.bin'),
        model,
        accelerator,
    )

    model, val_loader = accelerator.prepare(
        model, val_loader
    )

    dice_acc, dice_class, hd95_acc, hd95_class = val_one_epoch(
        model,
        inference,
        val_loader,
        metrics,
        val_step,
        post_trans,
        accelerator,
        False,
    )

    accelerator.print(f"dice acc: {dice_acc}")
    accelerator.print(f"dice class : {dice_class}")
    accelerator.print(f"hd95 acc: {hd95_acc}")
    accelerator.print(f"hd95 class : {hd95_class}")

    sys.exit(0)
