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
from src.loader import get_dataloader
from src.optimizer import LinearWarmupCosineAnnealingLR
from src.SlimUNETR.SlimUNETR import SlimUNETR
from src.utils import Logger, same_seeds


def train_one_epoch(
    model: torch.nn.Module,
    config: EasyDict,
    loss_functions: Dict[str, torch.nn.modules.loss._Loss],
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
    post_trans: monai.transforms.Compose,
    accelerator: Accelerator,
    epoch: int,
    step: int,
):
    # train
    model.train()
    for i, image_batch in enumerate(train_loader):
        # accelerator.print(str(i) + ' step started.')
        # accelerator.print(datetime.now())
        img_input = image_batch["image"]
        label_input = image_batch["label"]
        logits = model(img_input)
        total_loss = 0
        log = ""
        for name in loss_functions:
            alpth = 1
            loss = loss_functions[name](logits, label_input)
            accelerator.log({"Train/" + name: float(loss)}, step=step)
            total_loss += alpth * loss
        val_outputs = [post_trans(i) for i in logits]
        for metric_name in metrics:
            metrics[metric_name](y_pred=val_outputs, y=image_batch["label"])

        accelerator.backward(total_loss)
        optimizer.step()
        optimizer.zero_grad()
        accelerator.log(
            {
                "Train/Total Loss": float(total_loss),
            },
            step=step,
        )
        accelerator.print(
            f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] Training [{i + 1}/{len(train_loader)}] Loss: {total_loss:1.5f} {log}",
            flush=True,
        )
        step += 1
        # accelerator.print(str(i) + ' step finished.')
        # accelerator.print(datetime.now())
    scheduler.step(epoch)
    metric = {}
    for metric_name in metrics:
        batch_acc = metrics[metric_name].aggregate()
        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes
        metric.update(
            {
                f"Train/mean {metric_name}": float(batch_acc.mean()),
                f"Train/label1 {metric_name}": float(batch_acc[0]),
                f"Train/label2 {metric_name}": float(batch_acc[1]),
            }
        )
    accelerator.print(
        f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] Training metric {metric}"
    )
    accelerator.log(metric, step=epoch)
    return step


@torch.no_grad()
def val_one_epoch(
    model: torch.nn.Module,
    loss_functions: Dict[str, torch.nn.modules.loss._Loss],
    inference: monai.inferers.Inferer,
    val_loader: torch.utils.data.DataLoader,
    config: EasyDict,
    metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
    step: int,
    post_trans: monai.transforms.Compose,
    accelerator: Accelerator,
    epoch: int,
):
    # val
    model.eval()
    for i, image_batch in enumerate(val_loader):
        logits = inference(image_batch["image"], model)
        total_loss = 0
        log = ""
        for name in loss_functions:
            loss = loss_functions[name](logits, image_batch["label"])
            accelerator.log({"Val/" + name: float(loss)}, step=step)
            log += f" {name} {float(loss):1.5f} "
            total_loss += loss
        val_outputs = [post_trans(i) for i in logits]
        for metric_name in metrics:
            metrics[metric_name](y_pred=val_outputs, y=image_batch["label"])
        accelerator.log(
            {
                "Val/Total Loss": float(total_loss),
            },
            step=step,
        )
        accelerator.print(
            f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] Validation [{i + 1}/{len(val_loader)}] Loss: {total_loss:1.5f} {log}",
            flush=True,
        )
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
        metric.update(
            {
                f"Val/mean {metric_name}": float(batch_acc.mean()),
                f"Val/label1 {metric_name}": float(batch_acc[0]),
                f"Val/label2 {metric_name}": float(batch_acc[1])
            }
        )
    accelerator.print(
        f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] Validation metric {metric}"
    )
    accelerator.log(metric, step=epoch)
    return (
        torch.Tensor([metric["Val/mean dice_metric"]]).to(accelerator.device),
        batch_acc,
        step,
    )


if __name__ == "__main__":

    device_num = 4
    torch.cuda.set_device(device_num)

    # load yml
    config = EasyDict(
        yaml.load(open("config.yml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
    )

    same_seeds(50)
    #logging_dir = os.getcwd() + "/logs/" + str(datetime.now())
    logging_dir = os.path.join(f'{config.work_dir}','train','logs',f'{config.finetune.checkpoint}',f'{str(datetime.now())}')
    accelerator = Accelerator(
        cpu=False, log_with=["tensorboard"], project_dir=logging_dir
    )
    Logger(logging_dir if accelerator.is_local_main_process else None)
    accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])
    accelerator.print(objstr(config))

    accelerator.print("Load Model...")
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
        dropout=0.3,
    )
    '''
    model = SlimUNETR(**config.finetune.slim_unetr)
    image_size = config.trainer.image_size

    accelerator.print("Load Dataloader...")
    train_loader, val_loader = get_dataloader(config)

    inference = monai.inferers.SlidingWindowInferer(
        roi_size=ensure_tuple_rep(image_size, 3),
        overlap=0.5,
        sw_device=accelerator.device,
        device=accelerator.device,
    )
    metrics = {
        "dice_metric": monai.metrics.DiceMetric(
            include_background=True,
            reduction=monai.utils.MetricReduction.MEAN_BATCH,
            get_not_nans=False,
        ),
        # 'hd95_metric': monai.metrics.HausdorffDistanceMetric(percentile=95, include_background=True, reduction=monai.utils.MetricReduction.MEAN_BATCH, get_not_nans=False)
    }
    post_trans = monai.transforms.Compose(
        [
            monai.transforms.Activations(sigmoid=True),
            monai.transforms.AsDiscrete(threshold=0.5),
        ]
    )

    optimizer = optim_factory.create_optimizer_v2(
        model,
        opt=config.trainer.optimizer,
        weight_decay=config.trainer.weight_decay,
        lr=config.trainer.lr,
        betas=(0.9, 0.95),
    )
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=config.trainer.warmup,
        max_epochs=config.trainer.num_epochs,
    )
    loss_functions = {
        "focal_loss": monai.losses.FocalLoss(to_onehot_y=False),
        "dice_loss": monai.losses.DiceLoss(
            smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=True
        ),
    }

    step = 0
    best_eopch = -1
    val_step = 0
    starting_epoch = 0
    best_acc = 0
    best_class = []

    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader, val_loader
    )

    # resume training
    if config.trainer.resume:
        model, starting_epoch, step, val_step = utils.resume_train_state(
            model, "{}".format(config.finetune.checkpoint), train_loader, accelerator
        )

    # Start Training
    accelerator.print("Start Training！")
    accelerator.print(datetime.now())
    for epoch in range(starting_epoch, config.trainer.num_epochs):
        # train
        step = train_one_epoch(
            model,
            config,
            loss_functions,
            train_loader,
            optimizer,
            scheduler,
            metrics,
            post_trans,
            accelerator,
            epoch,
            step,
        )
        # val
        mean_acc, batch_acc, val_step = val_one_epoch(
            model,
            loss_functions,
            inference,
            val_loader,
            config,
            metrics,
            val_step,
            post_trans,
            accelerator,
            epoch,
        )

        accelerator.print(
            f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] lr = {scheduler.get_last_lr()} best acc: {best_acc}, mean acc: {mean_acc}, mean class: {batch_acc}"
        )

        # save model
        if mean_acc > best_acc:
            accelerator.save_state(
                # output_dir=f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/best"
                #output_dir=f"/data/jionkim/Slim_UNETR/model_store/{config.finetune.checkpoint}/best"
                output_dir=os.path.join(f'{config.work_dir}','train','model_store',f'{config.finetune.checkpoint}','best'),
                safe_serialization = False
            )
            best_acc = mean_acc
            best_class = batch_acc
            best_eopch = epoch
        if (epoch+1) % max(config.trainer.save_cycle,1)==0 or (epoch+1)==config.trainer.num_epochs:
            accelerator.save_state(
                # output_dir=f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/epoch_{epoch}"
                #output_dir=f"/data/jionkim/Slim_UNETR/model_store/{config.finetune.checkpoint}/epoch_{epoch}"
                output_dir=os.path.join(f'{config.work_dir}','train','model_store',f'{config.finetune.checkpoint}',f'epoch_{epoch+1:05d}'),
                safe_serialization = False
            )

    accelerator.print(f"best dice mean acc: {best_acc}")
    accelerator.print(f"best dice accs: {best_class}")
    accelerator.print(datetime.now())
    sys.exit(1)
