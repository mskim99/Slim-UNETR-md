import os
import sys
from datetime import datetime
from typing import Dict

import monai
import pytz
import torch
import yaml
import time
from accelerate import Accelerator
from easydict import EasyDict
from monai.utils import ensure_tuple_rep
from objprint import objstr
from timm.optim import optim_factory
import numpy as np

from src import utils
from src.loader import get_dataloader, get_dataloader_val_only
from src.optimizer import LinearWarmupCosineAnnealingLR
from src.SlimUNETR.SlimUNETR import SlimUNETR
from src.utils import Logger, load_pretrain_model

best_acc = 0
best_class = []

'''
def warm_up(
    model: torch.nn.Module,
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
    # warm_up
    model.train()
    accelerator.print(f"Start Warn Up!")
    for i, image_batch in enumerate(train_loader):
        logits = model(image_batch["image"])
        total_loss = 0
        # log = ""
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
    '''

@torch.no_grad()
def gen_seg_result(
    model: torch.nn.Module,
    # config: EasyDict,
    inference: monai.inferers.Inferer,
    val_loader: torch.utils.data.DataLoader,
    # metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
    step: int,
    post_trans: monai.transforms.Compose,
    accelerator: Accelerator,
):
    start = time.time()
    # inference
    time_accum = 0
    model.eval()
    '''
    dice_acc = 0
    dice_class = []
    hd95_acc = 0
    hd95_class = []
    '''
    for i, image_batch in enumerate(val_loader):
        '''
        print(image_batch["image"][0].shape)
        print(image_batch["label"][0].shape)
        '''
        logits = inference(image_batch["image"], model)
        val_outputs = [post_trans(i) for i in logits]
        img = image_batch["image"][0][0, :, :, :].cpu().detach().numpy()
        lab1_gen = val_outputs[0][0, :, :, :].cpu().detach().numpy().astype(bool)
        lab2_gen = val_outputs[0][1, :, :, :].cpu().detach().numpy().astype(bool)
        lab1_gt = image_batch["label"][0][0, :, :, :].cpu().detach().numpy().astype(bool)
        lab2_gt = image_batch["label"][0][1, :, :, :].cpu().detach().numpy().astype(bool)
        '''
        np.save('./interference/intf_img_' + str(i).zfill(2) + '.npy', img)
        np.save('./interference/intf_lab1_gen_' + str(i).zfill(2) + '.npy', lab1_gen)
        np.save('./interference/intf_lab2_gen_' + str(i).zfill(2) + '.npy', lab2_gen)
        np.save('./interference/intf_lab1_gt_' + str(i).zfill(2) + '.npy', lab1_gt)
        np.save('./interference/intf_lab2_gt_' + str(i).zfill(2) + '.npy', lab2_gt)
        '''

        np.save('/data/jionkim/Slim_UNETR/inference/intf_img_' + str(i).zfill(2) + '.npy', img)
        np.save('/data/jionkim/Slim_UNETR/inference/intf_lab1_gen_' + str(i).zfill(2) + '.npy', lab1_gen)
        np.save('/data/jionkim/Slim_UNETR/inference/intf_lab2_gen_' + str(i).zfill(2) + '.npy', lab2_gen)
        np.save('/data/jionkim/Slim_UNETR/inference/intf_lab1_gt_' + str(i).zfill(2) + '.npy', lab1_gt)
        np.save('/data/jionkim/Slim_UNETR/inference/intf_lab2_gt_' + str(i).zfill(2) + '.npy', lab2_gt)

        '''
        for metric_name in metrics:
            metrics[metric_name](y_pred=val_outputs, y=image_batch["label"])
            '''
        accelerator.print(f"[{i + 1}/{len(val_loader)}] Validation Loading", flush=True)
        step += 1

    accelerator.print(f"Generation Over!")
    done = time.time()
    elapsed = done - start
    print(elapsed)
    '''
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
    '''

if __name__ == "__main__":

    start = time.time()
    device_num = 4
    torch.cuda.set_device(device_num)

    # load yml
    config = EasyDict(
        yaml.load(open("config.yml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
    )

    utils.same_seeds(50)
    logging_dir = os.getcwd() + "/logs/" + str(datetime.now())
    accelerator = Accelerator(cpu=False)
    Logger(logging_dir if accelerator.is_local_main_process else None)
    accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])
    accelerator.print(objstr(config))

    accelerator.print("load model...")
    model = SlimUNETR(
        in_channels=1,
        out_channels=2,
        embed_dim=96,
        embedding_dim=216,
        channels=(24, 48, 60),
        blocks=(1, 2, 3, 2),
        heads=(1, 2, 4, 4),
        r=(4, 2, 2, 1),
        dropout=0.3)
    image_size = config.trainer.image_size

    accelerator.print("load dataset...")
    train_loader, val_loader = get_dataloader(config)
    # val_loader = get_dataloader_val_only(config)

    inference = monai.inferers.SlidingWindowInferer(
        roi_size=ensure_tuple_rep(image_size, 3),
        overlap=0.5,
        sw_device=accelerator.device,
        device=accelerator.device,
    )

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

    step = 0
    best_epoch = -1
    val_step = 0

    # load pre-train model
    model = load_pretrain_model(
        # f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/best/pytorch_model.bin",
        f"/data/jionkim/Slim_UNETR/model_store/{config.finetune.checkpoint}/best/pytorch_model.bin",
        model,
        accelerator,
    )
    '''
    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader, val_loader
    )
    '''

    model, optimizer, scheduler, val_loader = accelerator.prepare(
        model, optimizer, scheduler, val_loader
    )

    # start inference
    accelerator.print("Start Val！")

    gen_seg_result(
        model,
        # config,
        inference,
        val_loader,
        # metrics,
        val_step,
        post_trans,
        accelerator,
    )

    done = time.time()
    elapsed = done - start
    print(elapsed)

    sys.exit(1)
