import os
import sys
from datetime import datetime

import monai
import torch
import yaml
import time
from accelerate import Accelerator
from easydict import EasyDict
from monai.utils import ensure_tuple_rep
from objprint import objstr
import numpy as np
import nibabel as nib
import nrrd

from src import utils
from src.loader import get_dataloader
from src.SlimUNETR.SlimUNETR import SlimUNETR
from src.utils import Logger, load_pretrain_model

from skimage.transform import resize

best_acc = 0
best_class = []

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
    upsample: bool
):
    start = time.time()
    # inference
    time_accum = 0
    model.eval()
    
    savePath = os.path.join(config.work_dir,'generation',config.finetune.checkpoint)
    os.makedirs(savePath,exist_ok=True)

    for i, image_batch in enumerate(val_loader):

        logits = inference(image_batch["image"], model)
        val_outputs = [post_trans(i) for i in logits]
        val_outputs_ups = []

        if upsample:
            for val_output in val_outputs:
                lab_res_stores = torch.zeros(val_output.shape[0], 256, 256, 256)

                for j in range(0, val_output.shape[0]):

                    vo_np = val_output[j, :, :, :].detach().cpu().numpy()
                    lab_reshape = resize(vo_np, (256, 256, 256),
                                           mode='edge',
                                           anti_aliasing=False,
                                           anti_aliasing_sigma=None,
                                           preserve_range=True,
                                           order=0)

                    lab_res_store = torch.zeros([256, 256, 256])
                    lab_res_store[lab_reshape > 1e-5] = 1
                    lab_res_stores[j] = lab_res_store

                lab_res_stores = lab_res_stores.cuda()
                val_outputs_ups.append(lab_res_stores)

        img = image_batch["image"][0][0, :, :, :].cpu().detach().numpy()
        if upsample:
            img = resize(img, (256, 256, 256))
            lab_gen = val_outputs_ups[0].cpu().detach().numpy().astype(bool)
        else:
            lab_gen = val_outputs[0].cpu().detach().numpy().astype(bool)

        lab_gt = image_batch["label"][0].cpu().detach().numpy().astype(bool)
        
        np.save(os.path.join(savePath,f'inf_img_{i:02}.npy'),img)
        np.save(os.path.join(savePath,f'inf_lab_gen_{i:02}.npy'),lab_gen)
        np.save(os.path.join(savePath,f'inf_lab_gt_{i:02}.npy'),lab_gt)
        
        accelerator.print(f"[{i + 1}/{len(val_loader)}] Validation Loading", flush=True)
        step += 1

    accelerator.print(f"Generation Over!")
    done = time.time()
    elapsed = done - start
    print(elapsed)

if __name__ == "__main__":

    print(torch.cuda.is_available())

    start = time.time()

    device_num = 1
    torch.cuda.set_device(device_num)

    # load yml
    config = EasyDict(
        yaml.load(open("config.yml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
    )

    utils.same_seeds(50)
    # logging_dir = os.getcwd() + "/logs/" + str(datetime.now())
    accelerator = Accelerator(cpu=False)
    # Logger(logging_dir if accelerator.is_local_main_process else None)
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

    step = 0
    best_epoch = -1
    val_step = 0

    # load pre-train model
    model = load_pretrain_model(
        # f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/best/pytorch_model.bin",
        os.path.join(f'{config.work_dir}',f'{config.generation.pathseg}','model_store',f'{config.finetune.checkpoint}',f'{("epoch_"+f"{config.generation.epoch:05d}") if isinstance(config.generation.epoch, int) else "best"}','pytorch_model.bin'),
        #f"/data/jionkim/Slim_UNETR/model_store/{config.finetune.checkpoint}/best/pytorch_model.bin",
        # f"J:/Program/Slim-UNETR/output/231215_1_res_128_batch_4/ckpt/best/pytorch_model.bin",
        model,
        accelerator,
    )

    model, val_loader = accelerator.prepare(
        model, val_loader
    )

    # start inference
    accelerator.print("Start Val！")
    print(torch.cuda.memory_allocated() / 1024 / 1024,' MiB')

    gen_seg_result(
        model,
        # config,
        inference,
        val_loader,
        # metrics,
        val_step,
        post_trans,
        accelerator,
        False
    )

    done = time.time()
    elapsed = done - start
    print(elapsed)
    print(torch.cuda.max_memory_allocated() / 1024 / 1024,' MiB')

    sys.exit(1)

    '''
    np.save('/data/jionkim/Slim_UNETR/inference/intf_img_' + str(i).zfill(2) + '.npy', img)
    np.save('/data/jionkim/Slim_UNETR/inference/intf_lab1_gen_' + str(i).zfill(2) + '.npy', lab1_gen)
    np.save('/data/jionkim/Slim_UNETR/inference/intf_lab2_gen_' + str(i).zfill(2) + '.npy', lab2_gen)
    np.save('/data/jionkim/Slim_UNETR/inference/intf_lab1_gt_' + str(i).zfill(2) + '.npy', lab1_gt)
    np.save('/data/jionkim/Slim_UNETR/inference/intf_lab2_gt_' + str(i).zfill(2) + '.npy', lab2_gt)
    '''
    '''
    np.save('J:/Program/Slim-UNETR/inference/intf_img_' + str(i).zfill(2) + '.npy', img)
    np.save('J:/Program/Slim-UNETR/inference/intf_lab1_gen_' + str(i).zfill(2) + '.npy', lab1_gen)
    np.save('J:/Program/Slim-UNETR/inference/intf_lab2_gen_' + str(i).zfill(2) + '.npy', lab2_gen)
    np.save('J:/Program/Slim-UNETR/inference/intf_lab1_gt_' + str(i).zfill(2) + '.npy', lab1_gt)
    np.save('J:/Program/Slim-UNETR/inference/intf_lab2_gt_' + str(i).zfill(2) + '.npy', lab2_gt)
    '''

'''
def save_inference(i:int, imageList:list, mode:str = None, separate:bool = False):
    
            np.save('./inference/intf_img_' + str(i).zfill(2) + '.npy', img)
        np.save('./inference/intf_lab_gen_' + str(i).zfill(2) + '.npy', lab_gen)
        np.save('./inference/intf_lab_gt_' + str(i).zfill(2) + '.npy', lab_gt)
'''