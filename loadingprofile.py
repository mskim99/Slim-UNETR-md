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
#import nrrd
from torch.profiler import profile, record_function, ProfilerActivity

from src import utils
from src.loader import get_dataloader
from src.SlimUNETR.SlimUNETR import SlimUNETR
from src.utils import load_pretrain_model

from skimage.transform import resize


def testing_epoch(
    model: torch.nn.Module,
    config: EasyDict,
    train_loader: torch.utils.data.DataLoader,
    post_trans: monai.transforms.Compose,
    accelerator: Accelerator,
    step: int,
):
    accelerator.print(f'testing_start current GPU memory usage: {torch.cuda.memory_allocated()/1024/1024} Mib')
    
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA], record_shapes=True) as prof:
        for i, image_batch in enumerate(train_loader):

            img_input = image_batch["image"]
            label_input = image_batch["label"]
            
            if i > 10:
                break
            #logits = model(img_input)

    prof.export_chrome_trace("dataloader_trace.json")  # Chrome Trace 파일로 내보내기
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))  # CPU 시간 기준으로 상위 연산 출력
    accelerator.print(f'testing_start current GPU memory usage: {torch.cuda.memory_allocated()/1024/1024} Mib')
    
    
if __name__ == "__main__":
    
    print(torch.cuda.is_available())

    start = time.time()

    device_num = 0
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

    testing_epoch(
            model,
            config,
            train_loader,
            post_trans,
            accelerator,
            step,
            )

    done = time.time()
    elapsed = done - start
    print(elapsed)
    print(torch.cuda.max_memory_allocated() / 1024 / 1024,' MiB')

    sys.exit(0)