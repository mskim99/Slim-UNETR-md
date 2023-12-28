import time

import torch
import yaml
from easydict import EasyDict

from src.SlimUNETR.SlimUNETR import SlimUNETR


def weight_test(model, x):
    start_time = time.time()
    _ = model(x)
    end_time = time.time()
    need_time = end_time - start_time
    from thop import profile

    flops, params = profile(model, inputs=(x,))
    throughout = round(x.shape[0] / (need_time / 1), 3)
    return flops, params, throughout


def Unitconversion(flops, params, throughout):
    print("params : {} M".format(round(params / (1000**2), 2)))
    print("flop : {} G".format(round(flops / (1000**3), 2)))
    print("throughout: {} Images/Min".format(throughout * 60))


if __name__ == "__main__":
    device = "cpu"
    # load yml
    config = EasyDict(
        yaml.load(open("config.yml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
    )

    x = torch.rand(1, 4, 128, 128, 128).to(device)
    model = SlimUNETR(**config.slim_unetr).to(device)
    flops, param, throughout = weight_test(model, x)
    Unitconversion(flops, param, throughout)
