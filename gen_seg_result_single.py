import sys

import torch
import time
import nibabel as nib

from src.SlimUNETR.SlimUNETR import SlimUNETR

import SimpleITK as sitk

def AsDiscrete(arr, threshold):
    img_t = arr >= threshold
    # img_t = img_t.astype(bool)

    return img_t

def _normalize(arr):
    img_r = torch.zeros(arr.shape).cuda()

    slices = arr != 0
    masked_img = arr[slices]
    if not slices.any():
        return arr

    _sub = torch.mean(masked_img.float()).item()
    _div = torch.std(masked_img.float(), unbiased=False).item()

    img_r[slices] = (masked_img - _sub) / _div

    return img_r

def NormalizeIntensity(arr):

    img_t = torch.zeros(arr.shape).cuda()
    for i, d in enumerate(arr):
        img_t[i] = _normalize(d)

    return img_t

start = 0

if __name__ == "__main__":

    print(torch.cuda.is_available())

    start = time.time()

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

    # load pre-train model
    model = model.cuda()
    model.load_state_dict(torch.load("/data/jionkim/Slim_UNETR/model_store/tte_noSP_128_b4_2e-3/best/pytorch_model.bin"))

    input_image = nib.load('/data/jionkim/_TTE_images_Fix_128/image/image_000.nii.gz').get_fdata()
    input_image = torch.Tensor(input_image).cuda()

    input_image = NormalizeIntensity(input_image)

    input_image = torch.unsqueeze(input_image, dim=0)
    input_image = torch.unsqueeze(input_image, dim=0)

    # start inference
    done = time.time()
    elapsed = done - start
    print(elapsed)
    print("Start Val！")
    # print(torch.cuda.memory_allocated() / 1024 / 1024,' MiB')

    # Volume Generation
    model.eval()

    logits = model(input_image)

    val_outputs = [AsDiscrete(i, threshold=0.5) for i in logits]

    input_image = input_image.detach().cpu().numpy()
    lab1_gen = val_outputs[0][0, :, :, :].cpu().detach().numpy().astype(bool)
    lab2_gen = val_outputs[0][1, :, :, :].cpu().detach().numpy().astype(bool)

    sitk.WriteImage(sitk.GetImageFromArray(input_image), '/data/jionkim/Slim_UNETR/inf_img.nrrd')
    sitk.WriteImage(sitk.GetImageFromArray(lab1_gen.astype(int)), '/data/jionkim/Slim_UNETR/inf_lab1_gen.nrrd')
    sitk.WriteImage(sitk.GetImageFromArray(lab2_gen.astype(int)), '/data/jionkim/Slim_UNETR/inf_lab2_gen.nrrd')

    done = time.time()
    elapsed = done - start
    print(elapsed)
    # print(torch.cuda.max_memory_allocated() / 1024 / 1024,' MiB')

    sys.exit(1)