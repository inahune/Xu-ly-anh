import os
import sys
import gc

import numpy as np
from PIL import Image
import torch

# Giam so luong thread CPU de tiet kiem toi da RAM dem (scratchpad)
torch.set_num_threads(2)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GUIDED_DIFFUSION_DIR = os.path.join(BASE_DIR, "source", "DIRE", "guided-diffusion")
if GUIDED_DIFFUSION_DIR not in sys.path:
    sys.path.insert(0, GUIDED_DIFFUSION_DIR)

from guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)

DIFFUSION_BF16_PATH = os.path.join(BASE_DIR, "checkpoints", "256x256_diffusion_uncond_bf16.pt")
DIFFUSION_PATH = os.path.join(BASE_DIR, "checkpoints", "256x256_diffusion_uncond.pt")


def center_crop_arr(pil_image: Image.Image, image_size: int) -> np.ndarray:
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def main():
    if len(sys.argv) != 4:
        print("Usage: python dire_worker.py <input_img> <recons_out> <dire_out>", file=sys.stderr)
        sys.exit(2)

    input_path, recons_out, dire_out = sys.argv[1], sys.argv[2], sys.argv[3]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = 256 if device == "cuda" else 64
    steps = "ddim20" if device == "cuda" else "ddim5"

    ckpt_path = DIFFUSION_BF16_PATH if device == "cpu" else DIFFUSION_PATH
    if not os.path.exists(ckpt_path):
        print(f"Khong tim thay checkpoint: {ckpt_path}", file=sys.stderr)
        sys.exit(3)

    defaults = model_and_diffusion_defaults()
    defaults.update(
        dict(
            attention_resolutions="32,16,8",
            class_cond=False,
            diffusion_steps=1000,
            dropout=0.1,
            image_size=256,
            learn_sigma=True,
            noise_schedule="linear",
            num_channels=256,
            num_head_channels=64,
            num_res_blocks=2,
            resblock_updown=True,
            use_fp16=(device == "cuda"),
            use_scale_shift_norm=True,
            timestep_respacing=steps,
        )
    )

    with torch.device("meta"):
        model, diff = create_model_and_diffusion(**defaults)

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, assign=True)
    del sd
    gc.collect()

    if device == "cpu":
        for m in model.modules():
            if isinstance(m, torch.nn.GroupNorm):
                m.float()
    else:
        model.to("cuda")
        model.convert_to_fp16()

    model.eval()

    pil_image = Image.open(input_path).convert("RGB")
    arr = center_crop_arr(pil_image, res)
    arr_norm = arr.astype(np.float32) / 127.5 - 1.0
    imgs = torch.from_numpy(np.transpose(arr_norm, [2, 0, 1])).unsqueeze(0).to(device)

    ctx = (
        torch.autocast(device_type="cpu", dtype=torch.bfloat16)
        if device == "cpu"
        else torch.autocast(device_type="cuda", dtype=torch.float16)
    )
    with torch.no_grad(), ctx:
        latent = diff.ddim_reverse_sample_loop(
            model,
            (1, 3, res, res),
            noise=imgs,
            clip_denoised=True,
            model_kwargs={},
            real_step=0,
        )
        recons = diff.ddim_sample_loop(
            model,
            (1, 3, res, res),
            noise=latent,
            clip_denoised=True,
            model_kwargs={},
            real_step=0,
        )

    dire = torch.abs(imgs.float() - recons.float())
    recons_uint8 = ((recons.float() + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()
    dire_uint8 = (dire * 255.0 / 2.0).clamp(0, 255).to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()

    Image.fromarray(recons_uint8).resize((256, 256), Image.BICUBIC).save(recons_out, format="PNG")
    Image.fromarray(dire_uint8).resize((256, 256), Image.BICUBIC).save(dire_out, format="PNG")


if __name__ == "__main__":
    main()
