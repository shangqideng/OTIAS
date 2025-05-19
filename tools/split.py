import torch

import torch
import numpy as np
import torch.nn.functional as F

def model_fuse_patchwise(model, input_rgb, input_lr_u, ms, patch_size=512, stride=256, dim=31, rate=4):
    device = input_rgb.device
    _, c_rgb, h, w = input_rgb.shape
    _, c_ms, _, _ = ms.shape  # assume ms is 1/rate resolution

    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size

    input_rgb = F.pad(input_rgb, (0, pad_w, 0, pad_h), mode='reflect')        # [1, 3, H_pad, W_pad]
    input_lr_u = F.pad(input_lr_u, (0, pad_w, 0, pad_h), mode='reflect')      # [1, dim, H_pad, W_pad]
    ms = F.pad(ms, (0, pad_w//rate, 0, pad_h//rate), mode='reflect')                # [1, 31, H/rate_pad, W/rate_pad]

    _, _, H_pad, W_pad = input_rgb.shape
    output = torch.zeros((1, dim, H_pad, W_pad), device=device)
    count_map = torch.zeros_like(output)

    for i in range(0, H_pad - patch_size + 1, stride):
        for j in range(0, W_pad - patch_size + 1, stride):
            rgb_patch = input_rgb[:, :, i:i+patch_size, j:j+patch_size]
            lr_patch = input_lr_u[:, :, i:i+patch_size, j:j+patch_size]

            ms_patch = ms[:, :, i//rate:(i+patch_size)//rate, j//rate:(j+patch_size)//rate]

            with torch.no_grad():
                fused_patch = model(rgb_patch, lr_patch, ms_patch)

            output[:, :, i:i+patch_size, j:j+patch_size] += fused_patch
            count_map[:, :, i:i+patch_size, j:j+patch_size] += 1

    fused = output / (count_map + 1e-6)

    fused = fused[:, :, :h, :w]  # [1, 31, H, W]
    return fused

