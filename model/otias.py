from fvcore.nn import FlopCountAnalysis, flop_count_table
import __init__
import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from edsr import make_edsr_baseline, make_coord
from thop import profile
# from UDL.pansharpening.common.evaluate import analysis_accu
# from UDL.Basis.criterion_metrics import *


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        if hidden_list==None:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        else:
            for hidden in hidden_list:
                layers.append(nn.Linear(lastv, hidden))
                layers.append(nn.GELU())
                lastv = hidden
            layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class otias_c_x4(nn.Module):

    def __init__(self, n_select_bands, n_bands, feat_dim=128,
                 guide_dim=128, sz=64):
        super().__init__()
        self.feat_dim = feat_dim
        self.guide_dim = guide_dim
        self.n_select_bands = n_select_bands
        self.n_bands = n_bands
        self.down32 = nn.AdaptiveMaxPool2d((sz // 2, sz // 2))
        self.down16 = nn.AdaptiveMaxPool2d((sz // 4, sz // 4))
        self.spatial_encoder = make_edsr_baseline(n_resblocks=4, n_feats=self.guide_dim, n_colors=self.n_select_bands + self.n_bands)
        self.spectral_encoder = make_edsr_baseline(n_resblocks=4, n_feats=self.feat_dim, n_colors=self.n_bands)

        imnet_in_dim_l1 = self.feat_dim + self.guide_dim*2 + 2
        imnet_in_dim_l2 = self.feat_dim + self.guide_dim*2 + 2

        self.imnet_l1 = MLP(imnet_in_dim_l1, out_dim=self.feat_dim*2, hidden_list=None)
        self.wg_32 = MLP(self.feat_dim, out_dim=self.feat_dim, hidden_list=[64])
        self.ffn_32 = MLP(self.feat_dim, out_dim=self.feat_dim, hidden_list=[64])

        self.imnet_l2 = MLP(imnet_in_dim_l2, out_dim=self.feat_dim*2, hidden_list=None)
        self.wg_64 = MLP(self.feat_dim, out_dim=self.feat_dim, hidden_list=[64])
        self.ffn_64 = MLP(self.feat_dim, out_dim=n_bands, hidden_list=[64])

    def level1(self, feat, coord, hr_guide, hr_guide_lr):

        # feat: [B, C, h, w]
        # coord: [B, N, 2], N <= H * W

        b, c, h, w = feat.shape  # lr  7x128x8x8
        _, _, H, W = hr_guide.shape  # hr  7x128x64x64
        coord = coord.expand(b, H * W, 2)
        B, N, _ = coord.shape

        # LR centers' coords
        feat_coord = make_coord((h, w), flatten=False).to(feat.device).permute(2, 0, 1).unsqueeze(0).expand(b, 2, h,
                                                                                                            w).cuda()

        hr_guide_sample = F.grid_sample(hr_guide, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                     :].permute(0, 2, 1)  # [B, N, C]

        rx = 1 / h
        ry = 1 / w

        preds = []

        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()

                coord_[:, :, 0] += (vx) * rx
                coord_[:, :, 1] += (vy) * ry

                # feat: [B, c, h, w], coord_: [B, N, 2] --> [B, 1, N, 2], out: [B, c, 1, N] --> [B, c, N] --> [B, N, c]
                feat_sample = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                         :].permute(0, 2, 1)  # [B, N, c]
                hr_guide_lr_sample = F.grid_sample(hr_guide_lr, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                         :].permute(0, 2, 1)  # [B, N, c]
                coord_sample = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[
                          :, :, 0, :].permute(0, 2, 1)  # [B, N, 2]

                rel_coord = coord - coord_sample
                rel_coord[:, :, 0] *= h
                rel_coord[:, :, 1] *= w

                inp = torch.cat([feat_sample, hr_guide_sample, hr_guide_lr_sample, rel_coord], dim=-1)

                pred = self.imnet_l1(inp.view(B * N, -1)).view(B, N, -1)  # [B, N, 2]
                preds.append(pred[:, :, :self.feat_dim])
                preds.append(pred[:, :, self.feat_dim:])

        preds = torch.stack(preds, dim=-2)  # B x n x 8 X c

        weight = F.softmax(self.wg_32(preds), dim=-2)
        preds = preds.view(b, H*W, 8, self.feat_dim)
        recon = (preds * weight).sum(-2).view(b, H*W, -1)
        recon = self.ffn_32(recon) + recon

        ret = recon.permute(0, 2, 1).view(b, -1, H, W)

        return ret

    def level2(self, feat, coord, hr_guide, hr_guide_lr):

        # feat: [B, C, h, w]
        # coord: [B, N, 2], N <= H * W

        b, c, h, w = feat.shape  # lr  7x128x8x8
        _, _, H, W = hr_guide.shape  # hr  7x128x64x64
        coord = coord.expand(b, H * W, 2)
        B, N, _ = coord.shape

        # LR centers' coords
        feat_coord = make_coord((h, w), flatten=False).to(feat.device).permute(2, 0, 1).unsqueeze(0).expand(b, 2, h,
                                                                                                            w).cuda()

        hr_guide_sample = F.grid_sample(hr_guide, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                     :].permute(0, 2, 1)  # [B, N, C]

        rx = 1 / h
        ry = 1 / w

        preds = []

        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()

                coord_[:, :, 0] += (vx) * rx
                coord_[:, :, 1] += (vy) * ry

                # feat: [B, c, h, w], coord_: [B, N, 2] --> [B, 1, N, 2], out: [B, c, 1, N] --> [B, c, N] --> [B, N, c]
                feat_sample = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                         :].permute(0, 2, 1)  # [B, N, c]
                hr_guide_lr_sample = F.grid_sample(hr_guide_lr, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:,
                              :, 0,
                              :].permute(0, 2, 1)  # [B, N, c]
                coord_sample = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[
                          :, :, 0, :].permute(0, 2, 1)  # [B, N, 2]

                rel_coord = coord - coord_sample
                rel_coord[:, :, 0] *= h
                rel_coord[:, :, 1] *= w

                inp = torch.cat([feat_sample, hr_guide_sample, hr_guide_lr_sample, rel_coord], dim=-1)
                pred = self.imnet_l2(inp.view(B * N, -1)).view(B, N, -1)  # [B, N, 2]
                preds.append(pred[:, :, :self.feat_dim])
                preds.append(pred[:, :, self.feat_dim:])

        preds = torch.stack(preds, dim=-2)  # B x n x 8 X c

        weight = F.softmax(self.wg_64(preds), dim=-2)
        preds = preds.view(b, H*W, 8, self.feat_dim)
        recon = (preds * weight).sum(-2).view(b, H*W, -1)
        recon = self.ffn_64(recon)

        ret = recon.permute(0, 2, 1).view(b, -1, H, W)

        return ret

    def forward(self, HR_MSI, lms, LR_HSI):

        _, _, H, W = HR_MSI.shape
        coord_32 = make_coord([H // 2, W // 2]).cuda()
        coord_64 = make_coord([H, W]).cuda()
        # coord_128 = make_coord([H, W]).cuda()

        feat = torch.cat([HR_MSI, lms], dim=1)

        hr_spa = self.spatial_encoder(feat)
        guide32 = self.down32(hr_spa)
        guide16 = self.down16(hr_spa)

        lr_spe = self.spectral_encoder(LR_HSI)

        INR_feature_l1 = self.level1(lr_spe, coord_32, guide32, guide16)  # BxCxHxW
        INR_feature_l2 = self.level2(INR_feature_l1, coord_64, hr_spa, guide32)  # BxCxHxW

        output = lms + INR_feature_l2

        return output

    def train_step(self, batch, *args, **kwargs):
        gt, up, hsi, msi = batch['gt'].cuda(), \
                           batch['up'].cuda(), \
                           batch['lrhsi'].cuda(), \
                           batch['rgb'].cuda()
        sr = self(msi, up, hsi)
        loss = self.criterion(sr, gt, *args, **kwargs)
        log_vars = {}
        with torch.no_grad():
            metrics = analysis_accu(gt, sr, 4, choices=4)
            log_vars.update(metrics)

        return {'loss': loss, 'log_vars': log_vars}

    def eval_step(self, batch, *args, **kwargs):
        gt, up, hsi, msi = batch['gt'].cuda(), \
                           batch['up'].cuda(), \
                           batch['lrhsi'].cuda(), \
                           batch['rgb'].cuda()

        sr1 = self.forward(msi, up, hsi)

        with torch.no_grad():
            metrics = analysis_accu(gt[0].permute(1, 2, 0), sr1[0].permute(1, 2, 0), 4)
            metrics.update(metrics)

        return sr1, metrics

    def set_metrics(self, criterion, rgb_range=1.0):
        self.rgb_range = rgb_range
        self.criterion = criterion

class otias_c_x8(nn.Module):

    def __init__(self, n_select_bands, n_bands, feat_dim=128,
                 guide_dim=128, sz=64):
        super().__init__()
        self.feat_dim = feat_dim
        self.guide_dim = guide_dim
        self.n_select_bands = n_select_bands
        self.n_bands = n_bands
        self.down32 = nn.AdaptiveMaxPool2d((sz // 2, sz // 2))
        self.down16 = nn.AdaptiveMaxPool2d((sz // 4, sz // 4))
        self.spatial_encoder = make_edsr_baseline(n_resblocks=4, n_feats=self.guide_dim, n_colors=self.n_select_bands + self.n_bands)
        self.spectral_encoder = make_edsr_baseline(n_resblocks=4, n_feats=self.feat_dim, n_colors=self.n_bands)

        imnet_in_dim_l1 = self.feat_dim + self.guide_dim*2 + 2
        imnet_in_dim_l2 = self.feat_dim + self.guide_dim*2 + 2

        self.imnet_l1 = MLP(imnet_in_dim_l1, out_dim=self.feat_dim*2, hidden_list=None)
        self.wg_32 = MLP(self.feat_dim, out_dim=self.feat_dim, hidden_list=[64])
        self.ffn_32 = MLP(self.feat_dim, out_dim=self.feat_dim, hidden_list=[64])

        self.imnet_l2 = MLP(imnet_in_dim_l2, out_dim=self.feat_dim*2, hidden_list=None)
        self.wg_64 = MLP(self.feat_dim, out_dim=self.feat_dim, hidden_list=[64])
        self.ffn_64 = MLP(self.feat_dim, out_dim=n_bands, hidden_list=[64])

    def level1(self, feat, coord, hr_guide, hr_guide_lr):

        # feat: [B, C, h, w]
        # coord: [B, N, 2], N <= H * W

        b, c, h, w = feat.shape  # lr  7x128x8x8
        _, _, H, W = hr_guide.shape  # hr  7x128x64x64
        coord = coord.expand(b, H * W, 2)
        B, N, _ = coord.shape

        # LR centers' coords
        feat_coord = make_coord((h, w), flatten=False).to(feat.device).permute(2, 0, 1).unsqueeze(0).expand(b, 2, h,
                                                                                                            w).cuda()

        hr_guide_sample = F.grid_sample(hr_guide, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                     :].permute(0, 2, 1)  # [B, N, C]

        rx = 1 / h
        ry = 1 / w

        preds = []

        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()

                coord_[:, :, 0] += (vx) * rx
                coord_[:, :, 1] += (vy) * ry

                # feat: [B, c, h, w], coord_: [B, N, 2] --> [B, 1, N, 2], out: [B, c, 1, N] --> [B, c, N] --> [B, N, c]
                feat_sample = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                         :].permute(0, 2, 1)  # [B, N, c]
                hr_guide_lr_sample = F.grid_sample(hr_guide_lr, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                         :].permute(0, 2, 1)  # [B, N, c]
                coord_sample = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[
                          :, :, 0, :].permute(0, 2, 1)  # [B, N, 2]

                rel_coord = coord - coord_sample
                rel_coord[:, :, 0] *= h
                rel_coord[:, :, 1] *= w

                inp = torch.cat([feat_sample, hr_guide_sample, hr_guide_lr_sample, rel_coord], dim=-1)

                pred = self.imnet_l1(inp.view(B * N, -1)).view(B, N, -1)  # [B, N, 2]
                preds.append(pred[:, :, :self.feat_dim])
                preds.append(pred[:, :, self.feat_dim:])

        preds = torch.stack(preds, dim=-2)  # B x n x 8 X c

        weight = F.softmax(self.wg_32(preds), dim=-2)
        preds = preds.view(b, H*W, 8, self.feat_dim)
        recon = (preds * weight).sum(-2).view(b, H*W, -1)
        recon = self.ffn_32(recon) + recon

        ret = recon.permute(0, 2, 1).view(b, -1, H, W)

        return ret

    def level2(self, feat, coord, hr_guide, hr_guide_lr):

        # feat: [B, C, h, w]
        # coord: [B, N, 2], N <= H * W

        b, c, h, w = feat.shape  # lr  7x128x8x8
        _, _, H, W = hr_guide.shape  # hr  7x128x64x64
        coord = coord.expand(b, H * W, 2)
        B, N, _ = coord.shape

        # LR centers' coords
        feat_coord = make_coord((h, w), flatten=False).to(feat.device).permute(2, 0, 1).unsqueeze(0).expand(b, 2, h,
                                                                                                            w).cuda()

        hr_guide_sample = F.grid_sample(hr_guide, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                     :].permute(0, 2, 1)  # [B, N, C]

        rx = 1 / h
        ry = 1 / w

        preds = []

        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()

                coord_[:, :, 0] += (vx) * rx
                coord_[:, :, 1] += (vy) * ry

                # feat: [B, c, h, w], coord_: [B, N, 2] --> [B, 1, N, 2], out: [B, c, 1, N] --> [B, c, N] --> [B, N, c]
                feat_sample = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                         :].permute(0, 2, 1)  # [B, N, c]
                hr_guide_lr_sample = F.grid_sample(hr_guide_lr, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:,
                              :, 0,
                              :].permute(0, 2, 1)  # [B, N, c]
                coord_sample = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[
                          :, :, 0, :].permute(0, 2, 1)  # [B, N, 2]

                rel_coord = coord - coord_sample
                rel_coord[:, :, 0] *= h
                rel_coord[:, :, 1] *= w

                inp = torch.cat([feat_sample, hr_guide_sample, hr_guide_lr_sample, rel_coord], dim=-1)
                pred = self.imnet_l2(inp.view(B * N, -1)).view(B, N, -1)  # [B, N, 2]
                preds.append(pred[:, :, :self.feat_dim])
                preds.append(pred[:, :, self.feat_dim:])

        preds = torch.stack(preds, dim=-2)  # B x n x 8 X c

        weight = F.softmax(self.wg_64(preds), dim=-2)
        preds = preds.view(b, H*W, 8, self.feat_dim)
        recon = (preds * weight).sum(-2).view(b, H*W, -1)
        recon = self.ffn_64(recon)

        ret = recon.permute(0, 2, 1).view(b, -1, H, W)

        return ret

    def forward(self, HR_MSI, lms, LR_HSI):

        _, _, H, W = HR_MSI.shape
        coord_32 = make_coord([H // 2, W // 2]).cuda()
        coord_64 = make_coord([H, W]).cuda()
        # coord_128 = make_coord([H, W]).cuda()

        feat = torch.cat([HR_MSI, lms], dim=1)

        hr_spa = self.spatial_encoder(feat)
        guide32 = self.down32(hr_spa)
        guide16 = self.down16(hr_spa)

        lr_spe = self.spectral_encoder(LR_HSI)

        INR_feature_l1 = self.level1(lr_spe, coord_32, guide32, guide16)  # BxCxHxW
        INR_feature_l2 = self.level2(INR_feature_l1, coord_64, hr_spa, guide32)  # BxCxHxW

        output = lms + INR_feature_l2

        return output

    def train_step(self, batch, *args, **kwargs):
        gt, up, hsi, msi = batch['gt'].cuda(), \
                           batch['up'].cuda(), \
                           batch['lrhsi'].cuda(), \
                           batch['rgb'].cuda()
        sr = self(msi, up, hsi)
        loss = self.criterion(sr, gt, *args, **kwargs)
        log_vars = {}
        with torch.no_grad():
            metrics = analysis_accu(gt, sr, 4, choices=4)
            log_vars.update(metrics)

        return {'loss': loss, 'log_vars': log_vars}

    def eval_step(self, batch, *args, **kwargs):
        gt, up, hsi, msi = batch['gt'].cuda(), \
                           batch['up'].cuda(), \
                           batch['lrhsi'].cuda(), \
                           batch['rgb'].cuda()

        sr1 = self.forward(msi, up, hsi)

        with torch.no_grad():
            metrics = analysis_accu(gt[0].permute(1, 2, 0), sr1[0].permute(1, 2, 0), 4)
            metrics.update(metrics)

        return sr1, metrics

    def set_metrics(self, criterion, rgb_range=1.0):
        self.rgb_range = rgb_range
        self.criterion = criterion


class otias_h_x4(nn.Module):

    def __init__(self, n_select_bands, n_bands, feat_dim=128,
                 guide_dim=128, sz=64):
        super().__init__()
        self.feat_dim = feat_dim
        self.guide_dim = guide_dim
        self.n_select_bands = n_select_bands
        self.n_bands = n_bands
        self.down32 = nn.AdaptiveMaxPool2d((sz // 2, sz // 2))
        self.down16 = nn.AdaptiveMaxPool2d((sz // 4, sz // 4))
        self.spatial_encoder = make_edsr_baseline(n_resblocks=4, n_feats=self.guide_dim, n_colors=self.n_select_bands + self.n_bands)
        self.spectral_encoder = make_edsr_baseline(n_resblocks=4, n_feats=self.feat_dim, n_colors=self.n_bands)

        imnet_in_dim_l1 = self.feat_dim + self.guide_dim + 2
        imnet_in_dim_l2 = self.feat_dim + self.guide_dim + 2

        self.imnet_l1 = MLP(imnet_in_dim_l1, out_dim=self.feat_dim*2, hidden_list=None)
        self.wg_32 = MLP(self.feat_dim, out_dim=self.feat_dim, hidden_list=[64])
        self.ffn_32 = MLP(self.feat_dim, out_dim=self.feat_dim, hidden_list=[64])

        self.imnet_l2 = MLP(imnet_in_dim_l2, out_dim=self.feat_dim*2, hidden_list=None)
        self.wg_64 = MLP(self.feat_dim, out_dim=self.feat_dim, hidden_list=[64])
        self.ffn_64 = MLP(self.feat_dim, out_dim=n_bands, hidden_list=[64])

    def level1(self, feat, coord, hr_guide, hr_guide_lr):

        # feat: [B, C, h, w]
        # coord: [B, N, 2], N <= H * W

        b, c, h, w = feat.shape  # lr  7x128x8x8
        _, _, H, W = hr_guide.shape  # hr  7x128x64x64
        coord = coord.expand(b, H * W, 2)
        B, N, _ = coord.shape

        # LR centers' coords
        feat_coord = make_coord((h, w), flatten=False).to(feat.device).permute(2, 0, 1).unsqueeze(0).expand(b, 2, h,
                                                                                                            w).cuda()

        hr_guide_sample = F.grid_sample(hr_guide, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                     :].permute(0, 2, 1)  # [B, N, C]

        rx = 1 / h
        ry = 1 / w

        preds = []

        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()

                coord_[:, :, 0] += (vx) * rx
                coord_[:, :, 1] += (vy) * ry

                # feat: [B, c, h, w], coord_: [B, N, 2] --> [B, 1, N, 2], out: [B, c, 1, N] --> [B, c, N] --> [B, N, c]
                feat_sample = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                         :].permute(0, 2, 1)  # [B, N, c]
                hr_guide_lr_sample = F.grid_sample(hr_guide_lr, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                         :].permute(0, 2, 1)  # [B, N, c]
                coord_sample = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[
                          :, :, 0, :].permute(0, 2, 1)  # [B, N, 2]

                rel_coord = coord - coord_sample
                rel_coord[:, :, 0] *= h
                rel_coord[:, :, 1] *= w

                # inp = torch.cat([feat_sample, hr_guide_sample, hr_guide_lr_sample, rel_coord], dim=-1)
                inp = torch.cat([feat_sample, hr_guide_sample, rel_coord], dim=-1)

                pred = self.imnet_l1(inp.view(B * N, -1)).view(B, N, -1)  # [B, N, 2]
                preds.append(pred[:, :, :self.feat_dim])
                preds.append(pred[:, :, self.feat_dim:])

        preds = torch.stack(preds, dim=-2)  # B x n x 8 X c

        weight = F.softmax(self.wg_32(preds), dim=-2)
        preds = preds.view(b, H*W, 8, self.feat_dim)
        recon = (preds * weight).sum(-2).view(b, H*W, -1)
        recon = self.ffn_32(recon) + recon

        ret = recon.permute(0, 2, 1).view(b, -1, H, W)

        return ret

    def level2(self, feat, coord, hr_guide, hr_guide_lr):

        # feat: [B, C, h, w]
        # coord: [B, N, 2], N <= H * W

        b, c, h, w = feat.shape  # lr  7x128x8x8
        _, _, H, W = hr_guide.shape  # hr  7x128x64x64
        coord = coord.expand(b, H * W, 2)
        B, N, _ = coord.shape

        # LR centers' coords
        feat_coord = make_coord((h, w), flatten=False).to(feat.device).permute(2, 0, 1).unsqueeze(0).expand(b, 2, h,
                                                                                                            w).cuda()

        hr_guide_sample = F.grid_sample(hr_guide, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                     :].permute(0, 2, 1)  # [B, N, C]

        rx = 1 / h
        ry = 1 / w

        preds = []

        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()

                coord_[:, :, 0] += (vx) * rx
                coord_[:, :, 1] += (vy) * ry

                # feat: [B, c, h, w], coord_: [B, N, 2] --> [B, 1, N, 2], out: [B, c, 1, N] --> [B, c, N] --> [B, N, c]
                feat_sample = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                         :].permute(0, 2, 1)  # [B, N, c]
                hr_guide_lr_sample = F.grid_sample(hr_guide_lr, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:,
                              :, 0,
                              :].permute(0, 2, 1)  # [B, N, c]
                coord_sample = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[
                          :, :, 0, :].permute(0, 2, 1)  # [B, N, 2]

                rel_coord = coord - coord_sample
                rel_coord[:, :, 0] *= h
                rel_coord[:, :, 1] *= w

                # inp = torch.cat([feat_sample, hr_guide_sample, hr_guide_lr_sample, rel_coord], dim=-1)
                inp = torch.cat([feat_sample, hr_guide_sample, rel_coord], dim=-1)
                pred = self.imnet_l2(inp.view(B * N, -1)).view(B, N, -1)  # [B, N, 2]
                preds.append(pred[:, :, :self.feat_dim])
                preds.append(pred[:, :, self.feat_dim:])

        preds = torch.stack(preds, dim=-2)  # B x n x 8 X c

        weight = F.softmax(self.wg_64(preds), dim=-2)
        preds = preds.view(b, H*W, 8, self.feat_dim)
        recon = (preds * weight).sum(-2).view(b, H*W, -1)
        recon = self.ffn_64(recon)

        ret = recon.permute(0, 2, 1).view(b, -1, H, W)

        return ret

    def forward(self, HR_MSI, lms, LR_HSI):

        _, _, H, W = HR_MSI.shape
        coord_32 = make_coord([H // 2, W // 2]).cuda()
        coord_64 = make_coord([H, W]).cuda()
        # coord_128 = make_coord([H, W]).cuda()

        feat = torch.cat([HR_MSI, lms], dim=1)

        hr_spa = self.spatial_encoder(feat)
        guide32 = self.down32(hr_spa)
        guide16 = self.down16(hr_spa)

        lr_spe = self.spectral_encoder(LR_HSI)

        INR_feature_l1 = self.level1(lr_spe, coord_32, guide32, guide16)  # BxCxHxW
        INR_feature_l2 = self.level2(INR_feature_l1, coord_64, hr_spa, guide32)  # BxCxHxW

        output = lms + INR_feature_l2

        return output

    def train_step(self, batch, *args, **kwargs):
        gt, up, hsi, msi = batch['gt'].cuda(), \
                           batch['up'].cuda(), \
                           batch['lrhsi'].cuda(), \
                           batch['rgb'].cuda()
        sr = self(msi, up, hsi)
        loss = self.criterion(sr, gt, *args, **kwargs)
        log_vars = {}
        with torch.no_grad():
            metrics = analysis_accu(gt, sr, 4, choices=4)
            log_vars.update(metrics)

        return {'loss': loss, 'log_vars': log_vars}

    def eval_step(self, batch, *args, **kwargs):
        gt, up, hsi, msi = batch['gt'].cuda(), \
                           batch['up'].cuda(), \
                           batch['lrhsi'].cuda(), \
                           batch['rgb'].cuda()

        sr1 = self.forward(msi, up, hsi)

        with torch.no_grad():
            metrics = analysis_accu(gt[0].permute(1, 2, 0), sr1[0].permute(1, 2, 0), 4)
            metrics.update(metrics)

        return sr1, metrics

    def set_metrics(self, criterion, rgb_range=1.0):
        self.rgb_range = rgb_range
        self.criterion = criterion

class otias_h_x8(nn.Module):

    def __init__(self, n_select_bands, n_bands, feat_dim=128,
                 guide_dim=128, sz=64):
        super().__init__()
        self.feat_dim = feat_dim
        self.guide_dim = guide_dim
        self.n_select_bands = n_select_bands
        self.n_bands = n_bands
        self.down32 = nn.AdaptiveMaxPool2d((sz // 2, sz // 2))
        self.down16 = nn.AdaptiveMaxPool2d((sz // 4, sz // 4))
        self.spatial_encoder = make_edsr_baseline(n_resblocks=4, n_feats=self.guide_dim, n_colors=self.n_select_bands + self.n_bands)
        self.spectral_encoder = make_edsr_baseline(n_resblocks=4, n_feats=self.feat_dim, n_colors=self.n_bands)

        imnet_in_dim_l1 = self.feat_dim + self.guide_dim + 2
        imnet_in_dim_l2 = self.feat_dim + self.guide_dim + 2

        self.imnet_l1 = MLP(imnet_in_dim_l1, out_dim=self.feat_dim*2, hidden_list=None)
        self.wg_32 = MLP(self.feat_dim, out_dim=self.feat_dim, hidden_list=[64])
        self.ffn_32 = MLP(self.feat_dim, out_dim=self.feat_dim, hidden_list=[64])

        self.imnet_l2 = MLP(imnet_in_dim_l2, out_dim=self.feat_dim*2, hidden_list=None)
        self.wg_64 = MLP(self.feat_dim, out_dim=self.feat_dim, hidden_list=[64])
        self.ffn_64 = MLP(self.feat_dim, out_dim=n_bands, hidden_list=[64])

    def level1(self, feat, coord, hr_guide, hr_guide_lr):

        # feat: [B, C, h, w]
        # coord: [B, N, 2], N <= H * W

        b, c, h, w = feat.shape  # lr  7x128x8x8
        _, _, H, W = hr_guide.shape  # hr  7x128x64x64
        coord = coord.expand(b, H * W, 2)
        B, N, _ = coord.shape

        # LR centers' coords
        feat_coord = make_coord((h, w), flatten=False).to(feat.device).permute(2, 0, 1).unsqueeze(0).expand(b, 2, h,
                                                                                                            w).cuda()

        hr_guide_sample = F.grid_sample(hr_guide, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                     :].permute(0, 2, 1)  # [B, N, C]

        rx = 1 / h
        ry = 1 / w

        preds = []

        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()

                coord_[:, :, 0] += (vx) * rx
                coord_[:, :, 1] += (vy) * ry

                # feat: [B, c, h, w], coord_: [B, N, 2] --> [B, 1, N, 2], out: [B, c, 1, N] --> [B, c, N] --> [B, N, c]
                feat_sample = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                         :].permute(0, 2, 1)  # [B, N, c]
                hr_guide_lr_sample = F.grid_sample(hr_guide_lr, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                         :].permute(0, 2, 1)  # [B, N, c]
                coord_sample = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[
                          :, :, 0, :].permute(0, 2, 1)  # [B, N, 2]

                rel_coord = coord - coord_sample
                rel_coord[:, :, 0] *= h
                rel_coord[:, :, 1] *= w

                # inp = torch.cat([feat_sample, hr_guide_sample, hr_guide_lr_sample, rel_coord], dim=-1)
                inp = torch.cat([feat_sample, hr_guide_sample, rel_coord], dim=-1)

                pred = self.imnet_l1(inp.view(B * N, -1)).view(B, N, -1)  # [B, N, 2]
                preds.append(pred[:, :, :self.feat_dim])
                preds.append(pred[:, :, self.feat_dim:])

        preds = torch.stack(preds, dim=-2)  # B x n x 8 X c

        weight = F.softmax(self.wg_32(preds), dim=-2)
        preds = preds.view(b, H*W, 8, self.feat_dim)
        recon = (preds * weight).sum(-2).view(b, H*W, -1)
        recon = self.ffn_32(recon) + recon

        ret = recon.permute(0, 2, 1).view(b, -1, H, W)

        return ret

    def level2(self, feat, coord, hr_guide, hr_guide_lr):

        # feat: [B, C, h, w]
        # coord: [B, N, 2], N <= H * W

        b, c, h, w = feat.shape  # lr  7x128x8x8
        _, _, H, W = hr_guide.shape  # hr  7x128x64x64
        coord = coord.expand(b, H * W, 2)
        B, N, _ = coord.shape

        # LR centers' coords
        feat_coord = make_coord((h, w), flatten=False).to(feat.device).permute(2, 0, 1).unsqueeze(0).expand(b, 2, h,
                                                                                                            w).cuda()

        hr_guide_sample = F.grid_sample(hr_guide, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                     :].permute(0, 2, 1)  # [B, N, C]

        rx = 1 / h
        ry = 1 / w

        preds = []

        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()

                coord_[:, :, 0] += (vx) * rx
                coord_[:, :, 1] += (vy) * ry

                # feat: [B, c, h, w], coord_: [B, N, 2] --> [B, 1, N, 2], out: [B, c, 1, N] --> [B, c, N] --> [B, N, c]
                feat_sample = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                         :].permute(0, 2, 1)  # [B, N, c]
                hr_guide_lr_sample = F.grid_sample(hr_guide_lr, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:,
                              :, 0,
                              :].permute(0, 2, 1)  # [B, N, c]
                coord_sample = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[
                          :, :, 0, :].permute(0, 2, 1)  # [B, N, 2]

                rel_coord = coord - coord_sample
                rel_coord[:, :, 0] *= h
                rel_coord[:, :, 1] *= w

                # inp = torch.cat([feat_sample, hr_guide_sample, hr_guide_lr_sample, rel_coord], dim=-1)
                inp = torch.cat([feat_sample, hr_guide_sample, rel_coord], dim=-1)
                pred = self.imnet_l2(inp.view(B * N, -1)).view(B, N, -1)  # [B, N, 2]
                preds.append(pred[:, :, :self.feat_dim])
                preds.append(pred[:, :, self.feat_dim:])

        preds = torch.stack(preds, dim=-2)  # B x n x 8 X c

        weight = F.softmax(self.wg_64(preds), dim=-2)
        preds = preds.view(b, H*W, 8, self.feat_dim)
        recon = (preds * weight).sum(-2).view(b, H*W, -1)
        recon = self.ffn_64(recon)

        ret = recon.permute(0, 2, 1).view(b, -1, H, W)

        return ret

    def forward(self, HR_MSI, lms, LR_HSI):

        _, _, H, W = HR_MSI.shape
        coord_32 = make_coord([H // 2, W // 2]).cuda()
        coord_64 = make_coord([H, W]).cuda()
        # coord_128 = make_coord([H, W]).cuda()

        feat = torch.cat([HR_MSI, lms], dim=1)

        hr_spa = self.spatial_encoder(feat)
        guide32 = self.down32(hr_spa)
        guide16 = self.down16(hr_spa)

        lr_spe = self.spectral_encoder(LR_HSI)

        INR_feature_l1 = self.level1(lr_spe, coord_32, guide32, guide16)  # BxCxHxW
        INR_feature_l2 = self.level2(INR_feature_l1, coord_64, hr_spa, guide32)  # BxCxHxW

        output = lms + INR_feature_l2

        return output

    def train_step(self, batch, *args, **kwargs):
        gt, up, hsi, msi = batch['gt'].cuda(), \
                           batch['up'].cuda(), \
                           batch['lrhsi'].cuda(), \
                           batch['rgb'].cuda()
        sr = self(msi, up, hsi)
        loss = self.criterion(sr, gt, *args, **kwargs)
        log_vars = {}
        with torch.no_grad():
            metrics = analysis_accu(gt, sr, 4, choices=4)
            log_vars.update(metrics)

        return {'loss': loss, 'log_vars': log_vars}

    def eval_step(self, batch, *args, **kwargs):
        gt, up, hsi, msi = batch['gt'].cuda(), \
                           batch['up'].cuda(), \
                           batch['lrhsi'].cuda(), \
                           batch['rgb'].cuda()

        sr1 = self.forward(msi, up, hsi)

        with torch.no_grad():
            metrics = analysis_accu(gt[0].permute(1, 2, 0), sr1[0].permute(1, 2, 0), 4)
            metrics.update(metrics)

        return sr1, metrics

    def set_metrics(self, criterion, rgb_range=1.0):
        self.rgb_range = rgb_range
        self.criterion = criterion

class otias_ch_x4(nn.Module):

    def __init__(self, n_select_bands, n_bands, feat_dim=128,
                 guide_dim=128, sz=64):
        super().__init__()
        self.feat_dim = feat_dim
        self.guide_dim = guide_dim
        self.n_select_bands = n_select_bands
        self.n_bands = n_bands
        self.down32 = nn.AdaptiveMaxPool2d((sz // 2, sz // 2))
        self.down16 = nn.AdaptiveMaxPool2d((sz // 4, sz // 4))
        self.spatial_encoder = make_edsr_baseline(n_resblocks=6, n_feats=self.guide_dim, n_colors=self.n_select_bands + self.n_bands)
        self.spectral_encoder = make_edsr_baseline(n_resblocks=6, n_feats=self.feat_dim, n_colors=self.n_bands)

        imnet_in_dim_l1 = self.feat_dim + self.guide_dim*2 + 2
        imnet_in_dim_l2 = 192 + self.guide_dim*2 + 2

        self.imnet_l1 = MLP(imnet_in_dim_l1, out_dim=384, hidden_list=None)
        self.wg_32 = MLP(192, out_dim=192, hidden_list=[96])
        self.ffn_32 = MLP(192, out_dim=192, hidden_list=[96])

        self.imnet_l2 = MLP(imnet_in_dim_l2, out_dim=384, hidden_list=None)
        self.wg_64 = MLP(192, out_dim=192, hidden_list=[96])
        self.ffn_64 = MLP(192, out_dim=n_bands, hidden_list=[96])

    def level1(self, feat, coord, hr_guide, hr_guide_lr):

        # feat: [B, C, h, w]
        # coord: [B, N, 2], N <= H * W

        b, c, h, w = feat.shape  # lr  7x128x8x8
        _, _, H, W = hr_guide.shape  # hr  7x128x64x64
        coord = coord.expand(b, H * W, 2)
        B, N, _ = coord.shape

        # LR centers' coords
        feat_coord = make_coord((h, w), flatten=False).to(feat.device).permute(2, 0, 1).unsqueeze(0).expand(b, 2, h,
                                                                                                            w).cuda()

        hr_guide_sample = F.grid_sample(hr_guide, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                     :].permute(0, 2, 1)  # [B, N, C]

        rx = 1 / h
        ry = 1 / w

        preds = []

        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()

                coord_[:, :, 0] += (vx) * rx
                coord_[:, :, 1] += (vy) * ry

                # feat: [B, c, h, w], coord_: [B, N, 2] --> [B, 1, N, 2], out: [B, c, 1, N] --> [B, c, N] --> [B, N, c]
                feat_sample = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                         :].permute(0, 2, 1)  # [B, N, c]
                hr_guide_lr_sample = F.grid_sample(hr_guide_lr, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                         :].permute(0, 2, 1)  # [B, N, c]
                coord_sample = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[
                          :, :, 0, :].permute(0, 2, 1)  # [B, N, 2]

                rel_coord = coord - coord_sample
                rel_coord[:, :, 0] *= h
                rel_coord[:, :, 1] *= w

                inp = torch.cat([feat_sample, hr_guide_sample, hr_guide_lr_sample, rel_coord], dim=-1)
                pred = self.imnet_l1(inp.view(B * N, -1)).view(B, N, -1)  # [B, N, 2]
                preds.append(pred[:, :, :192])
                preds.append(pred[:, :, 192:])

        preds = torch.stack(preds, dim=-2)  # B x n x 8 X c

        weight = F.softmax(self.wg_32(preds), dim=-2)  # BxNx8x128
        preds = preds.view(b, H*W, 8, 192)  # BxNx8x128
        recon = (preds * weight).sum(-2).view(b, H*W, -1)
        recon = self.ffn_32(recon) + recon

        ret = recon.permute(0, 2, 1).view(b, -1, H, W)

        return ret

    def level2(self, feat, coord, hr_guide, hr_guide_lr):

        # feat: [B, C, h, w]
        # coord: [B, N, 2], N <= H * W

        b, c, h, w = feat.shape  # lr  7x128x8x8
        _, _, H, W = hr_guide.shape  # hr  7x128x64x64
        coord = coord.expand(b, H * W, 2)
        B, N, _ = coord.shape

        # LR centers' coords
        feat_coord = make_coord((h, w), flatten=False).to(feat.device).permute(2, 0, 1).unsqueeze(0).expand(b, 2, h,
                                                                                                            w).cuda()

        hr_guide_sample = F.grid_sample(hr_guide, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                     :].permute(0, 2, 1)  # [B, N, C]

        rx = 1 / h
        ry = 1 / w

        preds = []

        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()

                coord_[:, :, 0] += (vx) * rx
                coord_[:, :, 1] += (vy) * ry

                # feat: [B, c, h, w], coord_: [B, N, 2] --> [B, 1, N, 2], out: [B, c, 1, N] --> [B, c, N] --> [B, N, c]
                feat_sample = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                         :].permute(0, 2, 1)  # [B, N, c]
                hr_guide_lr_sample = F.grid_sample(hr_guide_lr, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:,
                              :, 0,
                              :].permute(0, 2, 1)  # [B, N, c]
                coord_sample = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[
                          :, :, 0, :].permute(0, 2, 1)  # [B, N, 2]

                rel_coord = coord - coord_sample
                rel_coord[:, :, 0] *= h
                rel_coord[:, :, 1] *= w

                inp = torch.cat([feat_sample, hr_guide_sample, hr_guide_lr_sample, rel_coord], dim=-1)
                pred = self.imnet_l2(inp.view(B * N, -1)).view(B, N, -1)  # [B, N, 2]
                preds.append(pred[:, :, :192])
                preds.append(pred[:, :, 192:])

        preds = torch.stack(preds, dim=-2)  # B x n x 8 X c

        weight = F.softmax(self.wg_64(preds), dim=-2)
        preds = preds.view(b, H*W, 8, 192)
        recon = (preds * weight).sum(-2).view(b, H*W, -1)
        recon = self.ffn_64(recon)

        ret = recon.permute(0, 2, 1).view(b, -1, H, W)

        return ret

    def forward(self, HR_MSI, lms, LR_HSI):

        _, _, H, W = HR_MSI.shape
        coord_32 = make_coord([H // 2, W // 2]).cuda()
        coord_64 = make_coord([H, W]).cuda()
        # coord_128 = make_coord([H, W]).cuda()

        feat = torch.cat([HR_MSI, lms], dim=1)

        hr_spa = self.spatial_encoder(feat)
        guide32 = self.down32(hr_spa)
        guide16 = self.down16(hr_spa)

        lr_spe = self.spectral_encoder(LR_HSI)

        INR_feature_l1 = self.level1(lr_spe, coord_32, guide32, guide16)  # BxCxHxW
        INR_feature_l2 = self.level2(INR_feature_l1, coord_64, hr_spa, guide32)  # BxCxHxW

        output = lms + INR_feature_l2

        return output

    def train_step(self, batch, *args, **kwargs):
        gt, up, hsi, msi = batch['gt'].cuda(), \
                           batch['up'].cuda(), \
                           batch['lrhsi'].cuda(), \
                           batch['rgb'].cuda()
        sr = self(msi, up, hsi)
        loss = self.criterion(sr, gt, *args, **kwargs)
        log_vars = {}
        with torch.no_grad():
            metrics = analysis_accu(gt, sr, 4, choices=4)
            log_vars.update(metrics)

        return {'loss': loss, 'log_vars': log_vars}

    def eval_step(self, batch, *args, **kwargs):
        gt, up, hsi, msi = batch['gt'].cuda(), \
                           batch['up'].cuda(), \
                           batch['lrhsi'].cuda(), \
                           batch['rgb'].cuda()

        sr1 = self.forward(msi, up, hsi)

        with torch.no_grad():
            metrics = analysis_accu(gt[0].permute(1, 2, 0), sr1[0].permute(1, 2, 0), 4)
            metrics.update(metrics)

        return sr1, metrics

    def set_metrics(self, criterion, rgb_range=1.0):
        self.rgb_range = rgb_range
        self.criterion = criterion


if __name__ == '__main__':
    model = otias_c_x4(n_select_bands=3, n_bands=31, feat_dim=128,
                      guide_dim=128, sz=64).cuda()
    hr = torch.rand(1, 3, 64, 64).cuda()
    lr = torch.rand(1, 31, 16, 16).cuda()
    lms = torch.rand(1, 31, 64, 64).cuda()
    T = model(hr, lms, lr)
    print(T[0].shape)
    print(flop_count_table(FlopCountAnalysis(model, (hr, lms, lr))))
    ### 2.99M                 | 8.722G ###

