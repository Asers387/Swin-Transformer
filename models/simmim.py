

# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_

from .swin_transformer import SwinTransformer
from .swin_transformer_v2 import SwinTransformerV2
from .swinir import SwinIR


def norm_targets(targets, patch_size):
    assert patch_size % 2 == 1
    
    targets_ = targets
    targets_count = torch.ones_like(targets)

    targets_square = targets ** 2.
    
    targets_mean = F.avg_pool2d(targets, kernel_size=patch_size, stride=1, padding=patch_size // 2, count_include_pad=False)
    targets_square_mean = F.avg_pool2d(targets_square, kernel_size=patch_size, stride=1, padding=patch_size // 2, count_include_pad=False)
    targets_count = F.avg_pool2d(targets_count, kernel_size=patch_size, stride=1, padding=patch_size // 2, count_include_pad=True) * (patch_size ** 2)
    
    targets_var = (targets_square_mean - targets_mean ** 2.) * (targets_count / (targets_count - 1))
    targets_var = torch.clamp(targets_var, min=0.)
    
    targets_ = (targets_ - targets_mean) / (targets_var + 1.e-6) ** 0.5
    
    return targets_


class SwinTransformerForSimMIM(SwinTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.mask_token, mean=0., std=.02)

    def forward(self, x, mask):
        x = self.patch_embed(x)

        assert mask is not None
        B, L, _ = x.shape

        mask_tokens = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
        x = x * (1. - w) + mask_tokens * w

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        x = x.transpose(1, 2)
        B, C, L = x.shape
        H = W = int(L ** 0.5)
        x = x.reshape(B, C, H, W)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {'mask_token'}


class SwinTransformerV2ForSimMIM(SwinTransformerV2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.mask_token, mean=0., std=.02)

    def forward(self, x, mask):
        x = self.patch_embed(x)

        assert mask is not None
        B, L, _ = x.shape

        mask_tokens = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
        x = x * (1. - w) + mask_tokens * w

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        x = x.transpose(1, 2)
        B, C, L = x.shape
        H = W = int(L ** 0.5)
        x = x.reshape(B, C, H, W)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {'mask_token'}


class SwinIRForSimMIM(SwinIR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.mask_token, mean=0., std=.02)

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x, mask):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        
        x = self.conv_first(x)

        assert mask is not None
        B, C, H, W = x.shape

        mask = mask.repeat_interleave(4, 1).repeat_interleave(4, 2).unsqueeze(1).contiguous()

        mask_tokens = self.mask_token.expand(B, H, W, -1).permute(0, 3, 1, 2)
        w = mask.type_as(mask_tokens)
        x = x * (1. - w) + mask_tokens * w

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            # for real-world SR
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            if self.upscale == 4:
                x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            raise NotImplementedError

        return x[:, :, :H*self.upscale, :W*self.upscale]

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {'mask_token'}


class SimMIM(nn.Module):
    def __init__(self, config, encoder, encoder_stride, in_chans, patch_size):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.encoder_stride = encoder_stride

        # self.decoder = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=self.encoder.num_features,
        #         out_channels=self.encoder_stride ** 2 * 3, kernel_size=1),
        #     nn.PixelShuffle(self.encoder_stride),
        # )
        
        # class EfficientUpscale(nn.Module):
        #     def __init__(self, num_features, upscale_factor=2):
        #         super().__init__()

        #         assert num_features % upscale_factor == 0, 'Invalid upsample factor'

        #         self.conv2d = nn.Conv2d(num_features, num_features * upscale_factor**2, kernel_size=1)
        #         self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        #         # self.layer_norm = nn.LayerNorm(num_features // upscale_factor)

        #     def forward(self, x):
        #         x = self.conv2d(x)
        #         x = self.pixel_shuffle(x)

        #         # x = x.permute(0, 2, 3, 1)
        #         # x = self.layer_norm(x)
        #         # x = x.permute(0, 3, 1, 2)
        #         return x

        # for i in range(4):
        #     for j in range(2):
        #         decoder_layers.append(EfficientUpscale(num_features, upscale_factor=2))
        #     num_features = num_features // 2

        # num_features = self.encoder.num_features

        # decoder_layers = []
        # for i in range(4):
        #     decoder_layers.append(nn.Conv2d(num_features, num_features * 4, kernel_size=1))
        #     decoder_layers.append(nn.PixelShuffle(2))
        #     decoder_layers.append(nn.Conv2d(num_features, num_features * 2, kernel_size=1))
        #     decoder_layers.append(nn.PixelShuffle(2))
        #     num_features //= 2
        # decoder_layers.append(nn.Conv2d(num_features, 3, kernel_size=1))
        # self.decoder = nn.Sequential(*decoder_layers)
        
        self.in_chans = in_chans
        self.patch_size = patch_size

    def forward(self, img_lr, img_vhr, mask_lr):
        z = self.encoder(img_lr, mask_lr)
        # img_vhr_rec = self.decoder(z)
        img_vhr_rec = z
        
        mask_vhr = mask_lr.repeat_interleave(self.patch_size * 8, 1).repeat_interleave(self.patch_size * 8, 2).unsqueeze(1).contiguous()
        
        # norm target as prompted
        if self.config.NORM_TARGET.ENABLE:
            img_vhr = norm_targets(img_vhr, self.config.NORM_TARGET.PATCH_SIZE * 8 - 1)
        
        loss_recon = F.l1_loss(img_vhr, img_vhr_rec, reduction='none')
        loss = (loss_recon * mask_vhr).sum() / (mask_vhr.sum() + 1e-5) / self.in_chans
        
        return img_vhr_rec, mask_vhr, loss

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}


def build_simmim(config, layernorm):
    model_type = config.MODEL.TYPE

    kwargs = dict(
        img_size=config.DATA.IMG_SIZE,
        patch_size=config.MODEL.SWIN.PATCH_SIZE,
        in_chans=config.MODEL.SWIN.IN_CHANS,
        drop_rate=config.MODEL.DROP_RATE,
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
        norm_layer=layernorm,
        use_checkpoint=config.TRAIN.USE_CHECKPOINT
    )

    encoder_stride = 32

    if model_type == 'swin':
        encoder = SwinTransformerForSimMIM(
            num_classes=0,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            **kwargs)
        in_chans = config.MODEL.SWIN.IN_CHANS
        patch_size = config.MODEL.SWIN.PATCH_SIZE
    elif model_type == 'swinv2':
        encoder = SwinTransformerV2ForSimMIM(
            num_classes=0,
            embed_dim=config.MODEL.SWINV2.EMBED_DIM,
            depths=config.MODEL.SWINV2.DEPTHS,
            num_heads=config.MODEL.SWINV2.NUM_HEADS,
            window_size=config.MODEL.SWINV2.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWINV2.MLP_RATIO,
            qkv_bias=config.MODEL.SWINV2.QKV_BIAS,
            ape=config.MODEL.SWINV2.APE,
            patch_norm=config.MODEL.SWINV2.PATCH_NORM,
            **kwargs)
        in_chans = config.MODEL.SWINV2.IN_CHANS
        patch_size = config.MODEL.SWINV2.PATCH_SIZE
    elif model_type == 'swinir':
        encoder = SwinIRForSimMIM(
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            upscale=8,
            img_range=1.,
            upsampler='pixelshuffledirect',
            **kwargs)
        in_chans = config.MODEL.SWIN.IN_CHANS
        patch_size = config.MODEL.SWIN.PATCH_SIZE
    else:
        raise NotImplementedError(f"Unknown pre-train model: {model_type}")

    model = SimMIM(config=config.MODEL.SIMMIM, encoder=encoder, encoder_stride=encoder_stride, in_chans=in_chans, patch_size=patch_size)

    return model