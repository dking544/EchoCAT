import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..basic.build_layer import build_conv_layer, build_norm_layer
from .base_module import BaseModule

from .helpers import to_2tuple


def resize_pos_embed(pos_embed,
                     src_shape,
                     dst_shape,
                     mode='bicubic',
                     num_extra_tokens=1):


    if src_shape[0] == dst_shape[0] and src_shape[1] == dst_shape[1]:
        return pos_embed
    assert pos_embed.ndim == 3, 'shape of pos_embed must be [1, L, C]'
    _, L, C = pos_embed.shape
    src_h, src_w = src_shape
    assert L == src_h * src_w + num_extra_tokens, \
        f"The length of `pos_embed` ({L}) doesn't match the expected " \
        f'shape ({src_h}*{src_w}+{num_extra_tokens}). Please check the' \
        '`img_size` argument.'
    extra_tokens = pos_embed[:, :num_extra_tokens]

    src_weight = pos_embed[:, num_extra_tokens:]
    src_weight = src_weight.reshape(1, src_h, src_w, C).permute(0, 3, 1, 2)

    dst_weight = F.interpolate(
        src_weight, size=dst_shape, align_corners=False, mode=mode)
    dst_weight = torch.flatten(dst_weight, 2).transpose(1, 2)

    return torch.cat((extra_tokens, dst_weight), dim=1)

def resize_relative_position_bias_table(src_shape, dst_shape, table, num_head):


    from scipy import interpolate

    def geometric_progression(a, r, n):
        return a * (1.0 - r**n) / (1.0 - r)

    left, right = 1.01, 1.5
    while right - left > 1e-6:
        q = (left + right) / 2.0
        gp = geometric_progression(1, q, src_shape // 2)
        if gp > dst_shape // 2:
            right = q
        else:
            left = q

    dis = []
    cur = 1
    for i in range(src_shape // 2):
        dis.append(cur)
        cur += q**(i + 1)

    r_ids = [-_ for _ in reversed(dis)]

    x = r_ids + [0] + dis
    y = r_ids + [0] + dis

    t = dst_shape // 2.0
    dx = np.arange(-t, t + 0.1, 1.0)
    dy = np.arange(-t, t + 0.1, 1.0)

    all_rel_pos_bias = []

    for i in range(num_head):
        z = table[:, i].view(src_shape, src_shape).float().numpy()
        f_cubic = interpolate.interp2d(x, y, z, kind='cubic')
        all_rel_pos_bias.append(
            torch.Tensor(f_cubic(dx,
                                 dy)).contiguous().view(-1,
                                                        1).to(table.device))
    new_rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
    return new_rel_pos_bias


class PatchEmbed(BaseModule):


    def __init__(self,
                 img_size=224,
                 in_channels=3,
                 embed_dims=768,
                 norm_cfg=None,
                 conv_cfg=None,
                 init_cfg=None):
        super(PatchEmbed, self).__init__(init_cfg)
        warnings.warn('The `PatchEmbed` in mmcls will be deprecated. '
                      'Please use `mmcv.cnn.bricks.transformer.PatchEmbed`. '
                      "It's more general and supports dynamic input shape")

        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, tuple):
            if len(img_size) == 1:
                img_size = to_2tuple(img_size[0])
            assert len(img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(img_size)}'

        self.img_size = img_size
        self.embed_dims = embed_dims


        conv_cfg = conv_cfg or dict()
        _conv_cfg = dict(
            type='Conv2d', kernel_size=16, stride=16, padding=0, dilation=1)
        _conv_cfg.update(conv_cfg)
        self.projection = build_conv_layer(_conv_cfg, in_channels, embed_dims)


        h_out, w_out = [(self.img_size[i] + 2 * self.projection.padding[i] -
                         self.projection.dilation[i] *
                         (self.projection.kernel_size[i] - 1) - 1) //
                        self.projection.stride[i] + 1 for i in range(2)]

        self.patches_resolution = (h_out, w_out)
        self.num_patches = h_out * w_out

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't " \
            f'match model ({self.img_size[0]}*{self.img_size[1]}).'

        x = self.projection(x).flatten(2).transpose(1, 2)

        if self.norm is not None:
            x = self.norm(x)

        return x


class HybridEmbed(BaseModule):


    def __init__(self,
                 backbone,
                 img_size=224,
                 feature_size=None,
                 in_channels=3,
                 embed_dims=768,
                 conv_cfg=None,
                 init_cfg=None):
        super(HybridEmbed, self).__init__(init_cfg)
        assert isinstance(backbone, nn.Module)
        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, tuple):
            if len(img_size) == 1:
                img_size = to_2tuple(img_size[0])
            assert len(img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(img_size)}'

        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():


                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(
                    torch.zeros(1, in_channels, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):

                    o = o[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        self.num_patches = feature_size[0] * feature_size[1]


        conv_cfg = conv_cfg or dict()
        _conv_cfg = dict(
            type='Conv2d', kernel_size=1, stride=1, padding=0, dilation=1)
        _conv_cfg.update(conv_cfg)
        self.projection = build_conv_layer(_conv_cfg, feature_dim, embed_dims)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):

            x = x[-1]
        x = self.projection(x).flatten(2).transpose(1, 2)
        return x


class PatchMerging(BaseModule):


    def __init__(self,
                 input_resolution,
                 in_channels,
                 expansion_ratio,
                 kernel_size=2,
                 stride=None,
                 padding=0,
                 dilation=1,
                 bias=False,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super().__init__(init_cfg)
        warnings.warn('The `PatchMerging` in mmcls will be deprecated. '
                      'Please use `mmcv.cnn.bricks.transformer.PatchMerging`. '
                      "It's more general and supports dynamic input shape")

        H, W = input_resolution
        self.input_resolution = input_resolution
        self.in_channels = in_channels
        self.out_channels = int(expansion_ratio * in_channels)

        if stride is None:
            stride = kernel_size
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        dilation = to_2tuple(dilation)
        self.sampler = nn.Unfold(kernel_size, dilation, padding, stride)

        sample_dim = kernel_size[0] * kernel_size[1] * in_channels

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, sample_dim)[1]
        else:
            self.norm = None

        self.reduction = nn.Linear(sample_dim, self.out_channels, bias=bias)


        H_out = (H + 2 * padding[0] - dilation[0] *
                 (kernel_size[0] - 1) - 1) // stride[0] + 1
        W_out = (W + 2 * padding[1] - dilation[1] *
                 (kernel_size[1] - 1) - 1) // stride[1] + 1
        self.output_resolution = (H_out, W_out)

    def forward(self, x):


        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'

        x = x.view(B, H, W, C).permute([0, 3, 1, 2])


        x = self.sampler(x)
        x = x.transpose(1, 2)

        x = self.norm(x) if self.norm else x
        x = self.reduction(x)

        return x
