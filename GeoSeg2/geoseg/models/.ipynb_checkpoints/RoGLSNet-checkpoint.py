import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Any, Optional, Tuple
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm
import os

from geoseg.models.dfformer import cdfformer_s18

# from dfformer import cdfformer_s18

HF_ENDPOINT = "HF_ENDPOINT=https://hf-mirror.com huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0"


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


class AuxHead(nn.Module):

    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x, h, w):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
        return feat


## RoPE functions
def init_t_xy(end_x: int, end_y: int, zero_center=False):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode='floor').float()

    return t_x, t_y

class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

def init_random_2d_freqs(head_dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
    freqs_x = []
    freqs_y = []
    theta = theta
    mag = 1 / (theta ** (torch.arange(0, head_dim, 4)[: (head_dim // 4)].float() / head_dim))
    for i in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)
        fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi / 2 + angles)], dim=-1)
        fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi / 2 + angles)], dim=-1)
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    freqs = torch.stack([freqs_x, freqs_y], dim=0)
    return freqs


def compute_cis(freqs, t_x, t_y):
    N = t_x.shape[0]
    # No float 16 for this range
    with torch.cuda.amp.autocast(enabled=False):
        freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2))
        freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2))
        freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)

    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # x.shape [1 128 64 32]     freqs_cis.shape  [8 64 8]
    ndim = x.ndim
    assert 0 <= 1 < ndim
    # assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 3 else 1 for i, d in enumerate(x.shape)]

    return freqs_cis.view(*shape)


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq [1 128 64 64] -> [1 128 64 32]    freqs_cis[8 64 8]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


class SpatialGatherModule(nn.Module):
    def __init__(self, scale=1):
        super(SpatialGatherModule, self).__init__()
        self.scale = scale

    def forward(self, features, probs):
        batch_size, num_classes, h, w = probs.size()
        #  把 HW 变为一行/一列，相当于把整个特征图变为一行/一列， softmax 相当于 每个通道的特征图之和为1
        probs = probs.view(batch_size, num_classes, -1)  # batch * k * hw
        probs = F.softmax(self.scale * probs, dim=2)

        features = features.view(batch_size, features.size(1), -1)
        features = features.permute(0, 2, 1)  # batch * hw * c

        # ocr_context 的一个值就相当于    prob的每个类别 的特征概率图  *   features 每个通道的 特征图，
        # 最后得到的结果是 类别1在通道1的值，类别1在通道2的值。。，类。。。。。。， 类别2在通道1的值别， 2在通道2的值。。。。。。
        ocr_context = torch.matmul(probs, features)  # (B, k, c)
        ocr_context = ocr_context.permute(0, 2, 1).contiguous().unsqueeze(-1)  # (B, C, K, 1)

        return ocr_context


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """

    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


def resize_complex_weight(origin_weight, new_h, new_w):
    h, w, num_heads = origin_weight.shape[0:3]  # size, w, c, 2
    origin_weight = origin_weight.reshape(1, h, w, num_heads * 2).permute(0, 3, 1, 2)
    new_weight = torch.nn.functional.interpolate(
        origin_weight,
        size=(new_h, new_w),
        mode='bicubic',
        align_corners=True
    ).permute(0, 2, 3, 1).reshape(new_h, new_w, num_heads, 2)
    return new_weight


class Mlp_FFT(nn.Module):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """

    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0.,
                 bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class VanillaAttentionBlock(nn.Module):
    def __init__(self, dim=128, num_heads=16, qkv_bias=False, window_size=8, relative_pos_embedding=False,
                 dropout=0.1, ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3 * dim, kernel_size=1, bias=qkv_bias)

        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1, padding=(window_size // 2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size // 2 - 1))

        self.relative_pos_embedding = relative_pos_embedding

        if self.relative_pos_embedding:
            """
            使用其他的相对位置编码方式，比如ROPE+
            """
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x):

        B, C, H, W = x.shape

        x = self.pad(x, self.ws)  # 是否能被窗口整除，不能则进行padding
        B, C, Hp, Wp = x.shape

        qkv = self.qkv(x)  # [1 128 * 3 8 8]
        # q,k,kv [1 16 64 8]
        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale  # (1 16 64 64)

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ v  # (1 16 64 8)

        # [1 128 8 8]
        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, ws1=self.ws, ws2=self.ws)

        attn = attn[:, :, :H, :W]

        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))

        out = self.pad_out(out)
        out = self.proj(out)
        # print(out.size())
        out = out[:, :, :H, :W]

        return out, q, k


class EfficientDynamicFilter(nn.Module):
    def __init__(self, dim, expansion_ratio=2, reweight_expansion_ratio=.25,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, num_filters=4, size=14, weight_resize=True,
                 **kwargs):
        super().__init__()
        size = to_2tuple(size)
        self.size = size[0]
        self.filter_size = size[1] // 2 + 1
        self.num_filters = num_filters
        self.dim = dim
        self.med_channels = int(expansion_ratio * dim)
        self.weight_resize = weight_resize
        self.pwconv1 = nn.Linear(dim, self.med_channels, bias=bias)
        self.act1 = act1_layer()
        self.reweight = Mlp_FFT(dim, reweight_expansion_ratio, num_filters * self.med_channels)
        self.complex_weights = nn.Parameter(
            torch.randn(self.size, self.filter_size, num_filters, 2,
                        dtype=torch.float32) * 0.02)
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(self.med_channels, dim, bias=bias)

    def forward(self, x):

        x = x.permute(0, 2, 3, 1)
        B, H, W, _ = x.shape

        routeing = self.reweight(x.mean(dim=(1, 2))).view(B, self.num_filters,
                                                          -1).softmax(dim=1)
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')

        if self.weight_resize:
            complex_weights = resize_complex_weight(self.complex_weights, x.shape[1],
                                                    x.shape[2])
            complex_weights = torch.view_as_complex(complex_weights.contiguous())
        else:
            complex_weights = torch.view_as_complex(self.complex_weights)

        routeing = routeing.to(torch.complex64)
        weight = torch.einsum('bfc,hwf->bhwc', routeing, complex_weights)
        if self.weight_resize:
            weight = weight.view(-1, x.shape[1], x.shape[2], self.med_channels)
        else:
            weight = weight.view(-1, self.size, self.filter_size, self.med_channels)
        #  x.shape [1 32 17 640]   weight.shape [1 14 8 640]
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')

        x = self.act2(x)
        x = self.pwconv2(x)

        x = x.permute(0, 3, 1, 2)

        return x


class GlobalCouplingAttention(nn.Module):
    def __init__(self, dim=256, num_heads=8, num_class=6, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.edf = EfficientDynamicFilter(dim, num_filters=4)
        self.vam = VanillaAttentionBlock(dim, num_heads=num_heads, window_size=window_size, qkv_bias=qkv_bias,
                                 relative_pos_embedding=False)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer(dim)

        self.global_gather = SpatialGatherModule()
        self.decoder_stage = nn.Sequential(ConvBNReLU(dim, dim),
                                           nn.Dropout2d(p=0.1, inplace=True),
                                           Conv(dim, num_class, kernel_size=1))
        self.post_conv = ConvBNReLU(dim, dim, kernel_size=3)

    def forward(self, x):
        x = x + self.drop_path(self.edf(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        pred = self.decoder_stage(x)
        global_center = self.global_gather(x, pred)
        gc = torch.mean(global_center, dim=2, keepdim=True)

        center_x = self.post_conv(x + gc * x)

        return center_x, pred


class Fusion(nn.Module):
    def __init__(self, dim=128, num_heads=16, num_class=6, qkv_bias=False, window_size=8, rope_theta=10.0,
                 relative_pos_embedding=True,
                 rope_mixed=True, patch_num=(4, 4)):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size
        self.patch_num = patch_num

        self.decoder_stage = nn.Sequential(ConvBNReLU(dim, dim),
                                           nn.Dropout2d(p=0.1, inplace=True),
                                           Conv(dim, num_class, kernel_size=1))

        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)

        self.CLC3x3_1 = nn.Sequential(
            nn.Conv2d(self.num_heads * 2, self.num_heads, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(self.num_heads, self.num_heads, kernel_size=3, stride=1, padding=1, bias=False))
        self.CLC3x3_2 = nn.Sequential(
            nn.Conv2d(self.num_heads * 2, self.num_heads, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(self.num_heads, self.num_heads, kernel_size=3, stride=1, padding=1, bias=False))

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1, padding=(window_size // 2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size // 2 - 1))
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.q = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
                               nn.BatchNorm2d(dim),
                               nn.ReLU(inplace=True)
                               )
        self.k = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
                               nn.BatchNorm2d(dim),
                               nn.ReLU(inplace=True)
                               )
        self.v = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
                               nn.BatchNorm2d(dim),
                               nn.ReLU(inplace=True)
                               )
        self.post_conv = ConvBNReLU(dim, dim, kernel_size=3)
        self.local_gather = SpatialGatherModule()

        self.relative_pos_embedding = relative_pos_embedding
        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

        self.rope_mixed = rope_mixed
        # t_x, t_y = init_t_xy(end_x=self.window_size[1], end_y=self.window_size[0])
        t_x, t_y = init_t_xy(end_x=self.ws, end_y=self.ws)
        self.register_buffer('rope_t_x', t_x)
        self.register_buffer('rope_t_y', t_y)

        freqs = init_random_2d_freqs(
            head_dim=dim // self.num_heads, num_heads=self.num_heads, theta=rope_theta,
            rotate=self.rope_mixed
        )
        if self.rope_mixed:
            self.rope_freqs = nn.Parameter(freqs, requires_grad=True)
        else:
            self.register_buffer('rope_freqs', freqs)
            freqs_cis = compute_cis(self.rope_freqs, self.rope_t_x, self.rope_t_y)
            self.rope_freqs_cis = freqs_cis

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, R, S):
        B, C, H, W = R.size()

        S = self.up4(S)

        D = self.decoder_stage(R)
        local_center = self.local_gather(R, D)  # #(B, C, K, 1)
        lc = torch.mean(local_center, dim=2, keepdim=True)
        center_R = self.post_conv(R + lc * R)

        S = self.pad(S, self.ws)  # 是否能被窗口整除，不能则进行padding
        R = self.pad(R, self.ws)  # 是否能被窗口整除，不能则进行padding
        center_R = self.pad(center_R, self.ws)

        B, C, Hp, Wp = S.shape

        q = self.q(R)
        # k = self.k(R)
        k = self.k(center_R)
        v = self.v(S)

        query = rearrange(q, 'b (h d) (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                          d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, ws1=self.ws, ws2=self.ws)
        key = rearrange(k, 'b (h d) (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                        d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, ws1=self.ws, ws2=self.ws)

        value = rearrange(v, 'b (h d) (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                          d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, ws1=self.ws, ws2=self.ws)

        if self.rope_mixed:
            freqs_cis = compute_cis(self.rope_freqs, self.rope_t_x, self.rope_t_y)
        else:
            freqs_cis = self.rope_freqs_cis.to(R.device)
        query, key = apply_rotary_emb(query.contiguous(), key.contiguous(), freqs_cis)

        dots = (query @ key.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ value

        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, ws1=self.ws, ws2=self.ws)

        attn = attn[:, :, :H, :W]

        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))

        out = self.pad_out(out)
        out = self.proj(out)
        # print(out.size())
        out = out[:, :, :H, :W]

        return out


class RoGLSNet(nn.Module):
    def __init__(self,
                 decode_channels=64,
                 dropout=0.1,
                 # backbone_name='swsl_resnet18',
                 pretrained=True,
                 window_size=8,
                 num_classes=6,
                 ):
        super().__init__()

        encoder_channels = (64, 128, 320, 512)
        self.backbone = cdfformer_s18(features_only=True, pretrained=True)

        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))

        self.pre_conv4 = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        self.pre_conv3 = ConvBN(encoder_channels[-2], decode_channels, kernel_size=1)
        self.pre_conv2 = ConvBN(encoder_channels[-3], decode_channels, kernel_size=1)
        self.pre_conv1 = ConvBN(encoder_channels[-4], decode_channels, kernel_size=1)

        self.gca = GlobalCouplingAttention(decode_channels, num_heads=8, window_size=window_size)

        self.fusion = Fusion(decode_channels, num_heads=8, window_size=window_size)

        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.CLC3x3_1 = nn.Sequential(
            nn.Conv2d(decode_channels * 3, decode_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(decode_channels, decode_channels, kernel_size=3, stride=1, padding=1, bias=False))

        self.CLC3x3_2 = nn.Sequential(
            nn.Conv2d(decode_channels * 2, decode_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(decode_channels, decode_channels, kernel_size=3, stride=1, padding=1, bias=False))

        self.aux_head = AuxHead(decode_channels, num_classes)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.wf = WF(decode_channels, decode_channels)

        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

        self.pa = nn.Sequential(
            nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1, groups=decode_channels),
            nn.Sigmoid())
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                Conv(decode_channels, decode_channels // 16, kernel_size=1),
                                nn.ReLU6(),
                                Conv(decode_channels // 16, decode_channels, kernel_size=1),
                                nn.Sigmoid())

        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ReLU6()

    def forward(self, x):
        h, w = x.size()[-2:]
        res1, res2, res3, res4 = self.backbone(x)

        res4 = self.pre_conv4(res4)
        res3 = self.pre_conv3(res3)
        res2 = self.pre_conv2(res2)
        res1 = self.pre_conv1(res1)

        # S, Q, K = self.gca(res4)
        S, ah = self.gca(res4)

        res4 = self.up4(res4)
        res3 = self.up2(res3)
        R = torch.cat([res2, res3, res4], dim=1)
        R = self.CLC3x3_1(R)

        feature = self.fusion(R, S)
        # feature = R

        S = self.up4(S)
        res = self.CLC3x3_2(torch.cat((S, feature), dim=1))

        out = self.wf(res, res1)  # res 需要上采样

        shortcut = self.shortcut(out)
        pa = self.pa(out) * out
        ca = self.ca(out) * out
        out = pa + ca
        out = self.proj(out) + shortcut
        out = self.act(out)

        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        out = self.segmentation_head(out)

        if self.training:
            return out, ah
        else:
            return out


if __name__ == '__main__':
    data = torch.rand(1, 3, 256, 256).to('cpu')
    # data = torch.rand(1, 3, 768, 768).to('cpu')
    net = RoGLSNet(num_classes=6).to('cpu')

    out = net(data)
    print(out[0].shape)
