from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath
from mmcv.cnn.bricks import build_activation_layer
import torch
import torch.nn as nn
from natten import NeighborhoodAttention2D as NeighborhoodAttention
from einops import rearrange


class MultiScaleBSConv(nn.Module):
    def __init__(self, dim, scale=(1, 3, 5, 7)):
        super().__init__()
        self.scale = scale
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(scale)):
            if i == 0:
                channels = dim - dim // len(scale) * (len(scale) - 1)
            else:
                channels = dim // len(scale)
            conv =BSConvU(channels, channels,
                             kernel_size=scale[i],
                             padding=scale[i] // 2)
            self.channels.append(channels)
            self.proj.append(conv)

    def forward(self, x):
        x = torch.split(x, split_size_or_sections=self.channels, dim=1)
        out = []
        for i, feat in enumerate(x):
            out.append(self.proj[i](feat))
        x = torch.cat(out, dim=1)
        return x

class MAFN(nn.Module):  ### MS-FFN
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_cfg=dict(type='GELU'),
                 drop=0, ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=False),
            build_activation_layer(act_cfg),
            nn.BatchNorm2d(hidden_features),
        )
        self.dwconv = MultiScaleBSConv(hidden_features)
        self.act = build_activation_layer(act_cfg)
        self.norm = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, in_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_features),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)

        x = self.dwconv(x)+x
        x = self.norm(self.act(x))

        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn
class Attention(nn.Module):
    def __init__(self, d_model,  num_heads=4):
        super().__init__()
        self.proj = nn.Conv2d(d_model, d_model * 2, 1)
        self.proj_1 = nn.Conv2d(d_model, d_model*2, 1)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)
        self.proj_3 = nn.Conv2d(d_model * 2, d_model, 1)
        self.project_out = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.oca = OCA(d_model)
        self.LKA = LKA(d_model)
        self.conv = nn.Conv2d(d_model*2, d_model*2, 3, padding=1, groups=d_model*2)
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.patch_embed = PatchEmbed(embed_dim=d_model)
        self.window_size = 8
        self.patch_unembed = PatchUnEmbed(embed_dim=d_model)
    def forward(self, x,relative_position_index_OCA):
        b, c, h, w = x.shape
        _, _, H, W = x.size()
        x_size = (H, W)
        x=self.proj(x)
        x,x1=x.chunk(2, dim=1)

        x1 = self.patch_embed(x1)
        x1 = self.oca(x1, x_size, relative_position_index_OCA)
        x1 = self.patch_unembed(x1, x_size)

        qk = self.conv(self.proj_1(x))
        v = self.LKA(self.proj_2(x))
        q, k = qk.chunk(2, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        x = (attn @ v)
        x= rearrange(x, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        x = self.project_out(x)

        out=self.proj_3(torch.cat([x1,x],dim=1))

        return out
class OCA(nn.Module):
    # overlapping cross-attention block
    def __init__(self, dim,
                 input_resolution=64,
                 window_size=8,
                 overlap_ratio=0.5,
                 num_heads=4,
                 qkv_bias=True,
                 qk_scale=None,

                 ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size), stride=window_size,
                                padding=(self.overlap_win_size - window_size) // 2)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((window_size + self.overlap_win_size - 1) * (window_size + self.overlap_win_size - 1),
                        num_heads))
        # 2*Wh-1 * 2*Ww-1, nH
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        self.proj = nn.Linear(dim, dim)

    def forward(self, x, x_size, rpi):
        h, w = x_size
        b, _, c = x.shape
        # shortcut = x

        x = x.view(b, h, w, c)

        ##########------Linear------##########
        qkv = self.qkv(x).reshape(b, h, w, 3, c).permute(3, 0, 4, 1, 2)  # 3, b, c, h, w
        q = qkv[0].permute(0, 2, 3, 1)  # b, h, w, c
        kv = torch.cat((qkv[1], qkv[2]), dim=1)  # b, 2*c, h, w
        ##########------Linear------##########

        ##########------q, k, v------##########
        q_windows = window_partition(q, self.window_size)  # nw*b, window_size, window_size, c
        q_windows = q_windows.view(-1, self.window_size * self.window_size, c)  # nw*b, window_size*window_size, c

        kv_windows = self.unfold(kv)  # b, c*w*w, nw
        kv_windows = rearrange(kv_windows, 'b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch', nc=2, ch=c,
                               owh=self.overlap_win_size, oww=self.overlap_win_size).contiguous()
        # 2, nw*b, ow*ow, c
        k_windows, v_windows = kv_windows[0], kv_windows[1]  # nw*b, ow*ow, c

        b_, nq, _ = q_windows.shape
        _, n, _ = k_windows.shape
        d = self.dim // self.num_heads
        q = q_windows.reshape(b_, nq, self.num_heads, d).permute(0, 2, 1, 3)  # nw*b, nH, nq, d
        k = k_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3)  # nw*b, nH, n, d
        v = v_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3)  # nw*b, nH, n, d
        ##########------q, k, v------##########

        ##########------q*k------##########
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        ##########------q*k------##########

        ##########------relative_position_bias------##########
        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size * self.window_size, self.overlap_win_size * self.overlap_win_size, -1)
        # ws*ws, wse*wse, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, ws*ws, wse*wse
        attn = attn + relative_position_bias.unsqueeze(0)
        ##########------relative_position_bias------##########

        ##########------k*v------##########
        attn = self.softmax(attn)
        attn_windows = (attn @ v).transpose(1, 2).reshape(b_, nq, self.dim)
        ##########------k*v------##########

        ##########------Merge_Windows------##########
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.dim)
        x = window_reverse(attn_windows, self.window_size, h, w)  # b h w c
        ##########------Merge_Windows------##########

        x = x.view(b, h * w, self.dim)
        x = self.proj(x)
        # x = self.proj(x) + shortcut
        #
        # ##########------LN+MLP------##########
        # x = x + self.mlp(self.norm2(x).reshape(b,h,w,c).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous().reshape(b,-1,c)
        # ##########------LN+MLP------##########

        return x

class FFBG(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=4,
        kernel_size=7,
        dilation=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 =nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.attn1 =Attention(dim)
        self.attn = NeighborhoodAttention(
            dim,
            kernel_size=kernel_size,
            dilation=dilation,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.mlp = MAFN(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       drop=drop)
        self.mlp1 = MAFN(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       drop=drop)
    def forward(self, x,relative_position_index_OCA):
        x = x + self.drop_path(self.attn(self.norm1(x.permute(0,2, 3,1).contiguous()))).permute(0,3,1,2).contiguous()
        x = x + self.drop_path(self.mlp(self.norm2(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()))

        x = x + self.drop_path(self.attn1(self.norm3(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1,2).contiguous(),relative_position_index_OCA))
        x = x + self.drop_path(self.mlp1(self.norm4(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()))

        return x

class PatchEmbed(nn.Module):
    def __init__(self, embed_dim=50, norm_layer=None):
        super().__init__()

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops
class PatchUnEmbed(nn.Module):
    def __init__(self, embed_dim=50):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops

def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)
def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)
def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act_type == 'silu':
        layer = nn.SiLU(inplace)
    elif act_type == 'gelu':
        layer = nn.GELU()
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) // 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)

def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

class BSConvU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None):
        super().__init__()
        self.with_ln = with_ln
        # check arguments
        if bn_kwargs is None:
            bn_kwargs = {}

        # pointwise
        self.pw=torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
        )

        # depthwise
        self.dw = torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=out_channels,
                bias=bias,
                padding_mode=padding_mode,
        )
        # self.project_out = nn.Conv2d(out_channels//2, out_channels, 1)
    def forward(self, fea):

        fea = self.pw(fea)
        fea= self.dw(fea)

        return fea


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)



