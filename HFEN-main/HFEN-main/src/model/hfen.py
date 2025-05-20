import torch
import torch.nn as nn
from . import Module
import torch.nn.functional as F
def make_model(args, parent=False):
    model = HFEN()
    return model

class FD(nn.Module):
    def __init__(self, dim):
        super(FD, self).__init__()

        self.blk = Module.FFBG(dim)
        self.window_size=8
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        if mod_pad_h != 0 or mod_pad_w != 0:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x, h, w
    def forward(self, x,relative_position_index_OCA):
        x, h, w = self.check_image_size(x)
        _, _, H, W = x.size()

        x=self.blk(x,relative_position_index_OCA)

        if h != H or w != W:
            x = x[:, :, 0:h, 0:w].contiguous()
        return x

class HFEN(nn.Module):
    def __init__(self, in_nc=3, nf=56, num_modules=5, out_nc=3, upscale=4):
        super(HFEN, self).__init__()
        self.fea_conv =Module.BSConvU(in_nc, nf, kernel_size=3)
        self.fea = Module.BSConvU(nf, nf, kernel_size=3)
        self.B1 = FD(nf)
        self.B2 = FD(nf)
        self.B3 = FD(nf)
        self.B4 = FD(nf)
        self.B5 = FD(nf)

        self.c = Module.conv_block(nf * num_modules, nf, kernel_size=1, act_type='gelu')
        upsample_block = Module.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)
        self.overlap_ratio = 0.5
        self.window_size = 8
    def calculate_rpi_oca(self):
        window_size_ori = self.window_size
        window_size_ext = self.window_size + int(self.overlap_ratio * self.window_size)
        coords_h = torch.arange(window_size_ori)
        coords_w = torch.arange(window_size_ori)
        coords_ori = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, ws, ws
        coords_ori_flatten = torch.flatten(coords_ori, 1)  # 2, ws*ws
        coords_h = torch.arange(window_size_ext)
        coords_w = torch.arange(window_size_ext)
        coords_ext = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, wse, wse
        coords_ext_flatten = torch.flatten(coords_ext, 1)  # 2, wse*wse
        relative_coords = coords_ext_flatten[:, None, :] - coords_ori_flatten[:, :, None]  # 2, ws*ws, wse*wse
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # ws*ws, wse*wse, 2
        relative_coords[:, :, 0] += window_size_ori - window_size_ext + 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size_ori - window_size_ext + 1
        relative_coords[:, :, 0] *= window_size_ori + window_size_ext - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index
    def forward(self, input):
        relative_position_index_OCA = self.calculate_rpi_oca()
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea,relative_position_index_OCA)
        out_B2 = self.B2(out_B1,relative_position_index_OCA)
        out_B3 = self.B3(out_B2,relative_position_index_OCA)
        out_B4 = self.B4(out_B3,relative_position_index_OCA)
        out_B5 = self.B5(out_B4,relative_position_index_OCA)

        out_B = self.c(torch.cat([out_B1,out_B2,out_B3,out_B4,out_B5], dim=1))
        out_lr = self.fea(out_B) + out_fea
        output = self.upsampler(out_lr)

        return output
    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
