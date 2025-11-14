import torch
import torch.nn as nn

from derivatives import rot_mac


from unet_parts import *

def get_Net(params):
	if params.net == "UNet":
		pde_cnn = ULite()
	return pde_cnn

class AxialDW(nn.Module):

	def __init__(self, dim, mixer_kernel, dilation=1):
		super().__init__()
		h, w = mixer_kernel
		self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding=(((h - 1) * dilation) // 2, 0), groups=dim,
							  dilation=dilation)
		self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding=(0, ((w - 1) * dilation) // 2), groups=dim,
							  dilation=dilation)

	def forward(self, x):
		x = x + self.dw_h(x) + self.dw_w(x)
		return x


class EncoderBlock(nn.Module):
	"""Encoding then downsampling"""

	def __init__(self, in_c, out_c, mixer_kernel=(7, 7)):
		super().__init__()
		self.dw = AxialDW(in_c, mixer_kernel=(7, 7))
		self.bn = nn.BatchNorm2d(in_c)
		self.pw = nn.Conv2d(in_c, out_c, kernel_size=1)
		self.down = nn.MaxPool2d((2, 2))
		self.act = nn.GELU()

	def forward(self, x):
		skip = self.bn(self.dw(x))
		x = self.act(self.down(self.pw(skip)))
		return x, skip


class DecoderBlock(nn.Module):
	"""Upsampling then decoding"""

	def __init__(self, in_c, out_c, mixer_kernel=(7, 7)):
		super().__init__()
		self.up = nn.Upsample(scale_factor=2)
		self.pw = nn.Conv2d(in_c + out_c, out_c, kernel_size=1)
		self.bn = nn.BatchNorm2d(out_c)
		self.dw = AxialDW(out_c, mixer_kernel=(7, 7))
		self.act = nn.GELU()
		self.pw2 = nn.Conv2d(out_c, out_c, kernel_size=1)

	def forward(self, x, skip):
		x = self.up(x)
		x = torch.cat([x, skip], dim=1)
		x = self.act(self.pw2(self.dw(self.bn(self.pw(x)))))
		return x


class BottleNeckBlock(nn.Module):
	"""Axial dilated DW convolution"""

	def __init__(self, dim):
		super().__init__()

		gc = dim // 4
		self.pw1 = nn.Conv2d(dim, gc, kernel_size=1)
		self.dw1 = AxialDW(gc, mixer_kernel=(3, 3), dilation=1)
		self.dw2 = AxialDW(gc, mixer_kernel=(3, 3), dilation=2)
		self.dw3 = AxialDW(gc, mixer_kernel=(3, 3), dilation=3)

		self.bn = nn.BatchNorm2d(4 * gc)
		self.pw2 = nn.Conv2d(4 * gc, dim, kernel_size=1)
		self.act = nn.GELU()

	def forward(self, x):
		x = self.pw1(x)
		x = torch.cat([x, self.dw1(x), self.dw2(x), self.dw3(x)], 1)
		x = self.act(self.pw2(self.bn(x)))
		return x


class ULite(nn.Module):
	def __init__(self):
		super().__init__()

		"""Encoder"""
		self.conv_in = nn.Conv2d(15, 16, kernel_size=7, padding=((7 - 1) * 1) // 2)
		self.e1 = EncoderBlock(16, 32)
		self.e1_cbam = CBAM(32)
		self.e2 = EncoderBlock(32, 64)
		self.e2_cbam = CBAM(64)
		self.e3 = EncoderBlock(64, 128)
		self.e3_cbam = CBAM(128)
		self.e4 = EncoderBlock(128, 256)
		self.e4_cbam = CBAM(256)
		self.e5 = EncoderBlock(256, 512)
		self.e5_cbam = CBAM(512)

		"""Bottle Neck"""
		self.b5 = BottleNeckBlock(512)

		"""Decoder"""
		self.d5 = DecoderBlock(512, 256)
		self.d5_cbam = CBAM(256)
		self.d4 = DecoderBlock(256, 128)
		self.d4_cbam = CBAM(128)
		self.d3 = DecoderBlock(128, 64)
		self.d3_cbam = CBAM(64)
		self.d2 = DecoderBlock(64, 32)
		self.d2_cbam = CBAM(32)
		self.d1 = DecoderBlock(32, 16)
		self.d1_cbam = CBAM(16)
		self.conv_out = nn.Conv2d(16, 2, kernel_size=1)

	def forward(self, a_old, p_old, mask_flow, v_cond, mask_cond):
		v_old = rot_mac(a_old)
		x = torch.cat(
			[p_old, a_old, v_old, mask_flow, v_cond * mask_cond, mask_cond, mask_flow * p_old, mask_flow * v_old,
			 v_old * mask_cond, v_cond], dim=1)
		"""Encoder"""
		x = self.conv_in(x)
		x, skip1 = self.e1(x)
		x = self.e1_cbam(x)
		x, skip2 = self.e2(x)
		x = self.e2_cbam(x)
		x, skip3 = self.e3(x)
		x = self.e3_cbam(x)
		x, skip4 = self.e4(x)
		x = self.e4_cbam(x)
		x, skip5 = self.e5(x)
		x = self.e5_cbam(x)

		"""BottleNeck"""
		# x = self.b5(x)         # (512, 8, 8)

		"""Decoder"""
		x = self.d5(x, skip5)
		x = self.d5_cbam(x)

		x = self.d4(x, skip4)
		x = self.d4_cbam(x)

		x = self.d3(x, skip3)
		x = self.d3_cbam(x)

		x = self.d2(x, skip2)
		x = self.d2_cbam(x)

		x = self.d1(x, skip1)
		x = self.d1_cbam(x)
		x = self.conv_out(x)

		a_new, p_new = 400 * torch.tanh((a_old + x[:, 0:1]) / 400), 10 * torch.tanh((p_old + x[:, 1:2]) / 10)
		return a_new, p_new
