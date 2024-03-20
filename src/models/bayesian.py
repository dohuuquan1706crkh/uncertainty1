import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor
from torch.distributions.normal import Normal
from models.attention import ChannelAttention, SpatialAttention

class ResidualConvBlock(nn.Module):
	"""Implements residual conv function.

	Args:
		channels (int): Number of channels in the input image.
	"""

	def __init__(self, channels: int) -> None:
		super(ResidualConvBlock, self).__init__()
		self.rcb = nn.Sequential(
			nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
			nn.BatchNorm2d(channels),
			nn.PReLU(),
			nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
			nn.BatchNorm2d(channels),
		)

	def forward(self, x: Tensor) -> Tensor:
		identity = x

		out = self.rcb(x)
		out = torch.add(out, identity)

		return out

class BayesCap(nn.Module):
	def __init__(self, in_channels=2, out_channels=2) -> None:
		super(BayesCap, self).__init__()
		# First conv layer.
		self.conv_block1 = nn.Sequential(
			nn.Conv2d(
				in_channels, 64, 
				kernel_size=9, stride=1, padding=4
			),
			nn.PReLU(),
		)

		# Features trunk blocks.
		trunk = []
		for _ in range(16):
			trunk.append(ResidualConvBlock(64))
		self.trunk = nn.Sequential(*trunk)

		# Second conv layer.
		self.conv_block2 = nn.Sequential(
			nn.Conv2d(
				64, 64, 
				kernel_size=3, stride=1, padding=1, bias=False
			),
			nn.BatchNorm2d(64),
		)

		# Output layer.
		self.conv_block3_mu = nn.Conv2d(
			64, out_channels=out_channels, 
			kernel_size=9, stride=1, padding=4
		)

		self.conv_block3_alpha = nn.Sequential(
			nn.Conv2d(
				64, 64, 
				kernel_size=9, stride=1, padding=4
			),
			nn.PReLU(),
			nn.Conv2d(
				64, 64, 
				kernel_size=9, stride=1, padding=4
			),
			nn.PReLU(),
			nn.Conv2d(
				64, 1, 
				kernel_size=9, stride=1, padding=4
			),
			nn.ReLU(),
		)
		self.conv_block3_beta = nn.Sequential(
			nn.Conv2d(
				64, 64, 
				kernel_size=9, stride=1, padding=4
			),
			nn.PReLU(),
			nn.Conv2d(
				64, 64, 
				kernel_size=9, stride=1, padding=4
			),
			nn.PReLU(),
			nn.Conv2d(
				64, 1, 
				kernel_size=9, stride=1, padding=4
			),
			nn.ReLU(),
		)

		# Initialize neural network weights.
		self._initialize_weights()

	def forward(self, x: Tensor) -> Tensor:
		return self._forward_impl(x)

	# Support torch.script function.
	def _forward_impl(self, x: Tensor) -> Tensor:
		out1 = self.conv_block1(x)
		out = self.trunk(out1)
		out2 = self.conv_block2(out)
		out = out1 + out2
		out_mu = self.conv_block3_mu(out)
		out_alpha = self.conv_block3_alpha(out)
		out_beta = self.conv_block3_beta(out)
		return out_mu, out_alpha, out_beta

	def _initialize_weights(self) -> None:
		for module in self.modules():
			if isinstance(module, nn.Conv2d):
				nn.init.kaiming_normal_(module.weight)
				if module.bias is not None:
					nn.init.constant_(module.bias, 0)
			elif isinstance(module, nn.BatchNorm2d):
				nn.init.constant_(module.weight, 1)

class BNN(nn.Module):
	"""
	This class is a family of BayesCap framework
	It outputs a sigma of gaussian distribution
	"""
	def __init__(self, in_channels=2, out_channels=2) -> None:
		super(BNN, self).__init__()
		# First conv layer.
		self.conv_block1 = nn.Sequential(
			nn.Conv2d(
				in_channels, 64, 
				kernel_size=9, stride=1, padding=4
			),
			nn.PReLU(),
		)

		# Features trunk blocks.
		trunk = []
		for _ in range(16):
			trunk.append(ResidualConvBlock(64))
		self.trunk = nn.Sequential(*trunk)

		# Second conv layer.
		self.conv_block2 = nn.Sequential(
			nn.Conv2d(
				64, 64, 
				kernel_size=3, stride=1, padding=1, bias=False
			),
			nn.BatchNorm2d(64),
		)

		# Output layer.
		self.conv_block3_mu = nn.Conv2d(
			64, out_channels=out_channels, 
			kernel_size=9, stride=1, padding=4
			# kernel_size=3, padding=1
		)

		## one_over_sigma
		self.conv_gauss = nn.Sequential(
			nn.Conv2d(
				64, 64, 
				kernel_size=9, stride=1, padding=4
			),
			nn.PReLU(),
			nn.Conv2d(
				64, 64, 
				kernel_size=9, stride=1, padding=4
			),
			nn.PReLU(),
			nn.Conv2d(
				64, 1, 
				kernel_size=9, stride=1, padding=4
			),
			nn.ReLU(),
		)
		## sigma_square
		# self.conv_gauss = nn.Sequential(
		# 	nn.Conv2d(
		# 		64, 64, 
		# 		kernel_size=9, stride=1, padding=4
		# 	),
		# 	nn.PReLU(),
		# 	nn.Conv2d(
		# 		64, 64, 
		# 		kernel_size=9, stride=1, padding=4
		# 	),
		# 	nn.PReLU(),
		# 	nn.Conv2d(
		# 		64, 1, 
		# 		kernel_size=9, stride=1, padding=4
		# 	),
		# 	nn.Sigmoid()
		# )
		# Initialize neural network weights
		self._initialize_weights()
		## initialize conv_block3_logvar
		# for m in self.conv_logvar.modules():
		# 	if isinstance(m, nn.Conv2d):
		# 		nn.init.normal_(m.weight, mean=0.0, std=1e-4)
		# 		nn.init.constant_(m.bias, 0)

	def forward(self, x: Tensor) -> Tensor:
		return self._forward_impl(x)

	# Support torch.script function.
	def _forward_impl(self, x: Tensor) -> Tensor:
		out1 = self.conv_block1(x)
		out = self.trunk(out1)
		out2 = self.conv_block2(out)
		out = out1 + out2
		mu = self.conv_block3_mu(out)
		gauss = self.conv_gauss(out)
		return mu, gauss

	def _initialize_weights(self) -> None:
		for module in self.modules():
			if isinstance(module, nn.Conv2d):
				nn.init.kaiming_normal_(module.weight)
				if module.bias is not None:
					nn.init.constant_(module.bias, 0)
			elif isinstance(module, nn.BatchNorm2d):
				nn.init.constant_(module.weight, 1)

## With using the context
class GaussCapContext(nn.Module):
	"""
	This class is a family of BayesCap framework
	It outputs a sigma of gaussian distribution
	"""
	def __init__(
			self,
			yhat_channels:int=2,
			img_channels:int=3,
			out_channels:int=2,
			fusion_mode: str="channel_att"
	) -> None:
		super(GaussCapContext, self).__init__()
		# First conv layer.
		in_channels = yhat_channels if fusion_mode != "concat" else (yhat_channels + img_channels)
		self.conv_yhat_block1 = nn.Sequential(
			nn.Conv2d(
				in_channels=in_channels,
				out_channels=64, 
				kernel_size=9,
				stride=1,
				padding=4
			),
			nn.PReLU(),
		)
		
		# Features trunk blocks.
		trunk = []
		for _ in range(16):	## trunk 16 blocks
			trunk.append(ResidualConvBlock(64))
		
		self.yhat_trunk = nn.Sequential(*trunk)

		# Second conv layer.
		self.conv_yhat_block2 = nn.Sequential(
			nn.Conv2d(
				64, 64, 
				kernel_size=3, stride=1, padding=1, bias=False
			),
			nn.BatchNorm2d(64),
		)

		self.fusion_mode = fusion_mode
		## add attention layers
		if self.fusion_mode in ["channel_att", "spatial_att", "cbam_att", "conv"]:
			self.conv_img_block1 = nn.Sequential(
				nn.Conv2d(
					img_channels,
					64, 
					kernel_size=9,
					stride=1,
					padding=4
				),
				nn.PReLU(),
			)
			self.img_trunk = nn.Sequential(*trunk)
			self.conv_img_block2 = nn.Sequential(
				nn.Conv2d(
					64, 64, 
					kernel_size=3, stride=1, padding=1, bias=False
				),
				nn.BatchNorm2d(64),
			)
			if self.fusion_mode == "channel_att":
				self.channel_att = ChannelAttention(
					in_planes=64
				)
			elif self.fusion_mode == "spatial_att":
				self.spatial_att = SpatialAttention()
			elif self.fusion_mode == "cbam_att":
				self.channel_att = ChannelAttention(
					in_planes=64
				)
				self.spatial_att = SpatialAttention()
			elif self.fusion_mode == "conv":
				self.conv_agg =  nn.Sequential(
					nn.Conv2d(
						128, 64, 
						kernel_size=9, stride=1, padding=4
					),
					nn.BatchNorm2d(64)
				)
		# else:
		# 	raise(f"Do not implement aggregation mode: {fusion_mode}")

		# Output layer.
		self.conv_block3_mu = nn.Conv2d(
			64, out_channels=out_channels, 
			kernel_size=9, stride=1, padding=4
			# kernel_size=3, padding=1
		)

		## one_over_sigma
		self.conv_gauss = nn.Sequential(
			nn.Conv2d(
				64, 64, 
				kernel_size=9, stride=1, padding=4
			),
			nn.PReLU(),
			nn.Conv2d(
				64, 64, 
				kernel_size=9, stride=1, padding=4
			),
			nn.PReLU(),
			nn.Conv2d(
				64, 1, 
				kernel_size=9, stride=1, padding=4
			),
			nn.ReLU(),
		)
		
		self._initialize_weights()

	def forward(
			self,
			yhat: Tensor,
			img: Tensor
	) -> Tensor:
		return self._forward_impl(yhat, img)

	# Support torch.script function.
	def _forward_impl(
			self,
			yhat: Tensor,
			img: Tensor
	) -> Tensor:
		"""
		out1 = self.conv_block1(x)
		out = self.trunk(out1)
		out2 = self.conv_block2(out)
		out = out1 + out2
		mu = self.conv_block3_mu(out)
		gauss = self.conv_gauss(out)
		"""
		
		if self.fusion_mode in ["channel_att", "spatial_att", "cbam_att", "conv"]:
			yhat_feat = self.conv_yhat_block1(yhat)
			yhat_feat_trunk = self.conv_yhat_block2(self.yhat_trunk(yhat_feat))
			
			img_feat = self.conv_img_block1(img)
			img_feat_trunk = self.conv_img_block2(self.img_trunk(img_feat))

			## exp1: sum
			# out = yhat_feat + yhat_feat_trunk

			agg_yhat = yhat_feat + yhat_feat_trunk
			agg_img = img_feat + img_feat_trunk

			if self.fusion_mode == "channel_att":
				## exp 5: channel attention
				channel_att_map = self.channel_att(agg_img)
				out = agg_yhat * channel_att_map
			elif self.fusion_mode == "spatial_att":
				# exp 6: spatial attention
				spatial_att_map = self.spatial_att(agg_img)
				out = agg_yhat * spatial_att_map
			elif self.fusion_mode == "cbam_att":
				## exp 7: cbam
				agg_img = self.channel_att(agg_img) *  agg_img
				cbam_att_map = self.spatial_att(agg_img) *  agg_img
				out = agg_yhat * cbam_att_map
			elif self.fusion_mode == "conv":
				## exp2: learn by conv
				out = self.conv_agg(
					torch.cat((agg_yhat, agg_img), dim=1)
				)
			# from matplotlib import pyplot as plt
			# plt.imshow(spatial_att_map[0][0].detach().numpy())
			# plt.savefig(f'../outputs/spatial_att_map.png')
			# out = spatial_att_map * agg_yhat
		elif self.fusion_mode == "concat":
			concat_input = torch.concat(
				(yhat, img),
				dim=1
			)
			# print(f"concat_input: {concat_input.shape, yhat.min(), yhat.max(), img.min(), img.max()}")
			yhat_feat = self.conv_yhat_block1(concat_input)
			yhat_feat_trunk = self.conv_yhat_block2(self.yhat_trunk(yhat_feat))
			out = yhat_feat + yhat_feat_trunk
		else:
			raise(f"Do not implement aggregation mode: {self.fusion_mode}")
		
		mu = self.conv_block3_mu(out)
		gauss = self.conv_gauss(out)
		return mu, gauss

	def _initialize_weights(self) -> None:
		for module in self.modules():
			if isinstance(module, nn.Conv2d):
				nn.init.kaiming_normal_(module.weight)
				if module.bias is not None:
					nn.init.constant_(module.bias, 0)
			elif isinstance(module, nn.BatchNorm2d):
				nn.init.constant_(module.weight, 1)

if __name__ == "__main__":
	yhat = torch.randn(1,3,256,256)
	img = torch.randn(1,3,256,256)

	model = GaussCapContext(
		yhat_channels=3,
		img_channels=1,
		out_channels=3
	)
	out_mu, out_logvar = model(yhat, img)
	print(out_mu.shape)