import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor
from torch.distributions.normal import Normal
# import sys
# sys.path.append(".")
from models.attention import ChannelAttention, SpatialAttention

"""
https://pyimagesearch.com/2023/10/02/a-deep-dive-into-variational-autoencoders-with-pytorch/
"""
class Sampling(nn.Module):
    def forward(self, z_mean, z_log_var):
        # get the shape of the tensor for the mean and log variance
        batch, dim = z_mean.shape
        # generate a normal random tensor (epsilon) with the same shape as z_mean
        # this tensor will be used for reparameterization trick
        epsilon = Normal(0, 1).sample((batch, dim)).to(z_mean.device)
        # apply the reparameterization trick to generate the samples in the
        # latent space
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

# define the encoder

class Encoder(nn.Module):
	def __init__(
			self,
			image_size:int=256,
			input_channels:int=3,
			embedding_dim:int=1024
	):
		super(Encoder, self).__init__()
		# define the convolutional layers for downsampling and feature
		# extraction
		self.feature_extractor = nn.Sequential(
			nn.Conv2d(input_channels, 16, 3, stride=2, padding=1),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.Conv2d(16, 32, 3, stride=2, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(32, 64, 3, stride=2, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 64, 3, stride=2, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
		)
		
		# define a flatten layer to flatten the tensor before feeding it into
		# the fully connected layer
		self.flatten = nn.Flatten()
		# define fully connected layers to transform the tensor into the desired
		# embedding dimensions
		## 8 = 2**3 (3 conv layers)
		n = 4
		self.fc_mean = nn.Linear(
			64 * (image_size // 2**n) * (image_size // 2**n), embedding_dim
		)
		# self.fc_log_var = nn.Linear(
		# 	128 * (image_size // 8) * (image_size // 8), embedding_dim
		# )
		# initialize the sampling layer
		# self.sampling = Sampling()
		self._initialize_weights()
	def forward(self, x):
		# apply convolutional layers with relu activation function
		x = self.feature_extractor(x)
		# flatten the tensor
		x = self.flatten(x)
		# get the mean and log variance of the latent space distribution
		z = self.fc_mean(x)
		# z_log_var = self.fc_log_var(x)
		# sample a latent vector using the reparameterization trick
		# z = self.sampling(z_mean, z_log_var)
		# z_log_var = z_log_var.clamp(min=-40, max=40)
		# return z_mean, z_log_var, z
		return z
	
	def _initialize_weights(self) -> None:
		for module in self.modules():
			if isinstance(module, nn.Conv2d):
				nn.init.kaiming_normal_(module.weight)
				if module.bias is not None:
					nn.init.constant_(module.bias, 0)
			elif isinstance(module, nn.BatchNorm2d):
				nn.init.constant_(module.weight, 1)
# define the decoder
class Decoder(nn.Module):
	def __init__(
			self,
			out_channels:int=3,
			embedding_dim:int=1024,
			shape_before_flattening:list=[128,40,40]
	):
		super(Decoder, self).__init__()
		# define a fully connected layer to transform the latent vector back to
		# the shape before flattening
		self.fc = nn.Linear(
			embedding_dim,
			shape_before_flattening[0]
			* shape_before_flattening[1]
			* shape_before_flattening[2],
		)
		# define a reshape function to reshape the tensor back to its original
		# shape
		self.reshape = lambda x: x.view(-1, *shape_before_flattening)
		# define the transposed convolutional layers for the decoder to upsample
		# and generate the reconstructed image
		self.reconstructor = nn.Sequential(
			 nn.ConvTranspose2d(
				shape_before_flattening[0], 64, 3, stride=2, padding=1, output_padding=1
			),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.ConvTranspose2d(
				64, 32, 3, stride=2, padding=1, output_padding=1
			),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.ConvTranspose2d(
				32, 16, 3, stride=2, padding=1, output_padding=1
			),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.ConvTranspose2d(
				16, 16, 3, stride=2, padding=1, output_padding=1
			),
			nn.BatchNorm2d(16),
			nn.ReLU()
		)
		
		self.conv_mu = nn.Conv2d(
			16, out_channels=out_channels, 
			kernel_size=9, stride=1, padding=4
		)
		# self.conv_logvar = nn.Conv2d(
		# 	16, out_channels=1, 
		# 	kernel_size=3, padding=1
		# )
				## one_over_sigma
		self.conv_gauss = nn.Sequential(
			nn.Conv2d(
				16, 16, 
				kernel_size=9, stride=1, padding=4
			),
			nn.PReLU(),
			# nn.Conv2d(
			# 	16, 16, 
			# 	kernel_size=9, stride=1, padding=4
			# ),
			# nn.PReLU(),
			nn.Conv2d(
				16, 1, 
				kernel_size=9, stride=1, padding=4
			),
			nn.ReLU(),
		)
		
		self._initialize_weights()
		# for m in self.conv_logvar.modules():
		# 	if isinstance(m, nn.Conv2d):
		# 		nn.init.normal_(m.weight, mean=0.0, std=1e-4)
		# 		nn.init.constant_(m.bias, 0)
		
	def forward(self, x):
		# pass the latent vector through the fully connected layer
		x = self.fc(x)
		# reshape the tensor
		x = self.reshape(x)
		# apply transposed convolutional layers with relu activation function
		x = self.reconstructor(x)
		x_mu = self.conv_mu(x)
		x_logvar = self.conv_gauss(x)
		return x_mu, x_logvar
	
	def _initialize_weights(self) -> None:
		for module in self.modules():
			if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
				nn.init.kaiming_normal_(module.weight)
				if module.bias is not None:
					nn.init.constant_(module.bias, 0)
			elif isinstance(module, nn.BatchNorm2d):
				nn.init.constant_(module.weight, 1)

class VAE(nn.Module):
	def __init__(
			self,
			image_size:int=256,
			input_channels:int=3,
			out_channels:int=3,
			embedding_dim:int=1024,
			shape_before_flattening:list=[128,256//8,256//8],	## 64 * (image_size // 2**n) * (image_size // 2**n)
			use_context:bool=False
	):
		super().__init__()
		self.use_context = use_context
		self.encoder = Encoder(
			image_size=image_size,
			input_channels=input_channels,
			embedding_dim=embedding_dim
		)
		if self.use_context:
			self.context_encoder = Encoder(
				image_size=image_size,
				input_channels=3,	## RGB
				embedding_dim=embedding_dim
			)
		self.decoder = Decoder(
			out_channels=out_channels,
			embedding_dim=embedding_dim,
			shape_before_flattening=shape_before_flattening
		)
		self.agg = nn.Linear(
			in_features=2*embedding_dim,
			out_features=embedding_dim
		)

	def forward(self, yhat, x=None):
		z = self.encoder(yhat)
		# if self.use_context:
		z_context = self.context_encoder(x)
		## exp 1: weight sum
		# z += z_context
		## exp 2: learn
		z_agg = self.agg(
			torch.cat((z, z_context), dim=1)
		)
		out_mu, out_logvar = self.decoder(z_agg)
		return out_mu, out_logvar, z, z_context

# Define Convolutional AutoEncoder Network
class ConvAutoencoder(torch.nn.Module):
	def __init__(self, 
			yhat_channels:int=2,
			out_channels:int=2,
			img_channels=None,
			fusion_mode=None,
		):
		super(ConvAutoencoder, self).__init__()

		self.img_channels = img_channels
		self.fusion_mode = fusion_mode

		if self.fusion_mode == "entrance":
			in_channels = yhat_channels + img_channels
		elif self.fusion_mode in ["channel_att", "spatial_att", "cbam_att", "conv"]:
			in_channels = yhat_channels
			self.img_encoder = torch.nn.Sequential(
				torch.nn.Conv2d(img_channels, 64, 3, stride=1, padding=1),  # 
				torch.nn.ReLU(True),
				torch.nn.MaxPool2d(2, stride=1),
				torch.nn.Conv2d(64, 16, 3, stride=1, padding=1),  # b, 8, 3, 3
				torch.nn.ReLU(True),
				torch.nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
			)
			if self.fusion_mode == "channel_att":
				self.channel_att = ChannelAttention(
					in_planes=16
				)
			elif self.fusion_mode == "spatial_att":
				self.spatial_att = SpatialAttention()
			elif self.fusion_mode == "cbam_att":
				self.channel_att = ChannelAttention(
					in_planes=16
				)
				self.spatial_att = SpatialAttention()
			elif self.fusion_mode == "conv":
				self.conv_agg =  nn.Sequential(
					nn.Conv2d(
						32, 16, 
						3, stride=1, padding=1
					),
					nn.BatchNorm2d(16)
				)
		else:
			in_channels = yhat_channels
		
		self.encoder = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels, 64, 3, stride=1, padding=1),  # 
			torch.nn.ReLU(True),
			torch.nn.MaxPool2d(2, stride=1),
			torch.nn.Conv2d(64, 16, 3, stride=1, padding=1),  # b, 8, 3, 3
			torch.nn.ReLU(True),
			torch.nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
		)
		
		self.decoder = torch.nn.Sequential(
			torch.nn.Upsample(scale_factor=1, mode='nearest'),
			torch.nn.Conv2d(16, 64, 3, stride=1, padding=1),  # b, 16, 10, 10
			torch.nn.ReLU(True),
			torch.nn.Upsample(scale_factor=1, mode='nearest'),
			# torch.nn.Conv2d(64, 64, 3, stride=1, padding=2),  # b, 8, 3, 3
		)
		self.output_mu = torch.nn.Conv2d(64, out_channels, 3, stride=1, padding=2)  # b, 8, 3, 3
		self.output_var = nn.Sequential(
			torch.nn.Conv2d(64, 1, 3, stride=1, padding=2),  # b, 8, 3, 3
			nn.ReLU()
		)
		self.norm = nn.BatchNorm2d(16)

	def forward(self, yhat, img=None):
		if self.fusion_mode in ["channel_att", "spatial_att", "cbam_att", "conv"]:
			yhat_feat = self.encoder(yhat)
			img_feat = self.img_encoder(img)
			if self.fusion_mode == "channel_att":
				## exp 5: channel attention
				channel_att_map = self.channel_att(img_feat)
				coded = yhat_feat * channel_att_map
			elif self.fusion_mode == "spatial_att":
				# exp 6: spatial attention
				spatial_att_map = self.spatial_att(img_feat)
				coded = yhat_feat * spatial_att_map
			elif self.fusion_mode == "cbam_att":
				## exp 7: cbam
				img_feat = self.channel_att(img_feat) *  img_feat
				cbam_att_map = self.spatial_att(img_feat) *  img_feat
				coded = yhat_feat * cbam_att_map
			elif self.fusion_mode == "conv":
				## exp2: learn by conv
				coded = self.conv_agg(
					torch.cat((yhat_feat, img_feat), dim=1)
				)
		elif self.fusion_mode == "entrance":
			## inspire topology-aware
			x = torch.concat(
				(yhat, img),
				dim=1
			)
			coded = self.encoder(x)
			# else:
			# 	raise ValueError("Check the fusion mode: {}".format(self.fusion_mode))
		else:
			coded = self.encoder(yhat)		
		coded = self.norm(coded)
		decoded = self.decoder(coded)
		mu = self.output_mu(decoded)
		var = self.output_var(decoded)
		return mu, var

if __name__ == "__main__":
	x = torch.randn(1,3,256,256)
	img = torch.randn(1,1,256,256)

	model = ConvAutoencoder(
		img_channels=1,
		yhat_channels=3,
		out_channels=3,
		fusion_mode="abc"
	)
	out_mu, out_logvar = model(x, img)
	print(out_mu.shape, out_logvar.shape)