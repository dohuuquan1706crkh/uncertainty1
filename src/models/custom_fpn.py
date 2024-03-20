from typing import Optional, Union
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.fpn.decoder import FPNDecoder

from models.custom_smp_model import UncSegmentationModel
from models.custom_head import SegmentationHead, ClassificationHead
from models.attention import ChannelAttention, SpatialAttention


class FPN(UncSegmentationModel):
    """FPN_ is a fully convolution neural network for image semantic segmentation.

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_pyramid_channels: A number of convolution filters in Feature Pyramid of FPN_
        decoder_segmentation_channels: A number of convolution filters in segmentation blocks of FPN_
        decoder_merge_policy: Determines how to merge pyramid features inside FPN. Available options are **add**
            and **cat**
        decoder_dropout: Spatial dropout rate in range (0, 1) for feature pyramid in FPN_
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
                **callable** and **None**.
            Default is **None**
        upsampling: Final upsampling factor. Default is 4 to preserve input-output spatial shape identity
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)

    Returns:
        ``torch.nn.Module``: **FPN**

    .. _FPN:
        http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf

    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_pyramid_channels: int = 256,
        decoder_segmentation_channels: int = 128,
        decoder_merge_policy: str = "add",
        decoder_dropout: float = 0.2,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,
        upsampling: int = 4,
        aux_params: Optional[dict] = None,
        fusion_mode=None,
        img_channels=None
    ):
        super().__init__()
        self.img_channels = img_channels
        self.fusion_mode = fusion_mode
        yhat_channels = in_channels
        ## TODO: add context here
        if self.fusion_mode == "entrance":
            in_channels = yhat_channels + img_channels
        elif self.fusion_mode in ["channel_att", "spatial_att", "cbam_att", "conv"]:
            in_channels = yhat_channels
            self.img_encoder = get_encoder(
                encoder_name,
                in_channels=img_channels,
                depth=encoder_depth,
                weights=encoder_weights,
            )
            if self.fusion_mode == "channel_att":
                self.channel_att = ChannelAttention(
                    in_planes=320    ## << check dim
                )
            elif self.fusion_mode == "spatial_att":
                self.spatial_att = SpatialAttention()
            elif self.fusion_mode == "cbam_att":
                self.channel_att = ChannelAttention(
                    in_planes=320    ## << check dim
                )
                self.spatial_att = SpatialAttention()
            # elif self.fusion_mode == "conv":
            #     self.conv_agg =  nn.Sequential(
            #         nn.Conv2d(
            #             32, 16,     ## << check dim
            #             3, stride=1, padding=1
            #         ),
            #         nn.BatchNorm2d(16)
            #     )
        else:
            in_channels = yhat_channels

        # validate input params
        if encoder_name.startswith("mit_b") and encoder_depth != 5:
            raise ValueError("Encoder {} support only encoder_depth=5".format(encoder_name))

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = FPNDecoder(
            encoder_channels=self.encoder.out_channels,
            encoder_depth=encoder_depth,
            pyramid_channels=decoder_pyramid_channels,
            segmentation_channels=decoder_segmentation_channels,
            dropout=decoder_dropout,
            merge_policy=decoder_merge_policy,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )
        
        ## uncertainty head is here
        self.uncertainty_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=1,
            activation=None,
            kernel_size=1,
            upsampling=upsampling,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = "fpn-{}".format(encoder_name)
        self.initialize()