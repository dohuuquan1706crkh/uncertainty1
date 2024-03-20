import torch
from segmentation_models_pytorch.base import initialization as init

class UncSegmentationModel(torch.nn.Module):
    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        ## uncertainty head is here
        init.initialize_head(self.uncertainty_head)

        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def check_input_shape(self, x):

        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
            new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    def forward(self, yhat, img=None):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(yhat)
        if self.fusion_mode in ["channel_att", "spatial_att", "cbam_att"]:
            yhat_feat = self.encoder(yhat)
            img_feat = self.img_encoder(img)
            if self.fusion_mode == "channel_att":
                ## exp 5: channel attention
                channel_att_map = self.channel_att(img_feat[-1])
                yhat_feat[-1] = yhat_feat[-1] * channel_att_map
                coded = yhat_feat
            elif self.fusion_mode == "spatial_att":
				# exp 6: spatial attention
                spatial_att_map = self.spatial_att(img_feat[0])
                yhat_feat[0] = yhat_feat[0] * spatial_att_map
                coded = yhat_feat
            elif self.fusion_mode == "cbam_att":
				## exp 7: cbam
                img_feat[0] = self.channel_att(img_feat[0]) *  img_feat[0]
                cbam_att_map = self.spatial_att(img_feat[0]) *  img_feat[0]
                yhat_feat[0] = yhat_feat[0] * cbam_att_map
                coded = yhat_feat
            # elif self.fusion_mode == "conv":
			# 	## exp2: learn by conv
            #     coded = []
            #     for si in len(yhat_feat):
            #         coded.append(
            #             self.conv_agg(
            #                 torch.cat((yhat_feat[si], img_feat[si]), dim=1)
            #             )
            #         )
        elif self.fusion_mode == "entrance":
            ## inspire topology-aware
            x = torch.concat(
                (yhat, img),
                dim=1
            )
            coded = self.encoder(x)
        else:
            coded = self.encoder(yhat)

        # features = self.encoder(yhat)
        decoder_output = self.decoder(*coded)

        masks = self.segmentation_head(decoder_output)
        ## output uncertainty
        logvars = self.uncertainty_head(decoder_output)

        # if self.classification_head is not None:
        #     labels = self.classification_head(features[-1])
        #     return masks, labels

        return masks, logvars

    @torch.no_grad()
    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        x = self.forward(x)

        return x