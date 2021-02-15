# Alex Ter-Sarkisov@City, University of London
# Dec 2020: Merry Christmas&A Happy New Year!!!!
# use this method as an extension to Mask R-CNN

import torch
import torch.nn as nn
import torch.nn.functional as F
from .faster_rcnn import TwoMLPHead, FastRCNNPredictor
from .mask_rcnn import MaskRCNN, MaskRCNNHeads, MaskRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign
from .backbone_utils_adjusted import resnet_fpn_backbone
from collections import OrderedDict
from torchvision.ops import misc as misc_nn_ops
from torch.nn.modules.loss import _Loss
from collections import OrderedDict

__all__ = [
    "get_attn_model"
]

# Main Attention class
class AttentionModel(MaskRCNN):
    def __init__(self, backbone, num_classes=2, 
                 # Faster and Mask R-CNN
                 min_size=512, max_size=512,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=400, rpn_pre_nms_top_n_test=400,
                 rpn_post_nms_top_n_train=200, rpn_post_nms_top_n_test=200,
                 rpn_nms_thresh=0.75,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.75,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.75, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=256, box_positive_fraction=0.75,
                 bbox_reg_weights=None,
                 # Mask parameters
                 mask_roi_pool=None, mask_head=None, mask_predictor=None,
                 # Alex - SSM
                 box_score_thresh_classifier=-0.01, box_nms_thresh_classifier=0.25, box_detections_per_img_s2new=8,
                 # Alex - Mask+Box Features extractor,
                 box_pool_s2=None, box_head_s2=None, box_predictor_s2=None,
                 mask_pool_s2=None, mask_head_s2=None, mask_predictor_s2=None,
                 # Alex - Attention model
                 x_stages=3, num_classes_img=3, sieve_layer=None, s2classifier=None,
                 img_classification_pos = None, img_classification_neg=None, lstm_feature_size = None, device=None, **kwargs):

        out_channels = backbone.out_channels
        # Mask features branch

        # Classification branch
        if box_pool_s2 is None:
            box_pool_s2 = MultiScaleRoIAlign(
                # single feature map
                featmap_names=['0'],
                output_size=7,
                sampling_ratio=2)

        if box_head_s2 is None:
            resolution = box_pool_s2.output_size[0]
            representation_size = 128
            box_head_s2 = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size)

        if box_predictor_s2 is None:
            representation_size = 128
            box_predictor_s2 = FastRCNNPredictor(
                representation_size,
                num_classes)

        if mask_pool_s2 is None:
            mask_pool_s2 = MultiScaleRoIAlign(
                #Alex: the key of the feature map
                featmap_names=['0'],
                output_size=14,
                sampling_ratio=2)

        if mask_head_s2 is None:
            mask_layers = (out_channels,)
            mask_dilation = 1
            mask_head_s2 = MaskRCNNHeads(out_channels, mask_layers, mask_dilation)

        # add mask predictor: upsample+bn+relu
        if mask_predictor_s2 is None:
            in_channels = mask_head_s2[-2].out_channels
            out_channels = in_channels
            mask_predictor_s2 = MaskRCNNPredictorTruncated(in_channels, out_channels, mask_dilation)

        # Attention layer,
        num_feature_maps = mask_predictor_s2.conv_reduce.out_channels
        num_reduce_feature_maps = int(num_feature_maps/2)
        if sieve_layer is None:
           sieve_layer = MaskFeaturesSieve(num_feature_maps=num_feature_maps, num_reduce_feature_maps=num_reduce_feature_maps, h=28, w=28, apply_linearity=False, final=False) 
        attn_layer = AttentionLayer(sieve_layer, x_stages=x_stages, num_features=num_feature_maps, lstm_feature_size=lstm_feature_size, device=device)

        # Positive + Negative Image classification branches
        if img_classification_pos is None:
           img_classification_pos = ImageClassificationLayerFromLSTM(input_features_size=lstm_feature_size, num_classes_img=num_classes_img)

        if img_classification_neg is None:
           img_classification_neg = ImageClassificationLayerFromLSTM(input_features_size=lstm_feature_size, num_classes_img=num_classes_img)

        # instantiate Mask R-CNN:
        #  and image classificiaotn module will be passed to the Generalized RCNN
        kwargs.update(attn=attn_layer, pos_classifier=img_classification_pos, neg_classifier=img_classification_neg)

        super(AttentionModel, self).__init__(backbone, num_classes,
                 # transform parameters
                 min_size, max_size,
                 image_mean, image_std,
                 # RPN parameters
                 rpn_anchor_generator, rpn_head,
                 rpn_pre_nms_top_n_train, rpn_pre_nms_top_n_test,
                 rpn_post_nms_top_n_train, rpn_post_nms_top_n_test,
                 rpn_nms_thresh,
                 rpn_fg_iou_thresh, rpn_bg_iou_thresh,
                 rpn_batch_size_per_image, rpn_positive_fraction,
                 # Box parameters
                 box_roi_pool, box_head, box_predictor,
                 box_score_thresh, box_nms_thresh, box_detections_per_img,
                 box_fg_iou_thresh, box_bg_iou_thresh,
                 box_batch_size_per_image, box_positive_fraction,
                 bbox_reg_weights,
                 # Mask parameters
                 mask_roi_pool=None, mask_head=None, mask_predictor=None, **kwargs)
        # Alex - SSM
        #
        self.roi_heads.score_thresh_classifier=box_score_thresh_classifier
        self.roi_heads.nms_thresh_classifier=box_nms_thresh_classifier
        self.roi_heads.detections_per_img_s2new = box_detections_per_img_s2new
        #
        # 
        self.roi_heads.box_pool_s2=box_pool_s2
        self.roi_heads.box_head_s2=box_head_s2
        self.roi_heads.box_predictor_s2=box_predictor_s2
        #
        # Alex - Mask Features extractor,
        self.roi_heads.mask_pool_s2=mask_pool_s2
        self.roi_heads.mask_head_s2=mask_head_s2
        self.roi_heads.mask_predictor_s2=mask_predictor_s2
        #
# Mask Feature Sive - contract and expand feature maps
# batch size (16) is not used here explicitly
class MaskFeaturesSieve(nn.Module):

      def __init__(self, num_feature_maps=128, num_reduce_feature_maps=64, h=28, w=28, out_linear_features = None, final=False, apply_linearity=False):
          super(MaskFeaturesSieve, self).__init__()
          # simple block for 'sieving' the features
          # halve the size
          self.conv_down = nn.Conv2d(in_channels=num_feature_maps, out_channels=num_reduce_feature_maps, kernel_size=(2,2), stride=2, padding=0)
          self.bn = nn.BatchNorm2d(num_features=num_reduce_feature_maps)
          # double the size
          self.conv_up = nn.ConvTranspose2d(in_channels=num_reduce_feature_maps, out_channels=num_feature_maps, kernel_size=(2,2), stride=2, padding=0)
          self.relu = nn.ReLU(inplace=False)
          self.apply_linearity = apply_linearity
          self.final=final
          if self.final and self.apply_linearity:
             self.feature_output = nn.Linear(num_feature_maps*h*w, out_linear_features)

          for name, param in self.named_parameters():
            if "weight" in name and 'bn' not in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
            elif "bias" in name:
                nn.init.constant_(param, 0.01)

      # forward method assumes x is the masks feature map, 256x14x14
      def forward(self, x):
          x = self.conv_down(x)
          x = self.relu(self.bn(x))
          x = self.conv_up(x)
          if self.apply_linearity and final:
              m1, m2, m3 = x.size()[1], x.size()[2], x.size()[3]
              x = x.view(-1, m1*m2*m3)
              x = self.linear_features(x)
          return x

# accepts the Bx256x14x14 mask features
# outputs the vector for attention computation
class AttentionLayer(nn.Module):

      def __init__(self, sieve_layer, x_stages=None, num_features=128, normalize=False, apply_linearity = False, final=False, lstm_feature_size = 256, device = 'cpu', **kwargs):    
          super(AttentionLayer, self).__init__()
          self.device = device
          # Mask sieve
          self.sieve_stages = x_stages
          self.sieve = []
          for l in range(self.sieve_stages):
                 s = sieve_layer
                 self.sieve.append(s)
          self.sieve = nn.Sequential(*self.sieve)
          # downsize to Cx1x1
          self.conv_d1 = nn.Conv2d(num_features, num_features, kernel_size=(2,2), stride=2, padding=0)
          self.bn1 = nn.BatchNorm2d(num_features=num_features)
          self.conv_d2 = nn.Conv2d(num_features, num_features, kernel_size=(2,2), stride=2, padding=0)
          self.bn2 = nn.BatchNorm2d(num_features=num_features)
          self.conv_d_final = nn.Conv2d(num_features, num_features, kernel_size=(7,7), stride=1, padding=0)
          self.bn3 = nn.BatchNorm2d(num_features=num_features)
          self.normalize=normalize
          self.num_rois = None
          # LSTM
          self.lstm_pos = nn.LSTM(input_size = num_features, hidden_size = lstm_feature_size, batch_first=True)
          self.lstm_neg = nn.LSTM(input_size = num_features, hidden_size = lstm_feature_size, batch_first=True)
          # Attn features
          self.final_linear_pos = nn.Linear(lstm_feature_size, lstm_feature_size)
          self.final_linear_neg = nn.Linear(lstm_feature_size, lstm_feature_size)
          #
          # 
          
          for name, param in self.named_parameters():
            if 'weight'in name and 'conv' in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
            elif "bias" in name:
                nn.init.constant_(param, 0.01)

      def get_attention(self, lstm_output, last_state):
          attn_vector = F.softmax(torch.bmm(lstm_output, last_state.transpose(2,1)), dim=1)
          new_hidden_state = torch.bmm(lstm_output.transpose(2,1), attn_vector)
          return new_hidden_state.squeeze(2)
          

      def forward(self, x):
          for s in range(self.sieve_stages):
              x = self.sieve(x)
          x = F.relu(self.conv_d1(x), inplace=False)
          x = self.bn1(x)
          x = F.relu(self.conv_d2(x), inplace=False)
          x = self.bn2(x)
          x = F.relu(self.conv_d_final(x), inplace=False)
          x = self.bn3(x)
          x=x.view(1, x.size()[0], -1)
          # input must be batch_size, seq_length, features
          relevant_features_lstm, (ht, ct) = self.lstm_pos(x)
          irrelevant_features_lstm,(hti, cti) = self.lstm_neg(x)
          linear_pos = self.final_linear_pos(ht)
          linear_neg = self.final_linear_neg(hti) 
          attn_hidden_state_r=self.get_attention(relevant_features_lstm, linear_pos)
          attn_hidden_state_ir=self.get_attention(irrelevant_features_lstm, linear_neg)
          # 1xlstmfeats
          new_attn_r, new_attn_ir = attn_hidden_state_r+linear_pos, attn_hidden_state_ir+linear_neg
          return new_attn_r, new_attn_ir 

# 02/01
# accepts the positive 
class ImageClassificationLayerFromLSTM(nn.Module):
      def __init__(self, input_features_size=None, linear_features = 128, num_classes_img = None):
          super(ImageClassificationLayerFromLSTM,self).__init__()
          self.fc_l1 = nn.Linear(input_features_size, linear_features)
          self.fc_l2 = nn.Linear(linear_features, linear_features)
          self.class_predict_logits = nn.Linear(linear_features, num_classes_img)

      # return
      def forward(self, x):
          x = x.view(1, -1)
          x = F.relu(self.fc_l1(x), inplace=False)
          x = F.relu(self.fc_l2(x), inplace=False)
          x = self.class_predict_logits(x)
          return x


class MaskRCNNPredictorTruncated(nn.Sequential):
    def __init__(self, in_channels, dim_reduced, num_classes):
        super(MaskRCNNPredictorTruncated, self).__init__(OrderedDict([
            ("conv_mask", misc_nn_ops.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0)),
            ("conv_reduce", nn.Conv2d(dim_reduced, dim_reduced, 3, 1, padding=1)),
            ("relu", nn.ReLU(inplace=False)),]))

        for name, param in self.named_parameters():
            if "weight" in name and 'bn' not in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
            elif "bias" in name:
                nn.init.constant_(param, 0.01)


# out_channels: 256
def get_attn_model(backbone=None, pretrained=False, out_channels=256, **kwargs):
    backbone_model = resnet_fpn_backbone (backbone_name=backbone, pretrained=pretrained, out_ch=out_channels)
    main_model = AttentionModel(backbone_model,**kwargs)
    return main_model


