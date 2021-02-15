# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""

Implements the Generalized R-CNN framework

"""

from collections import OrderedDict
import torch
from torch import nn

class GeneralizedRCNNAttention(nn.Module):
    """
    Main class for Generalized R-CNN.

    Arguments:
        backbone (nn.Module):
        rpn (nn.Module):
        heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, rpn, roi_heads, transform, attn=None, pos_classifier=None, neg_classifier=None):
        super(GeneralizedRCNNAttention, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.attn_layer = attn
        self.pos_classifier_attn = pos_classifier
        self.neg_classifier_attn = neg_classifier

    def forward(self, images, targets=None, feature_batch=None):
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        original_image_sizes = [img.shape[-2:] for img in images]
        if self.training:
           # targets for the segmenation and classification models
           if len(targets[0]['boxes']):
               images, targets = self.transform(images, targets=targets)
           # targets for the classification model only
           else:
               images, _ = self.transform(images, targets=None)
        # evaluation mode 
        else:
            images, _ = self.transform(images, targets=None)
        features = self.backbone(images.tensors)
        # Alex: LW Mask R-CNN outputs Cx16x16
        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])
        # without the targets, RPN/RoI losses will be empty
        proposals, proposal_losses = self.rpn(images, features, targets)
        # Alex
        detections, detector_losses, mask_features = self.roi_heads(features, proposals, images.image_sizes, targets)
        # Alex
        if not self.training:
           pos_vector, neg_vector = self.attn_layer(mask_features[0])
           # difference between positive and negative logits - final model output
           pos_img_class_logits = self.pos_classifier_attn(pos_vector)
           neg_img_class_logits = self.neg_classifier_attn(neg_vector)
           final_diff_logits = pos_img_class_logits-neg_img_class_logits
           scores_covid_img = [dict(final_scores=final_diff_logits)]
        else:
           scores_covid_img = None

        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        # 
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        # mask feature vector is added to the batch in the main step
        return losses, scores_covid_img, detections, #mask_feature_vector

