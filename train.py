# COVID Mask R-CNN project
# LSTM+Attention
# Developed by Alex Ter-Sarkisov@City, University of London
# alex.ter-sarkisov@city.ac.uk

import os
import pickle
import re
import sys
import time
from collections import OrderedDict
import cv2
import dataset_classification
import dataset_segmentation
import models.lstm_attn
import numpy as np
import torch
import torch.nn as nn
from torch.nn import NLLLoss as NLLLoss
import torch.nn.functional as F
import torchvision
import utils
from PIL import Image as PILImage
from matplotlib.patches import Rectangle
from models.lstm_attn import attn_model as attn_model
from models.lstm_attn.rpn import AnchorGenerator
from torch.utils import data
from torchvision import transforms

torch.autograd.set_detect_anomaly(True)
device = torch.device('cpu')

torch.manual_seed(time.time())
np.random.seed(int(time.time()))

if torch.cuda.is_available():
    device = torch.device('cuda')

bbn = 'resnext50_32x4d'
data_dir = 'imgs'
gt_dir = 'masks'
# parameters for the dataset
# if only signs are detected, gt_dir is gt1
# for characters gt_dir is gt3
# for masks gt2 and gt3
dataset_covid_pars = {'stage': 'train', 'gt': os.path.join('../covid_data/train', gt_dir),
                      'data': os.path.join('../covid_data/train', data_dir),
                      'mask_type': 'separate', 'ignore_small': True}
datapoint_covid = dataset_segmentation.CovidCTData(**dataset_covid_pars)
#
dataloader_covid_pars = {'shuffle': False, 'batch_size': 1}
dataloader_covid = data.DataLoader(datapoint_covid, **dataloader_covid_pars)
########################
dataset_covid_classification_pars = {'stage': 'train',
                                     'data': '../covid_data/cncb/train_very_large',
                                     'img_size': [512, 512]}
datapoint_classification_covid = dataset_classification.COVID_CT_DATA(**dataset_covid_classification_pars)
#
dataloader_covid_classification_pars = {'shuffle': False, 'batch_size': 1}
dataloader_classification_covid = data.DataLoader(datapoint_classification_covid,
                                                  **dataloader_covid_classification_pars)

# combine train and evaluation
attn_args = {'out_channels':256, 'min_size': 512, 'max_size': 512, 'rpn_batch_size_per_image': 256, 'rpn_positive_fraction': 0.75,
                 'box_positive_fraction': 0.5, 'rpn_pre_nms_top_n_train':400, 'rpn_post_nms_top_n_train':200,
                 'box_fg_iou_thresh': 0.75, 'box_bg_iou_thresh': 0.5, 'num_classes': 4, 'box_batch_size_per_image': 256,
                 'rpn_nms_thresh': 0.75, 'box_nms_thresh': 0.5,
                 'box_nms_thresh_classifier': 0.25, 'box_detections_per_img_s2new': 16, 'box_score_thresh': 0.05, 'box_score_thresh_classifier':-0.01,
                 'box_detections_per_img': 128, 'num_classes_img': 3, 'x_stages':1, 'lstm_feature_size':256, 'device':device}

# many small anchors
anchor_generator = AnchorGenerator(sizes=((2, 4, 8, 16, 32),), aspect_ratios=((0.1, 0.25, 0.5, 1, 1.5, 2),))
print(bbn)
# out_channels:2048
# num_classes: (1+3)
# background + GGO + C + Lungs
total_pars = 0
attn_args['rpn_anchor_generator'] = anchor_generator
attn_classifier = attn_model.get_attn_model(backbone=bbn, pretrained=True, **attn_args)
for _n, _p in attn_classifier.state_dict().items():
    total_pars += _p.numel()
print("Total pars", total_pars)
if device == torch.device('cuda'):
    attn_classifier = attn_classifier.to(device)
optimizer_pars = {'lr': 1e-5, 'weight_decay': 1e-3}
optimizer = torch.optim.Adam(list(attn_classifier.parameters()), **optimizer_pars)
total_epochs = 10
save_every = 10
start_time = time.time()
batch_size = 1
print(attn_classifier)
# batch_loss_function=nn.L1Loss()

nllloss = NLLLoss()
#cm=CM()
# switch_off_grad()
def step(stage, e, max_loss):
    epoch_loss_classification_indep = 0
    epoch_loss_segmentation = 0
    # repeat the total number of minibatches in the classifer dataset
    total_batches = int(len(dataloader_classification_covid) / batch_size)
    for r in range(total_batches):
            total_classifier_loss = 0
            optimizer.zero_grad()
            rand_im_segment_ind = torch.randint(len(dataloader_covid), size=(1,)).item()
            b = dataloader_covid.dataset[rand_im_segment_ind]
            X, y = b
            if device == torch.device('cuda'):
                X, y['labels'], y['boxes'], y['masks'] = X.to(device), y['labels'].to(device), y['boxes'].to(device), y[
                    'masks'].to(device)
            X = utils.normalize_img(X, device=device)
            images = [X]
            targets = []
            lab = {}
            lab['boxes'] = y['boxes']
            lab['labels'] = y['labels']
            lab['masks'] = y['masks']
            # train segmentation and classification or only classification?
            targets.append(lab)
            if len(targets[0]['boxes']):
                # pass
                attn_classifier.train()
                attn_classifier.attn_layer.eval()
                loss, _, _ = attn_classifier(images, targets)
                total_seg_loss = 0
                for k in loss.keys():
                    total_seg_loss += loss[k]
                # exclude all classification box and mask layers in RoI
                for _n, _p in attn_classifier.named_parameters():
                    if 's2' in _n or 'attn' in _n:
                        _p.grad = None
                # only if better
                epoch_loss_segmentation += total_seg_loss.item()
                if total_seg_loss.item() < max_loss:
                    utils.copy_weights(attn_classifier)
                total_seg_loss.backward()
                max_loss = total_seg_loss.item()
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
            else:
                total_seg_loss = 0
            # set all to eval, then switch back parameters upgrade
            # compute the image loss
            rand_im_class_ind = torch.randint(len(dataloader_classification_covid), size=(1,)).item()
            bt = dataloader_classification_covid.dataset[rand_im_class_ind]
            attn_classifier.eval()
            attn_classifier.backbone.train()
            attn_classifier.attn_layer.train()
            X_cl, y_cl = bt
            # print('targ', y_cl)
            if device == torch.device('cuda'):
                X_cl, y_cl = X_cl.to(device), y_cl.to(device)
            X_cl = X_cl.squeeze(0)
            X_cl = utils.normalize_img(X_cl, device=device)
            image = [X_cl]  # remove the batch dimension
            lab = dict(
                image_label=y_cl.squeeze(0))  # This is not used within thge model, keep it for the correct operations
            _, predict_score, _ = attn_classifier(image, targets=lab)
            # append the feature vector and its class
            correct_class_str = lab['image_label']
            #print('pred', predict_score)
            classifier_loss = F.binary_cross_entropy_with_logits(predict_score[0]['final_scores'].squeeze(0), y_cl)
            total_classifier_loss += classifier_loss
            total_classifier_loss.backward()
            for _n, _p in attn_classifier.named_parameters():
                if 'backbone' not in _n and 'attn' not in _n:
                   _p.grad = None
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            epoch_loss_classification_indep += total_classifier_loss
    # save model?
    if not e % save_every:
        attn_classifier.eval()
        state = {'epoch': str(e), 'model_weights': attn_classifier.state_dict(),
                 'optimizer_state': optimizer.state_dict(), 'lrate': 1e-5}
        torch.save(state, os.path.join("saved_models", bbn + "_16_4class_lstm_attn_ckpt_" + str(e) + ".pth"))
        attn_classifier.train()
    epoch_loss_classification_indep = epoch_loss_classification_indep / total_batches
    epoch_loss_segmentation = epoch_loss_segmentation / len(dataloader_covid)
    # print(epoch_loss)
    # print('total!', tots_pos, tots_neg)
    return epoch_loss_classification_indep, epoch_loss_segmentation, max_loss

max_loss = 1000
for e in range(total_epochs):
    train_class_indep_loss, epoch_loss_segmentation, max_loss = step("train", e, max_loss)
    print("Epoch {0:d}: train loss single classification = {1:.3f}, loss_segmentation = {2:.3f}, max_loss = {3:.3f},".format(
            e, train_class_indep_loss, epoch_loss_segmentation, max_loss))

end_time = time.time()
print("Training took {0:.1f} seconds".format(end_time - start_time))
# inference + save the model
attn_classifier.eval()
attn_classifier = attn_classifier.to(torch.device('cpu'))
state = {'epoch': str(total_epochs), 'model_weights': attn_classifier.state_dict(), 'optimizer_state': optimizer.state_dict(),
         'lrate': 1e-5}
torch.save(state, os.path.join("saved_models", bbn + "_16_4class_lstm_attn_ckpt_" + str(total_epochs) + ".pth"))
