# COVID Mask R-CNN project
# COVID Mask R-CNN project
# LSTM+Attention
# Developed by Alex Ter-Sarkisov@City, University of London
# alex.ter-sarkisov@city.ac.uk

import os
import re
import time
import cv2
import dataset_classification
#import matplotlib.patches as patches
#import matplotlib.pyplot as plt
import models.lstm_attn
import numpy as np
import torch
import torch.nn as nn
from torch.nn import NLLLoss as NLLLoss
import torch.nn.functional as F
#import torch.utils as utils
import torchvision
import utils
from PIL import Image as PILImage
from models.lstm_attn import attn_model as attention_model
from models.lstm_attn.rpn import AnchorGenerator
from torch.utils import data
#from torchvision import transforms

#torch.autograd.set_detect_anomaly(True)
device = torch.device('cpu')

#torch.manual_seed(time.time())
#np.random.seed(int(time.time()))

if torch.cuda.is_available():
    device = torch.device('cuda')

bbn = 'resnext50_32x4d'
data_dir = 'imgs'
gt_dir = 'masks'
########################
dataset_covid_classification_pars = {'stage': 'eval',
                                     'data':
                                     '../covid_data/cncb/ncov-ai.big.ac.cn/download/test',
                                     'img_size': [512, 512]}
datapoint_classification_covid = dataset_classification.COVID_CT_DATA(**dataset_covid_classification_pars)
#
dataloader_covid_classification_pars = {'shuffle': True, 'batch_size': 1}
dataloader_classification_covid = data.DataLoader(datapoint_classification_covid,
                                                  **dataloader_covid_classification_pars)

# combine train and evaluation
attn_args = {'out_channels':256, 'min_size': 512, 'max_size': 512, 'rpn_batch_size_per_image': 256, 'rpn_positive_fraction': 0.75,
                 'box_positive_fraction': 0.5,'rpn_pre_nms_top_n_test':400, 'rpn_post_nms_top_n_test':200,
                 'box_fg_iou_thresh': 0.75, 'box_bg_iou_thresh': 0.5, 'num_classes': 4, 'box_batch_size_per_image': 256,
                 'rpn_nms_thresh': 0.75, 'box_nms_thresh': 0.5,
                 'box_nms_thresh_classifier': 0.25, 'box_detections_per_img_s2new': 16, 'box_score_thresh': 0.75, 'box_score_thresh_classifier':-0.01,
                 'box_detections_per_img': 4, 'num_classes_img': 3, 'x_stages':1, 'lstm_feature_size':256, 'device':device}

# many small anchors
# for the large model add sizes=64,128
anchor_generator = AnchorGenerator(sizes=((2, 4, 8, 16, 32),), aspect_ratios=((0.1, 0.25, 0.5, 1, 1.5, 2),))
print(bbn)
# out_channels:2048
# num_classes: (1+4)
# background + GGO + C + Lungs
total_pars = 0
attn_args['rpn_anchor_generator'] = anchor_generator
weights = bbn+'_16_4class_lstm_attn_ckpt_10.pth'
print(weights)
covid_detector_weights = torch.load(os.path.join("saved_models", weights), map_location="cpu")
attn_classifier = attention_model.get_attn_model(backbone=bbn, pretrained=False, **attn_args)
print(attn_classifier)
# set backbone to train mode
attn_classifier.load_state_dict(covid_detector_weights['model_weights'])
attn_classifier.eval()
attn_classifier.backbone.train()
attn_classifier.attn_layer.train()

if device == torch.device('cuda'):
    attn_classifier = attn_classifier.to(device)
start = time.time()
batch_size = 1
confusion_matrix = torch.zeros(3,3, dtype=torch.int32).to(device)

for id, b in enumerate(dataloader_classification_covid):
    X,y = b
    #print(X)
    if device==torch.device('cuda'):
        X, y = X.to(device), y.to(device)
    image = [X.squeeze_(0)] #remove the batch dimension
    utils.normalize_img(image[0], device=device)
    _, predict_score,_ = attn_classifier(image, targets=None)    
    pred_class = predict_score[0]['final_scores'].argmax()
    confusion_matrix[torch.nonzero(y.view(-1)>0).item(), pred_class] += 1

end=time.time()
total=end-start
print(total)

cm = confusion_matrix.float()
print(cm, cm.sum())
print(cm.diagonal().sum().div(cm.sum()).item())
cm[0,:].div_(cm[0,:].sum())
cm[1,:].div_(cm[1,:].sum())
cm[2,:].div_(cm[2,:].sum())
print(cm)
cm_spec = confusion_matrix.float()

cm_spec[:,0].div_(cm_spec[:,0].sum())
cm_spec[:,1].div_(cm_spec[:,1].sum())
cm_spec[:,2].div_(cm_spec[:,2].sum())

cw=torch.tensor([0.45, 0.35, 0.2], dtype=torch.float).to(device)

f1_score = 2*cm.diag().mul(cm_spec.diag()).div(cm.diag()+cm_spec.diag()).dot(cw).item()
print("F1 score = {:1.4f}".format(f1_score))


print(cm)
print(cm_spec)
print('success')


