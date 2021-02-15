import os
import re
import numpy as np
from PIL import Image as PILImage
import torch
import torch.nn.functional as F
from torch.utils import data as data
from torchvision import transforms as transforms
#import matplotlib.pyplot as plt
from skimage.measure import label as method_label
from skimage.measure import regionprops


# dataset for GGO and C segmentation
class CovidCTData(data.Dataset):

    def __init__(self, **kwargs):
        self.mask_type = kwargs['mask_type']
        self.ignore_ = kwargs['ignore_small']
        # ignore small areas?
        if self.ignore_:
           self.area_th = 100
        else:
           self.area_th = 1
        self.stage = kwargs['stage']
        # this returns the path to imgs dir
        self.data = kwargs['data']
        # this returns the path to
        self.gt = kwargs['gt']
        # IMPORTANT: the order of images and masks must be the same
        self.content_data = os.listdir(self.data)
        self.content_gt = os.listdir(self.gt)
        self.fname = None
        self.img_fname = None

    # this method normalizes the image and converts it to Pytorch tensor
    # Here we use pytorch transforms functionality, and Compose them together,
    def transform_img(self, img):
        # Faster R-CNN does the normalization
        t_ = transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.Resize(img_size),
            transforms.ToTensor(),
        ])
        img = t_(img)
        return img

    # inputs: box coords (min_row, min_col, max_row, max_col)
    # array HxW from whic to extract a single object's mask
    # each isolated mask should have a different label, lab>0
    # masks are binary uint8 type
    def extract_single_mask(self, mask, lab):
        _mask = np.zeros(mask.shape, dtype=np.uint8)
        area = mask == lab
        _mask[area] = 1
        return _mask

    def load_img(self, idx):
        im = PILImage.open(os.path.join(self.data, self.content_data[idx]))
        self.img_fname = self.content_data[idx]
        im = self.transform_img(im)
        return im

    def load_labels_covid_ctscan_data(self, idx):
     list_of_bboxes = []
     labels = []
     list_of_masks = []
     # load bbox
     if self.img_fname.split('.')[0]+'.png' in os.listdir(self.gt):
        mask_name = os.path.join(self.gt, self.img_fname.split('.')[0]+'.png')
        self.fname=mask_name
        # extract bboxes from the mask
        mask = np.array(PILImage.open(mask_name))
        # only GGO: merge C and background
        # or merge GGO and C into a single mask
        # or keep separate masks
        if self.mask_type == "merge":
           mask[mask==3] = 2
           start_int = 2
        else:
           pass
        # add 1 if no lung, otherwise lesion masks are not detected
        if 1 not in np.unique(mask):
           mask[0,0]=1 
        # array  (NUM_CLASS_IN_IMNG, H,W) without bgr+lungs class (merge Class 0 and 1)
        # THIS IS IMPORTANT! CAN TRIGGER CUDA ERROR
        mask_classes = mask == np.unique(mask)[:, None, None][1:]
        # print(mask_classes)
        # extract bounding boxes and masks for each object
        for _idx, m in enumerate(mask_classes):
            lab_mask = method_label(m)
            regions = regionprops(lab_mask)
            for _i, r in enumerate(regions):
                # get rid of really small ones:
                if r.area > self.area_th:
                    # x_min, y_min, x_max, y_max
                    box_coords = (r.bbox[1], r.bbox[0], r.bbox[3], r.bbox[2])
                    list_of_bboxes.append(box_coords)
                    labels.append(_idx + 1)
                    # create a mask for one object, append to the list of masks
                    mask_obj = self.extract_single_mask(lab_mask, r.label)
                    list_of_masks.append(mask_obj)
     # create labels for Mask R-CNN
     # DO NOT CHANGE THESE DATATYPES!
     else:
        pass
     lab = self.return_lab(list_of_bboxes, labels, list_of_masks)
     return lab

    # add the image class
    def return_lab(self, lb, ll,lm):
        lab = {}
        list_of_bboxes = torch.as_tensor(lb, dtype=torch.float)
        labels = torch.tensor(ll, dtype=torch.int64)
        masks = torch.tensor(lm, dtype=torch.uint8)
        lab['labels'] = labels
        lab['boxes'] = list_of_bboxes
        lab['masks'] = masks
        lab['fname'] = self.fname
        lab['image_name'] = self.img_fname
        return lab

    # 'magic' method: size of the dataset
    def __len__(self):
        return len(os.listdir(self.data))

    # return one datapoint
    def __getitem__(self, idx):
        X = self.load_img(idx)
        y = self.load_labels_covid_ctscan_data(idx)
        #print(self.fname, self.img_fname, y)
        return X, y
