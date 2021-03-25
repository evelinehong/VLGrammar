"""
This code is based on the Torchvision repository, which was licensed under the BSD 3-Clause.
"""
import os
import pickle
import sys
import json
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from utils.mypath import MyPath
import json
from collections import Counter
import random
import copy
from torchvision import transforms

class PARTNET(Dataset):
    def __init__(self, root=MyPath.db_root_dir('partnet', ''), split='train', type='chair', transform=None):

        super(PARTNET, self).__init__()
        self.root = MyPath.db_root_dir('partnet', type)
        self.img_dir = self.root + split
        self.transform = transform
        self.images = []
        self.targets = []
        # split = 'train'
        self.split = split
        #self.part_order = {'bag_body': 0, 'handle': 1, 'shoulder_strap': 2}
        #self.classes = ['bag_body', 'handle', 'shoulder_strap']
        #self.part_order = {'headboard': 0, 'bed_sleep_area': 1, 'bed_frame_horizontal_surface':2, 'bed_side_surface_panel': 3, 'bed_post': 4, 'leg': 5, 'surface_base': 6, 'ladder':7}
        #self.classes = ['headboard', 'bed_sleep_area', 'bed_frame_horizontal_surface', 'bed_side_surface_panel', 'bed_post', 'leg', 'surface_base', 'ladder']
        #self.part_order = {'tabletop': 0, 'drawer': 1, 'cabinet_door': 2, 'side_panel': 3, 'bottom_panel': 4, 'central_support': 5, 'leg': 6, 'shelf': 7, 'leg_bar': 8, 'pedestal': 9,\
        #    'chair_head': 10, 'back_surface': 11, 'back_frame_vertical_bar': 12, 'back_frame_horizontal_bar': 13, 'chair_seat': 14, 'chair_arm': 15, 'arm_sofa_style': 16, 'arm_near_vertical_bar': 17, 'arm_horizontal_bar': 18}
        #self.classes = ['tabletop', 'drawer', 'cabinet_door', 'side_panel', 'bottom_panel', 'central_support', 'leg', 'shelf', 'leg_bar', 'pedestal',\
            #'chair_head', 'back_surface', 'back_frame_vertical_bar', 'back_frame_horizontal_bar', 'chair_seat', 'chair_arm', 'arm_sofa_style', 'arm_near_vertical_bar', 'arm_horizontal_bar']
        if type == 'chair':
            self.part_order = {'chair_head': 0, 'back_surface': 1, 'back_frame_vertical_bar': 2, 'back_frame_horizontal_bar': 3, 'chair_seat': 4, 'chair_arm': 5, 'arm_sofa_style': 6, 'arm_near_vertical_bar': 7, 'arm_horizontal_bar': 8, 'central_support': 9, 'leg': 10, 'leg_bar': 11, 'pedestal': 12}
            self.classes = ['chair_head', 'back_surface', 'back_frame_vertical_bar', 'back_frame_horizontal_bar', 'chair_seat', 'chair_arm', 'arm_sofa_style', 'arm_near_vertical_bar', 'arm_horizontal_bar', 'central_support', 'leg', 'leg_bar', 'pedestal']
        if type == 'table':
            self.part_order = {'tabletop': 0, 'drawer': 1, 'cabinet_door': 2, 'side_panel': 3, 'bottom_panel': 4, 'central_support': 5, 'leg': 6, 'shelf': 7, 'leg_bar': 8, 'pedestal': 9}
            self.classes = ['tabletop', 'drawer', 'cabinet_door', 'side_panel', 'bottom_panel', 'central_support', 'leg', 'shelf', 'leg_bar', 'pedestal']
        if type == 'bed':
            self.part_order = {'headboard': 0, 'bed_sleep_area': 1, 'bed_frame_horizontal_surface':2, 'bed_side_surface_panel': 3, 'bed_post': 4, 'leg': 5, 'surface_base': 6, 'ladder':7}
            self.classes = ['headboard', 'bed_sleep_area', 'bed_frame_horizontal_surface', 'bed_side_surface_panel', 'bed_post', 'leg', 'surface_base', 'ladder']
        if type == 'bag':
            self.part_order = {'bag_body': 0, 'handle': 1, 'shoulder_strap': 2}
            self.classes = ['bag_body', 'handle', 'shoulder_strap']
        for di in next(os.walk(self.img_dir))[1]:
            if di.endswith("npy"): continue
            else:
                images = os.listdir(os.path.join(self.img_dir, di))
                for img in images:
                    if (not img == "0.png") and (img.endswith("occluded.png")):
                        

                        name = img.replace("occluded.png", "")

                        f = open (os.path.join(self.img_dir, di, "cat2idx.json"))
                        part_dict = json.load(f)
                        for (key, val) in part_dict.items():
                            for value in val:
                                if name == str(value):
                                    self.images.append(os.path.join(self.img_dir, di, img))
                                    self.targets.append(self.part_order[key])
 
    def __getitem__(self, index):
        img = self.images[index]
        image = Image.open(img).convert('L')
        image = transforms.functional.resize(image, 256)
        image = ImageOps.invert(image)
        target = self.targets[index]

        img_size = image.size

        if self.transform is not None:
            image = self.transform(image)
        
        class_name = self.classes[target]

        out = {'image': image, 'target': target, 'meta': {'im_size': img_size, 'index': index, 'class_name': class_name}}
        return out
        
    def __len__(self):
        return len(self.targets)

    def get_image(self, index):
        img = self.images[index]
        return img

    def extra_repr(self):
        return "Split: {}".format(self.split)