import logging
import os
import random
import shutil
from os import listdir
from os.path import splitext
from pathlib import Path
import timm.loss
import torchvision.transforms.functional as TF
import PIL
import cv2
import pandas as pd
import torchvision
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
from torchvision import transforms
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import v2,InterpolationMode
from tqdm import tqdm
import csv
from torchvision.datasets.coco import CocoCaptions
from sklearn.model_selection import StratifiedKFold
def create_Liver_Dataset_df(data_dir):
    images, labels, diagnosis =  [], [], []
    for root, folders, files in os.walk(data_dir):  # 在目录树中游走,会打开子文件夹
        for file in files:
            images.append(os.path.join(root,file))
            if 'Normal' in root:
                labels.append(2)
            if 'Benign'in root:
                labels.append(0)
            if 'Malignant' in root:
                labels.append(1)
    PathDF = pd.DataFrame({ 'images': images, 'labels': labels})
    train_df, valid_df = train_test_split(PathDF, random_state=42, test_size=0.1, shuffle=True,stratify=PathDF.values[:,-1])

    # train_df, valid_df = train_test_split(PathDF, random_state=42, test_size=0.2,shuffle=True)
    # valid_df, test_df = train_test_split(valid_df, random_state=42, test_size=0.2,shuffle=True)

    return train_df, valid_df  # ,test_df


def create_3Liver_Dataset_df():
    data_dir='/home/uax/SCY/split_out/test'
    tumor_train, liver_train, original_train,label_train =  [], [], [],[]
    # tumor_test, liver_test, original_train, label_train = [], [], [], []
    for root, folders, files in os.walk(data_dir):  # 在目录树中游走,会打开子文件夹
        for filename in files:
            if filename.endswith('liver.png'):
                liver_abpath=os.path.join(root,filename)
                tumor_abpath = liver_abpath.replace('liver','tumor')
                original_abpath = liver_abpath.replace('liver','original')
                tumor_train.append(tumor_abpath)
                liver_train.append(liver_abpath)
                original_train.append(original_abpath)
                if 'Benign' in root:
                    label_train.append(0)
                if 'Malignant' in root:
                    label_train.append(1)

    PathDF = pd.DataFrame({ 'tumor': tumor_train, 'liver': liver_train,'original':original_train,'label':label_train})
    #train_df, valid_df = train_test_split(PathDF, random_state=42, test_size=0.1, shuffle=True,stratify=PathDF.values[:,-1])
    PathDF.to_csv("liver_3_test.csv", index=False)

    # train_df, valid_df = train_test_split(PathDF, random_state=42, test_size=0.2,shuffle=True)
    # valid_df, test_df = train_test_split(valid_df, random_state=42, test_size=0.2,shuffle=True)




def create_patient_id_Liver_df():

    n_splits=5
    Benign_path = '/home/uax/SCY/LiverClassification/dataset/data/new_data/Benign'
    Malignant_path = '/home/uax/SCY/LiverClassification/dataset/data/new_data/Malignant'
    Benign_id = os.listdir(Benign_path)
    Benign_label = [0] * len(Benign_id)
    Malignant_id = os.listdir(Malignant_path)
    Malignant_label = [1] * len(Malignant_id)
    all_id = Benign_id + Malignant_id
    all_label = Benign_label + Malignant_label
    random.shuffle(Benign_id)
    random.shuffle(Malignant_id)
    #fold_size = int (len(all_id)*0.1)
    fold_size=150
    for i in range(n_splits):
        test_start = i * fold_size  # 测试集开始位置
        test_end = (i + 1) * fold_size  # 测试集结束位置
        test_Benign_id = Benign_id[test_start:test_end]
        train_Benign_id = Benign_id[:test_start] + Benign_id[test_end:]
        test_Malignant_id = Malignant_id[test_start:test_end]
        train_Malignant_id = Malignant_id[:test_start] + Malignant_id[test_end:]
        test_list,train_list=[],[]
        for root, folders, files in os.walk(Benign_path):  # 在目录树中游走,会打开子文件夹
            for file in files:
                id = file.split('_')[0]
                if id in test_Benign_id:
                    test_list.append(os.path.join(root,file))
                elif id in train_Benign_id:
                    train_list.append(os.path.join(root, file))
                else:
                    print('err0')
                    exit()
        for root, folders, files in os.walk(Malignant_path):  # 在目录树中游走,会打开子文件夹
            for file in files:
                id = file.split('_')[0]
                if id in test_Malignant_id:
                    test_list.append(os.path.join(root,file))
                elif id in train_Malignant_id:
                    train_list.append(os.path.join(root, file))
                else:
                    print('err1')
                    exit()
        random.shuffle(train_list)
        random.shuffle(test_list)
        train_label_list,test_label_list=[],[]
        for img_path in train_list:
            if 'Benign' in img_path:
                train_label_list.append(0)
            elif 'Malignant' in img_path:
                train_label_list.append(1)
            else:
                print('err2')
                exit()
        for img_path in test_list:
            if 'Benign' in img_path:
                test_label_list.append(0)
            elif 'Malignant' in img_path:
                test_label_list.append(1)
            else:
                print('err3')
                exit()
        train_df = pd.DataFrame({'images': train_list, 'labels': train_label_list})
        valid_df = pd.DataFrame({'images': test_list, 'labels': test_label_list})
        train_df.to_csv("/home/uax/SCY/LiverClassification/dataset/KFold_5/train_Liver{}.csv".format(i), index=False)
        valid_df.to_csv("/home/uax/SCY/LiverClassification/dataset/KFold_5/val_Liver{}.csv".format(i), index=False)


train_transform =transforms.Compose([
            transforms.RandomOrder ([
            transforms.RandomApply([transforms.AutoAugment(interpolation=3),]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomRotation(degrees=45,interpolation=3),]),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4,hue=0.3)],p=0.5),
            transforms.RandomApply([transforms.RandomAffine(degrees=30, translate=(0, 0.2), scale=(0.9, 1),shear=(6, 9))],p=0.5),
            transforms.RandomChoice([
                                     transforms.RandomGrayscale(p=0.5),
                                     transforms.RandomAdjustSharpness(sharpness_factor=20, p=0.5),
                                     transforms.GaussianBlur(kernel_size=(5, 5)),
                                     transforms.RandomAutocontrast(p=0.5),
                                     transforms.RandomPerspective(p=0.5),
                                     ]),
            transforms.RandomResizedCrop(224,(0.6,1)),
                ]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2),
        ])

valid_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

class Liver_Dataset(Dataset):
    def __init__(self, path_df,mode='train'):
        self.path_df = path_df

        self.mode = mode

        if mode=='val' or mode=='test':
            self.transforms =  v2.Compose([
                v2.ToImage(),
                v2.Resize((224,224),interpolation=InterpolationMode.BILINEAR),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                # v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])
        else:
            self.transforms = v2.Compose([
                v2.ToImage(),
                # v2.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
                v2.RandomResizedCrop((224,224),scale=(0.7,1.0),interpolation=InterpolationMode.BILINEAR
                                     ,antialias=True),

                v2.AutoAugment(interpolation=InterpolationMode.BILINEAR),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomErasing(p=0.1),

                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                # v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])



        # self.strong_transforms = A.Compose([
        #     A.RandomResizedCrop(224, 224, scale=(0.2, 1.0), p=1.0),
        #     A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.1, rotate_limit=20,
        #                        border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
        #     A.HorizontalFlip(p=0.5),  # 水平翻转
        #     A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.7),
        #     A.OneOf([
        #     A.CLAHE(p=0.5),
        #     A.GridDistortion( border_mode=cv2.BORDER_CONSTANT, value=0,p=0.5),
        #     ],p=0.5),
        #     A.OneOf([
        #         A.CoarseDropout(max_holes=8, max_height=8, max_width=8, fill_value=0, p=0.5),
        #         A.ElasticTransform(alpha_affine=10, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
        #     ], p=0.5),
        #     A.OneOf([
        #         A.HueSaturationValue(p=0.5),
        #         A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
        #     ], p=0.5),
        #     A.OneOf([
        #         A.GaussianBlur(p=0.5),
        #         A.ToGray(p=0.5),
        #     ], p=0.5),
        #     A.Normalize(),
        #     ToTensorV2()
        #     ]
        # )

    def __len__(self):
        return self.path_df.shape[0]
    def __getitem__(self, idx):
        img_path =  self.path_df.iloc[idx]['images']
        # img_path='/home/uax/SCY/A_ADATA/classification_data/test/Benign/C31256_1_0_2_1.png'
        label = self.path_df.iloc[idx]['labels']

        image = self.transforms(Image.open(img_path).convert("RGB"))
        return image, label,img_path
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # if self.mode == 'train':
        #     augmentation = self.strong_transforms(image=img)
        #     img = augmentation['image']
        #     return img, label
        # if self.mode == 'val':
        #     augmentation = self.val_tranforms(image=img)
        #     img = augmentation['image']
        #     return  img,label
        # if self.mode == 'visualization':
        #     augmentation = self.val_tranforms(image=img)
        #     img = augmentation['image']
        #     # id = img_path.split('/')[-1].split('_')[0]
        #     return  img,label,img_path.split('/')[-1]
        #image, label=self.mixup()
        # if np.random.rand() > 0.5:
        #     all_idx = self.path_df[self.path_df['labels'] == label].index.tolist()
        #     rand_idx = random.choice(all_idx)
        #
        #     img2 = cv2.imread(self.path_df.iloc[rand_idx]['images'])
        #     img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)





class Liver_add_AFP_and_size_Dataset(Dataset):
    def __init__(self, path_df,mode='train',img_path=True):
        self.path_df = path_df
        self.mode = mode
        self.img_path =img_path
        self.val_tranforms = A.Compose([
            A.Resize(224, 224),
            A.Normalize(),
            ToTensorV2()])
        self.strong_transforms = A.Compose([
            A.RandomResizedCrop(224, 224, scale=(0.6, 1.0), p=1.0),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.1, rotate_limit=20,
                               border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
            A.HorizontalFlip(p=0.5),  # 水平翻转
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.7),
            A.OneOf([
            A.CLAHE(p=1),
            A.GridDistortion( border_mode=cv2.BORDER_CONSTANT, value=0,p=1),
            ],p=0.5),
            A.OneOf([
                A.CoarseDropout(max_holes=8, max_height=8, max_width=8, fill_value=0, p=1),
                A.ElasticTransform(alpha_affine=10, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
            ], p=0.6),
            A.OneOf([
                A.HueSaturationValue(p=1),
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1),
            ], p=0.4),
            A.OneOf([
                A.GaussianBlur(p=1),
                A.ToGray(p=1),
            ], p=0.5),
            A.Normalize(),
            ToTensorV2()
            ]
        )
        self.normal_tranforms = A.Compose([
            A.Normalize(),
            ToTensorV2()])

    def __len__(self):
        return self.path_df.shape[0]

    def __getitem__(self, idx):
        img_path =  self.path_df.iloc[idx]['images']
        label = self.path_df.iloc[idx]['labels']
        AFP = self.path_df.iloc[idx]['AFP']
        if AFP>=20:
            AFP=1
        else:
            AFP = 0
        size =  self.path_df.iloc[idx]['size']
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.mode == 'val':
            augmentation = self.val_tranforms(image=img)
            img = augmentation['image']
            if self.img_path:
                return img_path,img, label, AFP, size
            else:
                return  img,label,AFP,size
        #image, label=self.mixup()
        # if np.random.rand() > 0.5:
        #     all_idx = self.path_df[self.path_df['labels'] == label].index.tolist()
        #     rand_idx = random.choice(all_idx)
        #
        #     img2 = cv2.imread(self.path_df.iloc[rand_idx]['images'])
        #     img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)


        augmentation = self.strong_transforms(image=img)
        img = augmentation['image']
        return img, label,AFP,size

class Liver_and_size_Dataset(Dataset):
    def __init__(self, path_df,mode='train',img_path=False):
        self.path_df = path_df
        self.mode = mode
        self.img_path =img_path
        self.val_tranforms = A.Compose([
            A.Resize(224, 224),
            A.Normalize(),
            ToTensorV2()])
        self.strong_transforms = A.Compose([
            A.RandomResizedCrop(224, 224, scale=(0.6, 1.0), p=1.0),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.1, rotate_limit=20,
                               border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
            A.HorizontalFlip(p=0.5),  # 水平翻转
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.7),
            A.OneOf([
            A.CLAHE(p=1),
            A.GridDistortion( border_mode=cv2.BORDER_CONSTANT, value=0,p=1),
            ],p=0.5),
            A.OneOf([
                A.CoarseDropout(max_holes=8, max_height=8, max_width=8, fill_value=0, p=1),
                A.ElasticTransform(alpha_affine=10, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
            ], p=0.6),
            A.OneOf([
                A.HueSaturationValue(p=1),
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1),
            ], p=0.4),
            A.OneOf([
                A.GaussianBlur(p=1),
                A.ToGray(p=1),
            ], p=0.5),
            A.Normalize(),
            ToTensorV2()
            ]
        )
        self.normal_tranforms = A.Compose([
            A.Normalize(),
            ToTensorV2()])

    def __len__(self):
        return self.path_df.shape[0]

    def __getitem__(self, idx):
        img_path =  self.path_df.iloc[idx]['images']
        label = self.path_df.iloc[idx]['labels']
        # AFP = self.path_df.iloc[idx]['AFP']
        # if AFP>=20:
        #     AFP=1
        # else:
        #     AFP = 0
        size =  self.path_df.iloc[idx]['size']
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.mode == 'val':
            augmentation = self.val_tranforms(image=img)
            img = augmentation['image']
            if self.img_path:
                return img_path,img, label,  size
            else:
                return  img,label,size
        #image, label=self.mixup()
        # if np.random.rand() > 0.5:
        #     all_idx = self.path_df[self.path_df['labels'] == label].index.tolist()
        #     rand_idx = random.choice(all_idx)
        #
        #     img2 = cv2.imread(self.path_df.iloc[rand_idx]['images'])
        #     img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)


        augmentation = self.strong_transforms(image=img)
        img = augmentation['image']
        return img, label,size


class Liver_test_2classification_Dataset(Dataset):
    def __init__(self, path_df,mode='train',cam=None,id=False):
        self.id = id
        self.path_df = path_df
        self.mode = mode
        self.cam=cam
        self.TTA_tranforms = A.Compose([
            A.RandomResizedCrop(224, 224, scale=(0.6, 1.0), p=1.0),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.3),
            A.Normalize(),
            ToTensorV2()])
        # self.TTA_tranforms = A.Compose([
        #     A.RandomResizedCrop(224, 224, scale=(0.7, 1.0), p=1.0),
        #     A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.1, rotate_limit=20,
        #                        border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
        #     A.HorizontalFlip(p=0.5),  # 水平翻转
        #     A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.7),
        #     A.OneOf([
        #     A.CLAHE(p=1),
        #     A.GridDistortion( border_mode=cv2.BORDER_CONSTANT, value=0,p=1),
        #     ],p=0.5),
        #     A.OneOf([
        #         A.CoarseDropout(max_holes=8, max_height=8, max_width=8, fill_value=0, p=1),
        #         A.ElasticTransform(alpha_affine=10, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
        #     ], p=0.6),
        #     A.OneOf([
        #         A.HueSaturationValue(p=1),
        #         A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1),
        #     ], p=0.4),
        #     A.OneOf([
        #         A.GaussianBlur(p=1),
        #         A.ToGray(p=1),
        #     ], p=0.5),
        #     A.Normalize(),
        #     ToTensorV2()])
        self.val_tranforms = A.Compose([
            A.Resize(224, 224),
            A.Normalize(),
            ToTensorV2()])
        self.strong_transforms = A.Compose([
            A.RandomResizedCrop(224, 224, scale=(0.7, 1.0), p=1.0),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.1, rotate_limit=20,
                               border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
            A.HorizontalFlip(p=0.5),  # 水平翻转
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.7),
            A.OneOf([
            A.CLAHE(p=1),
            A.GridDistortion( border_mode=cv2.BORDER_CONSTANT, value=0,p=1),
            ],p=0.5),
            A.OneOf([
                A.CoarseDropout(max_holes=8, max_height=8, max_width=8, fill_value=0, p=1),
                A.ElasticTransform(alpha_affine=10, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
            ], p=0.6),
            A.OneOf([
                A.HueSaturationValue(p=1),
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1),
            ], p=0.4),
            A.OneOf([
                A.GaussianBlur(p=1),
                A.ToGray(p=1),
            ], p=0.5),
            A.Normalize(),
            ToTensorV2()
            ]
        )
        self.normal_tranforms = A.Compose([
            A.Normalize(),
            ToTensorV2()])

    def __len__(self):
        return self.path_df.shape[0]

    def mixup(self, image, label):
        all_idx = self.path_df[self.path_df['labels'] == label].index.tolist()
        rand_idx = random.choice(all_idx)

        img2 = cv2.imread(self.path_df.iloc[rand_idx]['images'])
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        lam = np.random.beta(1.0, 1.0)
        rand_idx = random.choice(range(len(self)))

        W, H = image.shape[1], image.shape[2]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        #预设两种方式，一种框的，一种整图的
        bbx1, bby1, bbx2, bby2 = self.get_rand_bbox(image.shape, lam)
        image[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image.shape[-2] * image.shape[-1]))

        label = lam * label + (1 - lam) * self.path_df.iloc[rand_idx]['labels']


        return image, label

    def __getitem__(self, idx):
        img_path =  self.path_df.iloc[idx]['images']
        label = self.path_df.iloc[idx]['labels']

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.mode == 'TTA':
            augmentation = self.TTA_tranforms(image=img)
            img = augmentation['image']
            #id = img_path.split('/')[-1].split('.')[0]
            if self.id:
                return  img_path.split('/')[-1].split('.')[0],img, label
            if self.cam:
                return img, label,img_path.split('/')[-1]
            return  img, label
        if self.mode == 'val':
            augmentation = self.val_tranforms(image=img)
            img = augmentation['image']
            #id = img_path.split('/')[-1].split('.')[0]
            if self.id:
                return  img_path.split('/')[-1].split('.')[0],img, label
            if self.cam:
                return img, label,img_path.split('/')[-1]
            return  img, label
            #return id, img,label
        #image, label=self.mixup()
        # if np.random.rand() > 0.5:
        #     all_idx = self.path_df[self.path_df['labels'] == label].index.tolist()
        #     rand_idx = random.choice(all_idx)
        #
        #     img2 = cv2.imread(self.path_df.iloc[rand_idx]['images'])
        #     img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)


        augmentation = self.strong_transforms(image=img)
        img = augmentation['image']

        return img_path.split('/')[-1].split('.')[0],img, label

class Liver2Dataset(Dataset):
    def __init__(self, path_df,mode='train'):
        self.path_df = path_df
        self.mode = mode
        self.val_tranforms = A.Compose([
            A.Resize(224, 224),
            A.Normalize(),
            ToTensorV2()],additional_targets={'image0': 'image'})
        self.strong_transforms = A.Compose([
            #A.Resize(224, 224),
            #A.RandomScale((0.5, 2.0), p=1),

            A.RandomResizedCrop(224, 224, scale=(0.7, 1.0), p=1.0),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.1, rotate_limit=30,
                               border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
            A.HorizontalFlip(p=0.5),  # 垂直翻转
            #A.VerticalFlip(p=0.5),  # 水平翻转
            #A.Rotate(35, p=0.5),  # 随机旋转0个或多个90度
            A.CoarseDropout(max_holes=8, max_height=8, max_width=8, fill_value=0, p=0.5),
            A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5, p=0.7),
            # A.GridDistortion(p=0.5),
            # A.ElasticTransform(p=0.5),
            A.ElasticTransform(p=0.5,alpha_affine=10,border_mode=cv2.BORDER_CONSTANT, value=0,),
            A.CLAHE(p=0.5),A.GridDistortion(border_mode=cv2.BORDER_CONSTANT, value=0,p=0.7),
                     A.HueSaturationValue(p=0.5), A.ChannelShuffle(p=0.5), #A.GridDropout(p=0.5),
                     A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
            # A.OneOf([A.ToGray(p=1),A.GridDistortion(p=1),A.ElasticTransform(p=1),A.CLAHE(p=1),
            #          A.HueSaturationValue(p=1),A.ChannelShuffle(p=1),A.GridDropout(p=1),
            #          A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1),
            #          ], p=0.5),  # 垂直翻转
            A.GaussianBlur(p=0.2),  # 水平翻转

            A.ToGray(p=0.5),
            A.Normalize(),
            ToTensorV2()],additional_targets={'image0': 'image'})
    def __len__(self):
        return self.path_df.shape[0]
        #return len(self.imgs_list)  # df结构的shape[0]就是样本数，shape[1]可以理解为特征数


    def __getitem__(self, idx):
        #tumor_path =  self.path_df.iloc[idx]['tumor']
        liver_path = self.path_df.iloc[idx]['liver']
        original_path = self.path_df.iloc[idx]['original']
        label = self.path_df.iloc[idx]['label']
        # tumor_img=cv2.imread(tumor_path)
        # tumor_img = cv2.cvtColor(tumor_img, cv2.COLOR_BGR2RGB)

        liver_img=cv2.imread(liver_path)
        liver_img = cv2.cvtColor(liver_img, cv2.COLOR_BGR2RGB)

        original_img = cv2.imread(original_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

        # self.check_image_range(tumor_img)
        # self.check_image_range(liver_img)
        # self.check_image_range(original_img)
        #_, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        if self.mode == 'val':
            augmentation = self.val_tranforms(image=liver_img, image0=original_img)
            liver_img,original_img = augmentation['image'],augmentation['image0']
            return liver_img,original_img,label


        augmentation = self.strong_transforms(image=liver_img, image0=original_img)
        liver_img,original_img = augmentation['image'],augmentation['image0']

        # original_img = original_img.permute(1, 2, 0).cpu().numpy()
        # liver_img = liver_img.permute(1, 2, 0).cpu().numpy()
        # tumor_img = tumor_img.permute(1, 2, 0).cpu().numpy()
        #
        # # 反归一化
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        # original_img = (original_img * std + mean) * 255.0
        # liver_img = (liver_img * std + mean) * 255.0
        # tumor_img = (tumor_img * std + mean) * 255.0
        # original_img = np.clip(original_img, 0, 255).astype(np.uint8)
        # liver_img = np.clip(liver_img, 0, 255).astype(np.uint8)
        # tumor_img = np.clip(tumor_img, 0, 255).astype(np.uint8)
        #
        # fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        # axes[0].imshow(original_img)
        # axes[0].set_title('Original')
        # axes[1].imshow(liver_img)
        # axes[1].set_title('Liver')
        # axes[2].imshow(tumor_img)
        # axes[2].set_title('Tumor')
        # plt.show()
        return  liver_img, original_img, label

#
def process_Liver_hepatitis_Dataset_df():
    random.seed(10086)
    np.random.seed(10086)

    normal_data_dir='/home/uax/SCY/A_ADATA/classification_data/normal/hepatitis'
    normal_hepatitis_list=[]
    for root, folders, files in os.walk(normal_data_dir):  # 在目录树中游走,会打开子文件夹
        for filename in files:
            id=filename.split('_')[0]
            if id not in normal_hepatitis_list:
                normal_hepatitis_list.append(id)
    tumor_Bdata_dir = '/home/uax/SCY/A_ADATA/classification_data/tumor/hepatitis/Benign'
    tumor_B_hepatitis_list=[]
    for root, folders, files in os.walk(tumor_Bdata_dir):  # 在目录树中游走,会打开子文件夹
        for filename in files:
            id=filename.split('_')[0]
            if id not in tumor_B_hepatitis_list:
                tumor_B_hepatitis_list.append(id)
    tumor_Mdata_dir = '/home/uax/SCY/A_ADATA/classification_data/tumor/hepatitis/Malignant'
    tumor_M_hepatitis_list=[]
    for root, folders, files in os.walk(tumor_Mdata_dir):  # 在目录树中游走,会打开子文件夹
        for filename in files:
            id=filename.split('_')[0]
            if id not in tumor_M_hepatitis_list:
                tumor_M_hepatitis_list.append(id)
    benign_count = len(tumor_B_hepatitis_list)
    malignant_count = len(tumor_M_hepatitis_list)
    normal_count = len(normal_hepatitis_list)

    total_count = benign_count + malignant_count + normal_count

    benign_ratio = benign_count / total_count
    malignant_ratio = malignant_count / total_count
    normal_ratio = normal_count / total_count

    test_size = 2027

    benign_test_count = int(benign_ratio * test_size)
    malignant_test_count = int(malignant_ratio * test_size)
    normal_test_count = test_size - benign_test_count - malignant_test_count

    benign_test = random.sample(tumor_B_hepatitis_list, benign_test_count)
    malignant_test = random.sample(tumor_M_hepatitis_list, malignant_test_count)
    normal_test = random.sample(normal_hepatitis_list, normal_test_count)

    malignant_train = list(set(tumor_M_hepatitis_list) - set(malignant_test))
    benign_train = list(set(tumor_B_hepatitis_list) - set(benign_test))
    normal_train = list(set(normal_hepatitis_list) - set(normal_test))
    test_set = set(benign_test + malignant_test + normal_test)
    train_set = set(malignant_train+benign_train+normal_train)
    test_out_abpath='/home/uax/SCY/A_ADATA/classification_data/test'
    train_out_abpath = '/home/uax/SCY/A_ADATA/classification_data/train'

    for root, folders, files in os.walk(normal_data_dir):  # 在目录树中游走,会打开子文件夹
        for filename in files:
            if filename.split('_')[0] in test_set:
                img_abpath=os.path.join(root,filename)
                out_dir=os.path.join(test_out_abpath,'Normal')
                os.makedirs(out_dir,exist_ok=True)
                out_img_path=os.path.join(out_dir,filename)
                shutil.copy(img_abpath,out_img_path)
            elif filename.split('_')[0] in train_set:
                img_abpath=os.path.join(root,filename)
                out_dir=os.path.join(train_out_abpath,'Normal')
                os.makedirs(out_dir,exist_ok=True)
                out_img_path=os.path.join(out_dir,filename)
                shutil.copy(img_abpath,out_img_path)
    #tumor_Bdata_dir = '/home/uax/SCY/A_DATA/classification_data/tumor/hepatitis/B'
    tumor_dir='/home/uax/SCY/A_ADATA/classification_data/tumor/hepatitis'
    for root, folders, files in os.walk(tumor_dir):  # 在目录树中游走,会打开子文件夹
        for filename in files:
            if filename.split('_')[0] in test_set:
                img_abpath=os.path.join(root,filename)
                out_dir=os.path.join(test_out_abpath,os.path.basename(root))
                os.makedirs(out_dir,exist_ok=True)
                out_img_path=os.path.join(out_dir,filename)
                shutil.copy(img_abpath,out_img_path)
            elif filename.split('_')[0] in train_set:
                img_abpath=os.path.join(root,filename)
                out_dir=os.path.join(train_out_abpath,os.path.basename(root))
                os.makedirs(out_dir,exist_ok=True)
                out_img_path=os.path.join(out_dir,filename)
                shutil.copy(img_abpath,out_img_path)

    # for root, folders, files in os.walk(tumor_Mdata_dir):  # 在目录树中游走,会打开子文件夹
    #     for filename in files:
    #         if filename.split('_')[0] in malignant_test:
    #             img_abpath=os.path.join(root,filename)
    #             out_dir=os.path.join(test_out_abpath,os.path.basename(root))
    #             os.makedirs(out_dir,exist_ok=True)
    #             out_img_path=os.path.join(out_dir,filename)
    #             shutil.copy(img_abpath,out_img_path)
    #         elif filename.split('_')[0] in malignant_train:
    #             img_abpath=os.path.join(root,filename)
    #             out_dir=os.path.join(train_out_abpath,os.path.basename(root))
    #             os.makedirs(out_dir,exist_ok=True)
    #             out_img_path=os.path.join(out_dir,filename)
    #             shutil.copy(img_abpath,out_img_path)
def id_test():
    random.seed(10086)
    np.random.seed(10086)
    test_dir = '/home/uax/SCY/A_ADATA/classification_data/test'
    out_test_dir = '/home/uax/SCY/A_ADATA/classification_data/id_test'
    test_hepatitis=[]
    err_label_test=[]
    for root, folders, files in os.walk(test_dir):  # 在目录树中游走,会打开子文件夹
        for filename in files:
            img_path=os.path.join(root,filename)
            id=filename.split('_')[0]
            id_dir=os.path.join(out_test_dir,id)
            os.makedirs(id_dir,exist_ok=True)
            new_path=os.path.join(id_dir,filename)
            shutil.copy(img_path, new_path)


def process_Liver_5split_Dataset_df():
    random.seed(10086)
    np.random.seed(10086)
    test3_dir = '/home/uax/SCY/A_ADATA/classification_data/test_3'
    test3_img,test3_label=[],[]
    for root, folders, files in os.walk(test3_dir):  # 在目录树中游走,会打开子文件夹
        for filename in files:
            img_abpath=os.path.join(root,filename)
            test3_img.append(img_abpath)
            dir_label = os.path.basename(root)
            if dir_label == 'Normal':
                test3_label.append(0)
            elif dir_label == 'Benign':
                test3_label.append(1)
            elif dir_label == 'Malignant':
                test3_label.append(2)
    PathDF_test3 = pd.DataFrame({'images': test3_img, 'labels': test3_label})
    PathDF_test3.to_csv("liver_direct_test_3.csv", index=False)

    test_dir = '/home/uax/SCY/A_ADATA/classification_data/test'
    test_img,test_label=[],[]
    for root, folders, files in os.walk(test_dir):  # 在目录树中游走,会打开子文件夹
        for filename in files:
            img_abpath=os.path.join(root,filename)
            test_img.append(img_abpath)
            dir_label = os.path.basename(root)
            if dir_label == 'Normal':
                test_label.append(0)
            elif dir_label == 'Benign':
                test_label.append(1)
            elif dir_label == 'Malignant':
                test_label.append(2)
    PathDF_test = pd.DataFrame({'images': test_img, 'labels': test_label})
    PathDF_test.to_csv("liver_direct_test.csv", index=False)

    data_dir='/home/uax/SCY/A_ADATA/classification_data/train'
    Normal_patient_list,Benign_patient_list,Malignant_patient_list=[],[],[]
    for root, folders, files in os.walk(data_dir):  # 在目录树中游走,会打开子文件夹
        for filename in files:
            if os.path.basename(root)=='Normal':
                id=filename.split('_')[0]
                if id not in Normal_patient_list:
                    Normal_patient_list.append(id)
            if os.path.basename(root)=='Benign':
                id=filename.split('_')[0]
                if id not in Benign_patient_list:
                    Benign_patient_list.append(id)
            if os.path.basename(root)=='Malignant':
                id=filename.split('_')[0]
                if id not in Malignant_patient_list:
                    Malignant_patient_list.append(id)
    # X = Normal_patient_list + Benign_patient_list + Malignant_patient_list
    # y = [0] * len(Normal_patient_list) + [1] * len(Benign_patient_list) + [2] * len(Malignant_patient_list)
    X = np.concatenate((Normal_patient_list, Benign_patient_list, Malignant_patient_list))
    y = np.concatenate((np.zeros(len(Normal_patient_list)),
                        np.ones(len(Benign_patient_list)),
                        2 * np.ones(len(Malignant_patient_list)))).astype(int)
    skf = StratifiedKFold(n_splits=5,shuffle=True)

    for Fold_i,(train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        #y_train, y_test = y[train_index], y[test_index]
        train_img,train_label = [],[]
        val_img,val_label = [],[]
        for root, folders, files in os.walk(data_dir):  # 在目录树中游走,会打开子文件夹
            for filename in files:
                img_abpath=os.path.join(root,filename)
                patient_id = filename.split('_')[0]
                if patient_id in X_train:
                    train_img.append(img_abpath)
                    dir_label = os.path.basename(root)
                    if dir_label=='Normal':
                        train_label.append(0)
                    elif dir_label=='Benign':
                        train_label.append(1)
                    elif dir_label=='Malignant':
                        train_label.append(2)
                    else:
                        raise ValueError('Invalid train label encountered: {}'.format(dir_label))
                elif patient_id in X_test:
                    val_img.append(img_abpath)
                    dir_label = os.path.basename(root)
                    if dir_label=='Normal':
                        val_label.append(0)
                    elif dir_label=='Benign':
                        val_label.append(1)
                    elif dir_label=='Malignant':
                        val_label.append(2)
                    else:
                        raise ValueError('Invalid test label encountered: {}'.format(dir_label))
                else:
                    raise ValueError('Invalid patient_id encountered: {}'.format(patient_id))

        set1 = set(train_img)
        set2 = set(val_img)

        if len(set1 & set2) > 0:
            print("有重复元素")

        PathDF_train = pd.DataFrame({'images': train_img, 'labels': train_label})
        PathDF_val = pd.DataFrame({'images': val_img, 'labels': val_label})
        PathDF_train.to_csv("liver_direct_train{}.csv".format(Fold_i), index=False)
        PathDF_val.to_csv("liver_direct_val{}.csv".format(Fold_i), index=False)



def process_Liver_hepatitis_2_classification_Dataset_df():
    random.seed(10086)
    np.random.seed(10086)

    tumor_Bdata_dir = '/home/uax/SCY/A_ADATA/classification_data/tumor/hepatitis/Benign'
    tumor_B_hepatitis_list=[]
    for root, folders, files in os.walk(tumor_Bdata_dir):  # 在目录树中游走,会打开子文件夹
        for filename in files:
            id=filename.split('_')[0]
            if id not in tumor_B_hepatitis_list:
                tumor_B_hepatitis_list.append(id)
    tumor_Mdata_dir = '/home/uax/SCY/A_ADATA/classification_data/tumor/hepatitis/Malignant'
    tumor_M_hepatitis_list=[]
    for root, folders, files in os.walk(tumor_Mdata_dir):  # 在目录树中游走,会打开子文件夹
        for filename in files:
            id=filename.split('_')[0]
            if id not in tumor_M_hepatitis_list:
                tumor_M_hepatitis_list.append(id)
    benign_count = len(tumor_B_hepatitis_list)
    malignant_count = len(tumor_M_hepatitis_list)


    total_count = benign_count + malignant_count

    benign_ratio = benign_count / total_count
    malignant_ratio = malignant_count / total_count

    test_size = 2000

    benign_test_count = int(benign_ratio * test_size)
    malignant_test_count = int(malignant_ratio * test_size)
    normal_test_count = test_size - benign_test_count - malignant_test_count

    benign_test = random.sample(tumor_B_hepatitis_list, benign_test_count)
    malignant_test = random.sample(tumor_M_hepatitis_list, malignant_test_count)
    normal_test = random.sample(normal_hepatitis_list, normal_test_count)

    malignant_train = list(set(tumor_M_hepatitis_list) - set(malignant_test))
    benign_train = list(set(tumor_B_hepatitis_list) - set(benign_test))
    normal_train = list(set(normal_hepatitis_list) - set(normal_test))
    test_set = set(benign_test + malignant_test + normal_test)
    train_set = set(malignant_train+benign_train+normal_train)
    test_out_abpath='/home/uax/SCY/A_ADATA/classification_data/test'
    train_out_abpath = '/home/uax/SCY/A_ADATA/classification_data/train'

    for root, folders, files in os.walk(normal_data_dir):  # 在目录树中游走,会打开子文件夹
        for filename in files:
            if filename.split('_')[0] in test_set:
                img_abpath=os.path.join(root,filename)
                out_dir=os.path.join(test_out_abpath,'Normal')
                os.makedirs(out_dir,exist_ok=True)
                out_img_path=os.path.join(out_dir,filename)
                shutil.copy(img_abpath,out_img_path)
            elif filename.split('_')[0] in train_set:
                img_abpath=os.path.join(root,filename)
                out_dir=os.path.join(train_out_abpath,'Normal')
                os.makedirs(out_dir,exist_ok=True)
                out_img_path=os.path.join(out_dir,filename)
                shutil.copy(img_abpath,out_img_path)
    #tumor_Bdata_dir = '/home/uax/SCY/A_DATA/classification_data/tumor/hepatitis/B'
    tumor_dir='/home/uax/SCY/A_ADATA/classification_data/tumor/hepatitis'
    for root, folders, files in os.walk(tumor_dir):  # 在目录树中游走,会打开子文件夹
        for filename in files:
            if filename.split('_')[0] in test_set:
                img_abpath=os.path.join(root,filename)
                out_dir=os.path.join(test_out_abpath,os.path.basename(root))
                os.makedirs(out_dir,exist_ok=True)
                out_img_path=os.path.join(out_dir,filename)
                shutil.copy(img_abpath,out_img_path)
            elif filename.split('_')[0] in train_set:
                img_abpath=os.path.join(root,filename)
                out_dir=os.path.join(train_out_abpath,os.path.basename(root))
                os.makedirs(out_dir,exist_ok=True)
                out_img_path=os.path.join(out_dir,filename)
                shutil.copy(img_abpath,out_img_path)
def process_Liver_5split_Dataset_df_2_classification():
    random.seed(10086)
    np.random.seed(10086)
    test3_dir = '/home/uax/SCY/A_ADATA/classification_data/test_3'
    test3_img,test3_label=[],[]
    for root, folders, files in os.walk(test3_dir):  # 在目录树中游走,会打开子文件夹
        for filename in files:
            img_abpath=os.path.join(root,filename)
            dir_label = os.path.basename(root)
            if dir_label == 'Benign':
                test3_label.append(0)
                test3_img.append(img_abpath)
            elif dir_label == 'Malignant':
                test3_label.append(1)
                test3_img.append(img_abpath)
    PathDF_test3 = pd.DataFrame({'images': test3_img, 'labels': test3_label})
    PathDF_test3.to_csv("liver_externaltest_2classification.csv", index=False)

    test_dir = '/home/uax/SCY/A_ADATA/classification_data/test'
    test_img,test_label=[],[]
    for root, folders, files in os.walk(test_dir):  # 在目录树中游走,会打开子文件夹
        for filename in files:
            img_abpath=os.path.join(root,filename)
            dir_label = os.path.basename(root)
            if dir_label == 'Benign':
                test_label.append(0)
                test_img.append(img_abpath)
            elif dir_label == 'Malignant':
                test_label.append(1)
                test_img.append(img_abpath)
    PathDF_test = pd.DataFrame({'images': test_img, 'labels': test_label})
    PathDF_test.to_csv("liver_test_2classification.csv", index=False)

    data_dir='/home/uax/SCY/A_ADATA/classification_data/train'
    Benign_patient_list,Malignant_patient_list=[],[]
    for root, folders, files in os.walk(data_dir):  # 在目录树中游走,会打开子文件夹
        for filename in files:

            if os.path.basename(root)=='Benign':
                id=filename.split('_')[0]
                if id not in Benign_patient_list:
                    Benign_patient_list.append(id)
            if os.path.basename(root)=='Malignant':
                id=filename.split('_')[0]
                if id not in Malignant_patient_list:
                    Malignant_patient_list.append(id)
    # X = Normal_patient_list + Benign_patient_list + Malignant_patient_list
    # y = [0] * len(Normal_patient_list) + [1] * len(Benign_patient_list) + [2] * len(Malignant_patient_list)
    X = np.concatenate((Benign_patient_list, Malignant_patient_list))
    y = np.concatenate((np.zeros(len(Benign_patient_list)),
                        np.ones(len(Malignant_patient_list))
                        )).astype(int)
    skf = StratifiedKFold(n_splits=5,shuffle=True)

    for Fold_i,(train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        #y_train, y_test = y[train_index], y[test_index]
        train_img,train_label = [],[]
        val_img,val_label = [],[]
        for root, folders, files in os.walk(data_dir):  # 在目录树中游走,会打开子文件夹
            for filename in files:
                img_abpath=os.path.join(root,filename)
                patient_id = filename.split('_')[0]
                if patient_id in X_train:

                    dir_label = os.path.basename(root)
                    if dir_label=='Benign':
                        train_label.append(0)
                        train_img.append(img_abpath)
                    elif dir_label=='Malignant':
                        train_label.append(1)
                        train_img.append(img_abpath)
                elif patient_id in X_test:

                    dir_label = os.path.basename(root)

                    if dir_label=='Benign':
                        val_label.append(0)
                        val_img.append(img_abpath)
                    elif dir_label=='Malignant':
                        val_label.append(1)
                        val_img.append(img_abpath)

        set1 = set(train_img)
        set2 = set(val_img)

        if len(set1 & set2) > 0:
            print("有重复元素")

        PathDF_train = pd.DataFrame({'images': train_img, 'labels': train_label})
        PathDF_val = pd.DataFrame({'images': val_img, 'labels': val_label})
        PathDF_train.to_csv("liver_train_2classification{}.csv".format(Fold_i), index=False)
        PathDF_val.to_csv("liver_val_2classification{}.csv".format(Fold_i), index=False)




def check_split_Dataset_df():
    random.seed(10086)
    np.random.seed(10086)
    test_dir = '/home/uax/SCY/A_ADATA/classification_data/test'
    test_hepatitis=[]
    err_label_test=[]
    for root, folders, files in os.walk(test_dir):  # 在目录树中游走,会打开子文件夹
        for filename in files:
            class_label=filename.split('_')[2]
            if '.' in class_label:
                class_label=class_label.split('.')[0]
            id=filename.split('_')[0]
            if id not in test_hepatitis and class_label!='2':
                test_hepatitis.append(id)
            dir_label=os.path.basename(root)
            if dir_label=='Benign' and class_label!='0':
                err_label_test.append(id)
            if dir_label=='Malignant' and class_label!='1':
                err_label_test.append(id)
            if dir_label=='Normal' and class_label!='2':
                err_label_test.append(id)
    train_dir = '/home/uax/SCY/A_ADATA/classification_data/train'
    train_repeat_hepatitis=[]
    err_label_train = []
    for root, folders, files in os.walk(train_dir):  # 在目录树中游走,会打开子文件夹
        for filename in files:
            id=filename.split('_')[0]
            if id in test_hepatitis:
                train_repeat_hepatitis.append(id)
            class_label=filename.split('_')[2]
            if '.' in class_label:
                class_label=class_label.split('.')[0]
            dir_label=os.path.basename(root)
            if dir_label=='Benign' and class_label!='0':
                err_label_train.append(id)
            if dir_label=='Malignant' and class_label!='1':
                err_label_train.append(id)
            if dir_label=='Normal' and class_label!='2':
                err_label_train.append(id)

    a=0

def process_Liver_7_test_df_2_classification():
    random.seed(10086)
    np.random.seed(10086)
    test7_dir = '/home/uax/SCY/A_ADATA/classification_data/test_7'
    test7_img,test7_label=[],[]
    # for root, folders, files in os.walk(test7_dir):  # 在目录树中游走,会打开子文件夹
    #     for filename in files:
    #         img_abpath=os.path.join(root,filename)
    #         dir_label = os.path.basename(root)
    #         if dir_label == 'Benign':
    #             test7_label.append(0)
    #             test7_img.append(img_abpath)
    #         elif dir_label == 'Malignant':
    #             test7_label.append(1)
    #             test7_img.append(img_abpath)
    external_independent_test_path='/home/uax/SCY/A_ADATA/classification_data/external_independent_test'
    for root, folders, files in os.walk(external_independent_test_path):  # 在目录树中游走,会打开子文件夹
        for filename in files:
            img_abpath=os.path.join(root,filename)
            dir_label = os.path.basename(root)
            if dir_label == 'Benign':
                test7_label.append(0)
                test7_img.append(img_abpath)
            elif dir_label == 'Malignant':
                test7_label.append(1)
                test7_img.append(img_abpath)

    PathDF_test3 = pd.DataFrame({'images': test7_img, 'labels': test7_label})
    PathDF_test3.to_csv("liver_external_independent_test_2classification.csv", index=False)


def new_val_dataset_without_3():
    random.seed(10086)
    np.random.seed(10086)
    test7_dir = '/home/uax/SCY/A_ADATA/classification_data/val'
    test7_img,test7_label=[],[]
    # for root, folders, files in os.walk(test7_dir):  # 在目录树中游走,会打开子文件夹
    #     for filename in files:
    #         img_abpath=os.path.join(root,filename)
    #         dir_label = os.path.basename(root)
    #         if dir_label == 'Benign':
    #             test7_label.append(0)
    #             test7_img.append(img_abpath)
    #         elif dir_label == 'Malignant':
    #             test7_label.append(1)
    #             test7_img.append(img_abpath)
    independent_val_path='/home/uax/SCY/A_ADATA/classification_data/val'
    for root, folders, files in os.walk(independent_val_path):  # 在目录树中游走,会打开子文件夹
        for filename in files:
            img_abpath=os.path.join(root,filename)
            dir_label = os.path.basename(root)
            if dir_label == 'Benign':
                test7_label.append(0)
                test7_img.append(img_abpath)
            elif dir_label == 'Malignant':
                test7_label.append(1)
                test7_img.append(img_abpath)

    PathDF_test3 = pd.DataFrame({'images': test7_img, 'labels': test7_label})
    PathDF_test3.to_csv("liver_independent_val.csv", index=False)


def resnet_rl_5split_Dataset():
    data_csv = pd.read_csv('/home/uax/LRF/RL_new/RL_HCC_resnet/resnet_train_afp_label.csv')
    patient_id_list=[]
    patient_id_label_list = []
    for i,row_data in data_csv.iterrows():
        id = row_data['images'].split('/')[-1].split('_')[0]
        if id not in patient_id_list:
            patient_id_list.append(id)
            patient_id_label_list.append(row_data['labels'])
    skf = StratifiedKFold(n_splits=5,shuffle=True)

    for Fold_i,(train_index, test_index) in enumerate(skf.split(patient_id_list, patient_id_label_list)):
        patient_id_list_train, patient_id_list_test = [patient_id_list[i] for i in train_index], [patient_id_list[i] for i in test_index]
        #y_train, y_test = y[train_index], y[test_index]
        train_img,train_label = [],[]
        val_img,val_label = [],[]
        for i, row_data in data_csv.iterrows():
            id = row_data['images'].split('/')[-1].split('_')[0]
            if id in patient_id_list_train:
                train_img.append(row_data)

            elif id in patient_id_list_test:
                val_img.append(row_data)

        train_img = pd.concat(train_img,axis=1).transpose()
        val_img = pd.concat(val_img,axis=1).transpose()
        train_img.to_csv("resnet_rl_4classification_train{}.csv".format(Fold_i), index=False)
        val_img.to_csv("resnet_rl_4classification_val{}.csv".format(Fold_i), index=False)


if __name__ == "__main__":
    #path = os.path.abspath(os.path.dirname(__file__))
    # a=os.getcwd()
    #b=os.path.abspath('')
    # new_val_dataset_without_3()
    #process_Liver_5split_Dataset_df_2_classification()
    #data_img = r"/home/gxmdjzx/.pycharm_helpers/pycharm/py_project/Dataset_BUSI_with_GT/"
    # data_dir = "/home/uax/SCY/LiverClassification/dataset/data/HCC/Benign"
    # create_patient_id_Liver_df()
    #test_df = create_Liver_Dataset_df(data_dir)
    #train_df, valid_df = create_Liver_Dataset_df(data_dir)
    #train_df.to_csv("/home/uax/SCY/LiverClassification/dataset/new_train_Liver.csv", index=False)
    #valid_df.to_csv("/home/uax/SCY/LiverClassification/dataset/new_val_Liver.csv", index=False)
    #test_df.to_csv("/home/uax/SCY/LiverClassification/dataset/test_Liver.csv", index=False)
    #check_split_Dataset_df()
    print('down')
    exit()



