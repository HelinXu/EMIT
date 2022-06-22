'''
Author: HelinXu xuhelin1911@gmail.com
Date: 2022-06-15 22:09:41
LastEditTime: 2022-06-23 03:32:58
Description: Taobao Dataset
'''

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from tqdm import tqdm
from glob import glob
import json
import numpy as np
from os.path import join as pjoin
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from misc import DATA_ROOT, LABEL_1


def tag2label(tag: str) -> np.array:
    label = np.zeros(len(LABEL_1) + 1)
    for l in LABEL_1:
        if l in tag:
            label[LABEL_1.index(l)] = 1
    if label.sum() == 0:
        label[-1] = 1
    # normalize
    label = label / label.sum()
    return label


# define dataset as TaobaoDataset
class TaobaoDataset(Dataset):
    def __init__(self,
                data_root=DATA_ROOT,
                transform=transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]),
                train=True):
        self.data_root = data_root
        self.transform = transform
        self.train = train
        self.data = [] # list of (img, label)

        if self.train:
            for file in tqdm(glob(pjoin(data_root, "train/*/profile.json"))):
                with open(file) as f:
                    jf = json.load(f)
                    for imgs_tags in jf['imgs_tags']:
                        for k, v in imgs_tags.items():
                            img_path = pjoin(data_root, "train", k.split('_')[0], k)
                            self.data.append((img_path, v))
        else: # val
            for file in tqdm(glob(pjoin(data_root, "val/*/profile.json"))):
                with open(file) as f:
                    jf = json.load(f)
                    for imgs_tags in jf['imgs_tags']:
                        for k, v in imgs_tags.items():
                            img_path = pjoin(data_root, "val", k.split('_')[0], k)
                            self.data.append((img_path, v))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx][0]
        img = Image.open(img_path)
        tag = self.data[idx][1]
        label = tag2label(tag)
        if self.transform:
            img = self.transform(img)
        return img, label
