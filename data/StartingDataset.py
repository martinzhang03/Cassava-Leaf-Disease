import torch
import pandas as pd
from PIL import Image as img
import os
import cv2
from torchvision import transforms as transformers
import constants

# wow = pd.read_csv("train.csv")
# print(wow)
# print(wow.groupby(["label"]).count())

class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self, csv_path, im_path, training_set=True):
        data = pd.read_csv(csv_path)
        self.image_id = data['image_id']
        self.labels = data['label']
        self.im_path = im_path

        if training_set:
            self.image_id = self.image_id[:constants.TRAIN_NUM]
            self.labels = self.labels[:constants.TRAIN_NUM]
        else:
            self.image_id = self.image_id[constants.TRAIN_NUM: ]
            self.labels = self.labels[constants.TRAIN_NUM: ]




    def __getitem__(self, index):
        id = self.image_id.iloc[index]
        label = torch.tensor(int(self.labels.iloc[index]))

        img_path = constants.IMG_PATH + id

        image = img.open(img_path).resize((800, 600))
        
        return (transformers.ToTensor()(image), label)

    def __len__(self):
        return len(self.labels)
