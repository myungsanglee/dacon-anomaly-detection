import os
import sys  
from glob import glob
sys.path.append(os.getcwd())

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import albumentations
import albumentations.pytorch
import cv2
import numpy as np
import pandas as pd


class DaconAnomalyDataset(Dataset):
    def __init__(self, dataset_dir, transforms, mode):
        super().__init__()
        self.transforms = transforms

        if mode == 'train':
            self.img_path_list = sorted(glob(os.path.join(dataset_dir, 'train/*.png')))
            
            self.labels = pd.read_csv(os.path.join(dataset_dir, 'train_df.csv'))['label']

            label_unique = sorted(np.unique(self.labels))
            label_unique = {key:value for key,value in zip(label_unique, range(len(label_unique)))}

            self.labels = [label_unique[k] for k in self.labels]

        elif mode == 'test':
            self.img_path_list = sorted(glob(os.path.join(dataset_dir, 'test/*.png')))
            self.labels = ["tmp"]*len(self.img_path_list)

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        img_file = self.img_path_list[index]
        img = cv2.imread(img_file)
        transformed = self.transforms(image=img)['image']

        label = self.labels[index]

        return transformed, label


class DaconAnomaly(pl.LightningDataModule):
    def __init__(self, dataset_dir, workers, batch_size, input_size):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.workers = workers
        self.batch_size = batch_size
        self.input_size = input_size

    def setup(self, stage=None):
        train_transforms = albumentations.Compose([
            albumentations.HorizontalFlip(),
            albumentations.Blur(),
            albumentations.ShiftScaleRotate(),
            albumentations.GaussNoise(),
            albumentations.Cutout(max_h_size=int(self.input_size*0.125), max_w_size=int(self.input_size*0.125)),
            # albumentations.ElasticTransform(),
            albumentations.RandomResizedCrop(self.input_size, self.input_size, (0.8, 1)),
            albumentations.Normalize(0, 1),
            albumentations.pytorch.ToTensorV2(),
        ],)

        valid_transform = albumentations.Compose([
            albumentations.Resize(self.input_size, self.input_size, always_apply=True),
            albumentations.Normalize(0, 1),
            albumentations.pytorch.ToTensorV2(),
        ],)

        self.train_dataset = DaconAnomalyDataset(
            dataset_dir=self.dataset_dir,
            transforms=train_transforms,
            mode='train'
        )

        # self.valid_dataset = DaconAnomalyDataset(
        #     dataset_dir=self.dataset_dir,
        #     transforms=valid_transform,
        #     is_train=False
        # )

        self.test_dataset = DaconAnomalyDataset(
            dataset_dir=self.dataset_dir,
            transforms=valid_transform,
            mode='test'
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            persistent_workers=self.workers > 0,
            pin_memory=self.workers > 0
        )

    # def val_dataloader(self):
    #     return DataLoader(
    #         self.valid_dataset,
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #         num_workers=self.workers,
    #         persistent_workers=self.workers > 0,
    #         pin_memory=self.workers > 0
    #     )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            persistent_workers=self.workers > 0,
            pin_memory=self.workers > 0
        )


if __name__ == '__main__':
    input_size = 64

    train_transform = albumentations.Compose([
        albumentations.HorizontalFlip(),
        albumentations.Blur(),
        albumentations.ShiftScaleRotate(),
        albumentations.GaussNoise(),
        albumentations.Cutout(max_h_size=int(input_size*0.125), max_w_size=int(input_size*0.125)),
        # albumentations.ElasticTransform(),
        albumentations.RandomResizedCrop(input_size, input_size, (0.8, 1)),
        albumentations.Normalize(0, 1),
        albumentations.pytorch.ToTensorV2(),
    ],)

    origin_transform = albumentations.Compose([
        albumentations.Resize(input_size, input_size, always_apply=True),
        albumentations.Normalize(0, 1),
        albumentations.pytorch.ToTensorV2(),
    ],)

    train_loader = DataLoader(
        DaconAnomalyDataset(
            dataset_dir='/home/fssv2/myungsang/datasets/dacon_anomaly_detection',
            transforms=train_transform,
            is_train=True
        ),
        batch_size=1,
        shuffle=False
    )

    # label_name_path = '/home/fssv2/myungsang/datasets/dacon_anomaly_detection/dacon_anomaly.names'
    # with open(label_name_path, 'r') as f:
    #     label_name_list = f.read().splitlines()

    for train_sample in train_loader:
        train_x, train_y = train_sample

        img = train_x[0].numpy()   
        img = (np.transpose(img, (1, 2, 0))*255.).astype(np.uint8).copy()

        print(train_y)
        # print(f'label: {label_name_list[train_y[0]]}\n')

        cv2.imshow('Train', img)
        key = cv2.waitKey(0)
        if key == 27:
            break

    cv2.destroyAllWindows()
