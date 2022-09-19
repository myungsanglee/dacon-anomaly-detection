import argparse
import platform
import pickle
import os
from glob import glob
from numpy import size

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
import torchsummary
import timm
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import albumentations
import albumentations.pytorch
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from utils.yaml_helper import get_configs
from utils.module_select import get_optimizer
from module.lr_scheduler import CosineAnnealingWarmUpRestarts



def get_img_mean_std(img_path_list, input_size):
    imgs = []
    for img_path in img_path_list:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_AREA)

        imgs.append(img)
    
    meanRGB = [np.mean(x, axis=(0,1)) for x in imgs]
    stdRGB = [np.std(x, axis=(0,1)) for x in imgs]
    
    meanR = np.mean([m[0] for m in meanRGB])/255
    meanG = np.mean([m[1] for m in meanRGB])/255
    meanB = np.mean([m[2] for m in meanRGB])/255

    stdR = np.mean([s[0] for s in stdRGB])/255
    stdG = np.mean([s[1] for s in stdRGB])/255
    stdB = np.mean([s[2] for s in stdRGB])/255

    print("평균",meanR, meanG, meanB)
    print("표준편차",stdR, stdG, stdB)
 
    print(f'meanRGB: {meanR:.6f}, {meanG:.6f}, {meanB:.6f}')
    print(f'stdRGB: {stdR:.6f}, {stdG:.6f}, {stdB:.6f}')


class AnomalyDataset(Dataset):
    def __init__(self, dataset_dir, transforms, mode):
        super().__init__()
        self.transforms = transforms
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset/name2label.pkl'), 'rb') as f:
            self.name2label = pickle.load(f)
        self.mode = mode

        if mode == 'train':
            self.img_path_list = sorted(glob(os.path.join(dataset_dir, 'train/*/*/*.png')))

        elif mode == 'valid':
            self.img_path_list = sorted(glob(os.path.join(dataset_dir, 'valid/*/*/*.png')))

        elif mode == 'test':
            self.img_path_list = sorted(glob(os.path.join(dataset_dir, 'test/*.png')))

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        img_file = self.img_path_list[index]
        img = cv2.imread(img_file, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transformed = self.transforms(image=img)['image']
        label = np.zeros((15, 10), dtype=np.float32)

        if self.mode != 'test':
            img_file_list = img_file.split(os.sep)
            class_idx, state_idx = self.name2label[img_file_list[-3]][img_file_list[-2]]
            label[..., 0][class_idx] = 1
            label[class_idx, 1:][state_idx] = 1

        return transformed, label


class AnomalyDataModule(pl.LightningDataModule):
    def __init__(self, dataset_dir, workers, batch_size, input_size):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.workers = workers
        self.batch_size = batch_size
        self.input_size = input_size

    def setup(self, stage=None):
        train_transforms = albumentations.Compose([
            albumentations.Transpose(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.ShiftScaleRotate(p=0.5),
            albumentations.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            albumentations.Resize(self.input_size, self.input_size, cv2.INTER_AREA),
            albumentations.Normalize(mean=[0.433038, 0.403458, 0.394151], std=[0.181572, 0.174035, 0.163234]),
            # albumentations.Normalize(0, 1),
            albumentations.pytorch.ToTensorV2(),
        ],)

        valid_transform = albumentations.Compose([
            albumentations.Resize(self.input_size, self.input_size, cv2.INTER_AREA),
            albumentations.Normalize(mean=[0.433038, 0.403458, 0.394151], std=[0.181572, 0.174035, 0.163234]),
            # albumentations.Normalize(0, 1),
            albumentations.pytorch.ToTensorV2(),
        ],)

        test_transform = albumentations.Compose([
            albumentations.Resize(self.input_size, self.input_size, cv2.INTER_AREA),
            albumentations.Normalize(mean=[0.418256, 0.393101, 0.386632], std=[0.195055, 0.190053, 0.185323]),
            # albumentations.Normalize(0, 1),
            albumentations.pytorch.ToTensorV2(),
        ],)

        self.train_dataset = AnomalyDataset(
            dataset_dir=self.dataset_dir,
            transforms=train_transforms,
            mode='train'
        )

        self.valid_dataset = AnomalyDataset(
            dataset_dir=self.dataset_dir,
            transforms=valid_transform,
            mode='valid'
        )

        self.test_dataset = AnomalyDataset(
            dataset_dir=self.dataset_dir,
            transforms=test_transform,
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

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            persistent_workers=self.workers > 0,
            pin_memory=self.workers > 0
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            persistent_workers=self.workers > 0,
            pin_memory=self.workers > 0
        )


class AnomalyLoss(nn.Module):
    def __init__(self):
        super(AnomalyLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(size_average=False)
        self.bce_loss = nn.BCELoss(size_average=False)
        self.mask = torch.FloatTensor(
            [[1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0]]
        )
        self.state_lambda = 1
        self.no_state_lambda = 0.5

    def forward(self, input, target):
        # encode input, [batch, 150] to [batch, 15, 10]
        pred = input.view(input.size(0), 15, 10)

        # class loss
        pred_class = pred[..., 0] # [batch, 15]
        # pred_class = torch.sigmoid(pred_class)
        class_loss = self.ce_loss(pred_class, target[..., 0])

        # state loss
        pred_state = pred[..., 1:] # [batch, 15, 9]
        pred_state = torch.sigmoid(pred_state)
        if torch.cuda.is_available:
            self.mask = self.mask.cuda()
        pred_state = pred_state * self.mask

        state_mask, no_state_mask = self.get_state_mask(target)
        if torch.cuda.is_available:
            state_mask = state_mask.cuda()
            no_state_mask = no_state_mask.cuda()

        state_loss = self.bce_loss(pred_state*state_mask, target[..., 1:])
        no_state_loss = self.bce_loss(pred_state*no_state_mask, target[..., 1:])

        loss = class_loss + (self.state_lambda*state_loss) + (self.no_state_lambda*no_state_loss)

        return loss

    def get_state_mask(self, target):
        batch_size = target.size(0)
        state_mask = torch.zeros((batch_size, 15, 9), dtype=torch.float32)
        no_state_mask = torch.ones((batch_size, 15, 9), dtype=torch.float32)

        for b in range(batch_size):
            true_class_idx = torch.argmax(target[b, :, 0])

            state_mask[b, true_class_idx, :] = 1
            no_state_mask[b, true_class_idx, :] = 0

        return state_mask, no_state_mask


class AnomalyModule(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore='model')
        self.anomaly_loss = AnomalyLoss()
        self.mask = torch.FloatTensor(
            [[1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0]]
        )
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset/label2name.pkl'), 'rb') as f:
            self.label2name = pickle.load(f)
        
        self.real = []
        self.pred = []

    def forward(self, x):
        pred = self.model(x)
        return self.decoder(pred)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.anomaly_loss(pred, y)

        self.log('train_loss', loss, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.anomaly_loss(pred, y)

        self.pred.extend(self.decoder(pred))
        self.real.extend(self.decoder(y.view(-1, 150)))

        self.log('val_loss', loss, prog_bar=True, logger=True)

    def validation_epoch_end(self, outputs):
        score = self.score_function(self.real, self.pred)
        self.log('val_f1-score', score, prog_bar=True, logger=True)
        self.real = []
        self.pred = []

    def configure_optimizers(self):
        cfg = self.hparams.cfg
        optim = get_optimizer(
            cfg['optimizer'],
            self.model.parameters(),
            **cfg['optimizer_options']
        )
        
        epoch_length = 120
        scheduler = CosineAnnealingWarmUpRestarts(
            optimizer=optim,
            T_0=epoch_length*4,
            T_mult=2,
            eta_max=0.001,
            T_up=epoch_length,
            gamma=0.96
        )

        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": scheduler,
                'interval': 'step'
            }
        } 

    def decoder(self, input):
        result = []

        # encode input, [batch, 150] to [batch, 15, 10]
        batch_size = input.size(0)
        pred = input.view(batch_size, 15, 10)

        # class
        pred_class = pred[..., 0] # [batch, 15]
        pred_class = torch.sigmoid(pred_class)

        # state 
        pred_state = pred[..., 1:] # [batch, 15, 9]
        pred_state = torch.sigmoid(pred_state)
        if torch.cuda.is_available:
            self.mask = self.mask.cuda()
        pred_state = pred_state * self.mask

        for b in range(batch_size):
            class_idx = torch.argmax(pred_class[b])
            state_idx = torch.argmax(pred_state[b, class_idx])

            label_name = self.label2name[int(class_idx)][int(state_idx)]
            result.append(label_name)
        
        return result

    def score_function(self, real, pred):
        score = f1_score(real, pred, average="macro")
        return score 


def train(cfg):
    data_module = AnomalyDataModule(
        dataset_dir=cfg['dataset_dir'],
        workers=cfg['workers'],
        batch_size=cfg['batch_size'],
        input_size=cfg['input_size']
    )

    model = timm.create_model('resnext50_32x4d', True, num_classes=150)
    torchsummary.summary(model, (3, cfg['input_size'], cfg['input_size']), batch_size=1, device='cpu')

    model_module = AnomalyModule(
        model=model,
        cfg=cfg
    )

    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(
            monitor='val_loss', 
            save_last=True, 
            every_n_epochs=cfg['save_freq']
        )
    ]

    trainer = pl.Trainer(
        max_epochs=cfg['epochs'],
        logger=TensorBoardLogger(cfg['save_dir'], 'resnext50_32x4d', default_hp_metric=False),
        accelerator=cfg['accelerator'],
        devices=cfg['devices'],
        plugins=DDPPlugin(find_unused_parameters=False) if platform.system() != 'Windows' else None,
        callbacks=callbacks,
        **cfg['trainer_options']
    )

    trainer.fit(model_module, data_module)


def test(cfg, ckpt_path):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= ','.join(str(num) for num in cfg['devices'])

    data_module = AnomalyDataModule(
        dataset_dir=cfg['dataset_dir'],
        workers=cfg['workers'],
        batch_size=cfg['batch_size'],
        input_size=cfg['input_size']
    )
    data_module.prepare_data()
    data_module.setup()

    '''
    Get Model by pytorch lightning
    '''
    model = timm.create_model('resnext50_32x4d', True, num_classes=150)
    torchsummary.summary(model, (3, cfg['input_size'], cfg['input_size']), batch_size=1, device='cpu')

    if torch.cuda.is_available:
        model = model.to('cuda')

    model_module =AnomalyModule.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        model=model,
        cfg=cfg
    )
    model_module.eval()

    # Inference
    pred_result = []
    for sample in data_module.test_dataloader():
        batch_x, batch_y = sample

        if torch.cuda.is_available:
            batch_x = batch_x.cuda()    
        
        with torch.no_grad():
            pred = model_module(batch_x)

        pred_result.extend(pred)
    print(len(pred_result))

    submission = pd.read_csv('/home/fssv2/myungsang/datasets/dacon_anomaly_detection/sample_submission.csv')

    submission["label"] = pred_result

    submission.to_csv("baseline.csv", index = False)


if __name__ == '__main__':
    '''Get Image Mean & Std'''
    # dataset_dir = '/home/fssv2/myungsang/datasets/dacon_anomaly_detection/total'
    # img_path_list = sorted(glob(os.path.join(dataset_dir, '*/*/*.png')))
    # print(len(img_path_list))
    # get_img_mean_std(img_path_list, 416)


    '''Data Module Test'''
    # dataset_dir = '/home/fssv2/myungsang/datasets/dacon_anomaly_detection/data'
    # data_module = AnomalyDataModule(
    #     dataset_dir=dataset_dir,
    #     workers=1,
    #     batch_size=32,
    #     input_size=416
    # )
    # data_module.prepare_data()
    # data_module.setup()

    # for data in data_module.train_dataloader():
    #     batch_x, batch_y = data
    #     print(f'batch_x.size(): {batch_x.size()}')
    #     print(f'batch_y.size(): {batch_y.size()}')
        
    #     img = batch_x[0].numpy()
    #     img = (np.transpose(img, (1, 2, 0))*255.).astype(np.uint8).copy()
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
    #     cv2.imshow('Train', img)
    #     key = cv2.waitKey(0)
    #     if key == 27:
    #         break

    # cv2.destroyAllWindows()

    '''Train'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str, help='config file')
    args = parser.parse_args()
    cfg = get_configs(args.cfg)

    train(cfg)

    '''Test'''
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--cfg', required=True, type=str, help='config file')
    # parser.add_argument('--ckpt', required=True, type=str, help='ckpt file path')
    # args = parser.parse_args()
    # cfg = get_configs(args.cfg)

    # test(cfg, args.ckpt)