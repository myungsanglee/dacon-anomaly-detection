import argparse
import platform
import pickle
import os
from glob import glob

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
from utils.module_select import get_optimizer, get_scheduler




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
        img = cv2.imread(img_file)
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
            albumentations.HorizontalFlip(),
            # albumentations.Blur(),
            albumentations.ShiftScaleRotate(),
            # albumentations.GaussNoise(),
            # albumentations.Cutout(max_h_size=int(self.input_size*0.125), max_w_size=int(self.input_size*0.125)),
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
        self.state_lambda = 5
        self.no_state_lambda = 0.5

    def forward(self, input, target):
        # encode input, [batch, 150] to [batch, 15, 10]
        pred = input.view(input.size(0), 15, 10)

        # class loss
        pred_class = pred[..., 0] # [batch, 15]
        pred_class = torch.sigmoid(pred_class)
        class_loss = self.bce_loss(pred_class, target[..., 0])

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

        self.log('train_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.anomaly_loss(pred, y)

        self.pred.extend(self.decoder(pred))
        self.real.extend(self.decoder(y.view(-1, 150)))

        self.log('val_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)

    def validation_epoch_end(self, outputs):
        score = self.score_function(self.real, self.pred)
        self.log('val_f1-score', score)
        self.real = []
        self.pred = []

    def configure_optimizers(self):
        cfg = self.hparams.cfg
        optim = get_optimizer(
            cfg['optimizer'],
            self.model.parameters(),
            **cfg['optimizer_options']
        )
        
        try:
            scheduler = get_scheduler(
                cfg['scheduler'],
                optim,
                **cfg['scheduler_options']
            )
    
            return {
                "optimizer": optim,
                "lr_scheduler": {
                    "scheduler": scheduler
                }
            } 
        
        except KeyError:
            return optim

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

    model = timm.create_model('resnet50', True, num_classes=150)
    torchsummary.summary(model, (3, cfg['input_size'], cfg['input_size']), batch_size=1, device='cpu')

    model_module = AnomalyModule(
        model=model,
        cfg=cfg
    )

    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(
            monitor='val_loss', 
            save_last=True, 
            every_n_epochs=cfg['save_freq']
        )
    ]

    trainer = pl.Trainer(
        max_epochs=cfg['epochs'],
        logger=TensorBoardLogger(cfg['save_dir'], 'resnet50', default_hp_metric=False),
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
    model = timm.create_model('resnet50', True, num_classes=150)
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
    '''Data Module Test'''
    # dataset_dir = '/home/fssv2/myungsang/datasets/dacon_anomaly_detection/data'
    # data_module = AnomalyDataModule(
    #     dataset_dir=dataset_dir,
    #     workers=1,
    #     batch_size=1,
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
        
    #     cv2.imshow('Train', img)
    #     key = cv2.waitKey(0)
    #     if key == 27:
    #         break

    # cv2.destroyAllWindows()

    '''Train'''
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--cfg', required=True, type=str, help='config file')
    # args = parser.parse_args()
    # cfg = get_configs(args.cfg)

    # train(cfg)

    '''Test'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str, help='config file')
    parser.add_argument('--ckpt', required=True, type=str, help='ckpt file path')
    args = parser.parse_args()
    cfg = get_configs(args.cfg)

    test(cfg, args.ckpt)