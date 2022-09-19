import math
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
from torch import optim
import albumentations
import albumentations.pytorch
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from utils.yaml_helper import get_configs
from utils.module_select import get_optimizer, get_scheduler
from module.lr_scheduler import CosineAnnealingWarmUpRestarts



class AnomalyDataset(Dataset):
    def __init__(self, img_path_list, label_list, transforms, mode='train'):
        super().__init__()
        self.transforms = transforms
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset/name2label.pkl'), 'rb') as f:
            self.name2label = pickle.load(f)
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.mode = mode

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        img_file = self.img_path_list[index]
        img = cv2.imread(img_file, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transformed = self.transforms(image=img)['image']

        if self.mode != 'test':
            label = np.zeros((15, 10), dtype=np.float32)

            class_key, state_key = self.label_list[index].split('-')
            class_idx, state_idx = self.name2label[class_key][state_key]
            label[..., 0][class_idx] = 1
            label[class_idx, 1:][state_idx] = 1
        
        else:
            label = self.label_list[0]

        return transformed, label


class AnomalyDataModule(pl.LightningDataModule):
    def __init__(self, dataset_dict, workers, batch_size, input_size):
        super().__init__()
        self.dataset_dict = dataset_dict
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

        self.train_dataset = AnomalyDataset(
            img_path_list=self.dataset_dict['train_img_list'],
            label_list=self.dataset_dict['train_label'],
            transforms=train_transforms
        )

        self.valid_dataset = AnomalyDataset(
            img_path_list=self.dataset_dict['val_img_list'],
            label_list=self.dataset_dict['val_label'],
            transforms=valid_transform
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


class AnomalyLoss(nn.Module):
    def __init__(self):
        super(AnomalyLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
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
        no_state_loss = self.bce_loss(pred_state*no_state_mask, target[..., 1:]*no_state_mask)

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
    def __init__(self, model, epoch_length):
        super().__init__()
        self.model = model
        self.epoch_length = epoch_length
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
        # return self.decoder(pred)
        return pred

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
        optimizer = optim.AdamW(
            params=self.model.parameters(), 
            lr=0, 
            weight_decay=1e-5
        )

        scheduler = CosineAnnealingWarmUpRestarts(
            optimizer=optimizer,
            T_0=self.epoch_length*4,
            T_mult=2,
            eta_max=0.001,
            T_up=self.epoch_length,
            gamma=0.9
        )

        return {
            "optimizer": optimizer,
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


def train(cfg, dataset):
    data_module = AnomalyDataModule(
        dataset_dict=dataset,
        workers=cfg['workers'],
        batch_size=cfg['batch_size'],
        input_size=cfg['input_size']
    )

    model = timm.create_model(cfg['model'], True, num_classes=150)
    # torchsummary.summary(model, (3, cfg['input_size'], cfg['input_size']), batch_size=1, device='cpu')

    model_module = AnomalyModule(
        model=model,
        epoch_length=math.ceil(len(dataset['train_img_list'])/cfg['batch_size'])
    )
    
    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(
            monitor='val_loss', 
            save_last=False, 
            # every_n_epochs=cfg['save_freq']
        )
    ]

    trainer = pl.Trainer(
        max_epochs=cfg['epochs'],
        logger=TensorBoardLogger(cfg['save_dir'], cfg['name'], default_hp_metric=False),
        accelerator=cfg['accelerator'],
        devices=cfg['devices'],
        plugins=DDPPlugin(find_unused_parameters=False) if platform.system() != 'Windows' else None,
        callbacks=callbacks,
        # **cfg['trainer_options']
    )

    trainer.fit(model_module, data_module)


def test(cfg, ckpt_path):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= ','.join(str(num) for num in cfg['devices'])

    test_img_list = sorted(glob('/home/fssv2/myungsang/datasets/dacon_anomaly_detection/test/*.png'))
    test_img_list = np.array(test_img_list)
 
    test_transform = albumentations.Compose([
        albumentations.Resize(cfg['input_size'], cfg['input_size'], cv2.INTER_AREA),
        albumentations.Normalize(mean=[0.418256, 0.393101, 0.386632], std=[0.195055, 0.190053, 0.185323]),
        # albumentations.Normalize(0, 1),
        albumentations.pytorch.ToTensorV2(),
    ],)

    test_dataset = AnomalyDataset(
        img_path_list=test_img_list,
        label_list=['test'],
        transforms=test_transform,
        mode='test'
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=cfg['workers'],
        persistent_workers=cfg['workers'] > 0,
        pin_memory=cfg['workers'] > 0
    )

    '''
    Get Model by pytorch lightning
    '''
    model = timm.create_model(cfg['model'], True, num_classes=150)
    # torchsummary.summary(model, (3, cfg['input_size'], cfg['input_size']), batch_size=1, device='cpu')

    if torch.cuda.is_available:
        model = model.to('cuda')

    model_module =AnomalyModule.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        model=model,
        epoch_length=math.ceil(len(test_img_list)/cfg['batch_size'])
    )
    model_module.eval()

    # Inference
    pred_result = 0
    for idx, sample in enumerate(test_loader):
        batch_x, _ = sample

        if torch.cuda.is_available:
            batch_x = batch_x.cuda()    
        
        with torch.no_grad():
            pred = model_module(batch_x)

        if idx == 0:
            pred_result = pred
        else:
            pred_result = torch.cat([pred_result, pred])
    
    return pred_result
    

    


if __name__ == '__main__':
    '''Set cfg'''
    cfg = dict()
    cfg['input_size'] = 384
    cfg['epochs'] = 100
    cfg['dataset_dir'] = '/home/fssv2/myungsang/datasets/dacon_anomaly_detection/total'
    cfg['workers'] = 32
    cfg['batch_size'] = 32
    cfg['save_dir'] = './saved'
    cfg['accelerator'] = 'gpu'
    cfg['devices'] = [1]
    cfg['model'] = 'efficientnet_b4'
    cfg['name'] = 'anomaly_final_06'
    print(f'----cfg setting----\n{cfg}\n')

    ''''Get Data'''
    # Total train/val data
    img_path_list = sorted(glob(cfg['dataset_dir'] + '/*/*/*.png'))
    img_path_list = np.array(img_path_list)

    label_list = []
    for img_path in img_path_list:
        label = img_path.split(os.sep)[-3] + '-' + img_path.split(os.sep)[-2]
        label_list.append(label)
    label_list = np.array(label_list)

    print(f'----Total train/val data num----\n{len(img_path_list)} / {len(label_list)}\n')
    
    '''Set StratifiedKFold'''
    skt = StratifiedKFold(n_splits=5, shuffle=False, random_state=None)

    '''Train 5 Fold'''
    for idx, (train_idx, val_idx) in enumerate(skt.split(img_path_list, label_list)):
        idx += 1
        print(f'\nTraining {idx} Fold\n')
        
        train_img_list = img_path_list[train_idx]
        train_label = label_list[train_idx]
        val_img_list = img_path_list[val_idx]
        val_label = label_list[val_idx]

        dataset = dict()
        dataset['train_img_list'] = train_img_list
        dataset['train_label'] = train_label
        dataset['val_img_list'] = val_img_list
        dataset['val_label'] = val_label

        print(f'----train data num----\n{len(train_img_list)}/{len(train_label)} ')
        print(f'----val data num----\n{len(val_img_list)}/{len(val_label)}\n')
        
        train(cfg, dataset)
        
    '''Test 5 Fold'''
    ckpt_path_list = glob(os.path.join(cfg['save_dir'], cfg['name'], '*/*/*.ckpt'))
    pred_result = 0
    for idx, ckpt_path in enumerate(ckpt_path_list):
        idx += 1
        print(f'\nTesting {idx} Fold\n')

        pred = test(cfg, ckpt_path)
        
        if idx == 0:
            pred_result = pred
        else:
            pred_result += pred
    pred_result /= 5
    
    '''Decode Result'''
    result = []
    mask = torch.FloatTensor(
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
        label2name = pickle.load(f)

    # encode input, [batch, 150] to [batch, 15, 10]
    batch_size = pred_result.size(0)
    pred = pred_result.view(batch_size, 15, 10)

    # class
    pred_class = pred[..., 0] # [batch, 15]
    pred_class = torch.sigmoid(pred_class)

    # state 
    pred_state = pred[..., 1:] # [batch, 15, 9]
    pred_state = torch.sigmoid(pred_state)
    if torch.cuda.is_available:
        mask = mask.cuda()
    pred_state = pred_state * mask

    for b in range(batch_size):
        class_idx = torch.argmax(pred_class[b])
        state_idx = torch.argmax(pred_state[b, class_idx])

        label_name = label2name[int(class_idx)][int(state_idx)]
        result.append(label_name)
    
    submission = pd.read_csv('/home/fssv2/myungsang/datasets/dacon_anomaly_detection/sample_submission.csv')

    submission["label"] = result

    submission.to_csv(f"{cfg['name']}.csv", index = False)
