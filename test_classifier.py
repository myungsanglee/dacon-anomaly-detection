import argparse
from asyncio.log import logger
import platform
import os

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
import torch
import pandas as pd
import numpy as np

from module.classifier import Classifier
from utils.module_select import get_model, get_data_module
from utils.yaml_helper import get_configs


def test(cfg, ckpt_path):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= ','.join(str(num) for num in cfg['devices'])

    data_module = get_data_module(cfg['dataset_name'])(
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
    model = get_model(cfg['model'])(in_channels=cfg['in_channels'], num_classes=cfg['num_classes'])

    if torch.cuda.is_available:
        model = model.to('cuda')

    model_module =Classifier.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        model=model,
        cfg=cfg
    )
    model_module.eval()

    # Inference
    train_y = pd.read_csv(os.path.join(cfg['dataset_dir'], "train_df.csv"))

    train_labels = train_y["label"]

    label_unique = sorted(np.unique(train_labels))
    label_unique = {key:value for key,value in zip(label_unique, range(len(label_unique)))}

    f_pred = []
    for sample in data_module.test_dataloader():
        batch_x, batch_y = sample

        if torch.cuda.is_available:
            batch_x = batch_x.cuda()    
        
        with torch.no_grad():
            pred = model_module(batch_x)
        f_pred.extend(pred.argmax(1).detach().cpu().numpy().tolist())

    label_decoder = {val:key for key, val in label_unique.items()}

    f_result = [label_decoder[result] for result in f_pred]

    submission = pd.read_csv(os.path.join(cfg['dataset_dir'], "sample_submission.csv"))

    submission["label"] = f_result

    submission.to_csv("baseline.csv", index = False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str, help='config file')
    parser.add_argument('--ckpt', required=True, type=str, help='ckpt file path')
    args = parser.parse_args()
    cfg = get_configs(args.cfg)

    test(cfg, args.ckpt)
