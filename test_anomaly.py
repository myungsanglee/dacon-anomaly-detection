import argparse
import os
import pickle

import torch
import pandas as pd
import numpy as np
from torchvision.models import efficientnet_b7

from module.anomaly_module import AnomalyModule01, AnomalyModule02
from models.anomaly_model import AnomalyModel01, AnomalyModel02, AnomalyModel03
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
    # model = AnomalyModel01(
    #     backbone=get_model(cfg['model']),
    #     in_channels=cfg['in_channels'],
    #     num_classes=cfg['num_classes'],
    #     input_size=cfg['input_size']
    # )

    # model = AnomalyModel02(
    #     backbone=get_model(cfg['model']),
    #     in_channels=cfg['in_channels'],
    #     num_classes=cfg['num_classes'],
    #     num_state=cfg['num_state'],
    #     input_size=cfg['input_size']
    # )

    model = AnomalyModel03(
        backbone=get_model(cfg['model']),
        in_channels=cfg['in_channels'],
        num_classes=cfg['num_classes'],
        num_state=cfg['num_state'],
        input_size=cfg['input_size']
    )

    if torch.cuda.is_available:
        model = model.to('cuda')

    # model_module =AnomalyModule01.load_from_checkpoint(
    #     checkpoint_path=ckpt_path,
    #     model=model,
    #     cfg=cfg
    # )
    # model_module.eval()

    model_module =AnomalyModule02.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        model=model,
        cfg=cfg
    )
    model_module.eval()

    # Inference
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset/label2name.pkl'), 'rb') as f:
            label2name = pickle.load(f)

    pred_class_list = []
    pred_state_list = []
    for sample in data_module.test_dataloader():
        batch_x, batch_y = sample

        if torch.cuda.is_available:
            batch_x = batch_x.cuda()    
        
        with torch.no_grad():
            pred_class, pred_state = model_module(batch_x)

        pred_class_list.extend(pred_class.argmax(1).detach().cpu().numpy().tolist())
        pred_state_list.extend(pred_state.argsort(1, True).detach().cpu().numpy().tolist())

    # print(pred_class_list)
    # print('')
    # print(pred_state_list)

    # pred_result = [label2name[class_idx][state_idx] for class_idx, state_idx in zip(pred_class_list, pred_state_list)]
    pred_result = []
    count = 0
    for class_idx, state_idx_list in zip(pred_class_list, pred_state_list):
        try:
            result = label2name[class_idx][state_idx_list[0]]
            pred_result.append(result)
        except KeyError:
            count += 1

            break_flag = False
            for state_idx in state_idx_list:
                for key in label2name[class_idx].keys():
                    if key == state_idx:
                        result = label2name[class_idx][key]
                        pred_result.append(result)
                        break_flag = True
                        break
                
                if break_flag:
                    break

            # print(f'label: {label2name[class_idx]} \npred_key: {state_idx_list[:5]} \nkeys: {label2name[class_idx].keys()}\n')
            # pass
            # try:
            #     result = label2name[class_idx][state_idx_list[1]]
            #     pred_result.append(result)
            # except KeyError:
            #     result = label2name[class_idx][state_idx_list[2]]
            #     pred_result.append(result)

    # print(pred_result)
    print(count)

    submission = pd.read_csv('/home/fssv2/myungsang/datasets/dacon_anomaly_detection/sample_submission.csv')

    submission["label"] = pred_result

    submission.to_csv("baseline.csv", index = False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str, help='config file')
    parser.add_argument('--ckpt', required=True, type=str, help='ckpt file path')
    args = parser.parse_args()
    cfg = get_configs(args.cfg)

    test(cfg, args.ckpt)
