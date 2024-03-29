from torch import optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts
from module.lr_scheduler import CosineAnnealingWarmUpRestarts
import timm

from models.backbone.vgg import vgg16, vgg16_bn
from models.backbone.darknet import darknet19
from dataset.anomaly_dataset import AnomalyDataModule


def get_model(model_name):
    model_dict = {
        'vgg16': vgg16,
        'vgg16_bn': vgg16_bn,
        'darknet19': darknet19,
        'resnet50': timm.create_model('resnet50', True, num_classes=0, global_pool='')
    }
    return model_dict.get(model_name)


def get_data_module(dataset_name):
    dataset_dict = {
        'dacon-anomaly': AnomalyDataModule
    }
    return dataset_dict.get(dataset_name)


def get_optimizer(optimizer_name, params, **kwargs):
    optim_dict = {
        'sgd': optim.SGD, 
        'adam': optim.Adam,
        'radam': optim.RAdam,
        'adamw': optim.AdamW
    }
    optimizer = optim_dict.get(optimizer_name)
    if optimizer:
        return optimizer(params, **kwargs)


def get_scheduler(scheduler_name, optim, **kwargs):
    scheduler_dict = {
        'multi_step': MultiStepLR, 
        'cosine_annealing_warm_restarts': CosineAnnealingWarmRestarts,
        'cosine_annealing_warm_up_restarts': CosineAnnealingWarmUpRestarts
    }
    optimizer = scheduler_dict.get(scheduler_name)
    if optimizer:
        return optimizer(optim, **kwargs)
