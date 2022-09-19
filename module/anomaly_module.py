import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics import Accuracy

from utils.module_select import get_optimizer, get_scheduler


class AnomalyModule01(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore='model')
        self.top_1 = Accuracy(top_k=1)
        self.top_5 = Accuracy(top_k=5)

    def forward(self, x):
        predictions = self.model(x)
        return predictions

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        loss = F.cross_entropy(y_pred, y)

        self.log('train_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log('train_top1', self.top_1(y_pred, y), logger=True, on_epoch=True, on_step=False)
        self.log('train_top5', self.top_5(y_pred, y), logger=True, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        loss = F.cross_entropy(y_pred, y)
        
        self.log('val_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log('val_top1', self.top_1(y_pred, y), logger=True, on_epoch=True, on_step=False)
        self.log('val_top5', self.top_5(y_pred, y), logger=True, on_epoch=True, on_step=False)
    
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


class AnomalyModule02(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore='model')
        self.class_acc = Accuracy(top_k=1)
        self.state_acc = Accuracy(top_k=1)

    def forward(self, x):
        predictions = self.model(x)
        return predictions

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred_class, pred_state = self.model(x)
        class_loss = F.cross_entropy(pred_class, y[0])
        state_loss = F.cross_entropy(pred_state, y[1])
        loss = class_loss + state_loss

        self.log('train_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log('train_class_loss', class_loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log('train_state_loss', state_loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log('train_class_top1', self.class_acc(pred_class, y[0]), logger=True, on_epoch=True, on_step=False)
        self.log('train_state_top1', self.state_acc(pred_state, y[1]), logger=True, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred_class, pred_state = self.model(x)
        class_loss = F.cross_entropy(pred_class, y[0])
        state_loss = F.cross_entropy(pred_state, y[1])
        loss = class_loss + state_loss
        
        self.log('val_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log('val_class_loss', class_loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log('val_state_loss', state_loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log('val_class_top1', self.class_acc(pred_class, y[0]), logger=True, on_epoch=True, on_step=False)
        self.log('val_state_top1', self.state_acc(pred_state, y[1]), logger=True, on_epoch=True, on_step=False)
    
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