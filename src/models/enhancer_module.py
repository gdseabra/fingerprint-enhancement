from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MinMetric
import numpy as np
from PIL import Image
# from torchmetrics.classification.accuracy import Accuracy

import torch.nn.functional as F


def bce_loss(pred, target, mask_label, mnt_label):
    bce_criterion = nn.functional.l1_loss
    image_loss = bce_criterion(pred, target, reduction = 'none')

    minutia_weighted_map = mnt_label
    image_loss *= minutia_weighted_map
    # image_loss = image_loss * mask_label
    # return torch.sum(image_loss) / (torch.sum(mask_label).clamp(min=1) + 1e-7)

    return torch.mean(torch.sum(image_loss, dim=(1,2)))

class MyCriterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, pixel_weight, mask):
        enh_loss = bce_loss(input, target, pixel_weight, mask)

        return enh_loss

class MyWeightedL1Loss(nn.L1Loss):
    def __init__(self, reduction='none'):
        super(MyWeightedL1Loss, self).__init__(reduction=reduction)

    def forward(self, input, target):
        pixel_mae = super(MyWeightedL1Loss, self).forward(input, target)
        loss = pixel_mae
        return loss.sum()/(loss.size(0)) # mean per-image loss (not per-pixel or per-batch).



class EnhancerLitModule(LightningModule):
    """
    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        output_path: str = None

    ) -> None:
        """Initialize a `EnhancerLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = MyWeightedL1Loss()


        # metric objects for calculating and averaging accuracy across batches
        # self.train_acc = Accuracy(task="multiclass", num_classes=10)
        # self.val_acc = Accuracy(task="multiclass", num_classes=10)
        # self.test_acc = Accuracy(task="multiclass", num_classes=10)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation loss
        self.val_loss_best = MinMetric()

        self.output_path = output_path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        # self.val_acc.reset()
        self.val_loss_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch
        yhat = self.forward(x)[:,0,:,:]

        y_orig = y[:,0,:,:]
        # y_skel = y[:,1,:,:]
        # mask = y[:,2,:,:]
        # mnt_map = y[:,3,:,:]


        loss = self.criterion(yhat, y_orig)

        # loss = self.mse_criterion(yhat, y_skel,  torch.ones_like(y_skel))

        data  = batch[0]
        names = batch[1]
        x, y = batch
        yhat = self.forward(x)

        # for i, name in enumerate(names):
        #     mnt = mnt_map[i, :, :]

        #     mnt = mnt.cpu().numpy()


        #     mnt = (255 * (mnt - np.min(mnt))/(np.max(mnt) - np.min(mnt))).astype('uint8')

        #     mnt = Image.fromarray(mnt)
        #     # print(name)
        #     mnt.save(self.output_path + '/mnt/' + str(i) + '.png')
        
        return loss

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        loss = self.val_loss.compute()  # get current val acc
        self.val_loss_best(loss)  # update best so far val acc
        # log `val_loss_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        # self.test_acc(preds, targets)
        self.log(
            "test/loss",
            self.test_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        # self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single prediction step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        data  = batch[0]
        names = batch[1]
        x, y = batch
        yhat = self.forward(x)

        for i, name in enumerate(names):
            skel = yhat[i, 0, :, :]

            skel = skel.cpu().numpy()


            skel = (255 * (skel - np.min(skel))/(np.max(skel) - np.min(skel))).astype('uint8')

            skel = Image.fromarray(skel)
            skel.save(self.output_path + '/skel/' + name + '.png')

        
        
        # return yhat
        # for i, name in enumerate(names):
        #     gabor = yhat[i, 1, :, :]
        #     bin   = torch.nn.functional.sigmoid(gabor)
        #     bin   = torch.round(bin)

        #     gabor = gabor.cpu().numpy()
        #     bin   = bin.cpu().numpy()


        #     gabor = (255 * (gabor - np.min(gabor))/(np.max(gabor) - np.min(gabor))).astype('uint8')
        #     bin   = (255 * (bin - np.min(bin))/(np.max(bin) - np.min(bin))).astype('uint8')

        #     gabor = Image.fromarray(gabor)
        #     gabor.save(self.output_path + '/gabor/' + name + '.png')

        #     bin = Image.fromarray(bin)
        #     bin.save(self.output_path + '/bin/' + name + '.png')



if __name__ == "__main__":
    _ = EnhancerLitModule(None, None, None, None)
