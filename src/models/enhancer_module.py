from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MinMetric
import numpy as np
import os
from PIL import Image
# from torchmetrics.classification.accuracy import Accuracy

import torch.nn.functional as F

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    # sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

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

class MaskedBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets, mask):
        """
        :param logits: Tensor of shape (N, 1, H, W) - raw model outputs
        :param targets: Tensor of shape (N, 1, H, W) - binary labels
        :param mask: Tensor of shape (N, 1, H, W) - binary mask (1=foreground, 0=background)
        """
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        foreground = mask.bool()

        # Avoid empty masks by clamping denominator
        foreground_loss = bce_loss[foreground].mean() if foreground.any() else torch.tensor(0.0, device=logits.device)

        return foreground_loss

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets, mask):
        """
        :param logits: Tensor of shape (N, 1, H, W) - raw model outputs
        :param targets: Tensor of shape (N, 1, H, W) - binary labels
        :param mask: Tensor of shape (N, 1, H, W) - binary mask (1=foreground, 0=background)
        """
        mse_loss = F.mse_loss(logits, targets, reduction='none')

        foreground = mask.bool()

        # Avoid empty masks by clamping denominator
        foreground_loss = mse_loss[foreground].mean() if foreground.any() else torch.tensor(0.0, device=logits.device)

        return foreground_loss


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
        output_path: str = None,
        patch_size: Tuple[int, int] = (128, 128),
        use_patches: bool = False,
        stride: int = 8,

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
        self.criterion = nn.BCEWithLogitsLoss()

        self.mse_criterion = torch.nn.functional.mse_loss
        self.bce_criterion = torch.nn.functional.binary_cross_entropy_with_logits


        self.patch_size = patch_size
        self.use_patches = use_patches
        self.stride = stride
        self.input_row = self.patch_size[0]
        self.input_col = self.patch_size[1]
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
        yhat = self.forward(x)

        pred_orig  = yhat[:,0,:,:]
        pred_bin = yhat[:,1,:,:]


        true_orig   = y[:,0,:,:]
        true_bin    = y[:,1,:,:]
        mask        = y[:,2,:,:]
        occ_mask    = y[:,3,:,:]


        masked_bce_criterion = MaskedBCELoss()
        masked_mse_criterion = MaskedMSELoss()
        
        foreground_loss = 0.5*masked_bce_criterion(pred_bin, true_bin, mask*occ_mask)+0.5*masked_mse_criterion(pred_bin, true_bin, mask*occ_mask)
        background_loss = 0.5*masked_bce_criterion(pred_bin, true_bin, (1-mask)) + 0.5*masked_mse_criterion(pred_bin, true_bin, (1-mask))
        occlusion_loss = 0.5*masked_bce_criterion(pred_bin, true_bin, (1-occ_mask)) + 0.5*masked_mse_criterion(pred_bin, true_bin, (1-occ_mask))

        w_occ, w_fg, w_bg = (0.2, 0.4, 0.4)

        total_loss = w_occ*occlusion_loss + w_fg*foreground_loss + w_bg*background_loss 

        # seg_loss_weight = 0.2
        
        # # MSE Loss com máscara
        # mse_loss_ridge = F.mse_loss(pred_orig * mask, true_orig * mask, reduction='sum')
        # mse_loss_ridge = mse_loss_ridge / (mask.sum() + 1e-8)  # média apenas nos pixels com máscara = 1

        # # BCE Loss com máscara
        # bce_loss_ridge = F.binary_cross_entropy_with_logits(
        #     pred_bin, true_bin, weight=mask, reduction='sum'
        # )
        # bce_loss_ridge = bce_loss_ridge / (mask.sum() + 1e-8)  # média apenas nos pixels com máscara = 1

        # mask_seg = 1 - mask
        # # MSE Loss de segmentação
        # mse_loss_seg = F.mse_loss(pred_orig * mask_seg, true_orig * mask_seg, reduction='sum')
        # mse_loss_seg = mse_loss_seg / (mask_seg.sum() + 1e-8)  # média apenas nos pixels com máscara = 1

        # # BCE Loss de segmentação
        # bce_loss_seg = F.binary_cross_entropy_with_logits(
        #     pred_bin, true_bin, weight=mask_seg, reduction='sum'
        # )
        # bce_loss_seg = bce_loss_seg / (mask_seg.sum() + 1e-8)  # média apenas nos pixels com máscara = 1

        # # Total loss ponderada de segmentação
        # total_loss = (1 - seg_loss_weight)* (0.5 * mse_loss_ridge + 0.5 * bce_loss_ridge) + seg_loss_weight*(0.5 * mse_loss_seg + 0.5 * bce_loss_seg)

        # total_loss = 0.5 * self.mse_criterion(pred_orig, true_orig) + 0.5 * self.bce_criterion(pred_bin, true_bin)
        

        # assert(2==1)

        # loss = (self.criterion(pred_bin, true_bin) + dice_loss(F.sigmoid(pred_bin), true_bin, multiclass=False))
        # loss += 0.5 * self.mse_criterion(pred_orig, true_orig)
        # loss = self.mse_criterion(yhat, y_skel,  torch.ones_like(y_skel))

        data  = batch[0]
        names = batch[1]

        # for i, name in enumerate(names):
        #     mnt = mnt_map[i, :, :]

        #     mnt = mnt.cpu().numpy()


        #     mnt = (255 * (mnt - np.min(mnt))/(np.max(mnt) - np.min(mnt))).astype('uint8')

        #     mnt = Image.fromarray(mnt)
        #     # print(name)
        #     mnt.save(self.output_path + '/mnt/' + str(i) + '.png')
        
        return total_loss

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

        # print(data.shape) # (28,1,128,128)
        # print(y) # tupla com todos os nomes das imagens do batch

        gabor_path = os.path.join(self.output_path, "gabor")
        if not os.path.exists(gabor_path):
            os.makedirs(gabor_path)

        bin_path = os.path.join(self.output_path, "bin")
        if not os.path.exists(bin_path):
            os.makedirs(bin_path)

        enh_path = os.path.join(self.output_path, "enh")
        if not os.path.exists(enh_path):
            os.makedirs(enh_path)

        if not self.use_patches:
            latent_en = self.forward(x)
        else:
            shape_latent = data.shape
            ROW = shape_latent[2]
            COL = shape_latent[3]
            row_list_1 = range(self.input_row, ROW+1, self.stride)
            row_list_2 = range(ROW, row_list_1[-1]-1,-self.stride)
            row_list = [*row_list_1, *row_list_2]
            
            col_list_1 = range(self.input_col, COL+1, self.stride)
            col_list_2 = range(COL, col_list_1[-1]-1, -self.stride)
            col_list = [*col_list_1,*col_list_2]

            patch_ind = 0

            latent_en = torch.zeros((data.shape[0], 2, data.shape[2], data.shape[3]), device=x.device)
            
            for row_ind in row_list:
                for col_ind in col_list:
                    patch_pred = self.forward(data[:,:,(row_ind-self.input_row):row_ind,(col_ind-self.input_col):col_ind])
                    latent_en[:,:,(row_ind-self.input_row):row_ind, (col_ind-self.input_col):col_ind] += patch_pred

        for i, name in enumerate(names):
            gabor   = latent_en[i, 1, :, :]
            orig    = latent_en[i, 0, :, :]

            bin   = torch.nn.functional.sigmoid(gabor)
            bin   = torch.round(bin)

            gabor = gabor.cpu().numpy()
            bin   = bin.cpu().numpy()
            orig  = orig.cpu().numpy()

            gabor = (255 * (gabor - np.min(gabor))/(np.max(gabor) - np.min(gabor))).astype('uint8')
            bin   = (255 * (bin - np.min(bin))/(np.max(bin) - np.min(bin))).astype('uint8')
            orig   = (255 * (orig - np.min(orig))/(np.max(orig) - np.min(orig))).astype('uint8')

            gabor = Image.fromarray(gabor)
            gabor.save(gabor_path + '/' + name + '.png')

            bin = Image.fromarray(bin)
            bin.save(bin_path + '/' + name + '.png')

            orig = Image.fromarray(orig)
            orig.save(enh_path + '/' + name + '.png')




if __name__ == "__main__":
    _ = EnhancerLitModule(None, None, None, None)
