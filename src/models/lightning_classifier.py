import logging
import sys

import torch
import torchmetrics
from pytorch_lightning.core import LightningModule
from torch.nn import functional as F

import src.utils.tensorboard as tb


class LightningClassifier(LightningModule):
    def __init__(self, batch_size: int, arch: str, lr: float, optimizer: str, criterion: str):
        super(LightningClassifier, self).__init__()
        self.batch_size = batch_size
        self.arch = arch
        self.lr = lr
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_classes = 10

        # CIFAR10 images are (3, 32, 32) (channels, width, height)
        self.layer_1 = torch.nn.Linear(3 * 32 * 32, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, self.num_classes)

        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.val_precision = torchmetrics.Precision(average='macro', num_classes=self.num_classes)
        self.val_recall = torchmetrics.Recall(average='macro', num_classes=self.num_classes)
        self.val_f1 = torchmetrics.F1(average='macro', num_classes=self.num_classes)
        self.confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=self.num_classes)

        self.logging = logging.getLogger(__name__)
        self.logging.setLevel(logging.DEBUG)
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] [%(name)s] %(message)s")
        sh.setFormatter(formatter)
        self.logging.addHandler(sh)

        self.epoch_train_loss = 0.0
        self.epoch_train_accuracy = 0.0
        self.epoch_val_loss = 0.0
        self.epoch_val_accuracy = 0.0

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # (b, 3, 32, 32) -> (b, 3*32*32)
        x = x.view(batch_size, -1)

        # layer 1 (b, 3*32*32) -> (b, 128)
        x = self.layer_1(x)
        x = torch.relu(x)

        # layer 2 (b, 128) -> (b, 256)
        x = self.layer_2(x)
        x = torch.relu(x)

        # layer 3 (b, 256) -> (b, 10)
        x = self.layer_3(x)

        # probability distribution over labels
        x = torch.log_softmax(x, dim=1)

        return x

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = F.__dict__[self.criterion](logits, y)
        _, preds = torch.max(logits, 1)
        self.train_accuracy(preds, y)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        # Log epoch loss and accuracy
        self.epoch_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.epoch_train_accuracy = self.train_accuracy.compute()
        self.log("loss/train_loss", self.epoch_train_loss, sync_dist=True)
        self.log("accuracy/train_accuracy", self.epoch_train_accuracy)
        self.log("overfit/loss", {"train": self.epoch_train_loss}, sync_dist=True)
        self.log("overfit/accuracy", {"train": self.epoch_train_accuracy}, sync_dist=True)
        self.log("step", self.current_epoch)

        # Log distributions and histograms for model weights and biases
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = F.__dict__[self.criterion](logits, y)
        _, preds = torch.max(logits, 1)
        self.val_accuracy(preds, y)
        self.val_precision(preds, y)
        self.val_recall(preds, y)
        self.val_f1(preds, y)
        return {"loss": loss, "preds": preds, "targets": y}

    def validation_epoch_end(self, outputs):
        # Log epoch loss, accuracy and validation metrics
        self.epoch_val_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.epoch_val_accuracy = self.val_accuracy.compute()
        self.log("loss/val_loss", self.epoch_val_loss, sync_dist=True)
        self.log("accuracy/val_accuracy", self.epoch_val_accuracy)
        self.log("overfit/loss", {"val": self.epoch_val_loss}, sync_dist=True)
        self.log("overfit/accuracy", {"val": self.epoch_val_accuracy}, sync_dist=True)
        self.log("metrics/precision", self.val_precision.compute())
        self.log("metrics/recall", self.val_recall.compute())
        self.log("metrics/f1", self.val_f1.compute())
        self.log("step", self.current_epoch)

        preds = torch.cat([x["preds"] for x in outputs])
        targets = torch.cat([x["targets"] for x in outputs])
        self.logger.experiment.add_pr_curve("pr_curve", targets, preds, self.current_epoch)
        tb.log_confusion_matrix(self.logger.experiment, self.confusion_matrix(preds, targets), self.current_epoch)

        # Log the learning rate
        self.log("learning_rate", self.lr)
        self.log("step", self.current_epoch)

        # Log epoch stats to console
        self.logging.info(
            f"epoch: {self.current_epoch + 1:03d}/{self.trainer.max_epochs:03d}    "
            f"training loss: {self.epoch_train_loss:.4f}    "
            f"training accuracy: {self.epoch_train_accuracy:.4f}    "
            f"validation loss: {self.epoch_val_loss:.4f}    "
            f"validation accuracy: {self.epoch_val_accuracy:.4f}"
        )

    def setup(self, stage):
        # Log example training images
        tb.log_image_examples(self.logger.experiment, self.train_dataloader())

        # Log model graph
        self.logger.experiment.add_graph(self, torch.rand((1, 3, 32, 32)))

    def configure_optimizers(self):
        return torch.optim.__dict__[self.optimizer](self.parameters(), lr=self.lr)

    def on_train_end(self):
        # Log example image classifications
        tb.log_image_classifications(self.logger.experiment, self.val_dataloader(), self.forward, self.current_epoch)

        # Log hyperparameters
        self.logger.experiment.add_hparams(
            {"learning rate": self.lr, "architecture": self.arch, "criterion": self.criterion, "optimizer": self.optimizer},
            {"hparam/loss": self.epoch_val_loss, "hparam/accuracy": self.epoch_val_accuracy},
        )

        # # Log image projector
        # tb.log_projector(self.logger.experiment, self.train_dataloader(), n=250)
