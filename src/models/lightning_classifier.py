import logging
import sys

import pytorch_lightning as pl
import torch
from pytorch_lightning.core import LightningModule
from torch.nn import functional as F
from torchvision import models

import src.utils.tensorboard as tb


class LightningClassifier(LightningModule):
    def __init__(self, batch_size: int, arch: str, pretrained: bool, lr: float, optimizer: str, criterion: str):
        super(LightningClassifier, self).__init__()
        self.batch_size = batch_size
        self.arch = arch
        self.model = models.__dict__[self.arch](pretrained=pretrained)
        self.lr = lr
        self.optimizer = optimizer
        self.criterion = criterion

        # Modify the last fully-connected ResNet architecture layers to be compatible with CIFAR10 data
        self.model.fc = torch.nn.Linear(512, 10)

        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()
        self.val_precision = pl.metrics.Precision(num_classes=10)
        self.val_recall = pl.metrics.Recall(num_classes=10)
        self.val_f1 = pl.metrics.F1(num_classes=10)
        self.confusion_matrix = pl.metrics.ConfusionMatrix(num_classes=10)

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
        return self.model(x)

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
        self.log("metrics/precision", self.val_precision.compute())
        self.log("metrics/recall", self.val_recall.compute())
        self.log("metrics/f1", self.val_f1.compute())
        self.log("step", self.current_epoch)
        preds = torch.cat([x["preds"] for x in outputs])
        targets = torch.cat([x["targets"] for x in outputs])
        # self.logger.experiment.add_pr_curve("pr_curve", targets, preds, self.current_epoch)
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

    def prepare_data(self):
        # Log example training images
        tb.log_image_examples(self.logger.experiment, self.train_dataloader())

        # Log model graph
        self.logger.experiment.add_graph(self.model, torch.rand((1, 3, 224, 224)))

    def configure_optimizers(self):
        return torch.optim.__dict__[self.optimizer](self.parameters(), lr=self.lr)

    def on_train_end(self):
        # Log example image classifications
        tb.log_image_classifications(self.logger.experiment, self.test_dataloader(), self.forward, self.current_epoch)

        # Log hyperparameters
        self.logger.experiment.add_hparams(
            {"learning rate": self.lr, "architecture": self.arch, "criterion": self.criterion, "optimizer": self.optimizer},
            {"hparam/loss": self.epoch_val_loss, "hparam/accuracy": self.epoch_val_accuracy},
        )

        # # Log image projector
        # tb.log_projector(self.logger.experiment, self.train_dataloader(), n=250)
