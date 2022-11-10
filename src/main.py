import logging
import os
import sys

import torch
import torchmetrics
from pytorch_lightning import Trainer
from pytorch_lightning.core import LightningModule
from pytorch_lightning.loggers import TestTubeLogger
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

import src.utils.tensorboard as tb


class LightningMNISTClassifier(LightningModule):
    def __init__(self, batch_size: int, num_workers: int, arch: str, lr: float, optimizer: str, criterion: str):
        super(LightningMNISTClassifier, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.arch = arch
        self.lr = lr
        self.optimizer = optimizer
        self.criterion = criterion

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)

        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.val_precision = torchmetrics.Precision(num_classes=10)
        self.val_recall = torchmetrics.Recall(num_classes=10)
        self.val_f1 = torchmetrics.F1(num_classes=10)
        self.confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=10)

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

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)

        # layer 1 (b, 1*28*28) -> (b, 128)
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
        data_directory = os.path.join(os.getcwd(), "data")

        # transforms for images
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        # prepare transforms standard to MNIST
        mnist_train = MNIST(data_directory, train=True, download=True, transform=transform)
        mnist_test = MNIST(data_directory, train=False, download=True, transform=transform)

        mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])

        # assign to use in dataloaders
        self.mnist_train = mnist_train
        self.mnist_val = mnist_val
        self.mnist_test = mnist_test

        # Log example training images
        tb.log_image_examples(self.logger.experiment, self.mnist_train)

        # Log model graph
        self.logger.experiment.add_graph(self, torch.rand((1, 1, 28, 28)))

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

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

        # Log image projector
        tb.log_projector(self.logger.experiment, self.mnist_train, n=250)


def main(hparams, *args):
    # Instantiate the PyTorch Lightning model
    model = LightningMNISTClassifier(
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        arch=hparams.arch,
        lr=hparams.lr,
        optimizer=hparams.optimizer,
        criterion=hparams.criterion,
    )

    # Set custom logger
    logger = TestTubeLogger(
        name=hparams.experiment_name,
        save_dir=hparams.log_path,
    )

    # Instantiate the Trainer
    trainer = Trainer(
        logger=logger,
        default_root_dir=os.path.join(hparams.log_path, hparams.experiment_name),
        max_epochs=hparams.max_epochs,
        progress_bar_refresh_rate=0,
        weights_summary="full",
        profiler="simple",
        num_sanity_val_steps=0,
    )

    # Start training
    trainer.fit(model)
