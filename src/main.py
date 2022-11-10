import os

import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


# -----------------
# MODEL
# -----------------
class LightningMNISTClassifier(pl.LightningModule):

    def __init__(self, batch_size: int, num_workers: int):
        super(LightningMNISTClassifier, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)

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

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        _, preds = torch.max(logits, 1)
        num_correct = torch.sum(preds == y.data)

        return {
            'loss': loss,
            'num_correct': num_correct
        }

    def training_epoch_end(self, outputs):
        train_loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        train_acc_mean = torch.stack([x['num_correct'] for x in outputs]).sum().float()
        train_acc_mean /= (len(outputs) * self.batch_size)
        self.epoch_train_loss = train_loss_mean
        self.epoch_train_accuracy = train_acc_mean

        self.log('train_loss', train_loss_mean)
        self.log('train_acc', train_acc_mean)
        self.log('step', self.current_epoch)

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        _, preds = torch.max(logits, 1)
        num_correct = torch.sum(preds == y.data)

        return {
            'loss': loss,
            'num_correct': num_correct
        }

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        val_acc_mean = torch.stack([x['num_correct'] for x in outputs]).sum().float()
        val_acc_mean /= (len(outputs) * self.batch_size)
        self.epoch_val_loss = val_loss_mean
        self.epoch_val_accuracy = val_acc_mean

        self.log('val_loss', val_loss_mean)
        self.log('val_acc', val_acc_mean)
        self.log('step', self.current_epoch)

        print(
            f"epoch: {self.current_epoch + 1:02d}/{self.trainer.max_epochs}    "
            f"training loss: {self.epoch_train_loss:.4f}    "
            f"training accuracy: {self.epoch_train_accuracy:.4f}    "
            f"validation loss: {self.epoch_val_loss:.4f}    "
            f"validation accuracy: {self.epoch_val_accuracy:.4f}",
            flush=True,
        )

    def prepare_data(self):
        data_directory = os.path.join(os.getcwd(), 'data')

        # transforms for images
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # prepare transforms standard to MNIST
        mnist_train = MNIST(data_directory, train=True, download=True, transform=transform)
        self.mnist_test = MNIST(data_directory, train=False, download=True, transform=transform)

        self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# Instantiate the PyTorch Lightning model
model = LightningMNISTClassifier(
    batch_size=100,
    num_workers=4
)

# Instantiate the Trainer
trainer = pl.Trainer(
    max_epochs=10,
    progress_bar_refresh_rate=0,
    num_sanity_val_steps=0,
)

# Start training
trainer.fit(model)
