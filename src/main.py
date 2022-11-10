import os

from pl_bolts.datamodules.cifar10_datamodule import CIFAR10DataModule
from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger

from src.models.lightning_classifier import LightningClassifier


def main(hparams, *args):
    # Instantiate data modules
    data_dir = os.path.join(os.getcwd(), "data")
    mnist = MNISTDataModule(
        data_dir=data_dir,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
    )
    cifar10 = CIFAR10DataModule(
        data_dir=data_dir,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
    )

    # Instantiate the PyTorch Lightning model
    model = LightningClassifier(
        batch_size=hparams.batch_size,
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
    trainer.fit(model, datamodule=cifar10)
