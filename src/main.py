import os

from pl_bolts.datamodules.cifar10_datamodule import CIFAR10DataModule
from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.plugins import DDPPlugin

from src.models.lightning_classifier import LightningClassifier


def main(hparams, *args):

    # Recalculate batch_size and num_workers for DDP to split across the total number of GPUs
    if hparams.accelerator == "ddp":
        hparams.batch_size = int(hparams.batch_size / max(1, hparams.gpus))
        hparams.num_workers = int(hparams.num_workers / max(1, hparams.gpus))

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
        pretrained=hparams.pretrained,
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
        num_nodes=hparams.num_nodes,
        gpus=hparams.gpus,
        accelerator=hparams.accelerator,
        max_epochs=hparams.max_epochs,
        progress_bar_refresh_rate=0,
        precision=16 if hparams.use_amp else 32,
        # plugins=DDPPlugin(find_unused_parameters=False),
        weights_summary="full",
        profiler="simple",
        num_sanity_val_steps=0,
        limit_train_batches=1,
        limit_val_batches=1,
        # fast_dev_run=True,
        # overfit_batches=10,
    )

    # Start training
    trainer.fit(model, datamodule=cifar10)
