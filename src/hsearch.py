import os

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.trial import TrialState
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger

from pl_bolts.datamodules.cifar10_datamodule import CIFAR10DataModule
from src.models.lightning_classifier import LightningClassifier


def objective(trial, hparams):

    # Define the variables for hyperparameter optimisation
    hparams.lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    # hparams.optimizer = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    # hparams.criterion = trial.suggest_categorical("criterion", ["cross_entropy", "nll_loss"])
    # hparams.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)

    # Instantiate the data module
    cifar10 = CIFAR10DataModule(
        data_dir=hparams.data_dir,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
    )

    # Instantiate the model
    model = LightningClassifier(
        batch_size=hparams.batch_size,
        arch=hparams.arch,
        pretrained=hparams.pretrained,
        lr=hparams.lr,
        optimizer=hparams.optimizer,
        criterion=hparams.criterion,
    )

    # Set custom logger
    log_sub_dir = "{}_{}_{:f}_{}".format(hparams.arch.lower(), hparams.optimizer.lower(), round(hparams.lr, 8), str(trial.number).zfill(3))
    logger = TestTubeLogger(
        name=log_sub_dir,
        save_dir=os.path.join(hparams.log_path, hparams.experiment_name),
    )

    # Initialise the Trainer
    trainer = Trainer(
        logger=logger,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="accuracy/val_accuracy")],
        default_root_dir=os.path.join(hparams.log_path, hparams.experiment_name, log_sub_dir),
        num_nodes=hparams.num_nodes,
        gpus=hparams.gpus,
        distributed_backend=hparams.distributed_backend,
        max_epochs=hparams.max_epochs,
        auto_scale_batch_size="binsearch",
        stochastic_weight_avg=True,
        progress_bar_refresh_rate=0,
        precision=16 if hparams.use_amp else 32,
        weights_summary=None,
        # profiler="simple",
        num_sanity_val_steps=0,
        limit_train_batches=1,
        limit_val_batches=1,
        # fast_dev_run=True,
        # overfit_batches=10,
    )

    # Start training
    trainer.fit(model, cifar10)

    return trainer.callback_metrics["accuracy/val_accuracy"].item()


def main(hparams, *args):

    # TODO: Need to update IP address of Postgres server for every hyperparameter optimisation run
    postgres_ip = "10.149.1.110"
    study = optuna.load_study(
        study_name=hparams.experiment_name,
        storage=f"postgresql://jac249@{postgres_ip}:5432/postgres",
    )
    study.optimize(lambda trial: objective(trial, hparams), n_trials=1)

    print("Number of finished trials: {}".format(len(study.trials)))

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print(optuna.importance.get_param_importances(study))
