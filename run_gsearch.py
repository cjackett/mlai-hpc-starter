import os
from datetime import datetime

from test_tube import HyperOptArgumentParser, SlurmCluster

from src.hsearch import main

# Set the experiment name
experiment_name = "arch-gsearch-10k"
# experiment_name = "optimiser-gsearch-10k"
# experiment_name = "criteria-gsearch-10k"

# Define the amount of compute
nodes = 1
gpus = 3
cpus_per_task = 4
tasks_per_node = gpus
num_workers = cpus_per_task * gpus

# Set up HyperOptArgumentParser and add job arguments
parser = HyperOptArgumentParser()

parser.add_argument("--log_path", default="/datasets/work/mlaifsp-objdet/work/coral-ml/logs")
parser.add_argument("--data_dir", default="/datasets/work/oa-mri-images/work/DWBE/data/448x448_cleaned")
parser.add_argument("--experiment_name", default=experiment_name)
parser.add_argument("--num_nodes", default=nodes, type=int)
parser.add_argument("--gpus", default=gpus, type=int)
parser.add_argument("--num_workers", default=num_workers, type=int)
parser.add_argument("--distributed_backend", default="ddp", type=str, choices=("dp", "ddp", "ddp2"))
parser.add_argument("--use_amp", default=True, action="store_true")
parser.add_argument("--pretrained", default=True, action="store_true")
parser.add_argument("--lr", default=0.0001, type=float)
parser.add_argument("--max_epochs", default=100, type=int)
parser.add_argument("--batch_size", default=300, type=int)
parser.add_argument("--dataset_limit", default=10000, type=int)
parser.add_argument("--weight_decay", default=0, type=int)

# parser.add_argument("--arch", default="inception_v3")
parser.opt_list("--arch", options=[
    "alexnet",
    "densenet121",
    # "densenet161",
    # "densenet169",
    # "densenet201",
    "googlenet",
    "inception_v3",
    # "mnasnet0_5",
    # "mnasnet1_0",
    "mobilenet_v2",
    # "mobilenet_v3_large",
    "mobilenet_v3_small",
    "resnet101",
    # "resnet152",
    "resnet18",
    # "resnet34",
    "resnet50",
    # "resnext101_32x8d",
    "resnext50_32x4d",
    # "shufflenet_v2_x0_5",
    "shufflenet_v2_x1_0",
    # "squeezenet1_0",
    "squeezenet1_1",
    "vgg11",
    "vgg11_bn",
    # "vgg13",
    # "vgg13_bn",
    # "vgg16",
    # "vgg16_bn",
    # "vgg19",
    # "vgg19_bn",
    # "wide_resnet101_2",
    "wide_resnet50_2",
], tunable=True)

parser.add_argument("--optimizer", default="Adam", type=str, choices=("Adam", "SGD", "RMSprop"))
# parser.opt_list(
#     "--optimizer",
#     options=[
#         "Adam",
#         "RMSprop",
#         "SGD",
#     ],
#     tunable=True,
# )

parser.add_argument("--criterion", default="cross_entropy", type=str, choices=("cross_entropy", "nll_loss"))
# parser.opt_list(
#     "--criterion",
#     options=[
#         "cross_entropy",
#         "nll_loss",
#     ],
#     tunable=True,
# )

hparams = parser.parse_args()

# Enable cluster training
cluster = SlurmCluster(
    hyperparam_optimizer=hparams,
    log_path=hparams.log_path,
    python_cmd="python3",
)

# Source local Python environment
cluster.add_command("source env/bin/activate")

# Set job compute details - this will apply per set of hyper-parameters
cluster.per_experiment_nb_nodes = nodes
cluster.per_experiment_nb_cpus = cpus_per_task
cluster.per_experiment_nb_gpus = gpus
cluster.memory_mb_per_node = "24g"
cluster.job_time = "0-01:59:59"

# Add tasks-per-node to Slurm script
cluster.add_slurm_cmd(cmd="tasks-per-node", value=tasks_per_node, comment="tasks per node")

# Email results
# cluster.notify_job_status(email="chris.jackett@csiro.au", on_done=True, on_fail=True)

# Submit job to Slurm workload manager
cluster.optimize_parallel_cluster_gpu(
    train_function=main,
    nb_trials=15,
    job_name=hparams.experiment_name,
    enable_auto_resubmit=False,
    job_display_name=experiment_name,
)
