import os
from datetime import datetime

from test_tube import HyperOptArgumentParser, SlurmCluster

from src.hsearch import main

# Set the experiment name
experiment_name = "arch-lr-hsearch-10k"
nb_trials = 200

# Define the amount of compute
nodes = 1
gpus = 1
cpus_per_task = 4
tasks_per_node = gpus
num_workers = cpus_per_task * gpus

# Set up HyperOptArgumentParser and add job arguments
parser = HyperOptArgumentParser()

# parser.add_argument("--log_path", default=os.path.join(os.getcwd(), 'logs'))
parser.add_argument("--log_path", default="/datasets/work/mlaifsp-objdet/work/coral-ml/logs")
parser.add_argument("--data_dir", default="/datasets/work/oa-mri-images/work/DWBE/data/448x448_cleaned")
parser.add_argument("--experiment_name", default=experiment_name)
parser.add_argument("--num_nodes", default=nodes, type=int)
parser.add_argument("--gpus", default=gpus, type=int)
parser.add_argument("--num_workers", default=num_workers, type=int)
parser.add_argument("--distributed_backend", default=None, type=str)
parser.add_argument("--use_amp", default=True, action="store_true")
parser.add_argument("--arch", default="inception_v3", type=str)
parser.add_argument("--pretrained", default=True, action="store_true")
parser.add_argument("--lr", default=0.00005, type=float)
parser.add_argument("--optimizer", default="Adam", type=str)
parser.add_argument("--criterion", default="cross_entropy", type=str)
parser.add_argument("--max_epochs", default=100, type=int)
parser.add_argument("--batch_size", default=150, type=int)
parser.add_argument("--dataset_limit", default=10000, type=int)
parser.add_argument("--weight_decay", default=0, type=int)

# Add a dummy variable range so that we can generate multiple SlurmCluster nb_trials
parser.opt_range('--dummy_var', default=0.5, type=float, tunable=True, low=0, high=1, nb_samples=nb_trials)

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
# cluster.notify_job_status(email="<email_address>", on_done=True, on_fail=True)

# Submit job to Slurm workload manager
cluster.optimize_parallel_cluster_gpu(
    train_function=main,
    nb_trials=nb_trials,
    job_name=hparams.experiment_name,
    enable_auto_resubmit=False,
    job_display_name=experiment_name
)
