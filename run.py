import os
from datetime import datetime

from test_tube import HyperOptArgumentParser, SlurmCluster

from src.main import main

# Set the experiment name
experiment_name = 'run'

# Define the amount of compute
nodes = 1
gpus = 0
tasks_per_node = 1
cpus_per_task = 5

# Set up HyperOptArgumentParser and add job arguments
parser = HyperOptArgumentParser()

parser.add_argument("--log_path", default=os.path.join(os.getcwd(), 'logs'))
parser.add_argument("--experiment_name", default=f"{datetime.now(tz=None).strftime('%Y-%m-%d_%H-%M-%S')}_{experiment_name}")
parser.add_argument("--num_nodes", default=nodes, type=int)
parser.add_argument("--gpus", default=gpus, type=int)
parser.add_argument("--num_workers", default=cpus_per_task, type=int)
parser.add_argument("--arch", default="custom", type=str, choices=("custom", "resnet18", "resnet34", "resnet50"))
parser.add_argument("--lr", default=0.0001, type=float)
parser.add_argument("--optimizer", default="Adam", type=str, choices=("Adam", "SGD", "RMSprop"))
parser.add_argument("--criterion", default="cross_entropy", type=str, choices=("cross_entropy", "nll_loss"))
parser.add_argument("--max_epochs", default=10, type=int)
parser.add_argument("--batch_size", default=100, type=int)

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
cluster.memory_mb_per_node = "4g"
cluster.job_time = "0-00:10:00"

# Add tasks-per-node to Slurm script
cluster.add_slurm_cmd(cmd="tasks-per-node", value=tasks_per_node, comment="tasks per node")

# Email results
# cluster.notify_job_status(email="<email_address>", on_done=True, on_fail=True)

# Submit job to Slurm workload manager
cluster.optimize_parallel_cluster_gpu(
    train_function=main,
    nb_trials=1,
    job_name=hparams.experiment_name,
    enable_auto_resubmit=False,
    job_display_name=experiment_name
)
