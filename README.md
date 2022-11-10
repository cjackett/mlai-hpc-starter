# MLAI HPC Starter
## Overview

This repository contains a basic starter kit for training machine learning models on High Performance Computers (HPCs). This guide will step through how to set up a local and HPC development environment, training simple classification models on the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database), and distributed multi-node, multi-GPU training of more complicated models on the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).

A range of standard machine learning software engineering techniques will be covered including interaction with the HPC Slurm Workload Manager, best practices for model training logging, automated model checkpointing, and a number of efficient training techniques such as memory pinning and automatic mixed precision. The aim of this guide is to develop a generalised and distributed machine learning training stack that can be easily customised for new datasets and models.

There are many ways to implement distributed model training in an HPC environment, and this guide will explore one
 approach based on the following Python frameworks and libraries:
 
 * [PyTorch](https://pytorch.org/) - open source machine learning library based on Torch
 * [PyTorch Lightning](https://www.pytorchlightning.ai/) - a lightweight PyTorch wrapper for high-performance AI research 
 * [Test Tube](https://williamfalcon.github.io/test-tube/) - log metadata and track your machine learning experiments
 * [TensorBoard](https://www.tensorflow.org/tensorboard) - visualisation and tooling needed for machine learning experimentation

Other tools that are used throughout this content include:

 * [BASH](https://en.wikipedia.org/wiki/Bash_(Unix_shell)) - Unix shell and command language
 * [Git](https://git-scm.com/) - free and open source distributed version control system
 * [Python Virtual Environments](https://docs.python.org/3/library/venv.html) - isolated Python environment containing the Python interpreter, libraries and scripts

This repository is structured into a number of different branches that contain various stages of the guide. The rest of this guide will step through the content in each branch. If you are experienced with machine learning in an HPC environment, you may want to skip straight to the final version of the code by checking out `8-distributed-training`. Otherwise, please checkout the first branch and follow along with the tutorials in this README.md file.

---

## Contents

* [Project preparation](#project-preparation)
    1. [Optional - Set up remote code deployment](#optional---set-up-remote-code-deployment)
    2. [Set up Python virtual environment](#set-up-python-virtual-environment)
        * [Optional - Set up local Python virtual environment for IDE integration](#optional---set-up-local-python-virtual-environment-for-ide-integration)
1. [Simple model training](#simple-model-training)
2. [PyTorch Lightning Module](#pytorch-lightning-module)
3. [Improved data loading with multiple workers](#improved-data-loading-with-multiple-workers)
4. [Automatic Slurm script generation](#automatic-slurm-script-generation)
   * [Optional bonus challenge 1 - HPC experiment number](#optional-bonus-challenge-1---hpc-experiment-number)
   * [Optional bonus challenge 2 - Fixing error logs](#optional-bonus-challenge-2---fixing-error-logs)
5. [Improved console and TensorBoard logging](#improved-console-and-tensorboard-logging)
   * [Optional bonus challenge 3 - Alternative logging dashboards](#optional-bonus-challenge-3---alternative-logging-dashboards)
   * [Optional bonus challenge 4 - Fix PR curve logging](#optional-bonus-challenge-4---fix-pr-curve-logging)
6. [PyTorch Lightning Data Module](#pytorch-lightning-data-module)
7. [Changing architecture](#changing-architecture)
8. [Distributed training with DDP](#distributed-training-with-ddp)
* [What now?](#what-now)
* [Acknowledgements](#acknowledgements)


![HackFest Roadmap](.img/0-roadmap.jpg?raw=true "HackFest Roadmap")

---

<a name="project-preparation"></a>
## Project preparation

<a name="optional---set-up-remote-code-deployment"></a>
### Optional - Set up remote code deployment

Actively working on machine learning code in an HPC environment can be achieved a number of ways. The approach taken here is to set up a remote code deployment using a modern Integrated Development Environment (IDE), which allows code to be developed on a local machine and continuously synchronised to a remote location. In practice, this works quite effectively as only individual text files are typically modified during development. The IDE maintains a SFTP (FTP over SSH) connection with the remote server, and files are dynamically synchronised through this secure channel.

The following blog posts show how to set up remote development in JetBrains IDEs and Visual Studio Code:

 * JetBrains - [Remote Development with Pycharm](https://jvision.medium.com/remote-development-with-pycharm-d741287e07de)
 * Visual Studio Code - [VS Code Remote Development](https://code.visualstudio.com/docs/remote/remote-overview)


<a name="set-up-python-virtual-environment"></a>
### Set up Python virtual environment

Let's go ahead and SSH into the HPC and set up our code:

```bash
ssh <hpc.domain.name>
```

```bash
git clone https://github.com/cjackett/mlai-hpc-starter.git -b 1-simple-model-training
cd mlai-hpc-starter
```

We're going to install a Python virtual environment into the `mlai-hpc-starter/env` directory. We will use the Python instance that is already installed on the HPC to create a Python virtual environment in our project working directory, and this Python virtual environment will then be activated and used when we train our models:

```bash
python3 -m venv env
source env/bin/activate
```

Upgrade pip and install Python packages from requirements.txt:

```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

Note here that the `--no-cache-dir` is important when installing packages in the HPC environment. Not using `--no-cache-dir` can result in the installation process repeatedly searching for the package cache directory and stalling indefinitely.

<a name="optional---set-up-local-python-virtual-environment-for-ide-integration"></a>
#### Optional - Set up local Python virtual environment for IDE integration

Unfortunately, with the Python virtual environment being located on the HPC, our local IDE does not have access to the environment to allow it to perform all of the advanced code features available in modern IDEs. This will result in your local IDE complaining that no Python interpreter has been set, package requirements are not satisfied, incorrect code checking, and an inability to inspect imported library and package code. All of these IDE errors can be avoided by setting up a local virtual environment:

In a terminal on your local machine, install and activate a local Python virtual environment:

```bash
python3 -m venv env
source env/bin/activate
```

Upgrade pip and install Python packages from requirements.txt:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

In PyCharm, you can now set the IDE Python interpreter at:
* File -> Settings -> Project: -> Python Interpreter -> Cog -> Add
    * Virtualenv Environment -> Existing environment -> Interpreter -> ... -> Select /mlai-hpc-starter/env/bin/python3

---

<a name="simple-model-training"></a>
## 1. Simple model training

![3-layer neural network](.img/1-neural-network.jpg?raw=true "3-layer neural network")


This first repository branch contains the minimal code necessary to train a simple 3-layered ML model on the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database) using a single CPU. This is a standard PyTorch implementation, and does not yet use the PyTorch Lightning framework. 

In your local terminal, double-check which git branch you are currently working on:

```bash
git branch
```

If you are on a branch other than `1-simple-model-training`, checkout the first repo branch:

```bash
git checkout 1-simple-model-training
```

If using a remote code deployment, any file changes that occur when checking-out git branches will automatically be synchronised to the remote location. This can be verified in PyCharm by inspecting the 'File Transfer' tab window at the bottom of the IDE.

There are only two files required to train this simple ML model: `src/main.py` and `run.sh`. The `src/main.py` file contains code from the official [PyTorch Lightning tutorial](https://colab.research.google.com/drive/1Mowb4NzWlRCxzAFjOIJqUmmk_wAT-XP3) which sets up a 3-layer fully-connected neural network and trains it on the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database). Our HPC-compatible version of the official tutorial contains a couple of minor changes, most notably the addition of a small amount of logging code so that we can log core training statistics at the end of every epoch.

But before we perform any model training, we need to fetch our training data. Normally, this would take place during the execution of our training code. However, in the `data` directory there is a BASH script to fetch the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) and [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) datasets. As bandwidth to the MNIST servers is often very poor, we have already pre-configured the datasets in this repository for convenience.

The `run.sh` file contains the Slurm commands required to configure and submit the batch job to the HPC Workload Manager. In a HPC SSH terminal, execute the run script:

```bash
sbatch ./run.sh
```

You should receive a response similar to the following:

```
Submitted batch job 49402998
```

This indicates that the batch job was successfully submitted to the Workload Manager, and is now in the queue. You can inspect the progress of the job with:

```bash
sacct
```

If your batch job is taking some time to start, you can ask the Workload Manager to give you an estimate of when the job is currently scheduled to start with:

```bash
squeue --user <ident> --start
```

After the batch job has started, the logging will be printed to a file in the working directory (`mlai-hpc-starter`) called `slurm.<job_id>.out`. In the PyCharm Remote Host browser on the right-hand panel, you can hit the refresh button and double-click on the log file. This will view a temporary copy of the log file, and you should see the epoch logging statistics for the training run:

```
epoch: 01/10    training loss: 0.0040    training accuracy 0.9228    validation loss: 0.0023    validation accuracy: 0.9512
epoch: 02/10    training loss: 0.0016    training accuracy 0.9681    validation loss: 0.0018    validation accuracy: 0.9666
epoch: 03/10    training loss: 0.0011    training accuracy 0.9780    validation loss: 0.0018    validation accuracy: 0.9688
epoch: 04/10    training loss: 0.0008    training accuracy 0.9830    validation loss: 0.0018    validation accuracy: 0.9690
epoch: 05/10    training loss: 0.0007    training accuracy 0.9863    validation loss: 0.0017    validation accuracy: 0.9720
epoch: 06/10    training loss: 0.0006    training accuracy 0.9874    validation loss: 0.0016    validation accuracy: 0.9764
epoch: 07/10    training loss: 0.0005    training accuracy 0.9894    validation loss: 0.0016    validation accuracy: 0.9756
epoch: 08/10    training loss: 0.0004    training accuracy 0.9911    validation loss: 0.0019    validation accuracy: 0.9752
epoch: 09/10    training loss: 0.0004    training accuracy 0.9915    validation loss: 0.0017    validation accuracy: 0.9792
epoch: 10/10    training loss: 0.0003    training accuracy 0.9929    validation loss: 0.0018    validation accuracy: 0.9756
```

The model training is limited to run for 10 epochs. When working correctly, you should see the training and validation accuracy increase, while the training and validation loss decrease.

---

<a name="pytorch-lightning-module"></a>
## 2. PyTorch Lightning Module

![PyTorch Lightning](.img/2-pytorch-lightning.png?raw=true "PyTorch Lightning")

In this branch we will see how to refactor the standard PyTorch model implementation into PyTorch Lightning Module format. Let's start by working through the official [PyTorch Lightning introduction tutorial](https://colab.research.google.com/drive/1Mowb4NzWlRCxzAFjOIJqUmmk_wAT-XP3?usp=sharing). This tutorial should take about 30 minutes to complete and will provide a good overview of the refactoring process, the PyTorch Lightning Module structure, important PyTorch Lightning methods, and many of the additional features you get for free by structuring your code in the PyTorch Lightning format.

Note: the official PyTorch Lightning tutorial appears to contain a number of typos and other errors, which will require small modifications to make the cells execute correctly.

In this repo branch there is an HPC-compatible version of the final official PyTorch Lightning tutorial code. In a local terminal, checkout the second repo branch:

```bash
git checkout 2-lightning-module
```

The `src/main.py` file now contains a refactored PyTorch Lightning version of the original training code from branch `1-simple-model-training`, which are both directly based on the examples in the official PyTorch Lightning introduction tutorial we just completed. In a HPC SSH terminal, execute the run script:

```bash
sbatch ./run.sh
```

We can look at the standard output of the job by inspecting the Slurm log file `slurm.<job_id>.out`. Also notice that we now get logging from PyTorch Lightning stored in the `/mlai-hpc-starter/lightning_logs` directory. Each subdirectory is a separate model run, and the subdirectory name contains the Slurm job ID. The directory structure should look like this:

```
mlai-hpc-starter
└───data                        - Data directory to store MNIST and CIFAR10
└───env                         - Python virtual environment
└───lightning_logs              - PyTorch Lightning default logging directory
│   └───version_52339445        - Logging directory for single model run
│       └───checkpoints         - Checkpoints direcory to store saved model
│   └───version_52340129        - Logging directory for single model run
│       └───checkpoints         - Checkpoints direcory to store saved model
└───src                         - Source directory
```

PyTorch Lightning automatically logs data to [TensorBoard](https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html) and also performs [automated checkpointing](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.model_checkpoint.html) into the `checkpoints` directory mentioned above.

If we wish to view the TensorBoard logs, we first need to download the logs from HPC. In the Remote Host browser on the right-hand panel in PyCharm, right-click on the `lightning_logs` directory and select `Download from here`. If we previously set up a local Python Virtual Environment, we can start TensorBoard in a local terminal with the following command:

```bash
tensorboard --logdir lightning_logs/
```

You should see the following output in the terminal:

```bash
TensorFlow installation not found - running with reduced feature set.<br>
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all<br>
TensorBoard 2.4.1 at http://localhost:6006/ (Press CTRL+C to quit)
```

Click on the link to view TensorBoard running on your local machine. This is the most basic usage of TensorBoard, and later on we will dive much deeper into model training logging in the repo branch `5-improved-logging`.

If we look at the Slurm log file `slurm.<job_id>.out` we find the following error:

```bash
UserWarning: The dataloader, train dataloader, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument in the `DataLoader` init to improve performance.
```

PyTorch Lightning has a good amount of targeted error messages that can be very helpful in avoiding common implementation mistakes. In this case, PyTorch Lightning is suggesting that we need more data loaders because this model is only using a single CPU to load images from disk and pass them to the model. We will address this in the next branch.

Now we have successfully trained a PyTorch Lightning model using a single CPU on a single node on HPC. Let's go and fix that DataLoader issue!

---

<a name="improved-data-loading-with-multiple-workers"></a>
## 3. Improved data loading with multiple workers

At this point, we can make some performance improvements to the way we load our data from disk. In a local terminal, checkout the next repo branch:

```bash
git checkout 3-improved-data-loading
```

PyTorch allows data to be loaded using [multiple processes simultaneously](https://pytorch.org/docs/stable/data.html#multi-process-data-loading). To enable this, all we need to do is provide the `num_workers` argument to the DataLoader class.

But how many CPUs should we use to simultaneously load our data from disk?

We are currently working on CPU-based HPC, but we will end up working on GPU-based HPC where each node contains 28 CPUs (2 x 14-core CPUs) and 4 GPUs. A general rule-of-thumb when performing GPU-based model training is:

* num_worker = 4 * num_gpu

Right now we are performing the model training on CPU, so we can simply change the `--cpus-per-task` Slurm command in the `run.sh` script to allocate 5 CPUs for the entire job, and then provide 4 CPUs for data loading in the LightningMNISTClassifier class in `src/main.py`, which is subsequently passed to the DataLoaders.

Increasing the value of `num_workers` has the side-effect of increasing the CPU memory consumption. To manage this, we have increased the `--mem` Slurm command in the `run.sh` script accordingly.

Because `num_workers` is dependent on the specific `batch_size` and the allocated compute resources, a more finely-tuned approach would be to increase the value `num_workers` slowly until there is no more improvement in the model training speed.

In a HPC SSH terminal, let's re-execute the run script:

```bash
sbatch ./run.sh
```

Using the `sacct` command, notice the significant reduction in model training time compared to the previous single CPU example.

References:

* [PyTorch multi-process data loading](https://pytorch.org/docs/stable/data.html#multi-process-data-loading)
* [7 Tips To Maximize PyTorch Performance](https://towardsdatascience.com/7-tips-for-squeezing-maximum-performance-from-pytorch-ca4a40951259)
* [Guidelines for assigning num_workers to DataLoader](https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813)

---

<a name="automatic-slurm-script-generation"></a>
## 4. Automatic Slurm script generation

![Slurm workload manager](.img/4-slurm-workload-manager.png?raw=true "Slurm workload manager")

So far, we have been executing our code on a HPC using a custom-crafted `run.sh` script containing Slurm commands that are interpreted by the HPC. However, a more flexible solution exists. In this branch we will use the [Test Tube](https://williamfalcon.github.io/test-tube/) Python package to automatically generate and submit our Slurm scripts to the Slurm work load manager.

In a local terminal, checkout the next repo branch:

```bash
git checkout 4-auto-slurm-script-generation
```

Notice that the `run.sh` script has now been replaced with a `run.py` file. This new Python script contains the minimal code and structure necessary to generate and execute our Slurm scripts on the HPC. As an overview, this script performs the following tasks:

* Specify the amount of compute resources required for the job (currently we are not using any GPUs)
* Define a number of useful model training arguments and pass them to the SlurmCluster constructor
* Set up SlurmCluster commands, which are equivalent to the Slurm `SBATCH` commands in the previous `run.sh` script
* Call the `cluster.optimize_parallel_cluster_gpu()` method which generates the Slurm scripts and submits them to the HPC workload manager

Other notable changes have occurred in the `src/main.py` file:

* A `main()` method has been added which contains the high-level structure for the model training process - this is called directly from the submitted Slurm scripts
* We have switched to a custom logger ([TestTubeLogger](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.loggers.test_tube.html)) which has a better logging structure than the default option
* The learning rate value has become a pass-through argument, allowing it to be set at the highest level
* The `train_dataloader()` has been set to randomly shuffle the data during loading

In a HPC SSH terminal, make sure you have activated the Python Virtual Environment and then execute the model training code using:

```bash
source env/bin/activate
python ./run.py
```

Now that we are using a custom logger, our new training logs will appear in the `/logs` directory with the following structure:

```
mlai-hpc-starter
...
└───logs                            - Base TestTube logging directory
    └───2021-05-12_12-19-30_run     - Logging directory for single model run
        └───slurm_err_logs          - Slurm error logs
        └───slurm_out_logs          - Slurm standard output logs
        └───slurm_scripts           - Slurm scirpts executed on the HPC
        └───version_0               - TensorBoard logs for trained model
...
```

Structuring our machine learning training code and logs in this way has a few nice properties:

* Machine learning experiments can be easily configured and executed at the top level in the `run.py` script
* In the `run.py` script, we are currently generating and executing single Slurm scripts with `nb_trials=1`, but it's very easy to extend this to generate many scripts and run broad-scale machine learning experiments
* Training logs are neatly accumulated, scaling nicely when running large numbers of model runs, and are easily targeted and managed in TensorBoard

<a name="optional-bonus-challenge-1---hpc-experiment-number"></a>
### Optional bonus challenge 1 - HPC experiment number

Currently, the `--experiment_name` in the `run.py` script sets the log directory name using a timestamp-based naming scheme. A neater solution would be to name the log directory using the submitted Slurm job allocation number. This would have the advantage of log directories sorting in chronological order, and would also match the `JobID` from the `sacct` Slurm history which would be very useful when auditing job completions. Hint: there is a `hpc_exp_number` argument in the TestTube SlurmCluster object, and this number gets added to the generated Slurm script found in the `slurm_scripts` directory.


<a name="optional-bonus-challenge-2---fixing-error-logs"></a>
### Optional bonus challenge 2 - Fixing error logs

For some reason, PyTorch Lightning likes to send its model logging to the Slurm error log file in the `slurm_err_logs` directory. It would make more sense to log this output to the Slurm out log file in the `slurm_out_logs` directory. The challenge here is to find and override the PyTorch Lightning default logging stream and send all model logging to the Slurm out log file.

---

<a name="improved-console-and-tensorboard-logging"></a>
## 5. Improved console and TensorBoard logging

![TensorBoard](.img/5-tensorboard.png?raw=true "TensorBoard")

In this branch we will upgrade the logging capability of our model training process using two approaches: 

 1. Implement standard Python module logging
 2. Further utilise the TensorBoard methods and metrics
    
In a local terminal, let's checkout the next repo branch:

```bash
git checkout 5-improved-logging
```

In the `run.py` script we have added a number of pass-through arguments including `--arch`, `--optimizer` and `--criterion` which will allow us to configure these values at the top level. We've also included `src/utils/tensorboard.py` containing a number of TensorBoard helper function that will help keep the code in `src/main.py` relatively clean.

Most of the changes in this branch occur in `src/main.py`. The additional arguments set up in `run.py` are passed through to the `LightningMNISTClassifier` and stored as private member variables so that they can be used later from within the class. In these places, we now use Python [`__dict__` attribute indexing](https://docs.python.org/3/library/stdtypes.html#object.__dict__) to allow for dynamically selecting the loss criteria and optimiser using our passed-in arguments. Later on, this will allow us to run experiments where we grid search over any of these variables and easily compare the resulting trained models. 

In the constructor of the `LightningMNISTClassifier` class we have instantiated and configured a standard Python logger, setting the logging level to DEBUG and customising the default format. This logger is used to report the text-based model training statistics at the end of each validation loop, which will appear in the log file in the `logs/<experiment_name>/slurm_out_logs` directory. Formalised logging is a good habit to get into, and can be extremely useful when chasing down errors and bugs.

Instead of manually calculating the mean and accuracy statistics for all batches in each epoch, we have now moved to the Python [TorchMetrics](https://torchmetrics.readthedocs.io/en/latest/) package that offers a number of benefits:

* Optimized for distributed-training
* A standardized interface to increase reproducibility
* Reduces Boilerplate
* Rigorously tested
* Automatic accumulation over batches
* Automatic synchronization between multiple devices

TorchMetrics will be useful in performing the heavy-lifting when computing statistics across multiple nodes and GPUs on the HPC. Specifically, we will be using the following TorchMetrics to calculate quantities for logging with TensorBoard:

* [Accuracy](https://torchmetrics.readthedocs.io/en/latest/references/modules.html#accuracy)
* [Precision](https://torchmetrics.readthedocs.io/en/latest/references/modules.html#precision)
* [Recall](https://torchmetrics.readthedocs.io/en/latest/references/modules.html#recall)
* [F1](https://torchmetrics.readthedocs.io/en/latest/references/modules.html#f1)
* [ConfusionMatrix](https://torchmetrics.readthedocs.io/en/latest/references/modules.html#confusionmatrix)

Finally, in the `main` method of `src/main.py` we have added the `weights_summary` and `profiler` flags to the PyTorch Lightning Trainer. This will add some useful model and performance statistics to the Slurm log files.

In a HPC SSH terminal, make sure you have activated the Python Virtual Environment and then execute the model training code using:

```bash
source env/bin/activate
python ./run.py
```

After locally copying the log directory, we can now start TensorBoard in a local terminal with the following command:

```bash
tensorboard --logdir logs/<experiment_name> --reload_multifile True
```

You should see the following output in the terminal:

```bash
TensorFlow installation not found - running with reduced feature set.<br>
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all<br>
TensorBoard 2.4.1 at http://localhost:6006/ (Press CTRL+C to quit)
```

Click on the link to view TensorBoard running on your local machine and inspect all of the new model training metrics that have been added.

If we look at the Slurm error log file in `logs/<experiment_name>/slurm_err_logs`, we now find a summary of the neural network layers, the number of trainable/non-trainable parameters, and any additional metrics used:

```bash
  | Name             | Type            | Params
-----------------------------------------------------
0 | layer_1          | Linear          | 100 K 
1 | layer_2          | Linear          | 33.0 K
2 | layer_3          | Linear          | 2.6 K 
3 | train_accuracy   | Accuracy        | 0     
4 | val_accuracy     | Accuracy        | 0     
5 | val_precision    | Precision       | 0     
6 | val_recall       | Recall          | 0     
7 | val_f1           | F1              | 0     
8 | confusion_matrix | ConfusionMatrix | 0     
-----------------------------------------------------
136 K     Trainable params
0         Non-trainable params
136 K     Total params
0.544     Total estimated model params size (MB)
```

Below this, we see the output of the performance profiler:

```bash
FIT Profiler Report

Action                             	|  Mean duration (s)	|Num calls      	|  Total time (s) 	|  Percentage %   	|
--------------------------------------------------------------------------------------------------------------------------------------
Total                              	|  -              	|_              	|  54.097         	|  100 %          	|
--------------------------------------------------------------------------------------------------------------------------------------
run_training_epoch                 	|  4.9396         	|10             	|  49.396         	|  91.31          	|
run_training_batch                 	|  0.0040289      	|5500           	|  22.159         	|  40.961         	|
optimizer_step_and_closure_0       	|  0.0037768      	|5500           	|  20.773         	|  38.399         	|
training_step_and_backward         	|  0.0029197      	|5500           	|  16.058         	|  29.684         	|
model_forward                      	|  0.0018828      	|5500           	|  10.355         	|  19.142         	|
get_train_batch                    	|  0.0018363      	|5500           	|  10.1           	|  18.67          	|
training_step                      	|  0.0017298      	|5500           	|  9.514          	|  17.587         	|
backward                           	|  0.00080742     	|5500           	|  4.4408         	|  8.209          	|
on_train_end                       	|  2.8797         	|1              	|  2.8797         	|  5.3233         	|
evaluation_step_and_end            	|  0.0045184      	|500            	|  2.2592         	|  4.1762         	|
validation_step                    	|  0.0044211      	|500            	|  2.2105         	|  4.0862         	|
cache_result                       	|  7.9011e-06     	|23625          	|  0.18666        	|  0.34505        	|
on_train_batch_end                 	|  2.2665e-05     	|5500           	|  0.12466        	|  0.23043        	|
on_validation_end                  	|  0.012174       	|10             	|  0.12174        	|  0.22505        	|
on_batch_start                     	|  1.5598e-05     	|5500           	|  0.085791       	|  0.15859        	|
on_after_backward                  	|  1.2969e-05     	|5500           	|  0.071327       	|  0.13185        	|
on_batch_end                       	|  1.1801e-05     	|5500           	|  0.064906       	|  0.11998        	|
on_before_zero_grad                	|  1.0726e-05     	|5500           	|  0.058992       	|  0.10905        	|
on_train_batch_start               	|  8.5534e-06     	|5500           	|  0.047044       	|  0.086962       	|
training_step_end                  	|  8.5012e-06     	|5500           	|  0.046756       	|  0.086431       	|
on_validation_batch_start          	|  1.454e-05      	|500            	|  0.0072698      	|  0.013438       	|
validation_step_end                	|  9.2029e-06     	|500            	|  0.0046015      	|  0.0085059      	|
on_validation_batch_end            	|  8.809e-06      	|500            	|  0.0044045      	|  0.0081418      	|
on_train_epoch_end                 	|  0.00015503     	|10             	|  0.0015503      	|  0.0028658      	|
on_validation_epoch_end            	|  3.0107e-05     	|10             	|  0.00030107     	|  0.00055653     	|
on_epoch_start                     	|  1.2418e-05     	|20             	|  0.00024836     	|  0.00045909     	|
on_epoch_end                       	|  1.2096e-05     	|20             	|  0.00024193     	|  0.00044721     	|
on_validation_start                	|  1.2134e-05     	|10             	|  0.00012134     	|  0.00022429     	|
on_train_epoch_start               	|  1.1144e-05     	|10             	|  0.00011144     	|  0.000206       	|
on_validation_epoch_start          	|  9.2154e-06     	|10             	|  9.2154e-05     	|  0.00017035     	|
on_train_dataloader                	|  2.491e-05      	|1              	|  2.491e-05      	|  4.6046e-05     	|
on_before_accelerator_backend_setup	|  2.2888e-05     	|1              	|  2.2888e-05     	|  4.2309e-05     	|
on_fit_start                       	|  1.4151e-05     	|1              	|  1.4151e-05     	|  2.6159e-05     	|
on_train_start                     	|  1.2853e-05     	|1              	|  1.2853e-05     	|  2.3759e-05     	|
on_val_dataloader                  	|  8.0927e-06     	|1              	|  8.0927e-06     	|  1.496e-05      	|
```

References:

* [Python Official - Logging HOWTO](https://docs.python.org/3/howto/logging.html)
* [Python Ultimate Guide to Logging](https://www.loggly.com/ultimate-guide/python-logging-basics/)
* [TorchMetrics](https://torchmetrics.readthedocs.io/en/latest/)
* [`__dict__` attribute indexing](https://docs.python.org/3/library/stdtypes.html#object.__dict__)

<a name="optional-bonus-challenge-3---alternative-logging-dashboards"></a>
### Optional bonus challenge 3 - Alternative logging dashboards

This branch has focussed on logging all of the model training metrics and statistics to TensorBoard (via [TestTubeLogger](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.loggers.test_tube.html)). However, PyTorch Lightning has native support for a [range of different loggers](https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html#supported-loggers), including:

* [Comet](https://www.comet.ml)
* [MLflow](https://mlflow.org/)
* [Neptune](https://neptune.ai/)
* [Weights and Biases](https://www.wandb.com)

This bonus challenge is to replace the current TestTubeLogger with any of the above, and install and run the appropriate frontend interface to view the training logs. Good luck!

<a name="optional-bonus-challenge-4---fix-pr-curve-logging"></a>
### Optional bonus challenge 4 - Fix PR curve logging

PR curves can give a visual indication of the trade-off between model precision and recall. However, it appears that the PR curve logging may currently not be working correctly. This challenge involves closely inspecting the PR curve logging and fixing any issues.

---

<a name="pytorch-lightning-data-module"></a>
## 6. PyTorch Lightning Data Module

In this branch, we will refactor our current codebase to use the PyTorch Lightning [DataModule](https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html) format, which will help encapsulate our datasets so that our project doesn't become too unwieldy. We will also refactor the [LightningModule](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html) code to achieve a better layout, and make the project structure more maintainable.

In a local terminal, let's git checkout the next repo branch:

```bash
git checkout 6-lightning-data-module
```

The most notable change in this branch is that we have moved all of the PyTorch [LightningModule](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html) code into the new file `src/models/lightning_classifier.py`.  Also, notice that we have reconfigured our simple 3-layered neural network to accept the [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) data, which contains slightly larger images (32x32 pixels), with more variation than the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset.  This will mean that:

1. The more difficult task will be harder for the neural network to learn, and
2. The larger images will require more allocated memory to complete the training

For reason (2) we double the memory in `run.py` to 4GB to handle this increase in image size.

To further refactor our code, PyTorch Lightning recommends that all dataset methods be encapsulated in a [DataModule](https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html):

> In normal PyTorch code, the data cleaning/preparation is usually scattered across many files. This makes sharing and reusing the exact splits and transforms across projects impossible.
> Datamodules are for you if you ever asked the questions:
> * what splits did you use?
> * what transforms did you use?
> * what normalization did you use?
> * how did you prepare/tokenize the data?

Nicely, both of the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) and [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) datasets have already been encapsulated into PyTorch Lightning [DataModules](https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html), and we can simply import them into `src/main.py` from the [PyTorch Lightning Bolts Data Modules](https://lightning-bolts.readthedocs.io/en/latest/datamodules.html) package.

Side-node: The [PyTorch Lightning Bolts](https://lightning-bolts.readthedocs.io/en/latest/) package contains a large number of cutting-edge implementations for SOTA pretrained models including self-supervised learning, contrastive learning, VAEs, GANs, as well as DataModules for standard datasets - all compatible with PyTorch Lightning and distributed training.

Side-side-note: Following the project structure that we have started here, if you want to implement your own DataModule that points to your own data, one option would be to create a dataset folder at `src/datasets` and write a custom PyTorch Lightning DataModule such as `src/datasets/custom_data_module.py` by referring to the required methods in the official documentation [official documentation](https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html).

At this point, the `src/main.py` file is exceptionally clean, containing only high-level instantiations of the DataModules, the model, the logger and the Trainer. This is good news.

In a HPC SSH terminal, make sure you have activated the Python Virtual Environment and then execute the model training code using:

```bash
source env/bin/activate
python ./run.py
```

When the model run is complete, the `sacct` command will produce output similar to the following:

```bash
       JobID    JobName  Partition     User AllocCPUS NNodes    Elapsed   TotalCPU      State  MaxVMSize     MaxRSS     ReqMem        NodeList ReqGRES 
------------ ---------- ---------- -------- --------- ------ ---------- ---------- ---------- ---------- ---------- ---------- --------------- -------      
53685816          runv0      h2gpu   <ident>        5      1   00:01:30  02:55.122  COMPLETED                              4Gn            b004         
53685816.ba+      batch                             5      1   00:01:30  00:00.050  COMPLETED    313560K     11204K        4Gn            b004         
53685816.0      python3                             5      1   00:01:29  02:55.071  COMPLETED  20517416K   3941784K        4Gn            b004        
```

Notice the `MaxRSS` memory consumption at 3941784K (almost 4GB). This confirms that a small increase in image size requires a corresponding increase in memory resources to complete the model training.

If we inspect either the training logs at `logs/<experiment_number>/slurm_out_logs/` or the TensorBoard logging, we find that [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) was more difficult to train our 3-layer neural network on than [MNIST](https://en.wikipedia.org/wiki/MNIST_database). In the next branch we will upgrade the neural network architecture to a much deeper model and see how that impacts model training.

References:

* [PyTorch Lightning LightningModule](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html)
* [PyTorch Lightning DataModule](https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html)
* [PyTorch Lightning Bolts](https://lightning-bolts.readthedocs.io/en/latest/)

---

<a name="changing-architecture"></a>
## 7. Changing architecture

We have seen how a small 3-layered neural network can be trained efficiently on the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) and [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) datasets. Now it's time to train a much deeper network. The main change we will see in this branch is to swap our 3-layered network for a ResNet-18 network imported from the [PyTorch Torchvision models subpackage](https://pytorch.org/vision/stable/models.html).

![ResNet-18](.img/7-resnet-18.png?raw=true "ResNet-18")

The following plot gives an indication of the performance, training complexity and size of a range of different modern deep learning networks.

![Deep learning architectures](.img/7-deep-learning-architectures.jpg?raw=true "Deep learning architectures")

In a local terminal, checkout the next repo branch:

```bash
git checkout 7-changing-architectures
```

In the `src/models/lightning_classifier.py` file, we have imported the `torchvision.models` subpackage, used `__dict__` attribute indexing to dynamically select the `torchvision` model using a pass-through argument, and added a pass-through `pretrained` argument to determine if we start with the Imagenet pretrained model weights. We have also reconfigured the last fully-connected ResNet-18 layer to be compatible with [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) data and output the correct number of classes. The `forward()` method becomes greatly simplified and there are some minor changes to logging our model graph.

The `src/main.py` file now includes the following arguments:
* In LightningClassifier()
    * `pretrained` - pass-through argument to select pretrained model weights
* In Trainer()
    * [accelerator](https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html#distributed-modes) - Distributed training mode argument. We are not using GPUs just yet, but PyTorch Lightning throws a warning if we don't include this here.
    * [limit_train_batches / limit_val_batches]((https://pytorch-lightning.readthedocs.io/en/stable/common/debugging.html#shorten-epochs)) - Sometimes it’s helpful to only use a percentage of your training, val or test data (or a set number of batches). For example, you can use 20% of the training set and 1% of the validation set.
    * [fast_dev_run](https://pytorch-lightning.readthedocs.io/en/stable/common/debugging.html#fast-dev-run) - This flag runs a “unit test” by running n if set to n (int) else 1 if set to True training and validation batch(es). The point is to detect any bugs in the training/validation loop without having to wait for a full epoch to crash.
    * [overfit_batches](https://pytorch-lightning.readthedocs.io/en/stable/common/debugging.html#make-model-overfit-on-subset-of-data) - A good debugging technique is to take a tiny portion of your data (say 2 samples per class), and try to get your model to overfit. If it can’t, it’s a sign it won’t work with large datasets.

Finally, the `run.py` script has seen a few minor modifications to some of the pass-through arguments as well as the requested memory resources and job wall time.

If you execute the `run.py` script now, PyTorch will attempt to download the pretrained model weights for the ResNet-18 model from the official PyTorch server, which will fail with the following error:

```bash
Downloading: "https://download.pytorch.org/models/resnet18-5c106cde.pth" to /home/<ident>/.cache/torch/hub/checkpoints/resnet18-5c106cde.pth
Traceback (most recent call last):
  File "/apps/python/3.7.2/lib/python3.7/urllib/request.py", line 1317, in do_open
    encode_chunked=req.has_header('Transfer-encoding'))
  File "/apps/python/3.7.2/lib/python3.7/http/client.py", line 1229, in request
    self._send_request(method, url, body, headers, encode_chunked)
  File "/apps/python/3.7.2/lib/python3.7/http/client.py", line 1275, in _send_request
    self.endheaders(body, encode_chunked=encode_chunked)
  File "/apps/python/3.7.2/lib/python3.7/http/client.py", line 1224, in endheaders
    self._send_output(message_body, encode_chunked=encode_chunked)
  File "/apps/python/3.7.2/lib/python3.7/http/client.py", line 1016, in _send_output
    self.send(msg)
  File "/apps/python/3.7.2/lib/python3.7/http/client.py", line 956, in send
    self.connect()
  File "/apps/python/3.7.2/lib/python3.7/http/client.py", line 1392, in connect
    server_hostname=server_hostname)
  File "/apps/python/3.7.2/lib/python3.7/ssl.py", line 412, in wrap_socket
    session=session
  File "/apps/python/3.7.2/lib/python3.7/ssl.py", line 853, in _create
    self.do_handshake()
  File "/apps/python/3.7.2/lib/python3.7/ssl.py", line 1117, in do_handshake
    self._sslobj.do_handshake()
ConnectionResetError: [Errno 104] Connection reset by peer
```

We will need to download the pretrained model weights on the HPC data mover node, which is optimised for downloading data from the internet. Start a new SSH connection to `hpc.data.mover.domain.name` and run the `src/utils/cache_models.sh` file to fetch the correct pretrained model weights:

```bash
ssh <hpc.data.mover.domain.name>
./mlai-hpc-starter/src/utils/cache_models.sh
exit
```

Now that we have cached the ResNet-18 pretrained model weights, in a HPC SSH terminal, go ahead and execute the model training code using:

```bash
source env/bin/activate
python ./run.py
```

Using the `sacct`, the job should complete in about 10 minutes using less than 10GB of memory:

```bash
       JobID    JobName  Partition      User  AllocCPUS   NNodes    Elapsed   TotalCPU      State  MaxVMSize     MaxRSS     ReqMem        NodeList 
------------ ---------- ---------- --------- ---------- -------- ---------- ---------- ---------- ---------- ---------- ---------- --------------- 
53710221          runv0         h2    <ident>         5        1   00:09:25  11:49.556  COMPLETED                             10Gn            c017 
53710221.ba+      batch                               5        1   00:09:25  00:00.051  COMPLETED    313552K     11228K       10Gn            c017 
53710221.0      python3                               5        1   00:09:24  11:49.504  COMPLETED  43131640K   8662484K       10Gn            c017 
```

After locally copying the log directory, check out the training logs by starting TensorBoard in a local terminal with:

```bash
tensorboard --logdir logs/<experiment_name> --reload_multifile True
```

Next up, we will make the jump from single-node, multi-CPU training all the way to multi-node, multi-GPU, multi-CPU deep network model training!

References:

* [7 Tips To Maximize PyTorch Performance](https://towardsdatascience.com/7-tips-for-squeezing-maximum-performance-from-pytorch-ca4a40951259)
* [PyTorch Lightning Accellerator](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.accelerators.Accelerator.html)
* [PyTorch Lightning Debugging](https://pytorch-lightning.readthedocs.io/en/stable/common/debugging.html)

---

<a name="distributed-training-with-ddp"></a>
## 8. Distributed training with DDP

Having completed much of the hard work and setup in the previous branches, this branch forges ahead to make good progress with only minimal changes to the code.

In a local terminal, let's checkout the last repo branch:

```bash
git checkout 8-distributed-training
```

In `src/main.py`, we recalculate the `batch_size` and  `workers` based on the number of GPUs requested so that DDP performs the correct splits. The `num_nodes` and `gpus` and `accelerator` are then passed through to the PyTorch Lightning Trainer.

Another speed improvement we can make at this stage is using [Automatic Mixed Precision (AMP)](https://pytorch.org/docs/stable/amp.html). It is often an unnecessary to use full 32-bit precision throughout the entire model during training, and we can use AMP to scale some, or all of the calculations back to 16-bit precision. In the PyTorch Lightning Trainer we set the `precision` argument to 16 to enable AMP training, and this has a couple of computational benefits:

1. Memory use is decreased by half, which means you can double batch size and cut training time in half
2. Certain GPUs (V100, 2080Ti) give you automatic speed-ups (3x-8x faster) because they are optimized for 16-bit computations

In the `run.py` script, we add the `accelerator` pass-through argument and set it to DDP (DistributedDataParallel). There are a number of different [distributed modes](https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html#distributed-modes) that PyTorch Lightning supports, but the punchline is that you should always use DDP if possible. This is because the DDP implementation is the moat efficient, only syncing gradients across GPUs and nodes. DDP works as follows:

1. Each GPU across each node gets its own process
2. Each GPU gets visibility into a subset of the overall dataset. It will only ever see that subset
3. Each process inits the model
4. Each process performs a full forward and backward pass in parallel
5. The gradients are synced and averaged across all processes
6. Each process updates its optimizer

![Distributed data parallel training](.img/8-distributed-data-parallel-training.png?raw=true "Distributed data parallel training")

In `run.py`, we also increase the requested compute resources and slightly modify the variables used to define these resources. The specific configuration of `cpus_per_task`, `tasks_per_node`, and `num_workers`, and how they are passed through as arguments and Slurm commands is critical for making DDP work correctly. This is because Slurm needs to know precisely how many, and which CPUs need to be connected to each GPU where the model is trained. With this configuration, you only need to change the `gpus` value (limited to a maximum of 4) and the `nodes` value (with a practical upper limit of say 20) to dramatically scale up the machine learning training process. The computational resources specified in `run.py` are the minimum to demonstrate full multi-node GPU training on the HPC.

Another performance improvement to know if we are implementing our own DataModule, is we can set the [`pin_memory` argument in each DataLoader to `True`](https://pytorch.org/docs/stable/data.html#memory-pinning).

> Host to GPU copies are much faster when they originate from pinned (page-locked) memory. For data loading, passing pin_memory=True to a DataLoader will automatically put the fetched data Tensors in pinned memory, and thus enables faster data transfer to CUDA-enabled GPUs.

This is already happening for us in the [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) DataModule implementation (see if you can navigate through the code to find it), but it's worth keeping in mind when working on your own custom DataModule.

And that's it. We have successfully implemented distributed machine learning model training in the CSIRO HPC environment!

References:

* [PyTorch - Automatic Mixed Precision](https://pytorch.org/docs/stable/amp.html)
* [7 Tips To Maximize PyTorch Performance](https://towardsdatascience.com/7-tips-for-squeezing-maximum-performance-from-pytorch-ca4a40951259)
* [PyTorch Lightning - Multi-GPU training](https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html)
* [Distributed Deep Learning With PyTorch Lightning](https://devblog.pytorchlightning.ai/distributed-deep-learning-with-pytorch-lightning-part-1-8df1d032e6d3)
* [Getting Started with Distributed Data Parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
* [7 Tips To Maximize PyTorch Performance](https://towardsdatascience.com/7-tips-for-squeezing-maximum-performance-from-pytorch-ca4a40951259)

---

<a name="what-now"></a>
## What now?

This guide has taken us from training a simple 3-layer neural network on the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset using the HPC, through to full distributed model training across multiple nodes, GPUs and CPUs on the HPC. But what could this be used for now?

Well, the structure of this codebase lends itself nicely to be extended in various ways. The first thing you might try is writing your own [DataModule](https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html) in the `src/datasets` directory that points to your own dataset.

Next, you will probably want to build your own model in the `src/models` directory that is more sophisticated than the classification model implemented here. PyTorch Lightning is really just organised PyTorch code, and so any type of machine learning model can be implemented in PyTorch Lightning and benefit from the many features that PyTorch Lightning provides. Well maintained [documentation](https://lightning-bolts.readthedocs.io/en/latest/index.html) and [code](https://github.com/PyTorchLightning/pytorch-lightning) exists for many example model implementations including:

* [Autoencoders](https://lightning-bolts.readthedocs.io/en/latest/autoencoders.html) such as [AE](https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/autoencoders/basic_ae/basic_ae_module.py) and [VAE](https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/autoencoders/basic_vae/basic_vae_module.py)
* [Convolutional architectures](https://lightning-bolts.readthedocs.io/en/latest/convolutional.html) such as [GPT-2](https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/vision/image_gpt/gpt2.py) and [Image GPT](https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/vision/image_gpt/igpt_module.py)
* [GANs](https://lightning-bolts.readthedocs.io/en/latest/gans.html) including [Basic GAN](https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/gans/basic/basic_gan_module.py) and [DCGAN](https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/gans/dcgan/dcgan_module.py)
* Object detection models such as [Faster R-CNN](https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/detection/faster_rcnn/faster_rcnn_module.py)
* [Reinforcement learning](https://lightning-bolts.readthedocs.io/en/latest/reinforce_learn.html) including [many...](https://github.com/PyTorchLightning/lightning-bolts/tree/master/pl_bolts/models/rl)
* [Self-supervised learning](https://lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html#simclr) including [SimCL](https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/self_supervised/simclr/simclr_module.py), [SwAV](https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/self_supervised/swav/swav_module.py) and other [contrastive learning models](https://github.com/PyTorchLightning/lightning-bolts/tree/master/pl_bolts/models/self_supervised)

As you progress through a machine learning research project, you might find yourself writing small experiments to investigate different neural network properties or machine learning training configurations. In this case you might think about creating a `src/tests` or `src/experiments` directory where you create a test class that inherits from your main model class, and then override one or more of the model methods. This would allow you to save and archive specific experiments that differ from your main model, while allowing you to return and re-run exactly the same experiment at a later date.

Anything is possible. Have fun!

---

<a name="acknowledgements"></a>
## Acknowledgements

Core contributors:
* Chris Jackett
* Abdelwahed Khamis
* Dan MacKinley

The creation of this learning resource was a collaborative effort within the Machine Learning and Artifical Intelligence Future Science Platform (MLAI FSP). The following people contributed content, suggestions, corrections, and generally improved the quality of this resource:

* Tom Blau
* Muming Zhao
* Kaya Baxter
* Ondrej Hlinka