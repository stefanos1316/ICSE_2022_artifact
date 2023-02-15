# Performance-Energy for Deep Learning Framework (PrEngDL)
This is a framework to faciliate the collection of energy and run-time performance measurements
for the DeepLearningExamples repository.
At the moment, it can collect the energy and run-time perfomrance of six different DL models.

# Prerequisites
- Ubuntu or Debian distro (we have tested it on Debian 10.04).
- GPU with at least 8 GiB of memory

# Execution environment
- We used NVIDIA Quadro P4000 GPU.
- The GPU was installed on a server equipped with two 6th generation Intel Xeon Gold 6154 CPU with 72 logical cores and 96 GB of main memory.
- We used Ubuntu 18.04 and Cuda version 11.2.
- The scripts were written using Python 3.7.

# Content
- `tools/compile_results.sh:` a shell script to collect the mean values of our tests, the results directory is required as a command line argument (for training or inference)
- `configs:` holds all the configurations that we used to perform our experiments
- `data/cProfile_results:` houses of the profiled functions from running the cProfile on the models' training and inference process.
- `DeepLearningExamples:` the repository with the tasks that we use in our experiment.
- `data/function_call_graphs:` houses the function call graphs from each profiled (cProfile) task in our experiment.
- `tools/governor.sh:` a small script that allows a user to switch the power governoer (po -> power saving, pe -> performance).
- `install_and_configure.yml:` the ansible script responsible of deploying the necessary software to run our experiment.
- `data/measurements_inference:` has the inference data from all the tasks of our experiment.
- `data/measurements_inference:` has the training data from all the tasks of our experiment.
- `run.sh:` the main script that runs all of our tasks. 

# Installation
After downloading the repository, execute the Ansible script:

```bash
apt install ansible -y
```
The execute the following to install docker and add the current user to the docker group
```bash
sudo ansible-playbook install_and_configure.yml --tags "docker" --extra-vars "user=system_user"
```
After, you have to log out and login from your system to apply the changes.
Then run the ansible script with the following flags, also do not worry about the password
field since we use bycrypt and no plain text is going to be written in the corresponding file:
```bash
sudo ansible-playbook install_and_configure.yml --tags "experiment" --extra-vars "user=system_user"
```


# Run experiments
To run the experiments execute run the following:
```bash
./run_experiments.sh -r 10 -t [train|test]
```
The `-r` defines how many times a task should run (to get statistical measurements),
while the `-t` presents a training or inference task is going to be executed.
When the experiments is done,
two directories will appear in the root directory of the project with all the results (`measurements_train` and `measurements_inference`)


# Compile results
After executing the training and inference process,
execute the following command to get statistical results
```bash
bash compile_results.sh ./results_directory
```
The `./results_directory` should be the path where the the results are stored
from the `run_experiments.sh` script.
