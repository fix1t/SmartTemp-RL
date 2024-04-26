# SmartTemp-RL

## Overview

SmartTemp-RL is a research project focused on applying deep reinforcement learning algorithms to optimize temperature control systems. The project utilizes two main RL algorithms: Deep Q-Learning (DQL) and Proximal Policy Optimization (PPO). This README provides instructions on how to set up the environment, run training sessions, test models, and evaluate different configurations as discussed in the thesis.

## Repository Structure

- **src/**: Contains all the source code for the SmartTemp-RL project.
- **doc/**: Contains diagrams of different moduls.
- **results/**: Contains data from configurations testing and final evaluation of algorithms.

## Getting Started

### Prerequisites

Before running the scripts, ensure that you have Python and Make installed on your system. The project's dependencies are managed using a virtual environment.

### Setup

Navigate to the source code directory:

```shell
cd src
```

Create a Python virtual environment and install the required dependencies:

```shell
make install
```

To activate the environment for running scripts directly run command:

```shell
source venv/bin/activate
```

### Running the Algorithms

To train the models using the default configurations for DQL and PPO, use the following commands:

```shell
make dql
make ppo
```

To test the latest learned models:

```shell
make dql-test
make ppo-test
```

### Parameter Tuning

Run all the configurations used in the Parameter Tuning section of the thesis:

```shell
make run-configurations-dql # For Hyperparameter combinations
make run-configurations-ppo

make run-configurations-nn-dql # For Neural Network architecture combinations
make run-configurations-nn-ppo

make run-configurations-top-dql # Run combinations of best performing
make run-configurations-top-ppo # configurations (Hyperparameters + NN)
```

### Final Evaluation

To run the final evaluation script for the DQL algorithm:

```shell
make final-evaluation-dql
make final-evaluation-ppo
```

## Additional Information

The Makefile automates the setup, training, testing, and evaluation processes, ensuring that the project can be run with minimal manual setup. For more detailed information on the specific parameters and configurations, please refer to the thesis document or the source code comments.

For usage of a specific script run `python3 <path-to-script> --help`
