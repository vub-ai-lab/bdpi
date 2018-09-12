# Sample-Efficient Reinforcement Learning with Bootstrapped Dual Policy Iteration

This repository contains the complete implementation of the Bootstrapped Dual Policy Iteration algorithm we developed over the past year and a half. The repository also contains scripts to re-run and re-plot our experiments.

## Organization

The following components are available in this repository

* The complete Python 3 source code of our algorithm;
* Two OpenAI Gym environments: Five Rooms and Table (FrozenLake8x8 is available in the OpenAI Gym);
* Scripts to re-run all our experiments in a fully automated way.

The files are organized as follows:

* `gym_envs/`: Gym environments.
* `main.py`: A simple RL agent that performs actions in a Gym environment, and learns using BDPI
* `bdpi.py`: The BDPI learning algorithm (actor, critics, and glue between them)
* `experiments_gym.sh`: A script that produces a job description for a given environment. Run an experiment with `./experiments_gym.sh table && cat commands_table.sh | parallel -jCORES`

## Dependencies

Reproducing our results require a computer with the following components:

* A recent Linux distribution
* Python 3, with `lzo` and PyTorch
* GNU Parallel
* Gnuplot
