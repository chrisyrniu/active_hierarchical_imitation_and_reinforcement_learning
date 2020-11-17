# AHIRL in Ant-Maze Environment
Active Hierarchical Imitation and Reinforcement Learning (AHIRL) implementation in Ant-Maze environment.

Please see our [report](https://chrisyrniu.github.io/files/report_ahirl.pdf) of this work.

## Run Training
`sh run_train.sh`

## Run Testing
`sh run_test.sh`

## Check Training Process and Results
* Use tensorboardx
* Use plot_script.py and saved log file:

`python plot_script.py saved/ name Reward`

`python plot_script.py saved/ name Steps-Taken`

## Acknowledgement
This repository is revised based on [Hierarchial Actor-Critic](https://github.com/andrew-j-levy/Hierarchical-Actor-Critc-HAC-)
