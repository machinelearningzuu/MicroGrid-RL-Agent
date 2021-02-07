# MicroGrid-RL-Agent

Main Idea behind the project is develop a method for minimize electric grid consumption and increase the utilization of renewable energy generation

# Project Overview

### Details of Main Agent
![Details of Main Agent](https://github.com/1zuu/MicroGrid-RL-Agent/blob/main/WhatsApp%20Image%202021-02-06%20at%2018.18.32.jpeg)

### Details of Power Shiftable Agents
![Details of Power Shiftable Agents](https://github.com/1zuu/MicroGrid-RL-Agent/blob/main/WhatsApp%20Image%202021-02-06%20at%2018.19.09.jpeg)

There are 3 power sources MicroGrid, SolarPower, battery used for this project. Microgrid is the main power source,  main network for delivering 
electricity from producers to consumers simply which we pay for what we use. Solar power is the intermedant source because its weather dependent. 
then finally rechargeable battery which can be recharge using solar power.Objective is to balance the power consumtion of these 3 sources while 
reducing the grid constmption. The reason for using a battery is if solar power is not available its more cost effective battery consumption than grid consumption.

the model consists 3 states  (solar_power, battery_value, demand) for a particular time step. there are 
2 actions battery charging or discharging and based on that it will decide how much it cost for
the grid. cost of the grid is much higher in time periods like 6.30 pm to 10 pm. So reward functions 
assigned to minimize the cost of the grid.
  
# Techniques
  - Deep Learning
  - Reinforcement Learning
  - Dense regression networks
  - Q-learning 
  - Deep Q-learning
# Tools

* TensorFlow - Deep Learning Model
* pandas - Data Extraction and Preprocessing
* numpy - numerical computations
* scikit learn - Advanced preprocessing and Machine Learning Models
* matplotlib - Visualization

### Installation

Install the dependencies and conda environment

```sh
$ conda create -n envname python=python_version
$ activate envname 
$ conda install -c anaconda tensorflow-gpu
$ conda install -c anaconda pandas
$ conda install -c anaconda matplotlib
$ conda install -c anaconda scikit-learn
```
