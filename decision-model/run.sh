#!/bin/bash

# run mujoco
python experiment.py --K 8 --env hopper --dataset medium --embed_dim 64
python experiment.py --K 8 --env walker2d --dataset medium --embed_dim 64
python experiment.py --K 8 --env halfcheetah --dataset medium --embed_dim 64
# run antmaze
python experiment.py --K 8 --env antmaze --dataset umaze --embed_dim 128
python experiment.py --K 8 --env antmaze --dataset umaze-diverse --embed_dim 128
# run kitchen
python experiment.py --K 8 --env mini_kitchen_microwave_kettle_light_slider --dataset complete --embed_dim 128
python experiment.py --K 8 --env kitchen_microwave_kettle_light_slider --dataset partial --embed_dim 128
python experiment.py --K 8 --env kitchen_microwave_kettle_bottomburner_light --dataset mixed --embed_dim 128

