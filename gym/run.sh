#!/bin/bash

envs=("hopper" "walker2d" "halfcheetah")
datasets=("medium" "medium-replay" "medium-expert")
token_mixers=("line" "conv")

for token_mixer in "${token_mixers[@]}"
do
    for env in "${envs[@]}"
    do
        for dataset in "${datasets[@]}"
        do
            for _ in {1..5}
            do
                python experiment.py --K 8 --env $env --dataset $dataset --embed_dim 128 --max_iters 10 --num_eval_episodes 100 --token_mixer $token_mixer
            done
        done
    done
done

envs=("hopper" "walker2d" "halfcheetah")
datasets=("medium" "medium-replay" "medium-expert")
token_mixers=("double-line" "double-conv")

for token_mixer in "${token_mixers[@]}"
do
    for env in "${envs[@]}"
    do
        for dataset in "${datasets[@]}"
        do
            for _ in {1..5}
            do
                python experiment.py --K 8 --env $env --dataset $dataset --embed_dim 64 --max_iters 10 --num_eval_episodes 100 --token_mixer $token_mixer
            done
        done
    done
done

envs=("antmaze")
datasets=("umaze" "umaze-diverse")
token_mixers=("line" "conv")

for token_mixer in "${token_mixers[@]}"
do
    for env in "${envs[@]}"
    do
        for dataset in "${datasets[@]}"
        do
            for _ in {1..5}
            do
                python experiment.py --K 8 --env $env --dataset $dataset --embed_dim 128 --max_iters 10 --num_eval_episodes 100 --token_mixer $token_mixer
            done
        done
    done
done

envs=("antmaze")
datasets=("umaze" "umaze-diverse")
token_mixers=("double-line" "double-conv")

for token_mixer in "${token_mixers[@]}"
do
    for env in "${envs[@]}"
    do
        for dataset in "${datasets[@]}"
        do
            for _ in {1..5}
            do
                python experiment.py --K 8 --env $env --dataset $dataset --embed_dim 64 --max_iters 10 --num_eval_episodes 100 --token_mixer $token_mixer
            done
        done
    done
done
