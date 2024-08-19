#!/bin/bash

games=("Breakout" "Pong" "Qbert", "Seaquest")
token_mixers=("line" "conv" "double-line" "double-conv")

for token_mixer in "${token_mixers[@]}"
do
    for game in "${games[@]}"
    do
        for seed in 123 231 312
            do
                python experiment.py --seed $seed --context_length 20 --game $game --batch_size 64 --token_mixer $token_mixer --lr 6e-4 --decay
        done
    done
done
