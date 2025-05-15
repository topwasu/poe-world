#!/bin/bash

# Define arrays for games and methods
games=("pong_agent" "montezuma_agent")
methods=("worldcoder" "poe")
modes=("full" "partial")
seeds=("0" "1" "2")
# Loop through each game
for game in "${games[@]}"; do
    # Loop through each method
    for method in "${methods[@]}"; do
        # Loop through each mode
        for mode in "${modes[@]}"; do
            echo "Running $game with $method and $mode"
            for seed in "${seeds[@]}"; do
                python run.py --config-name="$game" method="$method" post_synthesis_mode=evaluate eval.set=training eval.mode="$mode" seed="$seed"
            done
        done
    done
done

games=("pong_agent" "pong_alt_agent" "montezuma_agent" "montezuma_alt_agent")
# Loop through each game
for game in "${games[@]}"; do
    # Loop through each method
    for method in "${methods[@]}"; do
        # Loop through each mode
        for mode in "${modes[@]}"; do
            echo "Running $game with $method and $mode with random seeds"
            for seed in "${seeds[@]}"; do
                python run.py --config-name="$game" method="$method" post_synthesis_mode=evaluate eval.set=random eval.mode="$mode" seed="$seed"
            done
        done
    done
done
