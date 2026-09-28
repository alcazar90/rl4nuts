#!/bin/bash

# OMP_NUM_THREADS=1 poetry run python -m rl4nuts_utils.benchmark \
#     --env-ids "CartPole-v1" \
#     --command "poetry run python rl4nuts/reinforce.py --track --capture-video" \
#     --num-seeds 3 \
#     --start-seed 666 \
#     --workers 4

# The effect of discount factor gamma 
# TODO: add exp_name to identify these runs later
for gamma in 0.1 0.5 0.7 0.8 0.9 0.99 0.999; do
    OMP_NUM_THREADS=1 poetry run python -m rl4nuts_utils.benchmark \
        --env-ids "CartPole-v1" \
        --command "poetry run python rl4nuts/reinforce.py --track --capture-video --total_timesteps 225000 --gamma $gamma" \
        --num-seeds 3 \
        --start-seed 666 \
        --workers 5
done

echo "All experiments completed"
