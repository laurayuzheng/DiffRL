
# Roundabout (GradPPO)
# python ./examples/train_rl.py \
#     --cfg examples/cfg/grad_ppo/traffic_ring.yaml \
#     --logdir ./examples/logs/traffic_ring/grad_ppo \
#     --seed 1 \
#     --rl_device cpu

# Ring (PPO)
python ./examples/train_rl.py \
    --cfg examples/cfg/ppo/traffic_ring.yaml \
    --logdir ./examples/logs/traffic_ring/ppo \
    --seed 2 \
    --rl_device cpu

# PaceCar (GradPPO)
# python ./examples/train_rl.py \
    # --cfg examples/cfg/grad_ppo/traffic_single_pace_car/scenario_a.yaml \
    # --logdir ./examples/logs/traffic_pacecar/grad_ppo/scenario_a \
    # --seed 0