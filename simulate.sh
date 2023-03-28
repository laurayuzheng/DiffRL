# TrafficPaceCarEnv
# TrafficRoundaboutEnv

# python examples/test_env.py --env TrafficRingEnv --num-envs 4 --render

DEVICE='cpu'
ROUNDABOUT_ENV_CFG='examples/cfg/grad_ppo/traffic_roundabout.yaml' 
ROUNDABOUT_ENV_LOGDIR='examples/logs/traffic_roundabout/grad_ppo' 

PACECAR_ENV_CFG='examples/cfg/grad_ppo/traffic_single_pace_car/scenario_a.yaml'
PACECAR_ENV_LOGDIR='examples/logs/traffic_pacecar/grad_ppo/scenario_a'

RING_ENV_CFG='examples/cfg/grad_ppo/traffic_ring.yaml'
RING_ENV_LOGDIR='examples/logs/traffic_ring/grad_ppo'

# Simulate GradPPO on Traffic Roundabout
python examples/train_rl.py \
    --rl_device=${DEVICE} \
    --cfg=${RING_ENV_CFG} \
    --checkpoint=${RING_ENV_LOGDIR}/03-22-2023-21-02-27/nn/last_df_traffic_ring_grad_ppoep900rew[246.68898].pth \
    --play \
    --seed 1 \
    --render 

# python examples/train_rl.py \
#     --rl_device=${DEVICE} \
#     --cfg=${PACECAR_ENV_CFG} \
#     --checkpoint=${PACECAR_ENV_LOGDIR}/03-20-2023-17-22-31/nn/df_traffic_pace_car_grad_ppo.pth \
#     --play \
#     --render