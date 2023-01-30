ITER=10

# scenario a
for (( i=0; i<${ITER}; i++ ))
do
    python ./examples/train_shac.py --cfg ./examples/cfg/shac/traffic_single_pace_car/scenario_a.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_a/shac --seed ${i}
    python ./examples/train_rl.py --cfg ./examples/cfg/ppo/traffic_single_pace_car/scenario_a.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_a/ppo --seed ${i}
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/traffic_single_pace_car/scenario_a.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_a/grad_ppo --seed ${i}
done

# scenario b
for (( i=0; i<${ITER}; i++ ))
do
    python ./examples/train_shac.py --cfg ./examples/cfg/shac/traffic_single_pace_car/scenario_b.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_b/shac --seed ${i}
    python ./examples/train_rl.py --cfg ./examples/cfg/ppo/traffic_single_pace_car/scenario_b.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_b/ppo --seed ${i}
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/traffic_single_pace_car/scenario_b.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_b/grad_ppo --seed ${i}
done

# scenario c
for (( i=0; i<${ITER}; i++ ))
do
    python ./examples/train_shac.py --cfg ./examples/cfg/shac/traffic_single_pace_car/scenario_c.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_c/shac --seed ${i}
    python ./examples/train_rl.py --cfg ./examples/cfg/ppo/traffic_single_pace_car/scenario_c.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_c/ppo --seed ${i}
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/traffic_single_pace_car/scenario_c.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_c/grad_ppo --seed ${i}
done

# scenario d
for (( i=0; i<${ITER}; i++ ))
do
    python ./examples/train_shac.py --cfg ./examples/cfg/shac/traffic_single_pace_car/scenario_d.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_d/shac --seed ${i}
    python ./examples/train_rl.py --cfg ./examples/cfg/ppo/traffic_single_pace_car/scenario_d.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_d/ppo --seed ${i}
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/traffic_single_pace_car/scenario_d.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_d/grad_ppo --seed ${i}
done

# scenario e
for (( i=0; i<${ITER}; i++ ))
do
    python ./examples/train_shac.py --cfg ./examples/cfg/shac/traffic_single_pace_car/scenario_e.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_e/shac --seed ${i}
    python ./examples/train_rl.py --cfg ./examples/cfg/ppo/traffic_single_pace_car/scenario_e.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_e/ppo --seed ${i}
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/traffic_single_pace_car/scenario_e.yaml --logdir ./examples/logs/traffic_single_pace_car/scenario_e/grad_ppo --seed ${i}
done
