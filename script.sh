ITER=5


for (( i=0; i<${ITER}; i++ ))
do
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cartpole_swing_up.yaml --logdir ./examples/logs/cartpole/grad_ppo --gi_alpha 0.0
done


for (( i=0; i<${ITER}; i++ ))
do
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cartpole_swing_up.yaml --logdir ./examples/logs/cartpole/grad_ppo --gi_alpha 1.0
done


# for (( i=0; i<${ITER}; i++ ))
# do
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cartpole_swing_up.yaml --logdir ./examples/logs/cartpole/grad_ppo --gi_alpha 0.0
# done

# for (( i=0; i<${ITER}; i++ ))
# do
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cartpole_swing_up.yaml --logdir ./examples/logs/cartpole/grad_ppo --gi_alpha 0.2
# done


# for (( i=0; i<${ITER}; i++ ))
# do
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cartpole_swing_up.yaml --logdir ./examples/logs/cartpole/grad_ppo --gi_alpha 0.4
# done

# for (( i=0; i<${ITER}; i++ ))
# do
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cartpole_swing_up.yaml --logdir ./examples/logs/cartpole/grad_ppo --gi_alpha 0.002
# done


# for (( i=0; i<${ITER}; i++ ))
# do
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cartpole_swing_up.yaml --logdir ./examples/logs/cartpole/grad_ppo --gi_alpha 0.004
# done


# for (( i=0; i<${ITER}; i++ ))
# do
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cartpole_swing_up.yaml --logdir ./examples/logs/cartpole/grad_ppo --gi_alpha 0.01
# done


# for (( i=0; i<${ITER}; i++ ))
# do
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cartpole_swing_up.yaml --logdir ./examples/logs/cartpole/grad_ppo --gi_alpha 0.02
# done


# for (( i=0; i<${ITER}; i++ ))
# do
#     python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cartpole_swing_up.yaml --logdir ./examples/logs/cartpole/grad_ppo --gi_alpha 0.04
# done
