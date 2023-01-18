ITER=5

# ant
for (( i=0; i<${ITER}; i++ ))
do
    python ./examples/train_shac.py --cfg ./examples/cfg/shac/ant.yaml --logdir ./examples/logs/ant/shac --seed ${i}
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/ant.yaml --logdir ./examples/logs/ant/grad_ppo --seed ${i} --ppo_lr_threshold 0.0 --ppo_kl_threshold 0.0
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/ant.yaml --logdir ./examples/logs/ant/grad_ppo --seed ${i} --ppo_lr_threshold 0.0001 --ppo_kl_threshold 0.0001
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/ant.yaml --logdir ./examples/logs/ant/grad_ppo --seed ${i} --ppo_lr_threshold 0.0005 --ppo_kl_threshold 0.0005
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/ant.yaml --logdir ./examples/logs/ant/grad_ppo --seed ${i} --ppo_lr_threshold 0.001 --ppo_kl_threshold 0.001
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/ant.yaml --logdir ./examples/logs/ant/grad_ppo --seed ${i} --ppo_lr_threshold 0.005 --ppo_kl_threshold 0.005
done


# cartpole
for (( i=0; i<${ITER}; i++ ))
do
    python ./examples/train_shac.py --cfg ./examples/cfg/shac/cartpole_swing_up.yaml --logdir ./examples/logs/cartpole_swing_up/shac --seed ${i}
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cartpole_swing_up.yaml --logdir ./examples/logs/cartpole_swing_up/grad_ppo --seed ${i} --ppo_lr_threshold 0.0 --ppo_kl_threshold 0.0
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cartpole_swing_up.yaml --logdir ./examples/logs/cartpole_swing_up/grad_ppo --seed ${i} --ppo_lr_threshold 0.0001 --ppo_kl_threshold 0.0001
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cartpole_swing_up.yaml --logdir ./examples/logs/cartpole_swing_up/grad_ppo --seed ${i} --ppo_lr_threshold 0.0005 --ppo_kl_threshold 0.0005
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cartpole_swing_up.yaml --logdir ./examples/logs/cartpole_swing_up/grad_ppo --seed ${i} --ppo_lr_threshold 0.001 --ppo_kl_threshold 0.001
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cartpole_swing_up.yaml --logdir ./examples/logs/cartpole_swing_up/grad_ppo --seed ${i} --ppo_lr_threshold 0.005 --ppo_kl_threshold 0.005
done


# cheetah
for (( i=0; i<${ITER}; i++ ))
do
    python ./examples/train_shac.py --cfg ./examples/cfg/shac/cheetah.yaml --logdir ./examples/logs/cheetah/shac --seed ${i}
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cheetah.yaml --logdir ./examples/logs/cheetah/grad_ppo --seed ${i} --ppo_lr_threshold 0.0 --ppo_kl_threshold 0.0
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cheetah.yaml --logdir ./examples/logs/cheetah/grad_ppo --seed ${i} --ppo_lr_threshold 0.0001 --ppo_kl_threshold 0.0001
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cheetah.yaml --logdir ./examples/logs/cheetah/grad_ppo --seed ${i} --ppo_lr_threshold 0.0005 --ppo_kl_threshold 0.0005
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cheetah.yaml --logdir ./examples/logs/cheetah/grad_ppo --seed ${i} --ppo_lr_threshold 0.001 --ppo_kl_threshold 0.001
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/cheetah.yaml --logdir ./examples/logs/cheetah/grad_ppo --seed ${i} --ppo_lr_threshold 0.005 --ppo_kl_threshold 0.005
done


# hopper
for (( i=0; i<${ITER}; i++ ))
do
    python ./examples/train_shac.py --cfg ./examples/cfg/shac/hopper.yaml --logdir ./examples/logs/hopper/shac --seed ${i}
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/hopper.yaml --logdir ./examples/logs/hopper/grad_ppo --seed ${i} --ppo_lr_threshold 0.0 --ppo_kl_threshold 0.0
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/hopper.yaml --logdir ./examples/logs/hopper/grad_ppo --seed ${i} --ppo_lr_threshold 0.0001 --ppo_kl_threshold 0.0001
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/hopper.yaml --logdir ./examples/logs/hopper/grad_ppo --seed ${i} --ppo_lr_threshold 0.0005 --ppo_kl_threshold 0.0005
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/hopper.yaml --logdir ./examples/logs/hopper/grad_ppo --seed ${i} --ppo_lr_threshold 0.001 --ppo_kl_threshold 0.001
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/hopper.yaml --logdir ./examples/logs/hopper/grad_ppo --seed ${i} --ppo_lr_threshold 0.005 --ppo_kl_threshold 0.005
done


# humanoid
for (( i=0; i<${ITER}; i++ ))
do
    python ./examples/train_shac.py --cfg ./examples/cfg/shac/humanoid.yaml --logdir ./examples/logs/humanoid/shac --seed ${i}
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/humanoid.yaml --logdir ./examples/logs/humanoid/grad_ppo --seed ${i} --ppo_lr_threshold 0.0 --ppo_kl_threshold 0.0
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/humanoid.yaml --logdir ./examples/logs/humanoid/grad_ppo --seed ${i} --ppo_lr_threshold 0.0001 --ppo_kl_threshold 0.0001
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/humanoid.yaml --logdir ./examples/logs/humanoid/grad_ppo --seed ${i} --ppo_lr_threshold 0.0005 --ppo_kl_threshold 0.0005
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/humanoid.yaml --logdir ./examples/logs/humanoid/grad_ppo --seed ${i} --ppo_lr_threshold 0.001 --ppo_kl_threshold 0.001
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/humanoid.yaml --logdir ./examples/logs/humanoid/grad_ppo --seed ${i} --ppo_lr_threshold 0.005 --ppo_kl_threshold 0.005
done


# snu_humanoid
for (( i=0; i<${ITER}; i++ ))
do
    python ./examples/train_shac.py --cfg ./examples/cfg/shac/snu_humanoid.yaml --logdir ./examples/logs/snu_humanoid/shac --seed ${i}
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/snu_humanoid.yaml --logdir ./examples/logs/snu_humanoid/grad_ppo --seed ${i} --ppo_lr_threshold 0.0 --ppo_kl_threshold 0.0
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/snu_humanoid.yaml --logdir ./examples/logs/snu_humanoid/grad_ppo --seed ${i} --ppo_lr_threshold 0.0001 --ppo_kl_threshold 0.0001
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/snu_humanoid.yaml --logdir ./examples/logs/snu_humanoid/grad_ppo --seed ${i} --ppo_lr_threshold 0.0005 --ppo_kl_threshold 0.0005
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/snu_humanoid.yaml --logdir ./examples/logs/snu_humanoid/grad_ppo --seed ${i} --ppo_lr_threshold 0.001 --ppo_kl_threshold 0.001
    python ./examples/train_rl.py --cfg ./examples/cfg/grad_ppo/snu_humanoid.yaml --logdir ./examples/logs/snu_humanoid/grad_ppo --seed ${i} --ppo_lr_threshold 0.005 --ppo_kl_threshold 0.005
done