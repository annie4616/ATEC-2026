#!/bin/bash
# RGB+LiDAR-noise finetunes on GPU 5: depths 0.2 + 0.21 in parallel, then 0.22 solo.
# grasp1k generation: resume each depth's best noise-free checkpoint, train WITH
# GRASP_PERC_NOISE=rgbl so the policy learns to grasp from biased/held estimates.
set -u
RW=/hdd0/yewon/ATEC2026_Simulation_Challenge/real_world_preliminary/ATEC2026_Simulation_Challenge
PY=/home/yewon/.conda/envs/isaaclab/bin/python
TRAIN=$RW/grasp_rl/scripts/train_grasp.py
R=$RW/grasp_rl/runs
D=$RW/grasp_rl/detached
export GRASP_PERC_NOISE=rgbl
export GRASP_BANK_KOFILTER=1
export GRASP_CAMHIT_GRACE=10
resolve() { local d; d=$(ls -td $R/${1}_2* | head -1); echo "$d/$(ls "$d" | grep -oE 'model_[0-9]+\.pt' | sort -t_ -k2 -n | tail -1)"; }
cd $RW

run_one() {  # depth donor_prefix run_name
  CUDA_VISIBLE_DEVICES=5 $PY $TRAIN --headless --device cuda:0 --task ATEC-Grasp-B2Piper-Stage2-v0 \
    --crouch_depth "$1" --num_envs 1024 --max_iterations 1500 --run_name "$3" \
    --width_range 0.030,0.066 --length_range 0.030,0.250 --height_range 0.020,0.130 \
    --resume "$(resolve $2)" > $D/train_$3.log 2>&1
}

run_one 0.2  grasp1h_c020 grasp1k_c020 & P1=$!
sleep 90
run_one 0.21 grasp1i_c021 grasp1k_c021 & P2=$!
wait $P1 $P2
run_one 0.22 grasp1h_c022 grasp1k_c022
echo "GPU5_RGBL_TRAINS_DONE"
