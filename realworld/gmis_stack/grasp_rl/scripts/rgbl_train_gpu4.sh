#!/bin/bash
# RGB+LiDAR-noise finetunes on GPU 4: 0.23 + 0.24 in parallel, then 0.25 + 0.3.
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
  CUDA_VISIBLE_DEVICES=4 $PY $TRAIN --headless --device cuda:0 --task ATEC-Grasp-B2Piper-Stage2-v0 \
    --crouch_depth "$1" --num_envs 1024 --max_iterations 1500 --run_name "$3" \
    --width_range 0.030,0.066 --length_range 0.030,0.250 --height_range 0.020,0.130 \
    --resume "$(resolve $2)" > $D/train_$3.log 2>&1
}

run_one 0.23 grasp1i_c023 grasp1k_c023 & P1=$!
sleep 90
run_one 0.24 grasp1h_c024 grasp1k_c024 & P2=$!
wait $P1 $P2
run_one 0.25 grasp1i_c025 grasp1k_c025 & P3=$!
sleep 90
run_one 0.3  grasp1h_c030 grasp1k_c030 & P4=$!
wait $P3 $P4
echo "GPU4_RGBL_TRAINS_DONE"
