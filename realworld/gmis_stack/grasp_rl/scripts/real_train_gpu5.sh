#!/bin/bash
# Real-world robustness finetunes on GPU 5 (grasp1m): perception noise (rgbl) + sensor
# corruption + dynamics randomization. Resumes each depth's LADDER-SELECTED rgbl winner.
set -u
RW=/hdd0/yewon/ATEC2026_Simulation_Challenge/real_world_preliminary/ATEC2026_Simulation_Challenge
PY=/home/yewon/.conda/envs/isaaclab/bin/python
TRAIN=$RW/grasp_rl/scripts/train_grasp.py
R=$RW/grasp_rl/runs
D=$RW/grasp_rl/detached
export GRASP_PERC_NOISE=rgbl
export GRASP_REAL_NOISE=1
export GRASP_BANK_KOFILTER=1
export GRASP_CAMHIT_GRACE=10
resolve_iter() { local d; d=$(ls -td $R/${1}_2* | head -1); echo "$d/model_${2}.pt"; }
cd $RW

run_one() {  # depth donor_prefix donor_iter run_name
  CUDA_VISIBLE_DEVICES=5 $PY $TRAIN --headless --device cuda:0 --task ATEC-Grasp-B2Piper-Stage2-v0 \
    --crouch_depth "$1" --num_envs 1024 --max_iterations 1500 --run_name "$4" \
    --width_range 0.030,0.066 --length_range 0.030,0.250 --height_range 0.020,0.130 \
    --resume "$(resolve_iter $2 $3)" > $D/train_$4.log 2>&1
}

run_one 0.2  grasp1k_c020 900  grasp1m_c020 & P1=$!
sleep 90
run_one 0.21 grasp1k_c021 1200 grasp1m_c021 & P2=$!
wait $P1 $P2
run_one 0.22 grasp1k_c022 300  grasp1m_c022
echo "GPU5_REAL_TRAINS_DONE"
