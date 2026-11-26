#!/bin/bash
# Real-world robustness finetunes on GPU 4 (grasp1m): 0.23+0.24, then 0.25+0.3.
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
  CUDA_VISIBLE_DEVICES=4 $PY $TRAIN --headless --device cuda:0 --task ATEC-Grasp-B2Piper-Stage2-v0 \
    --crouch_depth "$1" --num_envs 1024 --max_iterations 1500 --run_name "$4" \
    --width_range 0.030,0.066 --length_range 0.030,0.250 --height_range 0.020,0.130 \
    --resume "$(resolve_iter $2 $3)" > $D/train_$4.log 2>&1
}

run_one 0.23 grasp1k_c023 1200 grasp1m_c023 & P1=$!
sleep 90
run_one 0.24 grasp1k_c024 1499 grasp1m_c024 & P2=$!
wait $P1 $P2
run_one 0.25 grasp1k_c025 1200 grasp1m_c025 & P3=$!
sleep 90
run_one 0.3  grasp1k_c030 300  grasp1m_c030 & P4=$!
wait $P3 $P4
echo "GPU4_REAL_TRAINS_DONE"
