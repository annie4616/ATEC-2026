#!/bin/bash
# Waits for both heuristic-ablation jobs, then writes the consolidated GRASP_SUMMARY.md
set -u
RW=/hdd0/yewon/ATEC2026_Simulation_Challenge/real_world_preliminary/ATEC2026_Simulation_Challenge
D=$RW/grasp_rl/detached
until [ -f "$D/heur5.done" ] && [ -f "$D/heur4.done" ]; do sleep 120; done
/home/yewon/.conda/envs/isaaclab/bin/python $RW/grasp_rl/scripts/summarize_grasp.py
