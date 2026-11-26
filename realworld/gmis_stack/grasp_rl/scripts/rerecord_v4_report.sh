#!/bin/bash
# Waits for both v4 re-record jobs, then regenerates FINAL_REPORT_V4.txt from the
# (unchanged) ko logs + fresh video logs.
set -u
RW=/hdd0/yewon/ATEC2026_Simulation_Challenge/real_world_preliminary/ATEC2026_Simulation_Challenge
D=$RW/grasp_rl/detached
until [ -f "$D/rerecord_v4_gpu5.done" ] && [ -f "$D/rerecord_v4_gpu4.done" ]; do sleep 60; done
bash $RW/grasp_rl/scripts/final_report_v4.sh
