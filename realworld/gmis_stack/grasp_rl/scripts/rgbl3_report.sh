#!/bin/bash
# Waits for both selection passes, then regenerates FINAL_REPORT_RGBL.txt from the
# (now best-checkpoint) korgbl after-logs and rgbl_d video logs.
set -u
RW=/hdd0/yewon/ATEC2026_Simulation_Challenge/real_world_preliminary/ATEC2026_Simulation_Challenge
D=$RW/grasp_rl/detached
until [ -f "$D/rgbl3_gpu5.done" ] && [ -f "$D/rgbl3_gpu4.done" ]; do sleep 120; done
bash $RW/grasp_rl/scripts/rgbl_report.sh
