#!/bin/bash
# 파인튜닝(PID) 완료 대기 후 final.pt를 submission_e/policy_act.pt로 배포
TRAIN_PID=1334903
CKPT_DIR=/hdd0/sohyun/atec_2026_tesk_e/ATEC2026_Simulation_Challenge/scripts/act/runs/taske_finetune_strict21/checkpoints
SUB=/hdd0/sohyun/atec_2026_tesk_e/ATEC2026_Simulation_Challenge/submission_e

# 학습 프로세스가 끝날 때까지 대기
while kill -0 $TRAIN_PID 2>/dev/null; do sleep 30; done
echo "[deploy] training finished"

# final.pt 우선, 없으면 best_loss.pt
if [ -f "$CKPT_DIR/final.pt" ]; then SRC="$CKPT_DIR/final.pt"; else SRC="$CKPT_DIR/best_loss.pt"; fi
cp "$SRC" "$SUB/policy_act.pt"
echo "[deploy] copied $SRC -> $SUB/policy_act.pt"
ls -la "$SUB/policy_act.pt"
