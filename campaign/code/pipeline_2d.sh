#!/bin/bash
# Hands-off 2D pipeline on node 109: wait for 4x4 training data,
# train oseq_rope on mixed 12q+16q, then evaluate at 16/20/24 qubits.
# Run from /data/diffushadow_2d/code. Good tenant: pinned to one GPU.
set -u
cd /data/diffushadow_2d/code
PY=/data/conda_envs/mypy312/bin/python
export CUDA_VISIBLE_DEVICES=${GPU_ID:-3}
mkdir -p checkpoints2d results2d logs2d

echo "[pipeline] waiting for 4x4 training data..."
until grep -qs "saved training json" logs2d/gen_4x4.log; do sleep 60; done
echo "[pipeline] 4x4 data ready; training starts $(date)"

$PY train_oseq.py \
  --model_type oseq_rope \
  --data_path data2d/tfi2d_4x3_train.json data2d/tfi2d_4x4_train.json \
  --qubits 12 16 \
  --hidden_dim 128 --layer_num 4 --head_num 8 \
  --rope_scaling_type none --rope_scaling_factor 1.0 --rope_theta 10000 \
  --epochs 100 --batch_size 256 \
  --model_save_path checkpoints2d/tfi2d_oseq_rope.pth \
  --loss_save_path logs2d/tfi2d_loss.json \
  > logs2d/train_2d.log 2>&1
rc=$?
echo "[pipeline] training finished rc=$rc $(date)"
[ $rc -ne 0 ] && exit $rc

for LY in 4 5 6; do
  if [ "$LY" = "4" ]; then EX=data2d/tfi2d_4x4_eval_exact.npz; else EX=data2d/tfi2d_4x${LY}_exact.npz; fi
  echo "[pipeline] waiting for exact cache $EX..."
  until [ -f "$EX" ]; do sleep 60; done
  echo "[pipeline] eval 4x${LY} starts $(date)"
  $PY tfi2d/eval_tfi2d.py \
    --model_path checkpoints2d/tfi2d_oseq_rope_final.pth \
    --lx 4 --ly $LY \
    --exact_npz "$EX" \
    --out_npz results2d/tfi2d_eval_4x${LY}.npz \
    --sample_size 20000 --diffusion_steps 4 --gen_batch_size 2048 \
    > logs2d/eval_4x${LY}.log 2>&1
  echo "[pipeline] eval 4x${LY} rc=$? $(date)"
done
echo "[pipeline] ALL DONE $(date)"
