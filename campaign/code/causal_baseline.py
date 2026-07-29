#!/usr/bin/env python3
"""WS3: causal (autoregressive) twin of Diffushadow's oseq_rope model.

Controlled baseline for the paper's central claim: identical token layout
[g, P1, b1, ..., PN, bN], identical embeddings/hidden size/layers/heads/RoPE,
identical training data — the ONLY differences are (i) a causal attention
mask and (ii) the autoregressive objective/decoding (fixed left-to-right
order instead of entropy-guided adaptive order).

Subcommands:
  train : AR teacher-forcing (predict b_i from the hidden state at P_i)
  eval  : left-to-right sampling + 2D TFI ZZ estimation vs exact cache
          (same estimator/median-of-means as tfi2d/eval_tfi2d.py)

Usage (node 109):
  python tfi2d/causal_baseline.py train \
      --data data2d/tfi2d_4x3_train.json data2d/tfi2d_4x4_train.json \
      --qubits 12 16 --epochs 100 --save checkpoints2d/tfi2d_causal.pth
  python tfi2d/causal_baseline.py eval \
      --model checkpoints2d/tfi2d_causal.pth --lx 4 --ly 6 \
      --exact_npz data2d/tfi2d_4x6_exact.npz --out campaign/tfi2d_causal_4x6.npz
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Qdmodel_oseq_rope import TransformerBlock, _build_llama_rotary_embedding  # noqa: E402
from eval_tfi2d import torus_pairs, shadow_zz, median_of_means  # noqa: E402


class CausalTransformerBlock(TransformerBlock):
    """Same block, causal mask injected before softmax."""

    def forward(self, x, key_padding_mask=None):
        import math
        batch, seq_len, _ = x.shape
        position_ids = torch.arange(seq_len, device=x.device,
                                    dtype=torch.long).unsqueeze(0).expand(batch, -1)
        q = self.q_proj(x).view(batch, seq_len, self.head_count, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.head_count, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.head_count, self.head_dim).transpose(1, 2)
        s = int(q.shape[2])
        if self._rope_uses_position_ids:
            cos, sin = self.rope(q, position_ids)
        elif self._rope_uses_seq_len:
            cos, sin = self.rope(q, seq_len=s)
        else:
            cos, sin = self.rope(q)
        from Qdmodel_oseq_rope import apply_rotary_pos_emb
        if self._apply_rotary_pass_position_ids:
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1)
        else:
            q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.triu(torch.ones(seq_len, seq_len, device=x.device,
                                       dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(causal, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_dim)
        x = self.norm1(x + self.out_proj(out))
        return self.norm2(x + self.ffn(x))


class CausalOseqModel(nn.Module):
    """AR twin: hidden state at token P_i predicts b_i."""

    def __init__(self, hidden_dim=128, num_layers=4, head_count=8,
                 max_seq_len=4096, rope_theta=10000.0):
        super().__init__()
        self.type_embedding = nn.Embedding(3, hidden_dim)
        self.g_proj = nn.Linear(1, hidden_dim)
        self.P_embedding = nn.Embedding(3, hidden_dim)
        self.b_embedding = nn.Embedding(3, hidden_dim)  # 0,1 + (unused) mask id 2
        head_dim = hidden_dim // head_count
        self.rope = _build_llama_rotary_embedding(head_dim, hidden_dim, head_count,
                                                  max_seq_len, rope_theta=rope_theta)
        self.layers = nn.ModuleList(
            [CausalTransformerBlock(hidden_dim, head_count, self.rope)
             for _ in range(num_layers)])
        self.output_layer = nn.Linear(hidden_dim, 2)

    def embed(self, x):
        # x: [B, L] float rows [g, P1, b1, ...]; unknown b may be any value —
        # they are never attended from positions <= their own P (causal mask),
        # so we can safely embed them as 0.
        B, L = x.shape
        g = x[:, :1]
        P = (x[:, 1:L:2] - 2).long().clamp(0, 2)
        b = x[:, 2:L:2].long().clamp(0, 1)
        h = torch.zeros(B, L, self.g_proj.out_features, device=x.device)
        h[:, 0] = self.g_proj(g)
        h[:, 1:L:2] = self.P_embedding(P)
        h[:, 2:L:2] = self.b_embedding(b)
        tid = torch.zeros(L, dtype=torch.long, device=x.device)
        tid[1:L:2] = 1
        tid[2:L:2] = 2
        return h + self.type_embedding(tid).unsqueeze(0)

    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        # logits for b_i from hidden at P_i (positions 1,3,5,...)
        return self.output_layer(h[:, 1::2])


def cmd_train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    datasets = []
    for path, n in zip(args.data, args.qubits):
        rows = json.load(open(path))
        datasets.append((n, torch.tensor(rows, dtype=torch.float32)))
        print(f"loaded {path}: {len(rows)} rows (N={n})")
    model = CausalOseqModel(args.hidden_dim, args.layer_num, args.head_num).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        total, nb = 0.0, 0
        batches = []
        for n, data in datasets:
            perm = torch.randperm(len(data))
            bs = max(1, int(args.batch_size * (1 + 2 * max(args.qubits)) / (1 + 2 * n)))
            batches += [data[perm[i:i + bs]] for i in range(0, len(data), bs)]
        random.shuffle(batches)
        for batch in batches:
            batch = batch.to(device)
            logits = model(batch)                      # [B, N, 2]
            target = batch[:, 2::2].round().long()     # [B, N]
            loss = loss_fn(logits.reshape(-1, 2), target.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); nb += 1
        print(f"epoch {epoch + 1}/{args.epochs} loss {total / nb:.6f}", flush=True)
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.save)
    print(f"saved {args.save}")


@torch.no_grad()
def cmd_eval(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CausalOseqModel(args.hidden_dim, args.layer_num, args.head_num).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()
    n = args.lx * args.ly
    L = 1 + 2 * n
    exact = np.load(args.exact_npz)
    gs = np.asarray(exact['gs'], dtype=float)
    h1, v1, h2 = torus_pairs(args.lx, args.ly)
    est = {'h1': [], 'v1': [], 'h2': []}
    for g in gs:
        P_all, b_all = [], []
        remaining = args.sample_size
        while remaining > 0:
            B = min(args.gen_batch_size, remaining)
            x = torch.zeros(B, L, device=device)
            x[:, 0] = float(g)
            P = torch.randint(2, 5, (B, n), device=device).float()
            x[:, 1::2] = P
            for i in range(n):                 # left-to-right AR sampling
                logits = model(x)[:, i]        # b_i from prefix
                p1 = torch.softmax(logits, dim=-1)[:, 1]
                x[:, 2 + 2 * i] = torch.bernoulli(p1)
            P_all.append(P.cpu().numpy()); b_all.append(x[:, 2::2].cpu().numpy())
            remaining -= B
        P = np.concatenate(P_all).astype(np.int8)
        b = np.concatenate(b_all).astype(np.int8)
        for key, pairs in (('h1', h1), ('v1', v1), ('h2', h2)):
            est[key].append(median_of_means(shadow_zz(P, b, pairs), 10))
        print(f"  g={g:.3f} h1={est['h1'][-1]:+.4f} (exact {float(exact['corr_h1'][len(est['h1'])-1]):+.4f})",
              flush=True)
    mse = {k: float(np.mean((np.array(est[k]) - np.asarray(exact[f'corr_{k}'])) ** 2))
           for k in ('h1', 'v1', 'h2')}
    print("MSE", {k: f"{v:.3e}" for k, v in mse.items()})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, gs=gs,
             gen_h1=np.array(est['h1']), gen_v1=np.array(est['v1']),
             gen_h2=np.array(est['h2']),
             exact_h1=exact['corr_h1'], exact_v1=exact['corr_v1'],
             exact_h2=exact['corr_h2'],
             mse_h1=mse['h1'], mse_v1=mse['v1'], mse_h2=mse['h2'])
    print(f"saved {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    tr = sub.add_parser('train')
    tr.add_argument('--data', nargs='+', required=True)
    tr.add_argument('--qubits', type=int, nargs='+', required=True)
    tr.add_argument('--hidden_dim', type=int, default=128)
    tr.add_argument('--layer_num', type=int, default=4)
    tr.add_argument('--head_num', type=int, default=8)
    tr.add_argument('--epochs', type=int, default=100)
    tr.add_argument('--batch_size', type=int, default=256)
    tr.add_argument('--lr', type=float, default=1e-4)
    tr.add_argument('--save', required=True)
    ev = sub.add_parser('eval')
    ev.add_argument('--model', required=True)
    ev.add_argument('--lx', type=int, default=4)
    ev.add_argument('--ly', type=int, required=True)
    ev.add_argument('--exact_npz', required=True)
    ev.add_argument('--out', required=True)
    ev.add_argument('--sample_size', type=int, default=20000)
    ev.add_argument('--gen_batch_size', type=int, default=2048)
    ev.add_argument('--hidden_dim', type=int, default=128)
    ev.add_argument('--layer_num', type=int, default=4)
    ev.add_argument('--head_num', type=int, default=8)
    args = ap.parse_args()
    if args.cmd == 'train':
        cmd_train(args)
    else:
        cmd_eval(args)


if __name__ == '__main__':
    main()
