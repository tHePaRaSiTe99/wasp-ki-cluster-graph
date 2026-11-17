"""
Trainiert das Mini-GPT (tiny_gpt.py) auf den Fake-Daten.

  python train_scratch.py --steps 3000
Ergebnis: checkpoints/tiny.pt  (+ Loss-Ausgabe alle 100 Schritte, Validierung alle 500)
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch

from fake_data import compact
from tiny_gpt import PAD, TinyGPT, encode_pair

HERE = Path(__file__).resolve().parent


def load(name, ctx):
    rows = [json.loads(l) for l in open(HERE / "data" / "fake" / f"{name}.jsonl", encoding="utf-8")]
    return [encode_pair(r["text"], compact(r["gold"]), ctx) for r in rows]


def batch(examples, idx):
    L = max(len(examples[i][0]) for i in idx)
    ids = torch.full((len(idx), L), PAD)
    lab = torch.full((len(idx), L), -100)
    for r, i in enumerate(idx):
        a, b = examples[i]
        ids[r, : len(a)] = torch.tensor(a)
        lab[r, : len(b)] = torch.tensor(b)
    return ids, lab


@torch.no_grad()
def val_loss(model, data, bs=32):
    model.eval()
    tot, n = 0.0, 0
    for s in range(0, len(data), bs):
        ids, lab = batch(data, list(range(s, min(s + bs, len(data)))))
        tot += model(ids, lab)[1].item(); n += 1
    model.train()
    return tot / n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=0, help="0 = automatisch")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(0); random.seed(0)

    ctx = 1024
    train, val = load("train", ctx), load("val", ctx)
    model = TinyGPT(d=args.d, layers=args.layers, ctx=ctx)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} Mio. Parameter | {len(train)} Train / {len(val)} Val")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)

    (HERE / "checkpoints").mkdir(exist_ok=True)
    t0, run = time.time(), 0.0
    for step in range(1, args.steps + 1):
        ids, lab = batch(train, random.sample(range(len(train)), args.batch))
        _, loss = model(ids, lab)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        run += loss.item()
        if step % 100 == 0:
            print(f"Schritt {step:5d} | train-loss {run / 100:.3f} | {time.time() - t0:5.0f}s", flush=True)
            run = 0.0
        if step % 500 == 0 or step == args.steps:
            print(f"   >> val-loss {val_loss(model, val):.3f}", flush=True)
            torch.save({"model": model.state_dict(), "d": args.d, "layers": args.layers, "ctx": ctx}, HERE / "checkpoints" / "tiny.pt")
