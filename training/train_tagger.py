"""
Trainiert den Wort-Tagger (tagger.py) auf den Fake-Daten.   python train_tagger.py --epochs 8
Ergebnis: checkpoints/tagger.pt
"""
import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from tagger import TAG_ID, Tagger, encode, gold_to_tags

HERE = Path(__file__).resolve().parent


def load(name, data_dir):
    rows = [json.loads(l) for l in open(HERE / data_dir / f"{name}.jsonl", encoding="utf-8")]
    out = []
    for r in rows:
        toks, tags = gold_to_tags(r["text"], r["gold"])
        out.append(([t[0] for t in toks], [TAG_ID[t] for t in tags]))
    return out


def batchify(data, vocab, idx):
    enc = [encode(data[i][0], vocab) for i in idx]
    T = max(len(e[0]) for e in enc)
    B = len(idx)
    wid = torch.zeros(B, T, dtype=torch.long); ch = torch.zeros(B, T, 20, dtype=torch.long)
    feat = torch.zeros(B, T, 4); lab = torch.full((B, T), -100)
    for r, (e, i) in enumerate(zip(enc, idx)):
        n = len(e[0])
        wid[r, :n], ch[r, :n], feat[r, :n] = e
        lab[r, :n] = torch.tensor(data[i][1])
    return wid, ch, feat, lab


@torch.no_grad()
def evaluate_loss(model, data, vocab):
    model.eval()
    wid, ch, feat, lab = batchify(data, vocab, list(range(min(200, len(data)))))
    lg = model(wid, ch, feat)
    acc_mask = lab != -100
    acc = (lg.argmax(-1) == lab)[acc_mask].float().mean().item()
    return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), lab.reshape(-1), ignore_index=-100).item(), acc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/fake")
    ap.add_argument("--extra-train", default="", help="weiteres Datenverzeichnis, dessen train.jsonl beigemischt wird (z.B. data/fake)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()
    torch.manual_seed(0); random.seed(0)

    train, val = load("train", args.data), load("val", args.data)
    if args.extra_train:
        train += load("train", args.extra_train)
    cnt = Counter(w.lower() for words, _ in train for w in words)
    vocab = {w: i + 2 for i, (w, c) in enumerate(cnt.most_common()) if c >= 5}
    model = Tagger(len(vocab) + 2)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} Mio. Parameter | {len(train)} Train, Wortschatz {len(vocab)}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train))); random.shuffle(order)
        tot = 0.0
        for s in range(0, len(order), args.batch):
            wid, ch, feat, lab = batchify(train, vocab, order[s:s + args.batch])
            lg = model(wid, ch, feat)
            loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), lab.reshape(-1), ignore_index=-100)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item()
        vl, va = evaluate_loss(model, val, vocab)
        print(f"Epoche {ep} | train-loss {tot / (len(order) / args.batch):.4f} | val-loss {vl:.4f} | val-Wortgenauigkeit {va:.3f} | {time.time() - t0:.0f}s", flush=True)
        (HERE / "checkpoints").mkdir(exist_ok=True)
        torch.save({"model": model.state_dict(), "vocab": vocab}, HERE / "checkpoints" / "tagger.pt")
