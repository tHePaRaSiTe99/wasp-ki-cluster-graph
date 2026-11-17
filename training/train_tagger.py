"""
Trainiert den Wort-Tagger (tagger.py) auf den Fake-Daten.   python train_tagger.py --epochs 8
Ergebnis: checkpoints/tagger.pt
"""
import argparse
import os
import sys
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from tagger import TAG_ID, Tagger, encode, gold_to_tags

HERE = Path(__file__).resolve().parent


def chunks(words, tags, size):
    """Lange (echte) Seiten in Stücke von ca. `size` Wörtern teilen - nur an Zeilenanfängen mit "*" oder "#",
    damit keine Literaturangabe zerrissen wird. Die Etiketten stammen von der ganzen Seite und bleiben gültig."""
    if not size or len(words) <= size:
        return [(words, tags)]
    out, start = [], 0
    cuts = [i + 1 for i, w in enumerate(words[:-1]) if w == "\n" and words[i + 1] in ("*", "#")]
    for c in cuts + [len(words)]:
        if c - start >= size or c == len(words):
            if c > start:
                out.append((words[start:c], tags[start:c]))
            start = c
    return out


def load(name, data_dir, chunk=0, repeat=1):
    rows = [json.loads(l) for l in open(HERE / data_dir / f"{name}.jsonl", encoding="utf-8")]
    out = []
    for r in rows:
        toks, tags = gold_to_tags(r["text"], r["gold"])
        out += chunks([t[0] for t in toks], [TAG_ID[t] for t in tags], chunk)
    return out * repeat


def encode_all(data, vocab):
    """Jedes Beispiel einmal vorab kodieren (statt in jeder Epoche neu). Wiederholte Beispiele (--repeat)
    sind dieselben Listen-Objekte und werden nur einmal kodiert."""
    cache = {}
    out = []
    for words, labels in data:
        key = id(words)
        if key not in cache:
            cache[key] = (*encode(words, vocab), torch.tensor(labels))
        out.append(cache[key])
    return out


def batchify(enc, idx, device="cpu"):
    T = max(len(enc[i][0]) for i in idx)
    B = len(idx)
    wid = torch.zeros(B, T, dtype=torch.long); ch = torch.zeros(B, T, 20, dtype=torch.long)
    feat = torch.zeros(B, T, 4); lab = torch.full((B, T), -100)
    for r, i in enumerate(idx):
        w, c, f, l = enc[i]
        n = len(w)
        wid[r, :n], ch[r, :n], feat[r, :n], lab[r, :n] = w, c, f, l
    return tuple(x.to(device, non_blocking=True) for x in (wid, ch, feat, lab))


@torch.no_grad()
def evaluate_loss(model, enc, device="cpu", batch=32):
    """Verlust und Wortgenauigkeit auf (bis zu) 200 Validierungsbeispielen, in Batches (GPU-Speicher)."""
    model.eval()
    idx = list(range(min(200, len(enc))))
    loss_sum, n_tok, n_ok = 0.0, 0, 0
    for s in range(0, len(idx), batch):
        wid, ch, feat, lab = batchify(enc, idx[s:s + batch], device)
        lg = model(wid, ch, feat)
        mask = lab != -100
        loss_sum += F.cross_entropy(lg.reshape(-1, lg.shape[-1]), lab.reshape(-1), ignore_index=-100, reduction="sum").item()
        n_tok += mask.sum().item()
        n_ok += ((lg.argmax(-1) == lab) & mask).sum().item()
    return loss_sum / max(n_tok, 1), n_ok / max(n_tok, 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/fake")
    ap.add_argument("--extra-train", default="", help="weiteres Datenverzeichnis, dessen train.jsonl beigemischt wird (z.B. data/fake)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--chunk", type=int, default=0, help="lange Seiten von --data in Stücke dieser Wortzahl teilen (z.B. 400)")
    ap.add_argument("--repeat", type=int, default=1, help="Trainingsseiten von --data so oft wiederholen (echte Seiten höher gewichten)")
    ap.add_argument("--device", default="auto", help="auto (GPU, falls vorhanden), cuda oder cpu")
    ap.add_argument("--out", default="checkpoints/tagger.pt", help="Zieldatei des Modells")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    torch.manual_seed(args.seed); random.seed(args.seed)
    out_path = HERE / args.out

    train, val = load("train", args.data, args.chunk, args.repeat), load("val", args.data, args.chunk)
    if args.extra_train:
        train += load("train", args.extra_train)
    cnt = Counter(w.lower() for words, _ in train for w in words)
    vocab = {w: i + 2 for i, (w, c) in enumerate(cnt.most_common()) if c >= 5}
    model = Tagger(len(vocab) + 2).to(device)
    train_enc, val_enc = encode_all(train, vocab), encode_all(val, vocab)
    dev_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} Mio. Parameter | {len(train)} Train, Wortschatz {len(vocab)} | Gerät: {dev_name}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    t0 = time.time()
    best = float("inf")
    for ep in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train))); random.shuffle(order)
        tot = 0.0
        for s in range(0, len(order), args.batch):
            wid, ch, feat, lab = batchify(train_enc, order[s:s + args.batch], device)
            lg = model(wid, ch, feat)
            loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), lab.reshape(-1), ignore_index=-100)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item()
        vl, va = evaluate_loss(model, val_enc, device, args.batch)
        improved = vl < best
        print(f"Epoche {ep} | train-loss {tot / (len(order) / args.batch):.4f} | val-loss {vl:.4f} | val-Wortgenauigkeit {va:.3f} | "
              f"{time.time() - t0:.0f}s" + ("  <- bisher bestes, gespeichert" if improved else ""), flush=True)
        if improved:  # nur das Modell mit dem kleinsten Validierungsverlust behalten (gegen Überanpassung)
            best = vl
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # immer als CPU-Tensoren speichern: Extraktion/Auswertung laden ohne GPU
            torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()}, "vocab": vocab, "epoch": ep, "val_loss": vl}, out_path)
    print(f"Bestes Modell: val-loss {best:.4f} -> {out_path}", flush=True)
    if device == "cuda":
        # Unter Windows stürzt der Prozess beim Aufräumen von CUDA gelegentlich ab (0xC0000409), obwohl alles
        # gespeichert ist. Das Modell liegt sicher auf der Platte -> Prozess direkt und sauber beenden.
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(0)
