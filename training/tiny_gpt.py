"""
Ein kleines GPT (Decoder-Transformer) von Grund auf in PyTorch - bewusst kurz gehalten (~100 Zeilen),
damit man jede Zeile nachvollziehen kann. Es arbeitet auf Byte-Ebene (kein Tokenizer nötig).

Aufgabe: Seitentext  ->  JSON nach GRAPH_SCHEMA.
Format einer Trainingssequenz:   <BOS> text-bytes <SEP> json-bytes <EOS>
Der Loss wird nur auf dem JSON-Teil berechnet (das Modell soll lernen zu extrahieren, nicht Seiten zu erfinden).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

BOS, SEP, EOS, PAD = 256, 257, 258, 259
VOCAB = 260


def encode_pair(text: str, target: str | None, max_len: int):
    src = [BOS, *text.encode("utf-8"), SEP]
    if target is None:
        return src, None
    tgt = [*target.encode("utf-8"), EOS]
    ids = (src + tgt)[:max_len]
    labels = ([-100] * len(src) + tgt)[:max_len]
    return ids, labels


def decode(ids) -> str:
    return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")


class Block(nn.Module):
    def __init__(self, d, heads, drop):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.heads, C // self.heads).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # Self-Attention, nur nach links schauen
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, T, C)))
        return x + self.drop(self.mlp(self.ln2(x)))


class TinyGPT(nn.Module):
    def __init__(self, d=128, layers=3, heads=4, ctx=1024, drop=0.1):
        super().__init__()
        self.ctx = ctx
        self.tok = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, heads, drop) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB, bias=False)
        self.head.weight = self.tok.weight  # Gewichte teilen spart Parameter
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, ids, labels=None):
        x = self.tok(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), labels[:, 1:].reshape(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt_ids, max_new=500):
        """Greedy-Decoding (immer das wahrscheinlichste Byte) - deterministisch, gut für Extraktion."""
        ids = torch.tensor([prompt_ids])
        for _ in range(max_new):
            if ids.shape[1] >= self.ctx:
                break
            nxt = self(ids)[0][0, -1].argmax().item()
            if nxt == EOS:
                break
            ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
        return ids[0, len(prompt_ids):].tolist()
