"""
Fine-Tuning eines vortrainierten kleinen Sprachmodells mit LoRA  -  der realistische Weg für deine Arbeit.

Statt ein Modell bei Null zu starten (wie tiny_gpt.py), nimmt man ein Modell, das schon Deutsch und Englisch
kann (z.B. Qwen2.5-0.5B/1.5B-Instruct), und trainiert nur kleine Zusatzmatrizen (LoRA, <1 % der Gewichte).
Das braucht wenige hundert Beispiele und eine GPU mit ~8-16 GB (Colab, Kaggle, TH-Köln-Cluster).

Installation (dort, wo trainiert wird):
  pip install torch transformers peft accelerate

Aufruf:
  python finetune_lora.py --data data/fake        # zuerst mit Fake-Daten Pipeline testen
  python finetune_lora.py --data data/real --model Qwen/Qwen2.5-1.5B-Instruct --epochs 3

Danach: das Ergebnis mit evaluate.py bewerten; zum Einbinden in src/extract.py nur _run_model austauschen.
HINWEIS: Dieses Skript konnte ich in meiner Umgebung nicht ausführen (kein Zugriff auf Modell-Downloads, keine GPU).
Es ist bewusst kurz und ohne Zusatz-Bibliotheken wie trl gehalten; bei Fehlermeldungen bitte melden.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
SYSTEM_PROMPT = ("Extrahiere aus der Seite einer Person des THK-AI Forschungsclusters Personen, Projekte, "
                 "Publikationen und Kooperationen als JSON.")


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def build(tok, row, max_len):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT + "\nAntworte nur mit JSON."},
            {"role": "user", "content": row["text"]}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    answer = json.dumps(row["gold"], ensure_ascii=False, separators=(",", ":")) + tok.eos_token
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    a = tok(answer, add_special_tokens=False)["input_ids"]
    ids = (p + a)[:max_len]
    labels = ([-100] * len(p) + a)[:max_len]  # Loss nur auf der Antwort
    return torch.tensor(ids), torch.tensor(labels)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/fake")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0, help="nur die ersten N Trainingsbeispiele (zum schnellen Testen)")
    ap.add_argument("--out", default="checkpoints/lora")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.print_trainable_parameters()

    rows = load(HERE / args.data / "train.jsonl")
    if args.limit:
        rows = rows[: args.limit]
    data = [build(tok, r, args.max_len) for r in rows]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    accum, step = 8, 0  # 8 Beispiele je Update (Batchgröße 1 spart Speicher)
    model.train()
    for ep in range(args.epochs):
        random.shuffle(data)
        for i, (ids, lab) in enumerate(data, 1):
            loss = model(input_ids=ids[None].to(dev), labels=lab[None].to(dev)).loss / accum
            loss.backward()
            if i % accum == 0:
                opt.step(); opt.zero_grad(); step += 1
                if step % 10 == 0:
                    print(f"Epoche {ep + 1} | Update {step} | loss {loss.item() * accum:.3f}", flush=True)
    model.save_pretrained(HERE / args.out)
    tok.save_pretrained(HERE / args.out)
    print("Gespeichert:", HERE / args.out)
