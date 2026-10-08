# src/models/distilbert_model.py — DistilBERT fine-tuning for fake news detection

import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OUTPUTS_DIR

MODEL_NAME   = "distilbert-base-uncased"
MAX_LEN      = 256      # truncate to 256 tokens (saves RAM vs 512)
BATCH_SIZE   = 8        # small batch for 8GB RAM
EPOCHS       = 2        # 2 epochs is enough for fine-tuning
LR           = 2e-5
WARMUP_RATIO = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ───────────────────────────────────────────────────────────────────
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ── Training ──────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs   = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            preds   = outputs.logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1":       f1_score(all_labels, all_preds),
        "roc_auc":  roc_auc_score(all_labels, all_probs),
        "y_pred":   np.array(all_preds),
        "y_proba":  np.array(all_probs),
        "y_true":   np.array(all_labels),
    }


# ── Main fine-tuning function ─────────────────────────────────────────────────
def run_distilbert(X_train, y_train, X_val, y_val, X_test, y_test):
    print(f"[distilbert] Device: {DEVICE}")
    print(f"[distilbert] Loading tokenizer and model: {MODEL_NAME}")

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    model     = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(DEVICE)

    print("[distilbert] Tokenizing datasets (this may take a few minutes)...")
    train_dataset = NewsDataset(X_train, y_train, tokenizer)
    val_dataset   = NewsDataset(X_val,   y_val,   tokenizer)
    test_dataset  = NewsDataset(X_test,  y_test,  tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )

    print(f"\n[distilbert] Fine-tuning for {EPOCHS} epochs...")
    print(f"  Batches per epoch : {len(train_loader)}")
    print(f"  Total steps       : {total_steps}")
    print(f"  NOTE: CPU training is slow (~1-2 hrs). Consider using Google Colab with GPU.\n")

    best_f1   = 0
    start     = time.time()

    for epoch in range(1, EPOCHS + 1):
        print(f"-- Epoch {epoch}/{EPOCHS} --")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler)
        val_metrics = evaluate(model, val_loader)
        elapsed = (time.time() - start) / 60

        print(f"  Train Loss : {train_loss:.4f}")
        print(f"  Val Acc    : {val_metrics['accuracy']:.4f}")
        print(f"  Val F1     : {val_metrics['f1']:.4f}")
        print(f"  Val AUC    : {val_metrics['roc_auc']:.4f}")
        print(f"  Elapsed    : {elapsed:.1f} min\n")

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            model.save_pretrained(OUTPUTS_DIR / "distilbert_model")
            tokenizer.save_pretrained(OUTPUTS_DIR / "distilbert_model")
            print(f"  [saved best model -> outputs/distilbert_model/]")

    print("\n[distilbert] Evaluating on test set...")
    test_metrics = evaluate(model, test_loader)
    print(f"  Test Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"  Test F1       : {test_metrics['f1']:.4f}")
    print(f"  Test ROC-AUC  : {test_metrics['roc_auc']:.4f}")

    return test_metrics, tokenizer
