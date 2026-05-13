#!/usr/bin/env python3
"""
AFRCC SOAPIE SLM - Clean Training Script (GitHub + Local / Databricks Ready)
Fine-tunes DistilBERT (66M params) for:
  - Binary classification: "good" vs "incomplete" case notes
  - Regression: quality_score (0-100)
Following DHA "Writing Case Notes" (SOAPIE) + AFRCC Training KPIs.

This is the production version of the Colab notebook.
No Jupyter magic, no `!pip`, fully executable with `python train_soapie_slm.py`.

USAGE (recommended):
  python train_soapie_slm.py --data_dir ./soapie_slm_small_afrcc --output_dir ./soapie_slm_output --epochs 8

For GitHub + Streamlit Cloud deployment flow:
  1. Push this script + train.jsonl / validation.jsonl / test.jsonl to your repo.
  2. Run locally or on a GPU machine / Colab to produce the model.
  3. Upload the saved `soapie_slm_small_distilbert/` folder to Hugging Face Hub or GitHub Releases.
  4. Update your Streamlit app (app.py) to download the model from HF on first run (`huggingface_hub.snapshot_download`).
  5. Deploy the clean app.py to Streamlit Cloud by pasting its raw GitHub URL.

Requirements (create requirements.txt):
  transformers==4.46.0
  datasets==3.1.0
  accelerate==1.1.1
  evaluate==0.4.3
  scikit-learn==1.5.2
  mlflow==2.17.0
  torch
  matplotlib
  seaborn

Author: AFRCC R&D | UNCLASSIFIED // Authorized training & development use only
"""

import argparse
import json
import os
import time
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    DistilBertModel,
    DistilBertPreTrainedModel,
)
from torch import nn
from datasets import Dataset
import evaluate
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_absolute_error, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import mlflow

# ====================== CONFIG ======================
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 512
NUM_LABELS_CLASS = 2
ID2LABEL = {0: "incomplete", 1: "good"}
LABEL2ID = {"incomplete": 0, "good": 1}


class MultiTaskDistilBERT(DistilBertPreTrainedModel):
    """Multi-task head: classification + quality regression (exact same as Colab version)."""
    def __init__(self, config):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.classifier = nn.Linear(config.dim, NUM_LABELS_CLASS)
        self.regressor = nn.Linear(config.dim, 1)
        self.dropout = nn.Dropout(0.1)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, quality_scores=None, **kwargs):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        class_logits = self.classifier(pooled)
        quality_pred = self.regressor(pooled).squeeze(-1)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            class_loss = loss_fct(class_logits, labels)
            if quality_scores is not None:
                reg_loss = nn.MSELoss()(quality_pred, quality_scores)
                loss = class_loss + 0.5 * reg_loss
            else:
                loss = class_loss
        return {"loss": loss, "logits": class_logits, "quality_pred": quality_pred}


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line.strip())
            data.append(ex)
    return data


def preprocess(examples, tokenizer):
    tokenized = tokenizer(
        examples["raw_case_note"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False
    )
    tokenized["labels"] = [LABEL2ID[ex["label"]] for ex in examples]
    tokenized["quality_scores"] = [float(ex["quality_score"]) / 100.0 for ex in examples]
    return tokenized


def compute_metrics(eval_pred, val_raw):
    logits, labels = eval_pred.predictions[0], eval_pred.label_ids
    quality_preds = eval_pred.predictions[1]

    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    prec = precision_score(labels, preds, average="macro", zero_division=0)
    rec = recall_score(labels, preds, average="macro", zero_division=0)

    # Regression MAE (scaled back to 0-100)
    true_quality = np.array([ex["quality_score"] for ex in val_raw]) / 100.0
    mae = mean_absolute_error(true_quality, quality_preds) * 100

    cm = confusion_matrix(labels, preds)
    incomplete_recall = recall_score(labels, preds, pos_label=0, zero_division=0)

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "quality_mae": round(mae, 2),
        "incomplete_recall": round(incomplete_recall, 4)
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune AFRCC SOAPIE SLM")
    parser.add_argument("--data_dir", type=str, default="./soapie_slm_small_afrcc",
                        help="Folder containing train.jsonl, validation.jsonl, test.jsonl")
    parser.add_argument("--output_dir", type=str, default="./soapie_slm_output",
                        help="Where to save the fine-tuned model and artifacts")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--mlflow_experiment", type=str, default="AFRCC_SOAPIE_SLM_Small_v1")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Using model: {MODEL_NAME}")
    print(f"Data dir: {args.data_dir}")
    print(f"Output dir: {args.output_dir}")

    # Load data
    train_raw = load_jsonl(os.path.join(args.data_dir, "train.jsonl"))
    val_raw = load_jsonl(os.path.join(args.data_dir, "validation.jsonl"))
    test_raw = load_jsonl(os.path.join(args.data_dir, "test.jsonl"))

    print(f"Train: {len(train_raw)} | Val: {len(val_raw)} | Test: {len(test_raw)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = Dataset.from_list(train_raw).map(
        lambda x: preprocess(x, tokenizer), batched=True,
        remove_columns=["raw_case_note", "id", "metadata", "issues", "condition", "interaction_type", "date"]
    )
    val_ds = Dataset.from_list(val_raw).map(
        lambda x: preprocess(x, tokenizer), batched=True,
        remove_columns=["raw_case_note", "id", "metadata", "issues", "condition", "interaction_type", "date"]
    )
    test_ds = Dataset.from_list(test_raw).map(
        lambda x: preprocess(x, tokenizer), batched=True,
        remove_columns=["raw_case_note", "id", "metadata", "issues", "condition", "interaction_type", "date"]
    )

    model = MultiTaskDistilBERT.from_pretrained(MODEL_NAME)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        logging_steps=5,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, val_raw),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print("🚀 Starting fine-tuning...")
    start = time.time()
    trainer.train()
    train_time = time.time() - start
    print(f"✅ Training complete in {train_time:.1f}s")

    # Final evaluation
    print("\n📊 Final Test Evaluation...")
    test_results = trainer.evaluate(test_ds)
    print(test_results)

    # Efficiency metrics
    model.eval()
    sample_note = test_raw[0]["raw_case_note"][:400]
    inputs = tokenizer(sample_note, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(model.device)

    for _ in range(3):
        with torch.no_grad():
            _ = model(**inputs)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    with torch.no_grad():
        out = model(**inputs)
    latency_ms = (time.time() - start) * 1000

    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0

    print(f"\n⚡ SLM Efficiency Metrics:")
    print(f"  Inference latency: {latency_ms:.1f} ms per note")
    print(f"  Model size: {model_size_mb:.1f} MB")
    print(f"  Peak VRAM: {peak_vram_gb:.2f} GB")

    # Confusion matrix + AFRCC compliance
    preds = trainer.predict(test_ds)
    class_preds = np.argmax(preds.predictions[0], axis=-1)
    true_labels = [LABEL2ID[ex["label"]] for ex in test_raw]

    cm = confusion_matrix(true_labels, class_preds, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["incomplete", "good"], yticklabels=["incomplete", "good"])
    plt.title("Test Confusion Matrix - Incomplete Detection (Safety Critical)")
    plt.tight_layout()
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")

    low_quality = sum(1 for ex in test_raw if ex["quality_score"] < 60 and ex["label"] == "incomplete")
    caught = sum(1 for i, ex in enumerate(test_raw) if ex["quality_score"] < 60 and class_preds[i] == 0)
    compliance_rate = (caught / max(low_quality, 1)) * 100
    print(f"AFRCC Compliance (catch low-quality notes): {compliance_rate:.1f}% (target >90%)")

    # Save model
    model_path = os.path.join(args.output_dir, "soapie_slm_small_distilbert")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"\n✅ Model saved to {model_path}")

    # MLflow logging (optional - set DATABRICKS_HOST + DATABRICKS_TOKEN env vars for remote)
    try:
        mlflow.set_experiment(args.mlflow_experiment)
        with mlflow.start_run(run_name=f"distilbert-soapie-{datetime.now().strftime('%Y%m%d-%H%M')}"):
            mlflow.log_metrics({k: float(v) for k, v in test_results.items() if isinstance(v, (int, float))})
            mlflow.log_metric("inference_latency_ms", round(latency_ms, 1))
            mlflow.log_metric("model_size_mb", round(model_size_mb, 1))
            mlflow.log_metric("afrcc_compliance_rate", round(compliance_rate, 1))
            mlflow.log_metric("train_time_seconds", round(train_time, 1))
            mlflow.pytorch.log_model(model, "model", registered_model_name="afrcc_soapie_slm_small")
            mlflow.log_artifact(cm_path)
        print("✅ MLflow run logged (local or Databricks if env vars set)")
    except Exception as e:
        print(f"MLflow logging skipped or failed: {e}")

    # Quick inference demo
    def score_new_case_note(note_text: str):
        inputs = tokenizer(note_text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(model.device)
        with torch.no_grad():
            out = model(**inputs)
        prob = torch.softmax(out["logits"], dim=-1)[0]
        pred_label = ID2LABEL[torch.argmax(prob).item()]
        quality = float(out["quality_pred"].item()) * 100
        confidence = float(torch.max(prob).item())
        flag_review = quality < 65 or pred_label == "incomplete"
        return {
            "predicted_label": pred_label,
            "quality_score": round(quality, 1),
            "confidence": round(confidence, 3),
            "recommend_human_review": flag_review,
            "afrcc_note": "Human RCC must always author final note. Model assists quality assurance only."
        }

    demo_note = """Case Note - PHONE FOLLOW-UP | 2025-10-18 | SM-772134 | SSgt Taylor Kim | JB Andrews
Condition: PTSD with depression. Caregiver (spouse) present.
SUBJECTIVE: SM reports "therapy is helping a bit, but nightmares still 4 nights/week."
OBJECTIVE: Voice flat, mentioned missing 2 PT sessions. Caregiver reports increased isolation.
ASSESSMENT: Possible medication side effect or emerging depressive episode.
PLAN: 1. Urgent BH follow-up within 72hrs. 2. Daily text check-ins for 7 days.
IMPLEMENTATION: Prior plan attended 3/3 PT. Med change implemented 10/14.
EVALUATION: Nightmares increased; engagement dropped. Immediate escalation needed."""

    print("\n🧪 Demo scoring on new note:")
    print(score_new_case_note(demo_note))

    print("\n🎉 Training complete! Next steps:")
    print("  1. Copy the 'soapie_slm_small_distilbert/' folder to your Streamlit app directory or upload to Hugging Face Hub.")
    print("  2. Update app.py MODEL_PATH to point to it (or use huggingface_hub to download in Cloud).")
    print("  3. For Unity/Quest 3+: torch.onnx.export(model, 'soapie_slm.onnx', ...)")
    print("  4. Deploy clean app.py to Streamlit Cloud using its raw GitHub URL.")


if __name__ == "__main__":
    main()