import os
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
import numpy as np
import torch

# Where will we be training
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

raw = load_dataset('fancyzhx/ag_news')

# Basic checks
assert raw["train"].column_names == ["text", "label"]
assert set(raw["train"]["label"]) == {0, 1, 2, 3}

cut = raw["train"].train_test_split(test_size=0.1, seed=42, shuffle=True)

train_ds = cut["train"]
assert len(train_ds) == 108000

val_ds = cut["test"]
assert len(val_ds) == 12000

tokenizer = AutoTokenizer.from_pretrained("./runs/day21/final")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=128)

train_tok = train_ds.map(tokenize, batched=True)
val_tok = val_ds.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained("./runs/day21/final")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": float((preds == labels).mean())}

args = TrainingArguments(
    output_dir="runs/day22",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    logging_steps=50,
    report_to="none",
    use_cpu=(device == "cpu"),
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,  # same function as yesterday
)

trainer.train()
trainer.save_model("runs/day22/final")
tokenizer.save_pretrained("runs/day22/final")