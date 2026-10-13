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

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=128)

train_tok = train_ds.map(tokenize, batched=True)
val_tok = val_ds.map(tokenize, batched=True)

id2label = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}
label2id = {v: k for k, v in id2label.items()}

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased",
    num_labels=4,
    id2label=id2label,
    label2id=label2id,
)

# print(sum(p.numel() for p in model.parameters()))
# 66956548

collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": float((preds == labels).mean())}

# smoke_args = TrainingArguments(
#     output_dir="runs/smoke",
#     learning_rate=2e-5,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=32,
#     num_train_epochs=1,
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     logging_steps=10,
#     report_to="none",
#     use_cpu=(device == "cpu"),
# )
# smoke = Trainer(
#     model=model,
#     args=smoke_args,
#     train_dataset=train_tok.select(range(512)),
#     eval_dataset=val_tok.select(range(128)),
#     processing_class=tokenizer,
#     data_collator=collator,
#     compute_metrics=compute_metrics,
# )
# smoke.train()
# smoke.save_model("runs/smoke/final")
# tokenizer.save_pretrained("runs/smoke/final")

args = TrainingArguments(
    output_dir="runs/day21",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
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
    data_collator=collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("runs/day21/final")
tokenizer.save_pretrained("runs/day21/final")