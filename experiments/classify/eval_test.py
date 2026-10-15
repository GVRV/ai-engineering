from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Where will we be evaluating?
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

raw = load_dataset("fancyzhx/ag_news")
test_ds = raw["test"]
assert len(test_ds) == 7600

tokenizer = AutoTokenizer.from_pretrained("./runs/day22/final")
model = AutoModelForSequenceClassification.from_pretrained(
    "./runs/day22/final"
)
model.eval()

LABELS = model.config.id2label

model.to(device)

def predict_split(split, batch_size=32):
    preds, labels, texts = [], [], []

    for start in range(0, len(split), batch_size):
        batch = split[start:start + batch_size]

        # this returns a dictionary with the tensor for the full batch
        # of input values. For a batch of size 1 input, it will return:
        # {
        #     'input_ids': tensor([
        #         [ 101, 10069,  2005,  1056,  1050, 11550,  2044,  7566,  9209,  5052,
        #           3667,  2012,  6769,  2047,  8095,  2360,  2027,  2024,  1005,  9364,
        #           1005,  2044,  7566,  2007, 16654,  6687,  3813,  2976,  9587, 24848,
        #           1012,   102
        #         ]
        #     ]),
        #     'token_type_ids': tensor([
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #          0, 0, 0, 0, 0, 0, 0, 0]
        #     ]),
        #     'attention_mask': tensor([
        #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #          1, 1, 1, 1, 1, 1, 1, 1
        #         ]
        #     ])
        # }
        enc = tokenizer(
            batch["text"], truncation=True, max_length=128,
            padding=True, return_tensors="pt",
        )

        # We basically push the tensors to the GPU here
        enc = {k: v.to(model.device) for k, v in enc.items()}

        # Without creating the graph needed for backpropagation
        with torch.no_grad():
            # logits will be a tensor where each row will contain the
            # logcounts for the classification id for each input
            # We have 4 IDs and considering we have 3 inputs in the batch,
            # it will return something like:
            # tensor([[-0.3222, -0.6178,  0.7770],
            #         [-1.4684, -0.0310, -0.5616],
            #         [-0.6865, -1.3338, -1.1234],
            #         [-0.1443,  0.6659, -1.4422]])
            logits = model(**enc).logits

        # We get the index of the highest log count in the row,
        # for the logits above, the preds will look like:
        # tensor([2, 1, 0, 1])
        preds.extend(logits.argmax(dim=-1).cpu().tolist())

        labels.extend(batch["label"])
        texts.extend(batch["text"])

    return preds, labels, texts

preds, labels, texts = predict_split(test_ds)

# This will result in a list of True/False values
classification_matches = np.array(preds) == np.array(labels)

# The mean operation on the list of True/False values
# will calculate the mean using True=1 and False=0
acc = float(np.mean(classification_matches))

print("test accuracy", acc, "n", len(labels))

confusion = np.zeros((4, 4), dtype=int)

for truth, pred in zip(labels, preds):
    confusion[truth, pred] += 1

print(confusion)

for i in range(4):
    total = confusion[i].sum()
    print(LABELS[i], confusion[i, i] / total, "n", total)

wrong = [i for i, (t, p) in enumerate(zip(labels, preds)) if t != p and t == 0 and p == 2]
# print three, preferably from the weak class or the big off-diagonal
for i in wrong[:5]:
    print("true", LABELS[labels[i]], "pred", LABELS[preds[i]])
    print(texts[i][:400])
    print("---")

wrong = [i for i, (t, p) in enumerate(zip(labels, preds)) if t != p and t == 0 and p == 3]
# print three, preferably from the weak class or the big off-diagonal
for i in wrong[:5]:
    print("true", LABELS[labels[i]], "pred", LABELS[preds[i]])
    print(texts[i][:400])
    print("---")