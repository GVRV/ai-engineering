from datasets import load_dataset, Dataset
from collections import Counter

ds = load_dataset("sh0416/ag_news")

print(ds)

print(ds["train"].column_names)
assert ds["train"].column_names == ['label', 'title', 'description']

print(len(ds["train"]))
print(len(ds["test"]))

print(ds["train"][0])
print(ds["train"][1])

def class_counts(split: Dataset) -> dict[int, int]:
    counts = Counter()
    for row in split:
        counts[row['label']] += 1
    return dict(counts)

print(class_counts(ds["train"]))
print(class_counts(ds["test"]))

LABELS = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech",
}

def print_samples(split: Dataset) -> None:
    shown = set()

    for row in split:
        if row["label"] not in shown:
            print(LABELS[row["label"]])
            print(row["title"][:240])
            print(row["description"][:240])
            shown.add(row["label"])

        if len(shown) == len(LABELS.keys()):
            break

print_samples(ds["train"])

ORIGINAL_LENGTH = len(ds["train"]) + len(ds["test"])

def make_splits(ds, val_ratio=0.1, seed=42):
    split_ds = ds["train"].train_test_split(test_size=val_ratio, seed=seed, shuffle=True)
    split_ds["validation"] = split_ds["test"]
    split_ds["test"] = ds["test"]
    return split_ds

ds = make_splits(ds)

print(len(ds["train"]))
print(len(ds["validation"]))
print(len(ds["test"]))

NEW_LENGTH = len(ds["train"]) + len(ds["validation"]) + len(ds["test"])
assert NEW_LENGTH == ORIGINAL_LENGTH