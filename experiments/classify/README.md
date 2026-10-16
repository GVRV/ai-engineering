# Classifier Model Project

## Introduction

The model is a DistilBERT encoder with 4-way head, fine-tuned to label a news blurb as World, Sports, Business, Sci/Tech. This is a classification model; it is not a chatbot or a generator.

### Data

The training data used is the [fancyzhx/ag_news](https://huggingface.co/datasets/fancyzhx/ag_news) dataset officially consisting of 120,000 training samples and 7,600 test samples. The training sample was cut into 108,000 sample training split and a 12,000 validation split (using seed `42` for deterministic splitting). The official test split was never seen during training or checkpoint evaluation.

### Training

Training started from the [distilbert/distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased) model and tokenizer. We tokenized each headline by truncating it to 128 tokens (or padding when necessary). We ran one epoch of the training split from the base model using a starting learning rate of `2e-5`. Then we reloaded the model from this checkpoint and trained again for two more epochs of the training split with a fresh starting learning rate of `2e-5` as the LR had decayed to almost zero during the previous run. During this run, we saved the model with the highest accuracy on the validation split epoch. The training used MPS device backend for PyTorch as it was performed on Apple Silicon.

### Accuracy

- Final Test split accuracy: 0.9441
- Validation split accuracy at selection: 0.9462

The minor gap of `0.0021` isn't significant enough to signal the model training was not successful.

#### Per-class Accuracy

The test split accuracy across the 4 classes is as follows:
- World 0.9511 (samples = 1900)
- Sports 0.9847 (samples = 1900)
- Business 0.9089 (samples = 1900)
- Sci/Tech 0.9316 (samples = 1900)

The model is quite accurate as classifying `World` and `Sports` headlines. The slight degradation in accuracy for `Business` and `Sci/Tech` is due to the model incorrectly classifying headlines which contain a mix of business and science/technology (like business articles about a technology company, or a technology article citing a business leader). For example:

```
Headline: Intel to delay product aimed for high-definition TVs SAN FRANCISCO -- In the latest of a series of product delays, Intel Corp. has postponed the launch of a video display chip it had previously planned to introduce by year end, putting off a showdown with Texas Instruments Inc. in the fast-growing market for high-definition television displays.

Truth: Business

Prediction: Sci/Tech

Reason: It seems like a lot of text does indeed have overlap between technology firms and technology centres like San Francisco, but the article is about business. The model is having a hard time with these distinctions when it deals with such companies.
```

## Run the model

Use the `classify.py` helper script to classify a headline:

```
> python classify.py
usage: python classify.py <headline>

> python classify.py "Phelps wins gold in the 200 meter freestyle"
[{'label': 'World', 'score': 0.8471630215644836}]

> python classify.py "Wall Street shares fall after the rate decision"
[{'label': 'Business', 'score': 0.9957653284072876}]

> python classify.py "IBM to hire even more new workers by the end of the year"
[{'label': 'Business', 'score': 0.8044154047966003}]
```

Please note: `runs/day22/final` is `.gitignored` so the weights will be unavailable if you have cloned this repository. You will also notice that the first headline mentioning legendary swimmer `Phelps` is mis-classified as `World` instead of `Sports`: this is a known failure case of the model.

## Caveats

This is a toy project and not a production grade classification model. We have ignored production ready polishing such as lowering response latency, adding monitoring and decisions on how to optimise classification tasks where the model is confused between two (or more) categories (like Business & Sci/Tech).

### TODO for Production Service

- A refusal or low-confidence bucket when the model is confused between 2 or more classes (especially as we know this is the case with `Business` & `Sci/Tech`)
- A proper test eval script that you can run when checkpoint changes easily. The accuracy number indicates one thing, but having a set of gotchas/known failures in the eval would help us understand real progress.
- A saved copy of the three splits (training, validation, test; generated with seed 42) so we have a deterministic dataset which can be used to replicate our accuracy on other machines.
- Compare the training performance on MPS with a small CPU box, to share performance expectations for anyone else running training.
- No fine-tuning on the test split even if there are customer complaints and eval failures.