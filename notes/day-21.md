DistilBERT is a pre-trained encoder: it converts text into vectors. We're not training it from scratch, we're adding a neural net layer at the end of it so that the vectors it encodes can pass through a NN layer and result in a classification score. We will then train this NN layer and the pre-trained DistilBERT model on the AG news dataset to classify headlines correctly.

The loss is cross-entropy. For each row, the model outputs logits which a softmax converts into probabilities. We take the probability of the correct classification, compare it with the probability given by the model and compute the loss. `Trainer` does the backward pass and optimiser step today, we're not writing the backward pass.

A checkpoint is the weights plus the tokenizer written to disk. It can be used to save progress on a training run and we can restart the training from this checkpoint in the future.

Test split will not be used at all today.

# Notes

Device: mps
Dataset id, model id: fancyzhx/ag_news, distilbert/distilbert-base-uncased
Train rows used (512 smoke / then the real number): 512 for smoke, 108000 for real
Val rows used: 128 for smoke, 12000 for real

Smoke checkpoint path, and the two files that prove it saved: `./runs/smoke/`
The directory contains the following files:
```
config.json
model.safetensors
optimizer.pt
rng_state.pth
scheduler.pt
tokenizer_config.json
tokenizer.json
trainer_state.json
training_args.bin
```

Real run train loss: {'loss': '0.1884', 'grad_norm': '0.2637', 'learning_rate': '2.963e-09', 'epoch': '1'}
Real run val accuracy: {'eval_loss': '0.1827', 'eval_accuracy': '0.9407', 'eval_runtime': '108.9', 'eval_samples_per_second': '110.2', 'eval_steps_per_second': '3.442', 'epoch': '1'}
Saved path: `./runs/final`

Two pipeline predictions:
- [{'label': 'Sports', 'score': 0.7880864143371582}]
- [{'label': 'Business', 'score': 0.9920560717582703}]

Test split touched: no

What I still find shaky, in full sentences:
- We're using all these lego blocks already available like `transformers` and `accelerate` library and `DistilBERT` model etc. How do we gain information about the available lego blocks? Which ones are good and industry standard and which ones are not? Where does it make sense to write things from scratch instead of building using these pre-made blocks? How do you develop this intuition and knowledge?
- "data_collator pads input_ids in a batch to the same length." - why is this required?
