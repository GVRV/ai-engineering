Yesterday’s last logged learning rate was 2.963e-09. The schedule started at 2e-5 and decayed to nearly zero by the end of the epoch. If you call trainer.train(resume_from_checkpoint=...) on that trainer folder, you also restore the scheduler. The next steps would use a learning rate of about zero. The weights would not move.

What you want is the weights only. AutoModelForSequenceClassification.from_pretrained("experiments/classify/runs/day21/final") loads those weights. A new Trainer then starts a new schedule at 2e-5 again. That is continued training, not a resume. The optimizer state from yesterday is thrown away. That is fine. Adam rebuilds its moments in a few hundred steps.

Overfitting is the thing you are watching. Train loss can keep falling while validation accuracy stalls or drops. The checkpoint you keep is the one with the best validation accuracy, not the one from the last epoch. load_best_model_at_end=True does that reload for you, if you set the metric.

# Notes

Loaded from: ./runs/day21/final
id2label printed:
```
(Pdb) model.config.id2label
{0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
```

Epoch 1 train loss, val accuracy: {'eval_loss': '0.1936', 'eval_accuracy': '0.9399', 'eval_runtime': '115.1', 'eval_samples_per_second': '104.3', 'eval_steps_per_second': '3.259', 'epoch': '1'}
Epoch 2 train loss, val accuracy: {'eval_loss': '0.2321', 'eval_accuracy': '0.9462', 'eval_runtime': '133.1', 'eval_samples_per_second': '90.16', 'eval_steps_per_second': '2.817', 'epoch': '2'}
Day 21 val accuracy: 0.9407

Which epoch was best: 2
final path, and the checkpoint folder it came from: "runs/day22/final" from checkpoint 2

Two probe sentences, day 21 vs day 22:
- [{'label': 'World', 'score': 0.8471630215644836}] (Changed from Yesterday's "Sports" classification with a lower probability 0.7880864143371582)
- [{'label': 'Business', 'score': 0.9957653284072876}] (Slightly higher than yesterday's 0.9920560717582703)

Did I resume the old scheduler: no
Test touched: no

What I still find shaky, in full sentences:
- Probably spent 95% of today's time just waiting for the model training to run. Can we pre-plan big training runs so that we're not including the training compute times within the day's allocated time budget (by maybe pre-planning tomorrow's training today and letting the computer run overnight, etc)