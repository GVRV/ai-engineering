Read the Parameter-efficient Fine-Tuning (PEFT) documentation [here](https://huggingface.co/docs/peft/en/index).

LoRA (Low-Rank Adaptation) is a method to improve the performance of general base models on specific tasks without optimizing all of its parameters (like during fine-tuning). The authors of the LoRA paper guessed that optimizing a model for a particular task should be "low-rank" (i.e. should not involve changing all of the parameters like a complete forward-pass during fine-tuning). Instead, optimizing the model on specific tasks can be performed in a low dimensional vector space (which makes sense considering this low dimensional vector space can then feed into the bigger model to correctly perform the optimized task). LoRA has been popular because it involves small task-specific adapters that can plug into general, large models. The time for LoRA optimization is lower considering parameters being updated are lower and it is resource efficient. We can merge back the LoRA weights into the base model to avoid inference time costs as well. For example, in a deployed classifier, you would merge the adapter into the model so that the inference is as quick as the general model. In a multi-tenant API, you can have a single general base model and tenant specific adapters for the specific tasks each tenant wants to optimise for. However, we must be clear that LoRA is not a smaller model, it relies on the base, general large model for capabilities and cannot be considered a magic pill if the base model is unsuitable for the optimization task.

# Optimization

- Full fine tuning: Optimise all of the parameters of the model to reduce the loss. Computational expensive, but works to teach the base model additional capabilities.
- LoRA: When optimising all parameters of the model is not viable (eg: using consumer hardware) but we have a capable base model we can optimise for specific tasks.
- Prompting only: specific task is close to what the base model can do, good prompt required with examples.
- Do not LoRA when the base model architecture is unsuitable for the task, or we need to change the embedding table or tokenizer.

Considering we already fine-tuned the AGNews model to improve the loss, using LoRA seems like a good second experiment to see if we can squeeze better performance from the model.

# LoRA Trainable Parameters

trainable params: 741,124 || all params: 67,697,672 || trainable%: 1.0948

# LoRA Accuracy

test accuracy 0.9026315789473685 n 7600

Compared to the fine-tuned model (parameters = 67M, accuracy = 0.9441), the accuracy is worse but we didn't spend a lot of resources on LoRA optimization.

# Notes

Rank, alpha, targets: 8, 16, 768 + 768
Trainable params: 741,124
Test accuracy, full: 0.9441
Test accuracy, LoRA: 0.9026
Gap: 0.0415
Intel headline, both models:
    - Fine tuned: [{'label': 'Sci/Tech', 'score': 0.9742323160171509}]
    - LoRA: [{'label': 'Sci/Tech', 'score': 0.902138888835907}]

Split matched the 0.9441 run: yes
Uploaded full model again: no

What I still find shaky, in full sentences:
- I still I've forgotten the basics of the self-attention architecture and transformers, might be worth it to revise it soon (maybe using some other exercise so that the context is different)?