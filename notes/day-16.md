# Block 0

Got the script working:
```
>>> from client import complete
>>> complete("You are a terse tutor.", "What is 17 + 4? Reply with the integer only.")
'21'
```

# Block A

A pretrained model continues documents. A chat model was then trained to treat a system message and a user message as a conversation.

Neither one looks up a fact at answer time unless that fact is already in the window, or a tool put it there.

If the fact is missing, the model still produces a fluent continuation. That continuation can be wrong. People call that a hallucination. It is the same next-token behaviour as the Shakespeare model, aimed at a helpful tone.

Prompting does not retrain weights. It only changes the text in the window.

Prompt Injection is introducing something in the prompt context window that will cause the model to do something that is not expected by the user (for example: a completely white image might be hiding nearly white text intructions to inform the user something incorrectly or a big document might be hiding some lines that ask the model to ignore previous instructions and do something nefarious).

# Notes

What temperature=0 means, in a sentence: We want to keep the entropy (?) low so that we're always getting the same answers instead of a little bit of probabilistic variance.

Bare vs constrained vs page-in-the-window, what changed: In the bare response, the model was very verbose and gave a structured, long reply (it was even cut short by the client). In the constrained prompt, the reply was short and correct. In the page-in-the-window prompt, the reply was short, and able to cite relevant information present in the context.

JSON parse failed on which runs: 0 (models are good now, I guess?) as both runs returned valid JSON and the relevant keys.

Score: grounded 10 / 10
The worst invention: None. The models are good now, I guess?

Prompt injection, in a sentence I could say to a teacher: I'm guessing we're not talking about injection in the security context here? so more like prompt engineering? It is the ability to shape the reply of the model by using the prompt instructions to be more effective and generate responses in a given structure, using relevant information (and maybe not its world of knowledge).

What I still find shaky, in full sentences: Apart from prompt engineering being a subtle art and not a science, and the constant change of model behviour between updates and different providers, nothing.