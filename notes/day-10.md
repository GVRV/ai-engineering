Autograd is a machine that applies the chain rule to every +, *, tanh I ran.

Today I apply the chain rule to whole matrices at once.

Same rule as Day 2: if z = f(y) and y = g(x), then the gradient of the loss wrt x is the gradient wrt y, multiplied by how y changes when x changes. Or in other words:

dz/dx = dz/dy * dy/dx

The only new skill is writing that multiply so it matches tensor shapes.

What I did today, in ordinary language:
I took one minibatch, let PyTorch compute gradients, then computed
the same gradients with matrix formulas and compared them.

Formulas that matched:
- classification loss → dlogits
- dW2 / db2 / dH
- tanh
- (BatchNorm gain/bias / preact — yes/no)

The linear-layer pattern I will reuse:
gradient wrt weights = incoming activations.T @ incoming gradient
gradient wrt activations = incoming gradient @ weights.T

What still does not match, and the max abs difference: Everything matches on the first run but I didn't proceed with simplifying the cross_entropy loss and the batch normalization operations (again, because I'm finding the math a little intimidating and I don't feel confident simplifying the derivations calculations right now). I watched the full video but only did Exercise 1 (and even that, while peeking at the video but trying to do the calculations by myself took like 4 hours).

I did not start the GPT video.

What I still find shaky, in full sentences: I think Andrej got a little carried away with this exercise. I understand that it is very useful to know the atomic mathematical operations and be able to do the backpropagation manually if needed but from a practical skills point of view, I highly doubt anyone who is training any networks of a certain size are doing this manually because it is super tedious.