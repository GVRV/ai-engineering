Yesterday we scaled the first weights so tanh would not start stuck at −1 and +1.

That fix is only about step 0. After many updates the preactivations can blow up or collapse again.

Batch Normalization forces each hidden unit, on each minibatch, to have average about 0 and spread about 1, then lets the net learn its own scale and shift.

It is a band-aid that made deep nets trainable. We implement the band-aid so we know what nn.BatchNorm1d is doing later.

As we covered yesterday that for our nonlinear layer of tanh, we want the values feeding in to be roughly gaussian with a mean of 0 and a standard deviation of 1 so that large values don't result in the tanh operation being 1 or -1 which won't get the gradients flow back using backpropagation but conversely, we also don't want all of the values to be too small because for very small values tanh(a) roughly equals a, so instead of a non-linear function, we get a linear function.

To avoid this problem of having too large values but also avoiding too small values, we want the distribution to be roughly gaussian with a mean of 0 and a std deviation of 1 and a team of google came up with a solution called BatchNormalisation which tackles this. The basic idea is that if we want the pre-activations to be roughly gaussian before the tanh operation, we can just do mathematical operations on them that make the values roughly gaussian (at least for that mini-batch) and these mathematical operations are differentiable so gradients will backpropagate fine.

The value to make a distribution roughly gaussian is using the mean and the std deviation of that sample of values:

roughly_gaussian_values = (values - mean) / std_deviation

Even without kaiming initialization, the number of tanh preactivations which are (1, -1) stays low:
Tanh has 12 flat values with zero gradient out of 3200: 0.375% flat

Results (hidden layer width = 100, embedding dimensions = 2, learning rate = 0.111, training runs = 500000 with BatchNorm and no Kaiming initialization):
Training loss (all training rows, end of run): 2.2249
Validation loss (all validation rows): 2.2238
Test loss (computed once, after I stopped changing things): 2.2334

What Batch Normalization does to one hidden unit on one minibatch:
1. subtract that unit's mean over the 32 examples
2. divide by that unit's spread over the 32 examples
3. multiply by a learned gain (starts at 1)
4. add a learned bias (starts at 0)

Why we bother:
So tanh keeps seeing numbers near 0 after step 0, not only at step 0.

Why evaluation cannot use the current batch's mean and spread:
A single generated name is a "batch" of 1. Its own mean/std is meaningless.
We store a slow running average of training batch statistics and use that.

What I had to add to the parameter list:
bn_gain, bn_bias  (the running mean/std are NOT learned by backward)

Step-0 loss with BatchNorm:
|tanh| > 0.99 fraction at step 0: 0.375%
Training loss: 2.2249
Validation loss (running stats): 2.2238
Test loss (once): 2.2334

Compared to Day 8 val 2.20: We're slightly worse but still comparable.

The linear bias just before BatchNorm is redundant because: Because we're adding a fixed value to all of the preactivation values and then regularalizing them by subtracting the mean from each of the value. Because the mean already has the bias included, the net result will be redundant. It's like taking a list of 10 random numbers, adding 2 to each and then subtracting the mean of the numbers from each of them or just taking the 10 random numbers and subtracting the mean from each without adding anything - the net result will be the same as in the mean for the former case is the mean of the latter case + 2.

What I still find shaky, in full sentences: Again, math is the weakest point, I'm trusting that the math on BatchNormalization done by the team at Google actually checks out (not only makes intuitive sense). What I find strange is that the batch normalization weights and biases work across batches because the next batch might be a complete random set of input values which might throw our current batch normalised weights and biases offtrack but I guess over a large number of training runs, this doesn't matter. Whoa! OK and the whole ResNet video and diagnostic techniques that Andrej flied through in the video, I can only understand them at the surface level but I don't really understand what's a good range and why.