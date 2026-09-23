Yesterday we asked “is the number on held-out names good?”

Today we ask “at step 0, before any learning, is the network behaving like a random guesser, or is it already broken?”

A random guesser over 27 letters should have loss about −log(1/27) ≈ 3.3.

If step-0 loss is 20, the network is not guessing — it is loudly wrong.

# Block A

With 100 hidden layer width, 2 dimensions, 200000 training runs, and a 0.111 learning rate (num_params=3481), we get the following losses:
Run 0: cross_entropy_loss=tensor(20.6873, grad_fn=<NllLossBackward0>)
Run 100: cross_entropy_loss=tensor(3.6226, grad_fn=<NllLossBackward0>)
Run 200: cross_entropy_loss=tensor(3.1645, grad_fn=<NllLossBackward0>)
Run 300: cross_entropy_loss=tensor(2.9716, grad_fn=<NllLossBackward0>)
...
Run 200000: cross_entropy_loss=tensor(2.4493, grad_fn=<NllLossBackward0>)
dataset_name='Training'
loss=tensor(2.4403)
dataset_name='Validation'
loss=tensor(2.4252)
dataset_name='Test'
loss=tensor(2.4301)

With torch.randn initialization, in the first run:
logits = tensor([  4.5147, -12.3937,  -7.5219,   8.5622,  -0.2654,  -8.7625,   4.9267,
          1.5101,   1.1081,  -1.8234, -13.8715,  -4.0687,  -6.1157,  -6.5670,
        -12.2107,  11.0033,   3.1504,   7.8640,  -3.2068, -11.7253,  -8.5989,
        -10.8444, -13.3724,   0.3356, -16.2003, -15.1165,  13.4284],
       grad_fn=<SelectBackward0>)
logits.std() = tensor(8.4455, grad_fn=<StdBackward0>)
logits.abs().max() = tensor(24.2089, grad_fn=<MaxBackward1>)
loss = cross_entropy_loss=tensor(19.2012, grad_fn=<NllLossBackward0>)

# Block B

Softmax on huge scores picks one letter with probability ~1.

If that letter is not the true next letter (almost always, at random init), the loss is enormous.

Shrinking the output weights makes the 27 scores close to each other, so the model starts near “I don’t know yet.”

See pictures for loss_with_random_initialization and loss_with_standard_initialization photos. The former has a hockey stick, the latter does not.

1000 runs with random init (losses every 100 runs):
cross_entropy_loss=tensor(15.8863, grad_fn=<NllLossBackward0>)
cross_entropy_loss=tensor(3.5190, grad_fn=<NllLossBackward0>)
cross_entropy_loss=tensor(3.1581, grad_fn=<NllLossBackward0>)
cross_entropy_loss=tensor(2.8747, grad_fn=<NllLossBackward0>)
cross_entropy_loss=tensor(2.8552, grad_fn=<NllLossBackward0>)
cross_entropy_loss=tensor(2.8933, grad_fn=<NllLossBackward0>)
cross_entropy_loss=tensor(2.3549, grad_fn=<NllLossBackward0>)
cross_entropy_loss=tensor(2.6623, grad_fn=<NllLossBackward0>)
cross_entropy_loss=tensor(2.7409, grad_fn=<NllLossBackward0>)
cross_entropy_loss=tensor(2.2429, grad_fn=<NllLossBackward0>)
dataset_name='Training'
loss=tensor(2.5358)
dataset_name='Validation'
loss=tensor(2.5436)
dataset_name='Test'
loss=tensor(2.5374)

1000 runs with standardised init (losses every 100 runs):
cross_entropy_loss=tensor(3.3146, grad_fn=<NllLossBackward0>)
cross_entropy_loss=tensor(2.7660, grad_fn=<NllLossBackward0>)
cross_entropy_loss=tensor(2.6123, grad_fn=<NllLossBackward0>)
cross_entropy_loss=tensor(2.7384, grad_fn=<NllLossBackward0>)
cross_entropy_loss=tensor(2.4991, grad_fn=<NllLossBackward0>)
cross_entropy_loss=tensor(2.5440, grad_fn=<NllLossBackward0>)
cross_entropy_loss=tensor(2.8426, grad_fn=<NllLossBackward0>)
cross_entropy_loss=tensor(2.8934, grad_fn=<NllLossBackward0>)
cross_entropy_loss=tensor(2.5733, grad_fn=<NllLossBackward0>)
cross_entropy_loss=tensor(2.6892, grad_fn=<NllLossBackward0>)
dataset_name='Training'
loss=tensor(2.4650)
dataset_name='Validation'
loss=tensor(2.4773)
dataset_name='Test'
loss=tensor(2.4696)

# Block C

Before shrinking:
Tanh hidden layer has 726 flat values with zero gradient out of 3200: 22.6875% flat

After shrinking:
Tanh has 34 flat values with zero gradient out of 3200: 1.0625% flat

After fixing both hidden layer and overconfidently wrong softmax
With 100 hidden layer width, 2 dimensions, 200000 training runs, and a 0.111 learning rate (num_params=3481), we get the following losses:
dataset_name='Training'
loss=tensor(2.2004)
dataset_name='Validation'
loss=tensor(2.2188)
dataset_name='Test'
loss=tensor(2.2146)

# Block D

After Kaiming initialization and fixing both hidden layer and overconfidently wrong softmax
With 100 hidden layer width, 2 dimensions, 200000 training runs, and a 0.111 learning rate (num_params=3481), we get the following losses:
dataset_name='Training'
loss=tensor(2.1929)
dataset_name='Validation'
loss=tensor(2.2020)
dataset_name='Test'
loss=tensor(2.2111)

# Block E

What “step-0 loss” means:
The loss after building the network, before any weight update.
For 27 equally likely letters it should be about 3.3 (-log(1/27)).

What I measured with raw randn output weights:
step-0 loss = 19.2012
logits typical size = 8.4455

After shrinking W2 and zeroing b2:
step-0 loss = 3.3146

Why huge logits are bad, in ordinary language: logits are log counts, so if there are huge logits, it means that our model is assigning a very high chance that to one or certain predictions but without any training. This means our model is confidently wrong.

What tanh saturation means:
tanh squashes to (−1, 1). Far out on the tails the slope is ~0,
so the backward pass cannot move those weights.

Fraction of hidden units with |tanh| > 0.99:
  before scaling W1: 22.6875%
  after scaling W1: 1.0625% (scaled using 0.4 using vibes)
  after scaling W1: 12.96875% (scaled using kaiming initialization scale of 0.6804)

The fan-in rule I used, and why dividing by sqrt(fan-in) exists: Assuming that our inputs are a gaussian distribution with mean 0 and std deviation 1, and our weights are a gaussian distribution with mean 0 and std deviation 1, when we multiply inputs with the weights, we get another distribution where the mean is still roughly 0 but the std deviation fans out as the multiplied numbers spread out both positively and negatively. To again make the std deviation close to 1, we need to normalise the weights so that the multiplication doesn't result in the spreading out of the numbers. This normalization was calculated by Kaiming and collaborators for different activation functions and is popularly used today.

I did not implement Batch Normalization today.

What I still find shaky, in full sentences: The math. I don't undestand it but I'm just taking it at face value and assuming the authors did their job properly and the peer reviews were tough, because I don't understand it and it doesn't look approachable (especially when Andrej mentioned that bigger, deeper models can run training but not optimise at all). Andrej also mentioned that initialising the biases to 0 causes some "smoothing problem" but then didn't get back to it except for mentioning the next time that he usually multiplies the biases with a small number like 0.01 to introduce some entropy. I undestand why introducing a very small amount of entropy might be a good idea but is that the smoothing issue he was referring to?