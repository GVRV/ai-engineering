What a gradient is: the gradient is the ratio how much a small change in a given parameter will change the loss value. Over time, we want to nudge parameters in the direction opposite to the gradient so that loss is minimised.

What the computation graph is: I'm not sure if this was covered in the video but I'm guessing this is the interconnected graph of various weights and biases that finally converge with a loss (something like what Andrej was displaying visually using the GraphViz library)

What .backward() does: The backward pass is the process of calculating the gradients for all of the parameters of the model using the chain rule we covered in Day 2. After the backward pass, each of the parameters (weights and biases) of the model have a numeric value "gradient" which tells us how a small change in this parameter will change the loss. Because our goal is to minimise the loss, we use the gradient in the opposite direction when making our adjustments after the backward pass is complete.

Why we zero grads: During backpropogation, the gradients in our parameters are added up cumulatively because as we discussed in Day 2, we want the cumulative effect of different chain rules to get the final gradient for a single parameter. However, on subsequent backward passes, we want to re-calculate the gradient of each parameter from 0 because as we have nudged their values in the meanwhile to minimise the loss. Retaining their old gradient value and then adding to it cumulatively during a subsequent backward pass will give us the wrong value for the gradient leading to incorrect network optimization.

What the training loop is (5 lines):
- Start with a forward pass to calculate the network predictions using inputs from the training set (or a subset of inputs from the training set)
- Use a loss calculation technique (eg: mean squared losses) to compare the network predictions with the actual expected values from our training set and generate a single scalar loss value for the network.
- Do a backward pass to calculate the gradients for all of the parameters (weights and biases, not input nodes) of our network using a zero grad starting point.
- Go through all of the parameters and nudge them a tiny bit in the direction opposite to the gradient (i.e. gradient descent) so that our loss value actually moves lower
- And then we restart the loop again with a forward pass and re-calculating the loss etc. There are many efficiencies available here (eg: batching inputs, how many loops to run, how much step size to use, etc) but we didn't cover that for now.

How this relates to PyTorch (Value ≈ tensor scalar + autograd): We didn't cover much of how things work in PyTorch except for the fact that PyTorch is highly optimised for efficiency and the API is kinda similar.