f(x) = 3x^2 - 4x + 5

If x = 2, then f(x) = 3 * 2^2 - 4 * 2 + 5 = 3 * 4 - 8 + 5 = 12 - 3 = 9
If x = 2.0001, then f(x) = 9.00080003


df/dx = 6x - 4
At x = 2, df/dx = 6*2 - 4 = 8

So, in our example, when we bumpled x by 0.0001 we saw a rough bump in f by 0.0008 matching our calculations.

The derivative of L wrt x is: if I bump x by a tiny h, L changes by dL/dx.

For multivariable functions, we can have derivatives for each variable:

f(a, b) = 3*(a^2)*b + 4*a*(b^2) + 5

df/da = 6ab + 4*(b^2)
df/db = 3*(a^2) + 8a


A gradient is: how much does a given value change if one of its dependent variables is changed by a very small amount. For the loss function of the neural network, it depends on many parameters and depending on the relationship of the parameter to the final loss, we use the gradient to nudge the parameter in the right direction.

The chain rule on a graph is: df/da = df/db * db/da i.e. if we know how a variable a affects a variable b and we know how the variable b affects a variable f, then we can easily find how how a affects f.

Why we += grads (the a+a bug): This was a python object reference bug. If the same object is memory is referenced by both self and other in certain operations, its grad value gets overwritten instead of being cumulatively affected.

What .backward() does in 3 steps:
1.Start with a root node where the gradient is 1 and then immediately calculate the gradients of its previous operation neighbours over a single mathematical (or logical unit of a number of mathematical) operation
2. Repeat this process with the neighbours of neighbours using topological sorting applying the chain rule as we already know the gradient from step 1
3. Continue this process until all nodes in the graph have been covered. At this point, even for leaf nodes, we know the gradient of how a small change in its value will affect the root node value we started with.
What I still don’t trust: Nothing really, everything made a sense today. I don't like using closures for calculating backward though (I guess it's memory efficient) and it made a lot more sense after clarifying that we need to use a reverse topologically sorted list.