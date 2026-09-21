Bigram NLL ≈ 2.45 is the MLE (Maximum Likelihood Estimation) for “P(next | one letter).”

To beat it we must change the assumption: more context, or shared structure across letters.

Today: context = 3 chars, each char → a short vector (embedding).

Why bigram couldn't beat 2.45: Because it was predicting the next character looking at only one previous character, there's a ceiling to its performance as there will always be a loss in the distributions of the predictions (eg: if 'a' and 'b' both follow 'c' 50% of the time, whenever 'a' is predicted, the model incurs the loss of not predicting 'b' and whenever 'b' is predicted, the model incurs the loss of not predicting 'a')

block_size means: the context size. How many characters does the model look at as input before giving you the output character.

C is: An embedding table. Instead of using a unique identifier for each different input character, it makes sense to encode input characters as low dimensional vectors. Apart from the performance improvements this provides, as the embedding table parameters are optimised by the training of the model, the model effectively encodes similar characters as roughly vectors pointing the same direction.

C[X] shape: if C is [27, 2] and X is [32, 3], then C[X] is [32, 3, 2]

why view / concat: Because we have a batch size of 3, and because we're using 2 dimensions to encode each character, we will need to use 6 (3 * 2) numbers as input to our model. Using `view` or `concat` are different ways to rearrage the data in the tensor so that it aligns with how we can easily use it to provide input to the hidden layer.

param count: 3481
overfit-32 final loss: 0.2538
full-data train loss after N steps: 2.2761
what I still don't trust: Figuring out the hyperparameters still seems like a bit of an art (yes, we can do some experiments and kinda figure out a good learning rate, etc but it seems like some experimentation is always required). Going back to the main topic for day, it seems like increasing the context window will always give us a method to decrease the loss and make the model better but it takes more computing for training and inference (so there will be a trade-off curve somewhere?)