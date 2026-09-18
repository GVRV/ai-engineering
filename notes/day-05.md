logits = unnormalised scores

softmax = exp + divide by row sum = a distribution

loss = NLL of the true next char

xs / ys are: xs is the list of all the indexes of the input characters. ys is the list of the all indexes of the ground truth predictions
one-hot does: It's a method of encoding integers to a vector such that the if the integer is x, the xth dimension of the vector is 1 but every other dimension is 0.
W is: The weights of our model is a matrix of size 27 x 27 which contains the log probabilities of the predictions.
logits vs probs: logits are the log counts, you need to exponentialise the logits to get the counts and then normalise the counts to get the probabitilies summing to 1.
softmax in one line: Softmax is an operation to turn a vector into a probability distribution
initial loss for random net: 3.819756269454956
initial loss: 3.8231
final loss (step N): 2.4807
count-model NLL to beat: 2.4544
why we shouldn't beat it by a lot: then we overfit the model to our data instead of remaining general.
regularisation is the neural twin of: smoothing the counts
what I still don't trust: why pytorch got so popular considering the API is horrendous. One hot seems like a very inefficient mechanism to encode integers but I guess it doesn't matter because modern neural nets are quite big (even converting integers to binary and then using those 1s as input might be better?) A lot of hyperparameters were just guessed (eg: how many iterations of training, the learning rate, the regularization parameters, etc) - is there a science to it or you just go with trial and error?

twenty names from today's model:
an.
syngesharowaimad.
roliaia.
try.
vy.
ste.
haneinde.
risonare.
aelendy.
radusthondrnsa.
en.
flalmafisonaijaljald.
koriarielvili.
latovetzaq.
e.
maya.
zaya.
ann.
mah.
kea.