A language model predicts: The next token/word given a number of input tokens/words.
A bigram model assumes: We want to guess the next letter given a current letter. Any previous letter already part of the word don't count towards the next letter prediction.
N is: A tensor of counts of the number of instances of pairs of letters we see in our training set. The row has the counts of all the other letters which might follow it for a particular letter.
P is: The probabilities of the counts normalised so that they add up to 1 (i.e. instead of here's how many times 'a' followed 'b', here's the probability of 'a' following 'b' where the sum of any letter following 'b' including the end character will total up to 1)
Sampling is: This was not covered? But I'm guessing it's keeping the training set smaller and then actually checking the loss against the probabilities calculated using the training set?
NLL is: Negative log likelihood. Likelikehood is the product of the probabilities of a given set of predictions. As probabilities are already tiny, and multiplying them can result in even tinier numbers, logging the likelihood gives us a sum ranging from -INF to 0 that penalises extremely unlikely predictions quite well. Negative log likelihood is used so that we have a positive number we can try to minimise like a conventional loss value.
My unsmoothed NLL: 2.4544
My N+1 NLL: 2.4544 (just adding +1 for smoothing didn't move the needle much, if I added +5 then it shows a worse NLL of 2.4616)
Why we add 1: Because the log of 0 is -INF, a single prediction that didn't occur in our distribution might result in our loss being infinite. To prevent this from happening, we smooth out the distribution by adding a constant count to all possible predictions.
20 sample names: cexze.momasurailezitynn.konimittain.llayn.ka.da.staiyaubrtthrigotai.moliellavo.ke.teda.ka.emimmsade.enkaviyny.ftlspihinivenvorhlasu.dsor.br.jol.pen.aisan.ja.
What I still don't trust: I don't get minimizing the log likelihood because to do that wouldn't we just go with the highest probability next token given a character always?

Train/Eval: change probabilities (`P`) look-up so that real bigrams have high probability (low NLL)

Generate: draw from finished probabilities look-up (`P`). Argmax != training.