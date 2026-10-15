Validation accuracy 0.9462 was the number I used to pick the checkpoint.

Test accuracy is the number I show someone else. I did not choose the model based on it.

I will score the official 7,600-row test split once. If it is worse than validation, I write that down. I do not train again today.

# Block A

Scoring loss/accuracy on the validation and test splits have different purposes.

The validation split is used to score loss/accuracy during the training at our pre-determined checkpoints (for example, each epoch of the training set yesterday, so we scored validation split twice yesterday and by co-incidence the last epoch scored higher – if our training run lasted many many more epoches, maybe the last epoch might not have the highest score).

The test split is used to score loss/accuracy on a set of data which the model has never seen. This the number we will share publicly for others to gauge the efficiency of our model.

We might also use more complicated scoring for the test split. Instead of looking at a single loss/accuracy number, we might want to score accuracy per class (for classification problems like the one we're dealing with). This might uncover issues where the loss is low, but there's a particular class the model is struggling to classify (like the Phelps headline we discovered yesterday).

# Notes

Model path: runs/day22/final
Test n, test accuracy: n 7600, test accuracy 0.944078947368421
Val accuracy this checkpoint was picked on: 0.9462
Gap: 0.0034

Confusion matrix:
[[1807   11   44   38]
 [  13 1871   10    6]
 [  33    4 1727  136]
 [  29    8   93 1770]]

Per-class accuracy:
World 0.9510526315789474 n 1900
Sports 0.9847368421052631 n 1900
Business 0.9089473684210526 n 1900
Sci/Tech 0.9315789473684211 n 1900

Hypothesis: Business and Sci/Tech often get confused with each other

Three mistakes:
1.  true: Sci/Tech
    pred: Business
    text: IBM to hire even more new workers By the end of the year, the computing giant plans to have its biggest headcount since 1991.
    why: IBM is a big technology company, so model leaned towards the business aspect of the headline rather than the technology side. A human can make the same mistake with this text.
2.  true: Business
    pred: Sci/Tech
    text: Intel to delay product aimed for high-definition TVs SAN FRANCISCO -- In the latest of a series of product delays, Intel Corp. has postponed the launch of a video display chip it had previously planned to introduce by year end, putting off a showdown with Texas Instruments Inc. in the fast-growing market for high-definition television displays.
    why: It seems like a lot of text does indeed have overlap between technology firms and technology centres like SF, but the article is about business. The model is having a hard time with these distinctions when it deals with such companies.
3.  true: World
    pred: Business
    text: A Founding Father? Give the guy some credit. Tung Chee-hwa, Hong Kong #39;s embattled Chief Executive, gets precious little of it from his people these daysand heaps of
    why: Probably because of "chief executive" reference.

Did I train today: no
Did I change any hyperparameter after seeing test: no

What I still find shaky, in full sentences:
- I didn't know what you meant when you referenced "turns off dropout". I googled it and now understand it, but this is an example where you need to be verbose.
- In today's approach, we tokenized the test split to pytorch tensors in one go before we passed them to the model for inference. But yesterday, we used the train/validation splits and tokenized them in-place by applying the tokenize function to each element. Why did we prefer the former approach yesterday and the latter today? Why didn't we have `padding=True` for tokenization yesterday but have it today?