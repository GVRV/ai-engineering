AG News is a public dataset of news blurbs, already split into train and test by the people who published it.

I am not allowed to look at the test split while I decide how to train. Test is for the final number.

I will cut a small validation slice out of train, so I can check progress without touching test.

Today I only inspect. I do not tokenize, I do not load DistilBERT, I do not call Trainer.

# Block A

- What a row is in our dataset. A row consists of 3 columns: A label (int64) to signify the class, a title (string) and a description (string). The labels are World (0), Sports (1), Business (2), Sci/Tech (3). An example of a row is below:

```
Label: 2
Title: Wall St. Bears Claw Back Into the Black (Reuters)
Description: Reuters - Short-sellers, Wall Street's dwindling\band of ultra-cynics, are seeing green again.
```

- What a split is: whenever we're training a model by minimising a loss on a certain dataset, we split the dataset into different chunks which are used for different purposes and these are called splits. The most case is to use 3 splits: Train, Validation and Test splits (roughly 80%, 10% and 10%). You use the training set to train the model which might result in some overfitting. After completing an epoch (1 run of the whole of test split), we do a single forward pass on the entire validation split to understand how the model behaves with data it sees rarely and to tune hyperparameters (if needed). Once you have a good model trained, you test it once on the test split to get a loss on the completely out of sample data which can be judged as how the model will behave in the real world where it might encouter data it has never seen during its training, this is the public loss you publish.

- Why we still cut validation: Answered above, we cut some of the training data so that there is some data we can test against (for tuning hyperparameters or getting an estimate of the real world loss) while ensuring the model isn't getting overfitted to this data.

- What we are not doing: No embedding for retreival, no writing datasets/articles, no fine-tuning a chat model to be a teacher. Just want to do classification on news articles (as training a model for this task is similar to training a model for language training but uses less computing power so that we can do it locally)

# Block C

Blurbs pasted below:

```
Business
Wall St. Bears Claw Back Into the Black (Reuters)
Reuters - Short-sellers, Wall Street's dwindling\band of ultra-cynics, are seeing green again.

Sci/Tech
'Madden,' 'ESPN' Football Score in Different Ways (Reuters)
Reuters - Was absenteeism a little high\on Tuesday among the guys at the office? EA Sports would like\to think it was because "Madden NFL 2005" came out that day,\and some fans of the football simulation are rabid enough to\take a sick day

Sports
Phelps, Thorpe Advance in 200 Freestyle (AP)
AP - Michael Phelps took care of qualifying for the Olympic 200-meter freestyle semifinals Sunday, and then found out he had been added to the American team for the evening's 400 freestyle relay final. Phelps' rivals Ian Thorpe and Pieter v

World
Venezuelans Vote Early in Referendum on Chavez Rule (Reuters)
Reuters - Venezuelans turned out early\and in large numbers on Sunday to vote in a historic referendum\that will either remove left-wing President Hugo Chavez from\office or give him a new mandate to govern for the next two\years.
```

# Block E

Dataset id I loaded: 'sh0416/ag_news' from HuggingFace
Train rows, test rows: 120000, 7600
Columns: ['label', 'title', 'description']
Label map: World (0), Sports (1), Business (2), Sci/Tech (3)

Train class counts: {3: 30000, 0: 30000, 2: 30000, 1: 30000}
Test class counts: {3: 1900, 0: 1900, 2: 1900, 1: 1900}

One blurb per class, first 200 characters: Blurbs pasted above.

Validation cut: ratio, seed, three lengths: 0.1 ratio, 42 seed, new lengths:
train: 108000
validation: 12000
test: 7600

HF_HOME path, and did the second run redownload: /Users/gaurav/.cache/huggingface No

What I still find shaky, in full sentences:
- Nothing so far, I understand today was a practical assignment in downloading a dataset from HuggingFace (a widely/industry accepted repository for datasets) and exploring its structure.