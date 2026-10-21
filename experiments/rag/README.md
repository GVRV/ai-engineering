# Retrieval Scoring Project

## Introduction

The idea was to create a retrieve then read loop over SQuAD validation split. The dataset contains a Wikipedia paragraph and a question that contains an answer from within that paragraph. The purpose of the project was to first embed all of the unique paragraphs, and then find the highest cosine similarity paragraph (aka Hit@1) for all questions. Augmenting the question with this Hit@1 paragraph, we used an open source model to generate the answer only if the model is able to find the answer from within the source text or else reply with `NOT_IN_SOURCE`.

## Dataset

We used the validation split of the `rajpurkar/squad` dataset. It contains 10,570 samples with unique questions, across 2067 unique Wikipedia paragraph contexts. We used this existing dataset instead of generating our own knowledge corpus like with the eariler Photosynthesis-based retrieval project.

## Retrieval Number

We used the `all-MiniLM-L6-v2` to embed the unique contexts and questions in 384 dimensional vector space. We used the cosine similarity dot product to get the similarity.

Our Hit@1 is 0.625 and Hit@5 is 0.864.

Hit@1 means that the context ranking highest using cosine similarity was the same context that is available in the dataset as the gold paragraph for that question. So, for 62.5% of the questions, we were able to fetch the context paragraph that is exactly the same as the paragraph used in the dataset.

Hit@5 means that the gold paragraph for the question is available in the top 5 ranking contexts using cosine similarity. So, for 86.4% of the questions, we are able to find the dataset context paragraph for that question in the top 5 paragraphs ranked via our dot product cosine similarity.

We did not try to bump up the numbers by judging that a hit means that the answer string appeared in the paragraph somewhere. For example, the question "What venue did Super Bowl 50 take place in?" had a Hit@1 gold-id miss but the answer "Levi's Stadium" appears in the context retrieved. We did not bump up our numbers artificially.

## Generation Number

We then used 20 questions (10 with a gold-id hit and 10 with a gold-id miss) found randomly using a numpy randomness seed of `42`. For each of these questions, we used the `openai/gpt-oss-20b` model (temperature `0`) to generate the answer to the questions from within the source paragraph sent to it or reply with `NOT_IN_SOURCE`. The augmented paragraph was capped at the first 1500 characters and we scored whether the model generated a grounded reply against this cropped text. The results are as follows:

- gold hit, grounded, useful: 9
- gold hit, not grounded, useful: 1
- gold hit, not useful: 0
- gold miss, refused or answered from paste: 6
- gold miss, answered with fact not in paste: 4

### The Invention

For the question `A complete loss of rainforest cover may be caused by what type of emissions?`, even though the answer didn't appear in the augmented context, the model used its world knowledge to generate the slightly reply: `Carbon emissions.` (gold reply was "Greenhouse emissions").

The context passed in was:
```
Environmentalists are concerned about loss of biodiversity that will result from destruction of the forest, and also about the release of the carbon contained within the vegetation, which could accelerate global warming. Amazonian evergreen forests account for about 10% of the world's terrestrial primary productivity and 10% of the carbon stores in ecosystems—of the order of 1.1 × 1011 metric tonnes of carbon. Amazonian forests are estimated to have accumulated 0.62 ± 0.37 tons of carbon per hectare per year between 1975 and 1996.
```

### The False Refusal

For the question `How much of Paris' population was killed by the plague?`, the model replied `NOT_IN_SOURCE` but the context passed in contained at least an estimate which could've been used for a more useful reply like `At least 40,000.`.

The context passed in was:
```
In 1466, perhaps 40,000 people died of the plague in Paris. During the 16th and 17th centuries, the plague was present in Paris around 30 per cent of the time. The Black Death ravaged Europe for three years before it continued on into Russia, where the disease was present somewhere in the country 25 times between 1350 to 1490. Plague epidemics ravaged London in 1563, 1593, 1603, 1625, 1636, and 1665, reducing its population by 10 to 30% during those years. Over 10% of Amsterdam's population died in 1623–25, and again in 1635–36, 1655, and 1664. Plague occurred in Venice 22 times between 1361 and 1528. The plague of 1576–77 killed 50,000 in Venice, almost a third of the population. Late outbreaks in central Europe included the Italian Plague of 1629–1631, which is associated with troop movements during the Thirty Years' War, and the Great Plague of Vienna in 1679. Over 60% of Norway's population died in 1348–50. The last plague outbreak ravaged Oslo in 1654.
```

## How to Run

Use the script `ask.py` to ask a question, find the highest ranking context that might contain the answer and send a request to Groq to generate an answer.

Note: It requires `GROQ_API_KEY` in an `.env` file to operate and has external Python dependencies like `sentence_transformers` and `datasets`.

```
> python ask.py "A complete loss of rainforest cover may be caused by what type of emissions?"
Answer: Carbon emissions.
Context: Environmentalists are concerned about loss of biodiversity that will result from destruction of the forest, and also about the release of the carbon contained within the vegetation, which could accelerate global warming. Amazonian evergreen forests account for about 10% of the world's terrestrial primary productivity and 10% of the carbon stores in ecosystems—of the order of 1.1 × 1011 metric tonnes of carbon. Amazonian forests are estimated to have accumulated 0.62 ± 0.37 tons of carbon per hectare per year between 1975 and 1996.

> python ask.py "What day was the Super Bowl played on?"
Answer: NOT_IN_SOURCE
Context: The game's media day, which was typically held on the Tuesday afternoon prior to the game, was moved to the Monday evening and re-branded as Super Bowl Opening Night. The event was held on February 1, 2016 at SAP Center in San Jose. Alongside the traditional media availabilities, the event featured an opening ceremony with player introductions on a replica of the Golden Gate Bridge.
```

### What the Project is Not

- We have not used a vector database. The corpus of 2067 paragraphs is small enough to embed and matrix multiply on a local machine. A larger corpus would need an index for ranking.
- We judged whether the response was grounded and useful manually by going through the augmented context and comparing it with the response generated by the model. We do not have an eval harness, everything has been hand scored.
- We are not claiming that the Hit@1 score of 0.625 is a leaderboard number. This just happened to be our observation for this particular dataset, using a particular split from it and calculating the number over dot products using a particular embedding model.
- The wrong page still generated the correct answer from the model because the model used its real world knowledge. This is the failure mode we need to focus on and correct.
- The hand-made photosynthesis files were crafted for an even simpler RAG experiment. You can read about it in `experiments/retrieval/README.md`. Both experiments are separate, there happen to be some similarities like the embedding model used, but the numbers are not to be confused to be same across both experiments.