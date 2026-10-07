Retrieval does not make the model know the textbook. It finds a passage and puts that passage in the window, which is the same trick as yesterday’s paste.

If the search returns the wrong passage, a fluent answer is still ungrounded.

I score the reply only against the chunk that was actually pasted, not against what I wish had been pasted.

Overlap score formula, in words: Using some tokenization scheme, we get the percentage of the question tokens which are found in the chunk of text. The higher the question tokens that appear in the text, the better the chunk is considered for ranking.

# Block A

A retrieval system has 3 steps:
- Chunking: create small chunks from the reference text small enough to be passed into the context of the chat with the model.
- Ranking: Given a question, rank the chunks for relevance so that only relevant chunks are passed to the model context
- Read: Including the relevant chunks in the user message (note: prompt injection vulnerable), we ask the question to the model to get a response.

# Block C

The question "A student says plants make food only in the leaves, and only in the day. What should I correct, for class 7?" results in the following hits:

    [
        (0.3684210526315789, '10_verbs.txt'),
        (0.3684210526315789, '12_school_timetable.txt'),
        (0.2631578947368421, '04_photosynthesis_advanced.txt')
    ]

which means that the chunk from `10_verbs.txt` is included in the model call and the model correctly replies `I am not sure`.

# Block D

Q: Are the chemicals involved in photosynthesis carbon dioxide and oxygen?
Reply: {'file': '05_respiration.txt', 'score': 0.8, 'reply': 'I am not sure.'}
Grounded: 1

Q: Is photosynthesis less effective during winter months?
Reply: {'file': '01_photosynthesis_process.txt', 'score': 0.42857142857142855, 'reply': 'I’m not sure. The source only explains how plants use CO₂ and H₂O to make glucose and oxygen, but it doesn’t say anything about winter or how effective photosynthesis is during that season.'}
Grounded: 1

Q: Is chlorophyll responsible for a plant's green colour?
Reply: {'file': '02_chlorophyll.txt', 'score': 0.7777777777777778, 'reply': 'Yes. The source says that chlorophyll is the pigment that gives the plant its green colour.'}
Grounded: 1

Q: Are fruits where plant stores the energy produced by photosynthesis?
Reply: {'file': '05_respiration.txt', 'score': 0.6, 'reply': 'I’m not sure.'}
Grounded: 1

Q: Does most life on Earth depend on photosynthesis?
Reply: {'file': '03_photosynthesis_reactions.txt', 'score': 0.2857142857142857, 'reply': 'I am not sure.'}
Grounded: 1

Q: Are school grammar classes voluntary for students to attend?
Reply: {'file': '07_human_digestion.txt', 'score': 0.3333333333333333, 'reply': 'I’m not sure. The source talks only about how the body digests food, not about school grammar classes.'}
Grounded: 1

Q: Did Gandhi's plant based diet give him the energy to fight for independence?
Reply: {'file': '09_gandhi.txt', 'score': 0.35714285714285715, 'reply': 'I’m not sure. The passage does not say anything about Gandhi’s diet or whether it gave him energy to fight for independence.'}
Grounded: 1

Q: Plants are green because that colour helps with plant reproduction?
Reply: {'file': '08_plant_reproduction.txt', 'score': 0.6, 'reply': 'I am not sure.'}
Grounded: 1

Q: The water cycle is essential for all plants to survive. True or False?
Reply: {'file': '04_photosynthesis_advanced.txt', 'score': 0.6153846153846154, 'reply': 'I am not sure.'}
Grounded: 1

Q: Who was the first human to land on the moon?
Reply: {'file': '09_gandhi.txt', 'score': 0.4444444444444444, 'reply': 'I am not sure.'}
Grounded: 1


A question where the right file won: Is chlorophyll responsible for a plant's green colour?
A question where the wrong file won: Are school grammar classes voluntary for students to attend?
What the reply did with the wrong file: Say not sure.

What I still find shaky, in full sentence: I get that yesterday's approach of adding all of the available corpus to the context was a bad idea (from token efficiency point of view, but also from prompt injection vulnerability), but I don't consider the overlap score a much better approach as tokens can have synonyms (like "two" and "2", etc) and so it's very likely that with a big corpus, the wrong chunks are included in the model call.