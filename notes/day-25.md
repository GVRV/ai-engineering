SQuAD is a public set of questions, each paired with a Wikipedia paragraph and a short answer span that appears in that paragraph.

I will treat the unique paragraphs as the corpus. I will not write any of them.

A hit is: the paragraph I ranked first contains the answer string.

A hit is not a generated answer. Generation is tomorrow. Today the ranker either finds the page or it does not.

# Block E - Notes

Dataset id and split: rajpurkar/squad dataset, validation split
Questions, unique contexts: 10570 questions, 2067 contexts
Embedder: all-MiniLM-L6-v2

Hit@1: 0.6251
Hit@5: 0.8640

Five misses, one line each:
Q:  What day was the game played on?
gold answer:  February 7, 2016
ranked:  The game's media day, which was typically held on the Tuesday afternoon prior to the game, was moved to the Monday evening and re-branded as Super Bowl Opening Night. The event was held on February 1, 2016 at SAP Center in San Jose. Alongsi
---
Q:  What day was the Super Bowl played on?
gold answer:  February 7, 2016
ranked:  The game's media day, which was typically held on the Tuesday afternoon prior to the game, was moved to the Monday evening and re-branded as Super Bowl Opening Night. The event was held on February 1, 2016 at SAP Center in San Jose. Alongsi
---
Q:  What venue did Super Bowl 50 take place in?
gold answer:  Levi's Stadium
ranked:  On May 21, 2013, NFL owners at their spring meetings in Boston voted and awarded the game to Levi's Stadium. The $1.2 billion stadium opened in 2014. It is the first Super Bowl held in the San Francisco Bay Area since Super Bowl XIX in 1985
---
Q:  If Roman numerals were used, what would Super Bowl 50 have been called?
gold answer:  Super Bowl L
ranked:  On June 4, 2014, the NFL announced that the practice of branding Super Bowl games with Roman numerals, a practice established at Super Bowl V, would be temporarily suspended, and that the game would be named using Arabic numerals as Super B
---
Q:  What year did the Denver Broncos secure a Super Bowl title for the third time?
gold answer:  2015
ranked:  The Broncos took an early lead in Super Bowl 50 and never trailed. Newton was limited by Denver's defense, which sacked him seven times and forced him into three turnovers, including a fumble which they recovered for a touchdown. Denver lin
---

Did I generate an answer today: no
Did I write any corpus text: no

What I still find shaky, in full sentences:
- In our earlier use of the embeddings when we were dealing with a single question, we used `doc_vecs(n_contexts, n_embedding) @ question(n_embedding)` operation but today we did a `questions(n_questions, n_embedding) @ doc_vecs(n_contexts, n_embedding).T` operation? I'm guessing ultimately it doesn't matter how you calculate the cosine similarities because it is just a multiplication operation over the vectors but I found this somewhat confusing.