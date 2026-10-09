Overlap asks: what fraction of the question’s words appear in this file?

Glue words like “the” and “only” appear in almost every school note, so they drown the rare words.

An embedding turns a whole sentence into a list of numbers, trained so that sentences about the same thing land near each other.

Cosine similarity asks how aligned those two lists are. It can match “green pigment” to “chlorophyll” without sharing the word.

It still does not answer the question. It only picks a file to paste.

# Block A

Stopwords fixed the problem! Kinda, it has three files all at 0.2 relevancy and we're only getting a chlorophyll file because of the sorted results on the file prefixes (so 01_photosynthesis_process.txt ranks better than 08_plant_reproduction.txt even though they have the same score). We lucked out!

Q: Plants make food using chrolophyll and photosynthesis only in the leaves, and only in the day. What should I correct, for class 7?

Result:
[(0.2, '01_photosynthesis_process.txt'), (0.2, '04_photosynthesis_advanced.txt'), (0.2, '08_plant_reproduction.txt')]
{'file': '01_photosynthesis_process.txt', 'score': 0.2, 'reply': 'I’m not sure.'}

It still picked the correct file/chunk for the chlorophyll question:

Q: Is chlorophyll responsible for a plant's green colour?
Result: {'file': '02_chlorophyll.txt', 'score': 0.6666666666666666, 'reply': 'Yes. The source says that chlorophyll is the pigment that gives the plant its green colour.'}

# Block D

Q: Are there 2 main stages of photosynthesis?
Correct file: 03_photosynthesis_reactions.txt
Embeddings top file: 03_photosynthesis_reactions.txt
Fast Embed top file: 03_photosynthesis_reactions.txt
Overlap with stopwords top file: 03_photosynthesis_reactions.txt
Better ranker: Both

Q: Is the potato plant the best at photosynthesis?
Correct file: None (but should be photosynthesis related)
Embeddings top file: 02_chlorophyll.txt
Fast Embed top file: 04_photosynthesis_advanced.txt
Overlap with stopwords top file: 01_photosynthesis_process.txt
Better ranker: Both

Q: Is the stomach the most important organ for digestion?
Correct file: 07_human_digestion.txt
Embeddings top file: 07_human_digestion.txt
Fast Embed top file: 07_human_digestion.txt
Overlap with stopwords top file: 07_human_digestion.txt
Better ranker: Both

Q: Plants make food using chrolophyll and photosynthesis only in the leaves, and only in the day. What should I correct, for class 7?
Correct file: 02_chlorophyll.txt
Embeddings top file: 02_chlorophyll.txt
Fast Embed top file: 02_chlorophyll.txt
Overlap with stopwords top file: 01_photosynthesis_process.txt
Better ranker: Embeddings (NOTE: This is amazing considering we have mis-spelled `chlorophyll` in the question but still embedding models are able to determine similarity!)

Q: Are the chemicals involved in photosynthesis carbon dioxide and oxygen?
Correct file: 01_photosynthesis_process.txt
Embeddings top file: 01_photosynthesis_process.txt
Fast Embed top file: 01_photosynthesis_process.txt
Overlap with stopwords top file: 01_photosynthesis_process.txt
Better ranker: Both

Q: Is photosynthesis less effective during winter months?
Correct file: None
Embeddings top file: 03_photosynthesis_reactions.txt
Fast Embed top file: 04_photosynthesis_advanced.txt
Overlap with stopwords top file: 01_photosynthesis_process.txt
Better ranker: None

Q: Is chlorophyll responsible for a plant's green colour?
Correct file: 02_chlorophyll.txt
Embeddings top file: 02_chlorophyll.txt
Fast Embed top file: 02_chlorophyll.txt
Overlap with stopwords top file: 02_chlorophyll.txt
Better ranker: Both

Q: Are fruits where plant stores the energy produced by photosynthesis?
Correct file: 01_photosynthesis_process.txt
Embeddings top file: 01_photosynthesis_process.txt
Fast Embed top file: 01_photosynthesis_process.txt
Overlap with stopwords top file: 01_photosynthesis_process.txt
Better ranker: Both

Q: Does most life on Earth depend on photosynthesis?
Correct file: 01_photosynthesis_process.txt
Embeddings top file: 03_photosynthesis_reactions.txt
Fast Embed top file: 04_photosynthesis_advanced.txt
Overlap with stopwords top file: 03_photosynthesis_reactions.txt
Better ranker: None

Q: Are school grammar classes voluntary for students to attend?
Correct file: 12_school_timetable.txt
Embeddings top file: 12_school_timetable.txt
Fast Embed top file: 12_school_timetable.txt
Overlap with stopwords top file: 12_school_timetable.txt
Better ranker: Both

Q: Did Gandhi's plant based diet give him the energy to fight for independence?
Correct file: 09_gandhi.txt
Embeddings top file: 09_gandhi.txt
Fast Embed top file: 09_gandhi.txt
Overlap with stopwords top file: 09_gandhi.txt
Better ranker: Both

Q: Plants are green because that colour helps with plant reproduction?
Correct file: 08_plant_reproduction.txt
Embeddings top file: 02_chlorophyll.txt
Fast Embed top file: 02_chlorophyll.txt
Overlap with stopwords top file: 08_plant_reproduction.txt
Better ranker: Overlap search with stopwords

Q: The water cycle is essential for all plants to survive. True or False?
Correct file: 06_water_cycle.txt
Embeddings top file: 06_water_cycle.txt
Fast Embed top file: 06_water_cycle.txt
Overlap with stopwords top file: 04_photosynthesis_advanced.txt
Better ranker: Embeddings

Q: Who was the first human to land on the moon?
Correct file: None
Embeddings top file: 09_gandhi.txt
Fast Embed top file: 09_gandhi.txt
Overlap with stopwords top file: 05_respiration.txt
Better ranker: None (Embeddings at least it found the chunk about some person, but that doesn't make it any better, it's equally wrong when compared to respiration)

# Block E

Class-7 top file, raw overlap: 10_verbs.txt
Class-7 top file, stopwords removed: 01_photosynthesis_process.txt
Class-7 top file, embeddings: 02_chlorophyll.txt
NOTE: Do not compare scores between different embedding models as it doesn't indicate objective more similarity (eg: 0.5 in both sentence-transformers and 0.5 in fastembed doesn't mean equally similar, they each have their own similarity scales)

Cosine, in a sentence: the process of getting the measure of a similarity of two vectors by measuring the angle between them (1 = identical, 0 = unrelated). When doing text embeddings, we can convert sentences/chunks/files to vectors using an embedding model and then get the relevant chunk(s) by calculating the cosine with the vector of the query.

A question embeddings got right and overlap got wrong: The water cycle is essential for all plants to survive. True or False?
A question both got wrong: Does most life on Earth depend on photosynthesis? (Kinda, because there is no "right" file for this one, but you can see embeddings being confused about it as well)

I did not call the chat model to paper over a bad rank.

What I still find shaky, in full sentences:
- Is there an widely used list of stopwords (otherwise figuring out a list for your own corpus involves a lot of experimentation/trial and error)? Especially if stopwords cause your results to be worse over time.
- These embedding models seem like black boxes themselves and are constantly being updated/changing, so you're not quite sure which one to use and how it will behave? Is the only way to overcome this by having good tests and testing each new update of all widely available embedding models?

