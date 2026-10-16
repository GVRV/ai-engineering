# Retreival-Augmented Generation (RAG) Project

## Introduction

RAG aims to solve the problem where a user wants to use a model to query a corpus of knowledge that isn't intrinsically available to the model. Breaking down RAG, we are left with the following simple steps in this process:

- Retreival: Look at the user's query (or requirements) and find the relevant chunks of knowledge from the corpus available with the user.
- Augmented: Take the retreived chunks and augment the user's query to the model with this additional information.
- Generation: Return the model's response given the input contained the user's query as well the necessary knowledge expected to answer it.

### Practical Example

We wanted to understand how RAG would help a teacher answer a question that they ask a model using the knowledge corpus available to them. A 7th standard math teacher would have a vastly different knowledge corpus than a 12th grade math teacher, even though the subject is the same, some advanced concepts are introduced later and cannot be used in an answer that is  targeted towards the objective understanding of 7th grade students.

Please note: For this project, I have manually written all of the corpus in `retreival/corpus` directory focusing mostly on Photosynthesis but they are not real school notes. The idea is that this dummy knowledge corpus can help us understand RAG and its approaches but can be easily mentally substituted for a bigger, more complex knowledge corpus for real-world usage in the future.

### Approaches

The most obvious method for this would be the "brute-force" approach where we augment the user's query with the entirety of the knowledge corpus available, but this approach has some limitations:
- The knowledge corpus might be bigger than the model's context length, so parts of it might be excluded (especially bad when these excluded parts contain the information needed for the answer)
- Crafting big queries with unnecessary augmented information might lead of inference issues like subpar response/latency times or hallucinations.

Therefore, we want to use a more calculated approach to retreiving the right chunks of information from knowledge corpus so that augmenting the user query with this information results in a effective answer by the model.

#### Overlap Search (Naive Approach)

We started with a very naive approach where to find the most relevant file from our knowledge corpus, we just looked at how words from the question are found in the file. This naive approach led to bugs because a query may contain a lot of stop words (like `is`, `a`, `the`, etc) which a lot of files also contain.

We made an incremental improvement in our approach by discarding a list of stop words from both the file content and the query text, which resulted in better matches but still subpar performance because high overlap doesn't necessarily mean high correlation.

#### Cosine similarity between embeddings (Better Approach)

A better approach to get semantic similarity between the query and a chunk of the knowledge corpus is to first use a lightweight model to embed both the query and the chunks of knowledge in n-dimensional vector space. Then the cosine between both vectors is how similar they are: ~1 means they're almost the same, 0 means they're not correlated at all. Ranking chunks of information using their vector cosine scores, gives us the most relevant chunks of information from the corpus of knowldge which can then use to augment our query.

For converting the text into n-dimensional vectors, we can use specialised small and performant embedding models like `all-MiniLM-L6-V2` or `BAAI/bge-small-en-v1.5`.

A follow-on issue with this approach is that we would ideally want to break up our knowledge corpus into manageable chunks which contain the necessary information (and hence result in higher cosine similarity) for our augmentation. Using too big of a chunk (like whole files) might mean that the cosine similarity is low because a huge file might contain a lot of irrelevant information apart from just a paragraph that might be relevant. Using too small chunks (like individual sentences or fixed character length or fixed word length texts) might mean that the relevant information is contained over more than one chunk in a chaotic manner; maybe even no single chunk contains all of the relevant information for effective augmentation.

Usually, a semantic breakdown of the knowledge corpus into well defined sections with a given meaning/context works best but the best case for your knowledge corpus and your target queries needs to be experimented with. This process of experimentation results in you coming up with "evals" that you can test (eg: this particular question on this particular knowledge corpus should get an answer, or that particular question on this particular knowledge corpus should result in a "I don't know" answer, etc)

### Caveats

This is not a production RAG setup with a vector database. This was a toy project to understand what RAG is and understand the theoretical foundations of how various approachs to RAG effectively over time emerged.

