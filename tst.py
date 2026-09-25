import bm25s

# 1. Define your corpus
corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence and machine learning are changing the world.",
    "Python is a popular programming language for data science.",
    "Retrieval-Augmented Generation helps LLMs query custom documents."
]

# 2. Tokenize the corpus (simple whitespace/punctuation splitting)
corpus_tokens = bm25s.tokenize(corpus, stopwords="english")

# 3. Initialize and build the BM25 index
retriever = bm25s.BM25()
retriever.index(corpus_tokens)

# 4. Query the index
query = "Python data science programming"
query_tokens = bm25s.tokenize(query)

# Retrieve the top 2 most relevant documents
documents, scores = retriever.retrieve(query_tokens, corpus=corpus, k=2)

for doc, score in zip(documents[0], scores[0]):
    print(f"Score: {score:.4f} | Document: {doc}")
