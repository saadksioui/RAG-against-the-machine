import bm25s

# 1. Tokenizing a corpus (list of documents)
corpus = [
    "a cat is a feline and likes to purr",
    "a dog is the human's best friend and loves to play"
]
corpus_tokens = bm25s.tokenize(corpus)
print(corpus_tokens)