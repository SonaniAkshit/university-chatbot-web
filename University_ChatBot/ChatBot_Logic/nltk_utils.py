import re
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

# Ensure punkt tokenizer is available (safe to call on startup)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# simple regex to keep words and numbers; split on spaces/punctuation fallback
_TOKEN_RE = re.compile(r"\w+|\S")

def tokenize(sentence):
    """
    Split sentence into tokens. Uses nltk.word_tokenize but falls back to regex if needed.
    """
    try:
        tokens = nltk.word_tokenize(sentence)
    except Exception:
        tokens = _TOKEN_RE.findall(sentence)
    return [t for t in tokens]

def stem(word):
    """
    Return stemmed lowercase form of word.
    """
    return stemmer.stem(word.lower())

def normalize_tokens(tokens):
    """
    Normalize tokens: lowercase and remove purely punctuation tokens.
    """
    norm = []
    for t in tokens:
        t = t.strip()
        if t == "":
            continue
        # drop tokens that are only punctuation
        if all(ch in ".,!?;:-_()[]{}\"'`" for ch in t):
            continue
        norm.append(t.lower())
    return norm

def bag_of_words(tokenized_sentence, words):
    """
    Return bag-of-words numpy array (float32).
    Stems tokens and matches against precomputed `words` (which should be stemmed).
    """
    sentence_words = [stem(w) for w in normalize_tokens(tokenized_sentence)]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1.0
    return bag
