"""Duplicate retrieval: BM25 over the time-aware candidate pools, and the retrieval metrics.

Used by notebooks/06_baseline_retrieval.ipynb, 07 (sentence vectors), 09 and later src/pipeline.py,
so every method is ranked and scored in exactly the same way.
"""
import re

import numpy as np
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")    # words of 2+ letters or digits (the same rule as the TF-IDF in 05)
_stemmer = PorterStemmer()
_stems = {}                                # each distinct word is stemmed only once


def bm25_tokens(text):
    """Lower-cased words without stop words, cut to their stem ("licensing" -> "licens")."""
    tokens = []
    for word in TOKEN_RE.findall(text.lower()):
        if word in ENGLISH_STOP_WORDS:
            continue
        if word not in _stems:
            _stems[word] = _stemmer.stem(word)
        tokens.append(_stems[word])
    return tokens


class BM25Index:
    """Stem counts of the published petitions, one row per petition (in CandidatePools.ids order).

    The BM25 statistics (number of petitions, how many contain each stem, average length) are computed
    inside each query's pool, so a petition outside the pool never changes the scores.
    """

    def __init__(self, texts):
        self.vectorizer = CountVectorizer(analyzer=bm25_tokens)
        self.counts = self.vectorizer.fit_transform(texts).tocsc()    # columns = stems; CSC is fast to slice by column
        self.doc_len = np.asarray(self.counts.sum(axis=1)).ravel()   # number of stems in each petition

    def query_columns(self, text):
        """Columns of the distinct query stems that occur in the published petitions."""
        vocabulary = self.vectorizer.vocabulary_
        return sorted({vocabulary[t] for t in bm25_tokens(text) if t in vocabulary})

    def scores(self, text, mask, k1=1.5, b=0.75):
        """BM25 score of every petition in the pool (mask: True for pool members, in CandidatePools.ids order)."""
        columns = self.query_columns(text)
        n = int(mask.sum())
        if n == 0 or not columns:
            return np.zeros(n)
        tf = self.counts[:, columns].toarray()[mask]        # how often each query stem occurs in each pool petition
        doc_len = self.doc_len[mask]
        df = (tf > 0).sum(axis=0)                           # pool petitions that contain each query stem
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5))       # rare stems count more; the "1 +" keeps it positive
        length_norm = k1 * (1 - b + b * doc_len / doc_len.mean())
        return (idf * tf * (k1 + 1) / (tf + length_norm[:, None])).sum(axis=1)


def hybrid_scores(bm25, cosine, alpha):
    """alpha x BM25 + (1 - alpha) x cosine, with BM25 divided by the pool's best BM25 score so both are about 0-1."""
    top = bm25.max() if len(bm25) else 0
    scaled = bm25 / top if top > 0 else bm25
    return alpha * scaled + (1 - alpha) * cosine


def gold_rank(scores, pool_ids, gold_ids):
    """Rank (1 = top) of the best-scored correct petition, or inf if no correct petition is in the pool.

    Ties count against the method: every other petition with the same score is placed before it.
    """
    is_gold = np.isin(pool_ids, list(gold_ids))
    if not is_gold.any():
        return np.inf
    best = scores[is_gold].max()
    return 1 + int((scores[~is_gold] >= best).sum())


def retrieval_metrics(ranks, ks=(1, 5, 10)):
    """Recall@k (share of queries with a correct petition in the top k) and MRR (mean of 1 / rank; 1 / inf = 0)."""
    ranks = np.asarray(ranks, dtype=float)
    metrics = {f"recall@{k}": float((ranks <= k).mean()) for k in ks}
    metrics["mrr"] = float((1 / ranks).mean())
    return metrics
