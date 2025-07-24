"""Tools to help with submitting jobs to the cluster."""

import ssl

import nltk
import numpy as np
from nltk.corpus import wordnet


def get_bigram(seed):
    """Return a random bigram of the form <adjective>_<noun>."""
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download("wordnet")  # Download WordNet data
    adjectives = [synset.lemmas()[0].name() for synset in wordnet.all_synsets(wordnet.ADJ)]
    nouns = [synset.lemmas()[0].name() for synset in wordnet.all_synsets(wordnet.NOUN)]

    adjectives = [
        adj for adj in adjectives if "-" not in adj and "_" not in adj and adj[0].islower()
    ]
    nouns = [noun for noun in nouns if "-" not in noun and "_" not in noun and noun[0].islower()]

    rng = np.random.default_rng(seed)
    i_adj, i_noun = rng.choice(len(adjectives)), rng.choice(len(nouns))

    # Return the bigram with the words capitalized

    return adjectives[i_adj].capitalize() + nouns[i_noun].capitalize()
