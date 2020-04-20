import json

import numpy as np

from whatlies.language import SpacyLanguage, Sense2VecLangauge, TFHuggingfaceLang
from whatlies.transformers import Pca
from whatlies import Embedding, EmbeddingSet


def test_sense2vec_not_break():
    lang = Sense2VecLangauge("s2v_old")

    words = ["bank|NOUN", "bank|VERB", "duck|NOUN", "duck|VERB",
            "dog|NOUN", "cat|NOUN", "jump|VERB", "run|VERB",
            "chicken|NOUN", "puppy|NOUN", "kitten|NOUN", "carrot|NOUN"]
    emb = lang[words]

    v1 = emb["duck|NOUN"]
    v2 = emb["duck|VERB"]

    assert not np.array_equal(v1, v2)


def test_huggingface_not_break():
    lang = TFHuggingfaceLang("bert-base-uncased")
    lang["cat"]
