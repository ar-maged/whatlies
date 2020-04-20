from typing import Union

import spacy
from spacy.language import Language
import numpy as np
from typing import Union, List
from sklearn.metrics import pairwise_distances
from sense2vec import Sense2Vec, Sense2VecComponent

from whatlies.embedding import Embedding
from whatlies.embeddingset import EmbeddingSet
from whatlies.language.common import _selected_idx_spacy

class SpacyLanguage:
    """
    This object is used to lazily fetch [Embedding][whatlies.embedding.Embedding]s or
    [EmbeddingSet][whatlies.embeddingset.EmbeddingSet]s from a spaCy language
    backend. This object is meant for retreival, not plotting.

    Arguments:
        model: name of the model to load, be sure that it's downloaded beforehand

    **Usage**:

    ```python
    > lang = SpacyLanguage("en_core_web_md")
    > lang['python']
    > lang[['python', 'snake', 'dog']]

    > lang = SpacyLanguage("en_trf_robertabase_lg")
    > lang['programming in [python]']
    ```
    """

    def __init__(self, model: Union[str, Language]):
        if isinstance(model, str):
            self.nlp = spacy.load(model)
        elif isinstance(model, Language):
            self.nlp = model
        else:
            raise ValueError("Language must be started with `str` or spaCy-langauge object.")

    @staticmethod
    def _input_str_legal(string):
        if sum(1 for c in string if c == "[") > 1:
            raise ValueError("only one opener `[` allowed ")
        if sum(1 for c in string if c == "]") > 1:
            raise ValueError("only one opener `]` allowed ")

    def __getitem__(self, query: Union[str, List[str]]):
        """
        Retreive a single embedding or a set of embeddings. Depending on the spaCy model
        the strings can support multiple tokens of text but they can also use the Bert DSL.
        See the Language Options documentation: https://rasahq.github.io/whatlies/tutorial/languages/#bert-style.

        Arguments:
            query: single string or list of strings

        **Usage**
        ```python
        > lang = SpacyLanguage("en_core_web_md")
        > lang['python']
        > lang[['python'], ['snake']]
        > lang[['nobody expects'], ['the spanish inquisition']]
        ```
        """
        if isinstance(query, str):
            self._input_str_legal(query)
            start, end = _selected_idx_spacy(query)
            clean_string = query.replace("[", "").replace("]", "")
            vec = self.nlp(clean_string)[start:end].vector
            return Embedding(query, vec)
        return EmbeddingSet(*[self[tok] for tok in query])

    def embset_similar(self, emb: Union[str, Embedding], n: int = 10, prob_limit=-15, lower=True, metric='cosine'):
        """
        Retreive an [EmbeddingSet][whatlies.embeddingset.EmbeddingSet] that are the most simmilar to the passed query.

        Arguments:
            emb: query to use
            n: the number of items you'd like to see returned
            prob_limit: likelihood limit that sets the subset of words to search
            metric: metric to use to calculate distance, must be scipy or sklearn compatible
            lower: only fetch lower case tokens

        Returns:
            An [EmbeddingSet][whatlies.embeddingset.EmbeddingSet] containing the similar embeddings.
        """
        embs = [w[0] for w in self.score_similar(emb, n, prob_limit, lower, metric)]
        return EmbeddingSet({w.name: w for w in embs})

    def score_similar(self, emb: Union[str, Embedding], n: int = 10, prob_limit=-15, lower=True, metric='cosine'):
        """
        Retreive a list of (Embedding, score) tuples that are the most simmilar to the passed query.

        Arguments:
            emb: query to use
            n: the number of items you'd like to see returned
            prob_limit: likelihood limit that sets the subset of words to search, to ignore set to `None`
            metric: metric to use to calculate distance, must be scipy or sklearn compatible
            lower: only fetch lower case tokens

        Returns:
            An list of ([Embedding][whatlies.embedding.Embedding], score) tuples.
        """
        if isinstance(emb, str):
            emb = self[emb]

        vec = emb.vector
        queries = [w for w in self.nlp.vocab]
        if prob_limit is not None:
            queries = [w for w in queries if w.prob >= prob_limit]
        if lower:
            queries = [w for w in queries if w.is_lower]
        if len(queries) == 0:
            raise ValueError(f"Language model has no tokens for this setting. Consider raising prob_limit={prob_limit}")

        vector_matrix = np.array([w.vector for w in queries])
        distances = pairwise_distances(vector_matrix, vec.reshape(1, -1), metric=metric)
        by_similarity = sorted(zip(queries, distances), key=lambda z: z[1])

        return [(self[q.text], float(d)) for q, d in by_similarity[:n]]
