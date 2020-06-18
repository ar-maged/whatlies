from bpemb import BPEmb

from whatlies import Embedding, EmbeddingSet


class BytePairLang:
    """
    This object is used to lazily fetch [Embedding][whatlies.embedding.Embedding]s or
    [EmbeddingSet][whatlies.embeddingset.EmbeddingSet]s from a Byte-Pair Encoding
    backend. This object is meant for retreival, not plotting.

    This language represents token-free pre-trained subword embeddings. Originally created by
    Benjamin Heinzerling and Michael Strube.

    Important:
        These vectors will auto-download. You can also specify "multi" to download
        embeddings for multiple language at the same time. A full list of available
        languages can be found [here](https://nlp.h-its.org/bpemb). The article that
        belongs to this work can be found [here](http://www.lrec-conf.org/proceedings/lrec2018/pdf/1049.pdf).

    Arguments:
        lang: name of the model to load
        vs: the vocabulary size that the byte pair embedding needs to use

    **Usage**:

    ```python
    > from whatlies.language import BytePairLang
    > lang = BytePairLang(lang="en")
    > lang['python']
    > lang = BytePairLang(lang="multi")
    > lang[['hund', 'hond', 'dog']]
    ```
    """
    def __init__(self, lang: str, vs: int = 10000, dim: int = 100):
        self.module = BPEmb(lang=lang, vs=vs, dim=dim)

    def __getitem__(self, item):
        """
        Retreive a single embedding or a set of embeddings. If an embedding contains multiple
        sub-tokens then we'll average them before retreival.

        Arguments:
            item: single string or list of strings

        **Usage**
        ```python
        > lang = BytePairLang(lang="en")
        > lang['python']
        > lang[['python', 'snake']]
        > lang[['nobody expects', 'the spanish inquisition']]
        ```
        """
        if isinstance(item, str):
            return Embedding(item, self.module.embed(item).mean(axis=0))
        if isinstance(item, list):
            return EmbeddingSet(*[self[i] for i in item])
        raise ValueError(f"Item must be list of string got {item}.")
