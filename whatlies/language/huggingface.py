import numpy as np
import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
from typing import Union, List

from whatlies.embedding import Embedding
from whatlies.embeddingset import EmbeddingSet


class TFHuggingfaceLang:
    def __init__(self, model='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model)
        self.model = TFBertModel.from_pretrained(model)

    def __getitem__(self, query: Union[str, List[str]]):
        """
        Retreive a single embedding or a set of embeddings.

        Arguments:
            query: single string or list of strings

        **Usage**
        ```python
        > lang = HuggingfaceTF("en_core_web_md")
        > lang['python']
        > lang[['python'], ['snake']]
        > lang[['nobody expects'], ['the spanish inquisition']]
        ```
        """
        if isinstance(query, str):
            input_ids = tf.constant(self.tokenizer.encode(query))[None, :]
            outputs = self.model(input_ids)
            vec = np.array(outputs[1]).reshape(-1)
            return Embedding(query, vec)
        return EmbeddingSet(*[self[tok] for tok in query])
