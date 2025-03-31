# semantic_text_segmentation
Split a text into semantically coherent subtexts

# Usage

Install with 

```
pip isntall semantic_text_segmentation
```

Call the function `segment_text` with a list of sentences and a SentenceTransformer model. 
This will output the start and end indices and the list of sentences for each segment
See [test.py](test.py) for an example script.

# Credits

This is a simplification and update of the code published with the paper "Unsupervised Topic Segmentation of Meetings with BERT Embeddings" by Solbiati et al. [paper](https://arxiv.org/pdf/2106.12978) | [code](https://github.com/gdamaskinos/unsupervised_topic_segmentation)