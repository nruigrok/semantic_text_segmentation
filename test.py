import sys
from sentence_transformers import SentenceTransformer

from semantic_text_segmentation import segment_text

sentences = [x.strip() for x in open(sys.argv[1]).readlines()]
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

for start, end, sentences in segment_text(sentences, model):
    print(" ".join(sentences[start:end]))
    print("-----")
