import sys
from sentence_transformers import SentenceTransformer

from semantic_text_segmentation.segmentation import segment_text

sentences = [x.strip() for x in open(sys.stdin).readlines()]
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

segments = segment_text(sentences, model)

for start, end in zip([0] + segments, segments + [len(sentences) + 1]):
    print(" ".join(sentences[start:end]))
    print("-----")
