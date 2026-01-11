from datetime import datetime
import sys
import nltk
from sentence_transformers import SentenceTransformer
import re
from semantic_text_segmentation import segment_text
from pathlib import Path

nltk.download("punkt_tab")

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def try_params(thr):
    segs = segment_text(
        model=model,
        sentences=sentences,
        tiling_comparison_window=4,
        smoothing_window=2,
        smoothing_passes=1,
        relative_depth_threshold=thr,
    )
    lengths = [len(s.sentences) for s in segs]
    print(f"thr={thr:.2f} → {len(segs)} segments, lengths={lengths}")
    return segs


if __name__ == "__main__":
    folder = Path("/home/nel/blout")
    folder = Path("/home/nel/alm_data/text/Omroep Brabant/TV")
    txt_files = list(folder.glob("*.txt"))
    for file in txt_files:
        with open(file, "r") as f:
            text = f.read()

            sentences = nltk.tokenize.sent_tokenize(text)
            segments = segment_text(
                model=model,
                sentences=sentences,
                tiling_comparison_window=8,  # ~5 sentences per comparison block
                smoothing_window=2,  # small smoothing window
                smoothing_passes=1,  # moderate smoothing
                relative_depth_threshold=0.45,  # moderate sensitivity
            )

            for i, segment in enumerate(segments, start=1):
                article = {}
                article["text"] = " ".join(segment.sentences)
                article["programma"] = file.stem.rsplit("_")[0]
                date_match = re.search(r"\d{4}-\d{2}-\d{2}", file.stem)
                date_str = date_match.group(0)
                article["date"] = datetime.strptime(date_str, "%Y-%m-%d")
                article["url"] = f"{article['programma']}_{article['date']}_item_{i}"
                print(article)
