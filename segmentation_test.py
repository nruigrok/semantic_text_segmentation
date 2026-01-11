from datetime import datetime
import logging
from pathlib import Path
import re
from types import SimpleNamespace

from amcat4py import AmcatClient
import nltk
import ollama
import numpy as np
from semantic_text_segmentation import segment_text

nltk.download("punkt", quiet=True)

# =========================
# Index velden
# =========================

FIELDS = {
    "url": {"type": "url", "identifier": "true"},
    "text": "text",
    "title": "text",
    "publisher": "keyword",
    "date": "date",
    "page": "tag",
    "programma": "keyword",
    "file": "keyword",
    "keywords": "tag",
    "tags": "tag",
    "platform": "keyword",
    "external_id": "keyword",
    "gemeente": "tag",
    "mediumtype": "keyword",
    "article_type": "keyword",
    "file_path": "keyword",
    "region": "keyword",
    "category": "keyword",
}

# =========================
# Metadata parsing
# =========================


def parse_metadata(file_path: Path) -> dict:
    """
    Haal publisher, platform, datum en programmanaam uit het pad/bestand.
    Voorbeeld bestandsnaam:
        Omroep/Radio/AFSLAG-RIJNMOND-20240401-1700.txt
    """
    stem = file_path.stem
    platform = file_path.parent.name.upper()
    publisher = file_path.parent.parent.name

    # Datum herkennen: 20240401 of 2024-04-01
    m = re.search(r"(\d{4})(\d{2})(\d{2})|(\d{4})-(\d{2})-(\d{2})", stem)
    date = None
    if m:
        year = int(m.group(1) or m.group(4))
        month = int(m.group(2) or m.group(5))
        day = int(m.group(3) or m.group(6))
        date = datetime(year, month, day).date().isoformat()

    # Programma afleiden uit bestandsnaam
    if "--" in stem:
        parts = stem.split("--")
        programma = parts[1] if len(parts) >= 3 else "Unknown"
    else:
        # alles na eerste '-' tot aan datum weghalen
        programma = re.sub(r"^.*?-", "", stem)
        programma = re.sub(r"-(?:\d{8}|\d{4}-\d{2}-\d{2}).*$", "", programma)

    return {
        "file": str(file_path),
        "publisher": publisher,
        "platform": platform,
        "date": date,
        "programma": programma.strip(),
    }


# =========================
# Embedding model (Ollama)
# =========================


class OllamaModel:
    def __init__(self, model: str = "mxbai-embed-large"):
        self.model = model

    def encode(self, sentences):
        """
        semantic_text_segmentation verwacht een model met een .encode()
        die een numpy-array teruggeeft met embeddings per zin.
        """
        resp = ollama.embed(model=self.model, input=sentences)
        return np.array(resp["embeddings"], dtype=np.float32)


# =========================
# Segment refinements
# =========================


def refine_segments(segments, min_sents_per_segment: int = 2):
    """
    Plak hele kleine segmentjes (korter dan min_sents_per_segment) aan hun buren.
    segments: lijst van SimpleNamespace(sentences=[...])
    """
    refined = []
    buffer = []

    for seg in segments:
        seg_sents = list(seg.sentences)

        if len(seg_sents) < min_sents_per_segment:
            # bufferen en later plakken
            buffer.extend(seg_sents)
            continue

        if buffer:
            seg_sents = buffer + seg_sents
            buffer = []

        refined.append(SimpleNamespace(sentences=seg_sents))

    if buffer:
        if refined:
            refined[-1].sentences.extend(buffer)
        else:
            refined.append(SimpleNamespace(sentences=buffer))

    return refined


# =========================
# Domeinspecifieke cues & reclamefilter
# =========================

AD_URL_RE = re.compile(r"\b[\w\-]+\.(nl|com|eu|net)\b", re.IGNORECASE)

CUE_PATTERNS = [
    r"^Het weer\.",  # Weerblok
    r"^En dan de files\.",  # Files
    r"^Dat was het nieuws\.",  # Afsluiting nieuws
    r"^Meer via onze app\b",
    r"^Meer via onze website\b",
]


def is_ad_sentence(s: str) -> bool:
    """
    Detecteer 'reclameachtige' zinnen.
    Kort + URL → grote kans op reclame.
    """
    s_strip = s.strip()
    if len(s_strip.split()) <= 15 and AD_URL_RE.search(s_strip):
        return True
    return False


def split_segment_on_cues(segment, cue_patterns=CUE_PATTERNS):
    """
    Neem één segment (SimpleNamespace(sentences=[...])) en splits op harde cues.
    Cues zelf blijven aan het begin van het nieuwe subsegment staan.
    """
    sentences = list(segment.sentences)
    if not sentences:
        return []

    new_segments = []
    current = []

    for sent in sentences:
        if any(re.search(p, sent) for p in cue_patterns) and current:
            # sluit huidig segment en begin nieuwe bij cue
            new_segments.append(SimpleNamespace(sentences=current))
            current = [sent]
        else:
            current.append(sent)

    if current:
        new_segments.append(SimpleNamespace(sentences=current))

    return new_segments


def drop_ad_segments(segments):
    """
    Verwijder segmenten die vrijwel zeker alleen reclame zijn:
    - korte tekst
    - met .nl / .com / ...
    """
    cleaned = []
    for seg in segments:
        text = " ".join(seg.sentences).strip()
        words = text.split()
        if len(words) <= 20 and AD_URL_RE.search(text):
            # segment lijkt puur reclame → skip
            continue
        cleaned.append(seg)
    return cleaned


def classify_segment_type(segment):
    """
    Heel simpele type-classificatie (optioneel, voor 'article_type'):
    - 'weer' als het begint met 'Het weer.'
    - 'files' als het begint met 'En dan de files.'
    - 'reclame' als erg kort + URL
    - anders 'nieuws'
    """
    if not segment.sentences:
        return "nieuws"
    first = segment.sentences[0].strip()

    if re.match(r"^Het weer\.", first):
        return "weer"
    if re.match(r"^En dan de files\.", first):
        return "files"

    text = " ".join(segment.sentences).strip()
    words = text.split()
    if len(words) <= 20 and AD_URL_RE.search(text):
        return "reclame"

    return "nieuws"


# =========================
# Main
# =========================

if __name__ == "__main__":
    logging.basicConfig(format="[%(levelname)-7s] %(message)s", level=logging.INFO)
    logger = logging.getLogger("segmentatie")
    logger.setLevel(logging.WARNING)

    # externe libs stiller
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("amcat4py").setLevel(logging.WARNING)
    logging.getLogger("ollama").setLevel(logging.WARNING)

    conn = AmcatClient("http://localhost/amcat")
    index = "svdj2024_audio"

    # Let op: delete_index alleen gebruiken als je echt opnieuw wilt beginnen!
    conn.delete_index(index)
    conn.create_index(index)
    conn.set_fields(index, FIELDS)

    folder = Path("/home/nel/alm_data/text/")
    txt_files = list(folder.rglob("*.txt"))
    logger.info("Found %d txt files", len(txt_files))

    model = OllamaModel("mxbai-embed-large")

    for file in txt_files:
        meta = parse_metadata(file)
        raw_text = file.read_text(encoding="utf-8", errors="replace").strip()
        if not raw_text:
            continue

        # 1. Zinnen
        sentences = nltk.tokenize.sent_tokenize(raw_text, language="dutch")
        logger.info("File %s: %d sentences", file.name, len(sentences))

        k = 4  # window voor semantic tiling

        # 2. Semantische segmentatie
        if len(sentences) <= 2 * k + 2:
            all_segments = [SimpleNamespace(sentences=sentences)]
        else:
            try:
                all_segments = segment_text(
                    model=model,
                    sentences=sentences,
                    tiling_comparison_window=k,
                    smoothing_window=2,
                    smoothing_passes=2,
                    relative_depth_threshold=0.3,
                )
            except (ValueError, ZeroDivisionError) as e:
                logger.warning("Segmentatie mislukt in %s (%s)", file.name, e)
                all_segments = [SimpleNamespace(sentences=sentences)]

        # 3. Kleine semantische segmenten aan elkaar plakken
        all_segments = refine_segments(all_segments, min_sents_per_segment=4)

        # 4. Domein-cues: 'Het weer', 'En dan de files', 'Dat was het nieuws', etc.
        split_segments = []
        for seg in all_segments:
            split_segments.extend(split_segment_on_cues(seg))

        # 5. Reclame-segmenten eruit
        all_segments = drop_ad_segments(split_segments)

        logger.info(
            "File %s: totaal %d segmenten na cues+ads",
            file.name,
            len(all_segments),
        )

        # 6. Voorbereiden voor upload
        articles = []
        for i, segment in enumerate(all_segments, start=1):
            seg_text = " ".join(segment.sentences).strip()
            if not seg_text:
                continue

            article_type = classify_segment_type(segment)

            doc = {
                **meta,
                "title": f"{meta['programma']}-{meta['date'] or 'unknown-date'}_{i}",
                "text": seg_text,
                "url": f"{file.stem}_seg_{i}",
                "file": f"{file.stem}_seg_{i}.txt",
                "article_type": article_type,
                "mediumtype": "Regionale Omroep",
                "platform": meta.get("platform", "AUDIO"),
            }
            articles.append(doc)

        if not articles:
            continue

        logger.info("Uploading %d segments from %s", len(articles), file.name)
        try:
            conn.upload_documents(index, articles)
        except Exception:
            logger.exception("Upload failed for %s", file)
            raise
