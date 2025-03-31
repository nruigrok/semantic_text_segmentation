#!/usr/bin/env python3

# This code is based on https://github.com/gdamaskinos/unsupervised_topic_segmentation/blob/main/core.py

from typing import Iterable, NamedTuple, Sequence
from sentence_transformers import SentenceTransformer
import numpy as np

from numpy import dot
from numpy.linalg import norm


def depth_score(timeseries):
    depth_scores = []
    for i in range(1, len(timeseries) - 1):
        left, right = i - 1, i + 1
        while left > 0 and timeseries[left - 1] > timeseries[left]:
            left -= 1
        while right < (len(timeseries) - 1) and timeseries[right + 1] > timeseries[right]:
            right += 1
        depth_scores.append((timeseries[right] - timeseries[i]) + (timeseries[left] - timeseries[i]))
    return depth_scores


def smooth(timeseries, n, s):
    smoothed_timeseries = timeseries[:]
    for _ in range(n):
        for index in range(len(smoothed_timeseries)):
            neighbours = smoothed_timeseries[max(0, index - s) : min(len(timeseries) - 1, index + s)]
            smoothed_timeseries[index] = sum(neighbours) / len(neighbours)
    return smoothed_timeseries


def cosine_similarity(a: np.array, b: np.array) -> float:
    return float(dot(a, b) / (norm(a) * norm(b)))


def block_comparison_score(embeddings, k):
    for i in range(k, len(embeddings) - k):
        yield cosine_similarity(embeddings[i - k : i + 1].max(axis=0), embeddings[i + 1 : i + k + 2].max(axis=0))


def get_local_maxima(array):
    # NOTE they keep slicing the tail and head to compare left and right regions
    # (in chunking, then in depth calc, and now in local maxima calc)
    # Probably OK as you don't want to cut near the edges anyway, but seems strange?
    for i in range(1, len(array) - 1):
        if array[i - 1] < array[i] and array[i] > array[i + 1]:
            yield i, array[i]


def depth_score_to_topic_change_indexes(depth_score_timeseries, relative_depth_threshold=0.6):
    """
    capped add a max segment limit so there are not too many segments, used for UI improvements on the Workplace TeamWork product
    """
    threshold = relative_depth_threshold * max(depth_score_timeseries)
    local_maxima = get_local_maxima(depth_score_timeseries)

    return [ix for ix, m in local_maxima if m > threshold]


def flatten_features(batches_features):
    res = []
    for batch_features in batches_features:
        res += batch_features
    return res


def split_list(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(min(len(a), n)))


def get_segments(
    sentences, model, tiling_comparison_window, smoothing_passes, smoothing_window, relative_depth_threshold
):
    embeddings = model.encode(sentences)
    scores = list(block_comparison_score(embeddings, k=tiling_comparison_window))
    scores = smooth(scores, n=smoothing_passes, s=smoothing_window)

    depth_scores = depth_score(scores)

    segments = depth_score_to_topic_change_indexes(depth_scores, relative_depth_threshold=relative_depth_threshold)

    # to get sentence index from the depth scores, "pad" the index with
    # the sentence comparison window plus one because depth calc doesn't use the edges
    segments = [x + tiling_comparison_window + 1 for x in segments]

    return segments


class Segment(NamedTuple):
    start: int
    end: int
    sentences: Sequence[str]


def segment_text(
    sentences: Sequence[str],
    model: SentenceTransformer,
    *,
    tiling_comparison_window=10,
    smoothing_passes=2,
    smoothing_window=1,
    relative_depth_threshold=0.6
) -> Iterable[Segment]:
    """Segement a text into semantically coherent subtexts

    :param sentences: a sequence of sentences (strings)
    :param model: a SentenceTransformer model
    :param tiling_comparison_window: How for to look 'left' and 'right' to generate chunks
    :param smoothing_passes: How many smoothing passes to make
    :param smoothing_window: In smoothing, size of the window to average
    :param relative_depth_threshold: The threshold (as percentage of highest distance) to select cuts
    """
    embeddings = model.encode(sentences)
    scores = list(block_comparison_score(embeddings, k=tiling_comparison_window))
    scores = smooth(scores, n=smoothing_passes, s=smoothing_window)

    depth_scores = depth_score(scores)

    segments = depth_score_to_topic_change_indexes(depth_scores, relative_depth_threshold=relative_depth_threshold)

    # to get sentence index from the depth scores, "pad" the index with
    # the sentence comparison window plus one because depth calc doesn't use the edges
    segments = get_segments(
        sentences,
        model,
        tiling_comparison_window=tiling_comparison_window,
        smoothing_passes=smoothing_passes,
        smoothing_window=smoothing_window,
        relative_depth_threshold=relative_depth_threshold,
    )

    return [
        Segment(start=start, end=end, sentences=sentences[start:end])
        for start, end in zip([0] + segments, segments + [len(sentences) + 1])
    ]
