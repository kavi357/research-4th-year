# backend/similarity/similarity_utils.py

import numpy as np
from numpy.linalg import norm

def cosine_similarity(v1, v2):
    """Compute cosine similarity between two 512-d vectors."""
    if norm(v1) == 0 or norm(v2) == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm(v1) * norm(v2)))

def euclidean_distance(v1, v2):
    """Euclidean distance for MFCC & Chroma."""
    return float(norm(v1 - v2))

def normalize(dist):
    """Convert distance → similarity score (0–1)."""
    return float(1 / (1 + dist))

def combined_similarity(embed_sim, mfcc_dist, chroma_dist, 
                        w1=0.70, w2=0.15, w3=0.15):
    """
    Weighted score based on:
    - w1: embedding similarity (cosine)
    - w2: MFCC (distance → similarity)
    - w3: Chroma (distance → similarity)
    """

    mfcc_sim = normalize(mfcc_dist)
    chroma_sim = normalize(chroma_dist)

    score = (w1 * embed_sim) + (w2 * mfcc_sim) + (w3 * chroma_sim)
    return float(score)
