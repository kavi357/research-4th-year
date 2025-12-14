import numpy as np

def tempo_similarity(t1, t2):
    diff = abs(t1 - t2)
    return max(0.0, 1 - diff / max(t1, t2, 1))


def pitch_similarity(p1, p2):
    diff = abs(p1 - p2)
    return max(0.0, 1 - diff / max(p1, p2, 1))


def hybrid_basic_score(tempo_sim, pitch_sim):
    return (tempo_sim + pitch_sim) / 2
