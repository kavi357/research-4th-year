import numpy as np

def estimate_key_from_chroma(chroma):
    """
    Simple tonic estimation (publishable baseline)
    """
    chroma_mean = chroma.mean(axis=1)
    return int(np.argmax(chroma_mean))
