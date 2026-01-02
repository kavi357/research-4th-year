import numpy as np

def melody_similarity_heatmap(fp_q, fp_db, window=5):
    """
    fp = (intervals, contour)
    Returns 2D similarity matrix
    """
    int_q, cont_q = fp_q
    int_d, cont_d = fp_db

    if int_q is None or int_d is None:
        return None

    nq = len(int_q) - window
    nd = len(int_d) - window

    if nq <= 0 or nd <= 0:
        return None

    heatmap = np.zeros((nq, nd), dtype=np.float32)

    for i in range(nq):
        q_win = int_q[i:i+window]
        for j in range(nd):
            d_win = int_d[j:j+window]

            # interval similarity (key-invariant)
            sim = np.mean(q_win == d_win)
            heatmap[i, j] = sim

    return heatmap
