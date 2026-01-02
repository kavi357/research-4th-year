##backend/melody/melody_simiarity
from .timeline_similarity import timeline_similarity

def lcs_with_indices(a, b):
    """
    Returns:
    length, a_start, a_end, b_start, b_end
    """
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n):
        for j in range(m):
            if a[i] == b[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])

    # traceback
    i, j = n, m
    a_idx = []
    b_idx = []

    while i > 0 and j > 0:
        if a[i-1] == b[j-1]:
            a_idx.append(i-1)
            b_idx.append(j-1)
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1

    if not a_idx:
        return 0, None, None, None, None

    a_idx.reverse()
    b_idx.reverse()

    return (
        len(a_idx),
        a_idx[0],
        a_idx[-1],
        b_idx[0],
        b_idx[-1]
    )


def melody_similarity(fp_q, fp_db):
    """
    fp = (intervals, contour)
    Returns:
    score, motif_len, q_start, q_end, db_start, db_end
    """
    int_q, cont_q = fp_q
    int_d, cont_d = fp_db

    if int_q is None or int_d is None:
        return 0.0, 0, None, None, None, None

    # Interval LCS
    l_int, qi_s, qi_e, di_s, di_e = lcs_with_indices(
        int_q.tolist(), int_d.tolist()
    )

    # Contour LCS
    l_cont, qc_s, qc_e, dc_s, dc_e = lcs_with_indices(
        cont_q.tolist(), cont_d.tolist()
    )

    score_int = l_int / min(len(int_q), len(int_d))
    score_cont = l_cont / min(len(cont_q), len(cont_d))
    score = 0.6 * score_int + 0.4 * score_cont

    # Choose stronger motif
    if l_int >= l_cont:
        motif_len = l_int
        q_start, q_end = qi_s, qi_e + 1
        db_start, db_end = di_s, di_e + 1

    else:
        motif_len = l_cont
        q_start, q_end = qc_s, qc_e
        db_start, db_end = dc_s, dc_e

    return score, motif_len, q_start, q_end, db_start, db_end


def analyze_melody_pair(query, db_song):
    """
    query, db_song contain extracted pitch contours
    """

    timeline = timeline_similarity(
        query_pitch=query["pitch"],
        db_pitch=db_song["pitch"],
        window_size=16,
        hop_size=8
    )

    return {
        "title": db_song["title"],
        "melody_similarity": db_song["overall_similarity"],
        "shared_motif_notes": db_song["shared_motif_notes"],
        "timeline_similarity": timeline
    }