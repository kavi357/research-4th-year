import numpy as np

def transpose_invariant(seq):
    if len(seq) == 0:
        return seq
    base = seq[0]
    return [(x - base) % 24 for x in seq]

def lcs_with_path(a, b):
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    back = [[None]*(len(b)+1) for _ in range(len(a)+1)]

    for i in range(len(a)):
        for j in range(len(b)):
            if a[i] == b[j]:
                dp[i+1][j+1] = dp[i][j] + 1
                back[i+1][j+1] = (i, j)
            else:
                if dp[i][j+1] >= dp[i+1][j]:
                    dp[i+1][j+1] = dp[i][j+1]
                else:
                    dp[i+1][j+1] = dp[i+1][j]

    # 🔁 BACKTRACK MATCHES
    i, j = len(a), len(b)
    matches = []

    while i > 0 and j > 0:
        if back[i][j] is not None:
            pi, pj = back[i][j]
            matches.append((pi, pj))
            i, j = pi, pj
        elif dp[i-1][j] >= dp[i][j-1]:
            i -= 1
        else:
            j -= 1

    matches.reverse()
    return dp[-1][-1], matches


def harmony_lcs_similarity(a, b):
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return 0.0, 0, []

    # 🔑 ONLY transpose if numeric (raw chords)
    if isinstance(a[0], (int, np.integer)):
        a_t = transpose_invariant(a)
        b_t = transpose_invariant(b)
    else:
        # Roman numerals → already key invariant
        a_t = a
        b_t = b

    l, matches = lcs_with_path(a_t, b_t)
    score = l / min(len(a), len(b))

    return float(score), int(l), matches

