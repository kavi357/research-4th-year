#backend/rhythm/fingerprint_simlarity.py
def lcs(a, b):
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)):
        for j in range(len(b)):
            if a[i] == b[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[-1][-1]

def rhythm_fingerprint_similarity(a, b):
    if a is None or b is None:
        return 0.0, 0

    l = lcs(a, b)
    return l / min(len(a), len(b)), l
