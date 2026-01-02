#backend/rhythm/rhythm_report.py

def rhythm_report(common_sim, fp_sim, lcs_len):
    return {
        "common_rhythm_similarity": round(common_sim, 3),
        "rhythm_fingerprint_similarity": round(fp_sim, 3),
        "fingerprint_match": fp_sim > 0.85,
        "shared_motif_beats": lcs_len,
        "interpretation": (
            "Strong rhythmic copying"
            if fp_sim > 0.85 else
            "No significant rhythmic copying"
        )
    }
