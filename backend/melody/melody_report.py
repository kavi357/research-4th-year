#backend/melody/melody_report

def melody_report(score, motif_len):
    return {
        "melody_similarity": round(score, 3),
        "shared_motif_notes": motif_len,
        "interpretation": (
            "High melodic similarity risk"
            if score > 0.8 and motif_len >= 8 else
            "Moderate melodic similarity"
            if score > 0.6 else
            "Low melodic similarity"
        )
    }
