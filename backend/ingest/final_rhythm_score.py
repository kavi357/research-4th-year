def final_rhythm_score(common_rhythm_sim, fingerprint_sim, evidence):
    """
    Industry-valid rhythm scoring.
    """

    # No fingerprint → cap similarity
    if not evidence.get("motif_match", False):
        return min(common_rhythm_sim, 0.65)

    # Strong motif evidence
    if fingerprint_sim > 0.8 and evidence["matched_beats"] >= 6:
        return min(0.95, fingerprint_sim)

    # Partial motif
    return 0.6 + (fingerprint_sim * 0.3)
