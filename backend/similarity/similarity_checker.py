# backend/similarity/similarity_checker.py

import numpy as np
from ingest.extract_audio_features import extract_audio_features
from ingest.extract_embeddings import extract_openl3_embedding
from similarity.similarity_utils import (
    cosine_similarity, euclidean_distance, combined_similarity
)
from similarity.load_from_db import load_reference_data


def compare_song_to_database(wav_path, db_path):
    print("\n🎵 Extracting features for uploaded song…")

    tempo, mfcc, chroma = extract_audio_features(wav_path)
    embed = extract_openl3_embedding(wav_path)

    # Load all reference data
    ref_embs, ref_feats = load_reference_data(db_path)

    results = []

    print("\n🔍 Comparing with database…\n")

    for track_id in ref_embs:
        ref_emb = ref_embs[track_id]
        ref_mfcc, ref_chroma = ref_feats[track_id]

        # Calculate metrics
        sim_embed = cosine_similarity(embed, ref_emb)
        dist_mfcc = euclidean_distance(mfcc, ref_mfcc)
        dist_chroma = euclidean_distance(chroma, ref_chroma)

        final_score = combined_similarity(sim_embed, dist_mfcc, dist_chroma)

        results.append({
            "track_id": track_id,
            "embedding_similarity": sim_embed,
            "mfcc_distance": dist_mfcc,
            "chroma_distance": dist_chroma,
            "final_similarity": final_score
        })

    # Sort by best match first
    results = sorted(results, key=lambda x: x["final_similarity"], reverse=True)

    return results
