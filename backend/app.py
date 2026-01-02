import uvicorn
import numpy as np
import sqlite3
import tempfile
import os
from pathlib import Path
import librosa
import sys

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# ---------------- RHYTHM ----------------
from ingest.preprocess import preprocess_audio
from ingest.preprocess import compute_audio_hash
from ingest.extract_rhythm_features_v2 import extract_rhythm_v2
from rhythm.fingerprint_similarity import rhythm_fingerprint_similarity

# ---------------- MELODY ----------------
from melody.melody_preprocess import preprocess_melody
from melody.melody_fingerprint import extract_melody_fingerprint
from melody.melody_similarity import melody_similarity
from ingest.extract_features import extract_audio_features  # pitch extractor
from melody.melody_heatmap import melody_similarity_heatmap

# ✅ ADD THIS IMPORT
from melody.timeline_similarity import timeline_similarity

# ---------------- EMBEDDINGS ----------------
from ingest.extract_embeddings import extract_fused_embedding  # your fused embedding extractor

# ---------------- HARMONY ----------------
from harmony.chord_extraction import extract_chord_sequence
from harmony.harmonic_similarity import harmony_lcs_similarity
from harmony.utils import convert_to_roman_sequence
from harmony.key_detection import estimate_key_from_chroma

from models.filter_net.metric_embedder import embed_feature_vector





# =====================================================
# CONFIG
# =====================================================
DB_PATH = Path(__file__).resolve().parents[1] / "database" / "music.db"
TOP_K = 5

# Rhythm
MIN_MOTIF = 4
MAX_MOTIF_REF = 32

# Melody
MIN_MELODY_MOTIF = 8
MAX_MELODY_REF = 16

# =====================================================
# APP
# =====================================================
app = FastAPI(title="Music Copyright Similarity Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# UTILS
# =====================================================

def build_query_feature_vector(
    fused_emb,
    mfcc,
    chroma,
    pitch_freqs,
    tempo
):
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)
    chroma_mean = chroma.mean(axis=1)

    pitch_median = np.median(pitch_freqs) if len(pitch_freqs) else 0.0
    pitch_std = np.std(pitch_freqs) if len(pitch_freqs) else 0.0

    return np.concatenate([
        fused_emb,            # 1536
        mfcc_mean,            # 20
        mfcc_std,             # 20
        chroma_mean,          # 12
        [pitch_median, pitch_std, tempo]
    ]).astype(np.float32)


def cosine_similarity(a, b):
    if a is None or b is None:
        return 0.0
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def feature_similarity(a, b):
    """
    Returns per-feature similarity in [0,1]
    """
    sims = []
    for i in range(len(a)):
        denom = abs(a[i]) + abs(b[i]) + 1e-6
        sims.append(1.0 - abs(a[i] - b[i]) / denom)
    return [float(np.clip(s, 0.0, 1.0)) for s in sims]


# ---------------- RHYTHM DB ----------------
def load_db_rhythm():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT t.id, t.title,
               a.common_rhythm,
               a.rhythm_fingerprint
        FROM audio_features a
        JOIN tracks t ON t.id = a.track_id
        WHERE a.common_rhythm IS NOT NULL
          AND a.rhythm_fingerprint IS NOT NULL
    """)
    rows = cur.fetchall()
    conn.close()
    return rows

# ---------------- MELODY DB ----------------
def load_db_melody():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT t.id, t.title,
               a.pitch_freqs,
               a.pitch_conf
        FROM audio_features a
        JOIN tracks t ON t.id = a.track_id
        WHERE a.pitch_freqs IS NOT NULL
          AND a.pitch_conf IS NOT NULL
    """)
    rows = cur.fetchall()
    conn.close()
    return rows

# ---------------- FUSED EMBEDDINGS DB ----------------
def load_db_fused_embeddings():
    """
    Load all precomputed fused embeddings from DB
    Returns: list of tuples (track_id, title, embedding)
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT t.id, t.title, f.embedding
        FROM fused_embeddings f
        JOIN tracks t ON t.id = f.track_id
    """)
    rows = cur.fetchall()
    conn.close()

    # Convert embeddings from BLOB to np.array
    result = []
    for tid, title, emb_blob in rows:
        emb = np.frombuffer(emb_blob, np.float32)
        result.append((tid, title, emb))
    return result

def load_db_harmony():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT h.track_id, t.title, h.chord_sequence
        FROM harmony_features h
        JOIN tracks t ON t.id = h.track_id
    """)
    rows = cur.fetchall()
    conn.close()

    result = []
    for tid, title, blob in rows:
        chords = np.frombuffer(blob, dtype=np.int8)
        result.append((tid, title, chords))
    return result

# ---------------- ROMAN NGRAM STATS ----------------
def load_roman_ngram_stats():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT ngram, n, frequency
        FROM roman_ngram_stats
    """)
    rows = cur.fetchall()
    conn.close()

    stats = {}
    for ngram, n, freq in rows:
        stats.setdefault(n, {})[ngram] = freq

    return stats


def compute_harmonic_rarity(roman_seq, roman_ngram_stats, n_sizes=(3,4,5)):
    """
    Returns:
      - list of matched ngrams
      - list of frequencies
      - rarity score in [0,1]
    """
    freqs = []
    used = []

    for n in n_sizes:
        if n not in roman_ngram_stats:
            continue

        for i in range(len(roman_seq) - n + 1):
            ng = "-".join(roman_seq[i:i+n])
            if "X" in ng:
                continue

            if ng in roman_ngram_stats[n]:
                f = roman_ngram_stats[n][ng]
                freqs.append(f)
                used.append(ng)

    if not freqs:
        return [], [], 0.0

    # rarity = inverse frequency
    rarity = 1.0 - float(np.mean(freqs))
    return used, freqs, round(rarity, 4)


# =====================================================
# API
# =====================================================
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        path = tmp.name

    try:
        # ---------------- LOAD AUDIO ----------------
        # Load first 60 seconds and downsample to 16kHz
        y, sr = librosa.load(path, sr=16000, mono=True, duration=60)


        # ================= SELF MATCH HASH (CORRECT) =================
        y_id, duration_id, sr_id = preprocess_audio(path)

        # ---------------- COMPUTE FUSED EMBEDDING FOR UPLOADED SONG ----------------
        y_emb, _, sr_emb = preprocess_audio(path)
        query_emb = extract_fused_embedding(y_emb, sr_emb)

        

        # ---------------- FIND TOP-N CANDIDATES BY EMBEDDING ----------------
        # ---------------- FIND TOP-N CANDIDATES BY METRIC EMBEDDING ----------------
        DB_EMBS = load_db_fused_embeddings()
        top_N = 50

        
# ---- QUERY SIDE FEATURES ----
        mfcc_q = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        chroma_q = librosa.feature.chroma_cqt(y=y, sr=sr)
        tempo_q, _ = librosa.beat.beat_track(y=y, sr=sr)

        _, _, _, _, pitch_freqs, pitch_conf, _ = extract_audio_features(y, sr)

        query_feature_vec = build_query_feature_vector(
            fused_emb=query_emb,
            mfcc=mfcc_q,
            chroma=chroma_q,
            pitch_freqs=pitch_freqs,
            tempo=tempo_q
        )

        query_metric_emb = embed_feature_vector(query_feature_vec)

        emb_sims = []

        for tid, title, fused_db in DB_EMBS:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("""
                SELECT tempo, mfcc, chroma, pitch_freqs
                FROM audio_features
                WHERE track_id=?
            """, (tid,))
            row = cur.fetchone()
            conn.close()

            if row is None:
                continue

            tempo_db, mfcc_db, chroma_db, pitch_db = row

            mfcc_db = np.frombuffer(mfcc_db, np.float32).reshape(20, -1)
            chroma_db = np.frombuffer(chroma_db, np.float32).reshape(12, -1)
            pitch_db = np.frombuffer(pitch_db, np.float32)

            db_feature_vec = build_query_feature_vector(
                fused_emb=fused_db,
                mfcc=mfcc_db,
                chroma=chroma_db,
                pitch_freqs=pitch_db,
                tempo=tempo_db
            )

            db_metric_emb = embed_feature_vector(db_feature_vec)

            sim = float(np.dot(query_metric_emb, db_metric_emb))
            emb_sims.append((tid, title, sim))

# ---- SORT & SELECT ----
        emb_sims.sort(key=lambda x: x[2], reverse=True)
        candidate_ids_fused = set(t[0] for t in emb_sims[:top_N])

        top_fused = emb_sims[0]
        fused_embedding_top = {
            "track_id": top_fused[0],
            "title": top_fused[1],
            "similarity": round(float(top_fused[2]), 4)
        }




        chroma_q = librosa.feature.chroma_cqt(y=y_id, sr=sr_id)
        tempo_q, _ = librosa.beat.beat_track(y=y_id, sr=sr_id)

        query_audio_hash = compute_audio_hash(chroma_q, tempo_q, duration_id)

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "SELECT id, title FROM tracks WHERE audio_hash=?",
            (query_audio_hash,)
        )
        self_match = cur.fetchone()
        conn.close()


        query_audio_hash = compute_audio_hash(chroma_q, tempo_q, duration_id)

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "SELECT id, title FROM tracks WHERE audio_hash=?",
            (query_audio_hash,)
        )
        self_match = cur.fetchone()
        conn.close()



        

        # =====================================================
        # RHYTHM ANALYSIS (UNCHANGED)
        # =====================================================
        y_rhythm, _, sr_rhythm = preprocess_audio(path)

        

        common_q, fp_q = extract_rhythm_v2(y_rhythm, sr_rhythm)

        if common_q is None or fp_q is None:
            common_q = np.zeros(6, dtype=np.float32)  # same dim as DB common_rhythm
            fp_q = np.zeros(1, dtype=np.int8)




        rhythm_rows = load_db_rhythm()

        # FILTER DB tracks by embedding candidates
        rhythm_rows = [r for r in rhythm_rows if r[0] in candidate_ids_fused]

        rhythm_matches = []
        motif_lengths = []

        for tid, title, common_db, fp_db in rhythm_rows:
            common_db = np.frombuffer(common_db, np.float32)
            fp_db = np.frombuffer(fp_db, np.int8)

            stat_sim = cosine_similarity(common_q, common_db)
            feat_sim = feature_similarity(common_q, common_db)

            fp_sim, lcs_len = rhythm_fingerprint_similarity(fp_q, fp_db)


            if lcs_len >= MIN_MOTIF:
                motif_lengths.append(lcs_len)

            score = 0.65 * fp_sim + 0.35 * stat_sim

            is_self = (self_match is not None and tid == self_match[0])


            rhythm_matches.append({
                "track_id": tid,
                "title": title,
                "rhythm_similarity": 1.0 if is_self else round(score, 4),
                "fingerprint_similarity": 1.0 if is_self else round(fp_sim, 4),
                "shared_motif_length": MAX_MOTIF_REF if is_self else int(lcs_len),
                "feature_similarity": {} if is_self else {
                    "tempo": round(feat_sim[0], 4),
                    "mean_ibi": round(feat_sim[1], 4),
                    "std_ibi": round(feat_sim[2], 4),
                    "beat_density": round(feat_sim[3], 4),
                    "mean_onset": round(feat_sim[4], 4),
                    "std_onset": round(feat_sim[5], 4)
                },
                "self_match": is_self
            })

        # ================= FORCE SELF MATCH =================
        

        rhythm_matches.sort(key=lambda x: x["rhythm_similarity"], reverse=True)
        rhythm_top_k = rhythm_matches[:TOP_K]

        best_rhythm = rhythm_top_k[0]["rhythm_similarity"] if rhythm_top_k else 0.0
        max_motif = max(motif_lengths) if motif_lengths else 0
        reps = sum(l >= max_motif for l in motif_lengths)

        motif_conf = min(1.0, max_motif / MAX_MOTIF_REF)
        repetition_conf = min(1.0, reps / 4)
        confidence = 0.7 * motif_conf + 0.3 * repetition_conf

        rhythm_overall = round(best_rhythm * confidence, 4)

        if rhythm_overall >= 0.85 and reps >= 3:
            rhythm_risk = "HIGH"
        elif rhythm_overall >= 0.6:
            rhythm_risk = "MEDIUM"
        else:
            rhythm_risk = "LOW"
            

        # =====================================================
        # MELODY ANALYSIS
        # =====================================================

        

        midi_q = preprocess_melody(pitch_freqs, pitch_conf)

        melody_matches = []
        melody_motifs = []
        melody_contours = []

        if midi_q is not None:
            fp_q = extract_melody_fingerprint(midi_q)
            melody_rows = load_db_melody()

            # ================= STEP 2B: FILTER =================
            for tid, title, pf_db, pc_db in melody_rows:
                if tid not in candidate_ids_fused:
                    continue

            
                pf_db = np.frombuffer(pf_db, np.float32)
                pc_db = np.frombuffer(pc_db, np.float32)

                midi_db = preprocess_melody(pf_db, pc_db)
                if midi_db is None:
                    continue

                fp_db = extract_melody_fingerprint(midi_db)
                score, motif_len, q_start, q_end, db_start, db_end = melody_similarity(fp_q, fp_db)


                is_self = (self_match is not None and tid == self_match[0])

                if is_self:
                    score = 1.0
                    motif_len = MAX_MELODY_REF

                # ✅ TIMELINE SIMILARITY (FIXED)
                


                heatmap = None
                if motif_len >= MIN_MELODY_MOTIF:
                    heatmap = melody_similarity_heatmap(fp_q, fp_db)

                if motif_len >= MIN_MELODY_MOTIF:
                    melody_motifs.append(motif_len)

                melody_matches.append({
                    "track_id": tid,
                    "title": title,
                    "melody_similarity": round(score, 4),
                    "shared_motif_notes": int(motif_len),

                    "motif": {
                        "query": [q_start, q_end],
                        "db": [db_start, db_end]
                    },

                    "heatmap": heatmap.tolist() if heatmap is not None else None,

                    # ✅ NOW VALID
                    
                })

            melody_matches.sort(key=lambda x: x["melody_similarity"], reverse=True)
            melody_top_k = melody_matches[:TOP_K]

            best_melody = melody_top_k[0]["melody_similarity"] if melody_top_k else 0.0
            max_m = max(melody_motifs) if melody_motifs else 0
            melody_conf = min(1.0, max_m / MAX_MELODY_REF)
            melody_overall = round(best_melody * melody_conf, 4)

            if melody_overall >= 0.85 and max_m >= MIN_MELODY_MOTIF:
                melody_risk = "HIGH"
            elif melody_overall >= 0.6:
                melody_risk = "MEDIUM"
            else:
                melody_risk = "LOW"

            for m in melody_top_k:
                conn = sqlite3.connect(DB_PATH)
                cur = conn.cursor()
                cur.execute(
                    "SELECT pitch_freqs, pitch_conf FROM audio_features WHERE track_id=?",
                    (m["track_id"],)
                )
                row = cur.fetchone()
                conn.close()

                if not row:
                    continue

                pf_db = np.frombuffer(row[0], np.float32)
                pc_db = np.frombuffer(row[1], np.float32)
                midi_db = preprocess_melody(pf_db, pc_db)

                timeline_local = timeline_similarity(
                    query_pitch=midi_q,
                    db_pitch=midi_db,
                    window_size=16,
                    hop_size=8
                )


                melody_contours.append({
                    "track_id": m["track_id"],
                    "title": m["title"],
                    "similarity": m["melody_similarity"],
                    "shared_motif_notes": m["shared_motif_notes"],

                    "motif": m["motif"],
                    "heatmap": m["heatmap"],
                    "timeline_similarity": timeline_local,

                    "query": {
                        "time": list(range(len(midi_q))),
                        "pitch": [float(p) for p in midi_q]
                    },
                    "db_song": {
                        "time": list(range(len(midi_db))),
                        "pitch": [float(p) for p in midi_db]
                    }
                })

        else:
            melody_top_k = []
            melody_overall = 0.0
            melody_risk = "LOW"

        # =====================================================
# HARMONY ANALYSIS (STAGE 3 — ONLY HIGH-RISK TRACKS)
# =====================================================

        roman_ngram_stats = load_roman_ngram_stats()


# ---- Intersection of Rhythm & Melody Top-K ----
        rhythm_ids = set(r["track_id"] for r in rhythm_top_k)
        melody_ids = set(m["track_id"] for m in melody_top_k)
        high_risk_ids = rhythm_ids.intersection(melody_ids)
        if not high_risk_ids:
                high_risk_ids = rhythm_ids.union(melody_ids)

# ---- Extract query chord sequence (ONCE) ----
        chords_q, _ = extract_chord_sequence(y, sr)

        # ---- Query chroma for key detection ----
        chroma_q_h = librosa.feature.chroma_cqt(y=y, sr=sr)
        query_key_pc = estimate_key_from_chroma(chroma_q_h)


        harmony_rows = load_db_harmony()
        print("HARMONY SAMPLE:", harmony_rows[0] if harmony_rows else "EMPTY")
        harmony_results = []

        for tid, title, chords_db in harmony_rows:
            if tid not in high_risk_ids:
                continue

            score, shared, matches = harmony_lcs_similarity(
                chords_q,
                chords_db,
            )


            # ---- DB chroma + key ----
            

# Fallback: key from chord histogram
            db_key_pc = int(np.bincount(chords_db % 12).argmax())

# ---- Roman sequences ----
            roman_q = convert_to_roman_sequence(chords_q, query_key_pc)
            roman_db = convert_to_roman_sequence(chords_db, db_key_pc)

            rarity_ngrams, rarity_freqs, rarity_score = compute_harmonic_rarity(
                roman_q,
                roman_ngram_stats
            )

# ---- Roman LCS (reuse same logic) ----
            roman_score, roman_shared, _ = harmony_lcs_similarity(
                np.array(roman_q, dtype=object),
                np.array(roman_db, dtype=object)
            )

            


            harmony_results.append({
                "track_id": tid,
                "title": title,
                "harmony_similarity": round(score, 4),
                "shared_chords": int(shared),
                "roman_similarity": round(roman_score, 4),
                "roman_shared": int(roman_shared),
                "roman_query": roman_q,
                "roman_db": roman_db,
                "harmony_rarity": {
                    "ngrams": rarity_ngrams[:20],
                    "frequencies": rarity_freqs[:20],
                    "rarity_score": rarity_score
                },

                "alignment": [
                    {
                        "query_index": int(qi),
                        "db_index": int(di),
                        "chord": int(chords_q[qi])
                    }
                    for qi, di in matches
                ],
                "query_chords": chords_q.tolist(),
                "db_chords": chords_db.tolist()
            })

        # ================= KEEP ONLY BEST HARMONY MATCH =================
        best_harmony = None
        if harmony_results:
            best_harmony = max(
                harmony_results,
                key=lambda x: x["harmony_similarity"]
    )

# ---- Harmony Risk Decision ----
        if best_harmony:
            harmony_overall = best_harmony["harmony_similarity"]
        
            if harmony_overall >= 0.75 and best_harmony["shared_chords"] >= 4:
                harmony_risk = "HIGH"
            elif harmony_overall >= 0.5:
                harmony_risk = "MEDIUM"
            else:
                harmony_risk = "LOW"
        else:
            harmony_overall = 0.0
            harmony_risk = "LOW"

        final_score = round(
            0.4 * rhythm_overall +
            0.4 * melody_overall +
            0.2 * harmony_overall,
            4
        )

        if fused_embedding_top["similarity"] != fused_embedding_top["similarity"]:  # NaN check
            fused_embedding_top["similarity"] = 0.0


        # ---------------- FINAL RISK DECISION (FIXED) ----------------
        if final_score >= 0.75 and (
            rhythm_risk == "HIGH" or melody_risk == "HIGH"
        ):
            final_risk = "HIGH"
        elif final_score >= 0.45:
            final_risk = "MEDIUM"
        else:
            final_risk = "LOW"

       

        # =====================================================
        # RESPONSE
        # =====================================================
        return {
            "status": "success",

            "fused_embedding_top": fused_embedding_top,

            "rhythm_summary": {
                "overall_similarity": rhythm_overall,
                "risk_level": rhythm_risk
            },

            "melody_summary": {
                "overall_similarity": melody_overall,
                "risk_level": melody_risk
            },

            "harmony_summary": {
                "overall_similarity": harmony_overall,
                "risk_level": harmony_risk
            },

            "harmony_analysis": {
                "best_match": best_harmony
            },

            "rhythm_analysis": {
                "top_k_matches": rhythm_top_k
            },

            "melody_analysis": {
                "top_k_matches": melody_top_k
            },

            "melody_contours": melody_contours,

            "self_match": {
                "detected": self_match is not None,
                "track_id": self_match[0] if self_match else None,
                "title": self_match[1] if self_match else None
            },

            "final_decision": {
                "score": final_score,
                "risk_level": final_risk
            }

        }

    finally:
        os.remove(path)

# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
