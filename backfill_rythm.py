# backend/maintenance/backfill_rhythm.py

import sqlite3
import numpy as np
import librosa
import sys
from pathlib import Path

# Add parent directory to path to import from backend
sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.ingest.extract_rhythm_tempo import extract_rhythm_features

DB_PATH = "database/music.db"


def validate_rhythm_vector(rhythm_vec):
    """
    Validate that the rhythm vector meets expected criteria.
    Returns (is_valid, error_message)
    """
    if rhythm_vec is None:
        return False, "Vector is None"
    
    if not isinstance(rhythm_vec, np.ndarray):
        return False, "Not a numpy array"
    
    if rhythm_vec.dtype != np.float32:
        return False, f"Wrong dtype: {rhythm_vec.dtype} (expected float32)"
    
    # Check expected dimension (should be ~42 based on new extraction)
    expected_dim = 42  # Adjust if your implementation differs
    if len(rhythm_vec) != expected_dim:
        return False, f"Wrong dimension: {len(rhythm_vec)} (expected {expected_dim})"
    
    # Check for NaN or Inf
    if np.any(np.isnan(rhythm_vec)):
        return False, "Contains NaN values"
    
    if np.any(np.isinf(rhythm_vec)):
        return False, "Contains Inf values"
    
    # Check if vector is all zeros (might indicate extraction failure)
    if np.all(rhythm_vec == 0):
        return False, "All zeros - extraction may have failed"
    
    return True, "Valid"


def backfill_rhythm(force_all=False):
    """
    Backfill rhythm features with the NEW enhanced extraction method.
    
    This will:
    1. Detect tracks with missing, old 7-dim, or corrupted rhythm features
    2. Re-extract using the new 42-dim method
    3. Validate and store the new features
    
    Args:
        force_all: If True, re-extract ALL tracks regardless of existing data
    """
    
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Detect tracks that need updating
    if force_all:
        cur.execute("""
            SELECT t.id, t.file_path, t.title, length(af.rhythm_pattern) as rhythm_len
            FROM tracks t
            JOIN audio_features af ON t.id = af.track_id
        """)
    else:
        cur.execute("""
            SELECT t.id, t.file_path, t.title, length(af.rhythm_pattern) as rhythm_len
            FROM tracks t
            JOIN audio_features af ON t.id = af.track_id
            WHERE af.rhythm_pattern IS NULL
               OR length(af.rhythm_pattern) = 28      -- old 7-dim
               OR length(af.rhythm_pattern) = 1024    -- old 256-dim
               OR length(af.rhythm_pattern) = 0       -- empty
               OR length(af.rhythm_pattern) != 168    -- not new 42-dim
        """)

    rows = cur.fetchall()
    print(f"\n{'='*60}")
    print(f"🔎 Found {len(rows)} tracks to update rhythm vectors")
    print(f"{'='*60}\n")

    if len(rows) == 0:
        print("✅ All tracks already have valid rhythm features!")
        conn.close()
        return

    success_count = 0
    failed_count = 0
    failed_tracks = []

    for idx, (track_id, file_path, title, rhythm_len) in enumerate(rows, 1):
        print(f"[{idx}/{len(rows)}] Processing: {title}")
        print(f"   Track ID: {track_id}")
        print(f"   Old rhythm length: {rhythm_len if rhythm_len else 'NULL'} bytes")
        
        try:
            # Check if file exists
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Audio file not found: {file_path}")
            
            # Load audio
            print(f"   Loading audio from: {file_path}")
            y, sr = librosa.load(file_path, sr=None, mono=True)
            print(f"   Loaded: {len(y)} samples @ {sr} Hz ({len(y)/sr:.2f} seconds)")

            # Extract NEW rhythm features
            print(f"   Extracting rhythm features...")
            rhythm_vec = extract_rhythm_features(y, sr)
            
            # The function now GUARANTEES correct output, but double-check
            if not isinstance(rhythm_vec, np.ndarray):
                raise TypeError(f"extract_rhythm_features returned {type(rhythm_vec)}, expected ndarray")
            
            if rhythm_vec.shape != (42,):
                raise ValueError(f"Wrong shape: {rhythm_vec.shape}, expected (42,)")
            
            if rhythm_vec.dtype != np.float32:
                raise ValueError(f"Wrong dtype: {rhythm_vec.dtype}, expected float32")
            
            print(f"   ✓ Extracted {len(rhythm_vec)}-dim rhythm vector")
            print(f"   Vector stats: min={rhythm_vec.min():.4f}, "
                  f"max={rhythm_vec.max():.4f}, "
                  f"mean={rhythm_vec.mean():.4f}, "
                  f"std={rhythm_vec.std():.4f}")

            # Ensure float32 (should already be, but safety check)
            rhythm_vec = rhythm_vec.astype(np.float32)

            # Update database
            cur.execute("""
                UPDATE audio_features
                SET rhythm_pattern = ?
                WHERE track_id = ?
            """, (rhythm_vec.tobytes(), track_id))

            # Verify update
            cur.execute("""
                SELECT length(rhythm_pattern) 
                FROM audio_features 
                WHERE track_id = ?
            """, (track_id,))
            new_len = cur.fetchone()[0]
            
            if new_len != len(rhythm_vec) * 4:  # 4 bytes per float32
                raise ValueError(f"Database update failed: expected {len(rhythm_vec)*4} bytes, got {new_len}")
            
            print(f"   ✅ Successfully updated (new size: {new_len} bytes)\n")
            success_count += 1

        except FileNotFoundError as e:
            print(f"   ❌ File not found: {e}\n")
            failed_count += 1
            failed_tracks.append((track_id, title, str(e)))
            
        except Exception as e:
            print(f"   ❌ Failed: {type(e).__name__}: {e}\n")
            failed_count += 1
            failed_tracks.append((track_id, title, str(e)))

    # Commit all changes
    conn.commit()
    conn.close()
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 BACKFILL SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successfully updated: {success_count} tracks")
    print(f"❌ Failed: {failed_count} tracks")
    print(f"{'='*60}\n")
    
    if failed_tracks:
        print("❌ Failed tracks details:")
        print(f"{'='*60}")
        for track_id, title, error in failed_tracks:
            print(f"Track ID {track_id}: {title}")
            print(f"  Error: {error}\n")
    
    if success_count > 0:
        print("🎉 Rhythm feature backfill complete!")
        print(f"\n💡 Next steps:")
        print(f"   1. Test with a query: python backend/app.py")
        print(f"   2. Upload a song that's IN your database")
        print(f"   3. Check if it gets 100% rhythm similarity")
        print(f"   4. Upload a DIFFERENT song")
        print(f"   5. Check if it gets < 70% rhythm similarity")


def verify_backfill():
    """
    Verify that all tracks have valid rhythm features after backfill.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # Get rhythm feature statistics
    cur.execute("""
        SELECT 
            COUNT(*) as total_tracks,
            SUM(CASE WHEN af.rhythm_pattern IS NULL THEN 1 ELSE 0 END) as null_count,
            SUM(CASE WHEN length(af.rhythm_pattern) = 168 THEN 1 ELSE 0 END) as valid_42dim,
            SUM(CASE WHEN length(af.rhythm_pattern) = 28 THEN 1 ELSE 0 END) as old_7dim,
            SUM(CASE WHEN length(af.rhythm_pattern) NOT IN (0, 28, 168, 1024) 
                     AND af.rhythm_pattern IS NOT NULL THEN 1 ELSE 0 END) as other
        FROM tracks t
        JOIN audio_features af ON t.id = af.track_id
    """)
    
    stats = cur.fetchone()
    total, null_count, valid_42dim, old_7dim, other = stats
    
    conn.close()
    
    print(f"\n{'='*60}")
    print(f"🔍 RHYTHM FEATURES VERIFICATION")
    print(f"{'='*60}")
    print(f"Total tracks: {total}")
    print(f"Valid 42-dim (168 bytes): {valid_42dim}")
    print(f"NULL rhythm: {null_count}")
    print(f"Old 7-dim (28 bytes): {old_7dim}")
    print(f"Other sizes: {other}")
    print(f"{'='*60}\n")
    
    if null_count == 0 and old_7dim == 0 and valid_42dim == total:
        print("✅ All tracks have valid 42-dim rhythm features!")
        return True
    else:
        print("⚠️  Some tracks still need updating. Run backfill_rhythm() again.")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Backfill rhythm features with new extraction method')
    parser.add_argument('--verify', action='store_true', 
                       help='Only verify existing features without updating')
    parser.add_argument('--force-all', action='store_true',
                       help='Force re-extract ALL tracks regardless of existing data')
    
    args = parser.parse_args()
    
    if args.verify:
        verify_backfill()
    else:
        backfill_rhythm(force_all=args.force_all)
        print("\n" + "="*60)
        verify_backfill()