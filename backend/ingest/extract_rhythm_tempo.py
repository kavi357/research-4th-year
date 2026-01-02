# extract_rhythm_tempo.py - BULLETPROOF VERSION

import numpy as np
import librosa

def extract_rhythm_features(y, sr):
    """
    Enhanced rhythm features with better discriminative power.
    Returns exactly 42-dim float32 vector.
    
    GUARANTEED OUTPUT: np.ndarray with shape (42,) and dtype float32
    """
    
    # Ensure float32 input
    y = np.asarray(y, dtype=np.float32)
    
    # Initialize all features as Python floats (not numpy scalars)
    tempo = 120.0
    tempo_strength = 0.0
    onset_density = 0.0
    ioi_mean = 0.0
    ioi_std = 0.0
    ioi_median = 0.0
    beat_regularity = 0.0
    beat_mean_interval = 0.0
    syncopation = 0.0
    rhythm_entropy = 0.0
    beat_strength_mean = 0.0
    beat_strength_std = 0.0
    
    # Initialize arrays
    ioi_hist = np.zeros(10, dtype=np.float32)
    onset_autocorr = np.zeros(20, dtype=np.float32)
    
    # ==============================================
    # 1) TEMPO with confidence
    # ==============================================
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo_val, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_env, 
            sr=sr,
            units='frames'
        )
        
        # Force scalar conversion
        tempo = float(np.asarray(tempo_val).item())
        
        # Get tempo strength (confidence)
        tempogram = librosa.feature.tempogram(
            onset_envelope=onset_env, 
            sr=sr
        )
        tempo_strength = float(np.max(tempogram))
        
    except Exception as e:
        tempo = 120.0
        tempo_strength = 0.0
        beat_frames = np.array([])
        onset_env = np.zeros(100, dtype=np.float32)
    
    # ==============================================
    # 2) ONSET PATTERN (key for rhythm identity)
    # ==============================================
    try:
        # Onset envelope statistics
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, 
            sr=sr,
            backtrack=True
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        duration = float(len(y) / sr)
        if duration > 0 and len(onset_times) > 0:
            onset_density = float(len(onset_times) / duration)
        else:
            onset_density = 0.0
        
        # Inter-onset intervals (IOI)
        if len(onset_times) > 1:
            ioi = np.diff(onset_times)
            ioi_mean = float(np.mean(ioi))
            ioi_std = float(np.std(ioi))
            ioi_median = float(np.median(ioi))
            
            # IOI distribution (histogram bins)
            hist_counts, _ = np.histogram(ioi, bins=10, range=(0, 2.0))
            ioi_hist = hist_counts.astype(np.float32)
            
            # Normalize histogram
            hist_sum = float(np.sum(ioi_hist))
            if hist_sum > 0:
                ioi_hist = ioi_hist / hist_sum
                
    except Exception as e:
        pass  # Keep defaults
    
    # ==============================================
    # 3) ONSET AUTOCORRELATION (rhythmic pattern)
    # ==============================================
    try:
        # This captures repeating rhythmic patterns
        autocorr = librosa.autocorrelate(onset_env, max_size=50)
        autocorr = autocorr[:20]  # Take first 20 lags
        
        # Normalize but preserve shape
        max_val = float(np.max(autocorr))
        if max_val > 0:
            onset_autocorr = (autocorr / max_val).astype(np.float32)
        else:
            onset_autocorr = autocorr.astype(np.float32)
            
    except Exception as e:
        pass  # Keep default zeros
    
    # ==============================================
    # 4) BEAT SYNCHRONOUS FEATURES
    # ==============================================
    try:
        if len(beat_frames) > 2:
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            beat_intervals = np.diff(beat_times)
            
            # Beat regularity
            std_val = float(np.std(beat_intervals))
            if std_val > 1e-5:
                beat_regularity = float(1.0 / std_val)
            else:
                beat_regularity = 0.0
                
            beat_mean_interval = float(np.mean(beat_intervals))
            
            # Beat strength variation
            # Ensure we don't go out of bounds
            valid_frames = beat_frames[beat_frames < len(onset_env)]
            if len(valid_frames) > 1:
                beat_onsets = onset_env[valid_frames[:-1]]
                beat_strength_mean = float(np.mean(beat_onsets))
                beat_strength_std = float(np.std(beat_onsets))
                
    except Exception as e:
        pass  # Keep defaults
    
    # ==============================================
    # 5) RHYTHMIC COMPLEXITY MEASURES
    # ==============================================
    try:
        # Syncopation index (how "off-beat" the rhythm is)
        if len(onset_times) > 0 and len(beat_frames) > 0:
            beat_times_all = librosa.frames_to_time(beat_frames, sr=sr)
            
            # Find closest beat for each onset
            syncopation_scores = []
            for onset_t in onset_times:
                distances = np.abs(beat_times_all - onset_t)
                min_dist = float(np.min(distances))
                syncopation_scores.append(min_dist)
            
            if len(syncopation_scores) > 0:
                syncopation = float(np.mean(syncopation_scores))
            
        # Rhythmic entropy
        if np.sum(ioi_hist) > 0:
            # Remove zeros to avoid log(0)
            nonzero_hist = ioi_hist[ioi_hist > 1e-10]
            if len(nonzero_hist) > 0:
                rhythm_entropy = float(-np.sum(nonzero_hist * np.log2(nonzero_hist)))
                
    except Exception as e:
        pass  # Keep defaults
    
    # ==============================================
    # COMBINE INTO FEATURE VECTOR - BULLETPROOF
    # ==============================================
    
    # Build feature list explicitly
    feature_list = []
    
    # Add scalar features (10 values)
    scalars = [
        tempo, tempo_strength, onset_density,
        ioi_mean, ioi_std, ioi_median,
        beat_regularity, beat_mean_interval,
        syncopation, rhythm_entropy
    ]
    for val in scalars:
        feature_list.append(float(val))
    
    # Add beat strength (2 values)
    feature_list.append(float(beat_strength_mean))
    feature_list.append(float(beat_strength_std))
    
    # Add IOI histogram (10 values)
    for val in ioi_hist:
        feature_list.append(float(val))
    
    # Add onset autocorrelation (20 values)
    for val in onset_autocorr:
        feature_list.append(float(val))
    
    # Convert to numpy array - GUARANTEED 1D float32
    feature_vec = np.array(feature_list, dtype=np.float32)
    
    # Final validation
    assert feature_vec.shape == (42,), f"Wrong shape: {feature_vec.shape}"
    assert feature_vec.dtype == np.float32, f"Wrong dtype: {feature_vec.dtype}"
    
    # Replace any NaN or Inf with 0
    feature_vec = np.nan_to_num(feature_vec, nan=0.0, posinf=0.0, neginf=0.0)
    
    return feature_vec