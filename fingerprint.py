import numpy as np
import scipy.signal as sig
import scipy.io.wavfile as wavfile
import pickle
import os
import io



TARGET_SR        = 8000    
N_FFT            = 1024  
HOP_LENGTH       = 128     
N_FREQ_BINS      = N_FFT // 2 + 1   


PEAK_NEIGHBORHOOD_FREQ  = 20
PEAK_NEIGHBORHOOD_TIME  = 20
PEAK_MIN_AMP_DB         = -60.0    


FAN_VALUE        = 15
MAX_PAIR_DT_FRAMES = 200  
MIN_PAIR_DT_FRAMES = 1


MIN_VOTES        = 5       




def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple polyphase resample using scipy."""
    if orig_sr == target_sr:
        return audio
    ratio_num = target_sr
    ratio_den = orig_sr
    from math import gcd
    g = gcd(ratio_num, ratio_den)
    return sig.resample_poly(audio, ratio_num // g, ratio_den // g)


def load_audio(path: str, sr: int = TARGET_SR) -> np.ndarray:
    """
    Load an MP3 or WAV file, convert to mono float32, resample to `sr`.
    Uses librosa for MP3 (handles all common formats).
    """
    import librosa
    y, orig_sr = librosa.load(path, sr=None, mono=True)
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
    return y.astype(np.float32)


def load_audio_bytes(data: bytes, sr: int = TARGET_SR) -> np.ndarray:
    """Load audio from raw bytes (for Streamlit file_uploader)."""
    import librosa
    y, orig_sr = librosa.load(io.BytesIO(data), sr=None, mono=True)
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
    return y.astype(np.float32)




def compute_spectrogram(y: np.ndarray,
                        n_fft: int = N_FFT,
                        hop: int = HOP_LENGTH) -> np.ndarray:


    window = np.hanning(n_fft)

    y_padded = np.pad(y, n_fft // 2, mode='reflect')

    n_frames = 1 + (len(y_padded) - n_fft) // hop
    spec = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float32)

    for i in range(n_frames):
        frame = y_padded[i * hop: i * hop + n_fft] * window
        fft_out = np.fft.rfft(frame)
        spec[:, i] = np.abs(fft_out)

    spec_db = 20.0 * np.log10(np.maximum(spec, 1e-10))
    return spec_db



def time_axis(spec_db: np.ndarray,
              sr: int = TARGET_SR,
              hop: int = HOP_LENGTH) -> np.ndarray:
    """Return the time (in seconds) corresponding to each spectrogram frame."""
    n_frames = spec_db.shape[1]
    return np.arange(n_frames) * hop / float(sr)


def freq_axis(spec_db: np.ndarray,
              sr: int = TARGET_SR,
              n_fft: int = N_FFT) -> np.ndarray:
    """Return the frequency (in Hz) corresponding to each spectrogram bin."""
    n_bins = spec_db.shape[0]
    return np.fft.rfftfreq(n_fft, d=1.0 / sr)[:n_bins]


def get_peaks(spec_db: np.ndarray,
              neighborhood_freq: int = PEAK_NEIGHBORHOOD_FREQ,
              neighborhood_time: int = PEAK_NEIGHBORHOOD_TIME,
              min_amp_db: float = PEAK_MIN_AMP_DB) -> list:

    from scipy.ndimage import maximum_filter

    struct = np.ones((2 * neighborhood_freq + 1,
                      2 * neighborhood_time + 1), dtype=bool)
    local_max = maximum_filter(spec_db, footprint=struct) == spec_db

    # Suppress edges and quiet points
    background = spec_db <= min_amp_db
    eroded_background = maximum_filter(background, footprint=struct)

    detected_peaks = local_max & ~eroded_background

    freq_idx, time_idx = np.where(detected_peaks)
    peaks = sorted(zip(time_idx.tolist(), freq_idx.tolist()))
    return peaks   # list of (t_frame, f_bin)




def generate_hashes(peaks: list,
                    fan_value: int = FAN_VALUE,
                    max_dt: int = MAX_PAIR_DT_FRAMES,
                    min_dt: int = MIN_PAIR_DT_FRAMES) -> list:

    hashes = []
    n = len(peaks)
    for i, (t1, f1) in enumerate(peaks):
        for j in range(i + 1, min(i + 1 + fan_value, n)):
            t2, f2 = peaks[j]
            dt = t2 - t1
            if dt < min_dt:
                continue
            if dt > max_dt:
                break
            h = (int(f1), int(f2), int(dt))
            hashes.append((h, t1))
    return hashes




def fingerprint_song(y: np.ndarray) -> list:

    spec = compute_spectrogram(y)
    peaks = get_peaks(spec)
    hashes = generate_hashes(peaks)
    return hashes


def build_database(songs_dir: str, db_path: str = "fingerprint_db.pkl",
                   verbose: bool = True) -> dict:

    db = {}
    song_list = []

    supported = ('.mp3', '.wav', '.flac', '.ogg', '.m4a')
    files = sorted([f for f in os.listdir(songs_dir)
                    if f.lower().endswith(supported)
                    and not f.startswith('.')])

    for idx, fname in enumerate(files):
        song_name = os.path.splitext(fname)[0]
        fpath = os.path.join(songs_dir, fname)

        if verbose:
            print(f"[{idx+1}/{len(files)}]  Indexing: {song_name}")

        try:
            y = load_audio(fpath)
            hashes = fingerprint_song(y)
            for h, t in hashes:
                if h not in db:
                    db[h] = []
                db[h].append((song_name, t))
            song_list.append(song_name)
        except Exception as e:
            print(f"  !! ERROR loading {fname}: {e}")

    db['__song_list__'] = song_list

    with open(db_path, 'wb') as f:
        pickle.dump(db, f)

    if verbose:
        print(f"\nDone. {len(song_list)} songs indexed → {db_path}")
        print(f"Total hash entries: {sum(len(v) for k, v in db.items() if k != '__song_list__')}")

    return db


def load_database(db_path: str = "fingerprint_db.pkl") -> dict:
    with open(db_path, 'rb') as f:
        return pickle.load(f)




def match_song(query_audio: np.ndarray,
               db: dict,
               min_votes: int = MIN_VOTES) -> dict:

    spec_q     = compute_spectrogram(query_audio)
    peaks_q    = get_peaks(spec_q)
    hashes_q   = generate_hashes(peaks_q)


    offset_hist = {}  

    for (h, t_q) in hashes_q:
        if h in db:
            for (song_name, t_s) in db[h]:
                offset = t_s - t_q
                if song_name not in offset_hist:
                    offset_hist[song_name] = {}
                offset_hist[song_name][offset] = \
                    offset_hist[song_name].get(offset, 0) + 1


    all_votes = {s: max(cnt.values()) for s, cnt in offset_hist.items()}

    if not all_votes:
        return dict(match=None, votes=0, offset_hist={},
                    all_votes={}, query_hashes=len(hashes_q))

    best_song  = max(all_votes, key=all_votes.get)
    best_votes = all_votes[best_song]

    return dict(
        match        = best_song if best_votes >= min_votes else None,
        votes        = best_votes,
        offset_hist  = offset_hist,
        all_votes    = all_votes,
        query_hashes = len(hashes_q),
    )




def generate_single_peak_hashes(peaks: list) -> list:

    hashes = []
    for (t, f) in peaks:
        h = (int(f), int(t % 100))
        hashes.append((h, t))
    return hashes


def build_database_single_peaks(songs_dir: str,
                                 db_path: str = "fingerprint_db_single.pkl",
                                 verbose: bool = True) -> dict:

    db = {}
    song_list = []
    supported = ('.mp3', '.wav', '.flac', '.ogg', '.m4a')
    files = sorted([f for f in os.listdir(songs_dir)
                    if f.lower().endswith(supported)
                    and not f.startswith('.')])

    for idx, fname in enumerate(files):
        song_name = os.path.splitext(fname)[0]
        fpath = os.path.join(songs_dir, fname)
        if verbose:
            print(f"[{idx+1}/{len(files)}]  Indexing (single peaks): {song_name}")
        try:
            y = load_audio(fpath)
            spec = compute_spectrogram(y)
            peaks = get_peaks(spec)
            hashes = generate_single_peak_hashes(peaks)
            for h, t in hashes:
                if h not in db:
                    db[h] = []
                db[h].append((song_name, t))
            song_list.append(song_name)
        except Exception as e:
            print(f"  !! ERROR: {e}")

    db['__song_list__'] = song_list
    with open(db_path, 'wb') as f:
        pickle.dump(db, f)
    if verbose:
        print(f"\nSingle-peak DB done → {db_path}")
    return db


def match_single_peaks(query_audio: np.ndarray, db: dict) -> dict:

    spec_q  = compute_spectrogram(query_audio)
    peaks_q = get_peaks(spec_q)
    hashes_q = generate_single_peak_hashes(peaks_q)

    votes = {}
    for (h, t_q) in hashes_q:
        if h in db:
            for (song_name, _) in db[h]:
                votes[song_name] = votes.get(song_name, 0) + 1

    if not votes:
        return dict(match=None, votes=0, all_votes={})
    best = max(votes, key=votes.get)
    return dict(match=best, votes=votes[best], all_votes=votes)




def add_noise(y: np.ndarray, snr_db: float) -> np.ndarray:
    """Add white Gaussian noise at the given SNR (dB)."""
    signal_power = np.mean(y ** 2)
    noise_power  = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(len(y)) * np.sqrt(noise_power)
    return (y + noise).astype(np.float32)


def pitch_shift_simple(y: np.ndarray, semitones: float,
                        sr: int = TARGET_SR) -> np.ndarray:

    factor = 2 ** (semitones / 12.0)

    n_orig  = len(y)
    n_new   = int(round(n_orig / factor))
    y_resampled = sig.resample(y, n_new)
    return y_resampled.astype(np.float32)


def time_stretch_simple(y: np.ndarray, rate: float) -> np.ndarray:

    n_new = int(round(len(y) / rate))
    return sig.resample(y, n_new).astype(np.float32)
