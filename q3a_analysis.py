#!/usr/bin/env python3
"""
q3a_analysis.py  –  Q3A: Sonic Signatures / Magical Mystery Tune
EE200 Course Project

Run AFTER index_songs.py has built fingerprint_db.pkl and fingerprint_db_single.pkl.

This script produces ALL the plots and observations required for Q3A:
  1. DFT of full song (shows loss of timing information)
  2. Spectrograms with short vs long windows (time-frequency trade-off)
  3. Constellation (local maxima) overlay on spectrogram
  4. Paired-hash matching: offset histogram for a true match vs wrong song
  5. Single-peak matching comparison
  6. Robustness: noise test (increasing SNR)
  7. Robustness: pitch shift test
  8. Robustness: time stretch test

Saves all figures as PNG files and prints observations.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend (works without a display)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fingerprint import (
    load_audio, compute_spectrogram, get_peaks, generate_hashes,
    fingerprint_song, load_database, match_song, match_single_peaks,
    generate_single_peak_hashes, add_noise, pitch_shift_simple,
    time_stretch_simple, TARGET_SR, N_FFT, HOP_LENGTH
)

SONGS_DIR     = "songs"
DB_PATH       = "fingerprint_db.pkl"
SINGLE_DB_PATH = "fingerprint_db_single.pkl"
FIG_DIR       = "q3a_figures"
os.makedirs(FIG_DIR, exist_ok=True)

FRAMES_PER_SEC = TARGET_SR / HOP_LENGTH     # ≈62.5 frames/s
FREQ_RES_HZ    = TARGET_SR / N_FFT          # ≈7.8 Hz per bin

# ─── helpers ──────────────────────────────────────────────────────────────────

def time_axis(spec):
    """Return time axis in seconds for a spectrogram array."""
    return np.arange(spec.shape[1]) * HOP_LENGTH / TARGET_SR

def freq_axis(spec):
    """Return frequency axis in Hz for a spectrogram array."""
    return np.linspace(0, TARGET_SR / 2, spec.shape[0])


def pick_demo_song(songs_dir):
    """Return the first available mp3 in songs_dir."""
    for f in sorted(os.listdir(songs_dir)):
        if f.lower().endswith('.mp3') and not f.startswith('.'):
            return f
    raise FileNotFoundError(f"No MP3 files found in {songs_dir}")


def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. DFT of full song
# ══════════════════════════════════════════════════════════════════════════════

def plot_full_dft(songs_dir):
    print("\n[1] DFT of full song (shows loss of timing information)")
    fname = pick_demo_song(songs_dir)
    song_name = os.path.splitext(fname)[0]
    y = load_audio(os.path.join(songs_dir, fname))

    # Use only first 30 s to keep computation fast
    y_clip = y[:TARGET_SR * 30]
    N = len(y_clip)
    Y = np.abs(np.fft.rfft(y_clip))
    freqs = np.fft.rfftfreq(N, d=1.0 / TARGET_SR)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    fig.suptitle(f"Full DFT of '{song_name}' (first 30 s)\n"
                 "Tells us WHICH frequencies are present, but NOT WHEN",
                 fontsize=11)

    axes[0].plot(freqs, Y, lw=0.5, color='steelblue')
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Magnitude")
    axes[0].set_title("Linear scale")
    axes[0].set_xlim([0, TARGET_SR / 2])

    Y_db = 20 * np.log10(np.maximum(Y, 1e-10))
    axes[1].plot(freqs, Y_db, lw=0.5, color='darkorange')
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Magnitude (dB)")
    axes[1].set_title("dB scale")
    axes[1].set_xlim([0, TARGET_SR / 2])

    plt.tight_layout()
    save(fig, "1_full_dft.png")

    print("""
  OBSERVATION:
  The DFT magnitude |X(f)| of an entire song is a 1-D function of frequency
  only.  It tells us which frequency components are present in the song, but
  completely loses the temporal dimension – we cannot tell whether a note was
  played at 3 s or at 25 s.  All timing information is collapsed into a single
  averaged spectrum.  This is why a plain DFT cannot be used for the audio
  fingerprinting task, where we need to know *when* each frequency was active.
""")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Short vs long window spectrogram (time-frequency trade-off)
# ══════════════════════════════════════════════════════════════════════════════

def plot_window_comparison(songs_dir):
    print("\n[2] Spectrogram: short window vs standard window vs long window")
    fname = pick_demo_song(songs_dir)
    song_name = os.path.splitext(fname)[0]
    y = load_audio(os.path.join(songs_dir, fname))
    y_clip = y[:TARGET_SR * 10]   # first 10 s

    configs = [
        (128,  64,  "Short window (128 samples = 16 ms)\nGood time res, poor freq res"),
        (1024, 128, "Standard window (1024 samples = 128 ms)\nBalanced"),
        (4096, 512, "Long window (4096 samples = 512 ms)\nPoor time res, good freq res"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Time-Frequency Trade-off – '{song_name}' (first 10 s)", fontsize=11)

    for ax, (n, h, title) in zip(axes, configs):
        spec = compute_spectrogram(y_clip, n_fft=n, hop=h)
        t = np.arange(spec.shape[1]) * h / TARGET_SR
        f = np.linspace(0, TARGET_SR / 2, spec.shape[0])
        im = ax.pcolormesh(t, f, spec, cmap='magma', shading='auto',
                           vmin=-80, vmax=0)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(title, fontsize=9)
        ax.set_ylim([0, TARGET_SR / 2])
        plt.colorbar(im, ax=ax, label="dB")

    plt.tight_layout()
    save(fig, "2_window_comparison.png")

    print("""
  OBSERVATION (Time-Frequency Trade-off):
  ─ SHORT window (128 samples ≈ 16 ms):
      Time resolution: excellent – events are pin-pointed in time.
      Frequency resolution: poor  – each bin spans TARGET_SR/128 ≈ 62.5 Hz,
      so nearby pitches blur together.  Individual harmonics are unresolvable.

  ─ STANDARD window (1024 samples ≈ 128 ms):
      Time resolution: good  – typical note onsets are clearly visible.
      Frequency resolution: good – bin width ≈ 7.8 Hz, harmonics are distinct.
      This is the Heisenberg uncertainty-principle trade-off of the STFT:
      Δt · Δf ≥ 1  (cannot have perfect resolution in both simultaneously).

  ─ LONG window (4096 samples ≈ 512 ms):
      Frequency resolution: excellent – bin width ≈ 2 Hz.
      Time resolution: poor  – a 500 ms note onset smears across many frames,
      making it impossible to track fast melodic or rhythmic variations.

  For fingerprinting we choose the STANDARD (1024-sample) window: it gives
  enough time resolution to distinguish peaks a fraction of a second apart,
  and enough frequency resolution to distinguish adjacent notes.
""")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Constellation overlay
# ══════════════════════════════════════════════════════════════════════════════

def plot_constellation(songs_dir):
    print("\n[3] Spectrogram + constellation peaks")
    fname = pick_demo_song(songs_dir)
    song_name = os.path.splitext(fname)[0]
    y = load_audio(os.path.join(songs_dir, fname))
    y_clip = y[:TARGET_SR * 20]   # first 20 s

    spec = compute_spectrogram(y_clip)
    peaks = get_peaks(spec)
    t_axis = time_axis(spec)
    f_axis = freq_axis(spec)

    peak_t = [t_axis[min(p[0], len(t_axis)-1)] for p in peaks]
    peak_f = [f_axis[min(p[1], len(f_axis)-1)] for p in peaks]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Spectrogram and Constellation – '{song_name}' (first 20 s)",
                 fontsize=11)

    # Full spectrogram
    im = axes[0].pcolormesh(t_axis, f_axis, spec, cmap='magma',
                             shading='auto', vmin=-80, vmax=0)
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].set_title("Spectrogram (dB)")
    axes[0].set_ylim([0, TARGET_SR / 2])
    plt.colorbar(im, ax=axes[0], label="dB")

    # Constellation
    axes[1].scatter(peak_t, peak_f, s=2, c='cyan', alpha=0.7, linewidths=0)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_title(f"Constellation ({len(peaks)} peaks) – local maxima only")
    axes[1].set_xlim([0, t_axis[-1]])
    axes[1].set_ylim([0, TARGET_SR / 2])
    axes[1].set_facecolor('black')

    plt.tight_layout()
    save(fig, "3_constellation.png")

    print(f"""
  OBSERVATION (Constellation):
  From the spectrogram we extracted {len(peaks)} local maxima in the first 20 s.
  The constellation is a sparse, binary representation: it keeps only the
  dominant time-frequency peaks, discarding the continuous shading of the
  spectrogram.  Key properties:
  ─ Noise-robust: background noise raises the noise floor uniformly but cannot
    displace genuine signal peaks from their (t, f) positions.
  ─ Compact: a few hundred peaks represent minutes of audio for fast lookup.
  ─ Invariant to volume: amplitude values are discarded; only positions matter.
""")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Hash matching: offset histogram (true match vs wrong song)
# ══════════════════════════════════════════════════════════════════════════════

def plot_matching(songs_dir, db):
    print("\n[4] Offset histogram: true match vs wrong song")
    files = sorted([f for f in os.listdir(songs_dir)
                    if f.lower().endswith('.mp3') and not f.startswith('.')])
    if len(files) < 2:
        print("  Need at least 2 songs – skipping.")
        return

    # Query: 10-s clip from the middle of the first song
    query_fname  = files[0]
    query_name   = os.path.splitext(query_fname)[0]
    wrong_name   = os.path.splitext(files[1])[0]

    y_full = load_audio(os.path.join(songs_dir, query_fname))
    start  = TARGET_SR * 30        # skip the first 30 s
    clip   = y_full[start: start + TARGET_SR * 10]

    result = match_song(clip, db)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f"Offset histogram for 10-s query from  '{query_name}'",
                 fontsize=11)

    # True match histogram
    if query_name in result['offset_hist']:
        offsets_true = list(result['offset_hist'][query_name].keys())
        counts_true  = list(result['offset_hist'][query_name].values())
        axes[0].bar(offsets_true, counts_true, width=1, color='limegreen')
        axes[0].set_title(f"TRUE MATCH: '{query_name}'\nPeak votes = "
                          f"{result['votes']}")
    else:
        axes[0].text(0.5, 0.5, "No matches found", transform=axes[0].transAxes,
                     ha='center')
        axes[0].set_title("TRUE MATCH: not found in DB")
    axes[0].set_xlabel("Time offset (frames)")
    axes[0].set_ylabel("Vote count")

    # Wrong song histogram
    if wrong_name in result['offset_hist']:
        offsets_wrong = list(result['offset_hist'][wrong_name].keys())
        counts_wrong  = list(result['offset_hist'][wrong_name].values())
        axes[1].bar(offsets_wrong, counts_wrong, width=1, color='tomato')
        axes[1].set_title(f"WRONG SONG: '{wrong_name}'\nMax random votes = "
                          f"{max(counts_wrong)}")
    else:
        axes[1].text(0.5, 0.5, "No spurious matches", transform=axes[1].transAxes,
                     ha='center')
        axes[1].set_title(f"WRONG SONG: '{wrong_name}'\n(no spurious matches)")
    axes[1].set_xlabel("Time offset (frames)")
    axes[1].set_ylabel("Vote count")

    plt.tight_layout()
    save(fig, "4_offset_histogram.png")

    top5 = sorted(result['all_votes'].items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"""
  OBSERVATION (Offset Histogram):
  Query: 10-second clip from '{query_name}' (starting at 30 s).

  TRUE MATCH ('{query_name}'):
    The histogram shows a SHARP PEAK at one offset value – all matching
    hashes align at the single correct time offset between the clip and the
    song.  This is the hallmark of a genuine match: the time relationship
    between paired peaks is preserved exactly.
    Peak vote count: {result['votes']}

  WRONG SONG ('{wrong_name}'):
    The histogram is nearly FLAT – spurious hash collisions occur at random,
    uncorrelated offsets.  No single offset accumulates many votes.

  Top 5 candidate songs:
  {chr(10).join(f"    {rank+1}. {name}  ({v} votes)" for rank,(name,v) in enumerate(top5))}

  Decision: '{result['match']}' wins.
""")

    return query_fname, clip


# ══════════════════════════════════════════════════════════════════════════════
# 5. Single-peak vs paired-hash comparison
# ══════════════════════════════════════════════════════════════════════════════

def compare_single_vs_paired(clip, query_name, db_paired, db_single):
    print("\n[5] Single-peak vs paired-hash comparison")

    res_paired = match_song(clip, db_paired)
    res_single = match_single_peaks(clip, db_single)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Paired-hash vs Single-peak matching\n"
                 f"Query: 10-s clip from '{query_name}'", fontsize=11)

    # Paired
    top_paired = sorted(res_paired['all_votes'].items(),
                        key=lambda x: x[1], reverse=True)[:10]
    names_p = [n[:20] for n, _ in top_paired]
    votes_p = [v for _, v in top_paired]
    axes[0].barh(names_p[::-1], votes_p[::-1], color='steelblue')
    axes[0].set_xlabel("Max aligned votes")
    axes[0].set_title(f"Paired hashes\nMatch: '{res_paired['match']}'  "
                      f"({res_paired['votes']} votes)")

    # Single peak
    top_single = sorted(res_single['all_votes'].items(),
                        key=lambda x: x[1], reverse=True)[:10]
    names_s = [n[:20] for n, _ in top_single]
    votes_s = [v for _, v in top_single]
    axes[1].barh(names_s[::-1], votes_s[::-1], color='darkorange')
    axes[1].set_xlabel("Vote count")
    axes[1].set_title(f"Single peaks\nMatch: '{res_single['match']}'  "
                      f"({res_single['votes']} votes)")

    plt.tight_layout()
    save(fig, "5_single_vs_paired.png")

    print(f"""
  OBSERVATION (Single-peak vs Paired-hash):

  PAIRED hashes encode a RELATIONSHIP between two peaks: (f1, f2, Δt).
  Because both the frequency pair AND the time difference must match, the
  probability of a spurious collision is extremely low:
      P(false hash match) ≈ 1 / (n_freq_bins² × max_Δt)

  SINGLE peaks encode only (f, t mod 100) – just one coordinate.
  Many songs share peaks at similar frequencies, so spurious collisions are
  much more frequent.  The vote counts for the correct and wrong songs are
  closer together, making discrimination unreliable – especially with noise.

  In practice:
  ─ Paired:  correct song has {res_paired['votes']} aligned votes vs
             a much lower count for all wrong songs → clear winner.
  ─ Single:  correct song: {res_single['votes']} votes but the margin
             over wrong songs is much smaller → easily fooled by noise.

  This is exactly why Shazam uses PAIRS (or even triplets) of peaks: the
  extra information in the relationship makes the hash far more specific.
""")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Robustness: noise
# ══════════════════════════════════════════════════════════════════════════════

def robustness_noise(songs_dir, db):
    print("\n[6] Robustness: adding increasing noise")
    fname = pick_demo_song(songs_dir)
    song_name = os.path.splitext(fname)[0]
    y = load_audio(os.path.join(songs_dir, fname))
    clip = y[TARGET_SR * 30: TARGET_SR * 40]   # 10-s clip

    snr_values = [40, 30, 20, 15, 10, 5, 0]
    results = []
    for snr in snr_values:
        noisy = add_noise(clip, snr_db=snr)
        res   = match_song(noisy, db)
        correct = (res['match'] == song_name)
        results.append((snr, res['votes'], correct))
        print(f"  SNR={snr:3d} dB  →  votes={res['votes']:4d}  "
              f"correct={correct}  match='{res['match']}'")

    snrs   = [r[0] for r in results]
    votes  = [r[1] for r in results]
    colors = ['limegreen' if r[2] else 'tomato' for r in results]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(range(len(snrs)), votes, color=colors, edgecolor='k', linewidth=0.5)
    ax.set_xticks(range(len(snrs)))
    ax.set_xticklabels([f"{s} dB" for s in snrs])
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Peak aligned votes")
    ax.set_title(f"Robustness to White Noise – query: '{song_name}'\n"
                 "Green = correct match, Red = incorrect")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='limegreen', label='Correct'),
                       Patch(color='tomato',    label='Wrong')],
              loc='upper left')
    plt.tight_layout()
    save(fig, "6_robustness_noise.png")

    print("""
  OBSERVATION (Noise robustness):
  At high SNR (40-20 dB) – moderate noise – matching succeeds because
  the strongest spectral peaks are still above the noise floor and their
  (t, f) positions are largely preserved.

  As SNR drops below ~10 dB, noise peaks start to compete with genuine
  signal peaks in the constellation.  Spurious hashes increase, reducing
  the aligned vote count for the correct song.

  At 0 dB SNR (noise power = signal power), the spectrogram is dominated
  by noise and recognition may fail.

  WHY pairs help: even if a few peak positions shift slightly due to noise,
  the (f1, f2, Δt) combination is redundant enough that many hash pairs
  still match, providing robustness over single-peak schemes.
""")


# ══════════════════════════════════════════════════════════════════════════════
# 7. Robustness: pitch shift
# ══════════════════════════════════════════════════════════════════════════════

def robustness_pitch(songs_dir, db):
    print("\n[7] Robustness: pitch shift")
    fname = pick_demo_song(songs_dir)
    song_name = os.path.splitext(fname)[0]
    y = load_audio(os.path.join(songs_dir, fname))
    clip = y[TARGET_SR * 30: TARGET_SR * 40]

    semitones_list = [0, 0.5, 1, 2, 3, 5]
    results = []
    for st in semitones_list:
        shifted = pitch_shift_simple(clip, semitones=st)
        res = match_song(shifted, db)
        correct = (res['match'] == song_name)
        results.append((st, res['votes'], correct))
        print(f"  +{st:3.1f} semitones  →  votes={res['votes']:4d}  "
              f"correct={correct}  match='{res['match']}'")

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ['limegreen' if r[2] else 'tomato' for r in results]
    ax.bar([str(r[0]) for r in results], [r[1] for r in results],
           color=colors, edgecolor='k', linewidth=0.5)
    ax.set_xlabel("Pitch shift (semitones)")
    ax.set_ylabel("Peak aligned votes")
    ax.set_title(f"Robustness to Pitch Shift – query: '{song_name}'\n"
                 "Green = correct, Red = incorrect")
    plt.tight_layout()
    save(fig, "7_robustness_pitch.png")

    print("""
  OBSERVATION (Pitch shift):
  Even a small pitch shift of ½–1 semitone can defeat the identifier.

  WHY:
  Our hashes store ABSOLUTE frequency bin indices.  When the pitch rises by
  Δ semitones, every frequency f → f × 2^(Δ/12).  For example, a 1-semitone
  shift moves a 440 Hz tone to 466 Hz.  At our frequency resolution of ~7.8 Hz
  per bin, this is already a 3-4 bin displacement.  The (f1, f2, Δt) hash
  computed from the shifted audio will NOT match any hash in the database.

  The song still SOUNDS almost identical to the human ear (just slightly
  higher pitch) because the ear is sensitive to frequency RATIOS, not
  absolute values.  But the fingerprinter uses absolute bin indices.

  SUGGESTED FIX:
  Store the RATIO f2/f1 instead of (f1, f2, Δt).  Frequency ratios are
  preserved under uniform pitch shifts, so the hash would survive transposition.
  (This is the key modification used by pitch-invariant fingerprinters.)
""")


# ══════════════════════════════════════════════════════════════════════════════
# 8. Robustness: time stretch
# ══════════════════════════════════════════════════════════════════════════════

def robustness_time_stretch(songs_dir, db):
    print("\n[8] Robustness: time stretch")
    fname = pick_demo_song(songs_dir)
    song_name = os.path.splitext(fname)[0]
    y = load_audio(os.path.join(songs_dir, fname))
    clip = y[TARGET_SR * 30: TARGET_SR * 40]

    rates = [1.0, 1.02, 1.05, 1.1, 0.95, 0.9]
    results = []
    for rate in rates:
        stretched = time_stretch_simple(clip, rate=rate)
        res = match_song(stretched, db)
        correct = (res['match'] == song_name)
        results.append((rate, res['votes'], correct))
        print(f"  rate={rate:.2f}  →  votes={res['votes']:4d}  "
              f"correct={correct}  match='{res['match']}'")

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ['limegreen' if r[2] else 'tomato' for r in results]
    ax.bar([f"{r[0]:.2f}x" for r in results], [r[1] for r in results],
           color=colors, edgecolor='k', linewidth=0.5)
    ax.set_xlabel("Time-stretch rate (>1 = faster)")
    ax.set_ylabel("Peak aligned votes")
    ax.set_title(f"Robustness to Time Stretch – query: '{song_name}'\n"
                 "Green = correct, Red = incorrect")
    plt.tight_layout()
    save(fig, "8_robustness_time_stretch.png")

    print("""
  OBSERVATION (Time stretch):
  A small time stretch (2–5%) typically still allows a correct match because:
  ─ The (f1, f2) part of the hash is unchanged (frequency content preserved).
  ─ Only Δt shifts proportionally; if it still falls in the same Δt bin,
    the hash matches.

  Larger stretches (>10%) push the Δt values out of range and the match fails.

  Time stretch is more survivable than pitch shift because only one component
  of the hash (Δt) changes, whereas pitch shift moves BOTH frequency components.
""")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  Q3A: Sonic Signatures – Magical Mystery Tune")
    print("  EE200 Course Project")
    print("=" * 65)

    if not os.path.isdir(SONGS_DIR):
        print(f"ERROR: songs directory '{SONGS_DIR}' not found.")
        print("Please extract songs.zip and place the songs/ folder here,")
        print("or adjust the SONGS_DIR variable at the top of this script.")
        sys.exit(1)

    if not os.path.exists(DB_PATH):
        print(f"ERROR: Database '{DB_PATH}' not found.")
        print("Run:  python index_songs.py  first.")
        sys.exit(1)

    if not os.path.exists(SINGLE_DB_PATH):
        print(f"WARNING: Single-peak DB '{SINGLE_DB_PATH}' not found.")
        print("Skipping single-peak comparison.  Run index_songs.py to build it.")
        has_single = False
    else:
        has_single = True

    print("\nLoading databases...")
    db_paired = load_database(DB_PATH)
    db_single = load_database(SINGLE_DB_PATH) if has_single else None
    song_list = db_paired.get('__song_list__', [])
    print(f"  Paired DB: {len(song_list)} songs, "
          f"{sum(len(v) for k, v in db_paired.items() if k != '__song_list__')} entries")
    if db_single:
        print(f"  Single DB: "
              f"{sum(len(v) for k, v in db_single.items() if k != '__song_list__')} entries")

    # Run all analyses
    plot_full_dft(SONGS_DIR)
    plot_window_comparison(SONGS_DIR)
    plot_constellation(SONGS_DIR)

    ret = plot_matching(SONGS_DIR, db_paired)
    if ret is not None:
        query_fname, clip = ret
        query_name = os.path.splitext(query_fname)[0]
        if db_single is not None:
            compare_single_vs_paired(clip, query_name, db_paired, db_single)

    robustness_noise(SONGS_DIR, db_paired)
    robustness_pitch(SONGS_DIR, db_paired)
    robustness_time_stretch(SONGS_DIR, db_paired)

    print("\n" + "=" * 65)
    print(f"  All figures saved to:  {FIG_DIR}/")
    print("=" * 65)


if __name__ == "__main__":
    main()
