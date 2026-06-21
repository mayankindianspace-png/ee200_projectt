# EE200 – Q3: Sonic Signatures / Zapptain America

Audio fingerprinting system (Shazam-style) built for the EE200 Course Project.

---

## Files

| File | Purpose |
|------|---------|
| `fingerprint.py` | Core library (STFT, peak picking, hashing, matching) |
| `index_songs.py` | **Run once** – builds the fingerprint database |
| `q3a_analysis.py` | **Q3A** – all experiments, plots, and observations |
| `app.py` | **Q3B** – Streamlit web app (single-clip + batch modes) |
| `requirements.txt` | Python dependencies |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Extract songs

Extract `songs.zip` so you have a `songs/` folder containing all the MP3 files:

```bash
unzip songs.zip
```

### 3. Build the fingerprint database (run once)

```bash
python index_songs.py --songs_dir songs/
```

This creates two files:
- `fingerprint_db.pkl` — paired-hash database (used for identification)
- `fingerprint_db_single.pkl` — single-peak database (used in Q3A comparison)

Indexing ~50 songs takes roughly 3–10 minutes depending on your machine.

---

## Q3A: Analysis

After indexing, run all experiments:

```bash
python q3a_analysis.py
```

Figures are saved to the `q3a_figures/` directory.  Observations are printed to
the terminal and should be incorporated into your report PDF.

---

## Q3B: Streamlit App

### Run locally

```bash
streamlit run app.py
```

The app opens in your browser at http://localhost:8501.

### Deploy to Streamlit Community Cloud (free)

1. Push all files to a **public GitHub repository**.
   - Include `fingerprint_db.pkl` and `fingerprint_db_single.pkl` in the repo
     (use Git LFS if they are larger than 100 MB each; typically they are ~50–150 MB).
2. Go to https://share.streamlit.io/ and click **New app**.
3. Point it to your repo, set the main file to `app.py`.
4. Deploy.  The live link is what you submit.

> **Note**: The database files MUST ship with the app so it works immediately
> without any setup step.  The submission instructions require this.

---

## Algorithm Summary

```
Audio file
    │
    ▼
compute_spectrogram()   →   2-D power spectrogram in dB
    │                       (STFT, Hann window, 1024-point FFT, hop=128)
    ▼
get_peaks()             →   Constellation: local maxima
    │                       (scipy.ndimage.maximum_filter, 2-D neighbourhood)
    ▼
generate_hashes()       →   Paired-peak hashes: (f1, f2, Δt)
    │                       (each peak paired with next FAN_VALUE=15 peaks)
    ▼
Database lookup         →   For each hash: retrieve (song_name, t_anchor)
    │
    ▼
Offset voting           →   offset = t_song − t_query
    │                       True match → all offsets align at ONE value
    ▼
Winner = song with max aligned-offset count
```

### Key parameters (in `fingerprint.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `TARGET_SR` | 8000 Hz | Resample rate (smaller = faster) |
| `N_FFT` | 1024 | FFT window length |
| `HOP_LENGTH` | 128 | Hop between frames |
| `PEAK_NEIGHBORHOOD_FREQ` | 20 bins | Peak isolation in frequency |
| `PEAK_NEIGHBORHOOD_TIME` | 20 frames | Peak isolation in time |
| `FAN_VALUE` | 15 | Peaks paired per anchor |
| `MAX_PAIR_DT_FRAMES` | 200 | Max Δt between paired peaks |
| `MIN_VOTES` | 5 | Minimum votes for a confident match |

---

## results.csv format (for batch auto-grading)

```csv
filename,prediction
clip1.mp3,Bohemian Rhapsody
clip2.wav,Hey Jude
clip3.mp3,(no match)
```

- `filename`: the uploaded filename exactly as provided
- `prediction`: matched song's filename **without extension**
