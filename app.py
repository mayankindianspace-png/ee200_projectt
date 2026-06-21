import os
import sys
import io
import csv
import time
import pickle
import tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fingerprint import (
    load_audio_bytes, compute_spectrogram, get_peaks, generate_hashes,
    match_song, load_database, time_axis, freq_axis,
    TARGET_SR, HOP_LENGTH, N_FFT
)


st.set_page_config(
    page_title="EE200 Audio Fingerprinter",
    page_icon="🎵",
    layout="wide",
)

DB_PATH     = "fingerprint_db.pkl"
SINGLE_DB_PATH = "fingerprint_db_single.pkl"


@st.cache_resource(show_spinner="Loading fingerprint database…")
def get_db(path: str):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)




def fig_to_image(fig):

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf


def plot_spectrogram(spec_db: np.ndarray, title: str = "Spectrogram",
                     max_seconds: float = 30.0):
    t = time_axis(spec_db)
    f = freq_axis(spec_db)

    max_frame = min(spec_db.shape[1], int(max_seconds * TARGET_SR / HOP_LENGTH))
    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.pcolormesh(t[:max_frame], f, spec_db[:, :max_frame],
                       cmap='magma', shading='auto', vmin=-80, vmax=0)
    plt.colorbar(im, ax=ax, label='dB')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    ax.set_ylim([0, TARGET_SR / 2])
    plt.tight_layout()
    return fig_to_image(fig)


def plot_constellation(spec_db: np.ndarray, peaks: list,
                       title: str = "Constellation",
                       max_seconds: float = 30.0):
    t = time_axis(spec_db)
    f = freq_axis(spec_db)
    max_frame = int(max_seconds * TARGET_SR / HOP_LENGTH)


    pts = [(t[min(p[0], len(t)-1)], f[min(p[1], len(f)-1)])
           for p in peaks if p[0] < max_frame]
    pt_t = [p[0] for p in pts]
    pt_f = [p[1] for p in pts]

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.scatter(pt_t, pt_f, s=3, c='cyan', alpha=0.8, linewidths=0)
    ax.set_facecolor('black')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"{title}  ({len(pts)} peaks shown)")
    ax.set_xlim([0, min(t[-1], max_seconds)])
    ax.set_ylim([0, TARGET_SR / 2])
    plt.tight_layout()
    return fig_to_image(fig)


def plot_offset_histogram(result: dict, top_n: int = 8):
    offset_hist = result['offset_hist']
    all_votes   = result['all_votes']


    top_songs = sorted(all_votes.items(), key=lambda x: x[1], reverse=True)[:top_n]
    best_song = result['match'] or (top_songs[0][0] if top_songs else "")

    n_cols = min(4, len(top_songs))
    n_rows = (len(top_songs) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 2.5 * n_rows))
    axes = np.array(axes).flatten()

    for idx, (song, votes) in enumerate(top_songs):
        ax = axes[idx]
        if song in offset_hist:
            offs = list(offset_hist[song].keys())
            cnts = list(offset_hist[song].values())
            color = 'limegreen' if song == best_song else 'steelblue'
            ax.bar(offs, cnts, width=1, color=color)
        ax.set_title(f"{song[:28]}\n{votes} votes",
                     fontsize=7,
                     color='green' if song == best_song else 'black',
                     fontweight='bold' if song == best_song else 'normal')
        ax.set_xlabel("Offset (frames)", fontsize=6)
        ax.tick_params(labelsize=6)


    for idx in range(len(top_songs), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Offset Histogram (peak = aligned match)", fontsize=9)
    plt.tight_layout()
    return fig_to_image(fig)


def plot_all_votes(all_votes: dict, match: str):
    top = sorted(all_votes.items(), key=lambda x: x[1], reverse=True)[:15]
    names = [n[:30] for n, _ in top]
    votes = [v for _, v in top]
    colors = ['limegreen' if n == match else 'steelblue' for n in names]

    fig, ax = plt.subplots(figsize=(8, max(3, len(top) * 0.35)))
    ax.barh(names[::-1], votes[::-1], color=colors[::-1], edgecolor='k', lw=0.3)
    ax.set_xlabel("Max aligned votes")
    ax.set_title("Top candidates (green = identified song)")
    plt.tight_layout()
    return fig_to_image(fig)



def identify(audio_bytes: bytes, db: dict, filename: str = "query") -> dict:

    y = load_audio_bytes(audio_bytes)
    spec = compute_spectrogram(y)
    peaks = get_peaks(spec)
    result = match_song(y, db)
    return dict(y=y, spec=spec, peaks=peaks, result=result, filename=filename)



st.title("EE200 Audio Fingerprinter")
st.markdown(
    """
    Upload a short audio clip and the app will identify the song by matching its
    spectrogram fingerprint against the indexed database.
    """)


db = get_db(DB_PATH)

if db is None:
    st.error(
        f"❌ Database not found at **{DB_PATH}**.\n\n"
        "Please run the indexer first:\n"
        "```\npython index_songs.py --songs_dir songs/\n```\n"
        "then restart this app."
    )
    st.stop()

song_list = db.get('__song_list__', [])
st.sidebar.success(f"✅ Database loaded\n\n**{len(song_list)} songs** indexed")
with st.sidebar.expander("Indexed songs"):
    for s in sorted(song_list):
        st.write(f"• {s}")


tab_single, tab_batch = st.tabs(["🎯 Single Clip", "📦 Batch Mode"])

with tab_single:
    st.header("Single-clip identification")
    st.markdown(
        "Upload any audio clip (MP3, WAV, etc.).  "
        "The app will show the spectrogram, constellation, "
        "offset histogram and the matched song."
    )

    uploaded = st.file_uploader(
        "Upload a query audio clip",
        type=['mp3', 'wav', 'flac', 'ogg', 'm4a'],
        key="single_upload"
    )

    if uploaded is not None:
        audio_bytes = uploaded.read()
        st.audio(audio_bytes, format=uploaded.type)

        with st.spinner("Fingerprinting and matching…"):
            t0 = time.time()
            info = identify(audio_bytes, db, filename=uploaded.name)
            elapsed = time.time() - t0

        result = info['result']
        match  = result['match']
        votes  = result['votes']

        if match:
            st.success(f"### 🎶 Identified: **{match}**\n"
                       f"Aligned votes: **{votes}**  |  "
                       f"Query hashes: {result['query_hashes']}  |  "
                       f"Time: {elapsed:.2f} s")
        else:
            st.warning(
                f"### ❓ No confident match found\n"
                f"Best candidate: **{max(result['all_votes'], key=result['all_votes'].get, default='—')}** "
                f"({votes} votes) — below the minimum threshold.\n\n"
                "Try a longer clip or a clip with less background noise."
            )

        st.divider()
        st.subheader("Intermediate steps")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Spectrogram** (first 30 s)")
            st.image(plot_spectrogram(info['spec'],
                                      title=f"Spectrogram – {uploaded.name}"))

        with col2:
            st.markdown("**Constellation** (local maxima)")
            st.image(plot_constellation(info['spec'], info['peaks'],
                                        title="Constellation"))

        st.markdown("**Offset histogram** – true match shows sharp spike at one offset")
        st.image(plot_offset_histogram(result))

        st.markdown("**All candidate votes**")
        st.image(plot_all_votes(result['all_votes'], match or ""))

        with st.expander("Raw vote table"):
            top_all = sorted(result['all_votes'].items(),
                             key=lambda x: x[1], reverse=True)
            st.table({"Song": [n for n, _ in top_all],
                      "Max aligned votes": [v for _, v in top_all]})


with tab_batch:
    st.header("Batch mode")
    st.markdown(
        "Upload multiple audio clips at once.  "
        "The app processes each file and produces a downloadable **results.csv** "
        "with exactly two columns: `filename`, `prediction`."
    )
    st.info(
        "`prediction` is the matched song's **filename without extension** "
        "(e.g. `Bohemian Rhapsody`), matching the format required for auto-grading."
    )

    batch_files = st.file_uploader(
        "Upload query clips (multiple allowed)",
        type=['mp3', 'wav', 'flac', 'ogg', 'm4a'],
        accept_multiple_files=True,
        key="batch_upload"
    )

    if batch_files:
        if st.button("▶ Run batch identification"):
            results_rows = []
            progress = st.progress(0.0, text="Processing…")
            status_box = st.empty()
            table_placeholder = st.empty()

            for i, f in enumerate(batch_files):
                progress.progress((i + 1) / len(batch_files),
                                  text=f"Processing {f.name}  ({i+1}/{len(batch_files)})")
                audio_bytes = f.read()
                info = identify(audio_bytes, db, filename=f.name)
                match = info['result']['match'] or "(no match)"
                results_rows.append({
                    "filename": f.name,
                    "prediction": match
                })

                table_placeholder.table(results_rows)

            progress.empty()
            st.success(f"✅ Done – {len(results_rows)} files processed.")


            csv_buf = io.StringIO()
            writer  = csv.DictWriter(csv_buf, fieldnames=["filename", "prediction"])
            writer.writeheader()
            writer.writerows(results_rows)
            csv_str = csv_buf.getvalue()

            st.download_button(
                label="⬇ Download results.csv",
                data=csv_str,
                file_name="results.csv",
                mime="text/csv",
            )

            st.divider()
            st.subheader("Results")
            st.dataframe(results_rows, use_container_width=True)



st.divider()
st.caption(
    "EE200: Signals, Systems and Networks – Course Project Q3B  |  "
    "Algorithm: STFT → constellation → paired-hash fingerprinting"
)
