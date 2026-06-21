

import argparse
import time
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fingerprint import build_database, build_database_single_peaks


def main():
    parser = argparse.ArgumentParser(
        description="Index songs into a fingerprint database.")
    parser.add_argument(
        "--songs_dir",
        type=str,
        default="songs",
        help="Path to folder containing MP3/WAV song files (default: ./songs)")
    parser.add_argument(
        "--db_path",
        type=str,
        default="fingerprint_db.pkl",
        help="Output path for the paired-hash database (default: fingerprint_db.pkl)")
    parser.add_argument(
        "--single_db_path",
        type=str,
        default="fingerprint_db_single.pkl",
        help="Output path for the single-peak database (default: fingerprint_db_single.pkl)")
    parser.add_argument(
        "--skip_single",
        action="store_true",
        help="Skip building the single-peak database (saves time if not needed)")
    args = parser.parse_args()

    if not os.path.isdir(args.songs_dir):
        print(f"ERROR: songs_dir '{args.songs_dir}' does not exist.")
        sys.exit(1)


    print("=" * 60)
    print("  STEP 1: Building paired-hash fingerprint database")
    print("=" * 60)
    t0 = time.time()
    db = build_database(args.songs_dir, db_path=args.db_path, verbose=True)
    elapsed = time.time() - t0
    print(f"\nPaired-hash indexing took {elapsed:.1f} s")

    song_list = db.get('__song_list__', [])
    print(f"\nIndexed {len(song_list)} songs:")
    for s in song_list:
        print(f"  • {s}")


    if not args.skip_single:
        print()
        print("=" * 60)
        print("  STEP 2: Building single-peak fingerprint database")
        print("  (used in Q3A to compare paired vs single-peak matching)")
        print("=" * 60)
        t0 = time.time()
        build_database_single_peaks(
            args.songs_dir,
            db_path=args.single_db_path,
            verbose=True)
        elapsed = time.time() - t0
        print(f"\nSingle-peak indexing took {elapsed:.1f} s")

    print()
    print("=" * 60)
    print("  ALL DONE")
    print(f"  Paired-hash DB  : {args.db_path}")
    if not args.skip_single:
        print(f"  Single-peak DB  : {args.single_db_path}")
    print()
    print("  Next steps:")
    print("  • For Q3A analysis:  run  python q3a_analysis.py")
    print("  • For Q3B Streamlit: run  streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
