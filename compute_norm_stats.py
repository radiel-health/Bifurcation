"""Quick script to compute normalization statistics from already-processed data."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from Preprocessing.pre_process import compute_normalization_stats

if __name__ == "__main__":
    print("Computing normalization statistics...")
    compute_normalization_stats()
    print("Done!")
