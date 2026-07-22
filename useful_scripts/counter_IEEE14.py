import argparse
import os
from collections import Counter

from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count islands in dataset records")
    parser.add_argument("--dataset_dir", required=True, type=str, help="Dataset directory to check")
    args = parser.parse_args()

    check_dir = args.dataset_dir

    unobservable_dir = os.path.join("islands", check_dir)

    island_counts = Counter()

    for filename in tqdm(sorted(os.listdir(unobservable_dir)), desc="Counting islands"):
        record_dir = os.path.join(unobservable_dir, filename)

        if not os.path.isdir(record_dir):
            continue

        num_islands = sum(
            1 for item in os.listdir(record_dir)
            if os.path.isdir(os.path.join(record_dir, item)) and item.startswith("island_")
        )

        island_counts[num_islands] += 1

    for num_islands in sorted(island_counts):
        print(f"{num_islands} islands: {island_counts[num_islands]} files")