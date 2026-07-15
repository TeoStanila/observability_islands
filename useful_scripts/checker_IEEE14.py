import os
import sys
import argparse
import json
from tqdm import tqdm
import pandapower as pp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from generation_IEEE14 import observability_analysis

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Checker for unobservable records to be truly unobservable")
    parser.add_argument("--dataset_dir", required=True, type=str, help="Dataset directory to check")
    args = parser.parse_args()

    check_dir = args.dataset_dir

    unobservable_dir = os.path.join("IEEE14_datasets", check_dir, "unobservable")

    incorrect = []

    for filename in tqdm(sorted(os.listdir(unobservable_dir)), desc="Checking unobservability"):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(unobservable_dir, filename)

        with open(path, "r") as f:
            record = json.load(f)

        net = pp.from_json_string(record["net_json"])
        result = observability_analysis(net)

        if result.observable:
            incorrect.append((filename))

    print(f"Incorrectly labeled unobservable: {len(incorrect)}")