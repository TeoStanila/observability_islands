import os
import sys
import json
from tqdm import tqdm
import pandapower as pp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from generation_IEEE14 import observability_analysis


if len(sys.argv) < 2:
    print("Specify dataset to check.")
    sys.exit()
check_dir = sys.argv[1]

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