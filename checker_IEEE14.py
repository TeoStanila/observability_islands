import os
import sys
import json
from tqdm import tqdm
import pandapower as pp

from generation_IEEE14 import observability_analysis

if len(sys.argv) < 2:
    print("Specify dataset to check.")
    sys.exit()
check_dir = sys.argv[1]

observable_dir = os.path.join("IEEE14_datasets", check_dir, "observable")

incorrect = []

for filename in tqdm(sorted(os.listdir(observable_dir)), desc="Checking observability"):
    if not filename.endswith(".json"):
        continue

    path = os.path.join(observable_dir, filename)

    with open(path, "r") as f:
        record = json.load(f)

    net = pp.from_json_string(record["net_json"])
    result = observability_analysis(net)

    if not result.observable:
        incorrect.append((filename, result.rank_deficiency))

print(f"Incorrectly labeled observable: {len(incorrect)}")