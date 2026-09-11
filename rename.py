import random
import shutil
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

IEEE14_DIR = Path("IEEE14_datasets")
ISLANDS_DIR = Path("islands")

OUTPUT_IEEE14 = IEEE14_DIR / "remixed_batch"
OUTPUT_ISLANDS = ISLANDS_DIR / "remixed_batch"

SAMPLE_FRACTION = 0.35

# Fixed seed = same random selection every time.
# Change it if you want a different selection.
RANDOM_SEED = 42


# ============================================================
# SETUP
# ============================================================

random.seed(RANDOM_SEED)

OUTPUT_IEEE14.mkdir(parents=True, exist_ok=True)
OUTPUT_ISLANDS.mkdir(parents=True, exist_ok=True)


# ============================================================
# FIND DATASETS
# ============================================================

dataset_folders = [
    p for p in IEEE14_DIR.iterdir()
    if p.is_dir() and p.name != "mixed_batch"
]

print(f"Found {len(dataset_folders)} datasets.")
print()


# ============================================================
# COLLECT 20% FROM EACH DATASET
# ============================================================

all_selected = []

for dataset_dir in sorted(dataset_folders):

    dataset_name = dataset_dir.name
    islands_dataset_dir = ISLANDS_DIR / dataset_name

    print("=" * 60)
    print(f"Processing: {dataset_name}")
    print("=" * 60)

    if not islands_dataset_dir.is_dir():
        print(f"WARNING: Missing islands folder: {islands_dataset_dir}")
        continue

    dataset_records = []

    # --------------------------------------------------------
    # Collect records from observable + unobservable
    # --------------------------------------------------------

    for category in ["observable", "unobservable"]:

        category_dir = dataset_dir / category

        if not category_dir.is_dir():
            print(f"WARNING: Missing {category} folder")
            continue

        # Get all record JSON files
        json_files = list(category_dir.glob("record_*.json"))

        for json_file in json_files:

            record_name = json_file.stem
            pkl_file = category_dir / f"{record_name}.pkl"

            # Both files must exist
            if not pkl_file.is_file():
                print(
                    f"WARNING: Missing PKL: "
                    f"{category}/{record_name}.pkl"
                )
                continue

            # Corresponding island directory
            island_dir = islands_dataset_dir / record_name

            if not island_dir.is_dir():
                print(
                    f"WARNING: Missing island: "
                    f"{dataset_name}/{record_name}"
                )
                continue

            dataset_records.append({
                "dataset": dataset_name,
                "category": category,
                "original_record": record_name,
                "json": json_file,
                "pkl": pkl_file,
                "island": island_dir,
            })

    # --------------------------------------------------------
    # Select 20% from THIS dataset
    # --------------------------------------------------------

    total = len(dataset_records)
    amount = round(total * SAMPLE_FRACTION)

    selected = random.sample(dataset_records, amount)

    print(f"Valid records: {total}")
    print(f"Selected:      {amount} (20%)")

    # Add them to the global selection
    all_selected.extend(selected)

    print()


# ============================================================
# SHUFFLE THE COMBINED SELECTION
# ============================================================

random.shuffle(all_selected)

print("=" * 60)
print(f"TOTAL SELECTED RECORDS: {len(all_selected)}")
print("=" * 60)
print()


# ============================================================
# COPY AND RENumber
# ============================================================

for new_number, record in enumerate(all_selected):

    new_record_name = f"record_{new_number}"

    category = record["category"]

    print(
        f"{record['dataset']}/"
        f"{category}/"
        f"{record['original_record']}"
        f"  ->  "
        f"{new_record_name}"
    )

    # --------------------------------------------------------
    # IEEE14 DATASET
    # --------------------------------------------------------

    output_category = OUTPUT_IEEE14 / category
    output_category.mkdir(
        parents=True,
        exist_ok=True
    )

    destination_json = (
        output_category /
        f"{new_record_name}.json"
    )

    destination_pkl = (
        output_category /
        f"{new_record_name}.pkl"
    )

    shutil.copy2(
        record["json"],
        destination_json
    )

    shutil.copy2(
        record["pkl"],
        destination_pkl
    )

    # --------------------------------------------------------
    # ISLAND
    # --------------------------------------------------------

    destination_island = (
        OUTPUT_ISLANDS /
        new_record_name
    )

    shutil.copytree(
        record["island"],
        destination_island
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("DONE")
print("=" * 60)

print(f"Total records in mixed_batch: {len(all_selected)}")

print()
print("IEEE14 output:")
print(f"  {OUTPUT_IEEE14 / 'observable'}")
print(f"  {OUTPUT_IEEE14 / 'unobservable'}")

print()
print("Islands output:")
print(f"  {OUTPUT_ISLANDS}")
print()