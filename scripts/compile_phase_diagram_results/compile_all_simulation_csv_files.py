# This script is to compile all the results of G-alpha pairs for example (or other parameter sweeps)
# into a single csv file which allows us to plot phase diagrams.
# Usually I would run this on the cluster where I performed the simulations.
# Change BASE_DIR to where the folder with sweep results is located.
import os
import pandas as pd

def collect_persistent_stats(base_dir, output_csv):
    """
    Walk through base_dir, find every data/persistent_activity_stats.csv,
    read it (assumed one row), tag it with its run path, and
    concatenate all into output_csv.
    """
    records = []
    for root, dirs, files in os.walk(base_dir):
        # only enter folders named "data"
        if os.path.basename(root) != "data":
            continue

        csv_path = os.path.join(root, "persistent_activity_stats.csv")
        if not os.path.isfile(csv_path):
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Skipping {csv_path}: could not read ({e})")
            continue

        if df.shape[0] != 1:
            print(f"Warning: {csv_path} has {df.shape[0]} rows (expected 1)")

        # record which run this came from, relative to base_dir
        rel_run = os.path.relpath(root, base_dir)
        df["run_path"] = rel_run

        records.append(df)

    if not records:
        print("No persistent_activity_stats.csv files found.")
        return

    all_df = pd.concat(records, ignore_index=True)

    # ensure consistent column order
    cols = [c for c in all_df.columns if c != "run_path"] + ["run_path"]
    all_df = all_df[cols]

    all_df.to_csv(output_csv, index=False)
    print(f"Combined {len(records)} runs into {output_csv}")

if __name__ == "__main__":
    BASE_DIR = "../../models/model_pet_nmda/results/g_alpha_random_seeds_per_G_alpha_corrected_spont_fun"
    OUTPUT_CSV = os.path.join(BASE_DIR, "persistent_activity_stats_all_runs.csv")
    collect_persistent_stats(BASE_DIR, OUTPUT_CSV)
