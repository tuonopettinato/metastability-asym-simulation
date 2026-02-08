#!/usr/bin/env python3

import os
import shutil
from parameters import N, multiple_dir_name

def delete_simulation_files(idx: int):
    """
    Deletes the following files for a given simulation number idx:
    - plots/plot_pattern_overlaps_{idx}.png
    - firing_rates/firing_rates_{idx}.npy
    - npy/ou_process/ou_process_{idx}.npy
    """
    base_dir = os.path.join(os.path.dirname(__file__), "..", f"{multiple_dir_name}_{N}", "npy")
    firing_dir = os.path.join(base_dir, "firing_rates")
    ou_dir = os.path.join(base_dir, "ou_process")
    plots_dir = os.path.join(base_dir, "..", "plots")

    files_to_delete = [
        os.path.join(plots_dir, f"pattern_overlaps_{idx}.png"),
        os.path.join(firing_dir, f"firing_rates_{idx}.npy"),
        os.path.join(ou_dir, f"ou_process_{idx}.npy")
    ]

    for fpath in files_to_delete:
        if os.path.exists(fpath):
            os.remove(fpath)
            print(f"Deleted: {fpath}")
        else:
            print(f"File not found, skipping: {fpath}")

if __name__ == "__main__":
    number = int(input("Enter the simulation number to delete: "))
    delete_simulation_files(number)
