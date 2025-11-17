import numpy as np
import os
import time
import gc
import logging
import glob  # <-- Added this import
from collections import defaultdict  # <-- Added this import
from typing import Dict, Optional, Tuple, Any, List

def _load_npz( path: str) -> Dict[str, np.ndarray]:
    """
    Robust loader:
    - Retries a few times on 'Too many open files' (Errno 23/24)
    - Copies arrays out of the NPZ file so no file descriptor is kept alive.
    """
    needed = [
        "image", "image_mask",
        "faces", "faces_mask",
        "obj", "obj_mask",
        "article", "article_mask",
        "caption_ids",
    ]

    max_retries = 5
    for attempt in range(max_retries):
        try:
            with np.load(path, allow_pickle=False) as z:
                # Check required keys
                for n in needed:
                    if n not in z.files:
                        raise KeyError(f"{os.path.basename(path)} missing '{n}'")

                if z["caption_ids"].size == 0:
                    raise ValueError(
                        f"{os.path.basename(path)} has empty caption_ids; "
                        "re-run precompute with tokenizer."
                    )

                # IMPORTANT: np.array(..., copy=True) ensures no lazy memmap
                out = {k: np.array(z[k], copy=True) for k in needed}
            # File is closed here because of the context manager
            return out

        except OSError as e:
            # 23 = ENFILE (Too many open files in system)
            # 24 = EMFILE (Too many open files for this process)
            if e.errno in (23, 24):
                logging.warning(
                    f"[WARN] OSError {e.errno} when opening {path} "
                    f"(attempt {attempt+1}/{max_retries}): {e}. "
                    "Forcing GC and retrying..."
                )
                gc.collect()
                time.sleep(0.5)  # small backoff
                continue
            # Other OS errors → re-raise
            raise
        except (KeyError, ValueError) as e:
            # Re-raise data validation errors immediately
            print(f"[ERROR] Data validation failed for {path}: {e}")
            raise

    # If we get here, all attempts failed
    raise OSError(
        f"Failed to open {path} after {max_retries} retries "
        "due to 'Too many open files'."
    )

# --- Your original test code ---
try:
    temp = _load_npz("/datastore/npl/ICEK/TnT/new_dataset/train/50000.npz")
    print("--- [DEBUG] Printing array shapes for 50000.npz ---")
    for key, array in temp.items():
        print(f"{key}: {array.shape}")
    print("-----------------------------------")
except Exception as e:
    print(f"Could not load 50000.npz: {e}\n")


# --- NEW FUNCTION TO CHECK ALL FILES ---
def check_directory_shapes(data_dir: str):
    """
    Loops through all .npz files in a directory and reports on
    which tensor keys have consistent vs. variable shapes.
    """
    print(f"\n--- Starting Shape Analysis for: {data_dir} ---")
    
    # Use defaultdict to store a set of shapes for each key
    all_shapes = defaultdict(set)
    error_files = []

    npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
    if not npz_files:
        print("No .npz files found.")
        return

    total_files = len(npz_files)
    print(f"Found {total_files} files. Starting check...")

    for i, file_path in enumerate(npz_files, 1):
        try:
            # Load the file using your robust loader
            data = _load_npz(file_path)
            
            # Add the shape of each array to its corresponding set
            for key, array in data.items():
                all_shapes[key].add(array.shape)
                
        except Exception as e:
            # If _load_npz fails (e.g., missing key, empty caption), log it
            error_files.append(f"{os.path.basename(file_path)}: {e}")
        
        # Print progress
        if i % 100 == 0 or i == total_files:
            print(f"Processed {i}/{total_files} files...")

    print("\n--- Shape Analysis Complete ---")
    
    # Report results
    for key in sorted(all_shapes.keys()):
        shapes = all_shapes[key]
        if len(shapes) == 1:
            print(f"✅ {key}: CONSISTENT - {list(shapes)[0]}")
        else:
            print(f"⚠️ {key}: VARIABLE - Found {len(shapes)} different shapes:")
            # Sort shapes to make the output readable
            for shape in sorted(list(shapes), key=str): 
                print(f"   {shape}")

    if error_files:
        print(f"\n--- Found {len(error_files)} Files With Errors ---")
        for err in error_files:
            print(err)

# --- RUN THE CHECK ---
# !!! IMPORTANT: Change this path to your directory !!!
train_dir = "/datastore/npl/ICEK/TnT/new_dataset/train"
check_directory_shapes(train_dir)
