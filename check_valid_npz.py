import numpy as np
import os
import time
import gc
import logging
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

        # If we get here, all attempts failed
        raise OSError(
            f"Failed to open {path} after {max_retries} retries "
            "due to 'Too many open files'."
        )


temp = _load_npz("/datastore/npl/ICEK/TnT/new_dataset/train/4000.npz")
print(temp)