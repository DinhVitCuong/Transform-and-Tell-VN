
import os, random, torch
from torch.utils.data import DataLoader
from main import NewsCaptionDataset   

class _DummyTokenizer:
    def encode(self, text, return_tensors="pt", truncation=True, max_length=512):
        ids = torch.randint(5, 30000, (min(len(text.split()), max_length),), dtype=torch.long)
        return ids.unsqueeze(0) if return_tensors == "pt" else ids

def summarize_tensor(t: torch.Tensor, name: str):
    t = t.detach()
    return f"{name}: shape={tuple(t.shape)}, dtype={t.dtype}, min={t.min():.3f}, max={t.max():.3f}, mean={t.mean():.3f}"

def print_sample(sample, idx, split):
    print(f"\n[{split}] idx={idx}  caption='{sample['caption']}'")
    print(" image_path:", sample["image_path"])
    print(" ", summarize_tensor(sample["image"], "image"))
    ctx = sample["contexts"]
    for k, v in ctx.items():
        if isinstance(v, torch.Tensor):
            print(" ", summarize_tensor(v, f"contexts.{k}"))
        else:
            print(f" contexts.{k}: {type(v)} -> {v}")
    print(" ", summarize_tensor(sample["caption_ids"], "caption_ids"))

def inspect_split(data_dir, h5_dir, split, k=10, mode="first", models=None):

    if models is None:
        models = {"tokenizer": _DummyTokenizer()}

    ds = NewsCaptionDataset(
        data_dir=data_dir,
        split=split,
        models=models,
        data_pt_dir=h5_dir,     # folder containing {split}_features.h5
        max_length=512,
    )

    n = len(ds)
    k = min(k, n)
    if mode == "random":
        indices = random.sample(range(n), k)
    else:
        indices = list(range(k))

    for i, idx in enumerate(indices):
        sample = ds[idx] 
        print_sample(sample, idx, split)

    ds.close()

if __name__ == "__main__":
    # CHANGE THESE PATHS:
    DATA_DIR = "/path/to/jsons"   # contains train.json / val.json / test.json
    H5_DIR   = "/path/to/h5"      # contains train_features.h5 / val_features.h5 / test_features.h5

    # If you have a real tokenizer, e.g.:
    # from transformers import AutoTokenizer
    # tok = AutoTokenizer.from_pretrained("vinai/phobert-base")
    # MODELS = {"tokenizer": tok}
    MODELS = None  # falls back to dummy tokenizer

    # First 10 of train, random 10 of val, first 5 of test:
    inspect_split(DATA_DIR, H5_DIR, "train", k=10, mode="first",  models=MODELS)
    inspect_split(DATA_DIR, H5_DIR, "val",   k=10, mode="random", models=MODELS)
    inspect_split(DATA_DIR, H5_DIR, "test",  k=5,  mode="first",  models=MODELS)
