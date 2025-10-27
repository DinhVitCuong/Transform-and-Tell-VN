# precompute_npz.py — save .npz using JSON keys
# Usage:
#   python precompute_npz.py \
#     --data-dir /path/to/dataset \
#     --image-root /path/to/images_resized \
#     --vncorenlp-path /path/to/VnCoreNLP \
#     --splits train val \
#     --out /path/to/cache_npz \
#     --device cuda:0 --max-faces 4 --max-objects 64 --compress

import os
import sys
import re
import json
import argparse
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.encoder import (
    setup_models, segment_text, detect_faces, detect_objects, image_feature
)

# -------- helpers --------
def _save_npz(path: str, compress: bool, **arrays: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if compress:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)

def _to_numpy(t: torch.Tensor, dtype=None) -> np.ndarray:
    if isinstance(t, torch.Tensor):
        t = t.detach().to("cpu").contiguous()
        if dtype is not None:
            t = t.to(dtype)
        return t.numpy()
    raise TypeError("Expected torch.Tensor")

def _sanitize_key(k: str) -> str:
    # keep only [A-Za-z0-9_-], replace others with '_'
    k = str(k)
    return re.sub(r"[^A-Za-z0-9_-]", "_", k)

def _resolve_items_with_keys(path: str) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Returns list of (key_str, item) in stable order:
    - dict: sorted by numeric key if possible, else lexicographic
    - list: key is the positional index as string
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [(str(i), data[i]) for i in range(len(data))]

    if isinstance(data, dict):
        def _key_sort(k):
            try:    return (0, int(k))
            except: return (1, str(k))
        keys_sorted = sorted(list(data.keys()), key=_key_sort)
        return [(str(k), data[k]) for k in keys_sorted]

    raise ValueError(f"Unsupported JSON at {path}: {type(data)}")

# -------- main split precompute --------
def precompute_split(
    split: str,
    data_dir: str,
    image_root: str,
    out_root: str,
    models: Dict[str, Any],
    max_faces: int,
    max_objects: int,
    compress: bool,
    overwrite: bool,
    max_len: int,
) -> None:
    index_path = os.path.join(data_dir, f"{split}.json")
    items_kv = _resolve_items_with_keys(index_path)
    out_dir = os.path.join(out_root, split)
    os.makedirs(out_dir, exist_ok=True)

    device = models["device"]
    tokenizer = models.get("tokenizer", None)
    embedder  = models["embedder"]
    vncore    = models["vncore"]
    mtcnn     = models["mtcnn"]
    facenet   = models["facenet"]
    yolo      = models["yolo"]
    resnet    = models["resnet"]
    resnet_obj= models["resnet_object"]
    preproc   = models["preprocess"]

    torch.set_grad_enabled(False)
    start = time.time()
    ok, fail = 0, 0
    total = len(items_kv)

    for i, (raw_key, item) in enumerate(items_kv, 1):
        key = _sanitize_key(raw_key)
        out_path = os.path.join(out_dir, f"{key}.npz")

        if (not overwrite) and os.path.exists(out_path):
            ok += 1
            if i % 100 == 0 or i == total:
                print(f"[{split}] {i}/{total} (skip existing) ✓")
            continue

        try:
            # ---- text ----
            segmented_context: List[str] = []
            for sentence in item.get("paragraphs", []):
                segmented_context.append(segment_text(sentence, vncore))
            caption_raw = item.get("caption", "")
            caption_seg = segment_text(caption_raw, vncore)

            # ---- image path (adapt to your dataset layout) ----
            if "image_path" in item:
                if "images/" in item["image_path"]:
                    img_id = item["image_path"].split("images/", 1)[-1]
                else:
                    img_id = item["image_path"]
            else:
                raise KeyError(f"Item with key {raw_key} missing 'image_path'")
            img_path = os.path.join(image_root, img_id)

            with Image.open(img_path).convert("RGB") as im:
                im = im.copy()

            # ---- vision compute ----
            img_feats, img_mask = image_feature(im, resnet, preproc, device)              # [49,2048], [49]
            face_feats, face_mask = detect_faces(im, mtcnn, facenet, device,
                                                 max_faces=max_faces, pad_to=max_faces)   # [F,512], [F]
            obj_feats, obj_mask = detect_objects(img_path, yolo, resnet_obj, preproc, device,
                                                 max_det=max_objects, pad_to=max_objects) # [O,2048], [O]

            # ---- article encode (final layer) ----
            try:
                feats_4d, attn_mask = embedder(segmented_context)    # [1,L,S,H], [1,S]
            except Exception:
                feats_4d, attn_mask = embedder([segmented_context])   # fallback
            final_layer = feats_4d[0, -1].contiguous()                # [S,1024]
            article_pad_mask = (~attn_mask[0].bool()).contiguous()    # [S], True=PAD

            # ---- caption ids (optional) ----
            if tokenizer is not None:
                cap_ids = tokenizer.encode(
                    caption_seg, return_tensors="pt",
                    truncation=True, max_length=max_len
                )[0]
                cap_ids_np = _to_numpy(cap_ids, dtype=torch.long).astype(np.int64)
            else:
                cap_ids_np = np.array([], dtype=np.int64)

            arrays = {
                "image":        _to_numpy(img_feats, dtype=torch.float32).astype(np.float32),
                "image_mask":   _to_numpy(img_mask,  dtype=torch.bool).astype(np.bool_),
                "faces":        _to_numpy(face_feats, dtype=torch.float32).astype(np.float32),
                "faces_mask":   _to_numpy(face_mask,  dtype=torch.bool).astype(np.bool_),
                "obj":          _to_numpy(obj_feats,  dtype=torch.float32).astype(np.float32),
                "obj_mask":     _to_numpy(obj_mask,   dtype=torch.bool).astype(np.bool_),
                "article":      _to_numpy(final_layer, dtype=torch.float16).astype(np.float16),
                "article_mask": _to_numpy(article_pad_mask, dtype=torch.bool).astype(np.bool_),
                "caption_ids":  cap_ids_np,
            }

            _save_npz(out_path, compress, **arrays)
            ok += 1

            if i % 100 == 0 or i == total:
                elapsed = time.time() - start
                print(f"[{split}] {i}/{total} ✓ ok={ok} fail={fail} elapsed={elapsed/60:.1f}m")

        except Exception as e:
            fail += 1
            print(f"[{split}] key={raw_key} FAILED: {e}")

    print(f"[{split}] DONE. ok={ok} fail={fail} -> {out_dir}")

def main():
    p = argparse.ArgumentParser("Precompute visual+text features to .npz by JSON key")
    p.add_argument("--data-dir", default="/data2/npl/ICEK/Wikipedia/content/ver4")
    p.add_argument("--image-root", default="/data2/npl/ICEK/Wikipedia/images_resized")
    p.add_argument("--vncorenlp-path", default="/data2/npl/ICEK/VnCoreNLP")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--out", default="/data2/npl/ICEK/TnT/dataset")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-faces", type=int, default=4)
    p.add_argument("--max-objects", type=int, default=64)
    p.add_argument("--compress", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-len", type=int, default=256)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    try:
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    models = setup_models(device, args.vncorenlp_path)
    models["device"] = device

    for split in args.splits:
        precompute_split(
            split=split,
            data_dir=args.data_dir,
            image_root=args.image_root,
            out_root=args.out,
            models=models,
            max_faces=args.max_faces,
            max_objects=args.max_objects,
            compress=args.compress,
            overwrite=args.overwrite,
            max_len=args.max_len,
        )

if __name__ == "__main__":
    main()
