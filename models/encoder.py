"""Convert raw ViWiki JSON files into the format consumed by
``ViWiki_face_ner_match.py`` and additionally compute the image and text
features described in that reader's documentation.

The input directory must contain ``train.json``, ``val.json`` and ``test.json``
with entries of the form::

    {
        "0": {
            "image_path": "/path/to/img.jpg",
            "paragraphs": [...],
            "scores": [...],
            "caption": "caption text",
            "context": ["sentence 1", "sentence 2", ...]
        },
        ...
    }

For each item this script will:

* copy the image to ``--image-out`` (if provided)
* detect faces with **MTCNN** and extract **FaceNet** embeddings with max faces = 4
* detect objects using **YoloV8** and encode them with **ResNet152** We filter out objects with a confidence less than 0.3 and select up to 64 objects
* run **VnCoreNLP** to for segment
* extract a global image feature with **ResNet152**
* embed the article text using a **RoBERTa** model (PhoBERT for Vietnamese)

The resulting ``splits.json``, ``articles.json`` and ``objects.json`` match the
schema in :mod:`tell.data.dataset_readers.ViWiki_face_ner_match`.  ``splits``
contains the face embeddings and global image features while object features are
stored in ``objects.json``.
"""

import argparse
import json
import os
import shutil
from typing import Dict, Tuple, List
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict
import re

import cv2
import numpy as np
import torch
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import py_vncorenlp
from transformers import AutoTokenizer, AutoModel
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision.models import resnet152, ResNet152_Weights
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from ultralytics import YOLO
import re
import logging
import h5py
import uuid
from typing import Optional, Tuple, List

Image.MAX_IMAGE_PIXELS = None
SHARD_THRESHOLD = 3000
SHARD_SIZE = 2000

def _rel_or_abs(path: str, base_dir: str, use_abs: bool = False) -> str:
    return path if (use_abs or os.path.isabs(path)) else os.path.relpath(path, start=base_dir)

# Set up logging
logging.basicConfig(
    filename="preprocess_data.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

class RobertaEmbedder(torch.nn.Module):

    def __init__(self, model, tokenizer, device, expected_layers: int = 25):
        super().__init__()
        self.model = model.eval()
        self.tok = tokenizer
        self.device = device
        self.expected_layers = int(expected_layers)   # 25 for *-large, 13 for *-base
        self.target_len = 256

        # quick sanity
        h = int(getattr(self.model.config, "hidden_size", 0))
        if h <= 0:
            raise ValueError("model.config.hidden_size is invalid; load a proper (PhoBERT/RoBERTa) checkpoint.")

    @torch.no_grad()
    def forward(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        proc = []
        for t in texts:
            t = (t or "").replace("<SEP>", " ")
            if not t.startswith(" "):
                t = " " + t
            proc.append(t)
        batch = self.tok(
            proc,
            padding="max_length",
            truncation=True,
            max_length=self.target_len,   
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = batch["input_ids"].to(self.device).long()
        attn_mask = batch["attention_mask"].to(self.device).long()  # keep 0/1 int

        # Hard guard (keeps CUDA from device-side assert later)
        vocab = int(self.model.config.vocab_size)
        mn, mx = int(input_ids.min().item()), int(input_ids.max().item())
        if mn < 0 or mx >= vocab:
            raise ValueError(f"OOR token id: min={mn}, max={mx}, vocab_size={vocab}")

        out = self.model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            output_hidden_states=True,
        )

        hs = torch.stack(list(out.hidden_states), dim=1).contiguous()  # [B, L_actual, 512, H]
        L_actual = hs.size(1)
        if L_actual < self.expected_layers:
            pad = self.expected_layers - L_actual
            hs = torch.cat([hs, hs.new_zeros(hs.size(0), pad, hs.size(2), hs.size(3))], dim=1)
        elif L_actual > self.expected_layers:
            hs = hs[:, -self.expected_layers:, :, :]

        attn_mask_bool = attn_mask.to(torch.bool)  # True=token, False=pad (flip later if needed)
        return hs, attn_mask_bool

def setup_models(device: torch.device, vncorenlp_path="/datastore/npl/ICEK/VnCoreNLP"):
    # (Optional) make HF strictly local if all files are on disk
    # os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    print("[DEBUG] SETTING UP MODELS")
    # --- VnCoreNLP ---
    py_vncorenlp.download_model(save_dir=vncorenlp_path)
    vncore = py_vncorenlp.VnCoreNLP(
        annotators=["wseg"],
        save_dir=vncorenlp_path,
        max_heap_size='-Xmx15g'
    )
    print("[DEBUG] LOADED VNCORENLP!")

    # --- Face detection + embedding ---
    mtcnn = MTCNN(keep_all=True, device=str(device))
    facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    print("[DEBUG] LOADED FaceNet!")

    # --- Global / object image features ---
    weights = ResNet152_Weights.IMAGENET1K_V1
    base = resnet152(weights=weights).eval().to(device)
    resnet = nn.Sequential(*list(base.children())[:-2]).eval().to(device)
    resnet_object = nn.Sequential(*list(base.children())[:-1]).eval().to(device)
    print("[DEBUG] LOADED ResNet152!")

    # --- YOLOv8 ---
    yolo = YOLO("yolov8m.pt")
    yolo.fuse()
    print("[DEBUG] LOADED YOLOv8!")

    # --- PhoBERT ---
    phoBERTlocal = "/datastore/npl/ICEK/TnT/phoBERT_large/phobert-large"
    tokenizer = AutoTokenizer.from_pretrained(phoBERTlocal, use_fast=False, local_files_only=True)
    # DO NOT force safetensors unless files exist
    roberta   = AutoModel    .from_pretrained(phoBERTlocal, local_files_only=True).eval().to(device)

    # Quick consistency checks (fail fast in Python instead of CUDA assert)
    assert len(tokenizer) == int(roberta.config.vocab_size), \
        f"Tokenizer size {len(tokenizer)} != model vocab_size {int(roberta.config.vocab_size)}"
    assert tokenizer.pad_token_id is not None and tokenizer.pad_token == "<pad>", \
        "PhoBERT pad token missing/misaligned."

    # (Optional) If you saw SDPA-related dtype issues, prefer math path:
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

    embedder = RobertaEmbedder(roberta, tokenizer, device).to(device)
    print("[DEBUG] LOADED phoBERT!")

    preprocess = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])

    return {
        "vncore": vncore,
        "mtcnn": mtcnn,
        "facenet": facenet,
        "resnet": resnet,
        "resnet_object": resnet_object,
        "roberta": roberta,
        "tokenizer": tokenizer,
        "embedder": embedder,
        "yolo": yolo,
        "preprocess": preprocess,
        "device": device,
    }


def segment_text(text: str, model) -> str:
    """Segment text using VnCoreNLP and join sentences with a separator"""
    sentences = re.split(r'(?<=[\.!?])\s+', text.strip())
    if not sentences:
        return ""
    segmented_sentences = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        try:
            segmented = model.word_segment(sent)[0]
            segmented_sentences.append(segmented)
        except Exception as e:
            logging.error(f"Error segmenting text: {e}")
            segmented_sentences.append(sent)
    return "<SEP>".join(segmented_sentences)

def _pad_to_len(x: torch.Tensor, target_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x: [S, D] on any device; returns (x_padded [target_len, D], mask [target_len] with True=PAD)
    If S >= target_len -> truncate and mask all False.
    """
    S, D = x.shape
    if S >= target_len:
        return x[:target_len], torch.zeros(target_len, dtype=torch.bool, device=x.device)
    pad = torch.zeros(target_len - S, D, dtype=x.dtype, device=x.device)
    out = torch.cat([x, pad], dim=0)
    mask = torch.zeros(target_len, dtype=torch.bool, device=x.device)
    mask[S:] = True
    return out, mask

def detect_faces(
    img: Image.Image,
    mtcnn,
    facenet,
    device: torch.device,
    max_faces: int = 4,
    pad_to: Optional[int] = None,   # set to an int (e.g., 4) if you want fixed length
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      feats: [S, 512] float32 (CPU)
      mask:  [S] bool (True=PAD)
    """
    with torch.no_grad():
        faces, probs = mtcnn(img, return_prob=True)

    if faces is None or len(faces) == 0:
        feats = torch.zeros((0, 512), dtype=torch.float32)
        if pad_to is not None:
            return _pad_to_len(feats, pad_to)
        return feats, torch.zeros((0,), dtype=torch.bool)

    if isinstance(probs, torch.Tensor):
        probs = probs.tolist()

    facelist = sorted(zip(faces, probs), key=lambda x: x[1], reverse=True)[:max_faces]
    face_tensors = torch.stack([fp[0] for fp in facelist]).to(device)  # [k,3,160,160]
    with torch.no_grad():
        embeds = facenet(face_tensors).float()  # [k,512] on device
    feats = embeds.detach().cpu().contiguous()

    if pad_to is not None:
        return _pad_to_len(feats, pad_to)
    return feats, torch.zeros(feats.size(0), dtype=torch.bool)

def detect_objects(
    image_path: str,
    yolo, resnet, preprocess, device: torch.device,
    conf: float = 0.3, iou: float = 0.45, max_det: int = 64,
    pad_to: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      feats: [S, 2048] float32 (CPU)
      mask:  [S] bool (True=PAD)
    """
    img = Image.open(image_path).convert("RGB")
    with torch.no_grad():
        results = yolo(image_path, conf=conf, iou=iou, max_det=max_det, verbose=False, show=False)

    dets: List[torch.Tensor] = []
    if results:
        res = results[0]
        xyxy = res.boxes.xyxy.cpu()
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = map(float, xyxy[i].tolist())
            crop = img.crop((x1, y1, x2, y2)).resize((224, 224))
            t = preprocess(crop).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = resnet(t).squeeze().detach().float()  # [2048]
            dets.append(feat.cpu())

    feats = torch.stack(dets, dim=0) if len(dets) else torch.zeros((0, 2048), dtype=torch.float32)
    if pad_to is not None:
        return _pad_to_len(feats, pad_to)
    return feats, torch.zeros(feats.size(0), dtype=torch.bool)

# ---------- IMAGE PATCHES (49 fixed) ----------
def image_feature(
    img: Image.Image,
    resnet: torch.nn.Module,
    preprocess: Compose,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      patches: [49, 2048] float32 (CPU)
      mask:    [49] bool (all False)
    """
    pre = Compose([
        Resize(256), CenterCrop(224), ToTensor(),
        Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    t = pre(img).unsqueeze(0).to(device)
    with torch.no_grad():
        fmap = resnet(t)                      # [1,2048,7,7]
    patches = fmap.squeeze(0).permute(1, 2, 0).reshape(-1, 2048).contiguous().float().cpu()  # [49,2048]
    mask = torch.zeros(49, dtype=torch.bool)  # fixed grid -> no padding
    return patches, mask

def load_split(path: str) -> Dict[str, dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _write_json_minified(path: str, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        # separators removes all extra spaces, indent=None avoids newlines
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"), indent=None)
    os.replace(tmp, path)

def save_checkpoint(samples: List[dict], articles: Dict[str, dict], objects: List[dict], out_dir: str, split: str):
    with open(os.path.join(out_dir, f"{split}.json"), "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    _write_json_minified(os.path.join(out_dir, f"articles_{split}.json"), articles)
    with open(os.path.join(out_dir, f"objects_{split}.json"), "w", encoding="utf-8") as f:
        json.dump(objects, f, ensure_ascii=False, indent=2)


def load_checkpoint(out_dir: str, split: str) -> Tuple[List[dict], Dict[str, dict], List[dict]]:
    samples, articles, objects = [], {}, []
    sp = os.path.join(out_dir, f"{split}.json")
    ap = os.path.join(out_dir, f"articles_{split}.json")
    op = os.path.join(out_dir, f"objects_{split}.json")
    if os.path.exists(sp):
        with open(sp, "r", encoding="utf-8") as f:
            samples = json.load(f)
    if os.path.exists(ap):
        with open(ap, "r", encoding="utf-8") as f:
            articles = json.load(f)
    if os.path.exists(op):
        with open(op, "r", encoding="utf-8") as f:
            objects = json.load(f)
    return samples, articles, objects

def process_item(sid: str, item: dict, segmented_context: str,
                 models: dict, image_out: str, split: str):
    sample_id = str(sid)

    # Prepare image path; optionally copy to image_out
    img_id = item["image_path"].split("images/")[-1]
    img_path = f"/datastore/npl/ICEK/Wikipedia/images_resized/{img_id}"
    # print(f"Image Path: {img_path}")
    if image_out:
        os.makedirs(image_out, exist_ok=True)
        dst = os.path.join(image_out, f"{sample_id}.jpg")
        if not os.path.exists(dst):
            try:
                shutil.copy(img_path, dst)
            except OSError as e:
                logging.error(f"Copy error for {img_path} → {dst}: {e}")
                return sample_id, None, None, None
        img_path = dst

    try:
        image = Image.open(img_path).convert("RGB")
    except OSError as e:
        logging.error(f"Load image error {img_path}: {e}")
        return sample_id, None, None, None

    mtcnn = models["mtcnn"]; facenet = models["facenet"]; resnet = models["resnet"]
    resnet_object = models["resnet_object"]; yolo = models["yolo"]; preprocess = models["preprocess"]
    embedder = models["embedder"]; device = models["device"]

    face_info = detect_faces(image, mtcnn, facenet, device)
    obj_feats = detect_objects(img_path, yolo, resnet_object, preprocess, device)
    img_feat = np.array(image_feature(image, resnet, preprocess, device), dtype=np.float32)
    art_embed = embedder([segmented_context]).cpu().numpy()

    features = {
        "image_feature": img_feat,
        "face_embeddings": np.array(face_info.get("embeddings", []), dtype=np.float32) if face_info.get("embeddings") is not None and len(face_info.get("embeddings")) else np.zeros((1, 512), dtype=np.float32),
        "face_detect_probs": np.array(face_info.get("detect_probs", []), dtype=np.float32) if face_info.get("detect_probs") else np.zeros((1,), dtype=np.float32),
        "face_n_faces": np.array(face_info.get("n_faces", 0), dtype=np.int32),
        "object_features": np.array(obj_feats, dtype=np.float32) if obj_feats else np.zeros((1, 2048), dtype=np.float32),
        "article_embed": art_embed,
    }

    article = {
        "_id": sample_id,
        "context": " ".join(item.get("paragraphs", [])),
        "images": [item.get("caption", "")],
        "web_url": ""
    }

    sample = {
        "_id": sample_id,
        "article_id": sample_id,
        "split": split,
        "image_index": 0
    }

    object_data = {"_id": sample_id, "object_features": obj_feats}

    return sample_id, features, sample, article, object_data

def _open_new_shard(output_dir: str, split: str, shard_idx: int):
    os.makedirs(output_dir, exist_ok=True)
    shard_path = os.path.join(output_dir, f"{split}_feat_{shard_idx:03d}.h5")
    h5 = h5py.File(shard_path, "a", libver="latest")
    root = h5.require_group("samples")
    return h5, root, shard_path
 

def _dump_index_json(output_dir: str, split: str, index: dict):
    idx_path = os.path.join(output_dir, f"{split}_h5_index.json")
    tmp = idx_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    os.replace(tmp, idx_path)


def _write_sample_to_h5(root: h5py.Group, sid: str, features: dict):
    grp = root.require_group(sid)
    # Clear existing datasets if any
    for name in list(grp.keys()):
        del grp[name]
    # Write datasets (uncompressed)
    grp.create_dataset("image_feature", data=features["image_feature"])
    grp.create_dataset("object_features", data=features["object_features"])
    grp.create_dataset("article_embed", data=features["article_embed"])
    grp.create_dataset("face_embeddings", data=features["face_embeddings"])
    grp.create_dataset("face_detect_probs", data=features["face_detect_probs"])
    # Attribute
    grp.attrs["face_n_faces"] = features["face_n_faces"]

def convert_items(items: Dict[str, dict], split: str, models: dict, output_dir: str,
                  image_out: str = None, checkpoint_interval: int = 100):
    samples_all, articles_all, objects_all = [], {}, []

    vncore = models["vncore"]

    # 1) Preprocess text (segmentation) first
    items_to_process = []
    logging.info(f"Preprocessing texts for split {split}")
    for sid, item in tqdm(list(items.items()), desc=f"Split {split}"):
        context = " ".join(item.get("paragraphs", []))
        caption = item.get("caption", "")
        try:
            segmented_context = segment_text(context, vncore)
            items_to_process.append((sid, item, segmented_context))
        except Exception as e:
            logging.error(f"Text preprocess error {sid}: {e}")
            continue

    total = len(items_to_process)
    logging.info(f"DONE TEXT SPLIT. Total record to process: {total}")

    # 2) Shard setup
    need_shard = total >= SHARD_THRESHOLD
    shard_size = SHARD_SIZE if need_shard else max(total, 1)

    shard_idx = 0
    in_shard = 0
    h5f, h5root, shard_path = _open_new_shard(output_dir, split, shard_idx)
    index = {}   # sample_id -> {"shard": basename, "group": f"/samples/{sample_id}"}
    logging.info(f"COMPLETE INIT FIRST H5 SHARD {split}")
    # 3) Iterate and write
    processed = 0
    for sid, item, segmented_context in tqdm(items_to_process, desc=f"Encode+Save {split}"):
        try:
            sample_id, features, sample, article, object_data = process_item(
                sid, item, segmented_context, models, image_out, split
            )
            if features is None:
                continue

            # Write features → current shard
            _write_sample_to_h5(h5root, sample_id, features)
            index[sample_id] = {
                "shard": os.path.basename(shard_path),
                "group": f"/samples/{sample_id}"
            }

            # Collect light metadata for JSON files
            samples_all.append(sample)
            articles_all[sample_id] = article
            # Point objects to H5 instead of inlining giant arrays
            objects_all.append({
                "_id": sample_id,
                "h5_shard": os.path.basename(shard_path),
                "group": f"/samples/{sample_id}",
                "n_objects": int(np.array(features["object_features"]).shape[0])
            })

            in_shard += 1
            processed += 1

            # Rotate shard
            if in_shard >= shard_size and processed < total:
                h5f.flush(); h5f.close()
                shard_idx += 1
                in_shard = 0
                h5f, h5root, shard_path = _open_new_shard(output_dir, split, shard_idx)

            # Periodic flush & index checkpoint
            if processed % checkpoint_interval == 0:
                h5f.flush()
                _dump_index_json(output_dir, split, index)

        except Exception as e:
            logging.exception(f"ERROR PROCESSING ITEM {sid}: {e}")
            continue

    # Finalize
    try:
        h5f.flush(); h5f.close()
    except Exception:
        pass
    _dump_index_json(output_dir, split, index)
    return samples_all, articles_all, objects_all

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create ViWiki dataset files")
    parser.add_argument("data_dir", nargs="?", default="/datastore/npl/ICEK/Wikipedia/content/ver4", help="Directory with train/val/test JSON")
    parser.add_argument("output_dir", nargs="?", default="/datastore/npl/ICEK/TnT/dataset/content", help="Directory to write converted files")
    parser.add_argument("--image-out", default=None, dest="image_out",
                        help="Optional directory to copy images")
    parser.add_argument("--vncorenlp", default="/datastore/npl/ICEK/VnCoreNLP",
                        help="Path to VnCoreNLP jar file")
    parser.add_argument("--checkpoint-interval", type=int, default=100,
                        help="Save checkpoint every N items")
    args = parser.parse_args()
    print("LOAD ARGS DONE!")
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = setup_models(device, args.vncorenlp)
    for split in ["demo10" ]:
        split_data = load_split(os.path.join(args.data_dir, f"{split}.json"))
        logging.info(f"Loaded data: {split}")
        print(f"Loaded data: {split}")
        samples, articles, objects = convert_items(split_data, split, models, args.output_dir, args.image_out, args.checkpoint_interval)
        save_checkpoint(samples, articles, objects, args.output_dir, split)
        logging.info(f"[{split}] saved: samples={len(samples)}, articles={len(articles)}, objects={len(objects)}")
        print(f"[{split}] saved: samples={len(samples)}, articles={len(articles)}, objects={len(objects)}")
    for split in ["test","val","train" ]:
        split_data = load_split(os.path.join(args.data_dir, f"{split}.json"))
        logging.info(f"Loaded data: {split}")
        print(f"Loaded data: {split}")
        samples, articles, objects = convert_items(split_data, split, models, args.output_dir, args.image_out, args.checkpoint_interval)
        save_checkpoint(samples, articles, objects, args.output_dir, split)
        logging.info(f"[{split}] saved: samples={len(samples)}, articles={len(articles)}, objects={len(objects)}")
        print(f"[{split}] saved: samples={len(samples)}, articles={len(articles)}, objects={len(objects)}")