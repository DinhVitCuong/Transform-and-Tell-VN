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
* run **VnCoreNLP** to obtain ``caption_ner`` and ``context_ner``
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
# class RobertaEmbedder(torch.nn.Module):
#     def __init__(self, model, tokenizer, segmenter, device):
#         super().__init__()
#         self.model = model
#         self.tokenizer = tokenizer
#         self.segmenter = segmenter
#         self.device = device
#         self.num_layers = model.config.num_hidden_layers + 1
#         self.hidden_size = model.config.hidden_size
#         self.max_position_embeddings = model.config.max_position_embeddings  # Usually 512
#         self.layer_weights = torch.nn.Parameter(torch.ones(self.num_layers) / self.num_layers)
        
#         # Pre-calculate maximum token length per sentence
#         self.max_tokens_per_sentence = self.max_position_embeddings - 2  # Reserve for special tokens

#     def forward(self, text: str) -> torch.Tensor:
#         """Process text with absolute safety against position embedding overflow"""
#         # Clean and segment text
#         sentences = re.split(r'(?<=[\.!?])\s+', text.strip())
#         if not sentences:
#             return torch.zeros(1, self.hidden_size, device=self.device)

#         # Process each sentence with strict length control
#         embeddings = []
#         current_length = 0
        
#         for sent in sentences:
#             # Skip if we've reached max length
#             if current_length >= self.max_position_embeddings:
#                 break
#             try:
#                 input = self.segmenter.word_segment(sent)[0]
#             except:
#                 input = sent    
#             # Calculate safe token limit
#             remaining = self.max_position_embeddings - current_length
#             max_length = min(remaining, self.max_tokens_per_sentence)
            
#             # Tokenize with safety margins
#             toks = self.tokenizer(
#                 input,
#                 truncation=True,
#                 max_length=max_length,
#                 return_tensors="pt",
#                 add_special_tokens=True
#             ).to(self.device)
            
#             # Skip empty tokenizations
#             if toks.input_ids.size(1) == 0:
#                 continue
                
#             # Manually create SAFE position IDs
#             seq_len = toks.input_ids.size(1)
#             position_ids = torch.arange(0, seq_len, dtype=torch.long, device=self.device)
#             position_ids = position_ids.clamp(max=self.max_position_embeddings-1)
#             toks['position_ids'] = position_ids.unsqueeze(0)
            
#             # Get hidden states
#             with torch.no_grad():
#                 outputs = self.model(**toks, output_hidden_states=True)
            
#             # Weighted sum of layers
#             hidden_states = torch.stack(outputs.hidden_states)  # [layers, 1, seq_len, hidden]
#             weights = torch.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
#             weighted_embedding = (weights * hidden_states).sum(dim=0).squeeze(0)  # [seq_len, hidden]
            
#             embeddings.append(weighted_embedding)
#             current_length += seq_len  # Track total tokens

#         if not embeddings:
#             return torch.zeros(1, self.hidden_size, device=self.device)
        
#         # Combine and truncate
#         full_embedding = torch.cat(embeddings, dim=0)[:self.max_position_embeddings]
#         return full_embedding

class RobertaEmbedder(torch.nn.Module):
    def __init__(self, model, tokenizer, device):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.hidden_size = model.config.hidden_size
        self.num_layers = model.config.num_hidden_layers + 1  # Include input embedding layer
        self.max_position_embeddings = model.config.max_position_embeddings  # Usually 512
        self.max_tokens_per_sentence = self.max_position_embeddings - 2  # Reserve for special tokens

    def forward(self, texts: List[str], segmented_texts: List[str]) -> torch.Tensor:
        """Process a list of pre-segmented texts, returning all PhoBERT hidden states"""
        embeddings = []
        for text, segmented in zip(texts, segmented_texts):
            sentences = re.split(r'(?<=[\.!?])\s+', text.strip())
            seg_sentences = segmented.split(' <SEP> ') if segmented else []
            if not sentences or not seg_sentences:
                embeddings.append(torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device))
                continue

            sent_embeddings = []
            current_length = 0
            for sent, seg_sent in zip(sentences, seg_sentences):
                if current_length >= self.max_position_embeddings:
                    break
                input = seg_sent.strip()
                if not input:
                    continue
                remaining = self.max_position_embeddings - current_length
                max_length = min(remaining, self.max_tokens_per_sentence)
                toks = self.tokenizer(
                    input,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                    add_special_tokens=True
                ).to(self.device)
                if toks.input_ids.size(1) == 0:
                    continue
                seq_len = toks.input_ids.size(1)
                position_ids = torch.arange(0, seq_len, dtype=torch.long, device=self.device)
                position_ids = position_ids.clamp(max=self.max_position_embeddings-1)
                toks['position_ids'] = position_ids.unsqueeze(0)
                with torch.no_grad():
                    outputs = self.model(**toks, output_hidden_states=True)
                hidden_states = torch.stack(outputs.hidden_states)  # [num_layers, 1, seq_len, hidden]
                sent_embeddings.append(hidden_states.squeeze(1))  # [num_layers, seq_len, hidden]
                current_length += seq_len
            if not sent_embeddings:
                embeddings.append(torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device))
            else:
                # Fix: Concatenate directly along sequence dimension
                full_embedding = torch.cat(sent_embeddings, dim=1)[:, :self.max_position_embeddings, :]
                embeddings.append(full_embedding)
        # Pad to max length
        max_len = max(e.size(1) for e in embeddings)
        padded_embeddings = torch.zeros(len(texts), self.num_layers, max_len, self.hidden_size, device=self.device)
        for i, emb in enumerate(embeddings):
            padded_embeddings[i, :, :emb.size(1), :] = emb
        return padded_embeddings  # [batch_size, num_layers, seq_len, hidden_size]

def setup_models(device: torch.device, vncorenlp_path="/data2/npl/ICEK/VnCoreNLP"):
    py_vncorenlp.download_model(save_dir=vncorenlp_path)
    vncore = py_vncorenlp.VnCoreNLP(
        annotators=["wseg", "pos", "ner", "parse"],
        save_dir=vncorenlp_path,
        max_heap_size='-Xmx6g'
    )
    print("LOADED VNCORENLP!")
    
    # Face detection + embedding
    mtcnn = MTCNN(keep_all=True, device=device)
    facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    print("LOADED FaceNet!")

    # Global image feature
    weights = ResNet152_Weights.IMAGENET1K_V1 
    base = resnet152(weights=weights).eval().to(device)
    resnet = nn.Sequential(*list(base.children())[:-2]).eval().to(device)
    
    resnet_object = nn.Sequential(*list(base.children())[:-1]).eval().to(device)
    print("LOADED ResNet152!")

    yolo = YOLO("yolov8m.pt")  # tải weight tự động lần đầu
    yolo.fuse()                # fuse model for speed
    print("LOADED YOLOv5!")
    
    phoBERTlocal = "/data2/npl/ICEK/TnT/phoBERT_large/phobert-large"
    tokenizer = AutoTokenizer.from_pretrained(phoBERTlocal, use_fast=False, local_files_only=True)
    roberta     = AutoModel    .from_pretrained(phoBERTlocal, use_safetensors=True, local_files_only=True).to(device).eval()
    embedder = RobertaEmbedder(roberta, tokenizer, device).to(device)
    print("LOADED phoBERT!")
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
    return " <SEP> ".join(segmented_sentences)

def extract_entities(text: str,
                     model
                    ) -> Dict[str, List[str]]:
    """
    Chia text thành từng câu, chạy NER trên mỗi câu bằng VnCoreNLP,
    rồi gộp kết quả (loại trùng). Nếu một câu lỗi hoặc không có entity,
    nó sẽ được bỏ qua.
    """
    label_mapping = {
        "PER":  "PERSON", "B-PER":  "PERSON", "I-PER":  "PERSON",
        "ORG":  "ORG",    "B-ORG":  "ORG",    "I-ORG":  "ORG",
        "LOC":  "LOC",    "B-LOC":  "LOC",    "I-LOC":  "LOC",
        "GPE":  "GPE",    "B-GPE":  "GPE",    "I-GPE":  "GPE",
        "NORP": "NORP",   "B-NORP": "NORP",   "I-NORP": "NORP",
        "MISC": "MISC",   "B-MISC": "MISC",   "I-MISC": "MISC",
    }
    entities = defaultdict(set)
    sentences = re.split(r'(?<=[\.!?])\s+', text.strip())
    if not sentences:
        return {}
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        try:
            annotated_text = model.annotate_text(sent)
        except Exception as e:
            print(f"Lỗi khi annotating text: {e}")
            return entities

        for subsent in annotated_text:

            for word in annotated_text[subsent]:
                ent_type = label_mapping.get(word.get('nerLabel', ''), '')
                ent_text = word.get('wordForm', '').strip()
                if ent_type and ent_text:
                    entities[ent_type].add(' '.join(ent_text.split('_')).strip("•"))
    return {typ: sorted(vals) for typ, vals in entities.items()}

def detect_faces(img: Image.Image, mtcnn: MTCNN, facenet: InceptionResnetV1, device: torch.device, max_faces=4) -> dict:
    with torch.no_grad():
        faces, probs = mtcnn(img, return_prob=True)
    if faces is None or len(faces) == 0:
        return {"n_faces": 0, "embeddings": [], "detect_probs": []}
    if isinstance(probs, torch.Tensor):
        probs = probs.tolist()
    facelist = sorted(zip(faces, probs), key=lambda x: x[1], reverse=True)
    facelist = facelist[:max_faces]

    face_tensors = torch.stack([fp[0] for fp in facelist]).to(device)  # (k,3,160,160)
    probs_top    = [float(fp[1]) for fp in facelist]

    with torch.no_grad():
        embeds = facenet(face_tensors).cpu().tolist()  # List[k][512]
    return {
        "n_faces": len(embeds),
        "embeddings": embeds,
        "detect_probs": probs[: len(embeds)].tolist(),
    }

def detect_objects(image_path: str, model, resnet, preprocess, device):
    img = Image.open(image_path).convert("RGB")
    results = model(image_path, conf=0.3, iou=0.45, max_det=64, verbose=False, show=False )     
    detections = []
    if not results:
        return detections
    res = results[0]                   # take the first Results object
    # lấy các tensor
    xyxy   = res.boxes.xyxy.cpu()      # shape (N,4)
    confs  = res.boxes.conf.cpu()      # shape (N,)
    classes= res.boxes.cls.cpu()       # shape (N,)
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i].tolist()
        conf = float(confs[i])
        cls  = int(classes[i])
        crop = img.crop((x1, y1, x2, y2)).resize((224,224))
        tensor = preprocess(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            feat_tensor = resnet(tensor).squeeze()   # still a Tensor
        # convert to plain Python list:
            feat = feat_tensor.cpu().tolist()
        detections.append(feat)
    return detections

def image_feature(img: Image.Image, resnet: torch.nn.Module, preprocess: Compose, device: torch.device) -> List[float]:
    preprocess_img_feat = Compose([
    # 1) Resize the shorter side to 256 (preserve aspect)
    Resize(256),
    # 2) Crop out the central 224×224 patch
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std =[0.229, 0.224, 0.225])
    ])
    tensor = preprocess_img_feat(img).unsqueeze(0).to(device)
    with torch.no_grad():
        fmap = resnet(tensor)                         # (1,2048,7,7)
    fmap = fmap.squeeze(0)                            # (2048,7,7)
    # move channel to last dim → (7,7,2048), then flatten to (49,2048)
    patches = fmap.permute(1, 2, 0).reshape(-1, 2048)  # (49,2048)
    return patches.cpu().tolist() 

def load_split(path: str) -> Dict[str, dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(samples: List[dict], articles: Dict[str, dict], objects: List[dict], out_dir: str, split: str):
    with open(os.path.join(out_dir, f"{split}.json"), "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, f"articles_{split}.json"), "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
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

def process_item(sid: str, item: dict, segmented_context: str, cap_ner: Dict[str, List[str]], ctx_ner: Dict[str, List[str]],
                 models: dict, image_out: str, split: str):
    sample_id = str(sid)

    # Prepare image path; optionally copy to image_out
    img_id = item["image_path"].split("images/")[-1]
    img_path = f"/data2/npl/ICEK/Wikipedia/images_resized/{img_id}"
    print(f"Image Path: {img_path}")
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
    art_embed = embedder([item.get("caption", "")], [segmented_context])[0].cpu().numpy()

    features = {
        "image_feature": img_feat,
        "face_embeddings": np.array(face_info.get("embeddings", []), dtype=np.float32) if face_info.get("embeddings") is not None and len(face_info.get("embeddings")) else np.zeros((1, 512), dtype=np.float32),
        "face_detect_probs": np.array(face_info.get("detect_probs", []), dtype=np.float32) if face_info.get("detect_probs") else np.zeros((1,), dtype=np.float32),
        "face_n_faces": np.array(face_info.get("n_faces", 0), dtype=np.int32),
        "object_features": np.array(obj_feats, dtype=np.float32) if obj_feats else np.zeros((1, 2048), dtype=np.float32),
        "article_embed": art_embed,
        "caption_ner": cap_ner,
        "context_ner": ctx_ner,
    }

    article = {
        "_id": sample_id,
        "context": " ".join(item.get("context", [])),
        "images": [item.get("caption", "")],
        "web_url": "",
        "caption_ner": [cap_ner],
        "context_ner": [ctx_ner],
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

def _write_sample_to_h5(root: h5py.Group, sample_id: str, features: dict):
    g = root.create_group(sample_id)
    # Numeric arrays (allow ragged shapes by separate datasets per sample)
    g.create_dataset("image_feature", data=features["image_feature"])
    g.create_dataset("face_embeddings", data=features["face_embeddings"])
    g.create_dataset("face_detect_probs", data=features["face_detect_probs"])
    g.attrs["face_n_faces"] = int(np.array(features["face_n_faces"]).item())
    g.create_dataset("object_features", data=features["object_features"])
    g.create_dataset("article_embed", data=features["article_embed"])
    # NER (store as UTF-8 JSON string)
    ner_json = json.dumps(
        {"caption_ner": features["caption_ner"], "context_ner": features["context_ner"]},
        ensure_ascii=False,
    )
    dt = h5py.string_dtype(encoding="utf-8")
    g.create_dataset("ner", data=ner_json, dtype=dt)

def _dump_index_json(output_dir: str, split: str, index: dict):
    idx_path = os.path.join(output_dir, f"{split}_h5_index.json")
    tmp = idx_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    os.replace(tmp, idx_path)

# ----------------------------
# Split conversion with H5 shards
# ----------------------------

# def convert_items(items: Dict[str, dict], split: str, models: dict, output_dir: str,
#                   image_out: str = None, checkpoint_interval: int = 100):
#     samples_all, articles_all, objects_all = [], {}, []

#     vncore = models["vncore"]

#     # Preprocess text (segmentation + NER) first
#     items_to_process = []
#     logging.info(f"Preprocessing texts for split {split}")
#     for sid, item in tqdm(list(items.items()), desc=f"Pre-NER {split}"):
#         context = " ".join(item.get("context", []))
#         caption = item.get("caption", "")
#         try:
#             segmented_context = segment_text(context, vncore)
#             cap_ner = extract_entities(caption, vncore)
#             ctx_ner = extract_entities(context, vncore)
#             items_to_process.append((sid, item, segmented_context, cap_ner, ctx_ner))
#         except Exception as e:
#             logging.error(f"Text preprocess error {sid}: {e}")
#             continue

#     total = len(items_to_process)
#     logging.info(f"DONE NER. Total record to process: {total}")
#     for sid, item, segmented_context, cap_ner, ctx_ner in items_to_process:
#         try:
#             sample_id, features, sample, article, object_data = process_item(
#                 sid, item, segmented_context, cap_ner, ctx_ner, models, image_out, split
#             )
#             print(f"DONE PROCESS {sid}")
#         except:
#             print(f"ERROR PROCESSING ITEM No.{sid}")
#     return samples_all, articles_all, objects_all

def convert_items(items: Dict[str, dict], split: str, models: dict, output_dir: str,
                  image_out: str = None, checkpoint_interval: int = 100):
    samples_all, articles_all, objects_all = [], {}, []

    vncore = models["vncore"]

    # 1) Preprocess text (segmentation + NER) first
    items_to_process = []
    logging.info(f"Preprocessing texts for split {split}")
    for sid, item in tqdm(list(items.items()), desc=f"Pre-NER {split}"):
        context = " ".join(item.get("context", []))
        caption = item.get("caption", "")
        try:
            segmented_context = segment_text(context, vncore)
            cap_ner = extract_entities(caption, vncore)
            ctx_ner = extract_entities(context, vncore)
            items_to_process.append((sid, item, segmented_context, cap_ner, ctx_ner))
        except Exception as e:
            logging.error(f"Text preprocess error {sid}: {e}")
            continue

    total = len(items_to_process)
    logging.info(f"DONE NER. Total record to process: {total}")

    # 2) Shard setup
    need_shard = total >= SHARD_THRESHOLD
    shard_size = SHARD_SIZE if need_shard else max(total, 1)

    shard_idx = 0
    in_shard = 0
    h5f, h5root, shard_path = _open_new_shard(output_dir, split, shard_idx)
    index = {}   # sample_id -> {"shard": basename, "group": f"/samples/{sample_id}"}

    # 3) Iterate and write
    processed = 0
    for sid, item, segmented_context, cap_ner, ctx_ner in tqdm(items_to_process, desc=f"Encode+Save {split}"):
        try:
            sample_id, features, sample, article, object_data = process_item(
                sid, item, segmented_context, cap_ner, ctx_ner, models, image_out, split
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
    parser.add_argument("data_dir", nargs="?", default="/data2/npl/ICEK/Wikipedia/content/ver4", help="Directory with train/val/test JSON")
    parser.add_argument("output_dir", nargs="?", default="/data2/npl/ICEK/TnT/dataset/content", help="Directory to write converted files")
    parser.add_argument("--image-out", default=None, dest="image_out",
                        help="Optional directory to copy images")
    parser.add_argument("--vncorenlp", default="/data2/npl/ICEK/VnCoreNLP",
                        help="Path to VnCoreNLP jar file")
    parser.add_argument("--checkpoint-interval", type=int, default=100,
                        help="Save checkpoint every N items")
    args = parser.parse_args()
    print("LOAD ARGS DONE!")
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = setup_models(device, args.vncorenlp)
    for split in ["demo10"]:  # DEBUG
        split_data = load_split(os.path.join(args.data_dir, f"{split}.json"))
        logging.info("Loaded data")
        samples, articles, objects = convert_items(split_data, split, models, args.output_dir, args.image_out, args.checkpoint_interval)
        save_checkpoint(samples, articles, objects, args.output_dir, split)
        logging.info(f"[{split}] saved: samples={len(samples)}, articles={len(articles)}, objects={len(objects)}")
    # for split in ["train", "val", "test"]:
    #     split_data = load_split(os.path.join(args.data_dir, f"{split}.json"))
    #     logging.info(f"Loaded data: {split}")
    #     samples, articles, objects = convert_items(split_data, split, models, args.output_dir, args.image_out, args.checkpoint_interval)
    #     save_checkpoint(samples, articles, objects, args.output_dir, split)
    #     logging.info(f"[{split}] saved: samples={len(samples)}, articles={len(articles)}, objects={len(objects)}")