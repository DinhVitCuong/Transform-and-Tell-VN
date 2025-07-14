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
* embed the article text using a **RoBERTa** model (PhoBERT for Vietnamese), saving raw last hidden state

The resulting features are stored in an HDF5 file (e.g., `train_features.h5`) with datasets for each sample ID.
"""

import argparse
import json
import os
import shutil
from typing import Dict, Tuple, List
from tqdm import tqdm
from collections import defaultdict
import re
import h5py
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
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

Image.MAX_IMAGE_PIXELS = None

# Set up logging
logging.basicConfig(filename="preprocess.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class RobertaEmbedder(torch.nn.Module):
    def __init__(self, model, tokenizer, segmenter, device):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.segmenter = segmenter
        self.device = device
        self.hidden_size = model.config.hidden_size
        self.num_layers = model.config.num_hidden_layers + 1  # Include input embedding layer
        self.max_position_embeddings = model.config.max_position_embeddings  # Usually 512
        self.max_tokens_per_sentence = self.max_position_embeddings - 2  # Reserve for special tokens

    def forward(self, texts: List[str]) -> torch.Tensor:
        """Process a batch of texts, returning all PhoBERT hidden states"""
        embeddings = []
        for text in texts:
            sentences = re.split(r'(?<=[\.!?])\s+', text.strip())
            if not sentences:
                embeddings.append(torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device))
                continue

            sent_embeddings = []
            current_length = 0
            for sent in sentences:
                if current_length >= self.max_position_embeddings:
                    break
                try:
                    input = self.segmenter.word_segment(sent)[0]
                except:
                    input = sent
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
                full_embedding = torch.cat([se.unsqueeze(1) for se in sent_embeddings], dim=1)[:, :self.max_position_embeddings]
                embeddings.append(full_embedding)
        # Pad to max length
        max_len = max(e.size(1) for e in embeddings)
        padded_embeddings = torch.zeros(len(texts), self.num_layers, max_len, self.hidden_size, device=self.device)
        for i, emb in enumerate(embeddings):
            padded_embeddings[i, :, :emb.size(1)] = emb
        return padded_embeddings  # [batch_size, num_layers, seq_len, hidden_size]

def setup_models(device: torch.device, vncorenlp_path="/data/npl/ICEK/VnCoreNLP"):
    py_vncorenlp.download_model(save_dir=vncorenlp_path)
    vncore = py_vncorenlp.VnCoreNLP(
        annotators=["wseg", "ner"],
        save_dir=vncorenlp_path,
        max_heap_size='-Xmx6g'
    )
    logging.info("Loaded VnCoreNLP")
    mtcnn = MTCNN(keep_all=True, device=device)
    facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    logging.info("Loaded FaceNet")
    weights = ResNet152_Weights.IMAGENET1K_V1
    base = resnet152(weights=weights).eval().to(device)
    resnet = nn.Sequential(*list(base.children())[:-2]).eval().to(device)
    resnet_object = nn.Sequential(*list(base.children())[:-1]).eval().to(device)
    logging.info("Loaded ResNet152")
    yolo = YOLO("yolov8m.pt")
    yolo.fuse()
    logging.info("Loaded YOLOv8")
    phoBERTlocal = "/data/npl/ICEK/TnT/phoBERT_large/phobert-large"
    tokenizer = AutoTokenizer.from_pretrained(phoBERTlocal, use_fast=False, local_files_only=True)
    roberta = AutoModel.from_pretrained(phoBERTlocal, use_safetensors=True, local_files_only=True).to(device).eval()
    embedder = RobertaEmbedder(roberta, tokenizer, vncore, device).to(device)
    logging.info("Loaded PhoBERT")
    preprocess = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

def extract_entities(text: str, model) -> Dict[str, List[str]]:
    label_mapping = {
        "PER": "PERSON", "B-PER": "PERSON", "I-PER": "PERSON",
        "ORG": "ORG", "B-ORG": "ORG", "I-ORG": "ORG",
        "LOC": "LOC", "B-LOC": "LOC", "I-LOC": "LOC",
        "GPE": "GPE", "B-GPE": "GPE", "I-GPE": "GPE",
        "NORP": "NORP", "B-NORP": "NORP", "I-NORP": "NORP",
        "MISC": "MISC", "B-MISC": "MISC", "I-MISC": "MISC",
    }
    entities = defaultdict(list)
    sentences = re.split(r'(?<=[\.!?])\s+', text.strip())
    if not sentences:
        return {}
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        try:
            annotated_text = model.annotate_text(sent)
            for subsent in annotated_text:
                for word in annotated_text[subsent]:
                    ent_type = label_mapping.get(word.get('nerLabel', ''), '')
                    ent_text = word.get('wordForm', '').strip()
                    if ent_type and ent_text:
                        ent_text_clean = ' '.join(ent_text.split('_')).strip("•")
                        if ent_text_clean not in entities[ent_type]:
                            entities[ent_type].append(ent_text_clean)
        except Exception as e:
            logging.error(f"Error annotating text: {e}")
            continue
    return {typ: sorted(vals) for typ, vals in entities.items()}

def detect_faces_batch(images: List[Image.Image], mtcnn: MTCNN, facenet: InceptionResnetV1, device: torch.device, max_faces=4) -> List[dict]:
    with torch.no_grad():
        faces_batch, probs_batch = mtcnn(images, return_prob=True)
    results = []
    for faces, probs in zip(faces_batch if faces_batch is not None else [], probs_batch if probs_batch is not None else []):
        if faces is None or len(faces) == 0:
            results.append({"n_faces": 0, "embeddings": [], "detect_probs": []})
            continue
        if isinstance(probs, torch.Tensor):
            probs = probs.tolist()
        facelist = sorted(zip(faces, probs), key=lambda x: x[1], reverse=True)[:max_faces]
        face_tensors = torch.stack([fp[0] for fp in facelist]).to(device)
        probs_top = [float(fp[1]) for fp in facelist]
        embeds = facenet(face_tensors).cpu().numpy()
        results.append({
            "n_faces": len(embeds),
            "embeddings": embeds.tolist(),
            "detect_probs": probs_top[:len(embeds)],
        })
    return results

def detect_objects_batch(image_paths: List[str], model, resnet, preprocess, device, batch_size=32) -> List[List[float]]:
    detections_batch = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        images = [Image.open(path).convert("RGB") for path in batch_paths]
        results = model(batch_paths, conf=0.3, iou=0.45, max_det=64, verbose=False, show=False)
        for img, res in zip(images, results):
            detections = []
            if not res:
                detections_batch.append(detections)
                continue
            xyxy = res.boxes.xyxy.cpu()
            confs = res.boxes.conf.cpu()
            classes = res.boxes.cls.cpu()
            crops = [img.crop(xyxy[j].tolist()).resize((224, 224)) for j in range(len(xyxy))]
            if not crops:
                detections_batch.append(detections)
                continue
            tensors = torch.stack([preprocess(crop) for crop in crops]).to(device)
            with torch.no_grad():
                feats = resnet(tensors).squeeze(-1).squeeze(-1).cpu().numpy()
            for feat in feats:
                detections.append(feat.tolist())
            detections_batch.append(detections)
    return detections_batch

def image_feature_batch(images: List[Image.Image], resnet: torch.nn.Module, device: torch.device, batch_size=32) -> List[List[float]]:
    preprocess_img_feat = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    features = []
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i+batch_size]
        tensors = torch.stack([preprocess_img_feat(img) for img in batch_images]).to(device)
        with torch.no_grad():
            fmaps = resnet(tensors)
        fmaps = fmaps.squeeze(0)
        patches = fmaps.permute(0, 2, 3, 1).reshape(fmaps.size(0), -1, 2048)
        features.extend(patches.cpu().numpy().tolist())
    return features

def process_batch(batch: List[Tuple[str, dict]], models: dict, image_out: str = None) -> Tuple[List[dict], Dict[str, dict], List[dict]]:
    vncore = models["vncore"]
    mtcnn = models["mtcnn"]
    facenet = models["facenet"]
    resnet = models["resnet"]
    resnet_object = models["resnet_object"]
    yolo = models["yolo"]
    embedder = models["embedder"]
    preprocess = models["preprocess"]
    device = models["device"]

    samples, articles, objects = [], {}, []
    image_paths = []
    images = []
    captions = []
    contexts = []
    sample_ids = []

    for sid, item in batch:
        img_path = item["image_path"]
        sample_id = str(sid)
        sample_ids.append(sample_id)
        image_paths.append(img_path)
        try:
            images.append(Image.open(img_path).convert("RGB"))
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {e}")
            continue
        captions.append(item.get("caption", ""))
        contexts.append(" ".join(item.get("context", [])))
        if image_out:
            os.makedirs(image_out, exist_ok=True)
            dst = os.path.join(image_out, f"{sample_id}.jpg")
            if not os.path.exists(dst):
                try:
                    shutil.copy(img_path, dst)
                except OSError as e:
                    logging.error(f"Error copying image {img_path} to {dst}: {e}")

    face_infos = detect_faces_batch(images, mtcnn, facenet, device)
    obj_feats_batch = detect_objects_batch(image_paths, yolo, resnet_object, preprocess, device)
    img_feats_batch = image_feature_batch(images, resnet, device)
    cap_ners = [extract_entities(cap, vncore) for cap in captions]
    ctx_ners = [extract_entities(ctx, vncore) for ctx in contexts]
    art_embeds = embedder(contexts).cpu().numpy()

    for i, (sid, item) in enumerate(batch):
        sample_id = str(sid)
        try:
            articles[sample_id] = {
                "_id": sample_id,
                "context": contexts[i],
                "images": [captions[i]],
                "web_url": "",
                "caption_ner": [cap_ners[i]],
                "context_ner": [ctx_ners[i]],
            }
            samples.append({
                "_id": sample_id,
                "article_id": sample_id,
                "split": split,
                "image_index": 0,
                "feature_path": f"{sample_id}",  # Reference to HDF5 dataset
            })
            objects.append({"_id": sample_id, "object_features": obj_feats_batch[i]})
            features = {
                "image_feature": np.array(img_feats_batch[i], dtype=np.float32),
                "face_info": face_infos[i],
                "object_features": np.array(obj_feats_batch[i], dtype=np.float32) if obj_feats_batch[i] else np.zeros((1, 2048), dtype=np.float32),
                "article_embed": art_embeds[i],
                "caption_ner": cap_ners[i],
                "context_ner": ctx_ners[i],
            }
            yield sample_id, features, samples, articles, objects
        except Exception as e:
            logging.error(f"Error processing sample {sample_id}: {e}")
            continue

def convert_items(items: Dict[str, dict], split: str, models: dict, output_dir: str, image_out: str = None, checkpoint_interval: int = 100, batch_size: int = 32, num_workers: int = 4) -> Tuple[List[dict], Dict[str, dict], List[dict]]:
    samples, articles, objects = [], {}, []
    hdf5_path = os.path.join(output_dir, f"{split}_features.h5")
    processed_ids = set()

    # Load existing HDF5 data if available
    if os.path.exists(hdf5_path):
        with h5py.File(hdf5_path, "r") as f:
            processed_ids = set(f.keys())
            samples_path = os.path.join(output_dir, f"{split}.json")
            articles_path = os.path.join(output_dir, f"articles_{split}.json")
            objects_path = os.path.join(output_dir, f"objects_{split}.json")
            if os.path.exists(samples_path):
                with open(samples_path, "r", encoding="utf-8") as sf:
                    samples = json.load(sf)
            if os.path.exists(articles_path):
                with open(articles_path, "r", encoding="utf-8") as af:
                    articles = json.load(af)
            if os.path.exists(objects_path):
                with open(objects_path, "r", encoding="utf-8") as of:
                    objects = json.load(of)
        logging.info(f"Loaded existing data for split {split}: {len(processed_ids)} samples")

    items_to_process = [(sid, item) for sid, item in items.items() if str(sid) not in processed_ids]
    total_items = len(items_to_process)
    processed_count = 0

    with h5py.File(hdf5_path, "a") as hdf5_file, ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(0, len(items_to_process), batch_size):
            batch = items_to_process[i:i+batch_size]
            futures.append(executor.submit(process_batch, batch, models, image_out))
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {split}"):
            try:
                for sample_id, features, batch_samples, batch_articles, batch_objects in future.result():
                    samples.extend(batch_samples)
                    articles.update(batch_articles)
                    objects.extend(batch_objects)
                    group = hdf5_file.create_group(sample_id)
                    for key, value in features.items():
                        if isinstance(value, np.ndarray):
                            group.create_dataset(key, data=value, compression="gzip")
                        else:
                            group.create_dataset(key, data=json.dumps(value), compression="gzip")
                    processed_count += 1
                    if processed_count % checkpoint_interval == 0:
                        save_checkpoint(samples, articles, objects, output_dir, split)
                        logging.info(f"Processed {processed_count}/{total_items} items for split {split}")
            except Exception as e:
                logging.error(f"Error in batch processing: {e}")
                continue
            torch.cuda.empty_cache()  # Clear GPU memory

    save_checkpoint(samples, articles, objects, output_dir, split)
    logging.info(f"Completed processing {processed_count}/{total_items} items for split {split}")
    return samples, articles, objects

def load_split(path: str) -> Dict[str, dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_checkpoint(samples: List[dict], articles: Dict[str, dict], objects: List[dict], output_dir: str, split: str):
    try:
        with open(os.path.join(output_dir, f"{split}.json"), "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        with open(os.path.join(output_dir, f"articles_{split}.json"), "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        with open(os.path.join(output_dir, f"objects_{split}.json"), "w", encoding="utf-8") as f:
            json.dump(objects, f, ensure_ascii=False, indent=2)
        logging.info(f"Checkpoint saved for split {split}")
    except Exception as e:
        logging.error(f"Error saving checkpoint for split {split}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create ViWiki dataset files")
    parser.add_argument("--data_dir", nargs="?", default="/data/npl/ICEK/Wikipedia/content/ver4", help="Directory with train/val/test JSON")
    parser.add_argument("--output_dir", nargs="?", default="/data/npl/ICEK/TnT/dataset/content", help="Directory to write converted files")
    parser.add_argument("--image-out", default=None, dest="image_out", help="Optional directory to copy images")
    parser.add_argument("--vncorenlp", default="/data/npl/ICEK/VnCoreNLP", help="Path to VnCoreNLP jar file")
    parser.add_argument("--checkpoint-interval", type=int, default=100, help="Save checkpoint every N items")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker processes")
    args = parser.parse_args()
    print("Loaded arguments")
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = setup_models(device, args.vncorenlp)
    for split in ["train", "val", "test"]:
        split_data = load_split(os.path.join(args.data_dir, f"{split}.json"))
        samples, articles, objects = convert_items(split_data, split, models, args.output_dir, args.image_out, args.checkpoint_interval, args.batch_size, args.num_workers)
        save_checkpoint(samples, articles, objects, args.output_dir, split)