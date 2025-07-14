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
Image.MAX_IMAGE_PIXELS = None
class RobertaEmbedder(torch.nn.Module):
    def __init__(self, model, tokenizer, segmenter, device):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.segmenter = segmenter
        self.device = device
        self.num_layers = model.config.num_hidden_layers + 1
        self.hidden_size = model.config.hidden_size
        self.max_position_embeddings = model.config.max_position_embeddings  # Usually 512
        self.layer_weights = torch.nn.Parameter(torch.ones(self.num_layers) / self.num_layers)
        
        # Pre-calculate maximum token length per sentence
        self.max_tokens_per_sentence = self.max_position_embeddings - 2  # Reserve for special tokens

    def forward(self, text: str) -> torch.Tensor:
        """Process text with absolute safety against position embedding overflow"""
        # Clean and segment text
        sentences = re.split(r'(?<=[\.!?])\s+', text.strip())
        if not sentences:
            return torch.zeros(1, self.hidden_size, device=self.device)

        # Process each sentence with strict length control
        embeddings = []
        current_length = 0
        
        for sent in sentences:
            # Skip if we've reached max length
            if current_length >= self.max_position_embeddings:
                break
            try:
                input = self.segmenter.word_segment(sent)[0]
            except:
                input = sent    
            # Calculate safe token limit
            remaining = self.max_position_embeddings - current_length
            max_length = min(remaining, self.max_tokens_per_sentence)
            
            # Tokenize with safety margins
            toks = self.tokenizer(
                input,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                add_special_tokens=True
            ).to(self.device)
            
            # Skip empty tokenizations
            if toks.input_ids.size(1) == 0:
                continue
                
            # Manually create SAFE position IDs
            seq_len = toks.input_ids.size(1)
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=self.device)
            position_ids = position_ids.clamp(max=self.max_position_embeddings-1)
            toks['position_ids'] = position_ids.unsqueeze(0)
            
            # Get hidden states
            with torch.no_grad():
                outputs = self.model(**toks, output_hidden_states=True)
            
            # Weighted sum of layers
            hidden_states = torch.stack(outputs.hidden_states)  # [layers, 1, seq_len, hidden]
            weights = torch.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
            weighted_embedding = (weights * hidden_states).sum(dim=0).squeeze(0)  # [seq_len, hidden]
            
            embeddings.append(weighted_embedding)
            current_length += seq_len  # Track total tokens

        if not embeddings:
            return torch.zeros(1, self.hidden_size, device=self.device)
        
        # Combine and truncate
        full_embedding = torch.cat(embeddings, dim=0)[:self.max_position_embeddings]
        return full_embedding

def setup_models(device: torch.device, vncorenlp_path="/data/npl/ICEK/VnCoreNLP"):
    py_vncorenlp.download_model(save_dir=vncorenlp_path)
    vncore = py_vncorenlp.VnCoreNLP(
        annotators=["wseg", "pos", "ner", "parse"],
        save_dir=vncorenlp_path,
        max_heap_size='-Xmx6qgZZ'
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
    
    phoBERTlocal = "/data/npl/ICEK/TnT/phoBERT_large/phobert-large"
    tokenizer = AutoTokenizer.from_pretrained(phoBERTlocal, use_fast=False, local_files_only=True)
    roberta     = AutoModel    .from_pretrained(phoBERTlocal, use_safetensors=True, local_files_only=True).to(device).eval()
    embedder = RobertaEmbedder(roberta, tokenizer, vncore, device).to(device)
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


# def convert_items(items: Dict[str, dict], split: str, models: dict, image_out: str = None
#                    ) -> Tuple[List[dict], Dict[str, dict], List[dict]]:
#     """Convert raw items into dataset entries with extracted features."""
#     samples: List[dict] = []
#     articles: Dict[str, dict] = {}
#     objects: List[dict] = []

#     vncore = models["vncore"]
#     mtcnn = models["mtcnn"]
#     facenet = models["facenet"]
#     resnet = models["resnet"]
#     resnet_object = models["resnet_object"]
#     yolo = models["yolo"]
#     tokenizer = models["tokenizer"]
#     roberta = models["roberta"]
#     preprocess = models["preprocess"]
#     device = models["device"]
#     for sid, item in tqdm(items.items(), desc=f"Processing {split}", unit="item"):
#         sample_id = str(sid)
#         img_path = item["image_path"]
#         print(img_path)
#         # Optionally copy image
#         if image_out:
#             os.makedirs(image_out, exist_ok=True)
#             dst = os.path.join(image_out, f"{sample_id}.jpg")
#             if not os.path.exists(dst):
#                 try:
#                     shutil.copy(img_path, dst)
#                     # print("sucess copy")
#                 except OSError:
#                     pass
#             img_path = dst

#         caption = item.get("caption", "")
#         context_txt = " ".join(item.get("context", []))
#         try:
#             image = Image.open(img_path).convert("RGB")
#         except OSError:
#             continue

#         cap_ner = extract_entities(caption, vncore)
#         ctx_ner = extract_entities(context_txt, vncore)
#         face_info = detect_faces(image, mtcnn, facenet, device)
#         # obj_feats = detect_objects(img_path, yolo, resnet, device)
#         obj_feats = detect_objects(img_path, yolo, resnet_object, preprocess, device)
#         img_feat = image_feature(image, resnet, preprocess, device)
#         # art_embed = roberta_embed(context_txt, tokenizer, roberta, vncore, device)

#         article = {
#             "_id": sample_id,
#             "context": context_txt,
#             "images": [caption],
#             "web_url": "",
#             "caption_ner": [cap_ner],
#             "context_ner": [ctx_ner],
#             # "article_embed": art_embed,
#         }
#         articles[sample_id] = article

#         samples.append({
#             "_id": sample_id,
#             "article_id": sample_id,
#             "split": split,
#             "image_index": 0,
#             "facenet_details": face_info,
#             "image_feature": img_feat,
#         })

#         objects.append({"_id": sample_id, "object_features": obj_feats})
#         break

#     return samples, articles, objects

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Create ViWiki dataset files")
#     parser.add_argument("data_dir", nargs="?", default="/data/npl/ICEK/Wikipedia/content/ver4", help="Directory with train/val/test JSON")
#     parser.add_argument("output_dir", nargs="?", default="/data/npl/ICEK/TnT/dataset/content", help="Directory to write converted files")
#     parser.add_argument("--image-out", default="/data/npl/ICEK/TnT/dataset/images", dest="image_out",
#                         help="Optional directory to copy images")
#     parser.add_argument("--vncorenlp",default="/data/npl/ICEK/VnCoreNLP",
#                         help="Path to VnCoreNLP jar file")
#     args = parser.parse_args()
#     print("LOAD ARGS DONE!")
#     os.makedirs(args.output_dir, exist_ok=True)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     models = setup_models(device, args.vncorenlp)


def load_split(path: str) -> Dict[str, dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(samples: List[dict], articles: Dict[str, dict], objects: List[dict], output_dir: str, split: str):
    """Save intermediate results to JSON files."""
    try:
        with open(os.path.join(output_dir, f"{split}.json"), "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        with open(os.path.join(output_dir, f"articles_{split}.json"), "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        with open(os.path.join(output_dir, f"objects_{split}.json"), "w", encoding="utf-8") as f:
            json.dump(objects, f, ensure_ascii=False, indent=2)
        print(f"Checkpoint saved for split {split}")
    except Exception as e:
        print(f"Error saving checkpoint for split {split}: {e}")

def load_checkpoint(output_dir: str, split: str) -> Tuple[List[dict], Dict[str, dict], List[dict]]:
    """Load existing samples, articles, and objects from JSON files."""
    samples = []
    articles = {}
    objects = []
    try:
        samples_path = os.path.join(output_dir, f"{split}.json")
        if os.path.exists(samples_path):
            with open(samples_path, "r", encoding="utf-8") as f:
                samples = json.load(f)
            print(f"Loaded existing samples for split {split}")
    except Exception as e:
        print(f"Error loading samples for split {split}: {e}")

    try:
        articles_path = os.path.join(output_dir, f"articles_{split}.json")
        if os.path.exists(articles_path):
            with open(articles_path, "r", encoding="utf-8") as f:
                articles = json.load(f)
            print(f"Loaded existing articles for split {split}")
    except Exception as e:
        print(f"Error loading articles for split {split}: {e}")

    try:
        objects_path = os.path.join(output_dir, f"objects_{split}.json")
        if os.path.exists(objects_path):
            with open(objects_path, "r", encoding="utf-8") as f:
                objects = json.load(f)
            print(f"Loaded existing objects for split {split}")
    except Exception as e:
        print(f"Error loading objects for split {split}: {e}")

    return samples, articles, objects

def convert_items(items: Dict[str, dict], split: str, models: dict, output_dir: str, image_out: str = None, checkpoint_interval: int = 100
                  ) -> Tuple[List[dict], Dict[str, dict], List[dict]]:
    """Convert raw items into dataset entries with extracted features, save to disk, and support resuming."""
    # Load existing checkpoint data
    samples, articles, objects = load_checkpoint(output_dir, split)
    processed_ids = {sample["article_id"] for sample in samples}
    
    # Create feature directory
    feature_dir = os.path.join(image_out, f"{split}_features") if image_out else os.path.join(output_dir, f"{split}_features")
    os.makedirs(feature_dir, exist_ok=True)

    vncore = models["vncore"]
    mtcnn = models["mtcnn"]
    facenet = models["facenet"]
    resnet = models["resnet"]
    resnet_object = models["resnet_object"]
    yolo = models["yolo"]
    tokenizer = models["tokenizer"]
    roberta = models["roberta"]
    embedder = models["embedder"]
    preprocess = models["preprocess"]
    device = models["device"]

    items_to_process = {sid: item for sid, item in items.items() if str(sid) not in processed_ids}
    total_items = len(items_to_process)
    processed_count = 0

    for sid, item in tqdm(items_to_process.items(), desc=f"Processing {split}", unit="item"):
        sample_id = str(sid)
        img_path = item["image_path"]

        # Skip if feature file already exists
        feature_path = os.path.join(feature_dir, f"{sample_id}.pt")
        if os.path.exists(feature_path):
            print(f"Skipping already processed item: {sample_id}")
            continue

        # Copy image only if image_out is provided
        if image_out:
            os.makedirs(image_out, exist_ok=True)
            dst = os.path.join(image_out, f"{sample_id}.jpg")
            if not os.path.exists(dst):
                try:
                    shutil.copy(img_path, dst)
                except OSError as e:
                    print(f"Error copying image {img_path} to {dst}: {e}")
                    continue
            img_path = dst

        try:
            caption = item.get("caption", "")
            context_txt = " ".join(item.get("context", []))
            image = Image.open(img_path).convert("RGB")

            cap_ner = extract_entities(caption, vncore)
            ctx_ner = extract_entities(context_txt, vncore)
            face_info = detect_faces(image, mtcnn, facenet, device)
            obj_feats = detect_objects(img_path, yolo, resnet_object, preprocess, device)
            img_feat = image_feature(image, resnet, preprocess, device)
            art_embed = embedder(context_txt)

            # Save precomputed features
            torch.save({
                "image_feature": torch.tensor(img_feat, dtype=torch.float),
                "face_info": face_info,
                "object_features": torch.tensor(obj_feats, dtype=torch.float) if obj_feats else torch.zeros((1, 2048)),
                "article_embed": art_embed,
                "caption_ner": cap_ner,
                "context_ner": ctx_ner,
            }, feature_path)

            article = {
                "_id": sample_id,
                "context": context_txt,
                "images": [caption],
                "web_url": "",
                "caption_ner": [cap_ner],
                "context_ner": [ctx_ner],
            }
            articles[sample_id] = article

            samples.append({
                "_id": sample_id,
                "article_id": sample_id,
                "split": split,
                "image_index": 0,
                "feature_path": feature_path,
            })

            objects.append({"_id": sample_id, "object_features": obj_feats})

            processed_count += 1

            # Save checkpoint every checkpoint_interval items
            if processed_count % checkpoint_interval == 0:
                save_checkpoint(samples, articles, objects, output_dir, split)
                print(f"Processed {processed_count}/{total_items} items for split {split}")

        except Exception as e:
            print(f"Error processing item {sample_id}: {e}")
            continue

    # Save final checkpoint
    save_checkpoint(samples, articles, objects, output_dir, split)
    print(f"Completed processing {processed_count}/{total_items} items for split {split}")

    return samples, articles, objects

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create ViWiki dataset files")
    parser.add_argument("data_dir", nargs="?", default="/data/npl/ICEK/Wikipedia/content/ver4", help="Directory with train/val/test JSON")
    parser.add_argument("output_dir", nargs="?", default="/data/npl/ICEK/TnT/dataset/content", help="Directory to write converted files")
    parser.add_argument("--image-out", default=None, dest="image_out",
                        help="Optional directory to copy images")
    parser.add_argument("--vncorenlp", default="/data/npl/ICEK/VnCoreNLP",
                        help="Path to VnCoreNLP jar file")
    parser.add_argument("--checkpoint-interval", type=int, default=100,
                        help="Save checkpoint every N items")
    args = parser.parse_args()
    print("LOAD ARGS DONE!")
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = setup_models(device, args.vncorenlp)
    for split in ["train", "val", "test"]:
        split_data = load_split(os.path.join(args.data_dir, f"{split}.json"))
        samples, articles, objects = convert_items(split_data, split, models, args.output_dir, args.image_out, args.checkpoint_interval)
        save_checkpoint(samples, articles, objects, args.output_dir, split)