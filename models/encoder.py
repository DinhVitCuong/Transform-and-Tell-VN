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


# class RobertaEmbedder(torch.nn.Module):
#     def __init__(self, model, tokenizer, segmenter, device):
#         super().__init__()
#         self.model = model
#         self.tokenizer = tokenizer
#         self.segmenter = segmenter
#         self.device = device
#         self.num_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer
#         self.hidden_size = model.config.hidden_size
#         # Learnable weights α_ℓ for each layer
#         self.layer_weights = torch.nn.Parameter(torch.ones(self.num_layers) / self.num_layers)
#     def forward(self, text: str) -> torch.Tensor:
#         sentences = re.split(r'(?<=[\.!?])\s+', text.strip())
#         token_embeds = []
#         total_tokens = 0
#         max_length = self.model.config.max_position_embeddings - 2 
#         for sent in sentences:
#             sent = sent.strip()
#             if not sent or sent.count('·') >= 5:
#                 continue

#             # 1) tokenize with truncation
#             toks = self.tokenizer(
#                 sent,
#                 truncation=True,
#                 max_length=max_length,
#                 max_length=self.model.config.max_position_embeddings,
#                 return_tensors="pt",
#             ).to(self.device)
#             # print("max token ID:", toks["input_ids"].max().item())
#             # print("vocab size   :", self.tokenizer.vocab_size)
#             # 2) build dummy token_type_ids
#             with torch.no_grad():
#                 out = self.model(
#                     **toks,
#                     output_hidden_states=True,
#                 )

#             # 3) layer-weighted sum of hidden states
#             hidden_states = torch.stack(out.hidden_states, dim=0)  # (layers, 1, seq_len, hidden)
#             weights       = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
#             weighted      = (weights * hidden_states).sum(dim=0).squeeze(0)  # (seq_len, hidden)
#             token_embeds.append(weighted)

#         if not token_embeds:
#             return torch.zeros(1, self.hidden_size, device=self.device)

#         # concatenate all sentence pieces
#         return torch.cat(token_embeds, dim=0)  # (total_seq_len, hidden)
class RobertaEmbedder(torch.nn.Module):
    def __init__(self, model, tokenizer, segmenter, device):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.segmenter = segmenter
        self.device = device
        self.num_layers = model.config.num_hidden_layers + 1
        self.hidden_size = model.config.hidden_size
        self.max_seq_len = 512  # Hard limit for RoBERTa
        self.layer_weights = torch.nn.Parameter(torch.ones(self.num_layers) / self.num_layers)

    def forward(self, text: str) -> torch.Tensor:
        # Clean and segment text
        sentences = re.split(r'(?<=[\.!?])\s+', text.strip())
        if not sentences:
            return torch.zeros(1, self.hidden_size, device=self.device)

        # Process sentences with strict length control
        embeddings = []
        current_length = 0
        
        for sent in sentences:
            sent = self.segmenter.word_segment(sent.strip())
            # Tokenize with strict length checking
            toks = self.tokenizer(
                sent,
                truncation=True,
                max_length=self.max_seq_len - current_length,
                return_tensors="pt",
                add_special_tokens=True
            ).to(self.device)

            # Skip if no space left
            if toks.input_ids.size(1) == 0:
                continue

            # Manually clamp position IDs
            position_ids = torch.arange(0, toks.input_ids.size(1), dtype=torch.long, device=self.device)
            position_ids = position_ids.unsqueeze(0)
            toks['position_ids'] = position_ids.clamp(max=self.max_seq_len-1)

            # Get hidden states
            with torch.no_grad():
                outputs = self.model(**toks, output_hidden_states=True)
            
            # Weighted sum of layers
            hidden_states = torch.stack(outputs.hidden_states)  # [layers, 1, seq_len, hidden_size]
            weights = torch.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
            weighted_embedding = (weights * hidden_states).sum(dim=0).squeeze(0)  # [seq_len, hidden_size]

            embeddings.append(weighted_embedding)
            current_length += weighted_embedding.size(0)

            if current_length >= self.max_seq_len:
                break

        if not embeddings:
            return torch.zeros(1, self.hidden_size, device=self.device)

        return torch.cat(embeddings)[:self.max_seq_len]  # Final truncation

def setup_models(device: torch.device, vncorenlp_path="/data/npl/ICEK/VnCoreNLP"):
    py_vncorenlp.download_model(save_dir=vncorenlp_path)
    vncore = py_vncorenlp.VnCoreNLP(annotators=["wseg", "pos", "ner", "parse"], save_dir=vncorenlp_path)
    print("LOADED VNCORENLP!")
    
    # Face detection + embedding
    mtcnn = MTCNN(keep_all=True, device=device)
    facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    print("LOADED FaceNet!")

    # Global image feature
    weights = ResNet152_Weights.IMAGENET1K_V1 
    base = resnet152(weights=weights).eval().to(device)
    # children() trả về: conv1, bn1, relu, maxpool, layer1…layer4, avgpool, fc
    # [-2] loại avgpool, [-1] loại fc
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
    projection = nn.Linear(2048, 1024).to(device).eval()
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
            feat_1024   = projection(feat_tensor)
        # convert to plain Python list:
        feat = feat_1024.cpu().tolist()
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

# def roberta_embed(
#     text: str,
#     tokenizer,
#     model,
#     segmenter,
#     device: torch.device
# ) -> List[float]:
#     # Filter out non-UTF-8 characters and strange symbols
#     try:
#         # Attempt to decode the text, ignore errors
#         text = text.encode('utf-8', 'ignore').decode('utf-8')
#     except Exception as e:
#         print(f"[WARN] Skipping text due to encoding issue: {e}")
#         return [0.0] * model.config.hidden_size  # Return a default vector or skip as needed

#     # Optional: further filter using regex to remove unusual characters
#     text = re.sub(r'[^\x00-\x7F]+', '', text)  # Removes non-ASCII characters

#     sentences = re.split(r'(?<=[\.!?])\s+', text.strip())
#     vecs = []
#     for sent in sentences:
#         sent = sent.strip()
#         if not sent:
#             continue
#         if sent.count('·') >= 5: 
#             continue

#         seg_sents = segmenter.word_segment(sent)
#         seg_text  = " ".join(seg_sents)

#         batch = tokenizer(
#             seg_text,
#             return_tensors="pt",
#             truncation=True,
#             max_length=model.config.max_position_embeddings
#         )

#         batch = {k: v.to(device) for k, v in batch.items()}

#         # Split if input exceeds the max_position_embeddings
#         if len(batch['input_ids'][0]) > model.config.max_position_embeddings:
#             chunk_size = model.config.max_position_embeddings
#             chunks = [batch['input_ids'][0][i:i + chunk_size] for i in range(0, len(batch['input_ids'][0]), chunk_size)]

#             # Create new batches for each chunk and pass through the model
#             for chunk in chunks:
#                 chunk_batch = {k: v.to(device) for k, v in batch.items()}
#                 chunk_batch['input_ids'] = chunk.unsqueeze(0).to(device)

#                 try:
#                     with torch.no_grad():
#                         out = model(**chunk_batch).last_hidden_state
#                 except RuntimeError as e:
#                     print(f"[WARN] phoBERT GPU failed on sent: {repr(sent)} → {e}")
#                     try:
#                         cpu_batch = {k: v.cpu() for k, v in chunk_batch.items()}
#                         model_cpu = model.to("cpu")
#                         with torch.no_grad():
#                             out = model_cpu(**cpu_batch).last_hidden_state
#                         model.to(device)
#                     except Exception as e2:
#                         print(f"[WARN] phoBERT CPU also failed, skip sent → {e2}")
#                         continue

#                 sent_vec = out.mean(dim=1).squeeze(0)  # (D,)
#                 vecs.append(sent_vec)
#         else:
#             try:
#                 with torch.no_grad():
#                     out = model(**batch).last_hidden_state
#             except RuntimeError as e:
#                 print(f"[WARN] phoBERT GPU failed on sent: {repr(sent)} → {e}")
#                 try:
#                     cpu_batch = {k: v.cpu() for k, v in batch.items()}
#                     model_cpu = model.to("cpu")
#                     with torch.no_grad():
#                         out = model_cpu(**cpu_batch).last_hidden_state
#                     model.to(device)
#                 except Exception as e2:
#                     print(f"[WARN] phoBERT CPU also failed, skip sent → {e2}")
#                     continue

#             sent_vec = out.mean(dim=1).squeeze(0)  # (D,)
#             vecs.append(sent_vec)

#     if not vecs:
#         return [0.0] * model.config.hidden_size

#     stacked = torch.stack(vecs, dim=0)     # (n_sent, D)
#     doc_vec = stacked.mean(dim=0)         # (D,)

def load_split(path: str) -> Dict[str, dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def convert_items(items: Dict[str, dict], split: str, models: dict, image_out: str = None
                   ) -> Tuple[List[dict], Dict[str, dict], List[dict]]:
    """Convert raw items into dataset entries with extracted features."""
    samples: List[dict] = []
    articles: Dict[str, dict] = {}
    objects: List[dict] = []

    vncore = models["vncore"]
    mtcnn = models["mtcnn"]
    facenet = models["facenet"]
    resnet = models["resnet"]
    resnet_object = models["resnet_object"]
    yolo = models["yolo"]
    tokenizer = models["tokenizer"]
    roberta = models["roberta"]
    preprocess = models["preprocess"]
    device = models["device"]
    for sid, item in tqdm(items.items(), desc=f"Processing {split}", unit="item"):
        sample_id = str(sid)
        img_path = item["image_path"]
        print(img_path)
        # Optionally copy image
        if image_out:
            os.makedirs(image_out, exist_ok=True)
            dst = os.path.join(image_out, f"{sample_id}.jpg")
            if not os.path.exists(dst):
                try:
                    shutil.copy(img_path, dst)
                    # print("sucess copy")
                except OSError:
                    pass
            img_path = dst

        caption = item.get("caption", "")
        context_txt = " ".join(item.get("context", []))
        try:
            image = Image.open(img_path).convert("RGB")
        except OSError:
            continue

        cap_ner = extract_entities(caption, vncore)
        ctx_ner = extract_entities(context_txt, vncore)
        face_info = detect_faces(image, mtcnn, facenet, device)
        # obj_feats = detect_objects(img_path, yolo, resnet, device)
        obj_feats = detect_objects(img_path, yolo, resnet_object, preprocess, device)
        img_feat = image_feature(image, resnet, preprocess, device)
        art_embed = roberta_embed(context_txt, tokenizer, roberta, vncore, device)

        article = {
            "_id": sample_id,
            "context": context_txt,
            "images": [caption],
            "web_url": "",
            "caption_ner": [cap_ner],
            "context_ner": [ctx_ner],
            "article_embed": art_embed,
        }
        articles[sample_id] = article

        samples.append({
            "_id": sample_id,
            "article_id": sample_id,
            "split": split,
            "image_index": 0,
            "facenet_details": face_info,
            "image_feature": img_feat,
        })

        objects.append({"_id": sample_id, "object_features": obj_feats})
        break

    return samples, articles, objects

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

