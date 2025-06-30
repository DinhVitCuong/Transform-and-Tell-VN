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
from torchvision.transforms import Compose, Normalize, ToTensor
import torch.nn as nn

import py_vncorenlp
from transformers import AutoTokenizer, AutoModel
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision.models import resnet152, ResNet152_Weights
from torchvision.transforms import Compose, Normalize, ToTensor
from ultralytics import YOLO


def setup_models(device: torch.device, vncorenlp_path="/data/npl/ICEK/VnCoreNLP"):
    py_vncorenlp.download_model(save_dir=vncorenlp_path)
    vncore = py_vncorenlp.VnCoreNLP(annotators=["wseg", "pos", "ner", "parse"], save_dir=vncorenlp_path)
    print("LOADED VNCORENLP!")

    # Face detection + embedding
    mtcnn = MTCNN(keep_all=True, device=device)
    facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    print("LOADED FaceNet!")

    # Global image feature
    base = resnet152(pretrained=True).eval().to(device)
    # children() trả về: conv1, bn1, relu, maxpool, layer1…layer4, avgpool, fc
    # [-2] loại avgpool, [-1] loại fc
    resnet = nn.Sequential(*list(base.children())[:-2]).eval().to(device)
    
    resnet_object = nn.Sequential(*list(base.children())[:-1]).eval().to(device)
    print("LOADED ResNet152!")

    yolo = YOLO("yolov8m.pt")  # tải weight tự động lần đầu
    yolo.fuse()                # fuse model for speed
    print("LOADED YOLOv5!")
    phoBERTlocal = "/data/npl/ICEK/TnT/phoBERT/phobert-base"

    tokenizer = AutoTokenizer.from_pretrained(phoBERTlocal, local_files_only=True)
    roberta     = AutoModel    .from_pretrained(phoBERTlocal, use_safetensors=True, local_files_only=True).to(device).eval()
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
        detections.append({
            "label": model.names[cls],
            "confidence": conf,
            "feature": feat
        })
    return detections

def image_feature(img: Image.Image, resnet: torch.nn.Module, preprocess: Compose, device: torch.device) -> List[float]:
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        fmap = resnet(tensor)                         # (1,2048,7,7)
    fmap = fmap.squeeze(0)                            # (2048,7,7)
    # move channel to last dim → (7,7,2048), then flatten to (49,2048)
    patches = fmap.permute(1, 2, 0).reshape(-1, 2048)  # (49,2048)
    return patches.cpu().tolist() 

def roberta_embed(
    text: str,
    tokenizer,
    model,
    segmenter,
    device: torch.device
) -> List[float]:
    """
    - text: cả đoạn context (có thể rất dài)
    - tokenizer/model: vinai/phobert-base đã load sẵn
    - segmenter: VnCoreNLP(tokenize) đã khởi tạo
    - device: cpu hoặc cuda
    Trả về: embedding 1-D (hidden_size) cho toàn bộ đoạn text
    """
    sentences = re.split(r'(?<=[\.!?])\s+', text.strip())
    vecs = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if sent.count('·') >= 5: 
            continue

        seg_sents = segmenter.word_segment(sent)
        seg_text  = " ".join(seg_sents)

        batch = tokenizer(
            seg_text,
            return_tensors="pt",
            truncation=True,
            max_length=model.config.max_position_embeddings
        )
        batch = {k: v.to(device) for k, v in batch.items()}

        try:
            with torch.no_grad():
                out = model(**batch).last_hidden_state  # (1, L, D)
        except RuntimeError as e:
            print(f"[WARN] RoBERTa GPU failed on sent: {repr(sent)} → {e}")
            try:
                cpu_batch = {k: v.cpu() for k, v in batch.items()}
                model_cpu = model.to("cpu")
                with torch.no_grad():
                    out = model_cpu(**cpu_batch).last_hidden_state
                model.to(device)
            except Exception as e2:
                print(f"[WARN] RoBERTa CPU also failed, skip sent → {e2}")
                continue

        sent_vec = out.mean(dim=1).squeeze(0)  # (D,)
        vecs.append(sent_vec)

    if not vecs:
        return [0.0] * model.config.hidden_size

    stacked = torch.stack(vecs, dim=0)     # (n_sent, D)
    doc_vec = stacked.mean(dim=0)         # (D,)
    return doc_vec.cpu().tolist()

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

