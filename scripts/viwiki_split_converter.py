#!/usr/bin/env python
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
* detect faces with **MTCNN** and extract **FaceNet** embeddings
* detect objects using **YoloV3** and encode them with **ResNet152**
* run **VnCoreNLP** to obtain ``caption_ner`` and ``context_ner``
* extract a global image feature with **ResNet152**
* embed the article text using a **RoBERTa** model

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

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor

from vncorenlp import VnCoreNLP
from transformers import AutoTokenizer, AutoModel

from tell.facenet import MTCNN, InceptionResnetV1
from tell.models.resnet import resnet152
from tell.yolov3.models import Darknet, load_darknet_weights
from tell.yolov3.utils.utils import non_max_suppression, scale_coords
from tell.yolov3.utils.datasets import letterbox


def setup_models(vncorenlp_path: str, device: torch.device):
    """Load all required models for the pipeline."""
    vncore = VnCoreNLP(vncorenlp_path, annotators="wseg,pos,ner", quiet=True)

    mtcnn = MTCNN(keep_all=True, device=str(device))
    facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    resnet = resnet152().to(device).eval()

    yolo_cfg = os.path.join("tell", "yolov3", "cfg", "yolov3-spp.cfg")
    yolo_weights = os.path.join("data", "yolov3-spp-ultralytics.pt")
    model = Darknet(yolo_cfg, img_size=416).to(device)
    if os.path.exists(yolo_weights):
        if yolo_weights.endswith(".pt"):
            model.load_state_dict(torch.load(yolo_weights, map_location=device)["model"])
        else:
            load_darknet_weights(model, yolo_weights)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    roberta = AutoModel.from_pretrained("vinai/phobert-base").to(device).eval()

    preprocess = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return {
        "vncore": vncore,
        "mtcnn": mtcnn,
        "facenet": facenet,
        "resnet": resnet,
        "yolo": model,
        "tokenizer": tokenizer,
        "roberta": roberta,
        "preprocess": preprocess,
        "device": device,
    }


def ner_text(nlp: VnCoreNLP, text: str) -> List[dict]:
    """Run NER on a single text using VnCoreNLP."""
    ner = []
    ann = nlp.annotate(text)
    for sent in ann.get("ner", []):
        for word, label in sent:
            if label != "O":
                ner.append({"text": word, "label": label.split("-")[-1]})
    return ner


def detect_faces(img: Image.Image, mtcnn: MTCNN, facenet: InceptionResnetV1, device: torch.device) -> dict:
    with torch.no_grad():
        faces, probs = mtcnn(img, return_prob=True)
    if faces is None or len(faces) == 0:
        return {"n_faces": 0, "embeddings": [], "detect_probs": []}
    faces = faces.to(device)
    with torch.no_grad():
        embeds = facenet(faces)
    return {
        "n_faces": len(embeds),
        "embeddings": embeds.cpu().tolist(),
        "detect_probs": probs[: len(embeds)].tolist(),
    }


def detect_objects(image_path: str, model: Darknet, resnet: torch.nn.Module, device: torch.device) -> List[List[float]]:
    img0 = cv2.imread(image_path)
    if img0 is None:
        return []
    img = letterbox(img0, new_shape=416)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img)[0]
        pred = non_max_suppression(pred, 0.3, 0.6)[0]

    feats = []
    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img0.shape).round()
        pil = Image.open(image_path).convert("RGB")
        for (*xyxy, conf, cls) in pred:
            obj = pil.crop((int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))).resize((224, 224))
            tensor = Compose([
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])(obj).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = resnet(tensor, pool=True)
            feats.append(feat.squeeze(0).cpu().tolist())
    return feats


def image_feature(img: Image.Image, resnet: torch.nn.Module, preprocess: Compose, device: torch.device) -> List[float]:
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = resnet(tensor, pool=True)
    return feat.squeeze(0).cpu().tolist()


def roberta_embed(text: str, tokenizer: AutoTokenizer, model: AutoModel, device: torch.device) -> List[float]:
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        hidden = model(**tokens).last_hidden_state.mean(dim=1)
    return hidden.squeeze(0).cpu().tolist()

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
    yolo = models["yolo"]
    tokenizer = models["tokenizer"]
    roberta = models["roberta"]
    preprocess = models["preprocess"]
    device = models["device"]

    for sid, item in items.items():
        sample_id = str(sid)
        img_path = item["image_path"]

        # Optionally copy image
        if image_out:
            os.makedirs(image_out, exist_ok=True)
            dst = os.path.join(image_out, f"{sample_id}.jpg")
            if not os.path.exists(dst):
                try:
                    shutil.copy(img_path, dst)
                except OSError:
                    pass
            img_path = dst

        caption = item.get("caption", "")
        context_txt = " ".join(item.get("context", []))

        try:
            image = Image.open(img_path).convert("RGB")
        except OSError:
            continue

        cap_ner = ner_text(vncore, caption)
        ctx_ner = ner_text(vncore, context_txt)

        face_info = detect_faces(image, mtcnn, facenet, device)
        obj_feats = detect_objects(img_path, yolo, resnet, device)
        img_feat = image_feature(image, resnet, preprocess, device)
        art_embed = roberta_embed(context_txt, tokenizer, roberta, device)

        article = {
            "_id": sample_id,
            "context": context_txt,
            "images": [caption],
            "web_url": "",
            "caption_ner": [cap_ner],
            "context_ner": ctx_ner,
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

    return samples, articles, objects


def main() -> None:
    parser = argparse.ArgumentParser(description="Create ViWiki dataset files")
    parser.add_argument("data_dir", help="Directory with train/val/test JSON")
    parser.add_argument("output_dir", help="Directory to write converted files")
    parser.add_argument("--image-out", dest="image_out", default=None,
                        help="Optional directory to copy images")
    parser.add_argument("--vncorenlp", dest="vncorenlp", required=True,
                        help="Path to VnCoreNLP jar file")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = setup_models(args.vncorenlp, device)

    all_samples: List[dict] = []
    article_map: Dict[str, dict] = {}
    object_entries: List[dict] = []

    for split_name in ["train", "val", "test"]:
        path = os.path.join(args.data_dir, f"{split_name}.json")
        items = load_split(path)
        samples, articles, objs = convert_items(items, split_name, models, args.image_out)
        all_samples.extend(samples)
        article_map.update(articles)
        object_entries.extend(objs)

    with open(os.path.join(args.output_dir, "splits.json"), "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.output_dir, "articles.json"), "w", encoding="utf-8") as f:
        json.dump(list(article_map.values()), f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.output_dir, "objects.json"), "w", encoding="utf-8") as f:
        json.dump(object_entries, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
