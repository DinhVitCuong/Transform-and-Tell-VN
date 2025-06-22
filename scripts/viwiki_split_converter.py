#!/usr/bin/env python
"""Convert simple ViWiki JSON data into the format required by
``ViWiki_face_ner_match.py``.

Input directory must contain ``train.json``, ``val.json`` and ``test.json``.
Each file has the structure::

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

This script generates ``splits.json`` and ``articles.json`` (and an empty
``objects.json``) compatible with the ``ViWiki_face_ner`` dataset reader.  Each
entry is assigned an ``_id`` equal to the key in the source JSON.
Images can optionally be copied into an output directory so that they are named
``<_id>.jpg`` as expected by the dataset reader.
"""

import argparse
import json
import os
import shutil
from typing import Dict, Tuple, List


def load_split(path: str) -> Dict[str, dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def convert_items(items: Dict[str, dict], split: str, image_out: str = None
                   ) -> Tuple[List[dict], List[dict]]:
    samples = []
    articles = []
    for sid, item in items.items():
        sample_id = str(sid)
        if image_out:
            os.makedirs(image_out, exist_ok=True)
            dst = os.path.join(image_out, f"{sample_id}.jpg")
            if not os.path.exists(dst):
                try:
                    shutil.copy(item["image_path"], dst)
                except OSError:
                    pass
        # Build article entry
        article = {
            "_id": sample_id,
            "context": " ".join(item.get("context", [])),
            "images": [item.get("caption", "")],
            "web_url": "",
            "caption_ner": [[]],
            "context_ner": [],
        }
        articles.append(article)
        # Build split sample
        samples.append({
            "_id": sample_id,
            "article_id": sample_id,
            "split": split,
            "image_index": 0,
        })
    return samples, articles


def main() -> None:
    parser = argparse.ArgumentParser(description="Create ViWiki dataset files")
    parser.add_argument("data_dir", help="Directory with train/val/test JSON")
    parser.add_argument("output_dir", help="Directory to write converted files")
    parser.add_argument("--image-out", dest="image_out", default=None,
                        help="Optional directory to copy images")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_samples: List[dict] = []
    all_articles: List[dict] = []

    for split_name in ["train", "val", "test"]:
        path = os.path.join(args.data_dir, f"{split_name}.json")
        items = load_split(path)
        samples, articles = convert_items(items, split_name, args.image_out)
        all_samples.extend(samples)
        all_articles.extend(articles)

    with open(os.path.join(args.output_dir, "splits.json"), "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.output_dir, "articles.json"), "w", encoding="utf-8") as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)

    # create empty objects.json for convenience
    with open(os.path.join(args.output_dir, "objects.json"), "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
