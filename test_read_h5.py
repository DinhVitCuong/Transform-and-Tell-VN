import os, json, h5py, numpy as np

def load_index(output_dir: str, split: str):
    idx_path = os.path.join(output_dir, f"{split}_h5_index.json")
    with open(idx_path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_sample(output_dir: str, split: str, sample_id: str):
    index = load_index(output_dir, split)
    info = index[sample_id]  # {"shard": "..._feat_000.h5", "group": "/samples/<id>"}
    shard_path = os.path.join(output_dir, info["shard"])
    with h5py.File(shard_path, "r") as h5:
        g = h5[info["group"]]
        data = {
            "image_feature":     g["image_feature"][()],       # np.ndarray
            "face_embeddings":   g["face_embeddings"][()],
            "face_detect_probs": g["face_detect_probs"][()],
            "face_n_faces":      int(g.attrs.get("face_n_faces", 0)),
            "object_features":   g["object_features"][()],
            "article_embed":     g["article_embed"][()],
        }
        ner_raw = g["ner"][()]
        ner_json = ner_raw.decode("utf-8") if isinstance(ner_raw, (bytes, bytearray)) else ner_raw
        data.update(json.loads(ner_json))  # adds: caption_ner, context_ner
        return data

# usage
idx = load_index("/data2/npl/ICEK/TnT/dataset/content", "demo10")
any_id = next(iter(idx))
sample = read_sample("/data2/npl/ICEK/TnT/dataset/content", "demo10", any_id)
print(sample["image_feature"].shape, sample["context_ner"])