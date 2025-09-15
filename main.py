import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import os
from tqdm import tqdm
# Import necessary components from previous implementations
from models.decoder import DynamicConvFacesObjectsDecoder
from models.encoder import setup_models
from tell.modules.token_embedders import AdaptiveEmbedding
from tell.modules import AdaptiveSoftmax
Image.MAX_IMAGE_PIXELS = None
import h5py
import logging
class EarlyStopper:
    def __init__(self, patience=5, min_delta=0):
        """
        Early stopper to halt training when validation loss doesn't improve.
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
 
def pad_and_collate(batch):
    from torch.nn.utils.rnn import pad_sequence

    def pad_context_list(tensors, padding_value=0.0):
        return pad_sequence(tensors, batch_first=True, padding_value=padding_value)

    images = [item["image"] for item in batch]
    captions = [item["caption"] for item in batch]
    caption_ids = [item["caption_ids"] for item in batch]
    image_paths = [item["image_path"] for item in batch]

    contexts = {
        "image": pad_context_list([item["contexts"]["image"] for item in batch]),
        "image_mask": pad_context_list([item["contexts"]["image_mask"] for item in batch], padding_value=True),
        "faces": pad_context_list([item["contexts"]["faces"] for item in batch]),
        "faces_mask": pad_context_list([item["contexts"]["faces_mask"] for item in batch], padding_value=True),
        "obj": pad_context_list([item["contexts"]["obj"] for item in batch]),
        "obj_mask": pad_context_list([item["contexts"]["obj_mask"] for item in batch], padding_value=True),
        "article": pad_context_list([item["contexts"]["article"] for item in batch]),
        "article_mask": pad_context_list([item["contexts"]["article_mask"] for item in batch], padding_value=True),
    }

    caption_ids = pad_sequence(caption_ids, batch_first=True, padding_value=0)
    return {
        "image": torch.stack(images),
        "caption": captions,
        "caption_ids": caption_ids,
        "contexts": contexts,
        "image_path": image_paths,
    }

def _h5_worker_init_fn(_):
    # Each worker gets its own H5 handle
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        ds = worker_info.dataset
        if hasattr(ds, "close"):
            ds.close()       # ensure clean state after fork
        # lazy open will occur in __getitem__

class NewsCaptionDataset(Dataset):
    """
    Expects:
      data_dir/
        train.json, val.json, test.json   # metadata (captions, image_path, _id, ...)
      features_dir/  (defaults to data_dir if not given)
        {split}_h5_index.json             # {"<id>": {"shard":"{split}_feat_000.h5","group":"/samples/<id>"}}
        {split}_feat_000.h5, {split}_feat_001.h5, ...

    H5 group per sample contains (uncompressed is fine):
      image_feature (Timg, Dimg)
      object_features (Kobj, Dobj)
      article_embed (L, Dart)
      face_embeddings (Nf, Df)           [optional]
      face_detect_probs (Nf,)            [optional]
      attr: face_n_faces                 [optional]
      ner: JSON string with {caption_ner, context_ner} [optional]
    """
    def __init__(self, data_dir, split, models, features_dir=None, max_length=512):
        super().__init__()
        self.data_dir     = data_dir
        self.features_dir = features_dir or data_dir
        self.split        = split
        self.models       = models
        self.tokenizer    = models.get("tokenizer", None)
        self.max_length   = max_length

        # ---- load metadata list (samples) ----
        meta_path = os.path.join(self.data_dir, f"{split}.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        # Map id -> record for O(1) access
        self.ids = [str(m["_id"]) for m in self.meta]
        # Stable order (numeric if possible)
        try:
            self.ids = sorted(self.ids, key=lambda x: int(x))
        except Exception:
            self.ids = sorted(self.ids)
        self._meta_by_id = {str(m["_id"]): m for m in self.meta}

        # ---- load H5 index ----
        idx_path = os.path.join(self.features_dir, f"{split}_h5_index.json")
        with open(idx_path, "r", encoding="utf-8") as f:
            self._h5_index = json.load(f)

        # per-worker shard handles
        self._open_files = {}  # shard_basename -> h5py.File

    def __len__(self):
        return len(self.ids)

    # ---------- H5 helpers ----------
    def _open_shard(self, shard_basename: str):
        f = self._open_files.get(shard_basename)
        if f is None:
            f = h5py.File(os.path.join(self.features_dir, shard_basename), "r", libver="latest")
            self._open_files[shard_basename] = f
        return f

    def _get_group(self, sample_id: str):
        info = self._h5_index[sample_id]  # {"shard": "..._feat_XXX.h5", "group": "/samples/<id>"}
        return self._open_shard(info["shard"])[info["group"]]

    # ---------- small utils ----------
    @staticmethod
    def _ensure_float32(arr):
        # Accept numpy or torch; return numpy float32 contiguous
        if arr is None:
            return np.zeros((0,), dtype=np.float32)
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        return np.ascontiguousarray(arr, dtype=np.float32)

    @staticmethod
    def _decode_json_dataset(dset):
        raw = dset[()]
        if isinstance(raw, (bytes, bytearray, np.bytes_)):
            raw = raw.decode("utf-8", errors="ignore")
        try:
            return json.loads(raw)
        except Exception:
            return {}

    @staticmethod
    def _pad_article(article_np: np.ndarray, target_len=512):
        """Pad/trim article to target_len along dim 0; return (np.float32 [L,D], mask [L], torch)."""
        L, D = article_np.shape if article_np.ndim == 2 else (0, 0)
        if L >= target_len:
            arr = article_np[:target_len]
            mask = torch.zeros((target_len,), dtype=torch.bool)
        else:
            arr = np.zeros((target_len, D), dtype=np.float32)
            if L > 0:
                arr[:L] = article_np
            mask = torch.zeros((target_len,), dtype=torch.bool)
            mask[L:] = True
        return arr, mask

    def __getitem__(self, idx):
        sid = self.ids[idx]
        rec = self._meta_by_id.get(sid, {})
        caption   = rec.get("caption", "")
        image_path = rec.get("image_path", "")

        # ---- read H5 features ----
        try:
            g = self._get_group(sid)

            img = self._ensure_float32(g["image_feature"][()])
            obj = self._ensure_float32(g["object_features"][()])
            art = self._ensure_float32(g["article_embed"][()])

            if "face_embeddings" in g:
                faces = self._ensure_float32(g["face_embeddings"][()])
            else:
                faces = np.zeros((0, 512), dtype=np.float32)

            # Optional NER (unused here, but handy to keep)
            caption_ner, context_ner = {}, {}
            if "ner" in g:
                ner_obj = self._decode_json_dataset(g["ner"])
                caption_ner = ner_obj.get("caption_ner", {})
                context_ner = ner_obj.get("context_ner", {})

            # Convert to tensors and sanitize NaN/inf
            image_tensor = torch.nan_to_num(torch.as_tensor(img, dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
            obj_tensor   = torch.nan_to_num(torch.as_tensor(obj, dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
            faces_tensor = torch.nan_to_num(torch.as_tensor(faces, dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
            art_np, art_mask = self._pad_article(art, target_len=512)
            art_tensor  = torch.nan_to_num(torch.as_tensor(art_np, dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)

            # Masks
            image_mask = torch.zeros((image_tensor.size(0),), dtype=torch.bool)
            obj_mask   = torch.zeros((obj_tensor.size(0),), dtype=torch.bool) if obj_tensor.size(0) > 0 \
                         else torch.ones((0,), dtype=torch.bool)
            faces_mask = torch.zeros((faces_tensor.size(0),), dtype=torch.bool) if faces_tensor.size(0) > 0 \
                         else torch.ones((0,), dtype=torch.bool)

            contexts = {
                "image": image_tensor,
                "image_mask": image_mask,
                "faces": faces_tensor,
                "faces_mask": faces_mask,
                "obj": obj_tensor,
                "obj_mask": obj_mask,
                "article": art_tensor,
                "article_mask": art_mask,
                # Optionally pass NER through if you need it later
                # "caption_ner": caption_ner,
                # "context_ner": context_ner,
            }

        except Exception as e:
            logging.error(f"[{self.split}] H5 read error for id={sid}: {e}")
            # Safe fallbacks
            contexts = {
                "image": torch.zeros((49, 2048), dtype=torch.float32),
                "image_mask": torch.ones((49,), dtype=torch.bool),
                "faces": torch.zeros((0, 512), dtype=torch.float32),
                "faces_mask": torch.ones((0,), dtype=torch.bool),
                "obj": torch.zeros((0, 2048), dtype=torch.float32),
                "obj_mask": torch.ones((0,), dtype=torch.bool),
                "article": torch.zeros((512, 1024), dtype=torch.float32),
                "article_mask": torch.ones((512,), dtype=torch.bool),
            }

        # Tokenize caption (stay on CPU; move to device later)
        if self.tokenizer is None:
            # Minimal fallback tokenizer to keep pipeline running
            cap_len = min(len(caption.split()), self.max_length)
            caption_ids = torch.randint(low=5, high=30000, size=(cap_len,), dtype=torch.long)
        else:
            caption_ids = self.tokenizer.encode(
                caption, return_tensors="pt", truncation=True, max_length=self.max_length
            )[0]

        return {
            "id": sid,
            "image": contexts["image"],  # convenience alias
            "contexts": contexts,
            "caption_ids": caption_ids,
            "caption": caption,
            "image_path": image_path,
        }

    # ---- cleanup ----
    def close(self):
        for f in self._open_files.values():
            try: f.close()
            except Exception: pass
        self._open_files.clear()

    def __del__(self):
        self.close()


class TransformAndTell(nn.Module):
    def __init__(self, vocab_size, embedder, decoder_params):
        """
        Full Transform and Tell model
        Args:
            vocab_size: Size of vocabulary
            embedder: Token embedder (AdaptiveEmbedding)
            decoder_params: Parameters for DynamicConvFacesObjectsDecoder
        """
        super().__init__()
        self.decoder = DynamicConvFacesObjectsDecoder(
            vocab_size=vocab_size,
            embedder=embedder,
            **decoder_params
        )
        
    def forward(self, prev_target, contexts, incremental_state=None):
        # Transpose context tensors to [seq_len, batch_size, hidden_dim]
        transposed_contexts = {}
        for key in ["image", "article", "faces", "obj"]:
            # [batch_size, seq_len, hidden_dim] -> [seq_len, batch_size, hidden_dim]
            transposed_contexts[key] = contexts[key].transpose(0, 1)
            transposed_contexts[f"{key}_mask"] = contexts[f"{key}_mask"]
            
        return self.decoder(prev_target, transposed_contexts, incremental_state)
    
    def generate(self, contexts, adaptive_softmax, max_length=100, temperature=1.0):
        """Generate caption from contexts using adaptive softmax for sampling"""
        self.eval()
        generated = []
        incremental_state = {}
        
        # Start with <s> token (assuming BOS is padding_idx + 1)
        device = next(self.parameters()).device
        current_token = torch.tensor([[self.decoder.padding_idx + 1]], device=device)
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get decoder hidden state
                hidden, _ = self(current_token, contexts, incremental_state)
                hidden = hidden[:, -1, :]  # [bs, hidden_dim]
                
                # Compute head logits
                head_logits = adaptive_softmax.head(hidden) / temperature
                head_logits = torch.nan_to_num(head_logits, nan=0.0, posinf=100.0, neginf=-100.0)
                head_logits -= head_logits.max(dim=-1, keepdim=True)[0]
                head_probs = torch.softmax(head_logits, dim=-1)
                
                # Sample from head
                cluster_index = torch.multinomial(head_probs, 1).item()
                
                if cluster_index < adaptive_softmax.cutoff[0]:
                    # Head token
                    next_token_id = cluster_index
                else:
                    # Tail cluster
                    tail_idx = cluster_index - adaptive_softmax.cutoff[0]
                    tail_module = adaptive_softmax.tail[tail_idx]
                    tail_logits = tail_module(hidden) / temperature
                    tail_logits = torch.nan_to_num(tail_logits, nan=0.0, posinf=100.0, neginf=-100.0)
                    tail_logits -= tail_logits.max(dim=-1, keepdim=True)[0]
                    tail_probs = torch.softmax(tail_logits, dim=-1)
                    
                    tail_token = torch.multinomial(tail_probs, 1).item()
                    next_token_id = adaptive_softmax.cutoff[tail_idx] + tail_token
                
                next_token = torch.tensor([[next_token_id]], device=device)
                
                # Stop if </s> token (assuming EOS is padding_idx)
                if next_token.item() == self.decoder.padding_idx:
                    break
                
                generated.append(next_token.item())
                current_token = next_token
        
        return generated


def train_model(config):
    """Training pipeline with early stopping"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = setup_models(device, config["vncorenlp_path"])

    train_loader = DataLoader(
        NewsCaptionDataset(config["data_dir"], "train", models),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        collate_fn=pad_and_collate,
        worker_init_fn=_h5_worker_init_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        NewsCaptionDataset(config["data_dir"], "val", models),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=pad_and_collate,
        worker_init_fn=_h5_worker_init_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    print("DATALOADER LOADED!")
    
    embedder = AdaptiveEmbedding(
        vocab_size=config["embedder"]["vocab_size"],
        padding_idx=config["embedder"]["padding_idx"],
        initial_dim=config["embedder"]["initial_dim"],
        factor=config["embedder"]["factor"],
        output_dim=config["embedder"]["output_dim"],
        cutoff=config["embedder"]["cutoff"],
        scale_embeds=True
    )
    print("EMBEDDERR LOADED!")
    model = TransformAndTell(
        vocab_size=config["vocab_size"],
        embedder=embedder,
        decoder_params=config["decoder_params"]
    ).to(device)
    print("TransformAndTell LOADED!")
    # Optimizer and loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=1e-5
    )
    # criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.tokenizer.pad_token_id)
    # Initialize AdaptiveSoftmax criterion
    criterion = AdaptiveSoftmax(
        vocab_size=config["vocab_size"],
        input_dim=config["decoder_params"]["decoder_output_dim"],
        cutoff=config["decoder_params"]["adaptive_softmax_cutoff"],
        factor=config["decoder_params"]["adaptive_softmax_factor"],
        dropout=config["decoder_params"]["adaptive_softmax_dropout"],
        adaptive_inputs=embedder if config["decoder_params"]["tie_adaptive_weights"] else None,
        tie_proj=config["decoder_params"]["tie_adaptive_proj"]
    ).to(device)
    print("CRITERION LOADED!")
    # Early stopper
    early_stopper = EarlyStopper(
        patience=config.get("early_stopping_patience", 5),
        min_delta=config.get("early_stopping_min_delta", 0.01)
    )
    print("START TRAINING!")
    # Training loop with validation
    best_val_loss = float('inf')
    ce_loss = nn.CrossEntropyLoss(ignore_index=config["decoder_params"]["padding_idx"])
    print(f"[DEBUG] num of epoch: {config['epochs']}")
    for epoch in range(config["epochs"]):
        model.train()
        total_tokens = 0
        total_loss_val=0
        # Training phase
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            caption_ids = batch["caption_ids"].to(device)
            contexts = {k: v.to(device) for k, v in batch["contexts"].items()}
            
            optimizer.zero_grad()
            logits, _ = model(caption_ids[:, :-1], contexts)
            
            # Sanitize logits for stability
            logits = torch.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0)
            logits = logits - logits.max(dim=-1, keepdim=True)[0]  # Subtract max for numerical stability
            
            output, new_target = criterion(
                logits.reshape(-1, logits.size(-1)),
                caption_ids[:, 1:].contiguous().view(-1)
            )
            
            total_loss = 0
            for out, tgt in zip(output, new_target):
                if out is not None and tgt is not None:
                    # out: (batch_size, num_classes), tgt: (batch_size,)
                    total_loss += ce_loss(out, tgt)
            
            # Normalize loss by number of tokens
            num_tokens = (caption_ids[:, 1:] != config["decoder_params"]["padding_idx"]).sum()
            normalized_loss = total_loss / num_tokens
            
            # Check for invalid loss before backward
            if torch.isnan(normalized_loss) or torch.isinf(normalized_loss):
                print(f"Skipping batch due to invalid loss: {normalized_loss.item()}")
                continue
            
            # Backward pass
            normalized_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.05)  # Stricter clipping
            optimizer.step()
            
            total_loss_val += total_loss.item()
            total_tokens += num_tokens.item()
        avg_loss = total_loss_val / total_tokens if total_tokens > 0 else 0.0
        print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Train Tokens: {total_tokens}")
        
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_total_tokens = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                caption_ids = batch["caption_ids"].to(device)
                contexts = {k: v.to(device) for k, v in batch["contexts"].items()}
                
                logits, _ = model(caption_ids[:, :-1], contexts)
                
                # Sanitize logits for stability
                logits = torch.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0)
                logits = logits - logits.max(dim=-1, keepdim=True)[0]  # Subtract max for numerical stability
                
                output, new_target = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    caption_ids[:, 1:].contiguous().view(-1)
                )
                
                
                # Compute total loss by summing losses for each cluster
                total_loss = 0
                for out, tgt in zip(output, new_target):
                    if out is not None and tgt is not None:
                        # out: (batch_size, num_classes), tgt: (batch_size,)
                        total_loss += ce_loss(out, tgt)
                
                # Normalize loss by number of tokens
                num_tokens = (caption_ids[:, 1:] != config["decoder_params"]["padding_idx"]).sum()
                normalized_loss = total_loss / num_tokens
                
                val_loss += total_loss.item()
                val_total_tokens += num_tokens.item()
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        print(f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}, Val Tokens: {val_total_tokens}")
        
        # Check for early stopping
        if early_stopper.early_stop(avg_val_loss):
            print(f"Early stopping triggered after {epoch+1} epochs!")
            break
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 
                       os.path.join(config["output_dir"], "best_model.pth"))
            print("Saved new best model")
    
    return model, models

def evaluate_model(model, models, config):
    """Evaluation pipeline"""
    device = next(model.parameters()).device
    
    # Load test dataset
    test_dataset = NewsCaptionDataset(config["data_dir"], "test", models)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=pad_and_collate,
        worker_init_fn=_h5_worker_init_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    # Initialize AdaptiveSoftmax criterion using the model's embedder
    criterion = AdaptiveSoftmax(
        vocab_size=config["vocab_size"],
        input_dim=config["decoder_params"]["decoder_output_dim"],
        cutoff=config["decoder_params"]["adaptive_softmax_cutoff"],
        factor=config["decoder_params"]["adaptive_softmax_factor"],
        dropout=config["decoder_params"]["adaptive_softmax_dropout"],
        adaptive_inputs=model.decoder.embedder if config["decoder_params"]["tie_adaptive_weights"] else None,
        tie_proj=config["decoder_params"]["tie_adaptive_proj"]
    ).to(device)

    model.eval()
    all_predictions = []
    ce_loss = nn.CrossEntropyLoss(ignore_index=config["decoder_params"]["padding_idx"])
    test_weighted_loss = 0.0
    test_total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            caption_ids = batch["caption_ids"].to(device)
            contexts = {k: v.to(device) for k, v in batch["contexts"].items()}
            
            # Forward pass
            logits, _ = model(caption_ids[:, :-1], contexts)
            
            # Sanitize logits for stability
            logits = torch.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0)
            logits = logits - logits.max(dim=-1, keepdim=True)[0]  # Subtract max for numerical stability
            
            # Compute loss using AdaptiveSoftmax output
            output, new_target = criterion(
                logits.reshape(-1, logits.size(-1)),
                caption_ids[:, 1:].contiguous().view(-1)
            )
            
            # Compute weighted batch loss
            batch_weighted_loss = 0.0
            batch_samples = 0
            for out, tgt in zip(output, new_target):
                if out is not None and tgt is not None:
                    cluster_mean_loss = ce_loss(out, tgt)
                    cluster_size = tgt.size(0)
                    batch_weighted_loss += cluster_mean_loss.item() * cluster_size
                    batch_samples += cluster_size
            
            # Normalize by number of tokens
            num_tokens = (caption_ids[:, 1:] != config["decoder_params"]["padding_idx"]).sum().item()
            assert batch_samples == num_tokens, f"Mismatch in token counts: {batch_samples} vs {num_tokens}"
            
            test_weighted_loss += batch_weighted_loss
            test_total_tokens += num_tokens
            
            # Generate captions
            for i in range(len(batch["image"])):
                contexts_i = {k: v[i].unsqueeze(0) for k, v in contexts.items()}
                generated_ids = model.generate(contexts_i, criterion, temperature=0.7)  # Lower temperature for stability
                generated_caption = test_dataset.tokenizer.decode(generated_ids)
                all_predictions.append({
                    "image_path": batch["image_path"][i],
                    "true_caption": batch["caption"][i],
                    "predicted_caption": generated_caption
                })

    if test_total_tokens > 0:
        test_avg_loss = test_weighted_loss / test_total_tokens
    else:
        test_avg_loss = 0.0
    print(f"Test Loss: {test_avg_loss:.4f}, Test Tokens: {test_total_tokens}")
        
    # Save predictions
    with open(os.path.join(config["output_dir"], "predictions.json"), "w",encoding="utf-8") as f:
        json.dump(all_predictions, f, indent=2)
    
    return test_avg_loss, all_predictions

# Standalone loading function for saved model
def load_saved_model(config, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = setup_models(device, config["vncorenlp_path"])
    
    embedder = AdaptiveEmbedding(
        vocab_size=config["embedder"]["vocab_size"],
        padding_idx=config["embedder"]["padding_idx"],
        initial_dim=config["embedder"]["initial_dim"],
        factor=config["embedder"]["factor"],
        output_dim=config["embedder"]["output_dim"],
        cutoff=config["embedder"]["cutoff"],
        scale_embeds=True
    )
    
    model = TransformAndTell(
        vocab_size=config["vocab_size"],
        embedder=embedder,
        decoder_params=config["decoder_params"]
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model, models
if __name__ == "__main__":
    # Configuration (parameters from paper and config.yaml)
    config = {
        "data_dir": "/data2/npl/ICEK/TnT/dataset/content",
        "output_dir": "/data2/npl/ICEK/TnT/output",
        "vncorenlp_path": "/data2/npl/ICEK/VnCoreNLP",
        "vocab_size": 64001,  # From phoBERT-base tokenizer
        "embed_dim": 1024,    # Hidden size
        "batch_size": 8,
        "num_workers": 0,
        "epochs": 400,
        "lr": 5e-5,  # Lowered learning rate for stability
        "embedder": {
            "vocab_size": 64001,
            "initial_dim": 1024,
            "output_dim": 1024,
            "factor": 1,
            "cutoff": [5000, 20000],
            "padding_idx": 0,
            "scale_embeds": True
        },
        "decoder_params": {
            "max_target_positions": 512,
            "dropout": 0.1,
            "share_decoder_input_output_embed": True,
            "decoder_output_dim": 1024,
            "decoder_conv_dim": 1024,
            "decoder_glu": True,
            "decoder_conv_type": "dynamic",
            "weight_softmax": True,
            "decoder_attention_heads": 16,
            "weight_dropout": 0.1,
            "relu_dropout": 0.0,
            "input_dropout": 0.1,
            "decoder_normalize_before": False,
            "attention_dropout": 0.1,
            "decoder_ffn_embed_dim": 4096,
            "decoder_kernel_size_list": [3, 7, 15, 31],
            "adaptive_softmax_cutoff": [5000, 20000],
            "adaptive_softmax_factor": 1,
            "tie_adaptive_weights": True,
            "adaptive_softmax_dropout": 0,
            "tie_adaptive_proj": False,
            "decoder_layers": 4,
            "final_norm": False,
            "padding_idx": 0,
            "swap": False
        },
        "early_stopping_patience": 10, 
        "early_stopping_min_delta": 0.001,
    }
    
    # Run training
    trained_model, models = train_model(config)
    
    # Save final model
    torch.save(trained_model.state_dict(), 
               os.path.join(config["output_dir"], "transform_and_tell_model.pth"))
    
    # Load best model for evaluation
    model_path = os.path.join(config["output_dir"], "best_model.pth")
    if os.path.exists(model_path):
        best_model, models = load_saved_model(config, model_path)
        print(f"Loaded best_model.pth for evaluation")
    else:
        print(f"best_model.pth not found, falling back to transform_and_tell_model.pth")
        best_model = trained_model
    
    # Run evaluation with best model
    test_loss, predictions = evaluate_model(best_model, models, config)
    print(f"Test Loss: {test_loss:.4f}")
    
    # Optionally, print some predictions
    for pred in predictions[:5]:
        print(f"Image: {pred['image_path']}")
        print(f"True Caption: {pred['true_caption']}")
        print(f"Predicted Caption: {pred['predicted_caption']}")
        print()