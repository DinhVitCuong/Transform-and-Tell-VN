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
from models.encoder import setup_models, segment_text, detect_faces, detect_objects, image_feature
from tell.modules.token_embedders import AdaptiveEmbedding
from tell.modules import AdaptiveSoftmax
Image.MAX_IMAGE_PIXELS = None
import h5py
import logging

MAX_FACES   = 4
MAX_OBJECTS = 32

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

    # contexts = {
    #     "image": pad_context_list([item["contexts"]["image"] for item in batch]),
    #     "image_mask": pad_context_list([item["contexts"]["image_mask"] for item in batch], padding_value=True),
    #     "faces": pad_context_list([item["contexts"]["faces"] for item in batch]),
    #     "faces_mask": pad_context_list([item["contexts"]["faces_mask"] for item in batch], padding_value=True),
    #     "obj": pad_context_list([item["contexts"]["obj"] for item in batch]),
    #     "obj_mask": pad_context_list([item["contexts"]["obj_mask"] for item in batch], padding_value=True),
    #     "article": pad_context_list([item["contexts"]["article"] for item in batch]),
    #     "article_mask": pad_context_list([item["contexts"]["article_mask"] for item in batch], padding_value=True),
    # }
    contexts = {
        # -> [B,49,2048]
        "image": pad_context_list([it["contexts"]["image"] for it in batch]),
        # -> [B,49] (bool)
        "image_mask": pad_context_list([it["contexts"]["image_mask"] for it in batch], padding_value=True),

        # -> [B,4,512]
        "faces": pad_context_list([it["contexts"]["faces"] for it in batch]),
        # -> [B,4] (bool)
        "faces_mask": pad_context_list([it["contexts"]["faces_mask"] for it in batch], padding_value=True),

        # -> [B,64,2048]
        "obj": pad_context_list([it["contexts"]["obj"] for it in batch]),
        # -> [B,64] (bool)
        "obj_mask": pad_context_list([it["contexts"]["obj_mask"] for it in batch], padding_value=True),

        # -> [B,L,S,H] (e.g., [B,25,512,H])  — không combine
        "article": pad_context_list([it["contexts"]["article"] for it in batch]),
        # -> [B,S] (bool, True=PAD)
        "article_mask": pad_context_list([it["contexts"]["article_mask"] for it in batch], padding_value=True),
    }
    
    for k in ("image_mask", "faces_mask", "obj_mask", "article_mask"):
        contexts[k] = contexts[k].to(torch.bool)

    caption_ids = pad_sequence(caption_ids, batch_first=True, padding_value=0)
    return {
        "image": torch.stack(images),
        "caption": captions,
        "caption_ids": caption_ids,
        "contexts": contexts,
        "image_path": image_paths,
    }

class NewsCaptionDataset(Dataset):
    def __init__(self, data_dir, split, models, max_length=512):
        super().__init__()
        self.data_dir     = data_dir
        self.split        = split
        self.models       = models
        self.tokenizer    = models.get("tokenizer", None)
        self.max_length   = max_length

        # Load the raw data
        data_file = os.path.join(data_dir, f"{split}.json")
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        if isinstance(self.data, dict):
            # Sort keys numerically when possible
            def _key(k):
                try:
                    return int(k)
                except Exception:
                    return k
            self.items = [self.data[k] for k in sorted(self.data.keys(), key=_key)]
        elif isinstance(self.data, list):
            self.items = self.data
        else:
            raise ValueError(f"Unsupported JSON structure in {data_file}: {type(self.data)}")

        # Stable ids for samplers
        self.ids = list(range(len(self.items)))

        # Setup model for feature extraction
        self.embedder = self.models["embedder"]
        self.device = self.models["device"]
        self.mtcnn = self.models["mtcnn"]
        self.facenet = self.models["facenet"]
        self.resnet = self.models["resnet"]
        self.resnet_object = self.models["resnet_object"]
        self.yolo = self.models["yolo"]
        self.preprocess = self.models["preprocess"]
        self.embedder = self.models["embedder"]
    def __len__(self):
        return len(self.ids)

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


    def __getitem__(self, idx):
        item = self.items[idx]

        segmented_context = []
        context = item.get("paragraphs", [])
        caption = item.get("caption", "")
        for sentence in context:
            segmented_context.append(segment_text(sentence, self.models["vncore"]))
        
        caption = segment_text(caption, self.models["vncore"])
    
        img_id = item["image_path"].split("images/")[-1]
        img_path = f"/data2/npl/ICEK/Wikipedia/images_resized/{img_id}"
        image = Image.open(img_path).convert("RGB")
        # mtcnn = self.models["mtcnn"]; facenet = self.models["facenet"]; resnet = self.models["resnet"]
        # resnet_object = self.models["resnet_object"]; yolo = self.models["yolo"]; preprocess = self.models["preprocess"]
        # embedder = self.models["embedder"]; device = self.models["device"]
        face_feats, face_mask = detect_faces(image, self.mtcnn, self.facenet, self.device, max_faces=MAX_FACES, pad_to=MAX_FACES)
        obj_feats, obj_mask = detect_objects(img_path, self.yolo, self.resnet_object, self.preprocess, self.device, max_det=MAX_OBJECTS, pad_to=MAX_OBJECTS)
        img_feats, img_mask = image_feature(image, self.resnet, self.preprocess, self.device)
        art_feats_b, attn_mask_b = self.embedder(segmented_context)   
        art_feats  = art_feats_b[0].contiguous()                    
        art_mask = (~attn_mask_b[0].bool()).contiguous()
        # art_feats, art_mask = embedder(segmented_context).cpu().numpy()
        
        contexts = {
            "image": img_feats,
            "image_mask": img_mask,
            "faces": face_feats,
            "faces_mask": face_mask,
            "obj": obj_feats,
            "obj_mask": obj_mask,
            "article": art_feats,
            "article_mask": art_mask,
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
            "id": idx,
            "image": contexts["image"],  # convenience alias
            "contexts": contexts,
            "caption_ids": caption_ids,
            "caption": caption,
            "image_path": item["image_path"],
        }


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
        # # Transpose context tensors to [seq_len, batch_size, hidden_dim]
        # transposed_contexts = {}
        # for key in ["image", "article", "faces", "obj"]:
        #     # [batch_size, seq_len, hidden_dim] -> [seq_len, batch_size, hidden_dim]
        #     transposed_contexts[key] = contexts[key].transpose(0, 1)
        #     transposed_contexts[f"{key}_mask"] = contexts[f"{key}_mask"]
            
        return self.decoder(prev_target, contexts, incremental_state)
    
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = setup_models(device, config["vncorenlp_path"])

    train_loader = DataLoader(
        NewsCaptionDataset(config["data_dir"], "train", models),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        collate_fn=pad_and_collate,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        NewsCaptionDataset(config["data_dir"], "val", models),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=pad_and_collate,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    print("[DEBUG] DATALOADER LOADED!")
    
    embedder = AdaptiveEmbedding(
        vocab_size=config["embedder"]["vocab_size"],
        padding_idx=config["embedder"]["padding_idx"],
        initial_dim=config["embedder"]["initial_dim"],
        factor=config["embedder"]["factor"],
        output_dim=config["embedder"]["output_dim"],
        cutoff=config["embedder"]["cutoff"],
        scale_embeds=True
    )
    print("[DEBUG] EMBEDDERR LOADED!")
    model = TransformAndTell(
        vocab_size=config["vocab_size"],
        embedder=embedder,
        decoder_params=config["decoder_params"]
    ).to(device)
    print("TransformAndTell LOADED!")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = AdaptiveSoftmax(
        vocab_size=config["vocab_size"],
        input_dim=config["decoder_params"]["decoder_output_dim"],
        cutoff=config["decoder_params"]["adaptive_softmax_cutoff"],
        factor=config["decoder_params"]["adaptive_softmax_factor"],
        dropout=config["decoder_params"]["adaptive_softmax_dropout"],
        adaptive_inputs=embedder if config["decoder_params"]["tie_adaptive_weights"] else None,
        tie_proj=config["decoder_params"]["tie_adaptive_proj"]
    ).to(device)
    print("[DEBUG] CRITERION LOADED!")
    early_stopper = EarlyStopper(
        patience=config.get("early_stopping_patience", 5),
        min_delta=config.get("early_stopping_min_delta", 0.01)
    )
    print("[DEBUG] START TRAINING!")
    best_val_loss = float('inf')
    ce_loss = nn.CrossEntropyLoss(ignore_index=config["decoder_params"]["padding_idx"])
    print(f"[DEBUG] num of epoch: {config['epochs']}")

    for epoch in range(config["epochs"]):
        model.train()
        total_tokens = 0
        total_loss_val = 0
        # Add diagnostics
        total_grad_norm = 0.0
        num_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            caption_ids = batch["caption_ids"].to(device)
            contexts = {k: v.to(device) for k, v in batch["contexts"].items()}
            
            optimizer.zero_grad()
            logits, _ = model(caption_ids[:, :-1], contexts)
            
            # Check for NaN/inf in logits
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"[DEBUG] NaN or inf in logits at epoch {epoch+1}, batch {num_batches}")
                continue
            
            logits = torch.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0)
            logits = logits - logits.max(dim=-1, keepdim=True)[0]
            
            output, new_target = criterion(
                logits.reshape(-1, logits.size(-1)),
                caption_ids[:, 1:].contiguous().view(-1)
            )
            
            total_loss = 0
            for out, tgt in zip(output, new_target):
                if out is not None and tgt is not None:
                    total_loss += ce_loss(out, tgt)
            
            num_tokens = (caption_ids[:, 1:] != config["decoder_params"]["padding_idx"]).sum()
            if num_tokens == 0:
                print(f"[DEBUG] Zero tokens in batch at epoch {epoch+1}, batch {num_batches}")
                continue
            
            normalized_loss = total_loss / num_tokens
            
            if torch.isnan(normalized_loss) or torch.isinf(normalized_loss):
                print(f"[DEBUG] Invalid loss: {normalized_loss.item()} at epoch {epoch+1}, batch {num_batches}")
                continue
            
            normalized_loss.backward()
            
            # Compute gradient norm before clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            total_grad_norm += grad_norm.item()
            num_batches += 1
            
            optimizer.step()
            
            total_loss_val += total_loss.item()
            total_tokens += num_tokens.item()
        
        avg_loss = total_loss_val / total_tokens if total_tokens > 0 else 0.0
        avg_grad_norm = total_grad_norm / num_batches if num_batches > 0 else 0.0
        print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Train Tokens: {total_tokens}, Avg Grad Norm: {avg_grad_norm:.6f}")
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_total_tokens = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                caption_ids = batch["caption_ids"].to(device)
                contexts = {k: v.to(device) for k, v in batch["contexts"].items()}
                
                logits, _ = model(caption_ids[:, :-1], contexts)
                
                logits = torch.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0)
                logits = logits - logits.max(dim=-1, keepdim=True)[0]
                
                output, new_target = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    caption_ids[:, 1:].contiguous().view(-1)
                )
                
                total_loss = 0
                for out, tgt in zip(output, new_target):
                    if out is not None and tgt is not None:
                        total_loss += ce_loss(out, tgt)
                
                num_tokens = (caption_ids[:, 1:] != config["decoder_params"]["padding_idx"]).sum()
                normalized_loss = total_loss / num_tokens
                
                val_loss += total_loss.item()
                val_total_tokens += num_tokens.item()
        
        avg_val_loss = val_loss / val_total_tokens if val_total_tokens > 0 else 0.0
        print(f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}, Val Tokens: {val_total_tokens}")
        scheduler.step(avg_val_loss)
        if early_stopper.early_stop(avg_val_loss):
            print(f"Early stopping triggered after {epoch+1} epochs!")
            break
        
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
def load_saved_model(config, model_path, models):
    """Load a saved model without reinitializing the models dictionary."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
        "data_dir": "/data2/npl/ICEK/Wikipedia/content/ver4",
        "output_dir": "/data2/npl/ICEK/TnT/output",
        "vncorenlp_path": "/data2/npl/ICEK/VnCoreNLP",
        "vocab_size": 64001,  # From phoBERT-base tokenizer
        "embed_dim": 1024,    # Hidden size
        "batch_size": 8,
        "num_workers": 0,
        "epochs": 40,
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
            "decoder_normalize_before": True,
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
        best_model, models = load_saved_model(config, model_path, models)
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