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
from typing import Dict, Optional, Tuple, Any
from tell.modules import AdaptiveSoftmax
from typing import Dict, Optional, Tuple, Any, List
Image.MAX_IMAGE_PIXELS = None
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
 
def _to_cpu(t, dtype=None):
        if isinstance(t, torch.Tensor):
            t = t.detach().to("cpu").contiguous()
            if dtype is not None:
                t = t.to(dtype)
        return t

def pad_and_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    from torch.nn.utils.rnn import pad_sequence

    def pad_list(xs: List[torch.Tensor], pad_val=0, dtype=None) -> torch.Tensor:
        if dtype is not None:
            xs = [x.to(dtype) for x in xs]
        return pad_sequence(xs, batch_first=True, padding_value=pad_val)

    contexts = {
        "image":       torch.stack([b["contexts"]["image"] for b in batch]),                 # [B,49,2048]
        "image_mask":  torch.stack([b["contexts"]["image_mask"] for b in batch]).to(bool),   # [B,49]
        "faces":       pad_list([b["contexts"]["faces"] for b in batch], 0.0),               # [B,F,512]
        "faces_mask":  pad_list([b["contexts"]["faces_mask"] for b in batch], True).to(bool),# [B,F]
        "obj":         pad_list([b["contexts"]["obj"] for b in batch], 0.0),                 # [B,O,2048]
        "obj_mask":    pad_list([b["contexts"]["obj_mask"] for b in batch], True).to(bool),  # [B,O]
        "article":     pad_list([b["contexts"]["article"] for b in batch], 0.0),             # [B,S,1024]
        "article_mask":pad_list([b["contexts"]["article_mask"] for b in batch], True).to(bool),# [B,S]
    }

    caption_ids = pad_list([b["caption_ids"] for b in batch], 0, dtype=torch.long)           # [B,T]
    ids = torch.tensor([b["id"] for b in batch], dtype=torch.long)

    return {"contexts": contexts, "caption_ids": caption_ids, "ids": ids}

class NewsCaptionDataset(Dataset):
    def __init__(self, data_dir, cache_dir, split, models, max_length=256):
        super().__init__()
        self.data_dir     = data_dir
        self.split        = split
        self.models       = models
        self.tokenizer    = models.get("tokenizer", None)
        self.max_length   = max_length
        self.cache_dir    = cache_dir

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
    
    def _load_npz(self, path: str) -> Dict[str, np.ndarray]:
        with np.load(path, allow_pickle=False) as z:
            needed = [
                "image","image_mask","faces","faces_mask","obj","obj_mask",
                "article","article_mask","caption_ids"
            ]
            for n in needed:
                if n not in z.files:
                    raise KeyError(f"{os.path.basename(path)} missing '{n}'")
            if z["caption_ids"].size == 0:
                raise ValueError(f"{os.path.basename(path)} has empty caption_ids; re-run precompute with tokenizer.")
            return {k: z[k] for k in z.files}
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        fname = f"{idx}.npz"  
        temp_path =  os.path.join(self.split_dir, fname)
        path = os.path.join(self.cache_dir, temp_path)
        arr = self._load_npz(path)

        # Cast to tensors (use fp32 at train time; masks -> bool)
        image        = torch.from_numpy(arr["image"]).float()              # [49,2048]
        image_mask   = torch.from_numpy(arr["image_mask"]).to(torch.bool)  # [49]

        faces        = torch.from_numpy(arr["faces"]).float()              # [F,512]
        faces_mask   = torch.from_numpy(arr["faces_mask"]).to(torch.bool)  # [F]

        obj          = torch.from_numpy(arr["obj"]).float()                # [O,2048]
        obj_mask     = torch.from_numpy(arr["obj_mask"]).to(torch.bool)    # [O]

        # article was saved as fp16; cast to fp32 for numerical stability
        article_np   = arr["article"].astype(np.float32, copy=False)
        article      = torch.from_numpy(article_np).float()                # [S,1024]
        article_mask = torch.from_numpy(arr["article_mask"]).to(torch.bool)# [S]

        caption_ids  = torch.from_numpy(arr["caption_ids"]).long()         # [T]

        return {
            "id": idx,
            "contexts": {
                "image": image, "image_mask": image_mask,
                "faces": faces, "faces_mask": faces_mask,
                "obj": obj, "obj_mask": obj_mask,
                "article": article, "article_mask": article_mask,
            },
            "caption_ids": caption_ids,
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
    
    @torch.no_grad()
    def generate(
        self,
        contexts: Dict[str, torch.Tensor],
        start_ids: torch.LongTensor,
        max_len: int,
        eos_id: int,
        pad_id: int,
        *,
        criterion: Optional[Any] = None,
        decoding: str = "greedy",           # "greedy" | "beam"
        temperature: float = 1.0,
        beam_size: int = 4,
        length_penalty: float = 1.0,        # >1.0 favors longer, <1.0 shorter
        return_logprobs: bool = False,
    ) -> Tuple[torch.LongTensor, Optional[torch.Tensor]]:
        """
        Decode captions using AdaptiveSoftmax without constructing a dense V projection.

        Args:
          contexts: batch-aligned dict of encoder/visual/text features (B, ...).
          start_ids: BOS tokens, shape (B, 1).
          max_len: total output length INCLUDING the given start token(s).
          eos_id/pad_id: vocabulary ids.
          criterion: must expose .log_prob(h) where h is (N, D).
          decoding: "greedy" or "beam".
          temperature: >0; applied on the last-step hidden before log_prob.
          beam_size: used when decoding == "beam".
          length_penalty: beam scoring denominator exponent (GNMT-style).
          return_logprobs: also return per-sample final logprob scores.

        Returns:
          (tokens[B, max_len], scores[B]) if return_logprobs else (tokens, None)
        """
        if criterion is None or not hasattr(criterion, "log_prob"):
            raise RuntimeError(
                "AdaptiveSoftmax criterion with `.log_prob(hidden)` is required for generation."
            )

        device = start_ids.device
        decoding = decoding.lower()
        assert decoding in {"greedy", "beam"}

        def _length_penalty_fn(t: torch.Tensor, alpha: float) -> torch.Tensor:
            # why: GNMT length norm to avoid short-hypothesis bias
            if alpha == 1.0:
                return torch.ones_like(t, dtype=torch.float, device=device)
            return ((5.0 + t.float()) ** alpha) / (6.0 ** alpha)

        def _log_probs(last_h: torch.Tensor) -> torch.Tensor:
            # why: keep inference numerically stable and respect temperature
            temp = max(temperature, 1e-5)
            last_h = last_h / temp
            return criterion.log_prob(last_h)  # [N, V], log-probabilities

        def _tile_ctx(ctx: Dict[str, torch.Tensor], k: int) -> Dict[str, torch.Tensor]:
            if k == 1:
                return ctx
            out = {}
            for name, tens in ctx.items():
                # (B, ...) -> (B*k, ...)
                out[name] = tens.repeat_interleave(k, dim=0)
            return out

        B = start_ids.size(0)
        max_len = int(max_len)
        assert start_ids.dim() == 2 and start_ids.size(1) >= 1, "start_ids shape must be (B, T0>=1)"

        if decoding == "greedy":
            seq = start_ids  # (B, t)
            finished = torch.zeros(B, dtype=torch.bool, device=device)
            scores = torch.zeros(B, dtype=torch.float, device=device)

            while seq.size(1) < max_len:
                logits, _ = self(seq, contexts)         # [B, t, D]
                last_h = logits[:, -1, :]               # [B, D]
                lp = _log_probs(last_h)                 # [B, V]

                next_tok = lp.argmax(dim=-1)            # [B]
                # why: do not change finished sequences further
                next_tok = torch.where(finished, torch.full_like(next_tok, pad_id), next_tok)
                step_lp = lp.gather(1, next_tok.unsqueeze(1)).squeeze(1)
                step_lp = torch.where(finished, torch.zeros_like(step_lp), step_lp)
                scores += step_lp

                seq = torch.cat([seq, next_tok.unsqueeze(1)], dim=1)
                finished = finished | (next_tok == eos_id)
                if bool(finished.all()):
                    break

            # pad tail if early stopped
            if seq.size(1) < max_len:
                tail = seq.new_full((B, max_len - seq.size(1)), pad_id)
                seq = torch.cat([seq, tail], dim=1)

            return (seq, scores) if return_logprobs else (seq, None)

        # ---- Beam search ----
        beam = int(max(1, beam_size))
        # bootstrap
        beam_seq = start_ids.repeat_interleave(beam, dim=0)          # (B*beam, t0)
        beam_scores = torch.zeros(B * beam, dtype=torch.float, device=device)
        beam_finished = torch.zeros(B * beam, dtype=torch.bool, device=device)
        ctx_beam = _tile_ctx(contexts, beam)                         # tile all contexts

        # step 1: expand from BOS (gives better initial diversity)
        cur_len = beam_seq.size(1)
        logits, _ = self(beam_seq, ctx_beam)                          # (B*beam, t0, D)
        lp = _log_probs(logits[:, -1, :]).view(B, beam, -1)           # (B, beam, V)
        # collapse to beam*V for each batch
        topk = min(beam, lp.size(-1))
        next_scores, next_ids = torch.topk(lp[:, 0, :], k=topk, dim=-1)  # from the single true beam at t0
        # seed beams
        beam_seq = beam_seq.view(B, beam, cur_len)
        beam_seq = beam_seq[:, :topk, :]                               # (B, topk, t0)
        next_tokens = next_ids                                         # (B, topk)
        beam_seq = torch.cat([beam_seq, next_tokens.unsqueeze(-1)], dim=-1)  # (B, topk, t0+1)
        beam_scores = next_scores                                      # (B, topk)
        beam_finished = next_tokens.eq(eos_id)                         # (B, topk)

        # if requested beam_size > topk, pad beams deterministically
        if topk < beam:
            pad_beams = beam - topk
            pad_seq = beam_seq[:, :1, :].expand(B, pad_beams, beam_seq.size(-1)).contiguous()
            pad_scores = torch.full((B, pad_beams), -1e9, device=device)  # dominated
            pad_finished = torch.ones((B, pad_beams), dtype=torch.bool, device=device)
            beam_seq = torch.cat([beam_seq, pad_seq], dim=1)
            beam_scores = torch.cat([beam_scores, pad_scores], dim=1)
            beam_finished = torch.cat([beam_finished, pad_finished], dim=1)

        beam_seq = beam_seq.view(B * beam, cur_len + 1)
        beam_scores = beam_scores.view(B * beam)
        beam_finished = beam_finished.view(B * beam)

        # main loop
        while beam_seq.size(1) < max_len:
            logits, _ = self(beam_seq, ctx_beam)                     # (B*beam, t, D)
            lp = _log_probs(logits[:, -1, :])                        # (B*beam, V)

            # freeze finished beams: only allow PAD with zero delta
            if beam_finished.any():
                lp[beam_finished] = -float("inf")
                lp[beam_finished, pad_id] = 0.0

            # candidate scores
            cand_scores = (beam_scores.unsqueeze(1) + lp)            # (B*beam, V)
            cand_scores = cand_scores.view(B, beam, -1)              # (B, beam, V)
            cand_scores = cand_scores.view(B, -1)                    # (B, beam*V)

            next_scores, next_pos = torch.topk(cand_scores, k=beam, dim=-1)  # (B, beam)
            vocab_size = lp.size(-1)
            next_beam_idx = next_pos // vocab_size                   # (B, beam)
            next_tokens = next_pos % vocab_size                      # (B, beam)

            # gather previous beam sequences
            old_seq = beam_seq.view(B, beam, -1)                     # (B, beam, t)
            gather_idx = next_beam_idx.unsqueeze(-1).expand(-1, -1, old_seq.size(-1))
            new_seq = torch.gather(old_seq, 1, gather_idx)           # (B, beam, t)
            new_seq = torch.cat([new_seq, next_tokens.unsqueeze(-1)], dim=-1)  # (B, beam, t+1)

            # update state
            beam_seq = new_seq.view(B * beam, -1)
            beam_scores = next_scores.view(B * beam)
            beam_finished = torch.gather(
                beam_finished.view(B, beam), 1, next_beam_idx
            ).contiguous().view(B * beam) | (next_tokens.view(-1) == eos_id)

            if bool(beam_finished.view(B, beam).all()):
                break

        # choose best beam per batch with length penalty
        t = beam_seq.size(1)
        lp_den = _length_penalty_fn(torch.full((B, beam), t, device=device), length_penalty)
        final_scores = beam_scores.view(B, beam) / lp_den
        best = final_scores.argmax(dim=-1)                            # (B,)
        best_idx = best + torch.arange(B, device=device) * beam
        out = beam_seq.index_select(0, best_idx)                      # (B, t)

        # pad tail if needed
        if out.size(1) < max_len:
            tail = out.new_full((B, max_len - out.size(1)), pad_id)
            out = torch.cat([out, tail], dim=1)

        # ensure sequences after first EOS are padded (clean output)
        eos_mask = (out == eos_id)
        if eos_mask.any():
            # first EOS position per row
            first_eos = torch.where(eos_mask, torch.arange(out.size(1), device=device), out.new_full(out.shape, out.size(1))).min(dim=1).values
            for i in range(B):
                pos = int(first_eos[i].item())
                if pos < out.size(1) - 1:
                    out[i, pos + 1 :] = pad_id

        final = final_scores.gather(1, best.unsqueeze(1)).squeeze(1)
        return (out, final) if return_logprobs else (out, None)

def train_model(config):

    print("[DEBUG] STARTING PREP")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cuda"
    models = setup_models(device, config["vncorenlp_path"])

    train_loader = DataLoader(
        NewsCaptionDataset(config["data_dir"], config["cache_dir"], "train", models),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        collate_fn=pad_and_collate,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        NewsCaptionDataset(config["data_dir"], config["cache_dir"], "val", models),
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
        min_delta=config.get("early_stopping_min_delta", 0.001)
    )
    print("[DEBUG] START TRAINING!")
    best_val_loss = float('inf')
    ce_loss = nn.CrossEntropyLoss(
        ignore_index=config["decoder_params"]["padding_idx"], reduction="sum"
    )

    print(f"[DEBUG] num of epoch: {config['epochs']}")

    for epoch in range(config["epochs"]):
        model.train()
        total_token_nll = 0.0  # summed NLL over the epoch
        total_tokens = 0
        total_grad_norm = 0.0
        num_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            # caption_ids = batch["caption_ids"].to(device)
            # contexts = {k: v.to(device) for k, v in batch["contexts"].items()}
            caption_ids = batch["caption_ids"].to(device, non_blocking=True)
            contexts    = {k: v.to(device, non_blocking=True) for k, v in batch["contexts"].items()}
            
            targets = caption_ids[:, 1:].contiguous().view(-1)
            num_tokens = (targets != config["decoder_params"]["padding_idx"]).sum()
            if num_tokens.item() == 0:
                continue

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(caption_ids[:, :-1], contexts)
            
            logits = torch.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0)
            logits = logits - logits.max(dim=-1, keepdim=True)[0]
            
            # Flatten to [B*(T-1), D] for AdaptiveSoftmax
            flat_logits = logits.reshape(-1, logits.size(-1))
            outputs, new_targets = criterion(flat_logits, targets)
            
            sum_loss = 0.0
            for out, tgt in zip(outputs, new_targets):
                if out is not None and tgt is not None and tgt.numel() > 0:
                    sum_loss = sum_loss + ce_loss(out, tgt)

            loss = sum_loss / num_tokens
            if not torch.isfinite(loss):
                print(f"[WARN] Non-finite loss; skipping batch.")
                continue
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # why: stabilize updates
            optimizer.step()

            total_grad_norm += float(grad_norm)
            num_batches += 1
            total_token_nll += float(sum_loss.detach())
            total_tokens += int(num_tokens.detach())
        
        train_avg_nll = (total_token_nll / total_tokens) if total_tokens > 0 else 0.0
        avg_grad_norm = (total_grad_norm / num_batches) if num_batches > 0 else 0.0
        print(f"Epoch {epoch+1} | Train NLL/token: {train_avg_nll:.4f} | Tokens: {total_tokens} | AvgGradNorm: {avg_grad_norm:.6f}")

        # ===== Validation (uses the same correct weighting) =====
        model.eval()
        val_sum_nll = 0.0
        val_tokens = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                caption_ids = batch["caption_ids"].to(device, non_blocking=True)
                contexts    = {k: v.to(device, non_blocking=True) for k, v in batch["contexts"].items()}

                targets = caption_ids[:, 1:].contiguous().view(-1)
                num_tokens = (targets != config["decoder_params"]["padding_idx"]).sum()
                if num_tokens.item() == 0:
                    continue

                logits, _ = model(caption_ids[:, :-1], contexts)
                logits = torch.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0)
                logits = logits - logits.max(dim=-1, keepdim=True)[0]
                flat_logits = logits.reshape(-1, logits.size(-1))

                outputs, new_targets = criterion(flat_logits, targets)

                batch_sum_nll = 0.0
                for out, tgt in zip(outputs, new_targets):
                    if out is not None and tgt is not None and tgt.numel() > 0:
                        batch_sum_nll = batch_sum_nll + ce_loss(out, tgt)

                val_sum_nll += float(batch_sum_nll)
                val_tokens += int(num_tokens)

        val_nll = (val_sum_nll / val_tokens) if val_tokens > 0 else 0.0
        print(f"Epoch {epoch+1} | Val NLL/token: {val_nll:.4f} | Tokens: {val_tokens}")

        scheduler.step(val_nll)
        if early_stopper.early_stop(val_nll):
            print(f"Early stopping triggered after {epoch+1} epochs!")
            break

        if val_nll < best_val_loss:
            best_val_loss = val_nll
            os.makedirs(config["output_dir"], exist_ok=True)
            torch.save(model.state_dict(), os.path.join(config["output_dir"], "best_model.pth"))
            print("Saved new best model")

    return model, models

def evaluate_model(model, models, config):
    """
    Evaluate with the new generate() and emit a predictions file that matches
    the input structure, adding a 'predict' field per item.
    """
    import os, json
    from tqdm import tqdm
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    device = next(model.parameters()).device

    # --- Dataset / Loader ---
    test_dataset = NewsCaptionDataset(config["data_dir"], config["cache_dir"], "test", models)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get("batch_size", 32),
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        collate_fn=pad_and_collate,
        pin_memory=torch.cuda.is_available(),
    )

    # --- Loss & Criterion (tie to model embedder like training) ---
    pad_idx = int(config["decoder_params"].get("padding_idx", 0))
    ce_loss = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="sum")
    criterion = AdaptiveSoftmax(
        vocab_size=config["vocab_size"],
        input_dim=config["decoder_params"]["decoder_output_dim"],
        cutoff=config["decoder_params"]["adaptive_softmax_cutoff"],
        factor=config["decoder_params"]["adaptive_softmax_factor"],
        dropout=config["decoder_params"]["adaptive_softmax_dropout"],
        adaptive_inputs=model.decoder.embedder if config["decoder_params"].get("tie_adaptive_weights") else None,
        tie_proj=config["decoder_params"].get("tie_adaptive_proj", False),
    ).to(device)

    # --- IDs & decoding params ---
    bos_id = int(config["decoder_params"].get("bos_idx", 1))
    eos_id = int(config["decoder_params"].get("eos_idx", 2))
    pad_id = pad_idx
    max_len = int(config.get("max_len", 40))
    decoding = str(config.get("decoding", "beam")).lower()
    beam_size = int(config.get("beam_size", 4))
    length_penalty = float(config.get("length_penalty", 1.0))
    temperature = float(config.get("temperature", 1.0))

    # --- Helpers: decode tokens back to text ---
    tok = models.get("tokenizer")
    vocab = models.get("vocab")

    def _decode_ids(ids):
        # strip after EOS, drop PAD
        clean = []
        for t in ids:
            if t == pad_id:
                continue
            if t == eos_id:
                break
            clean.append(int(t))
        if tok is not None:
            if hasattr(tok, "DecodeIds"):   # SentencePiece
                return tok.DecodeIds(clean)
            if hasattr(tok, "decode"):
                try:
                    return tok.decode(clean)
                except Exception:
                    pass
        if vocab is not None and hasattr(vocab, "itos"):
            pieces = [vocab.itos[i] for i in clean if 0 <= i < len(vocab.itos)]
            s = " ".join(pieces)
            return (
                s.replace(" @@ ", "")  # BPE cleanup if any
                 .replace(" ,", ",").replace(" .", ".")
                 .replace(" !", "!").replace(" ?", "?")
                 .strip()
            )
        return " ".join(map(str, clean))

    # --- Accumulators ---
    model.eval()
    test_sum_nll = 0.0
    test_total_tokens = 0
    preds = []  # aligned with dataset order (no shuffle)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            caption_ids = batch["caption_ids"].to(device, non_blocking=True)
            contexts = {k: v.to(device, non_blocking=True) for k, v in batch["contexts"].items()}

            # ----- Loss (sum over clusters, normalize by non-PAD tokens) -----
            targets = caption_ids[:, 1:].contiguous().view(-1)
            num_tokens = (targets != pad_idx).sum().item()
            if num_tokens > 0:
                logits, _ = model(caption_ids[:, :-1], contexts)  # [B, T-1, D]
                logits = torch.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0)
                logits = logits - logits.max(dim=-1, keepdim=True)[0]
                flat_logits = logits.reshape(-1, logits.size(-1))
                outputs, new_targets = criterion(flat_logits, targets)

                batch_sum = 0.0
                for out, tgt in zip(outputs, new_targets):
                    if out is not None and tgt is not None and tgt.numel() > 0:
                        batch_sum += float(ce_loss(out, tgt))
                test_sum_nll += batch_sum
                test_total_tokens += num_tokens

            # ----- Generation (batched) -----
            B = caption_ids.size(0)
            start_ids = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
            token_seqs, _ = model.generate(
                contexts=contexts,
                start_ids=start_ids,
                max_len=max_len,
                eos_id=eos_id,
                pad_id=pad_id,
                criterion=criterion,
                decoding=decoding,
                beam_size=beam_size,
                length_penalty=length_penalty,
                temperature=temperature,
                return_logprobs=False,
            )
            # remove BOS for decoding
            token_seqs = token_seqs[:, 1:].detach().cpu().tolist()
            preds.extend(_decode_ids(seq) for seq in token_seqs)

    test_avg_loss = (test_sum_nll / test_total_tokens) if test_total_tokens > 0 else 0.0
    print(f"Test NLL/token: {test_avg_loss:.4f} | Tokens: {test_total_tokens}")

    # --- Read input JSON and attach 'predict' preserving structure ---
    src_path = os.path.join(config["data_dir"], "test.json")
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Input file not found: {src_path}")

    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        assert len(data) == len(preds), f"Size mismatch: {len(data)} vs {len(preds)}"
        for i in range(len(data)):
            data[i]["predict"] = preds[i]
    elif isinstance(data, dict):
        def _key(k):
            try:
                return int(k)
            except Exception:
                return k
        keys_sorted = sorted(data.keys(), key=_key)
        assert len(keys_sorted) == len(preds), f"Size mismatch: {len(keys_sorted)} vs {len(preds)}"
        for i, k in enumerate(keys_sorted):
            data[k]["predict"] = preds[i]
    else:
        raise ValueError(f"Unsupported input JSON type: {type(data)}")

    # --- Save with same structure (JSON) ---
    os.makedirs(config["output_dir"], exist_ok=True)
    out_path = os.path.join(config["output_dir"], "predictions.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[OK] Predictions written to: {out_path}")

    return test_avg_loss, data

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
        "data_dir": "/datastore/npl/ICEK/Wikipedia/content/ver4",
        "cache_dir":  "/datastore/npl/ICEK/TnT/dataset",
        "output_dir": "/datastore/npl/ICEK/TnT/output",
        "vncorenlp_path": "/datastore/npl/ICEK/VnCoreNLP",
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
            "max_target_positions": 256,
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
    print("[DEBUG] ARGS PARSE DONE.")
    from gpu_select import get_visible_gpu, set_pytorch_device
    choosen_gpu = get_visible_gpu(vram_used=30000, total_vram=80000)
    device = set_pytorch_device(choosen_gpu)
    print(f"[DEBUG] GPU CHOOSEN {choosen_gpu}")
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
    
    # # Optionally, print some predictions
    # for pred in predictions[:5]:
    #     print(f"Image: {pred['image_path']}")
    #     print(f"True Caption: {pred['true_caption']}")
    #     print(f"Predicted Caption: {pred['predicted_caption']}")
    #     print()