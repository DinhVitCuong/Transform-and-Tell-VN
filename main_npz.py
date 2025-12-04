import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import os
import time
import gc
from tqdm import tqdm
import random
# Import necessary components from previous implementations
from models.decoder import DynamicConvFacesObjectsDecoder
from models.encoder import setup_models, segment_text, detect_faces, detect_objects, image_feature
from tell.modules.token_embedders import AdaptiveEmbedding
from typing import Dict, Optional, Tuple, Any
from tell.modules import AdaptiveSoftmax
from caption_metrics import evaluate
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
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
        self.data_dir   = data_dir
        self.split      = split
        self.models     = models
        self.tokenizer  = models.get("tokenizer", None)
        self.max_length = max_length
        self.cache_dir  = cache_dir

        data_file = os.path.join(data_dir, f"{split}.json")
        with open(data_file, "r") as f:
            self.data = json.load(f)

        # ---- Build items + IDs from JSON ----
        if isinstance(self.data, dict):
            def _key(k):
                try:
                    return int(k)
                except Exception:
                    return k

            keys_sorted = sorted(self.data.keys(), key=_key)
            self.items  = [self.data[k] for k in keys_sorted]
            # "true" IDs, e.g. 4001, 4002, ...
            self.ids    = [int(k) if str(k).isdigit() else k for k in keys_sorted]

        elif isinstance(self.data, list):
            self.items = self.data
            self.ids   = [item.get("id", i) for i, item in enumerate(self.items)]
        else:
            raise ValueError(f"Unsupported JSON structure in {data_file}: {type(self.data)}")

        # ---- Filter out samples whose .npz is missing ----
        valid_ids = []
        for fid in self.ids:
            fname = f"{fid}.npz"
            path  = os.path.join(self.cache_dir, self.split, fname)
            if os.path.exists(path):
                valid_ids.append(fid)
            else:
                logging.warning(
                    f"[WARN] Missing NPZ for split='{self.split}': {fname} "
                    f"under {self.cache_dir} – skipping this sample."
                )

        self.ids = valid_ids
        if len(self.ids) == 0:
            raise RuntimeError(
                f"No NPZ files found for split '{self.split}' in {self.cache_dir}"
            )

        # ---- Setup model handles ----
        self.device        = self.models["device"]
        self.mtcnn         = self.models["mtcnn"]
        self.facenet       = self.models["facenet"]
        self.resnet        = self.models["resnet"]
        self.resnet_object = self.models["resnet_object"]
        self.yolo          = self.models["yolo"]
        self.preprocess    = self.models["preprocess"]
        self.embedder      = self.models["embedder"]

    def __len__(self):
        return len(self.ids)

    # def _load_npz(self, path: str) -> Dict[str, np.ndarray]:
    #     with np.load(path, allow_pickle=False) as z:
    #         needed = [
    #             "image", "image_mask",
    #             "faces", "faces_mask",
    #             "obj", "obj_mask",
    #             "article", "article_mask",
    #             "caption_ids",
    #         ]
    #         for n in needed:
    #             if n not in z.files:
    #                 raise KeyError(f"{os.path.basename(path)} missing '{n}'")
    #         if z["caption_ids"].size == 0:
    #             raise ValueError(
    #                 f"{os.path.basename(path)} has empty caption_ids; "
    #                 "re-run precompute with tokenizer."
    #             )
    #         return {k: z[k] for k in z.files}
    def _load_npz(self, path: str) -> Dict[str, np.ndarray]:
        """
        Robust loader:
        - Retries a few times on 'Too many open files' (Errno 23/24)
        - Copies arrays out of the NPZ file so no file descriptor is kept alive.
        """
        needed = [
            "image", "image_mask",
            "faces", "faces_mask",
            "obj", "obj_mask",
            "article", "article_mask",
            "caption_ids",
        ]

        max_retries = 5
        for attempt in range(max_retries):
            try:
                with np.load(path, allow_pickle=False) as z:
                    # Check required keys
                    for n in needed:
                        if n not in z.files:
                            raise KeyError(f"{os.path.basename(path)} missing '{n}'")

                    if z["caption_ids"].size == 0:
                        raise ValueError(
                            f"{os.path.basename(path)} has empty caption_ids; "
                            "re-run precompute with tokenizer."
                        )

                    # IMPORTANT: np.array(..., copy=True) ensures no lazy memmap
                    out = {k: np.array(z[k], copy=True) for k in needed}
                # File is closed here because of the context manager
                return out

            except OSError as e:
                # 23 = ENFILE (Too many open files in system)
                # 24 = EMFILE (Too many open files for this process)
                if e.errno in (23, 24):
                    logging.warning(
                        f"[WARN] OSError {e.errno} when opening {path} "
                        f"(attempt {attempt+1}/{max_retries}): {e}. "
                        "Forcing GC and retrying..."
                    )
                    gc.collect()
                    time.sleep(0.5)  # small backoff
                    continue
                # Other OS errors → re-raise
                raise

        # If we get here, all attempts failed
        raise OSError(
            f"Failed to open {path} after {max_retries} retries "
            "due to 'Too many open files'."
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        file_id = self.ids[idx]                      # e.g. 4001
        fname   = f"{file_id}.npz"
        path    = os.path.join(self.cache_dir, self.split, fname)

        # If a file somehow disappears after init, warn & raise
        if not os.path.exists(path):
            logging.warning(
                f"[WARN] NPZ disappeared for split='{self.split}': {path}"
            )
            # You *could* return a dummy sample here instead of raising,
            # but raising is safer so you notice something is wrong.
            raise FileNotFoundError(path)

        arr = self._load_npz(path)

        # Cast to tensors
        image        = torch.from_numpy(arr["image"]).float()
        image_mask   = torch.from_numpy(arr["image_mask"]).to(torch.bool)

        faces        = torch.from_numpy(arr["faces"]).float()
        faces_mask   = torch.from_numpy(arr["faces_mask"]).to(torch.bool)

        obj          = torch.from_numpy(arr["obj"]).float()
        obj_mask     = torch.from_numpy(arr["obj_mask"]).to(torch.bool)

        article_np   = arr["article"].astype(np.float32, copy=False)
        article      = torch.from_numpy(article_np).float()
        article_mask = torch.from_numpy(arr["article_mask"]).to(torch.bool)

        caption_ids  = torch.from_numpy(arr["caption_ids"]).long()

        return {
            "id": file_id,   # return the real ID (4001, ...)
            "contexts": {
                "image": image, "image_mask": image_mask,
                "faces": faces, "faces_mask": faces_mask,
                "obj": obj,   "obj_mask": obj_mask,
                "article": article, "article_mask": article_mask,
            },
            "caption_ids": caption_ids,
        }


class TransformAndTell(nn.Module):
    def __init__(self, vocab_size, embedder, decoder_params,
                 sampling_temp: float = 1.0,
                 sampling_topk: int = 1,
                 padding_idx: int = 1,
                 ):
        """
        Full Transform and Tell model
        Args:
            vocab_size: Size of vocabulary
            embedder: Token embedder (AdaptiveEmbedding)
            decoder_params: Parameters for DynamicConvFacesObjectsDecoder
        """
        super().__init__()
        self.sampling_topk = sampling_topk
        self.sampling_temp = sampling_temp
        self.padding_idx = padding_idx
        self.decoder = DynamicConvFacesObjectsDecoder(
            vocab_size=vocab_size,
            embedder=embedder,
            **decoder_params
        )
        # WEIGHTED SUM FOR ARTICLE:
        # self.article_layer_alpha = nn.Parameter(torch.Tensor(25)) 
        # self.expected_article_layers = 25  # RoBERTa/PhoBERT-large
        
    def forward(self, prev_target, contexts, incremental_state=None, mode = "train"):
        # for key in ['image','article','faces','obj']:
        #     feat = contexts[key]
        #     mask = contexts[f"{key}_mask"]
        #     print(f"[DEBUG]: {key:>8} feat: {tuple(feat.shape)}, mask: {tuple(mask.shape)}")
        
        return self.decoder(prev_target, contexts, incremental_state, mode)

    def generate(self,  # type: ignore
                 context: Dict[str, torch.LongTensor],
                 image: torch.Tensor,
                 face_embeds,
                 obj_embeds,
                 metadata: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:

        B = image.shape[0]
        caption = {self.index: context[self.index].new_zeros(B, 2)}
        caption_ids, _, contexts = self._forward(
            context, image, caption, face_embeds, obj_embeds)

        _, gen_ids, attns = self._generate(caption_ids, contexts)

        gen_ids = gen_ids.cpu().numpy().tolist()
        attns_list: List[List[Dict[str, Any]]] = []

        for i, token_ids in enumerate(gen_ids):
            # Let's process the article text
            article_ids = context[self.index][i]
            article_ids = article_ids[article_ids != self.padding_idx]
            article_ids = article_ids.cpu().numpy()
            # article_ids.shape == [seq_len]

            # remove <s>
            if article_ids[0] == self.roberta.task.source_dictionary.bos():
                article_ids = article_ids[1:]

             # Ignore final </s> token
            if article_ids[-1] == self.roberta.task.source_dictionary.eos():
                article_ids = article_ids[:-1]

            # Sanity check. We plus three because we removed <s>, </s> and
            # the last two attention scores are for no attention and bias
            assert article_ids.shape[0] == attns[0][0]['article'][i][0].shape[0] - 4

            byte_ids = [int(self.roberta.task.source_dictionary[k])
                        for k in article_ids]
            # e.g. [16012, 17163, 447, 247, 82, 4640, 3437]

            byte_strs = [self.roberta.bpe.bpe.decoder.get(token, token)
                         for token in byte_ids]
            # e.g. ['Sun', 'rise', 'âĢ', 'Ļ', 's', 'Ġexecutive', 'Ġdirector']

            merged_article = []
            article_mask = []
            cursor = 0
            a: Dict[str, Any] = {}
            newline = False
            for j, b in enumerate(byte_strs):
                # Start a new word
                if j == 0 or b[0] == 'Ġ' or b[0] == 'Ċ' or newline:
                    if a:
                        byte_text = ''.join(a['tokens'])
                        a['text'] = bytearray([self.roberta.bpe.bpe.byte_decoder[c] for c in byte_text]).decode(
                            'utf-8', errors=self.roberta.bpe.bpe.errors)
                        merged_article.append(a)
                        cursor += 1
                    # Note that
                    #   len(attns) == generation_length
                    #   len(attns[j]) == n_layers
                    #   attns[j][l] is a dictionary
                    #   attns[j][l]['article'].shape == [batch_size, target_len, source_len]
                    #   target_len == 1 since we generate one word at a time
                    a = {'tokens': [b]}
                    article_mask.append(cursor)
                    newline = b[0] == 'Ċ'
                else:
                    a['tokens'].append(b)
                    article_mask.append(cursor)
            byte_text = ''.join(a['tokens'])
            a['text'] = bytearray([self.roberta.bpe.bpe.byte_decoder[c] for c in byte_text]).decode(
                'utf-8', errors=self.roberta.bpe.bpe.errors)
            merged_article.append(a)

            # Next let's process the caption text
            attn_dicts: List[Dict[str, Any]] = []
            # Ignore seed input <s>
            if token_ids[0] == self.roberta.task.source_dictionary.bos():
                token_ids = token_ids[1:]  # remove <s>
            # Now len(token_ids) should be the same of len(attns)

            assert len(attns) == len(token_ids)

            # Ignore final </s> token
            if token_ids[-1] == self.roberta.task.source_dictionary.eos():
                token_ids = token_ids[:-1]
            # Now len(token_ids) should be len(attns) - 1

            byte_ids = [int(self.roberta.task.source_dictionary[k])
                        for k in token_ids]
            # e.g. [16012, 17163, 447, 247, 82, 4640, 3437]

            byte_strs = [self.roberta.bpe.bpe.decoder.get(token, token)
                         for token in byte_ids]
            # e.g. ['Sun', 'rise', 'âĢ', 'Ļ', 's', 'Ġexecutive', 'Ġdirector']

            # Merge by space
            a: Dict[str, Any] = {}
            for j, b in enumerate(byte_strs):
                # Clean up article attention
                article_attns = copy.deepcopy(merged_article)
                start = 0
                for word in article_attns:
                    end = start + len(word['tokens'])
                    layer_attns = []
                    for layer in range(len(attns[j])):
                        layer_attns.append(
                            attns[j][layer]['article'][i][0][start:end].mean())
                    word['attns'] = layer_attns
                    start = end
                    del word['tokens']

                # Start a new word. Ġ is space
                if j == 0 or b[0] == 'Ġ':
                    if a:
                        for l in range(len(a['attns']['image'])):
                            for modal in ['image', 'faces', 'obj']:
                                a['attns'][modal][l] /= len(a['tokens'])
                                a['attns'][modal][l] = a['attns'][modal][l].tolist()
                            for word in a['attns']['article']:
                                word['attns'][l] /= len(a['tokens'])
                                word['attns'][l] = word['attns'][l].tolist()
                        byte_text = ''.join(a['tokens'])
                        a['tokens'] = bytearray([self.roberta.bpe.bpe.byte_decoder[c] for c in byte_text]).decode(
                            'utf-8', errors=self.roberta.bpe.bpe.errors)
                        attn_dicts.append(a)
                    # Note that
                    #   len(attns) == generation_length
                    #   len(attns[j]) == n_layers
                    #   attns[j][l] is a dictionary
                    #   attns[j][l]['article'].shape == [batch_size, target_len, source_len]
                    #   target_len == 1 since we generate one word at a time
                    a = {
                        'tokens': [b],
                        'attns': {
                            'article': article_attns,
                            'image': [attns[j][l]['image'][i][0] for l in range(len(attns[j]))],
                            'faces': [attns[j][l]['faces'][i][0] for l in range(len(attns[j]))],
                            'obj': [attns[j][l]['obj'][i][0] for l in range(len(attns[j]))],
                        }
                    }
                else:
                    a['tokens'].append(b)
                    for l in range(len(a['attns']['image'])):
                        for modal in ['image', 'faces', 'obj']:
                            a['attns'][modal][l] += attns[j][l][modal][i][0]
                        for w, word in enumerate(a['attns']['article']):
                            word['attns'][l] += article_attns[w]['attns'][l]

            for l in range(len(a['attns']['image'])):
                for modal in ['image', 'faces', 'obj']:
                    a['attns'][modal][l] /= len(a['tokens'])
                    a['attns'][modal][l] = a['attns'][modal][l].tolist()
                for word in a['attns']['article']:
                    word['attns'][l] /= len(a['tokens'])
                    word['attns'][l] = word['attns'][l].tolist()
            byte_text = ''.join(a['tokens'])
            a['tokens'] = bytearray([self.roberta.bpe.bpe.byte_decoder[c] for c in byte_text]).decode(
                'utf-8', errors=self.roberta.bpe.bpe.errors)
            attn_dicts.append(a)

            attns_list.append(attn_dicts)

            # gen_texts = [self.roberta.decode(
            #     x[x != self.padding_idx]) for x in gen_ids]

        return attns_list
    
    def _generate(self, caption_ids, contexts, attn_idx=None):
        incremental_state: Dict[str, Any] = {}
        seed_input = caption_ids[:, 0:1]
        log_prob_list = []
        index_path_list = [seed_input]
        eos = 2
        active_idx = seed_input[:, -1] != eos
        full_active_idx = active_idx
        gen_len = 100
        B = caption_ids.shape[0]
        attns = []
            # print(f"[DEBUG] full_active_idx {full_active_idx}")

        contexts = self.decoder.transpose(contexts)
        # for key in ['image','article','faces','obj']:
        #     feat = contexts[key]
        #     mask = contexts[f"{key}_mask"]
        #     print(f"[DEBUG]: {key:>8} feat: {tuple(feat.shape)}, mask: {tuple(mask.shape)}")
        for i in range(gen_len):
            if i == 0:
                prev_target = seed_input
            else:
                prev_target = seed_input[:, -1:]

            self.decoder.filter_incremental_state(
                incremental_state, active_idx)

            contexts_i = {
                'image': contexts['image'][:, full_active_idx],
                'image_mask': contexts['image_mask'][full_active_idx],
                'article': contexts['article'][:, full_active_idx],
                'article_mask': contexts['article_mask'][full_active_idx],
                'faces': contexts['faces'][:, full_active_idx],
                'faces_mask': contexts['faces_mask'][full_active_idx],
                'obj': contexts['obj'][:, full_active_idx],
                'obj_mask': contexts['obj_mask'][full_active_idx],
                'sections':  None,
                'sections_mask': None,
            }
            # for key in ['image','article','faces','obj']:
            #     feat = contexts_i[key]
            #     mask = contexts_i[f"{key}_mask"]
            #     print(f"[DEBUG]: {key:>8} feat: {tuple(feat.shape)}, mask: {tuple(mask.shape)}")

            decoder_out = self.decoder(
                prev_target,
                contexts_i,
                incremental_state=incremental_state,
                mode = "val")

            attns.append(decoder_out[1]['attn'])

            # We're only interested in the current final word
            decoder_out = (decoder_out[0][:, -1:], None)

            lprobs = self.decoder.get_normalized_probs(
                decoder_out, log_probs=True)
            # lprobs.shape == [batch_size, 1, vocab_size]

            lprobs = lprobs.squeeze(1)
            # lprobs.shape == [batch_size, vocab_size]

            topk_lprobs, topk_indices = lprobs.topk(self.sampling_topk)
            topk_lprobs = topk_lprobs.div_(self.sampling_temp)
            # topk_lprobs.shape == [batch_size, topk]

            # Take a random sample from those top k
            topk_probs = topk_lprobs.exp()
            sampled_index = torch.multinomial(topk_probs, num_samples=1)
            # sampled_index.shape == [batch_size, 1]

            selected_lprob = topk_lprobs.gather(
                dim=-1, index=sampled_index)
            # selected_prob.shape == [batch_size, 1]

            selected_index = topk_indices.gather(
                dim=-1, index=sampled_index)
            # selected_index.shape == [batch_size, 1]

            log_prob = selected_lprob.new_zeros(B, 1)
            log_prob[full_active_idx] = selected_lprob

            index_path = selected_index.new_full((B, 1), self.padding_idx)
            index_path[full_active_idx] = selected_index

            log_prob_list.append(log_prob)
            index_path_list.append(index_path)

            seed_input = torch.cat([seed_input, selected_index], dim=-1)

            is_eos = selected_index.squeeze(-1) == eos
            active_idx = ~is_eos

            full_active_idx[full_active_idx.nonzero()[~active_idx]] = 0

            seed_input = seed_input[active_idx]

            if active_idx.sum().item() == 0:
                break

        log_probs = torch.cat(log_prob_list, dim=-1)
        # log_probs.shape == [batch_size * beam_size, generate_len]

        token_ids = torch.cat(index_path_list, dim=-1)
        # token_ids.shape == [batch_size * beam_size, generate_len]

        return log_probs, token_ids, attns
    
def run_validation(
    model,
    val_loader,
    criterion,
    ce_loss,
    pad_idx,
    bos_id,
    eos_id,
    pad_id,
    device,
    _decode_ids,
    global_step: int,
    vncore
):
    model.eval()
    val_sum_nll = 0.0
    val_tokens = 0

    reference_captions = []
    candidate_captions = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validating @ step {global_step}", leave=False):
            caption_ids = batch["caption_ids"].to(device, non_blocking=True)
            contexts    = {k: v.to(device, non_blocking=True) for k, v in batch["contexts"].items()}

            targets = caption_ids[:, 1:].contiguous().view(-1)
            num_tokens = (targets != pad_idx).sum()
            if num_tokens.item() == 0:
                continue

            loss_contexts   = contexts.copy()
            loss_caption_ids = caption_ids[:, :-1].clone()

            logits, _ = model(loss_caption_ids, loss_contexts)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0)
            logits = logits - logits.max(dim=-1, keepdim=True)[0]
            flat_logits = logits.reshape(-1, logits.size(-1))

            outputs, new_targets = criterion(flat_logits, targets)

            batch_sum_nll = 0.0
            for out, tgt in zip(outputs, new_targets):
                if out is not None and tgt is not None and tgt.numel() > 0:
                    batch_sum_nll = batch_sum_nll + ce_loss(out, tgt)

            val_sum_nll += float(batch_sum_nll)
            val_tokens  += int(num_tokens)

            # ----- Generation for metrics -----
            B = caption_ids.size(0)
            _, token_seqs, attns = model._generate(caption_ids, contexts)

            gt_ids_batch   = caption_ids[:, 1:].detach().cpu().tolist()
            pred_ids_batch = token_seqs[:, 1:].detach().cpu().tolist()

            for gt_ids, pred_ids in zip(gt_ids_batch, pred_ids_batch):
                reference_captions.append(_decode_ids(gt_ids))
                candidate_captions.append(_decode_ids(pred_ids))

    val_nll = (val_sum_nll / val_tokens) if val_tokens > 0 else 0.0
    print(f"[VAL] step {global_step} | Val NLL/token: {val_nll:.4f} | Tokens: {val_tokens}")

    metrics = None
    if len(reference_captions) > 0:
        metrics = evaluate(reference_captions, candidate_captions, vncore)
        bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rougeL, cider = metrics
        print(
            f"[VAL] step {global_step} | "
            f"BLEU-1: {bleu1:.4f} | BLEU-2: {bleu2:.4f} | "
            f"BLEU-3: {bleu3:.4f} | BLEU-4: {bleu4:.4f} | "
            f"METEOR: {meteor:.4f} | "
            f"ROUGE-1: {rouge1:.4f} | ROUGE-L: {rougeL:.4f} | "
            f"CIDEr: {cider:.4f}"
        )
    else:
        print("[VAL] No captions collected for metrics at this step.")

    return val_nll, metrics

def train_model(config):

    print("[DEBUG] STARTING PREP")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = setup_models(device, config["vncorenlp_path"])
    vncore = models["vncore"]
    # train_loader = DataLoader(
    #     NewsCaptionDataset(config["data_dir"], config["cache_dir"], "demo20", models),
    #     shuffle=True,
    #     num_workers=config["num_workers"],
    #     collate_fn=pad_and_collate,
    #     pin_memory=True if torch.cuda.is_available() else False,
    # )

    # val_loader = DataLoader(
    #     NewsCaptionDataset(config["data_dir"], config["cache_dir"], "demo20", models),
    #     batch_size=config["batch_size"],
    #     shuffle=False,
    #     num_workers=config["num_workers"],
    #     collate_fn=pad_and_collate,
    #     pin_memory=True if torch.cuda.is_available() else False,
    # )
    train_loader = DataLoader(
        NewsCaptionDataset(config["data_dir"], config["cache_dir"], "train", models),
        shuffle=True,
        num_workers=config["num_workers"],
        batch_size=config["batch_size"], 
        collate_fn=pad_and_collate,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        NewsCaptionDataset(config["data_dir"], config["cache_dir"], "val", models),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        batch_size=config["batch_size"], 
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

    # ==========================
    # OPTIMIZER & SCHEDULE (BERT-STYLE)
    # ==========================
    opt_cfg = config.get("optimizer", {})

    base_lr       = opt_cfg.get("lr", 1e-4)
    b1            = opt_cfg.get("b1", 0.9)
    b2            = opt_cfg.get("b2", 0.98)
    eps           = opt_cfg.get("e", 1e-6)
    weight_decay  = opt_cfg.get("weight_decay", 0.00001)
    warmup_frac   = opt_cfg.get("warmup", 0.01)
    t_total       = int(opt_cfg.get("t_total", 0))  # total training steps
    schedule_type = opt_cfg.get("schedule", "warmup_linear")
    max_grad_norm = opt_cfg.get("max_grad_norm", 0.1)

    # AdamW is the usual "BERT Adam" variant
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        betas=(b1, b2),
        eps=eps,
        weight_decay=weight_decay,
    )

    # If t_total not set, fall back to epochs * steps per epoch
    if t_total <= 0:
        # NOTE: this assumes full epoch length is used
        steps_per_epoch = max(1, len(train_loader))
        t_total = steps_per_epoch * config["epochs"] 

    warmup_steps = int(warmup_frac * t_total)

    def lr_scale_fn(step: int) -> float:
        """
        Warmup + linear decay, like 'warmup_linear' in BERT.
        step is 1-based global step.
        """
        if step <= 0:
            return 0.0
        if step < warmup_steps:
            return float(step) / max(1.0, warmup_steps)
        # linear decay after warmup
        if step >= t_total:
            return 0.0
        return max(
            0.0,
            float(t_total - step) / max(1.0, t_total - warmup_steps)
        )

    print("[DEBUG] OPTIMIZER + WARMUP_LINEAR SCHED LOADED!")  

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

    # --- IDs & decoding params ---
    bos_id = int(config["decoder_params"].get("bos_idx", 0))
    eos_id = int(config["decoder_params"].get("eos_idx", 2))
    pad_idx = int(config["decoder_params"]["padding_idx"])
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
        clean = []
        for t in ids:
            if t == pad_id:
                continue
            if t == eos_id:
                break
            clean.append(int(t))

        if tok is not None:
            if hasattr(tok, "DecodeIds"):  # SentencePiece style
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
                s.replace(" @@ ", "")
                 .replace(" ,", ",").replace(" .", ".")
                 .replace(" !", "!").replace(" ?", "?")
                 .strip()
            )

        return " ".join(map(str, clean))

    print(f"[DEBUG] num of epoch: {config['epochs']}")

    # global step for warmup_linear
    global_step = 0  
    validate_every = int(config.get("validate_every_steps", 40000))  # default 30k
    print(f"[DEBUG] Validate every {validate_every} steps")

    for epoch in range(config["epochs"]):
        model.train()
        total_token_nll = 0.0
        total_tokens = 0
        total_grad_norm = 0.0
        num_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
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

            # ---- Grad clipping with max_grad_norm from config ----
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm
            )  

            # ---- Update LR with warmup_linear schedule ----
            global_step += 1                                       
            if schedule_type == "warmup_linear" and t_total > 0:   
                scale = lr_scale_fn(global_step)                   
                for group in optimizer.param_groups:               
                    group["lr"] = base_lr * scale                  

            optimizer.step()

            total_grad_norm += float(grad_norm)
            num_batches += 1
            total_token_nll += float(sum_loss.detach())
            total_tokens += int(num_tokens.detach())
            # ==========================
            # STEP-BASED VALIDATION
            # ==========================
            if global_step % validate_every == 0 or global_step == t_total:
                print(f"[DEBUG] Running validation at step {global_step}")
                val_nll, metrics = run_validation(
                    model=model,
                    val_loader=val_loader,
                    criterion=criterion,
                    ce_loss=ce_loss,
                    pad_idx=pad_idx,
                    bos_id=bos_id,
                    eos_id=eos_id,
                    pad_id=pad_id,
                    device=device,
                    _decode_ids=_decode_ids,
                    global_step=global_step,
                    vncore=vncore
                )

                # Early stopping now works on "validation calls" instead of epochs
                if early_stopper.early_stop(val_nll):
                    print(f"Early stopping triggered at step {global_step}!")
                    # Optional: save last model before breaking
                    os.makedirs(config["output_dir"], exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(config["output_dir"], "best_model_last.pth"))
                    return model, models

                if val_nll < best_val_loss:
                    best_val_loss = val_nll
                    os.makedirs(config["output_dir"], exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(config["output_dir"], "best_model.pth"))
                    print(f"Saved new best model at step {global_step}")
        
        train_avg_nll = (total_token_nll / total_tokens) if total_tokens > 0 else 0.0
        avg_grad_norm = (total_grad_norm / num_batches) if num_batches > 0 else 0.0
        print(f"Epoch {epoch+1} | Train NLL/token: {train_avg_nll:.4f} | Tokens: {total_tokens} | AvgGradNorm: {avg_grad_norm:.6f}")

        # ======================
        # VALIDATION
        # ======================
        model.eval()
        val_sum_nll = 0.0
        val_tokens = 0

        reference_captions = []
        candidate_captions = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                caption_ids = batch["caption_ids"].to(device, non_blocking=True)
                contexts    = {k: v.to(device, non_blocking=True) for k, v in batch["contexts"].items()}

                targets = caption_ids[:, 1:].contiguous().view(-1)
                num_tokens = (targets != pad_idx).sum()
                if num_tokens.item() == 0:
                    continue

                loss_contexts = contexts.copy()
                loss_caption_ids = caption_ids[:, :-1].clone()

                logits, _ = model(loss_caption_ids, loss_contexts)
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

                # ----- Generation for metrics -----
                B = caption_ids.size(0)
                start_ids = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
                _, token_seqs, attns = model._generate(caption_ids, contexts)

                gt_ids_batch   = caption_ids[:, 1:].detach().cpu().tolist()
                pred_ids_batch = token_seqs[:, 1:].detach().cpu().tolist()

                for gt_ids, pred_ids in zip(gt_ids_batch, pred_ids_batch):
                    reference_captions.append(_decode_ids(gt_ids))
                    candidate_captions.append(_decode_ids(pred_ids))

        val_nll = (val_sum_nll / val_tokens) if val_tokens > 0 else 0.0
        print(f"Epoch {epoch+1} | Val NLL/token: {val_nll:.4f} | Tokens: {val_tokens}")

        # ===== Text metrics =====
        if len(reference_captions) > 0:
            metrics = evaluate(reference_captions, candidate_captions, vncore)
            bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rougeL, cider = metrics
            print(
                f"Epoch {epoch+1} | "
                f"BLEU-1: {bleu1:.4f} | BLEU-2: {bleu2:.4f} | "
                f"BLEU-3: {bleu3:.4f} | BLEU-4: {bleu4:.4f} | "
                f"METEOR: {meteor:.4f} | "
                f"ROUGE-1: {rouge1:.4f} | ROUGE-L: {rougeL:.4f} | "
                f"CIDEr: {cider:.4f}"
            )
        else:
            print("No captions collected for metrics this epoch.")

        # NOTE: we removed ReduceLROnPlateau scheduler.step(val_nll)
        # because LR is now controlled per-step by warmup_linear.  

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
    vncore = models["vncore"]
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
    pad_idx = int(config["decoder_params"].get("padding_idx", 1))
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
    # For metrics
    reference_captions = []
    candidate_captions = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            caption_ids = batch["caption_ids"].to(device, non_blocking=True)
            contexts    = {k: v.to(device, non_blocking=True) for k, v in batch["contexts"].items()}

            targets = caption_ids[:, 1:].contiguous().view(-1)
            num_tokens = (targets != pad_idx).sum()
            if num_tokens.item() == 0:
                continue
            loss_contexts = contexts.copy()
            loss_caption_ids = caption_ids[:, :-1].clone()
            # ----- Loss (same as before) -----
            logits, _ = model(loss_caption_ids, loss_contexts)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0)
            logits = logits - logits.max(dim=-1, keepdim=True)[0]
            flat_logits = logits.reshape(-1, logits.size(-1))

            outputs, new_targets = criterion(flat_logits, targets)

            batch_sum_nll = 0.0
            for out, tgt in zip(outputs, new_targets):
                if out is not None and tgt is not None and tgt.numel() > 0:
                    batch_sum_nll = batch_sum_nll + ce_loss(out, tgt)

            test_sum_nll += float(batch_sum_nll)
            test_total_tokens += int(num_tokens)
            print("[DEBUG] DONE LOSS")
            # ----- Generation for metrics -----
            B = caption_ids.size(0)
            start_ids = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
            _, token_seqs, attns = model._generate(caption_ids, contexts)

            # remove BOS token before decoding
            gt_ids_batch   = caption_ids[:, 1:].detach().cpu().tolist()
            pred_ids_batch = token_seqs[:, 1:].detach().cpu().tolist()

            for gt_ids, pred_ids in zip(gt_ids_batch, pred_ids_batch):
                reference_captions.append(_decode_ids(gt_ids))
                candidate_captions.append(_decode_ids(pred_ids))

    val_nll = (test_sum_nll / test_total_tokens) if test_total_tokens > 0 else 0.0
    print(f"Val NLL/token: {val_nll:.4f} | Tokens: {test_total_tokens}")

    # ===== Text metrics =====
    if len(reference_captions) > 0:
        metrics = evaluate(reference_captions, candidate_captions, vncore)
        bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rougeL, cider = metrics
        print(
            f"BLEU-1: {bleu1:.4f} | BLEU-2: {bleu2:.4f} | "
            f"BLEU-3: {bleu3:.4f} | BLEU-4: {bleu4:.4f} | "
            f"METEOR: {meteor:.4f} | "
            f"ROUGE-1: {rouge1:.4f} | ROUGE-L: {rougeL:.4f} | "
            f"CIDEr: {cider:.4f}"
        )
    else:
        print("No captions collected for metrics this epoch.")

    # --- Read input JSON and attach 'predict' preserving structure ---
    preds = candidate_captions

    src_path = os.path.join(config["data_dir"], "test.json")
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Input file not found: {src_path}")

    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # We must match the sorting logic of NewsCaptionDataset exactly
    # to ensure predictions align with the correct input IDs.
    if isinstance(data, list):
        # If dataset filtered out missing NPZ files, lengths won't match.
        # We assume here strict 1:1 mapping as per your assert.
        if len(data) != len(preds):
            print(f"[WARN] Input size ({len(data)}) != Preds size ({len(preds)}). "
                  "Mapping might be misaligned if some files were skipped by the Dataset loader.")
        
        # Fill as much as we have
        limit = min(len(data), len(preds))
        for i in range(limit):
            data[i]["predict"] = preds[i]

    elif isinstance(data, dict):
        def _key(k):
            try:
                return int(k)
            except Exception:
                return k

        # Sort keys exactly how the Dataset loaded them
        keys_sorted = sorted(data.keys(), key=_key)
        
        if len(keys_sorted) != len(preds):
             print(f"[WARN] Input size ({len(keys_sorted)}) != Preds size ({len(preds)}). "
                   "Mapping might be misaligned if some files were skipped by the Dataset loader.")

        limit = min(len(keys_sorted), len(preds))
        for i in range(limit):
            k = keys_sorted[i]
            data[k]["predict"] = preds[i]

    else:
        raise ValueError(f"Unsupported input JSON type: {type(data)}")

    # --- Save with same structure (JSON) ---
    os.makedirs(config["output_dir"], exist_ok=True)
    out_path = os.path.join(config["output_dir"], "predictions.json")
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"[OK] Predictions written to: {out_path}")

    # Return the correct calculated loss variable (val_nll)
    return val_nll, data

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
        "seed": 42,
        "data_dir": "/datastore/npl/ICEK/Wikipedia/content/ver5",
        "cache_dir":  "/datastore/npl/ICEK/TnT/new_dataset",
        "output_dir": "/datastore/npl/ICEK/TnT/output/v3",
        "vncorenlp_path": "/datastore/npl/ICEK/VnCoreNLP",
        "vocab_size": 64001,  # From phoBERT-base tokenizer
        "embed_dim": 1024,    # Hidden size
        "batch_size": 32,
        "num_workers": 0,
        "epochs": 100,
        "decoding": "beam", 
        "lr": 0.0001,  
        "embedder": {
            "vocab_size": 64001,
            "initial_dim": 1024,
            "output_dim": 1024,
            "factor": 1,
            "cutoff": [5000, 20000],
            "padding_idx": 1,
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
            "padding_idx": 1,
            "swap": False
        },
        "early_stopping_patience": 10, 
        "early_stopping_min_delta": 0.001,
    }
    print("[DEBUG] ARGS PARSE DONE.")
    # from gpu_select import get_visible_gpu, set_pytorch_device
    # choosen_gpu = get_visible_gpu(vram_used=30000, total_vram=80000)
    # device = set_pytorch_device(choosen_gpu)
    # print(f"[DEBUG] GPU CHOOSEN {choosen_gpu}")
    # Run training
    seed_everything(config["seed"])
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
