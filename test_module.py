import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import os
from tqdm import tqdm
import py_vncorenlp
from transformers import AutoTokenizer, AutoModel
import re
import torch.nn.functional as F

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
#         # Clean input
#         try:
#             text = text.encode('utf-8', 'ignore').decode('utf-8')
#         except Exception as e:
#             print(f"[WARN] Encoding issue: {e}")
#             return torch.zeros(1, self.hidden_size).to(self.device)

#         text = re.sub(r'[^\x00-\x7F]+', '', text)
#         sentences = re.split(r'(?<=[\.!?])\s+', text.strip())

#         token_embeds = []
#         for sent in sentences:
#             if not sent.strip() or sent.count('·') >= 5:
#                 continue

#             seg_text = " ".join(self.segmenter.word_segment(sent.strip()))

#             batch = self.tokenizer(
#                 seg_text,
#                 return_tensors="pt",
#                 truncation=True,
#                 max_length=self.model.config.max_position_embeddings
#             ).to(self.device)

#             try:
#                 with torch.no_grad():
#                     out = self.model(**batch, output_hidden_states=True)
#             except RuntimeError as e:
#                 print(f"[WARN] RoBERTa GPU failed on sent: {sent} → {e}")
#                 try:
#                     model_cpu = self.model.to("cpu")
#                     with torch.no_grad():
#                         out = model_cpu(**batch.cpu(), output_hidden_states=True)
#                     self.model.to(self.device)
#                 except Exception as e2:
#                     print(f"[WARN] RoBERTa CPU also failed → {e2}")
#                     continue

#             hidden_states = torch.stack(out.hidden_states, dim=0)  # shape: (25, 1, seq_len, hidden_size)
#             weighted = (F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1) * hidden_states).sum(dim=0)
#             # weighted: (1, seq_len, hidden_size)
#             token_embeds.append(weighted.squeeze(0))  # shape: (seq_len, hidden_size)

#         if not token_embeds:
#             return torch.zeros(1, self.hidden_size).to(self.device)

#         return torch.cat(token_embeds, dim=0)  # shape: (total_seq_len, hidden_size)

class RobertaEmbedder(torch.nn.Module):
    def __init__(self, model, tokenizer, segmenter, device):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.segmenter = segmenter
        self.device = device
        self.num_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer
        self.hidden_size = model.config.hidden_size
        # Learnable weights α_ℓ for each layer
        self.layer_weights = torch.nn.Parameter(torch.ones(self.num_layers) / self.num_layers)

    def forward(self, text: str) -> torch.Tensor:
        # Clean input
        # try:
        #     text = text.encode('utf-8', 'ignore').decode('utf-8')
        # except Exception as e:
        #     print(f"[WARN] Encoding issue: {e}")
        #     return torch.zeros(1, self.hidden_size).to(self.device)

        # text = re.sub(r'[^\x00-\x7F]+', '', text)
        sentences = re.split(r'(?<=[\.!?])\s+', text.strip())

        token_embeds = []
        for sent in sentences:
            if not sent.strip() or sent.count('·') >= 5:
                continue
            # input_text = self.segmenter.word_segment(sent.strip())
            # seg_text = " ".join(self.segmenter.word_segment(sent.strip()))

            # batch = self.tokenizer(
            #     seg_text,
            #     return_tensors="pt",
            #     truncation=True,
            #     max_length=self.model.config.max_position_embeddings
            # ).to(self.device)
            toks = self.tokenizer(
                sent,
                return_tensors="pt",
                truncation=True,
                max_length=self.model.config.max_position_embeddings
            ).to(self.device)
            input_ids      = toks["input_ids"]
            input_ids = torch.tensor([self.tokenizer.encode(sent)]).to(device)
            with torch.no_grad():
                out = self.model(input_ids, output_hidden_states=True)

            hidden_states = torch.stack(out.hidden_states, dim=0)  # shape: (25, 1, seq_len, hidden_size)
            weighted = (F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1) * hidden_states).sum(dim=0)
            # weighted: (1, seq_len, hidden_size)
            token_embeds.append(weighted.squeeze(0))  # shape: (seq_len, hidden_size)

        if not token_embeds:
            return torch.zeros(1, self.hidden_size).to(self.device)

        return torch.cat(token_embeds, dim=0)  # shape: (total_seq_len, hidden_size)

with open("/data/npl/ICEK/Wikipedia/content/ver4/val.json", "r") as f:
    data = json.load(f)
keys = sorted(data.keys(), key=int)

# Get first 8 entries
sample_data = [data[k] for k in keys[:8]]
print("DATA LOADED")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
phoBERTlocal = "/data/npl/ICEK/TnT/phoBERT/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(phoBERTlocal, local_files_only=True)
roberta     = AutoModel    .from_pretrained(phoBERTlocal, use_safetensors=True, local_files_only=True).to(device).eval()
vncorenlp_path = "/data/npl/ICEK/VnCoreNLP"
py_vncorenlp.download_model(save_dir=vncorenlp_path)
vncore = py_vncorenlp.VnCoreNLP(annotators=["wseg", "pos", "ner", "parse"], save_dir=vncorenlp_path)
print("MODELS LOADED")
embedder = RobertaEmbedder(roberta, tokenizer, vncore, device).to(device)
print("EMBEDDER LOADED!")
# Optional: print or use the samples
for i, item in enumerate(sample_data):
    context_txt = " ".join(item.get("context", []))
    art_embed = embedder(context_txt)
    print(f"EMBEDDING {i} is shape {art_embed.shape}")
print(f"EMBEDDING {art_embed}")