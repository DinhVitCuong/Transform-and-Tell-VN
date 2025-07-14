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
from models.encoder import setup_models, extract_entities, detect_faces, detect_objects, image_feature
from tell.modules.token_embedders import AdaptiveEmbedding
from tell.modules import AdaptiveSoftmax
Image.MAX_IMAGE_PIXELS = None
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
     
# def pad_and_collate(batch):
#     from torch.nn.utils.rnn import pad_sequence

#     def pad_context_list(tensors, padding_value=0.0):
#         return pad_sequence(tensors, batch_first=True, padding_value=padding_value)

#     images = [item["image"] for item in batch]
#     captions = [item["caption"] for item in batch]
#     caption_ids = [item["caption_ids"] for item in batch]

#     contexts = {
#         "image": pad_context_list([item["contexts"]["image"] for item in batch]),
#         "image_mask": pad_context_list([item["contexts"]["image_mask"] for item in batch], padding_value=True),

#         "faces": pad_context_list([item["contexts"]["faces"] for item in batch]),
#         "faces_mask": pad_context_list([item["contexts"]["faces_mask"] for item in batch], padding_value=True),

#         "obj": pad_context_list([item["contexts"]["obj"] for item in batch]),
#         "obj_mask": pad_context_list([item["contexts"]["obj_mask"] for item in batch], padding_value=True),

#         "article": pad_context_list([item["contexts"]["article"] for item in batch]),
#         "article_mask": pad_context_list([item["contexts"]["article_mask"] for item in batch], padding_value=True),
#     }

#     caption_ids = pad_sequence(caption_ids, batch_first=True, padding_value=0)

#     return {
#         "image": torch.stack(images),
#         "caption": captions,
#         "caption_ids": caption_ids,
#         "contexts": contexts,
#     }

def pad_and_collate(batch):
    from torch.nn.utils.rnn import pad_sequence

    def pad_context_list(tensors, padding_value=0.0):
        max_len = max(t.size(0) for t in tensors)
        padded = torch.stack([torch.nn.functional.pad(t, (0, 0, 0, max_len - t.size(0)), value=padding_value) for t in tensors])
        return padded

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

# class NewsCaptionDataset(Dataset):
#     def __init__(self, data_dir, split, models, max_length=512):
#         """
#         Dataset for news image captioning
#         Args:
#             data_dir: Directory containing JSON files
#             split: 'train', 'val', or 'test'
#             models: Preloaded models dictionary from setup_models()
#             max_length: Maximum caption length
#         """
#         self.models = models
#         self.split = split
#         self.max_length = max_length
#         self.preprocess = transforms.Compose([
#             transforms.Resize((256, 256)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
#         ])
        
#         # Load data
#         self.data_pt_dir = "/data/npl/ICEK/TnT/dataset/content"
#         with open(os.path.join(data_dir, f"{split}.json"), "r") as f:
#             self.data = json.load(f)
#         # define a stable ordering of keys
#         self.keys = sorted(self.data.keys(), key=lambda x: int(x))
#         # Tokenizer for captions
#         self.tokenizer = models["tokenizer"]
        
#     def __len__(self):
#         return len(self.keys)
    
#     # def __getitem__(self, idx):
#     #     # Map the integer idx → the JSON key string
#     #     key = self.keys[idx]
#     #     item = self.data[key]
#     #     img_path = item["image_path"]
#     #     device = self.models["device"]
#     #     # Load and preprocess image
#     #     img = Image.open(img_path).convert("RGB")
#     #     img_tensor = self.preprocess(img)
        
#     #     # Extract features
#     #     contexts = {}
#     #     with torch.no_grad():

#     #         # Image features (49x2048)
#     #         img_feat = image_feature(img, self.models["resnet"], 
#     #                                 self.models["preprocess"], 
#     #                                 device)
#     #         img_tensor = torch.tensor(img_feat, device=device, dtype=torch.float)
#     #         contexts["image"] = img_tensor
#     #         contexts["image_mask"] = torch.zeros(
#     #             img_tensor.size(0), dtype=torch.bool, device=device
#     #         )
            
#     #         # print(f"[DEBUG] image tensor: {contexts['image'].shape}")
#     #         # print(f"[DEBUG] image mask tensor: {contexts['image_mask'].shape}")
            
#     #         # Face features (up to 4 faces)
#     #         face_info = detect_faces(img, self.models["mtcnn"], 
#     #                                 self.models["facenet"], 
#     #                                 device)
#     #         if face_info["n_faces"] > 0:
#     #             face_embeds = torch.tensor(face_info["embeddings"],
#     #                                     device=device, dtype=torch.float)
#     #         else:
#     #             face_embeds = torch.zeros((0, 512), device=device, dtype=torch.float)

#     #         faces_mask = torch.isnan(face_embeds).any(dim=-1)          
#     #         face_embeds[faces_mask] = 0.0                              
#     #         contexts["faces"] = face_embeds                          
#     #         contexts["faces_mask"] = faces_mask  
#     #         # print(f"[DEBUG] faces tensor: {contexts['faces'].shape}")
#     #         # print(f"[DEBUG] faces mask tensor: {contexts['faces_mask'].shape}")
            
#     #         # Object features (up to 64 objects)
#     #         obj_feats = detect_objects(img_path, self.models["yolo"],
#     #                                   self.models["resnet_object"],
#     #                                   self.models["preprocess"],
#     #                                   device)
#     #         if len(obj_feats) > 0:
#     #             obj_tensor = torch.tensor(obj_feats, device=device, dtype=torch.float)
#     #             obj_mask = torch.isnan(obj_tensor).any(dim=-1)
#     #             obj_tensor[obj_mask] = 0
#     #             contexts["obj"] = obj_tensor
#     #             contexts["obj_mask"] = obj_mask
#     #         else:
#     #             contexts["obj"] = torch.zeros((1, 2048), device=device, dtype=torch.float)
#     #             contexts["obj_mask"] = torch.ones((1,), dtype=torch.bool, device=device)
#     #         # Article features
#     #         # print(f"[DEBUG] obj tensor: {contexts['obj'].shape}")
#     #         # print(f"[DEBUG] obj mask tensor: {contexts['obj_mask'].shape}")
#     #         context_txt = " ".join(item.get("context", []))
#     #         art_embed = self.models["embedder"](context_txt)
#     #         # art_embed = roberta_embed(context_txt, 
#     #         #                           self.models["tokenizer"],
#     #         #                           self.models["roberta"],
#     #         #                           self.models["vncore"],
#     #         #                           self.models["device"])
#     #         # art_embed = art_embed.to(device, dtype=torch.float)
#     #         article_len = min(512, art_embed.size(0))
#     #         padded_article = torch.zeros((512, 1024), device=device)
#     #         padded_article[:article_len] = art_embed[:article_len]
#     #         contexts["article"] = padded_article
#     #         contexts["article_mask"] = torch.zeros(512, device=device, dtype=torch.bool)
#     #         contexts["article_mask"][article_len:] = True
#     #         # print(f"[DEBUG] article tensor: {contexts['article'].shape}")
#     #         # print(f"[DEBUG] article mask tensor: {contexts['article_mask'].shape}")
        
#     #     # Process caption
#     #     caption = item.get("caption", "")
#     #     caption_ids = self.tokenizer.encode(caption, 
#     #                                       return_tensors="pt",
#     #                                       truncation=True,
#     #                                       max_length=self.max_length)[0]
        
#     #     return {
#     #         "image": img_tensor,
#     #         "contexts": contexts,
#     #         "caption_ids": caption_ids,
#     #         "caption": caption
#     #     }
#     def __getitem__(self, idx):
#         key = self.keys[idx]
#         item = self.data[key]
#         img_path = item["image_path"]
#         device = self.models["device"]

#         # Load precomputed features
#         feature_path = os.path.join(self.data_pt_dir, f"{self.split}_features", f"{key}.pt")
#         features = torch.load(feature_path, map_location=device)

#         img_tensor = features["image_feature"]
#         contexts = {
#             "image": img_tensor,
#             "image_mask": torch.zeros(img_tensor.size(0), dtype=torch.bool, device=device),
#             "faces": torch.tensor(features["face_info"]["embeddings"], device=device, dtype=torch.float) if features["face_info"]["n_faces"] > 0 else torch.zeros((0, 512), device=device, dtype=torch.float),
#             "faces_mask": torch.isnan(torch.tensor(features["face_info"]["embeddings"], device=device)).any(dim=-1) if features["face_info"]["n_faces"] > 0 else torch.ones((0,), dtype=torch.bool, device=device),
#             "obj": features["object_features"],
#             "obj_mask": torch.isnan(features["object_features"]).any(dim=-1) if features["object_features"].size(0) > 0 else torch.ones((1,), dtype=torch.bool, device=device),
#             "article": features["article_embed"],
#             "article_mask": torch.zeros(512, device=device, dtype=torch.bool),
#         }
#         contexts["article_mask"][min(512, contexts["article"].size(0)):] = True

#         caption = item.get("caption", "")
#         caption_ids = self.tokenizer.encode(
#             caption,
#             return_tensors="pt",
#             truncation=True,
#             max_length=self.max_length
#         )[0]

#         return {
#         "image": img_tensor,
#         "contexts": contexts,
#         "caption_ids": caption_ids,
#         "caption": caption,
#         "image_path": img_path,
#     }

class NewsCaptionDataset(Dataset):
    def __init__(self, data_dir, split, models, max_length=512):
        self.models = models
        self.split = split
        self.max_length = max_length
        self.preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.data_pt_dir = "/data/npl/ICEK/TnT/dataset/content"
        with open(os.path.join(data_dir, f"{split}.json"), "r") as f:
            self.data = json.load(f)
        self.keys = sorted([d["_id"] for d in self.data], key=lambda x: int(x))
        self.tokenizer = models["tokenizer"]
        self.hdf5_path = os.path.join(self.data_pt_dir, f"{split}_features.h5")
        self.hdf5_file = h5py.File(self.hdf5_path, "r", swmr=True)  # Single-Writer Multiple-Reader mode

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        item = next(d for d in self.data if d["_id"] == key)
        img_path = item["image_path"]
        device = self.models["device"]

        try:
            group = self.hdf5_file[key]
            img_tensor = torch.tensor(group["image_feature"][:], device=device, dtype=torch.float)
            face_info = json.loads(group["face_info"][()])
            obj_tensor = torch.tensor(group["object_features"][:], device=device, dtype=torch.float)
            art_embed = torch.tensor(group["article_embed"][:], device=device, dtype=torch.float)
            cap_ner = json.loads(group["caption_ner"][()])
            ctx_ner = json.loads(group["context_ner"][()])

            contexts = {
                "image": img_tensor,
                "image_mask": torch.zeros(img_tensor.size(0), dtype=torch.bool, device=device),
                "faces": torch.tensor(face_info["embeddings"], device=device, dtype=torch.float) if face_info["n_faces"] > 0 else torch.zeros((0, 512), device=device, dtype=torch.float),
                "faces_mask": torch.ones((face_info["n_faces"],), dtype=torch.bool, device=device) if face_info["n_faces"] == 0 else torch.isnan(torch.tensor(face_info["embeddings"], device=device)).any(dim=-1),
                "obj": obj_tensor,
                "obj_mask": torch.ones((obj_tensor.size(0),), dtype=torch.bool, device=device) if obj_tensor.size(0) == 0 else torch.isnan(obj_tensor).any(dim=-1),
                "article": art_embed,
                "article_mask": torch.zeros(512, device=device, dtype=torch.bool),
            }
            contexts["article_mask"][min(512, contexts["article"].size(0)):] = True
        except Exception as e:
            logging.error(f"Error loading features for {key}: {e}")
            # Return dummy data to skip problematic sample
            contexts = {
                "image": torch.zeros((49, 2048), device=device, dtype=torch.float),
                "image_mask": torch.ones((49,), dtype=torch.bool, device=device),
                "faces": torch.zeros((0, 512), device=device, dtype=torch.float),
                "faces_mask": torch.ones((0,), dtype=torch.bool, device=device),
                "obj": torch.zeros((1, 2048), device=device, dtype=torch.float),
                "obj_mask": torch.ones((1,), dtype=torch.bool, device=device),
                "article": torch.zeros((512, 1024), device=device, dtype=torch.float),
                "article_mask": torch.ones((512,), dtype=torch.bool, device=device),
            }
            cap_ner = {}
            ctx_ner = {}

        caption = item.get("caption", "")
        caption_ids = self.tokenizer.encode(
            caption,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        )[0]

        return {
            "image": contexts["image"],
            "contexts": contexts,
            "caption_ids": caption_ids,
            "caption": caption,
            "image_path": img_path,
        }

    def __del__(self):
        self.hdf5_file.close()

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
    
    def generate(self, contexts, max_length=100, temperature=1.0):
        """Generate caption from contexts"""
        # The quirks of dynamic convolution implementation: The context
        # embedding has dimension [seq_len, batch_size], but the mask has
        # dimension [batch_size, seq_len].
        # print(f"""CONTEXTS IMAGE_FEAT SHAPE: {contexts["image"].shape}""")
        # print(f"""CONTEXTS ARTICLE SHAPE: {contexts["article"].shape}""")
        # print(f"""CONTEXTS OBJECT SHAPE: {contexts["obj"].shape}""")
        # print(f"""CONTEXTS FACES SHAPE: {contexts["faces"].shape}""")
        contexts["image"]=contexts["image"].transpose(0, 1)
        contexts["article"]=contexts["article"].transpose(0, 1)
        contexts["obj"]=contexts["obj"].transpose(0, 1)
        contexts["faces"]=contexts["faces"].transpose(0, 1)
        transposed_contexts = {}
        for key in ["image", "article", "faces", "obj"]:
            transposed_contexts[key] = contexts[key].transpose(0, 1)
            transposed_contexts[f"{key}_mask"] = contexts[f"{key}_mask"]
        self.eval()
        generated = []
        incremental_state = {}
        
        # Start with <s> token
        current_token = torch.tensor([[self.decoder.padding_idx + 1]]).to(next(self.parameters()).device)
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get next token probabilities
                logits, _ = self(current_token, contexts, incremental_state)
                logits = logits[:, -1, :] / temperature
                
                # Sample next token
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Stop if </s> token is generated
                if next_token.item() == self.decoder.padding_idx:
                    break
                    
                generated.append(next_token.item())
                current_token = next_token
                
        return generated


def train_model(config):
    """Training pipeline with early stopping"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = setup_models(device, config["vncorenlp_path"])
    
    # Initialize datasets and dataloaders
    train_dataset = NewsCaptionDataset(config["data_dir"], "train", models)
    val_dataset = NewsCaptionDataset(config["data_dir"], "val", models)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        collate_fn=pad_and_collate  
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=pad_and_collate
    )
    print("DATALOADER LOADED!")
    
    # Initialize model components
    # embedder = nn.Embedding(
    #     num_embeddings=config["vocab_size"],
    #     embedding_dim=config["embed_dim"]
    # )
    # Initialize adaptive embedder
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
            
            # loss, nll_loss = criterion(
            #     logits.reshape(-1, logits.size(-1)),
            #     caption_ids[:, 1:].contiguous().view(-1)
            # )
            # # Normalize loss by number of tokens
            # num_tokens = (caption_ids[:, 1:] != config["decoder_params"]["padding_idx"]).sum()
            # normalized_loss = loss / num_tokens
            # # loss.backward()
            # # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            # # optimizer.step()
            
            # # train_loss += loss.item()
            # # Backward pass
            # optimizer.zero_grad()
            # normalized_loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            # optimizer.step()
            
            # total_loss += loss.item()
            # total_tokens += num_tokens.item()# Compute loss using AdaptiveSoftmax output
            output, new_target = criterion(
                logits.reshape(-1, logits.size(-1)),
                caption_ids[:, 1:].contiguous().view(-1)
            )
            
            # Initialize cross-entropy loss with ignore_index for padding
            
            # Compute total loss by summing losses for each cluster
            total_loss = 0
            for out, tgt in zip(output, new_target):
                if out is not None and tgt is not None:
                    # out: (batch_size, num_classes), tgt: (batch_size,)
                    total_loss += ce_loss(out, tgt)
            
            # Normalize loss by number of tokens
            num_tokens = (caption_ids[:, 1:] != config["decoder_params"]["padding_idx"]).sum()
            normalized_loss = total_loss / num_tokens
            
            # Backward pass
            normalized_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            
            total_loss_val += total_loss.item()
            total_tokens += num_tokens.item()
        avg_loss = total_loss_val / total_tokens
        print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Train Tokens: {total_tokens}")
        
        # avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                caption_ids = batch["caption_ids"].to(device)
                contexts = {k: v.to(device) for k, v in batch["contexts"].items()}
                
                logits, _ = model(caption_ids[:, :-1], contexts)
                # loss = criterion(
                #     logits.view(-1, config["vocab_size"]),
                #     caption_ids[:, 1:].contiguous().view(-1)
                # )
                # val_loss += loss.item()
                
                # loss, nll_loss = criterion(
                #     logits.view(-1, logits.size(-1)),
                #     caption_ids[:, 1:].contiguous().view(-1)
                # )
                loss, nll_loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    caption_ids[:, 1:].contiguous().view(-1)
                )
                # Normalize loss by number of tokens
                num_tokens = (caption_ids[:, 1:] != config["decoder_params"]["padding_idx"]).sum()
                val_loss += loss.item()
                val_total_tokens += num_tokens.item()
        
        avg_val_loss = val_loss / len(val_loader)
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
            print("  Saved new best model")
    
    return model

def evaluate_model(model, config):
    """Evaluation pipeline"""
    device = next(model.parameters()).device
    models = setup_models(device, config["vncorenlp_path"])
    
    # Load validation dataset
    test_dataset = NewsCaptionDataset(config["data_dir"], "test", models)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=pad_and_collate
    )
    # Initialize adaptive embedder
    embedder = AdaptiveEmbedding(
        vocab_size=config["embedder"]["vocab_size"],
        padding_idx=config["embedder"]["padding_idx"],
        initial_dim=config["embedder"]["initial_dim"],
        factor=config["embedder"]["factor"],
        output_dim=config["embedder"]["output_dim"],
        cutoff=config["embedder"]["cutoff"],
        scale_embeds=True
    )
    criterion = AdaptiveSoftmax(
        vocab_size=config["vocab_size"],
        input_dim=config["decoder_params"]["decoder_output_dim"],
        cutoff=config["decoder_params"]["adaptive_softmax_cutoff"],
        factor=config["decoder_params"]["adaptive_softmax_factor"],
        dropout=config["decoder_params"]["adaptive_softmax_dropout"],
        adaptive_inputs=embedder if config["decoder_params"]["tie_adaptive_weights"] else None,
        tie_proj=config["decoder_params"]["tie_adaptive_proj"]
    ).to(device)

    model.eval()
    total_loss = 0
    all_predictions = []

    # criterion = nn.CrossEntropyLoss(ignore_index=test_dataset.tokenizer.pad_token_id)
    
    # with torch.no_grad():
    #     for batch in tqdm(test_loader, desc="Evaluating"):
    #         caption_ids = batch["caption_ids"].to(device)
    #         contexts = {k: v.to(device) for k, v in batch["contexts"].items()}
            
    #         # Forward pass
    #         logits, _ = model(caption_ids[:, :-1], contexts)
            
    #         # Calculate loss
            
    #         loss, nll_loss = criterion(
    #             logits.view(-1, logits.size(-1)),
    #             caption_ids[:, 1:].contiguous().view(-1)
    #         )
    #         print(f"loss type: {type(loss)}, loss: {loss}")
    #         print(f"nll_loss type: {type(nll_loss)}, nll_loss: {nll_loss}")
    #         # Normalize loss by number of tokens
    #         num_tokens = (caption_ids[:, 1:] != config["decoder_params"]["padding_idx"]).sum()
    #         test_loss += loss.item()
    #         test_total_tokens += num_tokens.item()
            
    #         # Generate captions
    #         for i in range(len(batch["image"])):
    #             contexts_i = {k: v[i].unsqueeze(0) for k, v in contexts.items()}
    #             generated_ids = model.generate(contexts_i)
    #             generated_caption = test_dataset.tokenizer.decode(generated_ids)
    #             all_predictions.append({
    #                 "image_path": batch["image_path"][i],
    #                 "true_caption": batch["caption"][i],
    #                 "predicted_caption": generated_caption
    #             })
    
    # test_avg_loss = total_loss / len(test_loader)    
    # print(f"Test Loss: {test_avg_loss:.4f}, Test Tokens: {test_total_tokens}")
    ce_loss = nn.CrossEntropyLoss(ignore_index=config["decoder_params"]["padding_idx"])
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            caption_ids = batch["caption_ids"].to(device)
            contexts = {k: v.to(device) for k, v in batch["contexts"].items()}
            
            # Forward pass
            logits, _ = model(caption_ids[:, :-1], contexts)
            
            # Compute loss using AdaptiveSoftmax output
            output, new_target = criterion(
                logits.reshape(-1, logits.size(-1)),
                caption_ids[:, 1:].contiguous().view(-1)
            )
            
            # Compute total loss
            total_loss = 0
            for out, tgt in zip(output, new_target):
                if out is not None and tgt is not None:
                    total_loss += ce_loss(out, tgt)
            
            # Normalize loss by number of tokens
            num_tokens = (caption_ids[:, 1:] != config["decoder_params"]["padding_idx"]).sum()
            test_loss += total_loss.item()
            test_total_tokens += num_tokens.item()
            
            # Generate captions (unchanged)
            for i in range(len(batch["image"])):
                contexts_i = {k: v[i].unsqueeze(0) for k, v in contexts.items()}
                generated_ids = model.generate(contexts_i)
                generated_caption = test_dataset.tokenizer.decode(generated_ids)
                all_predictions.append({
                    "image_path": batch["image_path"][i],
                    "true_caption": batch["caption"][i],
                    "predicted_caption": generated_caption
                })

    test_avg_loss = test_loss / test_total_tokens
    print(f"Test Loss: {test_avg_loss:.4f}, Test Tokens: {test_total_tokens}")
        
    
    # Save predictions
    with open(os.path.join(config["output_dir"], "predictions.json"), "w") as f:
        json.dump(all_predictions, f, indent=2)
    
    return test_avg_loss, all_predictions

if __name__ == "__main__":
    # Configuration (parameters from paper and config.yaml)
    config = {
        "data_dir": "/data/npl/ICEK/Wikipedia/content/ver4",
        "output_dir": "/data/npl/ICEK/TnT/output",
        "vncorenlp_path": "/data/npl/ICEK/VnCoreNLP",
        "vocab_size": 64001,  # From phoBERT-base tokenizer
        "embed_dim": 1024,    # Hidden size
        "batch_size": 8,
        "num_workers": 0,
        "epochs": 400,
        "lr": 1e-4,
        "embedder":{
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
    trained_model = train_model(config)
    
    # Save model
    torch.save(trained_model.state_dict(), 
              os.path.join(config["output_dir"], "transform_and_tell_model.pth"))
    
    # Run evaluation
    test_loss, predictions = evaluate_model(trained_model, config)