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
from models.encoder import setup_models, extract_entities, detect_faces, detect_objects, image_feature, roberta_embed
from tell.modules.token_embedders import AdaptiveEmbedding
from tell.modules import AdaptiveSoftmax
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

class NewsCaptionDataset(Dataset):
    def __init__(self, data_dir, split, models, max_length=512):
        """
        Dataset for news image captioning
        Args:
            data_dir: Directory containing JSON files
            split: 'train', 'val', or 'test'
            models: Preloaded models dictionary from setup_models()
            max_length: Maximum caption length
        """
        self.models = models
        self.split = split
        self.max_length = max_length
        self.preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Load data
        with open(os.path.join(data_dir, f"{split}.json"), "r") as f:
            self.data = json.load(f)
        # define a stable ordering of keys
        self.keys = sorted(self.data.keys(), key=lambda x: int(x))
        # Tokenizer for captions
        self.tokenizer = models["tokenizer"]
        
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        # Map the integer idx → the JSON key string
        key = self.keys[idx]
        item = self.data[key]
        img_path = item["image_path"]
        
        # Load and preprocess image
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.preprocess(img)
        
        # Extract features
        contexts = {}
        with torch.no_grad():
            # Image features (49x2048)
            img_feat = image_feature(img, self.models["resnet"], 
                                    self.models["preprocess"], 
                                    self.models["device"])
            contexts["image"] = torch.tensor(img_feat).to(self.models["device"])
            contexts["image_mask"] = None
            
            # Face features (up to 4 faces)
            face_info = detect_faces(img, self.models["mtcnn"], 
                                    self.models["facenet"], 
                                    self.models["device"])
            contexts["faces"] = torch.tensor(face_info["embeddings"]).to(self.models["device"]) \
                                if face_info["n_faces"] > 0 else torch.zeros((0, 512))
            contexts["faces_mask"] = None
            
            # Object features (up to 64 objects)
            obj_feats = detect_objects(img_path, self.models["yolo"],
                                      self.models["resnet_object"],
                                      self.models["preprocess"],
                                      self.models["device"])
            contexts["obj"] = torch.tensor([obj["feature"] for obj in obj_feats]).to(self.models["device"])
            contexts["obj_mask"] = None
            
            # Article features
            context_txt = " ".join(item.get("context", []))
            art_embed = roberta_embed(context_txt, 
                                      self.models["tokenizer"],
                                      self.models["roberta"],
                                      self.models["vncore"],
                                      self.models["device"])
            contexts["article"] = torch.tensor(art_embed).unsqueeze(0)  # Add sequence dim
            contexts["article_mask"] = None
        
        # Process caption
        caption = item.get("caption", "")
        caption_ids = self.tokenizer.encode(caption, 
                                          return_tensors="pt",
                                          truncation=True,
                                          max_length=self.max_length)[0]
        
        return {
            "image": img_tensor,
            "contexts": contexts,
            "caption_ids": caption_ids,
            "caption": caption
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
        return self.decoder(prev_target, contexts, incremental_state)
    
    def generate(self, contexts, max_length=100, temperature=1.0):
        """Generate caption from contexts"""
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
        num_workers=config["num_workers"]
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"]
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
    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0
        total_tokens = 0

        # Training phase
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            caption_ids = batch["caption_ids"].to(device)
            contexts = {k: v.to(device) for k, v in batch["contexts"].items()}
            
            optimizer.zero_grad()
            logits, _ = model(caption_ids[:, :-1], contexts)
            # loss = criterion(
            #     logits.view(-1, config["vocab_size"]),
            #     caption_ids[:, 1:].contiguous().view(-1)
            # )
            loss, nll_loss = criterion(
                logits.view(-1, logits.size(-1)),
                caption_ids[:, 1:].contiguous().view(-1)
            )
            # Normalize loss by number of tokens
            num_tokens = (caption_ids[:, 1:] != config["decoder_params"]["padding_idx"]).sum()
            normalized_loss = loss / num_tokens
            # loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            # optimizer.step()
            
            # train_loss += loss.item()
            # Backward pass
            optimizer.zero_grad()
            normalized_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            
            total_loss += loss.item()
            total_tokens += num_tokens.item()
        avg_loss = total_loss / total_tokens
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
                
                loss, nll_loss = criterion(
                    logits.view(-1, logits.size(-1)),
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
        num_workers=config["num_workers"]
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
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            caption_ids = batch["caption_ids"].to(device)
            contexts = {k: v.to(device) for k, v in batch["contexts"].items()}
            
            # Forward pass
            logits, _ = model(caption_ids[:, :-1], contexts)
            
            # Calculate loss
            
            loss, nll_loss = criterion(
                logits.view(-1, logits.size(-1)),
                caption_ids[:, 1:].contiguous().view(-1)
            )
            # Normalize loss by number of tokens
            num_tokens = (caption_ids[:, 1:] != config["decoder_params"]["padding_idx"]).sum()
            test_loss += loss.item()
            test_total_tokens += num_tokens.item()
            
            # Generate captions
            for i in range(len(batch["image"])):
                contexts_i = {k: v[i].unsqueeze(0) for k, v in contexts.items()}
                generated_ids = model.generate(contexts_i)
                generated_caption = test_dataset.tokenizer.decode(generated_ids)
                all_predictions.append({
                    "image_path": batch["image_path"][i],
                    "true_caption": batch["caption"][i],
                    "predicted_caption": generated_caption
                })
    
    test_avg_loss = total_loss / len(test_loader)    
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
        "batch_size": 16,
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