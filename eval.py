import torch
from main import load_saved_model, evaluate_model, NewsCaptionDataset, pad_and_collate, _h5_worker_init_fn
from torch.utils.data import DataLoader
import os

# Configuration (must match the training configuration)
config = {
    "data_dir": "/data2/npl/ICEK/TnT/dataset/content",
    "output_dir": "/data2/npl/ICEK/TnT/output",
    "vncorenlp_path": "/data2/npl/ICEK/VnCoreNLP",
    "vocab_size": 64001,
    "embed_dim": 1024,
    "batch_size": 8,
    "num_workers": 0,
    "epochs": 400,
    "lr": 1e-4,
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

# Path to the saved model
model_path = os.path.join(config["output_dir"], "best_model.pth")  # or "transform_and_tell_model.pth"

# Load the model
loaded_model, loaded_models = load_saved_model(config, model_path)

# Run evaluation
test_loss, predictions = evaluate_model(loaded_model, loaded_models, config)

# Print results
print(f"Test Loss: {test_loss:.4f}")
for pred in predictions[:5]:  # Print first 5 predictions as an example
    print(f"Image: {pred['image_path']}")
    print(f"True Caption: {pred['true_caption']}")
    print(f"Predicted Caption: {pred['predicted_caption']}")
    print()