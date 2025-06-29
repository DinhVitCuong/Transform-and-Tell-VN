from transformers import AutoModel
from pathlib import Path

# Thư mục chứa pytorch_model.bin + config.json của PhoBERT
src = "/data/npl/ICEK/TnT/phoBERTv2/phobert-base-v2"
dst = "/data/npl/ICEK/TnT/phoBERTv2/phobert-base-v2-safe"

# Load model từ local
model = AutoModel.from_pretrained(src, local_files_only=True)

# Tạo thư mục đích
Path(dst).mkdir(exist_ok=True)

# Lưu lại với safe_serialization=True
model.save_pretrained(dst, safe_serialization=True)

print("Đã convert xong, safetensors nằm trong:", dst)
