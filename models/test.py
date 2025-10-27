
import argparse
import json
import os
import shutil
from typing import Dict, Tuple, List
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict
import re

import cv2
import numpy as np
import torch
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import py_vncorenlp
from transformers import AutoTokenizer, AutoModel
import re
import logging
from typing import Optional, Tuple, List

Image.MAX_IMAGE_PIXELS = None
SHARD_THRESHOLD = 3000
SHARD_SIZE = 2000

def _rel_or_abs(path: str, base_dir: str, use_abs: bool = False) -> str:
    return path if (use_abs or os.path.isabs(path)) else os.path.relpath(path, start=base_dir)

# Set up logging
logging.basicConfig(
    filename="preprocess_data.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class RobertaEmbedder(torch.nn.Module):

    def __init__(self, model, tokenizer, device, expected_layers: int = 25):
        super().__init__()
        self.model = model.eval()
        self.tok = tokenizer
        self.device = device
        self.expected_layers = int(expected_layers)   # 25 for *-large, 13 for *-base
        self.target_len = 256

        # quick sanity
        h = int(getattr(self.model.config, "hidden_size", 0))
        if h <= 0:
            raise ValueError("model.config.hidden_size is invalid; load a proper (PhoBERT/RoBERTa) checkpoint.")

    @torch.no_grad()
    def forward(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        proc = []
        for t in texts:
            t = (t or "").replace("<SEP>", " ")
            if not t.startswith(" "):
                t = " " + t
            proc.append(t)
        batch = self.tok(
            proc,
            padding="max_length",
            truncation=True,
            max_length=self.target_len,   
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = batch["input_ids"].to(self.device).long()
        attn_mask = batch["attention_mask"].to(self.device).long()  # keep 0/1 int

        # Hard guard (keeps CUDA from device-side assert later)
        vocab = int(self.model.config.vocab_size)
        mn, mx = int(input_ids.min().item()), int(input_ids.max().item())
        if mn < 0 or mx >= vocab:
            raise ValueError(f"OOR token id: min={mn}, max={mx}, vocab_size={vocab}")

        out = self.model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            output_hidden_states=True,
        )

        hs = torch.stack(list(out.hidden_states), dim=1).contiguous()  # [B, L_actual, 512, H]
        L_actual = hs.size(1)
        if L_actual < self.expected_layers:
            pad = self.expected_layers - L_actual
            hs = torch.cat([hs, hs.new_zeros(hs.size(0), pad, hs.size(2), hs.size(3))], dim=1)
        elif L_actual > self.expected_layers:
            hs = hs[:, -self.expected_layers:, :, :]

        attn_mask_bool = attn_mask.to(torch.bool)  # True=token, False=pad (flip later if needed)
        return hs, attn_mask_bool

def setup_models(device: torch.device, vncorenlp_path="/data2/npl/ICEK/VnCoreNLP"):
    # (Optional) make HF strictly local if all files are on disk
    # os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    # --- VnCoreNLP ---
    py_vncorenlp.download_model(save_dir=vncorenlp_path)
    vncore = py_vncorenlp.VnCoreNLP(
        annotators=["wseg"],
        save_dir=vncorenlp_path,
        max_heap_size='-Xmx15g'
    )
    print("LOADED VNCORENLP!")
    phoBERTlocal = "/data2/npl/ICEK/TnT/phoBERT_large/phobert-large"
    tokenizer = AutoTokenizer.from_pretrained(phoBERTlocal, use_fast=False, local_files_only=True)
    # DO NOT force safetensors unless files exist
    roberta   = AutoModel    .from_pretrained(phoBERTlocal, local_files_only=True).eval().to(device)

    # Quick consistency checks (fail fast in Python instead of CUDA assert)
    assert len(tokenizer) == int(roberta.config.vocab_size), \
        f"Tokenizer size {len(tokenizer)} != model vocab_size {int(roberta.config.vocab_size)}"
    assert tokenizer.pad_token_id is not None and tokenizer.pad_token == "<pad>", \
        "PhoBERT pad token missing/misaligned."

    # (Optional) If you saw SDPA-related dtype issues, prefer math path:
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

    embedder = RobertaEmbedder(roberta, tokenizer, device).to(device)
    print("LOADED phoBERT!")

    return {
        "vncore": vncore,
        "tokenizer": tokenizer,
        "embedder": embedder,
        "device": device,
    }


def segment_text(text: str, model) -> str:
    """Segment text using VnCoreNLP and join sentences with a separator"""
    sentences = re.split(r'(?<=[\.!?])\s+', text.strip())
    if not sentences:
        return ""
    segmented_sentences = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        try:
            segmented = model.word_segment(sent)[0]
            segmented_sentences.append(segmented)
        except Exception as e:
            logging.error(f"Error segmenting text: {e}")
            segmented_sentences.append(sent)
    return "<SEP>".join(segmented_sentences)

def _pad_to_len(x: torch.Tensor, target_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x: [S, D] on any device; returns (x_padded [target_len, D], mask [target_len] with True=PAD)
    If S >= target_len -> truncate and mask all False.
    """
    S, D = x.shape
    if S >= target_len:
        return x[:target_len], torch.zeros(target_len, dtype=torch.bool, device=x.device)
    pad = torch.zeros(target_len - S, D, dtype=x.dtype, device=x.device)
    out = torch.cat([x, pad], dim=0)
    mask = torch.zeros(target_len, dtype=torch.bool, device=x.device)
    mask[S:] = True
    return out, mask
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create ViWiki dataset files")
    parser.add_argument("data_dir", nargs="?", default="/data2/npl/ICEK/Wikipedia/content/ver4", help="Directory with train/val/test JSON")
    parser.add_argument("output_dir", nargs="?", default="/data2/npl/ICEK/TnT/dataset/content", help="Directory to write converted files")
    parser.add_argument("--image-out", default=None, dest="image_out",
                        help="Optional directory to copy images")
    parser.add_argument("--vncorenlp", default="/data2/npl/ICEK/VnCoreNLP",
                        help="Path to VnCoreNLP jar file")
    parser.add_argument("--checkpoint-interval", type=int, default=100,
                        help="Save checkpoint every N items")
    args = parser.parse_args()
    print("LOAD ARGS DONE!")
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = setup_models(device, args.vncorenlp)
    segmented_context=[]
    context = [' Qatar có quan_hệ hỗn_hợp với các láng_giềng trong khu_vực vịnh Ba Tư . Qatar ký một thoả_thuận hợp_tác phòng_thủ với Iran , hai quốc_gia chỉa sẻ mỏ khí_đốt đơn_lẻ lớn nhất thế_giới . Qatar là quốc_gia thứ nhì sau Pháp công_khai tuyên_bố công_nhận Hội_đồng Chuyển_tiếp Quốc_gia Libya là chính_phủ hợp_pháp của Libya trong bối_cảnh nội_chiến Libya 2011 .', ' Năm 1991 , Qatar đóng vai_trò quan_trọng trong Chiến_tranh Vùng Vịnh , đặc_biệt là trong trận Khafji khi mà xe_tăng của Qatar lăn trên đường_phố thị_trấn và hỗ_trợ hoả_lực cho Vệ_binh quốc_gia Ả_Rập_Xê_Út giao_tranh với Quân_đội Iraq . Qatar cho_phép binh_sĩ liên_quân từ Canada sử_dụng lãnh_thổ làm căn_cứ không_quân , và cũng cho_phép không_quân Hoa_Kỳ và Pháp hoạt_động trên lãnh_thổ của mình .', ' Tháng 6 năm 2017 , Bahrain , Ả_Rập_Xê_Út , Các Tiểu vương_quốc Ả_Rập_Thống nhất , Ai_Cập , Yemen ( chính_phủ Hadi ) , Libya ( chính_phủ Hoà_hợp Quốc_gia ) và Maldives chấm_dứt quan_hệ ngoại_giao với Qatar , cáo_buộc nước này " ủng_hộ chủ_nghĩa khủng_bố Anh_em Hồi_giáo " . Ả_Rập_Xê_Út giải_thích động_thái này là một biện_pháp cần_thiết trong việc bảo_vệ an_ninh của vương_quốc . Quân_đội Qatar cũng bị loại khỏi liên_minh quân_sự ở Yemen . Ai_Cập đã đóng_cửa không_phận và cảng biển cho tất_cả các phương_tiện giao_thông của Qatar .', ' Quân_đội Qatar đã tham_gia vào_cuộc tấn_công của người Ả_Rập_Xê_Út dẫn_đầu ở Yemen chống lại Shia_Houthis . Vào năm 2015 , Al_Jazeera_Mỹ đã báo_cáo : " Nhiều báo_cáo cho thấy liên_minh do Ả_Rập dẫn_đầu chống lại các nhóm đối_lập ở Yemen đã tấn_công bừa_bãi dân_thường và sử_dụng bom chùm ở các khu_vực dân_sự , vi_phạm luật_pháp quốc_tế . " Các bệnh_viện cũng đã bị ném bom bởi Ả_Rập_Xê_Út và những người hoạt_động cùng với họ . Qatar đã bị đình_chỉ khỏi liên_minh tại Yemen do cuộc khủng_hoảng ngoại_giao năm 2017 .', ' Viện Nghiên_cứu Hoà_bình Quốc_tế Stockholm ( SIPRI ) cho rằng vào giai_đoạn 2010 – 14 Qatar là nước nhập_khẩu vũ_khí lớn thứ 46 trên thế_giới . Tuy_nhiên , báo_cáo của SIPRI viết rằng Qatar đã tăng_tốc các kế_hoạch nhằm chuyển_đổi và mở_rộng đáng_kể lực_lượng_vũ_trang . Đơn đặt_hàng năm 2013 cho 62 xe_tăng và 24 pháo_tự_hành từ Đức đã được tiếp_nối vào năm 2014 bằng một_số hợp_đồng khác , bao_gồm 24 máy_bay_trực_thăng chiến_đấu và 3 máy_bay cảnh_báo và kiểm_soát sớm từ Mỹ và 2 máy_bay chở dầu từ Tây_Ban_Nha . Năm 2015 , Qatar là nhà nhập_khẩu vũ_khí lớn thứ 16 trên thế_giới và năm 2016 , xếp thứ 11 , theo SIPRI .', ' Qatar duy_trì lực_lượng quân_sự khiêm_tốn gồm khoảng 11.800 người , trong đó có lục_quân ( 8.500 ) , hải_quân ( 1.800 ) và không_quân ( 1.500 ) . Qatar gần đây đã ký_kết các thoả_ước phòng_thủ với Hoa_Kỳ và Anh Quốc , trước đó từng ký_kết với Pháp vào năm 1994 . Qatar giữ vai_trò tích_cực trong các nỗ_lực phòng_thủ tập_thể của Hội_đồng Hợp_tác Vùng Vịnh ; năm thành_viên còn lại là Ả_Rập_Xê_Út , Kuwait , Bahrain , UAE và Oman . Qatar có một căn_cứ không_quân lớn do Hoa_Kỳ vận_hành , tạo ra một nguồn đảm_bảo về quốc_phòng và an_ninh . Chi_tiêu quốc_phòng của Qatar chiếm khoảng 4,2% GDP vào năm 1993 và 1,5% tổng_sản_phẩm quốc_nội trong năm 2010 , năm gần đây nhất có sẵn trong cơ_sở_dữ_liệu thống_kê SIPRI . Sự hiện_diện của căn_cứ không_quân Al_Udeid được vận_hành bởi Hoa_Kỳ và một_số quốc_gia khác của Liên_Hợp_Quốc , cung_cấp một nguồn bảo_đảm quốc_phòng và an_ninh quốc_gia . Năm 2008 , Qatar chi_tiêu 2,355 tỷ USD cho quân_sự , chiếm 2,3% GDP . Lực_lượng đặc_biệt của Qatar do Pháp và các quốc_gia phương Tây khác huấn_luyện và được cho là có kỹ_năng đáng_kể . Họ từng giúp phiến_quân Libya trong trận Tripoli ( 2011 ) .', ' Năm 2015 , Qatar tham_gia chiến_dịch can_thiệp quân_sự do Ả_Rập_Xê_Út lãnh_đạo tại Yemen chống lại phiến_quân Houthis và lực_lượng trung_thành với cựu tổng_thống Ali_Abdullah_Saleh , người đã bị phế_truất trong cuộc nổi_dậy Mùa xuân Ả_Rập năm 2011 .', ' Năm 2014 , quan_hệ giữa Qatar với Bahrain , Ả_Rập_Xê_Út , và Các Tiểu vương_quốc Ả_Rập_Thống nhất trở_nên căng_thẳng do Qatar ủng_hộ Tổ_chức Anh_em Hồi_giáo và các nhóm cực_đoan tại Syria . Điều tạo nên đỉnh_điểm căng_thẳng trong ba quốc_gia nói trên là việc rút đại_sứ của họ khỏi Qatar vào tháng 3 năm 2014 . Qatar cũng tham_gia vào chiến_dịch bí_mật Timber_Sycamore do CIA dẫn_đầu để huấn_luyện và vũ_trang phiến_quân Syria .', ' Trong những năm gần đây , Qatar đã sử_dụng các chiến_binh Hồi_giáo ở một_số quốc_gia bao_gồm Ai_Cập , Syria , Libya , Somalia và Mali để tiếp_tục chính_sách đối_ngoại của mình . Các nhóm Hồi_giáo từ các nhóm Anh_em Hồi_giáo đến các nhóm Salafist đã phục_vụ như một bộ_khuếch_đại quyền_lực cho đất_nước vì họ tin rằng từ đầu Mùa xuân Ả_Rập , các nhóm này đại_diện cho làn_sóng của tương_lai . David_Cohen , Bộ_trưởng Bộ Khủng_bố và Tình_báo tài_chính tại Kho_bạc Hoa_Kỳ , nói rằng Qatar là một " khu_vực pháp_lý cho_phép tài_trợ khủng_bố " ở miền bắc Syria . Tính đến năm 2015 [ cập_nhật ] , Qatar , Ả_Rập_Xê_Út và Thổ_Nhĩ_Kỳ công_khai hỗ_trợ Quân_đội Chinh_phục , một nhóm chống chính_phủ trong Nội_chiến Syria bao_gồm Mặt_trận Al-Nusra và liên_minh Salafi khác là Ahrar ash-Sham.', ' Năm 2003 , Qatar trở_thành đại_bản_doanh Bộ_Tư_lệnh Trung_ương Hoa_Kỳ và là một trong các địa_điểm chính phát_động xâm_chiếm Iraq . Trong tháng 3 năm 2005 , một vụ đánh bom tự_sát làm một người thiệt_mạng và 15 người bị_thương tại Doha gây chấn_động toàn_quốc , do trước đó Qatar chưa từng xảy ra hành_động khủng_bố nào . Vụ đánh bom được thực_hiện bởi Omar_Ahmed_Abdullah_Ali , một cư_dân Ai_Cập ở Qatar , người đã nghi_ngờ có quan_hệ với Al-Qaeda ở Bán_đảo Ả_Rập . Năm 2011 , Qatar tham_gia can_thiệp quân_sự tại Libya và được tường_thuật là trang_bị vũ_khí cho các tổ_chức đối_lập Libya . Qatar cũng là một nhà_tài_trợ vũ_khí chủ_yếu cho các nhóm phiến_quân trong nội_chiến Syria . Qatar đang theo_đuổi thoả_thuận hoà_bình Afghanistan và vào tháng 1 năm 2012 , Taliban_Afghanistan cho biết họ đang thành_lập một văn_phòng chính_trị ở Qatar để tạo điều_kiện cho các cuộc đàm_phán . Điều này đã được thực_hiện để tạo điều_kiện cho các cuộc đàm_phán hoà_bình và với sự hỗ_trợ của các quốc_gia khác bao_gồm Hoa_Kỳ và Afghanistan . Ahmed_Rashid , viết trên tờ Financial_Times , tuyên_bố rằng thông_qua văn_phòng , Qatar đã " tạo điều_kiện cho các cuộc họp giữa Taliban và nhiều quốc_gia và tổ_chức , bao_gồm cả bộ ngoại_giao Hoa_Kỳ , Liên_Hợp_Quốc , Nhật_Bản , một_số chính_phủ châu_Âu và các tổ_chức phi_chính_phủ , tất_cả Những người đã cố_gắng thúc_đẩy ý_tưởng về các cuộc đàm_phán hoà_bình . Các đề_xuất vào tháng 9 năm 2017 của các tổng_thống của cả Hoa_Kỳ và Afghanistan đã dẫn đến sự phản_đối từ các quan_chức cấp cao của Bộ Ngoại_giao Hoa_Kỳ .']
    for sentence in context:
        segmented_context.append(segment_text(sentence, models["vncore"]))
    art_feats_b, attn_mask_b = models["embedder"](segmented_context)   
    art_feats  = art_feats_b[0].contiguous()                    
    art_mask = (~attn_mask_b[0].bool()).contiguous()
    print(f"[DEBUG] art_feats shape: {art_feats.shape}, art_mask: {art_mask.shape}")