import os
import torch
import tempfile
import soundfile as sf
from datasets import load_dataset

from openvoice import se_extractor
from openvoice.api import ToneColorConverter

# Cấu hình
ckpt_converter = 'checkpoints_v2/converter'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
output_dir = 'outputs_v2'
embedding_name = 'vi-default'

os.makedirs(output_dir, exist_ok=True)
os.makedirs('checkpoints_v2/base_speakers/ses', exist_ok=True)

# Khởi tạo converter
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

# Tải dataset dạng stream từ Hugging Face
dataset = load_dataset("capleaf/viVoice", split="train", streaming=True)

# Số lượng audio mẫu để trích xuất (tùy chỉnh theo nhu cầu)
max_samples = 100000

print(f"⏳ Đang xử lý tối đa {max_samples} mẫu từ dataset viVoice...")

all_embeddings = []
for i, example in enumerate(dataset):
    if i >= max_samples:
        break

    duration = len(example["audio"]["array"]) / example["audio"]["sampling_rate"]
    if duration < 3.0:
        print(f"   ⏭ Bỏ qua mẫu quá ngắn ({duration:.2f} giây)")
        continue
    
    print(f"🔊 Mẫu {i + 1}")

    try:
        # Ghi tạm file audio ra đĩa (bắt buộc vì openvoice dùng file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            sf.write(tmp_wav.name, example["audio"]["array"], example["audio"]["sampling_rate"])

            # Trích xuất speaker embedding
            embedding, _ = se_extractor.get_se(tmp_wav.name, tone_color_converter, vad=True)
            all_embeddings.append(embedding)
            print("   ✔ Đã trích xuất embedding")

    except Exception as e:
        print(f"   ❌ Lỗi khi xử lý: {e}")

# Trung bình và lưu embedding
if all_embeddings:
    avg_embedding = torch.mean(torch.stack(all_embeddings), dim=0)
    save_path = f'checkpoints_v2/base_speakers/ses/{embedding_name}.pth'
    torch.save(avg_embedding, save_path)
    print(f"✅ Đã lưu embedding tại: {save_path}")
else:
    print("❌ Không có embedding nào được tạo.")
