import os
import torch
import glob
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

# Cấu hình
ckpt_converter = 'checkpoints_v2/converter'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
output_dir = 'outputs_v2'
os.makedirs(output_dir, exist_ok=True)
os.makedirs('checkpoints_v2/base_speakers/ses', exist_ok=True)

# Khởi tạo tone color converter
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

# Thư mục chứa các file âm thanh tiếng Việt
# Đặt các file .wav hoặc .mp3 vào thư mục này
audio_dir = '/home/hin/MyProject/python/tools/dataset/wavs'

# Tên cho speaker embedding (ví dụ: vi-default, vi-female, vi-male)
embedding_name = 'vi-hue'

# Lấy danh sách các file âm thanh
audio_files = []
for ext in ['*.wav', '*.mp3']:
    audio_files.extend(glob.glob(os.path.join(audio_dir, ext)))

if not audio_files:
    print(f"Không tìm thấy file âm thanh nào trong thư mục {audio_dir}")
    exit(1)

print(f"Tìm thấy {len(audio_files)} file âm thanh")

# Trích xuất speaker embedding từ mỗi file âm thanh
all_embeddings = []
for audio_file in audio_files:
    print(f"Đang xử lý file: {audio_file}")
    try:
        # Trích xuất embedding với VAD (Voice Activity Detection)
        embedding, _ = se_extractor.get_se(audio_file, tone_color_converter, vad=True)
        all_embeddings.append(embedding)
        print(f"  Đã trích xuất embedding thành công")
    except Exception as e:
        print(f"  Lỗi khi xử lý file {audio_file}: {str(e)}")

# Tính trung bình các embedding
if all_embeddings:
    avg_embedding = torch.mean(torch.stack(all_embeddings), dim=0)
    
    # Lưu embedding
    embedding_path = f'checkpoints_v2/base_speakers/ses/{embedding_name}.pth'
    torch.save(avg_embedding, embedding_path)
    print(f"Đã lưu embedding tại: {embedding_path}")
else:
    print("Không thể tạo embedding do không có file âm thanh hợp lệ")

print("Hoàn thành việc tạo speaker embedding!")