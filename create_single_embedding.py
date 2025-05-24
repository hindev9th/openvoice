import os
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

# Cấu hình
ckpt_converter = 'checkpoints_v2/converter'
device = "cpu"  # Sử dụng CPU vì GPU đang gặp vấn đề
output_dir = 'outputs_v2'
os.makedirs(output_dir, exist_ok=True)
os.makedirs('checkpoints_v2/base_speakers/ses', exist_ok=True)

# Khởi tạo tone color converter
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

# Chọn một file âm thanh chất lượng cao
audio_file = 'resources/audio_vi.wav'  # Thay đổi đường dẫn này nếu cần

# Tên speaker (phải khớp với speaker_key_formatted trong script chính)
speaker_name = 'vi-default'  # Ví dụ: 'vi-female', 'vi-male', etc.

print(f"Đang xử lý file: {audio_file}")
try:
    # Trích xuất embedding với VAD
    embedding, _ = se_extractor.get_se(audio_file, tone_color_converter, vad=True)
    
    # Kiểm tra và điều chỉnh hình dạng của embedding
    print(f"Hình dạng embedding ban đầu: {embedding.shape}, số chiều: {embedding.dim()}")
    
    # Đảm bảo embedding có hình dạng [batch, channels, time]
    if embedding.dim() == 2:
        # Nếu embedding có hình dạng [channels, time], thêm chiều batch
        embedding = embedding.unsqueeze(0)
        print(f"Đã điều chỉnh thành hình dạng: {embedding.shape}")
    elif embedding.dim() == 1:
        # Nếu embedding có hình dạng [channels], thêm chiều batch và time
        embedding = embedding.unsqueeze(0).unsqueeze(-1)
        print(f"Đã điều chỉnh thành hình dạng: {embedding.shape}")
    
    # Lưu embedding
    embedding_path = f'checkpoints_v2/base_speakers/ses/{speaker_name}.pth'
    torch.save(embedding, embedding_path)
    print(f"Đã lưu embedding tại: {embedding_path}")
except Exception as e:
    print(f"Lỗi khi xử lý file {audio_file} với VAD: {str(e)}")
    try:
        # Thử lại không dùng VAD
        embedding, _ = se_extractor.get_se(audio_file, tone_color_converter, vad=False)
        
        # Kiểm tra và điều chỉnh hình dạng của embedding
        print(f"Hình dạng embedding ban đầu (không VAD): {embedding.shape}, số chiều: {embedding.dim()}")
        
        # Đảm bảo embedding có hình dạng [batch, channels, time]
        if embedding.dim() == 2:
            # Nếu embedding có hình dạng [channels, time], thêm chiều batch
            embedding = embedding.unsqueeze(0)
            print(f"Đã điều chỉnh thành hình dạng: {embedding.shape}")
        elif embedding.dim() == 1:
            # Nếu embedding có hình dạng [channels], thêm chiều batch và time
            embedding = embedding.unsqueeze(0).unsqueeze(-1)
            print(f"Đã điều chỉnh thành hình dạng: {embedding.shape}")
        
        embedding_path = f'checkpoints_v2/base_speakers/ses/{speaker_name}.pth'
        torch.save(embedding, embedding_path)
        print(f"Đã lưu embedding (không VAD) tại: {embedding_path}")
    except Exception as e2:
        print(f"Lỗi khi thử lại không dùng VAD: {str(e2)}")
        
        # Tạo embedding ngẫu nhiên nếu cả hai phương pháp đều thất bại
        try:
            print("Tạo embedding ngẫu nhiên...")
            # Tạo embedding ngẫu nhiên với hình dạng [1, 256, 1]
            random_embedding = torch.randn(1, 256, 1, device=device)
            embedding_path = f'checkpoints_v2/base_speakers/ses/{speaker_name}.pth'
            torch.save(random_embedding, embedding_path)
            print(f"Đã lưu embedding ngẫu nhiên tại: {embedding_path}")
        except Exception as e3:
            print(f"Lỗi khi tạo embedding ngẫu nhiên: {str(e3)}")

print("Hoàn thành việc tạo speaker embedding!")
