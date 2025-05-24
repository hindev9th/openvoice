import torch
import os

# Đường dẫn đến file embedding
embedding_path = '/media/hin/data/python_project/OpenVoice_ONNX/checkpoints_v2/base_speakers/ses/vi-default.pth'

# Kiểm tra xem file có tồn tại không
if not os.path.exists(embedding_path):
    print(f"Lỗi: File không tồn tại: {embedding_path}")
else:
    try:
        # Sử dụng torch.load thay vì torch.jit.load
        embedding = torch.load(embedding_path, map_location='cpu')
        
        # In thông tin về embedding
        print(f"Đã tải thành công embedding từ: {embedding_path}")
        print(f"Loại dữ liệu: {type(embedding)}")
        
        if isinstance(embedding, torch.Tensor):
            print(f"Hình dạng: {embedding.shape}")
            print(f"Số chiều: {embedding.dim()}")
            print(f"Kiểu dữ liệu: {embedding.dtype}")
            print(f"Thiết bị: {embedding.device}")
            
            # In một số giá trị mẫu
            if embedding.numel() > 0:
                print(f"Giá trị đầu tiên: {embedding.flatten()[0]}")
                if embedding.numel() > 1:
                    print(f"Giá trị thứ hai: {embedding.flatten()[1]}")
            
            # Kiểm tra xem embedding có đúng hình dạng không
            if embedding.dim() == 3:
                print("Embedding có hình dạng đúng [batch, channels, time]")
            elif embedding.dim() == 2:
                print("Embedding có hình dạng 2 chiều [channels, time], cần thêm chiều batch")
                # Thêm chiều batch
                fixed_embedding = embedding.unsqueeze(0)
                print(f"Hình dạng sau khi sửa: {fixed_embedding.shape}")
            elif embedding.dim() == 1:
                print("Embedding có hình dạng 1 chiều [channels], cần thêm chiều batch và time")
                # Thêm chiều batch và time
                fixed_embedding = embedding.unsqueeze(0).unsqueeze(-1)
                print(f"Hình dạng sau khi sửa: {fixed_embedding.shape}")
            
            # Tính tổng số tham số
            total_params = embedding.numel()
            print(f"Tổng số tham số: {total_params}")
        else:
            print("Embedding không phải là tensor, mà là:", type(embedding))
            print("Nội dung:", embedding)
    except Exception as e:
        print(f"Lỗi khi tải embedding: {str(e)}")
