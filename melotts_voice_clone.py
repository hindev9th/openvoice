import os
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from openvoice.download_utils import load_or_download_config, load_or_download_model
from melo import api

# Configuration
device = "cuda:0" if torch.cuda.is_available() else "cpu"
output_dir = 'outputs_v2'
os.makedirs(output_dir, exist_ok=True)

config = load_or_download_config()
tone_color_converter = ToneColorConverter(config, device=device)

ckpt = load_or_download_model()
tone_color_converter.load_ckpt(ckpt)

# Get reference speaker embedding (the voice you want to clone)
# reference_speaker = '/mnt/data/python_project/OpenVoice_ONNX/resources/seps.wav'
reference_speaker = 'resources/demo_speaker0.mp3'
# reference_speaker = '/home/hin/MyProject/python/tools/dataset/wavs/segment10_46309.wav'
target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=True)

print(target_se)
print(audio_name)
# Define texts for different languages
texts = {
    # 'EN': "I’m Vu Quyet, CEO of Loca AI Technology Joint Stock Company. Welcome to our company.",
    'VI': '''Nó thét lên trước cảnh tượng hãi hùng. Mế nó đầu tóc rũ rượi, mặt mũi tím bầm, toàn thân giãy giụa, hơi thở khò khè, dồn dập bởi bàn tay nặng trịch của cha nó đang siết chặt vào cổ họng. Trời ơi, lão ta với bộ râu tóc rối bù bạc trắng, người nồng nặc mùi rượu như một con ma rừng. Không kịp suy nghĩ gì nữa, nó lấy hết sức mà bổ cả cái gùi ngô vào đầu lão. Lão hơi choáng, buông tay ra. Mế ngã vật xuống đất như một chiếc lá rụng. Nó ngồi thụp xuống, đỡ lấy mế thì bỗng bị lão đạp cho một cái ngã sõng soài. Rồi lão lại nhảy xổ vào mế nó mà đấm đá túi bụi. Ngay lúc ấy, nó quáng quàng vơ lấy khúc gỗ đập vào đầu lão. Lão ôm đầu rít lên, loạng choạng, trượt chân vào bắp ngô đang lăn lóc dưới sàn nhà rồi ngã ngửa ra, đập đầu vào cạnh bàn. Một dòng máu đỏ tươi rỉ ra từ mái tóc bạc xác xơ của lão. Cái miệng lão méo xệch. Đôi mắt mở trắng dã. Nó hét lên thất kinh. Mế run rẩy ôm ghì nó vào lòng.
- Ông ta… cha… chết rồi! Mế ơi… con giết người! - lời nó tắc nghẹn trong nức nở - làm sao đây?..''',
}

src_path = f'{output_dir}/tmp.wav'
speed = 0.9  # Adjustable speed

# Process each language
for language, text in texts.items():
    print(f"Processing {language} text: {text}")
    
    # Initialize MeloTTS for the current language
    model = api.TTS(
        language=language, 
        device=device, 
        # config_path="/mnt/data/python_project/OpenVoice_ONNX/finetuned_model/hue/config.json", 
        # ckpt_path="/mnt/data/python_project/OpenVoice_ONNX/finetuned_model/hue/G_97000.pth",
    )
    
    speaker_ids = model.hps.data.spk2id
    
    # Process each available speaker for this language
    for speaker_key in speaker_ids.keys():
        speaker_id = speaker_ids[speaker_key]
        speaker_key_formatted = speaker_key.lower().replace('_', '-')
        
        print(f"  Using speaker: {speaker_key_formatted}")
        
        # Load source speaker embedding
        source_se = torch.load(f'checkpoints_v2/base_speakers/ses/{speaker_key_formatted}.pth', map_location=device)
        
        # Generate speech with MeloTTS
        model.tts_to_file(text, speaker_id, src_path, speed=speed)
        save_path = f'{output_dir}/output_v2_{language}_{speaker_key_formatted}.wav'

        # Convert using tone color converter
        encode_message = "@MyShell"
        tone_color_converter.convert(
            audio_src_path=src_path, 
            src_se=source_se, 
            tgt_se=target_se, 
            output_path=save_path,
            message=encode_message)
        
        print(f"  Saved to: {save_path}")

print("Voice cloning completed for all languages!")