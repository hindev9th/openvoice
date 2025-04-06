import onnxruntime as ort
import numpy as np
import librosa
import torch
import os
import argparse
from openvoice.mel_processing import spectrogram_torch

def load_audio(audio_path, sr=16000):
    """Load and preprocess audio file"""
    audio, _ = librosa.load(audio_path, sr=sr)
    return audio

def extract_speaker_embedding(audio_path, session, config):
    """Extract speaker embedding from reference audio using ONNX model"""
    # Load the audio
    audio = load_audio(audio_path)
    
    # Convert to tensor and add batch dimension
    y = torch.FloatTensor(audio).unsqueeze(0)
    
    # Convert to spectrogram using config parameters
    spec = spectrogram_torch(y, 
                           config['data']['filter_length'],
                           config['data']['sampling_rate'], 
                           config['data']['hop_length'], 
                           config['data']['win_length'],
                           center=False)
    spec_lengths = torch.LongTensor([spec.size(-1)])
    
    # Create dummy speaker embeddings
    dummy_src_se = np.zeros((1, 256, 1), dtype=np.float32)
    dummy_tgt_se = np.zeros((1, 256, 1), dtype=np.float32)
    
    # Run speaker embedding extraction
    outputs = session.run(
        None,
        {
            'spectrogram_input': spec.numpy(),
            'spectrogram_lengths': spec_lengths.numpy(),
            'src_speaker_embedding': dummy_src_se,
            'tgt_speaker_embedding': dummy_tgt_se
        }
    )
    
    # Get speaker embedding and ensure correct shape [batch, 256, 1]
    speaker_embedding = torch.from_numpy(outputs[0])  # Get the raw output
    if speaker_embedding.dim() > 2:  # If we have extra dimensions
        speaker_embedding = speaker_embedding.squeeze()  # Remove all single dimensions
    
    # Ensure we have exactly 256 channels
    if speaker_embedding.size(0) > 256:
        # If we have more channels, take the first 256
        speaker_embedding = speaker_embedding[:256]
    elif speaker_embedding.size(0) < 256:
        # If we have fewer channels, pad with zeros
        padding = torch.zeros(256 - speaker_embedding.size(0))
        speaker_embedding = torch.cat([speaker_embedding, padding])
    
    # Reshape to [batch, 256, 1]
    speaker_embedding = speaker_embedding.reshape(1, 256, 1)
    return speaker_embedding

def convert_voice(input_audio, source_se, target_se, session, config):
    """Convert voice using ONNX model"""
    # Convert to spectrogram
    y = torch.FloatTensor(input_audio).unsqueeze(0)
    spec = spectrogram_torch(y, config['data']['filter_length'],
                           config['data']['sampling_rate'], config['data']['hop_length'], config['data']['win_length'],
                           center=False)
    spec_lengths = torch.LongTensor([spec.size(-1)])
    
    # Ensure speaker embeddings have the correct shape [batch, 256, 1]
    source_se = source_se.reshape(1, 256, 1)
    target_se = target_se.reshape(1, 256, 1)
    
    # Run inference
    outputs = session.run(
        None,
        {
            'spectrogram_input': spec.numpy(),
            'spectrogram_lengths': spec_lengths.numpy(),
            'src_speaker_embedding': source_se.numpy(),
            'tgt_speaker_embedding': target_se.numpy()
        }
    )
    
    # Ensure the output is in the correct format (float32, mono)
    converted_audio = outputs[0][0].astype(np.float32)
    if len(converted_audio.shape) > 1:
        converted_audio = converted_audio.squeeze()
    return converted_audio

def main():
    parser = argparse.ArgumentParser(description='Convert audio using OpenVoice')
    parser.add_argument('--config_path', type=str, required=True, help='Path to the model config file')
    parser.add_argument('--input_audio', type=str, required=True, help='Path to input audio file')
    parser.add_argument('--reference_audio', type=str, required=True, help='Path to reference audio file')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save output')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Load config
    import json
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize ONNX runtime session
    session = ort.InferenceSession("tone_converter.onnx")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract speaker embeddings using ONNX model
    print("Extracting speaker embeddings...")
    source_se = extract_speaker_embedding(args.input_audio, session, config)
    target_se = extract_speaker_embedding(args.reference_audio, session, config)
    
    # Load input audio
    print("Loading input audio...")
    input_audio = load_audio(args.input_audio)
    
    # Convert voice
    print("Running voice conversion...")
    converted_audio = convert_voice(input_audio, source_se, target_se, session, config)
    
    # Save the converted audio
    output_filename = os.path.basename(args.input_audio)
    output_path = os.path.join(args.output_dir, f"converted_{output_filename}")
    print(f"Saving converted audio to {output_path}")
    
    # Ensure the audio is in the correct format and range
    converted_audio = np.clip(converted_audio, -1.0, 1.0)
    import soundfile as sf
    sf.write(output_path, converted_audio, config['data']['sampling_rate'], format='WAV', subtype='PCM_16')
    
    print("Conversion completed!")

if __name__ == "__main__":
    main() 