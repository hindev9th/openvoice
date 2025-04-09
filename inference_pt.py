import torch
import librosa
import soundfile as sf
import os
import argparse
import numpy as np
from openvoice.api import ToneColorConverter
from openvoice.mel_processing import spectrogram_torch

def load_audio(audio_path, sr=16000):
    """Load and preprocess audio file"""
    audio, _ = librosa.load(audio_path, sr=sr)
    return audio

def extract_speaker_embedding(model, audio_path, config):
    """Extract speaker embedding from audio file"""
    audio = load_audio(audio_path, sr=config['data']['sampling_rate'])
    y = torch.FloatTensor(audio).to(model.device)
    y = y.unsqueeze(0)
    spec = spectrogram_torch(y, 
                           config['data']['filter_length'],
                           config['data']['sampling_rate'],
                           config['data']['hop_length'],
                           config['data']['win_length'],
                           center=False).to(model.device)
    
    with torch.no_grad():
        g = model.model.ref_enc(spec.transpose(1, 2)).unsqueeze(-1)
    
    return g

def convert_voice(model, input_audio, source_se, target_se, config):
    """Convert voice using PyTorch model"""
    # Convert to spectrogram
    y = torch.FloatTensor(input_audio).to(model.device).unsqueeze(0)
    spec = spectrogram_torch(y, 
                           config['data']['filter_length'],
                           config['data']['sampling_rate'],
                           config['data']['hop_length'],
                           config['data']['win_length'],
                           center=False).to(model.device)
    spec_lengths = torch.LongTensor([spec.size(-1)]).to(model.device)
    
    # Run inference
    with torch.no_grad():
        output, mask, _ = model.model.voice_conversion(
            spec, spec_lengths, source_se, target_se, tau=0.3
        )
    
    # Convert output to audio
    converted_audio = output.squeeze().cpu().numpy()
    
    # Normalize the audio
    if np.max(np.abs(converted_audio)) > 0:
        converted_audio = converted_audio / np.max(np.abs(converted_audio))
    
    return converted_audio

def main():
    parser = argparse.ArgumentParser(description='Convert audio using OpenVoice (PyTorch)')
    parser.add_argument('--config_path', type=str, required=True, help='Path to the model config file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the PyTorch model file')
    parser.add_argument('--input_audio', type=str, required=True, help='Path to input audio file')
    parser.add_argument('--reference_audio', type=str, required=True, help='Path to reference audio file')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save output')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cpu or cuda)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to print more information')
    
    args = parser.parse_args()
    
    # Load config
    import json
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize model
    print("Loading model...")
    try:
        # First try loading with ToneColorConverter
        model = ToneColorConverter(args.model_path, device=args.device)
    except Exception as e:
        print(f"Error loading model with ToneColorConverter: {e}")
        print("Trying alternative loading method...")
        try:
            # Try loading the state dict directly
            checkpoint = torch.load(args.model_path, map_location=args.device)
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    model = ToneColorConverter(config_path=args.config_path, device=args.device)
                    model.model.load_state_dict(checkpoint['model'])
                elif 'model_state_dict' in checkpoint:
                    model = ToneColorConverter(config_path=args.config_path, device=args.device)
                    model.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    # Try to find any key that might contain the model state
                    for key in checkpoint.keys():
                        if isinstance(checkpoint[key], dict) and any('weight' in k for k in checkpoint[key].keys()):
                            model = ToneColorConverter(config_path=args.config_path, device=args.device)
                            model.model.load_state_dict(checkpoint[key])
                            break
            else:
                model = ToneColorConverter(config_path=args.config_path, device=args.device)
                model.model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Error loading model state dict: {e}")
            raise
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract speaker embeddings
    print("Extracting speaker embeddings...")
    source_se = extract_speaker_embedding(model, args.input_audio, config)
    target_se = extract_speaker_embedding(model, args.reference_audio, config)
    
    if args.debug:
        print(f"Source speaker embedding shape: {source_se.shape}")
        print(f"Target speaker embedding shape: {target_se.shape}")
    
    # Load input audio
    print("Loading input audio...")
    input_audio = load_audio(args.input_audio, sr=config['data']['sampling_rate'])
    
    # Convert voice
    print("Running voice conversion...")
    converted_audio = convert_voice(model, input_audio, source_se, target_se, config)
    
    if args.debug:
        print(f"Converted audio shape: {converted_audio.shape}")
        print(f"Converted audio min/max: {np.min(converted_audio)}/{np.max(converted_audio)}")
    
    # Save the converted audio
    output_filename = os.path.basename(args.input_audio)
    output_path = os.path.join(args.output_dir, f"converted_{output_filename}")
    print(f"Saving converted audio to {output_path}")
    
    # Ensure the audio is in the correct format and range
    converted_audio = np.clip(converted_audio, -1.0, 1.0)
    sf.write(output_path, converted_audio, config['data']['sampling_rate'], format='WAV', subtype='PCM_16')
    
    print("Conversion completed!")

if __name__ == "__main__":
    main() 