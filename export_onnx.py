import torch
import torch.onnx
from openvoice.api import ToneColorConverter
import argparse
import numpy as np
import os
import json
from huggingface_hub import login, HfFolder

def check_huggingface_auth():
    """Check if Hugging Face authentication is set up"""
    token = HfFolder.get_token()
    if not token:
        print("Warning: Hugging Face authentication not found.")
        print("Please set up authentication using one of these methods:")
        print("1. Set environment variable: export HUGGING_FACE_HUB_TOKEN=your_token_here")
        print("2. Run: huggingface-cli login")
        print("3. Visit: https://huggingface.co/settings/tokens to get your token")
        return False
    return True

def export_tone_converter(model, output_path, device='cpu'):
    """Export the ToneColorConverter to ONNX"""
    # Get the filter length from the model's config
    filter_length = model.hps.data.filter_length
    spec_channels = filter_length // 2 + 1
    
    # Create dummy inputs for the model
    # 1. Spectrogram input (source audio)
    dummy_audio = torch.randn(1, spec_channels, 100).to(device)  # [batch, n_fft//2+1, time]
    # 2. Audio lengths
    dummy_lengths = torch.tensor([100]).to(device)
    # 3. Source speaker embedding [batch, channels, 1]
    dummy_src_se = torch.randn(1, 256, 1).to(device)
    # 4. Target speaker embedding [batch, channels, 1]
    dummy_tgt_se = torch.randn(1, 256, 1).to(device)
    
    # Create a wrapper class that uses the voice_conversion method
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, spec, spec_lengths, src_se, tgt_se):
            # Call voice_conversion with default parameters
            output, _, _ = self.model.voice_conversion(spec, spec_lengths, src_se, tgt_se, tau=0.3)
            return output
    
    # Create the wrapper and export
    wrapper = ModelWrapper(model.model)
    wrapper.eval()
    
    # Export the model
    torch.onnx.export(
        wrapper,
        (dummy_audio, dummy_lengths, dummy_src_se, dummy_tgt_se),
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['spectrogram_input', 'spectrogram_lengths', 'src_speaker_embedding', 'tgt_speaker_embedding'],
        output_names=['output_audio'],
        dynamic_axes={
            'spectrogram_input': {0: 'batch_size', 2: 'time'},
            'spectrogram_lengths': {0: 'batch_size'},
            'src_speaker_embedding': {0: 'batch_size'},
            'tgt_speaker_embedding': {0: 'batch_size'},
            'output_audio': {0: 'batch_size', 1: 'time'}
        }
    )
    print(f"ToneColorConverter exported to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Export OpenVoice ToneColorConverter to ONNX')
    parser.add_argument('--config_path', type=str, required=True, help='Path to the model config file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the ONNX model')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu or cuda)')
    parser.add_argument('--hf_token', type=str, help='Hugging Face access token')
    
    args = parser.parse_args()
    
    # Check and set up Hugging Face authentication
    if args.hf_token:
        login(args.hf_token)
    elif not check_huggingface_auth():
        print("Continuing without Hugging Face authentication. This may cause issues with watermark model download.")
    
    # Load config
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    
    try:
        # Initialize the model with config file
        model = ToneColorConverter(args.config_path, device=args.device)
        
        # Disable watermark model
        model.watermark_model = None
        
        # Load model weights
        checkpoint = torch.load(args.model_path, map_location=args.device, weights_only=True)
        model.model.load_state_dict(checkpoint['model'])
        
        # Set model to evaluation mode
        model.model.eval()
        
        # Export the tone converter
        export_tone_converter(model, args.output_path, args.device)
    except Exception as e:
        print(f"Error during model initialization: {str(e)}")
        print("If this is related to Hugging Face authentication, please provide your token using --hf_token")

if __name__ == "__main__":
    main() 