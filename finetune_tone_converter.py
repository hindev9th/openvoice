import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
import librosa
import numpy as np
import os
import json
import argparse
from openvoice.api import ToneColorConverter
from openvoice.mel_processing import spectrogram_torch
from tqdm import tqdm
import random
import gc

# Set PyTorch memory allocation settings
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Maximum length for spectrograms (in time steps)
MAX_SPECTROGRAM_LENGTH = 1000  # Adjust this based on your GPU memory

class VoiceDataset(Dataset):
    def __init__(self, data_dir, config, device='cuda'):
        self.data_dir = data_dir
        self.config = config
        self.device = device
        self.audio_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
        self.n_fft = config['data']['filter_length']
        self.n_freqs = self.n_fft // 2 + 1
        
        # Find max length for padding (capped at MAX_SPECTROGRAM_LENGTH)
        self.max_length = min(MAX_SPECTROGRAM_LENGTH, self._compute_max_length())
        
    def _compute_max_length(self):
        max_len = 0
        for audio_file in self.audio_files:
            audio_path = os.path.join(self.data_dir, audio_file)
            audio, _ = librosa.load(audio_path, sr=self.config['data']['sampling_rate'])
            y = torch.FloatTensor(audio).unsqueeze(0)
            spec = spectrogram_torch(y, 
                                   self.config['data']['filter_length'],
                                   self.config['data']['sampling_rate'],
                                   self.config['data']['hop_length'],
                                   self.config['data']['win_length'],
                                   center=False)
            max_len = max(max_len, spec.size(-1))
            # Clear memory
            del y, spec
            torch.cuda.empty_cache()
            gc.collect()
        return max_len
        
    def __len__(self):
        return len(self.audio_files)
    
    def load_audio(self, audio_path):
        audio, _ = librosa.load(audio_path, sr=self.config['data']['sampling_rate'])
        return torch.FloatTensor(audio)
    
    def process_audio(self, audio):
        # Convert to spectrogram
        y = audio.unsqueeze(0)
        spec = spectrogram_torch(y, 
                               self.config['data']['filter_length'],
                               self.config['data']['sampling_rate'],
                               self.config['data']['hop_length'],
                               self.config['data']['win_length'],
                               center=False)
        
        # Ensure shape is [n_freqs, time_steps]
        if spec.size(1) != self.n_freqs:
            spec = spec.transpose(1, 2)
        
        # Truncate or pad to max length
        if spec.size(-1) > self.max_length:
            spec = spec[..., :self.max_length]
        elif spec.size(-1) < self.max_length:
            pad_size = self.max_length - spec.size(-1)
            spec = torch.nn.functional.pad(spec, (0, pad_size))
        
        # Remove any extra dimensions
        spec = spec.squeeze()
        
        # Verify final shape
        assert spec.size(0) == self.n_freqs, f"Expected {self.n_freqs} frequency bins, got {spec.size(0)}"
        
        # Clear memory
        del y
        torch.cuda.empty_cache()
        
        return spec
    
    def __getitem__(self, idx):
        audio_path = os.path.join(self.data_dir, self.audio_files[idx])
        audio = self.load_audio(audio_path)
        spec = self.process_audio(audio)
        return spec.to(self.device)

def extract_speaker_embedding(model, audio_path, config):
    """Extract speaker embedding from audio file"""
    audio, _ = librosa.load(audio_path, sr=config['data']['sampling_rate'])
    y = torch.FloatTensor(audio).to(model.device)
    y = y.unsqueeze(0)
    spec = spectrogram_torch(y, 
                           config['data']['filter_length'],
                           config['data']['sampling_rate'],
                           config['data']['hop_length'],
                           config['data']['win_length'],
                           center=False).to(model.device)
    
    # Truncate if necessary
    if spec.size(-1) > MAX_SPECTROGRAM_LENGTH:
        spec = spec[..., :MAX_SPECTROGRAM_LENGTH]
    
    with torch.no_grad():
        g = model.model.ref_enc(spec.transpose(1, 2)).unsqueeze(-1)
    
    # Clear memory
    del y, spec
    torch.cuda.empty_cache()
    
    return g

def train(model, train_loader, optimizer, device, epoch, reference_audio_path, config):
    model.model.train()
    total_loss = 0
    
    # Extract reference speaker embedding
    ref_se = extract_speaker_embedding(model, reference_audio_path, config)
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, spec in enumerate(progress_bar):
        optimizer.zero_grad()
        
        # Print original shape for debugging
        print(f"Original spec shape: {spec.shape}")
        
        # Ensure spec has shape [batch_size, n_freqs, time_steps]
        if len(spec.shape) == 2:
            spec = spec.unsqueeze(0)  # Add batch dimension
        
        # Get dimensions
        batch_size = spec.size(0)
        n_freqs = config['data']['filter_length'] // 2 + 1
        time_steps = spec.size(-1)
        
        # Reshape spec to match model's expected input shape [N, 1, Ty, n_freqs]
        spec = spec.unsqueeze(1)  # Add channel dimension
        if spec.size(2) != n_freqs:
            spec = spec.transpose(2, 3)  # Swap frequency and time dimensions
        
        print(f"Reshaped spec: {spec.shape}")
        
        # Extract source speaker embedding from input audio
        try:
            src_se = model.model.ref_enc(spec.squeeze(1).transpose(1, 2)).unsqueeze(-1)
        except RuntimeError as e:
            print(f"Error in ref_enc: {e}")
            print(f"Spec shape: {spec.shape}")
            continue
        
        # Forward pass through voice conversion with gradient checkpointing
        try:
            def forward_func(spec, lengths, src_se, ref_se):
                return model.model.voice_conversion(spec, lengths, src_se, ref_se, tau=0.3)
            
            spec_lengths = torch.LongTensor([time_steps]).to(device)
            output, mask, (z, z_p, z_hat) = checkpoint(
                forward_func,
                spec,
                spec_lengths,
                src_se,
                ref_se
            )
        except RuntimeError as e:
            print(f"Error in voice_conversion: {e}")
            print(f"Spec shape: {spec.shape}")
            print(f"src_se shape: {src_se.shape}")
            print(f"ref_se shape: {ref_se.shape}")
            continue
        
        # Ensure output and spec have the same dimensions
        if output.size(-1) != spec.size(-1):
            # Truncate or pad output to match spec length
            if output.size(-1) > spec.size(-1):
                output = output[..., :spec.size(-1)]
                mask = mask[..., :spec.size(-1)]
            else:
                pad_size = spec.size(-1) - output.size(-1)
                output = torch.nn.functional.pad(output, (0, pad_size))
                mask = torch.nn.functional.pad(mask, (0, pad_size))
        
        # Calculate multiple losses
        # 1. L1 loss between input and output spectrograms (with mask)
        l1_loss = nn.L1Loss()(output * mask, spec * mask)
        
        # 2. Consistency loss between z and z_hat (should be similar after conversion)
        consistency_loss = nn.L1Loss()(z, z_hat)
        
        # 3. Speaker embedding similarity loss
        # Extract speaker embedding from output audio
        output_se = model.model.ref_enc(output.squeeze(1).transpose(1, 2)).unsqueeze(-1)
        speaker_loss = nn.L1Loss()(output_se, ref_se)
        
        # 4. Total loss is a weighted sum
        loss = l1_loss + 0.1 * consistency_loss + 0.5 * speaker_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({
            'total_loss': total_loss / (batch_idx + 1),
            'l1_loss': l1_loss.item(),
            'consistency_loss': consistency_loss.item(),
            'speaker_loss': speaker_loss.item()
        })
        
        # Clear memory
        del output, mask, z, z_p, z_hat, l1_loss, consistency_loss, speaker_loss, loss
        torch.cuda.empty_cache()
        gc.collect()
    
    return total_loss / len(train_loader)

def remove_weight_norm_for_onnx(model):
    """Remove weight normalization from all layers for ONNX export"""
    for module in model.modules():
        if hasattr(module, 'remove_weight_norm'):
            module.remove_weight_norm()
        elif isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            try:
                remove_weight_norm(module)
            except:
                pass
    return model

def main():
    parser = argparse.ArgumentParser(description='Fine-tune ToneColorConverter model')
    parser.add_argument('--config_path', type=str, required=True, help='Path to config file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing training audio files')
    parser.add_argument('--reference_audio', type=str, required=True, help='Path to reference audio file')
    parser.add_argument('--output_dir', type=str, default='finetuned_model', help='Directory to save fine-tuned model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--max_length', type=int, default=1000, help='Maximum spectrogram length')
    parser.add_argument('--export_onnx', action='store_true', help='Export model to ONNX format')
    parser.add_argument('--onnx_path', type=str, default='model.onnx', help='Path to save ONNX model')
    
    args = parser.parse_args()
    
    # Update global max length
    global MAX_SPECTROGRAM_LENGTH
    MAX_SPECTROGRAM_LENGTH = args.max_length
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load config
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize model
    model = ToneColorConverter(args.config_path, device=args.device)
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=args.device)
    
    # If exporting to ONNX, remove weight norm before loading state dict
    if args.export_onnx:
        print("Removing weight normalization for ONNX export...")
        # First remove weight norm from the model
        model.model = remove_weight_norm_for_onnx(model.model)
        
        # Create a new state dict without weight norm parameters
        new_state_dict = {}
        for key, value in checkpoint['model'].items():
            # Skip weight norm parameters
            if 'weight_g' in key or 'weight_v' in key:
                continue
            new_state_dict[key] = value
        
        # Load the modified state dict
        model.model.load_state_dict(new_state_dict, strict=False)
        
        # Export to ONNX
        print(f"Exporting model to {args.onnx_path}...")
        dummy_input = torch.randn(1, 513, 100).to(args.device)  # Adjust shape based on your model's input
        torch.onnx.export(
            model.model,
            dummy_input,
            args.onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'},
                         'output': {0: 'batch_size'}}
        )
        print("ONNX export completed!")
        return
    else:
        # Normal training mode - load state dict as is
        model.model.load_state_dict(checkpoint['model'])
    
    # Create dataset and dataloader
    dataset = VoiceDataset(args.data_dir, config, device=args.device)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.model.parameters(), lr=args.learning_rate)
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(args.epochs):
        # Randomly select a reference voice for this epoch
        ref_voice = random.choice(os.listdir(args.reference_audio))
        ref_se = extract_speaker_embedding(model, os.path.join(args.reference_audio, ref_voice), config)
        
        loss = train(model, train_loader, optimizer, args.device, epoch, os.path.join(args.reference_audio, ref_voice), config)
        
        # Save checkpoint if loss improved
        if loss < best_loss:
            best_loss = loss
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, checkpoint_path)
            print(f'Saved checkpoint to {checkpoint_path}')
        
        # Save final model
        if epoch == args.epochs - 1:
            final_model_path = os.path.join(args.output_dir, 'final_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, final_model_path)
            print(f'Saved final model to {final_model_path}')
        
        # Clear memory between epochs
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == '__main__':
    main() 