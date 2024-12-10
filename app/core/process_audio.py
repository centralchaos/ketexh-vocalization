"""



Audio Preprocessing Module for Chicken Vocalization Analysis

This module handles the conversion of audio files to spectrograms for model input.

Dependencies:
- librosa: Audio processing and feature extraction
- numpy: Numerical operations
- torch: Tensor operations
- pathlib: File path handling

Key Features:
- Audio segmentation into 1-second chunks
- Mel spectrogram generation
- Normalization and formatting for model input
"""

import librosa
import numpy as np
import torch
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class AudioPreprocessor:
    """
    Handles audio file preprocessing for model inference.
    
    Configuration:
        sample_rate: 22050 Hz (standard for audio processing)
        n_mels: 88 (number of mel bands)
        n_fft: 2048 (FFT window size)
        hop_length: 512 (number of samples between frames)
        fmin: 0 Hz (minimum frequency)
        fmax: 11025 Hz (maximum frequency, sr/2)
    """
    
    def __init__(self):
        """
        
        Parameters:
        - sample_rate: 44.1kHz for high-quality audio capture
        - segment_duration: 1 second for consistent analysis windows
        - n_fft: 2048 points for detailed frequency resolution
        - hop_length: 512 (75% overlap) for smooth temporal transitions
        - n_mels: 88 mel bands for frequency analysis
        - fmin/fmax: 2048-4096 Hz range focusing on relevant frequencies
        
        Raises:
            ValueError: If frequency or FFT parameters are invalid
        """
        self.sample_rate = 44100
        self.segment_duration = 1  # 1-second segments
        self.n_fft = 2048         # STFT window length
        self.hop_length = 512     # 75% overlap
        self.n_mels = 88          # Number of mel bands
        self.fmin = 2048          # Min frequency (Hz)
        self.fmax = 4096          # Max frequency (Hz)
        
        # Validate parameters
        if self.fmax <= self.fmin:
            raise ValueError("fmax must be greater than fmin")
        if self.hop_length >= self.n_fft:
            raise ValueError("hop_length must be less than n_fft")
    
    def segment_audio(self, audio_path):
        """
        Load and segment audio file into 1-second chunks.
        
        Args:
            audio_path (str or Path): Path to the audio file
            
        Returns:
            list: List of 1-second audio segments as numpy arrays
                 Empty list if file cannot be loaded
                 
        Process:
            1. Loads audio file at specified sample rate
            2. Divides into non-overlapping 1-second segments
            3. Discards incomplete segments (< 1 second)
        """
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return []
        
        samples_per_segment = self.sample_rate * self.segment_duration
        
        segments = []
        for start in range(0, len(y), samples_per_segment):
            segment = y[start:start + samples_per_segment]
            if len(segment) == samples_per_segment:
                segments.append(segment)
                
        return segments
        
    def create_spectrogram(self, audio_segment):
        """
        Convert audio segment to mel spectrogram.
        
        Args:
            audio_segment (numpy.ndarray): 1-second audio segment
            
        Returns:
            torch.Tensor: Normalized mel spectrogram
                         Shape: (88, 87) - (mel_bands, time_steps)
                         None if processing fails
                         
        Process:
            1. Compute mel spectrogram using STFT with Hann window
            2. Convert to log scale (dB)
            3. Normalize to range [0,1]
            4. Convert to PyTorch tensor
            
        Note:
            Expected output shape is (88, 87) for 1-second segments
            with given parameters
        """
        try:
            mel_spec = librosa.feature.melspectrogram(
                y=audio_segment,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                fmin=self.fmin,
                fmax=self.fmax,
                window='hann'
            )
            
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            normalized_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min())
            
            if normalized_spec.shape != (self.n_mels, 87):
                print(f"Warning: Unexpected spectrogram shape: {normalized_spec.shape}")
            
            return torch.FloatTensor(normalized_spec)
            
        except Exception as e:
            print(f"Error creating spectrogram: {e}")
            return None

    def process_file(self, audio_path):
        """
        Process complete audio file through the preprocessing pipeline.
        
        Args:
            audio_path (str or Path): Path to audio file
            
        Returns:
            torch.Tensor: Stacked spectrograms for all segments
                         Shape: (N, 88, 87) where N is number of segments
                         None if processing fails
                         
        Process:
            1. Segment audio file into 1-second chunks
            2. Create mel spectrogram for each segment
            3. Stack all spectrograms into single tensor
            
        Note:
            - Skips failed segments (returns None for that segment)
            - Returns None if no valid spectrograms are created
        """
        try:
            segments = self.segment_audio(audio_path)
            spectrograms = []
            
            for segment in segments:
                spec = self.create_spectrogram(segment)
                if spec is not None:
                    spectrograms.append(spec)
                
            if not spectrograms:
                return None
                
            return torch.stack(spectrograms)
            
        except Exception as e:
            print(f"Error processing file {audio_path}: {e}")
            return None