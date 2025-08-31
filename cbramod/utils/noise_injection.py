"""
Noise Injection Module for EEG Robustness Analysis

This module implements realistic noise injection for ear-EEG signals to evaluate
model robustness against real-world artifacts commonly found in wearable devices.

Implements various types of realistic artifacts:
- EMG artifacts (muscle activity from jaw/neck)
- Movement artifacts (head motion, electrode displacement)
- Electrode impedance issues (poor contact, high impedance)
- Environmental noise (power line, ambient electrical)
- Realistic mixed artifacts (combination of above)

  # EMG artifacts only
  python cbramod/training/finetuning/finetune_main.py --downstream_dataset IDUN_EEG --datasets_dir
  data/datasets/final_dataset --datasets ORP --use_pretrained_weights True --model_dir "./saved_models" --epochs 20
  --batch_size 32 --run_name "emg_10pct" --noise_level 0.10 --noise_type emg --seed 3407 --noise_seed 42

  # Movement artifacts only
  python cbramod/training/finetuning/finetune_main.py --downstream_dataset IDUN_EEG --datasets_dir
  data/datasets/final_dataset --datasets ORP --use_pretrained_weights True --model_dir "./saved_models" --epochs 20
  --batch_size 32 --run_name "movement_10pct" --noise_level 0.10 --noise_type movement --seed 3407 --noise_seed 42

  # Electrode artifacts only
  python cbramod/training/finetuning/finetune_main.py --downstream_dataset IDUN_EEG --datasets_dir
  data/datasets/final_dataset --datasets ORP --use_pretrained_weights True --model_dir "./saved_models" --epochs 20
  --batch_size 32 --run_name "electrode_10pct" --noise_level 0.10 --noise_type electrode --seed 3407 --noise_seed 42

  Key parameters:
  - --noise_level: 0.0=clean, 0.05=5%, 0.10=10%, 0.20=20%
  - --noise_type: realistic, emg, movement, electrode, gaussian
  - --noise_seed 42: Ensures reproducible noise patterns
  - --seed 3407: Ensures reproducible training
  
Author: CBraMod Research Team
Date: 2024
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal
from typing import Union, Tuple, Optional
import warnings

class EEGNoiseInjector:
    """
    Comprehensive EEG noise injection for robustness evaluation.
    
    Designed specifically for ear-EEG artifacts commonly encountered in 
    wearable sleep monitoring devices.
    """
    
    def __init__(self, sample_rate: float = 200.0, seed: Optional[int] = None):
        """
        Initialize noise injector.
        
        Args:
            sample_rate: EEG sampling rate in Hz
            seed: Random seed for reproducible noise generation
        """
        self.sample_rate = sample_rate
        self.rng = np.random.RandomState(seed) if seed is not None else np.random
        
        # Precompute filter coefficients for efficiency
        self._setup_filters()
    
    def _setup_filters(self):
        """Setup bandpass filters for different artifact types."""
        nyquist = self.sample_rate / 2
        
        # EMG filter (20-150 Hz) - muscle artifacts
        self.emg_sos = signal.butter(4, [20/nyquist, min(150/nyquist, 0.95)], 
                                    btype='band', output='sos')
        
        # Movement filter (0.5-10 Hz) - low frequency drift
        self.movement_sos = signal.butter(2, [0.5/nyquist, 10/nyquist], 
                                         btype='band', output='sos')
        
        # High frequency noise (50-100 Hz) - electrical interference
        self.electrical_sos = signal.butter(2, [50/nyquist, min(100/nyquist, 0.95)], 
                                           btype='band', output='sos')

    def inject_gaussian_noise(self, eeg_data: np.ndarray, noise_level: float) -> np.ndarray:
        """
        Inject simple Gaussian noise.
        
        Args:
            eeg_data: EEG signal [channels, time] or [time]
            noise_level: Noise level as fraction of signal power
            
        Returns:
            Noisy EEG signal
        """
        signal_power = np.var(eeg_data)
        noise_power = signal_power * noise_level
        noise = self.rng.normal(0, np.sqrt(noise_power), eeg_data.shape)
        
        return eeg_data + noise
    
    def inject_emg_noise(self, eeg_data: np.ndarray, noise_level: float) -> np.ndarray:
        """
        Inject realistic EMG (electromyographic) artifacts.
        
        EMG artifacts are high-frequency (20-150 Hz) bursts caused by muscle activity,
        particularly common in ear-EEG due to jaw/neck muscle contractions.
        
        Args:
            eeg_data: EEG signal [channels, time] or [time]
            noise_level: Noise level as fraction of signal power
            
        Returns:
            EEG signal with EMG artifacts
        """
        # Generate high-frequency EMG-like noise
        raw_noise = self.rng.normal(0, 1, eeg_data.shape)
        
        # Filter to EMG frequency range (20-150 Hz)
        if len(eeg_data.shape) == 1:
            emg_noise = signal.sosfilt(self.emg_sos, raw_noise)
        else:
            emg_noise = np.zeros_like(raw_noise)
            for ch in range(eeg_data.shape[0]):
                emg_noise[ch] = signal.sosfilt(self.emg_sos, raw_noise[ch])
        
        # Create burst patterns (EMG is typically bursty)
        burst_prob = 0.3  # 30% of time has EMG bursts
        burst_mask = self.rng.random(eeg_data.shape) < burst_prob
        emg_noise *= burst_mask
        
        # Scale noise relative to signal power
        signal_power = np.var(eeg_data)
        noise_power = signal_power * noise_level
        emg_noise = emg_noise * np.sqrt(noise_power / np.var(emg_noise))
        
        return eeg_data + emg_noise
    
    def inject_movement_artifacts(self, eeg_data: np.ndarray, noise_level: float) -> np.ndarray:
        """
        Inject movement artifacts (low-frequency drift and sudden jumps).
        
        Movement artifacts include:
        - Slow baseline drift from head movements
        - Sudden amplitude changes from electrode displacement
        - Low-frequency oscillations from breathing/heartbeat
        
        Args:
            eeg_data: EEG signal [channels, time] or [time]
            noise_level: Noise level as fraction of signal power
            
        Returns:
            EEG signal with movement artifacts
        """
        # 1. Slow baseline drift (0.5-2 Hz)
        drift_noise = self.rng.normal(0, 1, eeg_data.shape)
        if len(eeg_data.shape) == 1:
            drift = signal.sosfilt(self.movement_sos, drift_noise)
        else:
            drift = np.zeros_like(drift_noise)
            for ch_idx in range(eeg_data.shape[0]):
                drift[ch_idx] = signal.sosfilt(self.movement_sos, drift_noise[ch_idx])
        
        # 2. Sudden jumps (electrode displacement)
        jump_prob = 0.05  # 5% chance of sudden jumps
        if len(eeg_data.shape) == 1:
            jump_times = self.rng.random(eeg_data.shape[0]) < jump_prob
            jump_magnitudes = self.rng.normal(0, np.std(eeg_data) * 2, eeg_data.shape[0])
            jumps = np.cumsum(jump_times * jump_magnitudes)
        else:
            jumps = np.zeros_like(eeg_data)
            # Apply jumps across the time dimension (last axis)
            time_samples = eeg_data.shape[-1]
            for channel_idx in range(eeg_data.shape[0]):
                jump_times = self.rng.random(time_samples) < jump_prob
                jump_magnitudes = self.rng.normal(0, np.std(eeg_data[channel_idx]) * 2, time_samples)
                jump_series = np.cumsum(jump_times * jump_magnitudes)
                jumps[channel_idx, :] = jump_series
        
        # Combine drift and jumps
        movement_noise = drift + jumps
        
        # Scale relative to signal power
        signal_power = np.var(eeg_data)
        noise_power = signal_power * noise_level
        movement_noise = movement_noise * np.sqrt(noise_power / np.var(movement_noise))
        
        return eeg_data + movement_noise
    
    def inject_electrode_artifacts(self, eeg_data: np.ndarray, noise_level: float) -> np.ndarray:
        """
        Inject electrode impedance artifacts.
        
        Poor electrode contact causes:
        - Increased high-frequency noise
        - Intermittent signal dropout
        - Impedance fluctuations
        
        Args:
            eeg_data: EEG signal [channels, time] or [time]
            noise_level: Noise level as fraction of signal power
            
        Returns:
            EEG signal with electrode artifacts
        """
        # 1. High-frequency impedance noise
        hf_noise = self.rng.normal(0, 1, eeg_data.shape)
        if len(eeg_data.shape) == 1:
            impedance_noise = signal.sosfilt(self.electrical_sos, hf_noise)
        else:
            impedance_noise = np.zeros_like(hf_noise)
            for ch in range(eeg_data.shape[0]):
                impedance_noise[ch] = signal.sosfilt(self.electrical_sos, hf_noise[ch])
        
        # 2. Intermittent dropout (poor contact)
        dropout_prob = 0.1  # 10% chance of dropout periods
        if len(eeg_data.shape) == 1:
            dropout_mask = self.rng.random(eeg_data.shape[0]) > dropout_prob
        else:
            dropout_mask = self.rng.random(eeg_data.shape) > dropout_prob
        
        # Apply dropout by zeroing signal periods
        corrupted_signal = eeg_data * dropout_mask
        
        # Scale impedance noise
        signal_power = np.var(eeg_data)
        noise_power = signal_power * noise_level
        impedance_noise = impedance_noise * np.sqrt(noise_power / np.var(impedance_noise))
        
        return corrupted_signal + impedance_noise
    
    def inject_realistic_mixed_noise(self, eeg_data: np.ndarray, noise_level: float) -> np.ndarray:
        """
        Inject realistic mixed artifacts combining multiple noise types.
        
        This simulates real-world ear-EEG conditions with multiple simultaneous
        artifact sources at different intensities.
        
        Args:
            eeg_data: EEG signal [channels, time] or [time]
            noise_level: Overall noise level as fraction of signal power
            
        Returns:
            EEG signal with mixed realistic artifacts
        """
        # Distribute noise budget across artifact types
        # Based on typical ear-EEG artifact prevalence
        noise_distribution = {
            'emg': 0.4,        # 40% - most common in ear-EEG
            'movement': 0.3,   # 30% - frequent due to head motion
            'electrode': 0.2,  # 20% - contact issues
            'gaussian': 0.1    # 10% - background electrical noise
        }
        
        noisy_signal = eeg_data.copy()
        
        # Apply each noise type with proportional intensity
        for noise_type, proportion in noise_distribution.items():
            type_noise_level = noise_level * proportion
            
            if noise_type == 'emg':
                noisy_signal = self.inject_emg_noise(noisy_signal, type_noise_level)
            elif noise_type == 'movement':
                noisy_signal = self.inject_movement_artifacts(noisy_signal, type_noise_level)
            elif noise_type == 'electrode':
                noisy_signal = self.inject_electrode_artifacts(noisy_signal, type_noise_level)
            elif noise_type == 'gaussian':
                noisy_signal = self.inject_gaussian_noise(noisy_signal, type_noise_level)
        
        return noisy_signal
    
    def inject_noise(self, eeg_data: np.ndarray, noise_type: str, noise_level: float) -> np.ndarray:
        """
        Main noise injection interface.
        
        Args:
            eeg_data: EEG signal [channels, time] or [time]
            noise_type: Type of noise ('gaussian', 'emg', 'movement', 'electrode', 'realistic')
            noise_level: Noise level as fraction of signal power (0.0-1.0)
            
        Returns:
            Noisy EEG signal
        """
        if noise_level <= 0.0:
            return eeg_data
        
        # Ensure input is numpy array
        if isinstance(eeg_data, torch.Tensor):
            was_tensor = True
            device = eeg_data.device
            dtype = eeg_data.dtype
            eeg_data = eeg_data.cpu().numpy()
        else:
            was_tensor = False
        
        # Apply noise injection
        if noise_type == 'gaussian':
            noisy_data = self.inject_gaussian_noise(eeg_data, noise_level)
        elif noise_type == 'emg':
            noisy_data = self.inject_emg_noise(eeg_data, noise_level)
        elif noise_type == 'movement':
            noisy_data = self.inject_movement_artifacts(eeg_data, noise_level)
        elif noise_type == 'electrode':
            noisy_data = self.inject_electrode_artifacts(eeg_data, noise_level)
        elif noise_type == 'realistic':
            noisy_data = self.inject_realistic_mixed_noise(eeg_data, noise_level)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        # Convert back to tensor if input was tensor
        if was_tensor:
            noisy_data = torch.tensor(noisy_data, device=device, dtype=dtype)
        
        return noisy_data


def create_noise_injector(sample_rate: float = 200.0, seed: Optional[int] = None) -> EEGNoiseInjector:
    """
    Factory function to create noise injector.
    
    Args:
        sample_rate: EEG sampling rate in Hz
        seed: Random seed for reproducible experiments
        
    Returns:
        EEGNoiseInjector instance
    """
    return EEGNoiseInjector(sample_rate=sample_rate, seed=seed)


# Example usage and testing
if __name__ == "__main__":
    # Test the noise injection with synthetic EEG data
    import matplotlib.pyplot as plt
    
    # Create synthetic EEG signal (2 channels, 10 seconds at 200 Hz)
    t = np.linspace(0, 10, 2000)
    eeg_clean = np.array([
        np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t),  # Channel 1
        np.cos(2 * np.pi * 8 * t) + 0.3 * np.cos(2 * np.pi * 15 * t)    # Channel 2
    ])
    
    # Initialize noise injector
    noise_injector = create_noise_injector(sample_rate=200.0, seed=42)
    
    # Test different noise types at 10% level
    noise_types = ['gaussian', 'emg', 'movement', 'electrode', 'realistic']
    
    print("Testing EEG Noise Injection...")
    for noise_type in noise_types:
        noisy_eeg = noise_injector.inject_noise(eeg_clean, noise_type, 0.1)
        snr_db = 10 * np.log10(np.var(eeg_clean) / np.var(noisy_eeg - eeg_clean))
        print(f"{noise_type:>10} noise: SNR = {snr_db:.1f} dB")
    
    print("Noise injection module ready for integration!")