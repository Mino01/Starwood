import torch
import torch.nn as nn
import numpy as np

class DDSPCore(nn.Module):
    """
    A simplified implementation of the DDSP Core synthesizer.
    It combines a Harmonic Oscillator and a Noise Generator, both controlled
    by neural network outputs (f0, loudness, harmonic_mix).
    """
    def __init__(self, sample_rate: int = 44100, n_samples: int = 64000):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_samples = n_samples
        self.n_harmonics = 60 # Number of harmonics for the oscillator
        
        # Pre-calculate time steps
        self.register_buffer('time_steps', torch.linspace(0, 1, n_samples, dtype=torch.float32))

    def _safe_f0_to_hz(self, f0):
        """Converts f0 in normalized log-scale to Hz, handling potential zeros."""
        f0_hz = 20.0 * torch.exp(f0)
        return f0_hz

    def _harmonic_oscillator(self, f0_hz, loudness):
        """
        Generates the harmonic component using a bank of oscillators.
        
        Args:
            f0_hz (Tensor): Fundamental frequency in Hz (time-varying).
            loudness (Tensor): Loudness control signal (time-varying).
        
        Returns:
            Tensor: Harmonic audio waveform.
        """
        # 1. Calculate instantaneous phase
        # f0_hz is (batch, time)
        # Cumulative sum of phase increment (2 * pi * f0 / sr)
        phase_increment = 2.0 * np.pi * f0_hz / self.sample_rate
        phase = torch.cumsum(phase_increment, dim=1)
        
        # 2. Generate harmonics
        # Harmonics are multiples of the fundamental frequency
        harmonics = torch.arange(1, self.n_harmonics + 1, device=f0_hz.device).float()
        
        # Expand phase for all harmonics: (batch, time, n_harmonics)
        harmonic_phase = phase.unsqueeze(-1) * harmonics.unsqueeze(0).unsqueeze(0)
        
        # Simple amplitude decay for higher harmonics (can be learned)
        harmonic_amps = 1.0 / harmonics.unsqueeze(0).unsqueeze(0)
        
        # 3. Sum sinusoids
        # Sum over the harmonic dimension: (batch, time)
        harmonic_waveform = (torch.sin(harmonic_phase) * harmonic_amps).sum(dim=-1)
        
        # 4. Apply loudness envelope
        # Normalize and apply loudness (loudness is a placeholder for amplitude envelope)
        harmonic_waveform = harmonic_waveform * loudness
        
        return harmonic_waveform

    def _noise_generator(self, loudness):
        """
        Generates the noise component (filtered white noise).
        
        Args:
            loudness (Tensor): Loudness control signal (time-varying).
        
        Returns:
            Tensor: Noise audio waveform.
        """
        # 1. Generate white noise
        noise = torch.randn(loudness.shape, device=loudness.device)
        
        # 2. Apply loudness envelope (simple amplitude modulation for now)
        noise_waveform = noise * loudness
        
        # In a full DDSP, this would be filtered with a time-varying filter bank
        
        return noise_waveform

    def forward(self, ddsp_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Synthesizes audio from DDSP control parameters.
        
        Args:
            ddsp_params (Dict[str, torch.Tensor]): Dictionary containing control signals.
                Expected keys: 'f0', 'loudness', 'harmonic_mix'.
                All tensors should be of shape (batch_size, n_samples).
        
        Returns:
            torch.Tensor: The synthesized audio waveform of shape (batch_size, n_samples).
        """
        f0 = ddsp_params['f0']
        loudness = ddsp_params['loudness']
        harmonic_mix = ddsp_params['harmonic_mix'] # Between 0 and 1
        
        # 1. Convert f0 from normalized log-scale to Hz
        f0_hz = self._safe_f0_to_hz(f0)
        
        # 2. Generate harmonic component
        harmonic_audio = self._harmonic_oscillator(f0_hz, loudness)
        
        # 3. Generate noise component
        noise_audio = self._noise_generator(loudness)
        
        # 4. Mix components
        # Mix = harmonic_mix * harmonic + (1 - harmonic_mix) * noise
        audio_out = (harmonic_mix * harmonic_audio) + ((1.0 - harmonic_mix) * noise_audio)
        
        # 5. Normalize (optional, but good practice)
        audio_out = audio_out / (audio_out.abs().max(dim=1, keepdim=True).values + 1e-6)
        
        return audio_out

# Example of expected input shape (batch_size=1, n_samples=44100 for 1 second)
if __name__ == '__main__':
    # Create a dummy DDSPCore instance
    sr = 44100
    n_s = sr * 1 # 1 second of audio
    ddsp_synth = DDSPCore(sample_rate=sr, n_samples=n_s)
    
    # Create dummy control parameters (simulating neural network output)
    batch_size = 1
    
    # f0: A constant note (e.g., A4 = 440Hz) in normalized log-scale
    # log(440 / 20) = log(22) ~ 3.09
    f0_log_scale = torch.full((batch_size, n_s), 3.09)
    
    # Loudness: A simple fade-in envelope
    loudness_envelope = torch.linspace(0.0, 1.0, n_s).unsqueeze(0)
    
    # Harmonic Mix: Mostly harmonic sound
    harmonic_mix = torch.full((batch_size, n_s), 0.9)
    
    dummy_params = {
        'f0': f0_log_scale,
        'loudness': loudness_envelope,
        'harmonic_mix': harmonic_mix
    }
    
    # Synthesize
    audio_waveform = ddsp_synth(dummy_params)
    
    print(f"Synthesized audio shape: {audio_waveform.shape}")
    print(f"Max amplitude: {audio_waveform.abs().max()}")
    
    # To save the audio for listening:
    # from scipy.io.wavfile import write as write_wav
    # write_wav('ddsp_test.wav', sr, audio_waveform.squeeze().numpy())
