import torch
import torch.nn as nn
from typing import Dict

class DDSPRefinementNet(nn.Module):
    """
    The DDSP Refinement Network (formerly DDSPControlNet).
    
    This network takes the continuous RAVE latent vector (z_gen) and translates
    it into DDSP control parameters. This allows for fine-grained, interpretable
    control and refinement of the RAVE output's timbre.
    """
    def __init__(self, 
                 latent_dim: int = 512, 
                 control_dim: int = 3, # f0, loudness, harmonic_mix
                 n_samples: int = 44100):
        super().__init__()
        self.n_samples = n_samples
        self.latent_dim = latent_dim
        self.control_dim = control_dim
        
        # Simple MLP to map RAVE latent vector to control signals
        # The RAVE latent vector is a compressed representation of the entire piece.
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, control_dim * n_samples) # Output is flattened control signals
        )
        
        # Output activation layers to ensure control signals are in the correct range
        self.f0_activation = nn.Identity() # f0 is in log-scale, no strict bounds
        self.loudness_activation = nn.Sigmoid() # Loudness between 0 and 1
        self.harmonic_mix_activation = nn.Sigmoid() # Mix between 0 and 1

    def forward(self, latent_vector: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Maps a RAVE latent vector to DDSP control parameters.
        
        Args:
            latent_vector (torch.Tensor): A tensor of shape (batch_size, latent_dim).
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary of control signals, each of shape
                                     (batch_size, n_samples).
        """
        batch_size = latent_vector.shape[0]
        
        # 1. Pass through MLP
        # Output shape: (batch_size, control_dim * n_samples)
        raw_output = self.mlp(latent_vector)
        
        # 2. Reshape and split into individual control signals
        # Reshape to (batch_size, control_dim, n_samples)
        reshaped_output = raw_output.view(batch_size, self.control_dim, self.n_samples)
        
        # Split: f0, loudness, harmonic_mix
        f0_raw = reshaped_output[:, 0, :]
        loudness_raw = reshaped_output[:, 1, :]
        harmonic_mix_raw = reshaped_output[:, 2, :]
        
        # 3. Apply activations
        f0 = self.f0_activation(f0_raw)
        loudness = self.loudness_activation(loudness_raw)
        harmonic_mix = self.harmonic_mix_activation(harmonic_mix_raw)
        
        return {
            'f0': f0,
            'loudness': loudness,
            'harmonic_mix': harmonic_mix
        }

# Example usage
if __name__ == '__main__':
    # Simulate 1 second of audio at 44100 Hz
    sr = 44100
    n_s = sr * 1
    
    # Simulate a batch of 4 RAVE latent vectors
    batch_size = 4
    latent_dim = 512
    latent_input = torch.randn(batch_size, latent_dim)
    
    refinement_net = DDSPRefinementNet(latent_dim=latent_dim, n_samples=n_s)
    
    ddsp_params = refinement_net(latent_input)
    
    print(f"Generated DDSP Parameters:")
    for key, tensor in ddsp_params.items():
        print(f"- {key}: shape {tensor.shape}")
        
    # Expected output shape for each: (4, 44100)
