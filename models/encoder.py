import torch
import torch.nn as nn
import torch.nn.functional as F

class VIMEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        """
        VIME-based self-supervised tabular encoder.
        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of hidden units in the encoder.
            latent_dim (int): Dimensionality of the latent space.
        """
        super(VIMEEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        # Encode input to latent space
        z = self.encoder(x)
        # Decode latent representation back to input space
        x_reconstructed = self.decoder(z)
        return z, x_reconstructed


class VIMEMaskGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        Mask generator for VIME.
        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of hidden units in the mask generator.
        """
        super(VIMEMaskGenerator, self).__init__()
        self.mask_generator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Generate mask probabilities
        mask = self.mask_generator(x)
        return mask


class VIMESelfSupervisedModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        """
        VIME self-supervised model combining encoder and mask generator.
        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of hidden units.
            latent_dim (int): Dimensionality of the latent space.
        """
        super(VIMESelfSupervisedModel, self).__init__()
        self.encoder = VIMEEncoder(input_dim, hidden_dim, latent_dim)
        self.mask_generator = VIMEMaskGenerator(input_dim, hidden_dim)

    def forward(self, x):
        # Generate mask
        mask = self.mask_generator(x)
        # Apply mask to input
        x_masked = x * (1 - mask)
        # Encode and reconstruct
        z, x_reconstructed = self.encoder(x_masked)
        return mask, x_masked, z, x_reconstructed


def vime_loss(x, x_reconstructed, mask, mask_pred, alpha=1.0, beta=1.0):
    """
    Compute the VIME loss.
    Args:
        x (torch.Tensor): Original input.
        x_reconstructed (torch.Tensor): Reconstructed input.
        mask (torch.Tensor): Ground truth mask.
        mask_pred (torch.Tensor): Predicted mask.
        alpha (float): Weight for reconstruction loss.
        beta (float): Weight for mask prediction loss.
    Returns:
        torch.Tensor: Total loss.
    """
    # Reconstruction loss (e.g., MSE)
    reconstruction_loss = F.mse_loss(x_reconstructed, x)
    # Mask prediction loss (e.g., Binary Cross-Entropy)
    mask_loss = F.binary_cross_entropy(mask_pred, mask)
    # Total loss
    total_loss = alpha * reconstruction_loss + beta * mask_loss
    return total_loss