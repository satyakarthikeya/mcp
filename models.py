import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    1D CNN-based encoder for EEG signals.
    Reduces the dimensionality of EEG data to a latent representation.
    """
    
    def __init__(self, input_size=3000, latent_dim=128):
        super(Encoder, self).__init__()
        
        # First convolutional layer
        # Input: [batch_size, 1, 3000]
        # Output: [batch_size, 32, 1476]
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=32,
            kernel_size=50,
            stride=2,
            padding=0
        )
        
        # Calculate the output size of the first conv layer
        conv1_output_size = (input_size - 50) // 2 + 1
        
        # Second convolutional layer
        # Input: [batch_size, 32, 1476]
        # Output: [batch_size, 64, 734]
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=10,
            stride=2,
            padding=0
        )
        
        # Calculate the output size of the second conv layer
        conv2_output_size = (conv1_output_size - 10) // 2 + 1
        
        # Calculate the flattened size
        self.flat_size = 64 * conv2_output_size
        
        # Linear layer to latent space
        self.fc = nn.Linear(self.flat_size, latent_dim)
        
    def forward(self, x):
        # x shape: [batch_size, 1, 3000]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)  # Map to latent space
        return x


class Decoder(nn.Module):
    """
    1D CNN-based decoder for EEG signals.
    Reconstructs EEG data from latent representation.
    """
    
    def __init__(self, output_size=3000, latent_dim=128):
        super(Decoder, self).__init__()
        
        # Calculate the expected sizes at each stage of decoding
        # Working backwards from the output size
        conv2_output_size = (output_size - 50) // 2 + 1
        conv1_output_size = (conv2_output_size - 10) // 2 + 1
        
        # The flattened size at the input of the decoding process
        self.flat_size = 64 * conv1_output_size
        
        # Linear layer from latent space to flattened convolutional features
        self.fc = nn.Linear(latent_dim, self.flat_size)
        
        # First transposed convolutional layer
        # Input: [batch_size, 64, conv1_output_size]
        # Output: [batch_size, 32, conv2_output_size]
        self.conv_transpose1 = nn.ConvTranspose1d(
            in_channels=64,
            out_channels=32,
            kernel_size=10,
            stride=2,
            padding=0
        )
        
        # Second transposed convolutional layer
        # Input: [batch_size, 32, conv2_output_size]
        # Output: [batch_size, 1, output_size]
        self.conv_transpose2 = nn.ConvTranspose1d(
            in_channels=32,
            out_channels=1,
            kernel_size=50,
            stride=2,
            padding=0
        )
        
        # Store expected shapes for reshaping
        self.conv1_output_size = conv1_output_size
        
    def forward(self, x):
        # x shape: [batch_size, latent_dim]
        x = self.fc(x)
        x = x.view(x.size(0), 64, self.conv1_output_size)  # Reshape
        x = F.relu(self.conv_transpose1(x))
        x = torch.sigmoid(self.conv_transpose2(x))  # Use sigmoid for output in range [0, 1]
        return x


class Autoencoder(nn.Module):
    """
    Complete autoencoder model combining encoder and decoder.
    """
    
    def __init__(self, input_size=3000, latent_dim=128):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_size, latent_dim)
        self.decoder = Decoder(input_size, latent_dim)
        
    def forward(self, x):
        # Encode input to latent representation
        latent = self.encoder(x)
        # Decode latent representation back to input space
        reconstruction = self.decoder(latent)
        return reconstruction, latent


class SleepStageClassifier(nn.Module):
    """
    MLP classifier for sleep stage classification.
    Takes latent representations from the encoder as input.
    """
    
    def __init__(self, input_dim=128, hidden_dim=64, num_classes=8, dropout_rate=0.5):
        super(SleepStageClassifier, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x shape: [batch_size, input_dim]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SleepClassificationSystem(nn.Module):
    """
    Complete system combining encoder and classifier.
    Used for inference only.
    """
    
    def __init__(self, encoder, classifier):
        super(SleepClassificationSystem, self).__init__()
        self.encoder = encoder
        self.classifier = classifier
        
    def forward(self, x):
        # Extract features using the encoder
        with torch.no_grad():
            features = self.encoder(x)
        
        # Classify the features
        logits = self.classifier(features)
        return logits