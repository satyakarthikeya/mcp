import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from models import Autoencoder

def train_autoencoder(train_loader, val_loader, input_size=3000, latent_dim=128,
                     learning_rate=0.001, num_epochs=10, device='cuda'):
    """
    Train the autoencoder model.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        input_size: Size of input signal
        latent_dim: Size of latent space
        learning_rate: Learning rate for optimization
        num_epochs: Number of training epochs
        device: Device to train on ('cuda' or 'cpu')
    
    Returns:
        model: Trained autoencoder model
        history: Training history
    """
    # Initialize model
    model = Autoencoder(input_size, latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize history
    history = {
        'train_loss': [],
        'val_loss': [],
    }
    
    best_val_loss = float('inf')
    patience = 5  # Early stopping patience
    patience_counter = 0
    
    print(f"\nTraining autoencoder for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_batches = 0
        
        # Use tqdm for progress bar
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for data in train_pbar:
            inputs = data[0].to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            train_batches += 1
            
            # Update progress bar
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Free up memory
            del outputs, loss
            if device == 'cuda':
                torch.cuda.empty_cache()
        
        avg_train_loss = train_loss / train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for data in val_pbar:
                inputs = data[0].to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                
                val_loss += loss.item()
                val_batches += 1
                
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                # Free up memory
                del outputs, loss
                if device == 'cuda':
                    torch.cuda.empty_cache()
        
        avg_val_loss = val_loss / val_batches
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}')
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping triggered after {epoch + 1} epochs')
                break
    
    return model, history

def extract_latent_features(data_loader, encoder, device='cuda'):
    """
    Extract latent features using the trained encoder.
    
    Args:
        data_loader: Data loader
        encoder: Trained encoder model
        device: Device to use for computation
    
    Returns:
        features: Extracted features
        labels: Corresponding labels
    """
    features = []
    labels = []
    
    encoder.eval()
    with torch.no_grad():
        for data in tqdm(data_loader, desc='Extracting features'):
            inputs = data[0].to(device)
            batch_labels = data[1]
            
            # Extract features
            batch_features = encoder(inputs)
            
            # Move to CPU and convert to numpy
            features.append(batch_features.cpu().numpy())
            labels.append(batch_labels.numpy())
            
            # Free up memory
            del batch_features
            if device == 'cuda':
                torch.cuda.empty_cache()
    
    return np.vstack(features), np.concatenate(labels)