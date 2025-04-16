import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc

from data_preprocessing import load_dataset, create_data_loaders
from models import Autoencoder, SleepStageClassifier, SleepClassificationSystem, Encoder
from train_autoencoder import train_autoencoder, extract_latent_features
from train_classifier import train_classifier, evaluate_classifier
from utils import plot_confusion_matrix, plot_hypnogram, evaluate_model, SleepDataset

def main():
    """Main function to orchestrate the sleep stage classification pipeline."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Enable CUDA memory optimization
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
        torch.backends.cuda.matmul.allow_tf32 = True  # Enable TensorFloat-32
        torch.backends.cudnn.allow_tf32 = True
        
        # Set memory management
        torch.cuda.empty_cache()
    
    # Check for GPU and set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Parameters - reduced epochs and optimized for memory
    base_dir = '.'
    batch_size = 32  # Reduced batch size
    input_size = 3000  # 30 seconds at 100 Hz
    latent_dim = 128
    learning_rate = 0.001
    num_epochs_autoencoder = 10  # Reduced from 40010 to 10
    num_epochs_classifier = 10    # Also set to 10 for initial testing
    
    # Create output directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Step 1: Load and preprocess the dataset
    print("Loading and preprocessing the dataset...")
    X, y, subject_ids = load_dataset(None, base_dir, channel='EEG Fpz-Cz')  # Updated channel name
    
    if X is None or y is None:
        print("Error loading dataset. Exiting.")
        return
    
    # Clear memory after dataset loading
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    # Step 2: Create data loaders with memory optimization
    print("Creating data loaders...")
    train_loader, val_loader, test_loader, test_data = create_data_loaders(
        X, y, subject_ids, 
        batch_size=batch_size,
        num_workers=2,  # Reduced workers
        pin_memory=True  # Enable faster data transfer to GPU
    )
    
    # Clear memory after data loader creation
    del X, y
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    # Step 3: Train the autoencoder
    print(f"Training the autoencoder for {num_epochs_autoencoder} epochs...")
    autoencoder, autoencoder_history = train_autoencoder(
        train_loader, val_loader,
        input_size=input_size,
        latent_dim=latent_dim,
        learning_rate=learning_rate,
        num_epochs=num_epochs_autoencoder,
        device=device
    )
    
    # Step 4: Extract latent features using the trained encoder
    print("Extracting latent features...")
    encoder = autoencoder.encoder
    
    # Extract features in batches to manage memory
    train_features, train_labels = extract_latent_features(train_loader, encoder, device=device)
    val_features, val_labels = extract_latent_features(val_loader, encoder, device=device)
    test_features, test_labels = extract_latent_features(test_loader, encoder, device=device)
    
    # Clear encoder from memory as it's no longer needed
    del autoencoder
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    # Create new data loaders with the extracted features
    train_feature_dataset = TensorDataset(torch.FloatTensor(train_features), torch.LongTensor(train_labels))
    val_feature_dataset = TensorDataset(torch.FloatTensor(val_features), torch.LongTensor(val_labels))
    test_feature_dataset = TensorDataset(torch.FloatTensor(test_features), torch.LongTensor(test_labels))
    
    train_feature_loader = DataLoader(train_feature_dataset, batch_size=batch_size, shuffle=True)
    val_feature_loader = DataLoader(val_feature_dataset, batch_size=batch_size)
    test_feature_loader = DataLoader(test_feature_dataset, batch_size=batch_size)
    
    # Clear feature arrays from memory
    del train_features, val_features, test_features
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    # Step 5: Train the classifier
    print(f"Training the classifier for {num_epochs_classifier} epochs...")
    classifier, classifier_history = train_classifier(
        train_feature_loader, val_feature_loader,
        input_dim=latent_dim,
        hidden_dim=64,
        num_classes=8,
        learning_rate=learning_rate,
        num_epochs=num_epochs_classifier,
        device=device
    )
    
    # Step 6: Evaluate the classifier
    print("Evaluating the classifier...")
    accuracy, y_true, y_pred = evaluate_classifier(classifier, test_feature_loader, device=device)
    
    # Step 7: Plot additional results
    # Get test data for visualization
    X_test, y_test, test_subjects = test_data
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, title=f'Sleep Stage Classification (Accuracy: {accuracy:.4f})')
    
    # Calculate additional metrics
    metrics = evaluate_model(y_true, y_pred)
    
    # Plot hypnograms for a few subjects
    unique_subjects = np.unique(test_subjects)
    for subject in unique_subjects[:3]:  # Plot for the first 3 subjects
        # Get data for this subject
        subject_mask = test_subjects == subject
        subject_y_true = y_test[subject_mask]
        subject_indices = np.where(subject_mask)[0]
        
        # Get predictions for this subject (same order as in y_test)
        subject_y_pred = y_pred[np.isin(np.arange(len(y_pred)), subject_indices)]
        
        # Plot hypnogram
        plot_hypnogram(subject_y_true, subject_y_pred, subject)
    
    # Step 8: Save the complete system
    complete_system = SleepClassificationSystem(encoder, classifier)
    torch.save(complete_system.state_dict(), 'models/complete_system.pt')
    
    print("Pipeline completed successfully!")
    print(f"Models saved in the 'models' directory")
    print(f"Plots saved in the current directory")
    
    return complete_system, (encoder, classifier), (y_true, y_pred)

if __name__ == "__main__":
    main()