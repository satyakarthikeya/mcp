import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns

from models import SleepStageClassifier
from utils import STAGE_NAMES

def train_classifier(train_loader, val_loader, input_dim=128, hidden_dim=64, 
                     num_classes=8, learning_rate=0.001, num_epochs=10, device='cuda'):
    """
    Train the sleep stage classifier model.
    
    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        input_dim: Dimension of input features
        hidden_dim: Dimension of hidden layer
        num_classes: Number of sleep stage classes
        learning_rate: Learning rate for Adam optimizer
        num_epochs: Number of training epochs
        device: Device to train on ('cuda' or 'cpu')
        
    Returns:
        model: Trained classifier model
        history: Training history (loss and accuracy values)
    """
    # Check if CUDA is available
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available. Using CPU instead.")
        device = 'cpu'
    
    # Initialize model, loss function, and optimizer
    model = SleepStageClassifier(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # For tracking the training progress
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        # Training phase
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for features, labels in progress_bar:
            features = features.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(features)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for features, labels in progress_bar:
                features = features.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(features)
                
                # Calculate loss
                loss = criterion(outputs, labels)
                
                # Update statistics
                val_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()
                
                progress_bar.set_postfix({'loss': loss.item()})
        
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.4f}')
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/classifier.pt')
    
    # Plot the loss curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Classifier Training Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Classifier Training Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('classifier_training.png')
    plt.close()
    
    return model, history


def evaluate_classifier(model, test_loader, device='cuda'):
    """
    Evaluate the classifier model on test data.
    
    Args:
        model: Trained classifier model
        test_loader: DataLoader for test data
        device: Device to use for evaluation
        
    Returns:
        accuracy: Classification accuracy
        y_true: True labels
        y_pred: Predicted labels
    """
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available. Using CPU instead.")
        device = 'cpu'
    
    model.eval()
    model = model.to(device)
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for features, labels in tqdm(test_loader, desc='Evaluating'):
            features = features.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate accuracy
    accuracy = np.mean(y_true == y_pred)
    
    # Print accuracy
    print(f'Test Accuracy: {accuracy:.4f}')
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=STAGE_NAMES, yticklabels=STAGE_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('classifier_confusion_matrix.png')
    plt.close()
    
    return accuracy, y_true, y_pred