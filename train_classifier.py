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

def train_classifier(train_loader, val_loader, input_dim=128, hidden_dim=64, num_classes=8,
                    learning_rate=0.001, num_epochs=10, device='cuda'):
    """
    Train the sleep stage classifier.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        input_dim: Input dimension (latent space dimension)
        hidden_dim: Hidden layer dimension
        num_classes: Number of sleep stages
        learning_rate: Learning rate for optimization
        num_epochs: Number of training epochs
        device: Device to train on ('cuda' or 'cpu')
    
    Returns:
        model: Trained classifier model
        history: Training history
    """
    # Initialize model
    model = SleepStageClassifier(input_dim, hidden_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0
    patience = 5  # Early stopping patience
    patience_counter = 0
    
    print(f"\nTraining classifier for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        # Use tqdm for progress bar
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for features, labels in train_pbar:
            features = features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            train_acc = 100 * train_correct / train_total
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{train_acc:.2f}%'})
            
            # Free up memory
            del outputs, loss
            if device == 'cuda':
                torch.cuda.empty_cache()
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for features, labels in val_pbar:
                features = features.to(device)
                labels = labels.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Update progress bar
                val_acc = 100 * val_correct / val_total
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{val_acc:.2f}%'})
                
                # Free up memory
                del outputs, loss
                if device == 'cuda':
                    torch.cuda.empty_cache()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        
        # Early stopping based on validation accuracy
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping triggered after {epoch + 1} epochs')
                break
    
    return model, history

def evaluate_classifier(model, test_loader, device='cuda'):
    """
    Evaluate the classifier on test data.
    
    Args:
        model: Trained classifier model
        test_loader: Test data loader
        device: Device to use for computation
    
    Returns:
        accuracy: Classification accuracy
        y_true: True labels
        y_pred: Predicted labels
    """
    model.eval()
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for features, labels in tqdm(test_loader, desc='Evaluating'):
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            
            # Free up memory
            del outputs, predicted
            if device == 'cuda':
                torch.cuda.empty_cache()
    
    accuracy = correct / total
    
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
    
    return accuracy, np.array(y_true), np.array(y_pred)