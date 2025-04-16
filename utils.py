import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import seaborn as sns
import torch
from torch.utils.data import Dataset

# Constants
SLEEP_STAGES = {
    'W': 0,  # Wake
    'R': 1,  # REM
    '1': 2,  # NREM 1
    '2': 3,  # NREM 2
    '3': 4,  # NREM 3
    '4': 5,  # NREM 4
    'M': 6,  # Movement
    '?': 7   # Unknown
}

STAGE_NAMES = ['Wake', 'REM', 'N1', 'N2', 'N3', 'N4', 'Movement', 'Unknown']

class SleepDataset(Dataset):
    """Custom PyTorch Dataset for Sleep EEG data"""
    
    def __init__(self, X, y=None, transform=None):
        """
        Args:
            X (numpy.ndarray): EEG signals, shape (n_samples, n_channels, n_timesteps)
            y (numpy.ndarray, optional): Labels, shape (n_samples,)
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) if y is not None else None
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        sample = self.X[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        if self.y is not None:
            return sample, self.y[idx]
        else:
            return sample


def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    """
    Plot confusion matrix for sleep stage classification.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        title: Title for the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=STAGE_NAMES, yticklabels=STAGE_NAMES)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(title)
    plt.savefig('confusion_matrix.png')
    plt.close()


def plot_hypnogram(y_true, y_pred, subject_id, title='Hypnogram Comparison'):
    """
    Plot hypnogram comparison between ground truth and prediction.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        subject_id: Subject identifier
        title: Title for the plot
    """
    # Only include meaningful sleep stages (exclude M and ?)
    mask = (y_true < 6)
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    
    epochs = np.arange(len(y_true_filtered))
    
    plt.figure(figsize=(15, 6))
    plt.plot(epochs, 5 - y_true_filtered, 'b-', label='Ground Truth')
    plt.plot(epochs, 5 - y_pred_filtered, 'r-', alpha=0.5, label='Predicted')
    
    plt.yticks([0, 1, 2, 3, 4, 5], ['N4', 'N3', 'N2', 'N1', 'REM', 'Wake'])
    plt.xlabel('Epoch')
    plt.ylabel('Sleep Stage')
    plt.title(f'{title} - Subject {subject_id}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'hypnogram_{subject_id}.png')
    plt.close()


def evaluate_model(y_true, y_pred):
    """
    Calculate and print evaluation metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    
    metrics = {
        'accuracy': acc,
        'f1_score': f1
    }
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1-Score: {f1:.4f}")
    
    return metrics