import os
import numpy as np
import mne
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader
from utils import SleepDataset, SLEEP_STAGES
import glob
import warnings

# Suppress specific warnings from MNE
warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")

def load_edf_file(file_path, channel='EEG Fpz-Cz'):  # Updated default channel name
    """
    Load EEG data from an EDF file.
    
    Args:
        file_path: Path to the EDF file
        channel: EEG channel to extract (default is 'EEG Fpz-Cz')
    
    Returns:
        raw_data: MNE Raw object containing the EEG data
    """
    try:
        # Load the EDF file with verbose=False to reduce output
        raw_data = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        
        # Check if the specified channel exists
        if channel not in raw_data.ch_names:
            available_channels = raw_data.ch_names
            print(f"Channel {channel} not found. Available channels: {available_channels}")
            
            # Try to find a similar channel
            for ch in available_channels:
                if channel.replace('EEG ', '').lower() in ch.lower():
                    channel = ch
                    print(f"Using alternative channel: {channel}")
                    break
            else:
                return None
        
        # Pick the channel and return
        raw_data.pick_channels([channel])
        return raw_data
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_hypnogram(file_path):
    """
    Load sleep stage annotations from hypnogram file.
    
    Args:
        file_path: Path to the hypnogram file
    
    Returns:
        annotations: List of sleep stage annotations
    """
    try:
        annotations = mne.read_annotations(file_path)
        return annotations
    except Exception as e:
        print(f"Error loading hypnogram {file_path}: {e}")
        return None

def preprocess_eeg(raw_data, channel='EEG Fpz-Cz', l_freq=0.5, h_freq=30.0, epoch_sec=30):
    """
    Preprocess EEG data: filter, segment into epochs, normalize.
    
    Args:
        raw_data: MNE Raw object
        channel: EEG channel to extract
        l_freq: Lower frequency bound for bandpass filter
        h_freq: Upper frequency bound for bandpass filter
        epoch_sec: Duration of each epoch in seconds
    
    Returns:
        epochs_data: Numpy array of preprocessed epochs [n_epochs, 1, samples_per_epoch]
    """
    if raw_data is None:
        return None
    
    try:
        # Extract sampling frequency
        sfreq = raw_data.info['sfreq']
        
        # Apply bandpass filter
        raw_data.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
        
        # Get the data
        eeg_data = raw_data.get_data()
        
        if len(eeg_data) == 0:
            print("No data found after preprocessing")
            return None
        
        # Calculate the number of samples per epoch
        samples_per_epoch = int(epoch_sec * sfreq)
        
        # Determine how many complete epochs we can extract
        n_epochs = len(eeg_data[0]) // samples_per_epoch
        
        if n_epochs == 0:
            print("Recording too short to extract epochs")
            return None
        
        # Segment the data into epochs
        epochs_data = []
        for i in range(n_epochs):
            start_idx = i * samples_per_epoch
            end_idx = start_idx + samples_per_epoch
            epoch = eeg_data[0, start_idx:end_idx]
            
            # Skip epochs with invalid data
            if np.isnan(epoch).any() or np.isinf(epoch).any():
                continue
            
            # Z-score normalization
            epoch = (epoch - np.mean(epoch)) / (np.std(epoch) + 1e-10)
            epochs_data.append(epoch)
        
        if not epochs_data:
            print("No valid epochs found after preprocessing")
            return None
        
        # Convert to numpy array and reshape for CNNs [n_epochs, n_channels, samples_per_epoch]
        epochs_data = np.array(epochs_data).reshape(-1, 1, samples_per_epoch)
        return epochs_data
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None

def extract_labels_from_annotations(annotations, sfreq, n_epochs, epoch_sec=30):
    """
    Extract sleep stage labels from annotations.
    
    Args:
        annotations: MNE Annotations object
        sfreq: Sampling frequency
        n_epochs: Number of epochs
        epoch_sec: Duration of each epoch in seconds
    
    Returns:
        labels: List of sleep stage labels
    """
    if annotations is None:
        return None
    
    # Initialize labels for each epoch
    labels = ['?'] * n_epochs
    
    # Convert annotations to epoch labels
    for annot in annotations:
        onset_sec = annot['onset']
        duration_sec = annot['duration']
        description = annot['description'].strip()
        
        # Extract the sleep stage from the description
        stage = description[-1] if description and description[-1] in SLEEP_STAGES else '?'
        
        # Calculate which epochs this annotation covers
        start_epoch = int(onset_sec // epoch_sec)
        end_epoch = int((onset_sec + duration_sec) // epoch_sec)
        
        # Assign the sleep stage to those epochs
        for epoch_idx in range(start_epoch, min(end_epoch + 1, n_epochs)):
            if 0 <= epoch_idx < n_epochs:
                labels[epoch_idx] = stage
    
    # Convert string labels to integers
    int_labels = [SLEEP_STAGES.get(label, SLEEP_STAGES['?']) for label in labels]
    
    return np.array(int_labels)

def find_matching_hypnogram(psg_file, base_dir):
    """
    Find the matching hypnogram file for a given PSG file, handling the specific
    naming conventions in the Sleep-EDF dataset.
    
    Args:
        psg_file: Path to the PSG file
        base_dir: Base directory containing the dataset
        
    Returns:
        hypno_file: Path to the matching hypnogram file or None if not found
    """
    # Extract the subject ID and recording number from the PSG filename
    # Example: SC4001E0-PSG.edf -> subject_id = SC4001
    base_name = os.path.basename(psg_file)
    if '-PSG.edf' in base_name:
        subject_id = base_name.split('-')[0][:-1]  # Remove the last character (E0, F0, G0, etc.)
    else:
        subject_id = os.path.splitext(base_name)[0]  # Remove file extension
    
    # Check if it's from the sleep-cassette dataset
    if 'sleep-cassette' in psg_file:
        # The hypnogram files are in the annotations folder
        hypno_dir = os.path.join(base_dir, 'dataset', 'sleep-cassette', 'annotations')
        
        # Search for any hypnogram file matching the subject ID pattern
        hypno_pattern = os.path.join(hypno_dir, f"{subject_id}*-Hypnogram.edf")
        hypno_candidates = glob.glob(hypno_pattern)
        
        if hypno_candidates:
            return hypno_candidates[0]
    
    # Check if it's from the sleep-telemetry dataset
    elif 'sleep-telemetry' in psg_file:
        # For telemetry, the hypnogram files are in the same directory
        hypno_dir = os.path.dirname(psg_file)
        
        # Search for any hypnogram file matching the subject ID pattern
        hypno_pattern = os.path.join(hypno_dir, f"{subject_id}*-Hypnogram.edf")
        hypno_candidates = glob.glob(hypno_pattern)
        
        if hypno_candidates:
            return hypno_candidates[0]
    
    return None

def load_sleep_edf_dataset(base_dir, channel='EEG Fpz-Cz'):
    """
    Load and preprocess the Sleep-EDF dataset, which has a specific directory structure.
    
    Args:
        base_dir: Base directory containing the dataset
        channel: EEG channel to extract
    
    Returns:
        all_epochs: Combined epochs data
        all_labels: Combined labels
        subject_ids: List of subject IDs
    """
    all_epochs = []
    all_labels = []
    subject_ids = []
    
    # Process sleep-cassette dataset
    cassette_dir = os.path.join(base_dir, 'dataset', 'sleep-cassette')
    if os.path.exists(cassette_dir):
        # The PSG files are in the 'data' subfolder
        psg_dir = os.path.join(cassette_dir, 'data')
        if not os.path.exists(psg_dir):
            # If 'data' folder doesn't exist, look directly in the sleep-cassette folder
            psg_dir = cassette_dir
        
        # Find all PSG files
        psg_files = glob.glob(os.path.join(psg_dir, '*-PSG.edf'))
        
        for psg_file in psg_files:
            # Extract subject ID from the filename
            subject_id = os.path.basename(psg_file).split('-')[0]
            
            # Find the matching hypnogram file using the new function
            hypno_file = find_matching_hypnogram(psg_file, base_dir)
            
            if hypno_file is None:
                print(f"No matching hypnogram found for {psg_file}")
                continue
            
            print(f"Processing {subject_id}: {psg_file} with hypnogram {hypno_file}")
            
            # Load PSG and hypnogram files
            raw_data = load_edf_file(psg_file, channel=channel)
            annotations = load_hypnogram(hypno_file)
            
            if raw_data is None or annotations is None:
                print(f"Skipping {subject_id} due to loading errors")
                continue
            
            # Preprocess the EEG data
            epochs_data = preprocess_eeg(raw_data, channel=channel)
            
            if epochs_data is None:
                print(f"Skipping {subject_id} due to preprocessing errors")
                continue
            
            # Extract labels from annotations
            n_epochs = len(epochs_data)
            sfreq = raw_data.info['sfreq']
            labels = extract_labels_from_annotations(annotations, sfreq, n_epochs)
            
            if labels is None or len(labels) != n_epochs:
                print(f"Skipping {subject_id} due to label extraction errors")
                continue
            
            # Append to the dataset
            all_epochs.append(epochs_data)
            all_labels.append(labels)
            subject_ids.extend([subject_id] * n_epochs)
    
    # Process sleep-telemetry dataset
    telemetry_dir = os.path.join(base_dir, 'dataset', 'sleep-telemetry')
    if os.path.exists(telemetry_dir):
        # Find all PSG files
        psg_files = glob.glob(os.path.join(telemetry_dir, '*-PSG.edf'))
        
        for psg_file in psg_files:
            # Extract subject ID from the filename
            subject_id = os.path.basename(psg_file).split('-')[0]
            
            # Find the matching hypnogram file using the new function
            hypno_file = find_matching_hypnogram(psg_file, base_dir)
            
            if hypno_file is None:
                print(f"No matching hypnogram found for {psg_file}")
                continue
            
            print(f"Processing {subject_id}: {psg_file} with hypnogram {hypno_file}")
            
            # Load PSG and hypnogram files
            raw_data = load_edf_file(psg_file, channel=channel)
            annotations = load_hypnogram(hypno_file)
            
            if raw_data is None or annotations is None:
                print(f"Skipping {subject_id} due to loading errors")
                continue
            
            # Preprocess the EEG data
            epochs_data = preprocess_eeg(raw_data, channel=channel)
            
            if epochs_data is None:
                print(f"Skipping {subject_id} due to preprocessing errors")
                continue
            
            # Extract labels from annotations
            n_epochs = len(epochs_data)
            sfreq = raw_data.info['sfreq']
            labels = extract_labels_from_annotations(annotations, sfreq, n_epochs)
            
            if labels is None or len(labels) != n_epochs:
                print(f"Skipping {subject_id} due to label extraction errors")
                continue
            
            # Append to the dataset
            all_epochs.append(epochs_data)
            all_labels.append(labels)
            subject_ids.extend([subject_id] * n_epochs)
    
    # Combine all recordings
    if all_epochs:
        all_epochs = np.vstack(all_epochs)
        all_labels = np.concatenate(all_labels)
        subject_ids = np.array(subject_ids)
        
        print(f"Dataset loaded: {len(all_epochs)} epochs, {len(np.unique(subject_ids))} subjects")
        return all_epochs, all_labels, subject_ids
    else:
        print("No valid recordings found")
        return None, None, None

def load_dataset(records_file, base_dir, channel='EEG Fpz-Cz'):
    """
    Load and preprocess the entire dataset based on RECORDS file.
    
    Args:
        records_file: Path to the RECORDS file (can be None for Sleep-EDF dataset)
        base_dir: Base directory containing the dataset
        channel: EEG channel to extract
    
    Returns:
        all_epochs: Combined epochs data
        all_labels: Combined labels
        subject_ids: List of subject IDs
    """
    # For Sleep-EDF dataset, use the specialized loader
    return load_sleep_edf_dataset(base_dir, channel=channel)

def create_data_loaders(X, y, subject_ids, test_size=0.15, val_size=0.15, batch_size=64, random_state=42,
                        num_workers=4, pin_memory=True):
    """
    Split the dataset and create PyTorch DataLoaders.
    
    Args:
        X: EEG epochs
        y: Labels
        subject_ids: Subject identifiers
        test_size: Proportion of data for testing
        val_size: Proportion of data for validation
        batch_size: Batch size for DataLoaders
        random_state: Random seed
        num_workers: Number of CPU workers for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
    
    Returns:
        train_loader, val_loader, test_loader: PyTorch DataLoaders
    """
    # First split to separate test set
    unique_subjects = np.unique(subject_ids)
    n_subjects = len(unique_subjects)
    n_test_subjects = max(1, int(n_subjects * test_size))
    n_val_subjects = max(1, int(n_subjects * val_size))
    
    # Shuffle subjects
    np.random.seed(random_state)
    np.random.shuffle(unique_subjects)
    
    test_subjects = unique_subjects[:n_test_subjects]
    val_subjects = unique_subjects[n_test_subjects:n_test_subjects+n_val_subjects]
    train_subjects = unique_subjects[n_test_subjects+n_val_subjects:]
    
    # Split data by subject
    test_mask = np.isin(subject_ids, test_subjects)
    val_mask = np.isin(subject_ids, val_subjects)
    train_mask = np.isin(subject_ids, train_subjects)
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"Train set: {len(X_train)} epochs, {len(train_subjects)} subjects")
    print(f"Validation set: {len(X_val)} epochs, {len(val_subjects)} subjects")
    print(f"Test set: {len(X_test)} epochs, {len(test_subjects)} subjects")
    
    # Create datasets
    train_dataset = SleepDataset(X_train, y_train)
    val_dataset = SleepDataset(X_val, y_val)
    test_dataset = SleepDataset(X_test, y_test)
    
    # Create data loaders with optimized settings
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             num_workers=num_workers, pin_memory=pin_memory)
    
    return train_loader, val_loader, test_loader, (X_test, y_test, test_subjects)