"""
Bi-LSTM Feature Extractor - Extracts market state from price sequences.

Architecture:
- Input: 60 candles × N features (sequence)
- Bi-LSTM: 2 layers, 128 hidden units
- Output: 64-dimensional state vector

Pre-training:
- Task: Predict next candle direction (up/down)
- After training, freeze weights and use as feature extractor for PPO
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
from typing import Optional
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import (
    SEQUENCE_LENGTH, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, 
    STATE_DIM, MODELS_DIR, DEVICE_LSTM as DEVICE
)


class BiLSTMFeatureExtractor(nn.Module):
    """
    Bi-directional LSTM for extracting market state features.
    
    Args:
        input_size: Number of features per timestep
        hidden_size: LSTM hidden layer size
        num_layers: Number of LSTM layers
        output_size: Size of output state vector
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = LSTM_HIDDEN_SIZE,
        num_layers: int = LSTM_NUM_LAYERS,
        output_size: int = STATE_DIM,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bi-directional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection: Bi-LSTM outputs 2*hidden_size
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
        
        # Classification head (for pre-training)
        self.classifier = nn.Sequential(
            nn.Linear(output_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, return_state: bool = True) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            return_state: If True, return state vector. If False, return classification.
            
        Returns:
            State vector (batch, output_size) or classification (batch, 1)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last timestep output (combines forward and backward)
        last_output = lstm_out[:, -1, :]
        
        # Project to state vector
        state = self.fc(last_output)
        
        if return_state:
            return state
        else:
            # Classification mode (for pre-training)
            return self.classifier(state)
    
    def extract_features(self, x: torch.Tensor) -> np.ndarray:
        """
        Extract features without gradient computation.
        
        Args:
            x: Input tensor
            
        Returns:
            Numpy array of state vectors
        """
        self.eval()
        with torch.no_grad():
            state = self.forward(x, return_state=True)
        return state.cpu().numpy()


class SequenceDataset(Dataset):
    """
    Dataset for creating sequences from time series data.
    """
    
    def __init__(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        sequence_length: int = SEQUENCE_LENGTH
    ):
        """
        Args:
            data: Feature array of shape (n_samples, n_features)
            targets: Target array of shape (n_samples,)
            sequence_length: Length of each sequence
        """
        self.data = torch.FloatTensor(data)
        self.targets = torch.FloatTensor(targets)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        # Get sequence
        x = self.data[idx:idx + self.sequence_length]
        # Target is the value at the end of the sequence
        y = self.targets[idx + self.sequence_length - 1]
        return x, y


def create_sequences(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = 'target',
    sequence_length: int = SEQUENCE_LENGTH
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create sequences from DataFrame.
    
    Args:
        df: DataFrame with features
        feature_cols: List of feature column names
        target_col: Name of target column
        sequence_length: Length of each sequence
        
    Returns:
        Tuple of (features, targets) arrays
    """
    features = df[feature_cols].values.astype(np.float32)
    targets = df[target_col].values.astype(np.float32)
    
    # Handle NaN/Inf
    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return features, targets


def train_bilstm(
    model: BiLSTMFeatureExtractor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    patience: int = 10,
    class_weights: Optional[object] = None
) -> dict:
    """
    Pre-train Bi-LSTM on direction prediction task.
    
    Args:
        model: BiLSTM model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Maximum epochs
        learning_rate: Learning rate
        patience: Early stopping patience
        
    Returns:
        Training history dict
    """
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Use BCELoss but support class weights by applying per-sample weights in the batch
    use_weighting = class_weights is not None
    if use_weighting:
        cw = class_weights.to(DEVICE)
        print(f"Using class weights: {cw.cpu().numpy()}")
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            pred = model(x, return_state=False).squeeze()

            if use_weighting:
                # element-wise BCE and weighting
                bce = nn.BCELoss(reduction='none')
                loss_raw = bce(pred, y)
                sample_weights = torch.where(y == 1, cw[1], cw[0])
                loss = (loss_raw * sample_weights).mean()
            else:
                loss = nn.BCELoss()(pred, y)

            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        
        # Validation
        model.eval()
        val_losses = []
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x, return_state=False).squeeze()

                if use_weighting:
                    bce = nn.BCELoss(reduction='none')
                    loss_raw = bce(pred, y)
                    sample_weights = torch.where(y == 1, cw[1], cw[0])
                    loss = (loss_raw * sample_weights).mean()
                else:
                    loss = nn.BCELoss()(pred, y)

                val_losses.append(loss.item())

                # Accuracy
                predicted = (pred > 0.5).float()
                correct += (predicted == y).sum().item()
                total += y.size(0)

        avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
        val_acc = correct / total if total > 0 else 0.0

        # Update scheduler
        scheduler.step(avg_val_loss)

        # Logging
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            save_model(model, 'bilstm_best.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return history


def save_model(model: BiLSTMFeatureExtractor, filename: str = 'bilstm.pt'):
    """Save model weights."""
    filepath = MODELS_DIR / filename
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': model.lstm.input_size,
        'hidden_size': model.hidden_size,
        'num_layers': model.num_layers,
        'output_size': model.fc[-1].out_features
    }, filepath)
    print(f"Model saved to {filepath}")


def load_model(filename: str = 'bilstm_best.pt') -> BiLSTMFeatureExtractor:
    """Load model from weights file."""
    filepath = MODELS_DIR / filename
    checkpoint = torch.load(filepath, map_location=DEVICE)
    
    model = BiLSTMFeatureExtractor(
        input_size=checkpoint['input_size'],
        hidden_size=checkpoint['hidden_size'],
        num_layers=checkpoint['num_layers'],
        output_size=checkpoint['output_size']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    print(f"Model loaded from {filepath}")
    return model


def freeze_model(model: BiLSTMFeatureExtractor):
    """Freeze all parameters (for use with PPO)."""
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    print("Model frozen for inference")


if __name__ == "__main__":
    # Test model creation
    batch_size = 32
    seq_len = 60
    n_features = 10
    
    model = BiLSTMFeatureExtractor(input_size=n_features)
    print(model)
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, n_features)
    state = model(x, return_state=True)
    print(f"State shape: {state.shape}")  # Should be (32, 64)
    
    pred = model(x, return_state=False)
    print(f"Prediction shape: {pred.shape}")  # Should be (32, 1)
