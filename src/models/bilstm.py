"""
Bi-LSTM Feature Extractor - Enhanced architecture for 70-75% accuracy.

Architecture (from research guide):
- Input: 60 candles × ~22 features
- BatchNorm1d on input features
- Bi-LSTM Layer 1: 128 hidden units, dropout=0.2
- Bi-LSTM Layer 2: 64 hidden units, dropout=0.2
- MultiheadAttention (128 dim, 4 heads) ← +2-4% accuracy
- LayerNorm(128)
- Dense head: 128 → 64 → 32 → output (64-dim state)
- Classifier: output → 32 → 1 (Sigmoid) for pre-training

Pre-training:
- Task: Predict next candle direction (up/down)
- Loss: Focal Loss (γ=2) for class imbalance
- After training, freeze weights and use as feature extractor for PPO
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix
from pathlib import Path
from tqdm import tqdm
import sys
from typing import Optional
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import (
    SEQUENCE_LENGTH, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, 
    STATE_DIM, MODELS_DIR, DEVICE_LSTM as DEVICE,
    LSTM_HIDDEN_SIZE_L2, ATTENTION_HEADS, FOCAL_LOSS_GAMMA
)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in binary classification.
    
    From "Focal Loss for Dense Object Detection" (Lin et al., 2017).
    Down-weights easy examples and focuses training on hard misclassified ones.
    
    With γ=2: well-classified examples (pt>0.6) get 4x less weight.
    """
    
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Focusing parameter. Higher = more focus on hard examples.
                   γ=0 is equivalent to BCE. γ=2 is recommended default.
            alpha: Balancing factor for positive class (1-alpha for negative).
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted probabilities [0, 1], shape (batch,)
            target: Binary targets {0, 1}, shape (batch,)
        """
        # Clamp predictions to avoid log(0)
        pred = pred.clamp(1e-7, 1 - 1e-7)
        
        # Standard BCE (element-wise)
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        
        # Probability of correct classification
        pt = torch.where(target == 1, pred, 1 - pred)
        
        # Class-balancing weight
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        
        # Focal modulating factor: (1 - pt)^γ
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        
        return (focal_weight * bce).mean()


class TemporalAttentionPool(nn.Module):
    """
    Learn to weight timesteps based on importance instead of using only
    the last timestep. This preserves information from all 60 candles.
    
    Architecture: 2-layer MLP with tanh activation produces per-timestep
    importance scores, softmax normalizes them, then weighted sum.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (batch, seq_len, dim). Returns: (batch, dim)"""
        weights = self.attention(x)           # (batch, seq_len, 1)
        weights = torch.softmax(weights, dim=1)
        return (weights * x).sum(dim=1)       # (batch, dim)


class BiLSTMFeatureExtractor(nn.Module):
    """
    Enhanced Bi-directional LSTM with self-attention for market state extraction.
    
    Architecture follows research guide recommendations:
    - 2-layer Bi-LSTM with decreasing width (128→64)
    - Multi-head self-attention after LSTM
    - BatchNorm at input, LayerNorm after attention
    - Deeper dense head with dropout
    
    Args:
        input_size: Number of features per timestep
        hidden_size: LSTM hidden layer size (layer 1)
        hidden_size_l2: LSTM hidden layer size (layer 2)
        num_layers: Kept for backward compat, but layers are separate now
        output_size: Size of output state vector
        dropout: Dropout probability
        n_heads: Number of attention heads
        use_attention: Whether to use self-attention
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = LSTM_HIDDEN_SIZE,
        num_layers: int = LSTM_NUM_LAYERS,
        output_size: int = STATE_DIM,
        dropout: float = 0.2,
        hidden_size_l2: int = LSTM_HIDDEN_SIZE_L2,
        n_heads: int = ATTENTION_HEADS,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.hidden_size_l2 = hidden_size_l2
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # ====================================================================
        # Input normalization — stabilizes training across different features
        # ====================================================================
        self.batch_norm = nn.BatchNorm1d(input_size)
        
        # ====================================================================
        # Bi-LSTM Layer 1: larger capacity for initial feature extraction
        # ====================================================================
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )
        self.dropout1 = nn.Dropout(dropout)
        
        # ====================================================================
        # Bi-LSTM Layer 2: smaller, for refined representation
        # ====================================================================
        self.lstm2 = nn.LSTM(
            input_size=hidden_size * 2,  # BiLSTM L1 output
            hidden_size=hidden_size_l2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )
        self.dropout2 = nn.Dropout(dropout)
        
        # Attention dimension = L2 BiLSTM output = hidden_size_l2 * 2
        attn_dim = hidden_size_l2 * 2
        
        # ====================================================================
        # Self-Attention — captures which timesteps matter most (+2-4% acc)
        # ====================================================================
        # Temporal attention pooling (replaces last-timestep-only)
        self.temporal_pool = TemporalAttentionPool(attn_dim)
        
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=attn_dim,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True
            )
            self.layer_norm = nn.LayerNorm(attn_dim)
        
        # ====================================================================
        # Dense head: project to state vector
        # ====================================================================
        self.fc = nn.Sequential(
            nn.Linear(attn_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )
        
        # Classification head (for pre-training on direction prediction)
        self.classifier = nn.Sequential(
            nn.Linear(output_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
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
        batch_size, seq_len, n_features = x.shape
        
        # ====================================================================
        # BatchNorm: normalize across features for each sample
        # BatchNorm1d expects (batch, features) or (batch, features, length)
        # We have (batch, seq_len, features) → permute to (batch, features, seq_len)
        # ====================================================================
        x = x.permute(0, 2, 1)  # (batch, features, seq_len)
        x = self.batch_norm(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, features)
        
        # ====================================================================
        # Bi-LSTM Layer 1
        # ====================================================================
        lstm1_out, _ = self.lstm1(x)  # (batch, seq_len, hidden*2)
        lstm1_out = self.dropout1(lstm1_out)
        
        # ====================================================================
        # Bi-LSTM Layer 2
        # ====================================================================
        lstm2_out, _ = self.lstm2(lstm1_out)  # (batch, seq_len, hidden_l2*2)
        lstm2_out = self.dropout2(lstm2_out)
        
        # ====================================================================
        # Self-Attention (optional but recommended)
        # ====================================================================
        if self.use_attention:
            # Self-attention: query=key=value=lstm_output
            attn_out, _ = self.attention(lstm2_out, lstm2_out, lstm2_out)
            # Residual connection + LayerNorm
            attn_out = self.layer_norm(attn_out + lstm2_out)
            # Temporal attention pooling across ALL timesteps
            last_output = self.temporal_pool(attn_out)
        else:
            # Temporal attention pooling without self-attention
            last_output = self.temporal_pool(lstm2_out)
        
        # ====================================================================
        # Dense projection to state vector
        # ====================================================================
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
    epochs: int = 100,
    learning_rate: float = 1e-4,
    patience: int = 15,
    class_weights: Optional[object] = None,
    use_focal_loss: bool = False,
    focal_gamma: float = FOCAL_LOSS_GAMMA,
    label_smoothing: float = 0.05,
    use_mixup: bool = True,
    mixup_alpha: float = 0.2
) -> dict:
    """
    Pre-train Bi-LSTM on direction prediction task.
    
    v3 defaults: BCE loss (not Focal — classes are balanced), label smoothing
    (ε=0.05 to soften ambiguous near-zero-return samples), and Mixup
    augmentation (α=0.2 to increase effective dataset size).
    
    Args:
        model: BiLSTM model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Maximum epochs
        learning_rate: Initial learning rate
        patience: Early stopping patience
        class_weights: Optional class weights tensor
        use_focal_loss: Whether to use Focal Loss (recommended)
        focal_gamma: Focal Loss gamma parameter
        
    Returns:
        Training history dict
    """
    model = model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    # Loss function selection
    if use_focal_loss:
        criterion = FocalLoss(gamma=focal_gamma, alpha=0.25)
        print(f"Using Focal Loss (gamma={focal_gamma}, alpha=0.25)")
    else:
        criterion = None  # will use BCE with optional class weights
    
    use_weighting = (not use_focal_loss) and (class_weights is not None)
    if use_weighting:
        cw = class_weights.to(DEVICE)
        print(f"Using class-weighted BCE: {cw.cpu().numpy()}")
    
    print(f"Label smoothing: epsilon={label_smoothing}")
    print(f"Mixup augmentation: {'ON (alpha=' + str(mixup_alpha) + ')' if use_mixup else 'OFF'}")
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # Apply label smoothing: soften hard 0/1 targets
            if label_smoothing > 0:
                y = y * (1 - label_smoothing) + 0.5 * label_smoothing
            
            # Apply Mixup augmentation
            if use_mixup:
                x, y, _ = mixup_data(x, y, alpha=mixup_alpha)

            optimizer.zero_grad()
            pred = model(x, return_state=False).squeeze()

            if use_focal_loss:
                loss = criterion(pred, y)
            elif use_weighting:
                bce = nn.BCELoss(reduction='none')
                loss_raw = bce(pred, y)
                sample_weights = torch.where(y > 0.5, cw[1], cw[0])
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
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x, return_state=False).squeeze()

                if use_focal_loss:
                    loss = criterion(pred, y)
                elif use_weighting:
                    bce = nn.BCELoss(reduction='none')
                    loss_raw = bce(pred, y)
                    sample_weights = torch.where(y == 1, cw[1], cw[0])
                    loss = (loss_raw * sample_weights).mean()
                else:
                    loss = nn.BCELoss()(pred, y)

                val_losses.append(loss.item())

                # Collect predictions/targets for imbalanced metrics
                val_preds.append(pred.detach().cpu().numpy())
                val_targets.append(y.detach().cpu().numpy())

                # Accuracy
                predicted = (pred > 0.5).float()
                correct += (predicted == y).sum().item()
                total += y.size(0)

        avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
        val_acc = correct / total if total > 0 else 0.0

        # Imbalance-aware metrics
        if val_preds and val_targets:
            y_true = np.concatenate(val_targets).astype(np.float32)
            y_prob = np.concatenate(val_preds).astype(np.float32)

            thresholds = [0.05, 0.10, 0.20, 0.30, 0.50]
            best_f1 = -1.0
            best_thr = 0.50
            best_stats = (0.0, 0.0, 0.0)
            best_cm = np.array([[0, 0], [0, 0]])

            for thr in thresholds:
                y_hat = (y_prob > thr).astype(np.float32)
                tp = float(((y_hat == 1) & (y_true == 1)).sum())
                fp = float(((y_hat == 1) & (y_true == 0)).sum())
                fn = float(((y_hat == 0) & (y_true == 1)).sum())

                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)

                if f1 > best_f1:
                    best_f1 = f1
                    best_thr = thr
                    best_stats = (precision, recall, f1)
                    best_cm = confusion_matrix(y_true, y_hat, labels=[0.0, 1.0])

            precision, recall, f1 = best_stats
            pr_auc = average_precision_score(y_true, y_prob)
            cm = best_cm
            baseline_acc = 1.0 - float(y_true.mean())
        else:
            precision = recall = f1 = pr_auc = 0.0
            best_thr = 0.50
            cm = np.array([[0, 0], [0, 0]])
            baseline_acc = 0.0

        # Update scheduler
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Logging
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}, "
              f"LR={current_lr:.2e}")
        print(
            f"  BestThr={best_thr:.2f}  Pos P/R/F1={precision:.4f}/{recall:.4f}/{f1:.4f}  "
            f"PR-AUC={pr_auc:.4f}  Baseline Acc={baseline_acc:.4f}"
        )
        print(
            f"  Confusion Matrix [[TN, FP], [FN, TP]] = "
            f"[[{cm[0,0]}, {cm[0,1]}], [{cm[1,0]}, {cm[1,1]}]]"
        )
        
        # Track best accuracy for reporting
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        # Early stopping (based on validation loss)
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
    
    print(f"\n[BEST] Best validation accuracy: {best_val_acc:.4f}")
    return history


def save_model(model: BiLSTMFeatureExtractor, filename: str = 'bilstm.pt'):
    """Save model weights with architecture metadata."""
    filepath = MODELS_DIR / filename
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': model.lstm1.input_size,
        'hidden_size': model.hidden_size,
        'hidden_size_l2': model.hidden_size_l2,
        'num_layers': model.num_layers,
        'output_size': model.fc[-1].out_features,
        'use_attention': model.use_attention,
        'arch_version': 'v2_attention'  # Flag for new architecture
    }, filepath)
    print(f"Model saved to {filepath}")


def load_model(filename: str = 'bilstm_best.pt') -> BiLSTMFeatureExtractor:
    """Load model from weights file, supporting both v1 and v2 architectures."""
    filepath = MODELS_DIR / filename
    checkpoint = torch.load(filepath, map_location=DEVICE)
    
    # Detect architecture version
    arch_version = checkpoint.get('arch_version', 'v1')
    
    if arch_version == 'v2_attention':
        model = BiLSTMFeatureExtractor(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            hidden_size_l2=checkpoint.get('hidden_size_l2', LSTM_HIDDEN_SIZE_L2),
            num_layers=checkpoint['num_layers'],
            output_size=checkpoint['output_size'],
            use_attention=checkpoint.get('use_attention', True)
        )
    else:
        # Legacy v1 model — create with backward-compatible settings
        print("⚠️  Loading legacy v1 model — consider retraining with v2 architecture")
        model = BiLSTMFeatureExtractor(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers'],
            output_size=checkpoint['output_size'],
            use_attention=False  # v1 had no attention
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    print(f"Model loaded from {filepath} (arch: {arch_version})")
    return model


def freeze_model(model: BiLSTMFeatureExtractor):
    """Freeze all parameters (for use with PPO)."""
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    print("Model frozen for inference")


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """
    Apply Mixup augmentation — interpolates between random pairs of
    training samples to create synthetic data points.
    
    This acts as a strong regularizer and improves generalization,
    especially when training data is limited (~13K sequences).
    
    Args:
        x: Input tensor (batch, seq_len, features)
        y: Target tensor (batch,)
        alpha: Beta distribution parameter (lower = less mixing)
        
    Returns:
        Mixed inputs, mixed targets, mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    
    return mixed_x, mixed_y, lam


if __name__ == "__main__":
    # Test model creation
    batch_size = 32
    seq_len = 60
    n_features = 22  # ~22 features in the enhanced set
    
    model = BiLSTMFeatureExtractor(input_size=n_features)
    print(model)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {n_params:,}")
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, n_features)
    state = model(x, return_state=True)
    print(f"State shape: {state.shape}")  # Should be (32, 64)
    
    pred = model(x, return_state=False)
    print(f"Prediction shape: {pred.shape}")  # Should be (32, 1)
    
    # Test Focal Loss
    loss_fn = FocalLoss(gamma=2.0)
    target = torch.randint(0, 2, (batch_size,)).float()
    loss = loss_fn(pred.squeeze(), target)
    print(f"Focal Loss: {loss.item():.4f}")
