"""
Purged Walk-Forward Cross-Validation - Lopez de Prado methodology.

Prevents data leakage by:
1. Only training on past data
2. Adding embargo (purge) period between train and test
3. Walking forward through time
"""
import numpy as np
import pandas as pd
from typing import Generator, Tuple
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import N_SPLITS, EMBARGO_PCT, TRAIN_DAYS, TEST_DAYS

# Try to import mlfinpy, fall back to custom implementation
try:
    from mlfinpy.cross_validation import PurgedKFold
    HAS_MLFINPY = True
except ImportError:
    HAS_MLFINPY = False
    print("Warning: mlfinpy not installed. Using custom PurgedKFold implementation.")


class CustomPurgedKFold:
    """
    Custom implementation of Purged K-Fold Cross-Validation.
    
    Based on Lopez de Prado's methodology from 
    "Advances in Financial Machine Learning".
    
    Key concepts:
    - Purge: Remove samples that could leak information
    - Embargo: Add gap between train and test sets
    - Walk-forward: Only train on past data
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01
    ):
        """
        Args:
            n_splits: Number of cross-validation folds
            embargo_pct: Percentage of data to embargo (purge) after training set
        """
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
    
    def split(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        timestamps: pd.DatetimeIndex = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices with purging.
        
        Args:
            X: Features array
            y: Targets array (unused but kept for sklearn compatibility)
            timestamps: DatetimeIndex for time-based splitting
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        embargo_size = int(n_samples * self.embargo_pct)
        
        # Calculate fold size
        fold_size = n_samples // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            # Training set: all data before the test set minus embargo
            test_start = (i + 1) * fold_size
            test_end = min((i + 2) * fold_size, n_samples)
            
            # Purge: remove embargo_size samples before test set
            train_end = max(0, test_start - embargo_size)
            
            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices


class PurgedWalkForward:
    """
    Walk-Forward validation with purging for time series.
    
    This is the preferred method for financial ML:
    - Train on fixed window of past data
    - Purge overlapping samples
    - Test on future window
    - Walk forward and repeat
    """
    
    def __init__(
        self,
        train_period: int = TRAIN_DAYS,
        test_period: int = TEST_DAYS,
        embargo_period: int = 4,  # 4 candles = 1 hour for 15-min data
        step_size: int = None
    ):
        """
        Args:
            train_period: Number of periods (e.g., days) for training
            test_period: Number of periods for testing
            embargo_period: Number of samples to purge between train/test
            step_size: How far to walk forward each iteration (default = test_period)
        """
        self.train_period = train_period
        self.test_period = test_period
        self.embargo_period = embargo_period
        self.step_size = step_size or test_period
    
    def split(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        timestamps: pd.DatetimeIndex = None,
        samples_per_day: int = 96  # 96 15-min candles per day
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate walk-forward train/test splits.
        
        Args:
            X: Features array
            y: Targets (unused)
            timestamps: DatetimeIndex
            samples_per_day: Number of samples per day
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        train_size = self.train_period * samples_per_day
        test_size = self.test_period * samples_per_day
        step_size = self.step_size * samples_per_day
        
        current_pos = 0
        fold = 0
        
        while current_pos + train_size + self.embargo_period + test_size <= n_samples:
            # Training indices
            train_start = current_pos
            train_end = current_pos + train_size
            
            # Test indices (after embargo)
            test_start = train_end + self.embargo_period
            test_end = min(test_start + test_size, n_samples)
            
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)
            
            fold += 1
            print(f"Fold {fold}: Train [{train_start}:{train_end}] "
                  f"Purge [{train_end}:{test_start}] "
                  f"Test [{test_start}:{test_end}]")
            
            yield train_indices, test_indices
            
            # Walk forward
            current_pos += step_size
    
    def get_n_splits(self, X: np.ndarray, samples_per_day: int = 96) -> int:
        """Calculate number of splits."""
        n_samples = len(X)
        train_size = self.train_period * samples_per_day
        test_size = self.test_period * samples_per_day
        step_size = self.step_size * samples_per_day
        
        n_splits = 0
        current_pos = 0
        
        while current_pos + train_size + self.embargo_period + test_size <= n_samples:
            n_splits += 1
            current_pos += step_size
        
        return n_splits


def get_purged_cv(use_mlfinpy: bool = True, **kwargs):
    """
    Factory function to get appropriate cross-validator.
    
    Args:
        use_mlfinpy: Whether to use mlfinpy if available
        **kwargs: Arguments for the cross-validator
        
    Returns:
        Cross-validator instance
    """
    if use_mlfinpy and HAS_MLFINPY:
        return PurgedKFold(**kwargs)
    else:
        return CustomPurgedKFold(**kwargs)


def get_walk_forward_cv(**kwargs):
    """Get walk-forward cross-validator."""
    return PurgedWalkForward(**kwargs)


def visualize_splits(
    cv,
    n_samples: int,
    timestamps: pd.DatetimeIndex = None
):
    """
    Visualize cross-validation splits.
    
    Args:
        cv: Cross-validator instance
        n_samples: Total number of samples
        timestamps: Optional timestamps for display
    """
    X_dummy = np.zeros((n_samples, 1))
    
    print("\n" + "=" * 60)
    print("PURGED WALK-FORWARD SPLITS")
    print("=" * 60)
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X_dummy)):
        train_start, train_end = train_idx[0], train_idx[-1]
        test_start, test_end = test_idx[0], test_idx[-1]
        purge_size = test_start - train_end - 1
        
        if timestamps is not None:
            print(f"\nFold {fold + 1}:")
            print(f"  Train: {timestamps[train_start]} to {timestamps[train_end]} "
                  f"({len(train_idx)} samples)")
            print(f"  Purge: {purge_size} samples")
            print(f"  Test:  {timestamps[test_start]} to {timestamps[test_end]} "
                  f"({len(test_idx)} samples)")
        else:
            print(f"\nFold {fold + 1}:")
            print(f"  Train: [{train_start}:{train_end}] ({len(train_idx)} samples)")
            print(f"  Purge: {purge_size} samples")
            print(f"  Test:  [{test_start}:{test_end}] ({len(test_idx)} samples)")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Test with dummy data
    n_samples = 96 * 60  # 60 days of 15-min data
    X = np.random.randn(n_samples, 10)
    
    print("Testing PurgedWalkForward:")
    cv = PurgedWalkForward(train_period=30, test_period=7, embargo_period=4)
    visualize_splits(cv, n_samples)
    
    print(f"\nTotal splits: {cv.get_n_splits(X)}")
