"""
Feature Selection using LightGBM importance ranking.

Uses LightGBM's built-in feature importance to identify and rank predictive
features, allowing the pipeline to drop the bottom 30% of noisy features
and prevent the curse of dimensionality in the LSTM input.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))


def rank_features_lgbm(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = 'target',
    n_estimators: int = 200
) -> pd.DataFrame:
    """
    Rank features by LightGBM importance.
    
    Args:
        df: DataFrame with features and target
        feature_cols: List of feature column names to evaluate
        target_col: Name of target column
        n_estimators: Number of boosting rounds
        
    Returns:
        DataFrame with columns ['feature', 'importance', 'rank']
        sorted by importance descending
    """
    import lightgbm as lgb
    
    available = [c for c in feature_cols if c in df.columns]
    if len(available) < 2:
        return pd.DataFrame(columns=['feature', 'importance', 'rank'])
    
    # Prepare data (drop NaN rows for this subset)
    subset = df[available + [target_col]].dropna()
    X = subset[available].values
    y = subset[target_col].values
    
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
        n_jobs=-1
    )
    model.fit(X, y)
    
    importances = model.feature_importances_
    
    result = pd.DataFrame({
        'feature': available,
        'importance': importances,
    })
    result = result.sort_values('importance', ascending=False).reset_index(drop=True)
    result['rank'] = range(1, len(result) + 1)
    
    return result


def select_top_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = 'target',
    keep_pct: float = 0.7
) -> list[str]:
    """
    Select top features using LightGBM importance, keeping top keep_pct%.
    
    Args:
        df: DataFrame with features and target
        feature_cols: Candidate feature columns
        target_col: Target column name
        keep_pct: Fraction of features to keep (default 0.7 = drop bottom 30%)
        
    Returns:
        List of selected feature column names
    """
    ranking = rank_features_lgbm(df, feature_cols, target_col)
    
    if ranking.empty:
        return feature_cols
    
    n_keep = max(3, int(len(ranking) * keep_pct))  # Keep at least 3
    selected = ranking.head(n_keep)['feature'].tolist()
    
    print(f"LightGBM feature selection: kept {n_keep}/{len(ranking)} features")
    print(f"  Dropped: {[f for f in ranking['feature'].tolist() if f not in selected]}")
    
    return selected


if __name__ == "__main__":
    from src.data.features import load_featured_data, get_feature_columns
    
    df = load_featured_data()
    feature_cols = get_feature_columns()
    feature_cols = [c for c in feature_cols if c in df.columns and c not in ('close', 'volume')]
    
    print("Ranking features...")
    ranking = rank_features_lgbm(df, feature_cols)
    print("\nFeature Importance Ranking:")
    print(ranking.to_string(index=False))
    
    print("\n\nSelecting top 70%:")
    selected = select_top_features(df, feature_cols)
    print(f"Selected: {selected}")
