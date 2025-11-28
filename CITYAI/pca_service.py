import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from typing import Tuple, List
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import io
import base64


def detect_numeric_columns(df: pd.DataFrame) -> List[str]:
    """
    Auto-detect numeric columns suitable for PCA.
    Excludes ID columns and columns with too many missing values.
    """
    numeric_cols = []
    
    for col in df.columns:
        # Skip if column name suggests it's an ID or index
        col_lower = str(col).lower()
        if any(x in col_lower for x in ['id', 'index', 'name', 'cluster', 'group', 'pc']):
            continue
        
        # Check if column is numeric or can be converted
        try:
            numeric_data = pd.to_numeric(df[col], errors='coerce')
            # Only include if less than 50% missing values
            if numeric_data.notna().sum() / len(df) >= 0.5:
                numeric_cols.append(col)
        except Exception:
            continue
    
    return numeric_cols


def run_pca(
    df: pd.DataFrame,
    oxide_cols: List[str] = None,
    normalize: bool = True,
    scale_features: bool = True,
    n_components: int = 2
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Perform PCA on the dataframe.
    
    Args:
        df: Input dataframe
        oxide_cols: Specific columns to use. If None, auto-detect numeric columns.
        normalize: Whether to normalize rows to sum to 1 (default: True)
        scale_features: Whether to standardize features (default: True)
        n_components: Number of principal components (default: 2)
    
    Returns:
        Tuple of (scores, explained_variance, loadings_df)
    """
    # Auto-detect columns if not provided
    if oxide_cols is None:
        oxide_cols = detect_numeric_columns(df)
    
    if not oxide_cols:
        raise ValueError("No numeric columns found for PCA")
    
    if n_components > len(oxide_cols):
        n_components = len(oxide_cols)
    
    # Extract and prepare data
    X = df[oxide_cols].fillna(0).values.astype(float)
    
    # Normalize rows (each row sums to 1)
    if normalize:
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # Avoid division by zero
        X = X / row_sums
    
    # Standardize features (mean=0, std=1)
    if scale_features:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)
    explained_variance = pca.explained_variance_ratio_
    
    # Create loadings dataframe
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=oxide_cols
    )
    
    return scores, explained_variance, loadings


def create_pca_plot(
    scores: np.ndarray,
    explained_variance: np.ndarray,
    df: pd.DataFrame = None
) -> str:
    """
    Create PCA scatter plot and return as base64 encoded PNG.
    
    Args:
        scores: PCA scores (n_samples x n_components)
        explained_variance: Explained variance ratio for each PC
        df: Original dataframe (optional, for group coloring)
    
    Returns:
        Base64 encoded PNG image string
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Check if there's a Group column for coloring
    if df is not None and "Group" in df.columns:
        groups = df["Group"].values
        unique_groups = sorted(set(groups))
        colors_map = {g: plt.cm.tab20(i % 20) for i, g in enumerate(unique_groups)}
        
        for g in unique_groups:
            idx = (groups == g)
            ax.scatter(
                scores[idx, 0], scores[idx, 1],
                label=str(g), s=50,
                color=colors_map[g],
                edgecolors='black',
                linewidths=0.5,
                alpha=0.7
            )
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    else:
        ax.scatter(
            scores[:, 0], scores[:, 1],
            s=50,
            edgecolors='black',
            linewidths=0.5,
            alpha=0.7,
            color='#2b88ff'
        )
    
    # Labels with explained variance
    ax.set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}%)', fontsize=11)
    ax.set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}%)', fontsize=11)
    ax.set_title('PCA Analysis', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Convert plot to base64
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    
    return img_base64


def process_pca_job(
    input_file: Path,
    output_dir: Path,
    n_components: int = 2
) -> dict:
    """
    Process a PCA job from file input to file output.
    
    Args:
        input_file: Path to input CSV or Excel file
        output_dir: Directory to save outputs
        n_components: Number of principal components
    
    Returns:
        Metadata dictionary with processing information
    """
    # Read input file
    if input_file.suffix.lower() == '.csv':
        df = pd.read_csv(input_file)
    elif input_file.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(input_file)
    else:
        raise ValueError(f"Unsupported file format: {input_file.suffix}")
    
    # Run PCA
    scores, explained_variance, loadings = run_pca(
        df,
        n_components=n_components
    )
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save scores CSV
    scores_df = df.copy()
    for i in range(n_components):
        scores_df[f'PC{i+1}'] = scores[:, i]
    scores_path = output_dir / "pca_scores.csv"
    scores_df.to_csv(scores_path, index=False)
    
    # Save loadings CSV
    loadings_path = output_dir / "pca_loadings.csv"
    loadings.to_csv(loadings_path)
    
    # Save explained variance
    variance_df = pd.DataFrame({
        'Component': [f'PC{i+1}' for i in range(len(explained_variance))],
        'Explained_Variance': explained_variance,
        'Cumulative_Variance': np.cumsum(explained_variance)
    })
    variance_path = output_dir / "explained_variance.csv"
    variance_df.to_csv(variance_path, index=False)
    
    # Create and save plot
    plot_base64 = create_pca_plot(scores, explained_variance, df)
    plot_path = output_dir / "pca_plot.png"
    with open(plot_path, 'wb') as f:
        f.write(base64.b64decode(plot_base64))
    
    # Prepare metadata
    metadata = {
        "n_components": n_components,
        "n_samples": len(df),
        "explained_variance": explained_variance.tolist(),
        "cumulative_variance": np.cumsum(explained_variance).tolist(),
        "columns_used": loadings.index.tolist(),
        "scores_file": str(scores_path),
        "loadings_file": str(loadings_path),
        "variance_file": str(variance_path),
        "plot_file": str(plot_path),
        "plot_base64": plot_base64
    }
    
    return metadata
