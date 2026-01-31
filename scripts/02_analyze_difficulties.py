"""
난이도 분석 및 시각화

레이블링된 난이도 데이터를 분석하고 시각화합니다.
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

from meta_token_difference import MetaTokenDifferenceAnalyzer


def load_difficulties(path: str):
    """난이도 레이블 로드"""
    return MetaTokenDifferenceAnalyzer.load_difficulties(path)


def analyze_difficulty_distribution(difficulties, output_dir: Path):
    """난이도 분포 분석"""
    
    print("Analyzing difficulty distribution...")
    
    # Extract scores
    difficulty_metrics = ['mean', 'max', 'topk_mean', 'weighted_mean', 'median']
    scores_by_metric = defaultdict(list)
    
    for d in difficulties:
        for metric in difficulty_metrics:
            if metric in d.difficulty_scores:
                scores_by_metric[metric].append(d.difficulty_scores[metric])
    
    # Plot distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(difficulty_metrics):
        ax = axes[idx]
        scores = scores_by_metric[metric]
        
        ax.hist(scores, bins=50, alpha=0.7, edgecolor='black')
        ax.set_title(f'{metric.upper()} Distribution')
        ax.set_xlabel('Difficulty Score')
        ax.set_ylabel('Frequency')
        ax.grid(alpha=0.3)
        
        # Add statistics
        mean = np.mean(scores)
        std = np.std(scores)
        ax.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.3f}')
        ax.axvline(mean + std, color='orange', linestyle=':', label=f'+1 Std')
        ax.axvline(mean - std, color='orange', linestyle=':')
        ax.legend()
    
    # Remove extra subplot
    fig.delaxes(axes[-1])
    
    plt.tight_layout()
    save_path = output_dir / 'difficulty_distributions.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {save_path}")
    plt.close()


def analyze_layer_differences(difficulties, output_dir: Path):
    """레이어별 차이 분석"""
    
    print("Analyzing layer-wise differences...")
    
    # Extract layer diffs
    num_layers = len(difficulties[0].layer_diffs)
    layer_diff_matrix = np.zeros((len(difficulties), num_layers))
    
    for i, d in enumerate(difficulties):
        for layer_idx in range(num_layers):
            layer_diff_matrix[i, layer_idx] = d.layer_diffs[layer_idx]
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sample for visualization (max 100 samples)
    if len(difficulties) > 100:
        sample_indices = np.random.choice(len(difficulties), 100, replace=False)
        sample_indices = sorted(sample_indices)
        layer_diff_matrix_sample = layer_diff_matrix[sample_indices, :]
    else:
        layer_diff_matrix_sample = layer_diff_matrix
    
    sns.heatmap(
        layer_diff_matrix_sample.T,
        cmap='YlOrRd',
        cbar_kws={'label': 'Meta Token Difference'},
        ax=ax,
    )
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Layer Index')
    ax.set_title('Layer-wise Meta Token Differences')
    
    plt.tight_layout()
    save_path = output_dir / 'layer_differences_heatmap.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {save_path}")
    plt.close()
    
    # Plot layer-wise statistics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Mean per layer
    layer_means = layer_diff_matrix.mean(axis=0)
    layer_stds = layer_diff_matrix.std(axis=0)
    
    ax = axes[0]
    ax.plot(range(num_layers), layer_means, marker='o', label='Mean')
    ax.fill_between(
        range(num_layers),
        layer_means - layer_stds,
        layer_means + layer_stds,
        alpha=0.3,
        label='±1 Std',
    )
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Meta Token Difference')
    ax.set_title('Layer-wise Difficulty (Mean ± Std)')
    ax.grid(alpha=0.3)
    ax.legend()
    
    # Difficulty variance per layer
    ax = axes[1]
    ax.bar(range(num_layers), layer_stds, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Standard Deviation')
    ax.set_title('Layer-wise Difficulty Variance')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / 'layer_statistics.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {save_path}")
    plt.close()


def analyze_curriculum_stages(difficulties, output_dir: Path):
    """커리큘럼 스테이지 분석"""
    
    print("Analyzing curriculum stages...")
    
    # Stage distribution
    num_stages = max(d.curriculum_stage for d in difficulties) + 1
    stage_counts = [
        sum(1 for d in difficulties if d.curriculum_stage == s)
        for s in range(num_stages)
    ]
    
    # Plot stage distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, num_stages))
    bars = ax.bar(range(num_stages), stage_counts, color=colors, edgecolor='black', alpha=0.8)
    
    # Add percentages
    total = len(difficulties)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        percentage = height / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{int(height)}\n({percentage:.1f}%)',
            ha='center',
            va='bottom',
        )
    
    ax.set_xlabel('Curriculum Stage')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Curriculum Stage Distribution')
    ax.set_xticks(range(num_stages))
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = output_dir / 'curriculum_stage_distribution.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {save_path}")
    plt.close()
    
    # Difficulty by stage
    fig, ax = plt.subplots(figsize=(10, 6))
    
    stage_difficulties = [
        [d.difficulty_scores['topk_mean'] for d in difficulties if d.curriculum_stage == s]
        for s in range(num_stages)
    ]
    
    bp = ax.boxplot(
        stage_difficulties,
        labels=[f'Stage {s}' for s in range(num_stages)],
        patch_artist=True,
        showmeans=True,
    )
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Curriculum Stage')
    ax.set_ylabel('Difficulty Score (Top-K Mean)')
    ax.set_title('Difficulty Distribution by Curriculum Stage')
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = output_dir / 'difficulty_by_stage.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {save_path}")
    plt.close()


def analyze_correlations(difficulties, output_dir: Path):
    """메트릭 간 상관관계 분석"""
    
    print("Analyzing metric correlations...")
    
    # Extract all metrics
    difficulty_metrics = ['mean', 'max', 'topk_mean', 'weighted_mean', 'median', 'std']
    metric_values = defaultdict(list)
    
    for d in difficulties:
        for metric in difficulty_metrics:
            if metric in d.difficulty_scores:
                metric_values[metric].append(d.difficulty_scores[metric])
    
    # Compute correlation matrix
    metrics_available = [m for m in difficulty_metrics if m in metric_values]
    n = len(metrics_available)
    corr_matrix = np.zeros((n, n))
    
    for i, m1 in enumerate(metrics_available):
        for j, m2 in enumerate(metrics_available):
            corr_matrix[i, j] = np.corrcoef(
                metric_values[m1],
                metric_values[m2],
            )[0, 1]
    
    # Plot correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.3f',
        xticklabels=metrics_available,
        yticklabels=metrics_available,
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        ax=ax,
        cbar_kws={'label': 'Correlation'},
    )
    ax.set_title('Difficulty Metric Correlations')
    
    plt.tight_layout()
    save_path = output_dir / 'metric_correlations.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize difficulty labels")
    
    parser.add_argument(
        '--difficulty_path',
        type=str,
        required=True,
        help='Path to difficulty labels JSON',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='curriculum/analysis',
        help='Directory to save analysis outputs',
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load difficulties
    print(f"Loading difficulties from {args.difficulty_path}")
    difficulties = load_difficulties(args.difficulty_path)
    print(f"  ✓ Loaded {len(difficulties)} samples")
    
    # Run analyses
    print("\n" + "=" * 60)
    print("Running Analyses")
    print("=" * 60)
    
    analyze_difficulty_distribution(difficulties, output_dir)
    analyze_layer_differences(difficulties, output_dir)
    analyze_curriculum_stages(difficulties, output_dir)
    analyze_correlations(difficulties, output_dir)
    
    print("\n" + "=" * 60)
    print(f"✓ All analyses saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
