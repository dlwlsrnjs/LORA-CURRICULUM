"""
커리큘럼 러닝 학습 통합

기존 train_dialogue_lora.py에 커리큘럼 러닝을 통합합니다.
"""

import argparse
import yaml
import torch
import os
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from curriculum.meta_token_difference import MetaTokenDifferenceAnalyzer
from curriculum.curriculum_scheduler import (
    CurriculumConfig,
    CurriculumDataset,
    CurriculumTrainingScheduler,
)
from torch.utils.data import DataLoader


def load_curriculum_config(config_path: str) -> CurriculumConfig:
    """커리큘럼 설정 로드"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    curriculum_cfg = config_dict.get('curriculum', {})
    
    config = CurriculumConfig(
        strategy=curriculum_cfg.get('strategy', 'easy_to_hard'),
        num_stages=curriculum_cfg.get('num_stages', 3),
        stage_percentiles=curriculum_cfg.get('stage_percentiles'),
        difficulty_metric=curriculum_cfg.get('difficulty_metric', 'topk_mean'),
        loss_threshold=curriculum_cfg.get('loss_threshold', 2.0),
        difficulty_step_up=curriculum_cfg.get('difficulty_step_up', 0.05),
        difficulty_step_down=curriculum_cfg.get('difficulty_step_down', 0.02),
        curriculum_start_epoch=curriculum_cfg.get('curriculum_start_epoch', 0),
        curriculum_end_epoch=curriculum_cfg.get('curriculum_end_epoch', 5),
        warmup_epochs=curriculum_cfg.get('warmup_epochs', 1),
    )
    
    return config


def create_curriculum_dataloader(
    base_dataset,
    difficulties,
    config: CurriculumConfig,
    epoch: int,
    current_loss: float = None,
    batch_size: int = 4,
    num_workers: int = 4,
    num_layers: int = 32,
):
    """
    커리큘럼 DataLoader 생성
    """
    # Wrap dataset with curriculum info
    curriculum_dataset = CurriculumDataset(
        base_dataset=base_dataset,
        difficulties=difficulties,
        config=config,
    )
    
    # Create scheduler
    total_epochs = max(config.curriculum_end_epoch + 5, epoch + 1)
    scheduler = CurriculumTrainingScheduler(
        config=config,
        total_epochs=total_epochs,
    )
    
    # Get sampler for current epoch
    sampler = scheduler.get_sampler_for_epoch(
        epoch=epoch,
        difficulties=difficulties,
        current_loss=current_loss,
        num_layers=num_layers,
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        curriculum_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return dataloader


def integrate_with_existing_training(
    original_train_script: str,
    difficulty_path: str,
    curriculum_config_path: str,
):
    """
    기존 학습 스크립트에 커리큘럼 통합
    
    This is a template function showing how to integrate.
    Actual integration should be done in the training script.
    """
    
    print("=" * 60)
    print("Curriculum Learning Integration")
    print("=" * 60)
    
    # 1. Load difficulties
    print(f"\n1. Loading difficulty labels from {difficulty_path}")
    difficulties = MetaTokenDifferenceAnalyzer.load_difficulties(difficulty_path)
    print(f"   ✓ Loaded {len(difficulties)} difficulty labels")
    
    # 2. Load curriculum config
    print(f"\n2. Loading curriculum config from {curriculum_config_path}")
    curriculum_config = load_curriculum_config(curriculum_config_path)
    print(f"   ✓ Strategy: {curriculum_config.strategy}")
    print(f"   ✓ Stages: {curriculum_config.num_stages}")
    print(f"   ✓ Warmup epochs: {curriculum_config.warmup_epochs}")
    
    # 3. Create training scheduler
    print(f"\n3. Creating curriculum training scheduler")
    total_epochs = 10  # Example
    scheduler = CurriculumTrainingScheduler(
        config=curriculum_config,
        total_epochs=total_epochs,
    )
    scheduler.print_schedule()
    
    print("\n" + "=" * 60)
    print("Integration Guide")
    print("=" * 60)
    print("""
To integrate curriculum learning into your training loop:

1. Load difficulty labels at the beginning:
   ```python
   from curriculum.meta_token_difference import MetaTokenDifferenceAnalyzer
   difficulties = MetaTokenDifferenceAnalyzer.load_difficulties('path/to/difficulties.json')
   ```

2. Replace DataLoader creation in each epoch:
   ```python
   from curriculum.train_with_curriculum import create_curriculum_dataloader
   
   for epoch in range(num_epochs):
       # Create curriculum-aware DataLoader
       train_loader = create_curriculum_dataloader(
           base_dataset=train_dataset,
           difficulties=difficulties,
           config=curriculum_config,
           epoch=epoch,
           current_loss=avg_loss,  # from previous epoch
           batch_size=batch_size,
           num_layers=model.num_layers,
       )
       
       # Training loop as usual
       for batch in train_loader:
           # batch now includes 'difficulty_info'
           ...
   ```

3. Optional: Use difficulty info for weighted loss
   ```python
   difficulty_weight = batch['difficulty_info']['difficulty_percentile']
   weighted_loss = loss * (1 + difficulty_weight)  # harder samples get more weight
   ```
    """)


def main():
    parser = argparse.ArgumentParser(description="Integrate curriculum learning")
    
    parser.add_argument(
        '--difficulty_path',
        type=str,
        required=True,
        help='Path to difficulty labels JSON',
    )
    parser.add_argument(
        '--curriculum_config',
        type=str,
        required=True,
        help='Path to curriculum config YAML',
    )
    parser.add_argument(
        '--original_train_script',
        type=str,
        default='train_dialogue_lora.py',
        help='Original training script to integrate with',
    )
    
    args = parser.parse_args()
    
    integrate_with_existing_training(
        original_train_script=args.original_train_script,
        difficulty_path=args.difficulty_path,
        curriculum_config_path=args.curriculum_config,
    )


if __name__ == '__main__':
    main()
