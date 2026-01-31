"""
커리큘럼 러닝 스케줄러

난이도 정보를 바탕으로 학습 샘플의 순서를 결정하고,
레이어별 학습 전략을 관리합니다.
"""

import torch
from torch.utils.data import Dataset, Sampler
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import json


@dataclass
class CurriculumConfig:
    """커리큘럼 설정"""
    strategy: str = "easy_to_hard"  # easy_to_hard, layer_wise, dynamic, hybrid
    num_stages: int = 3
    stage_percentiles: List[float] = None  # [0.33, 0.66, 1.0]
    difficulty_metric: str = "topk_mean"
    
    # Layer-wise progressive
    layer_schedule: List[Tuple[int, int]] = None  # [(start_layer, end_layer), ...]
    
    # Dynamic pacing
    loss_threshold: float = 2.0
    difficulty_step_up: float = 0.05
    difficulty_step_down: float = 0.02
    
    # Training schedule
    curriculum_start_epoch: int = 0
    curriculum_end_epoch: int = 5
    warmup_epochs: int = 1
    
    def __post_init__(self):
        if self.stage_percentiles is None:
            # 균등 분할
            self.stage_percentiles = [
                (i + 1) / self.num_stages for i in range(self.num_stages)
            ]


class CurriculumSampler(Sampler):
    """
    커리큘럼 러닝을 위한 커스텀 샘플러
    """
    
    def __init__(
        self,
        difficulties: List,  # MetaTokenDifference objects
        config: CurriculumConfig,
        current_epoch: int = 0,
        current_loss: Optional[float] = None,
        num_layers: int = 32,
    ):
        self.difficulties = difficulties
        self.config = config
        self.current_epoch = current_epoch
        self.current_loss = current_loss
        self.num_layers = num_layers
        
        # 난이도 점수 추출
        self.difficulty_scores = np.array([
            d.difficulty_scores[config.difficulty_metric]
            for d in difficulties
        ])
        
        # 샘플 인덱스
        self.indices = np.arange(len(difficulties))
        
        # 전략별 샘플 순서 결정
        self.ordered_indices = self._compute_sample_order()
    
    def _compute_sample_order(self) -> np.ndarray:
        """전략에 따라 샘플 순서 결정"""
        
        if self.config.strategy == "easy_to_hard":
            return self._easy_to_hard_order()
        
        elif self.config.strategy == "layer_wise_progressive":
            return self._layer_wise_progressive_order()
        
        elif self.config.strategy == "dynamic_pacing":
            return self._dynamic_pacing_order()
        
        elif self.config.strategy == "hybrid":
            return self._hybrid_order()
        
        else:
            # Random (baseline)
            return np.random.permutation(self.indices)
    
    def _easy_to_hard_order(self) -> np.ndarray:
        """쉬운 것부터 어려운 것까지 정렬"""
        sorted_indices = np.argsort(self.difficulty_scores)
        return sorted_indices
    
    def _layer_wise_progressive_order(self) -> np.ndarray:
        """
        레이어별 점진적 학습
        
        현재 epoch에 따라 사용할 레이어 범위와 난이도 결정
        """
        # 현재 스테이지 결정
        if self.current_epoch < self.config.warmup_epochs:
            stage = 0
        elif self.current_epoch >= self.config.curriculum_end_epoch:
            stage = self.config.num_stages - 1
        else:
            # 선형 증가
            progress = (self.current_epoch - self.config.warmup_epochs) / \
                       (self.config.curriculum_end_epoch - self.config.warmup_epochs)
            stage = int(progress * self.config.num_stages)
            stage = min(stage, self.config.num_stages - 1)
        
        # 스테이지별 난이도 threshold
        difficulty_threshold = self.config.stage_percentiles[stage]
        
        # 난이도가 threshold 이하인 샘플 선택
        percentiles = np.array([d.difficulty_percentile for d in self.difficulties])
        mask = percentiles <= difficulty_threshold
        
        # 선택된 샘플을 난이도 순으로 정렬
        selected_indices = self.indices[mask]
        selected_scores = self.difficulty_scores[mask]
        sorted_order = np.argsort(selected_scores)
        
        return selected_indices[sorted_order]
    
    def _dynamic_pacing_order(self) -> np.ndarray:
        """
        손실 기반 동적 난이도 조절
        """
        if self.current_loss is None or self.current_epoch < self.config.warmup_epochs:
            # 초기에는 easy-to-hard
            return self._easy_to_hard_order()
        
        # 손실에 따라 난이도 threshold 조절
        if self.current_loss < self.config.loss_threshold:
            # 손실이 낮으면 난이도 증가
            self.current_difficulty_percentile = getattr(
                self, 'current_difficulty_percentile', 0.33
            ) + self.config.difficulty_step_up
        else:
            # 손실이 높으면 난이도 감소 (복습)
            self.current_difficulty_percentile = getattr(
                self, 'current_difficulty_percentile', 0.33
            ) - self.config.difficulty_step_down
        
        # 범위 제한
        self.current_difficulty_percentile = np.clip(
            self.current_difficulty_percentile, 0.1, 1.0
        )
        
        # Threshold 이하 샘플 선택
        percentiles = np.array([d.difficulty_percentile for d in self.difficulties])
        mask = percentiles <= self.current_difficulty_percentile
        
        selected_indices = self.indices[mask]
        selected_scores = self.difficulty_scores[mask]
        sorted_order = np.argsort(selected_scores)
        
        return selected_indices[sorted_order]
    
    def _hybrid_order(self) -> np.ndarray:
        """
        Layer-wise progressive + Dynamic pacing 결합
        """
        # Layer-wise로 기본 선택
        layer_order = self._layer_wise_progressive_order()
        
        # Dynamic pacing으로 미세 조정
        if self.current_loss is not None and self.current_epoch >= self.config.warmup_epochs:
            if self.current_loss < self.config.loss_threshold:
                # 손실이 낮으면 더 어려운 샘플 추가
                all_indices = set(self.indices)
                current_indices = set(layer_order)
                remaining_indices = list(all_indices - current_indices)
                
                # 남은 샘플 중 쉬운 것 일부 추가
                remaining_scores = self.difficulty_scores[remaining_indices]
                add_count = int(len(remaining_indices) * self.config.difficulty_step_up)
                if add_count > 0:
                    add_order = np.argsort(remaining_scores)[:add_count]
                    add_indices = np.array(remaining_indices)[add_order]
                    layer_order = np.concatenate([layer_order, add_indices])
        
        return layer_order
    
    def __iter__(self):
        # 샘플 순서 재계산 (매 epoch마다)
        self.ordered_indices = self._compute_sample_order()
        return iter(self.ordered_indices.tolist())
    
    def __len__(self):
        return len(self.ordered_indices)


class LayerSpecificCurriculumScheduler:
    """
    레이어별 독립적인 커리큘럼 스케줄링
    
    각 레이어가 자신의 난이도 메트릭에 따라 학습
    """
    
    def __init__(
        self,
        difficulties: List,
        num_layers: int,
        num_stages: int = 3,
        hierarchical: bool = True,  # True: 레이어별 → Joint, False: 레이어별만
    ):
        self.difficulties = difficulties
        self.num_layers = num_layers
        self.num_stages = num_stages
        self.hierarchical = hierarchical
        
        # 레이어별 난이도 순서 미리 계산
        self.layer_orderings = self._precompute_layer_orderings()
    
    def _precompute_layer_orderings(self) -> Dict[int, List[np.ndarray]]:
        """
        각 레이어에 대해 난이도 순서 계산
        
        Returns:
            {layer_idx: [stage0_order, stage1_order, stage2_order]}
        """
        orderings = {}
        
        for layer_idx in range(self.num_layers):
            layer_orderings = []
            
            # 레이어별 난이도 추출
            layer_diffs = np.array([
                d.layer_diffs[layer_idx]
                for d in self.difficulties
            ])
            
            # 스테이지별로 샘플 선택
            for stage in range(self.num_stages):
                percentile_threshold = (stage + 1) / self.num_stages
                
                # Percentile 계산
                sorted_diffs = np.sort(layer_diffs)
                threshold_value = sorted_diffs[int(len(sorted_diffs) * percentile_threshold)]
                
                # Threshold 이하 샘플 선택
                mask = layer_diffs <= threshold_value
                indices = np.where(mask)[0]
                
                # 난이도 순 정렬
                selected_diffs = layer_diffs[indices]
                sorted_order = np.argsort(selected_diffs)
                ordered_indices = indices[sorted_order]
                
                layer_orderings.append(ordered_indices)
            
            orderings[layer_idx] = layer_orderings
        
        return orderings
    
    def get_samples_for_layer_and_stage(
        self,
        layer_idx: int,
        stage: int,
    ) -> np.ndarray:
        """
        특정 레이어와 스테이지에 대한 샘플 순서 반환
        """
        return self.layer_orderings[layer_idx][stage]
    
    def get_joint_samples_for_stage(
        self,
        stage: int,
    ) -> np.ndarray:
        """
        모든 레이어에 대한 Joint 학습 샘플
        
        각 레이어의 stage 샘플을 합집합
        """
        all_indices = set()
        for layer_idx in range(self.num_layers):
            layer_samples = self.layer_orderings[layer_idx][stage]
            all_indices.update(layer_samples)
        
        # 난이도 순 정렬 (전체 평균 기준)
        all_indices = list(all_indices)
        all_scores = np.array([
            self.difficulties[i].difficulty_scores['mean']
            for i in all_indices
        ])
        sorted_order = np.argsort(all_scores)
        
        return np.array(all_indices)[sorted_order]


class CurriculumDataset(Dataset):
    """
    커리큘럼 러닝을 위한 Dataset wrapper
    
    난이도 정보와 함께 샘플을 반환
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        difficulties: List,
        config: CurriculumConfig,
    ):
        self.base_dataset = base_dataset
        self.difficulties = difficulties
        self.config = config
        
        assert len(base_dataset) == len(difficulties), \
            f"Dataset size mismatch: {len(base_dataset)} vs {len(difficulties)}"
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # 기본 데이터
        sample = self.base_dataset[idx]
        
        # 난이도 정보 추가
        difficulty = self.difficulties[idx]
        sample['difficulty_info'] = {
            'sample_id': difficulty.sample_id,
            'difficulty_scores': difficulty.difficulty_scores,
            'difficulty_percentile': difficulty.difficulty_percentile,
            'curriculum_stage': difficulty.curriculum_stage,
            'layer_diffs': difficulty.layer_diffs,
        }
        
        return sample


class CurriculumTrainingScheduler:
    """
    학습 전체를 관리하는 스케줄러
    
    Epoch별로 어떤 전략을 사용할지 결정
    """
    
    def __init__(
        self,
        config: CurriculumConfig,
        total_epochs: int,
    ):
        self.config = config
        self.total_epochs = total_epochs
        
        # Epoch별 전략 스케줄
        self.epoch_strategies = self._build_epoch_schedule()
    
    def _build_epoch_schedule(self) -> Dict[int, str]:
        """
        Epoch별 전략 스케줄 생성
        
        Returns:
            {epoch: strategy_name}
        """
        schedule = {}
        
        for epoch in range(self.total_epochs):
            if epoch < self.config.warmup_epochs:
                # Warmup: 가장 쉬운 샘플만
                schedule[epoch] = "warmup"
            
            elif epoch < self.config.curriculum_start_epoch:
                # 커리큘럼 시작 전: 랜덤
                schedule[epoch] = "random"
            
            elif epoch < self.config.curriculum_end_epoch:
                # 커리큘럼 진행 중
                if self.config.strategy == "layer_wise_progressive":
                    # 스테이지 계산
                    progress = (epoch - self.config.curriculum_start_epoch) / \
                               (self.config.curriculum_end_epoch - self.config.curriculum_start_epoch)
                    stage = int(progress * self.config.num_stages)
                    schedule[epoch] = f"stage_{stage}"
                else:
                    schedule[epoch] = self.config.strategy
            
            else:
                # 커리큘럼 종료 후: 전체 데이터
                schedule[epoch] = "full"
        
        return schedule
    
    def get_sampler_for_epoch(
        self,
        epoch: int,
        difficulties: List,
        current_loss: Optional[float] = None,
        num_layers: int = 32,
    ) -> CurriculumSampler:
        """
        해당 epoch에 맞는 샘플러 생성
        """
        strategy_name = self.epoch_strategies[epoch]
        
        # Config 복사 및 수정
        config = CurriculumConfig(
            strategy=self.config.strategy,
            num_stages=self.config.num_stages,
            stage_percentiles=self.config.stage_percentiles,
            difficulty_metric=self.config.difficulty_metric,
            layer_schedule=self.config.layer_schedule,
            loss_threshold=self.config.loss_threshold,
            difficulty_step_up=self.config.difficulty_step_up,
            difficulty_step_down=self.config.difficulty_step_down,
            curriculum_start_epoch=self.config.curriculum_start_epoch,
            curriculum_end_epoch=self.config.curriculum_end_epoch,
            warmup_epochs=self.config.warmup_epochs,
        )
        
        # Warmup인 경우 첫 스테이지만
        if strategy_name == "warmup":
            config.strategy = "layer_wise_progressive"
            config.curriculum_end_epoch = self.config.warmup_epochs
        
        # Full인 경우 모든 데이터 사용
        elif strategy_name == "full":
            config.strategy = "easy_to_hard"  # 순서는 유지하되 모든 데이터
            config.curriculum_end_epoch = epoch  # 현재 epoch
        
        sampler = CurriculumSampler(
            difficulties=difficulties,
            config=config,
            current_epoch=epoch,
            current_loss=current_loss,
            num_layers=num_layers,
        )
        
        return sampler
    
    def print_schedule(self):
        """스케줄 출력"""
        print("=" * 60)
        print("Curriculum Training Schedule")
        print("=" * 60)
        for epoch, strategy in self.epoch_strategies.items():
            print(f"Epoch {epoch:3d}: {strategy}")
        print("=" * 60)
