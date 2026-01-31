"""
메타토큰 추출 및 차이 계산 모듈

Cloud LLM과 Edge LLM에서 레이어별 메타토큰을 추출하고,
차이를 계산하여 샘플의 난이도를 측정합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from transformers import AutoModel, AutoTokenizer
import numpy as np
from dataclasses import dataclass
import json


@dataclass
class MetaTokenDifference:
    """메타토큰 차이 정보"""
    sample_id: str
    layer_diffs: Dict[int, float]  # {layer_idx: difference}
    difficulty_scores: Dict[str, float]  # {metric_name: score}
    difficulty_percentile: float
    curriculum_stage: int


class MetaTokenExtractor:
    """
    Cloud/Edge LLM에서 레이어별 메타토큰 추출
    """
    
    def __init__(
        self,
        cloud_model_name: str = "meta-llama/Meta-Llama-3-8B",
        edge_model_name: str = "Qwen/Qwen2.5-1.5B",
        system_prompt: str = "Analyze the following dialogue and generate embeddings for task understanding.",
        device: str = "cuda",
        use_fp16: bool = True,
    ):
        self.cloud_model_name = cloud_model_name
        self.edge_model_name = edge_model_name
        self.system_prompt = system_prompt
        self.device = device
        self.use_fp16 = use_fp16
        
        print(f"Loading Cloud LLM: {cloud_model_name}")
        self.cloud_tokenizer = AutoTokenizer.from_pretrained(cloud_model_name)
        if self.cloud_tokenizer.pad_token is None:
            self.cloud_tokenizer.pad_token = self.cloud_tokenizer.eos_token
        self.cloud_model = AutoModel.from_pretrained(
            cloud_model_name,
            torch_dtype=torch.float16 if use_fp16 else torch.float32,
            device_map=device,
            trust_remote_code=True,
        )
        self.cloud_model.eval()
        for param in self.cloud_model.parameters():
            param.requires_grad = False
        print(f"  ✓ Cloud LLM loaded ({self.cloud_model.config.num_hidden_layers} layers)")
        
        print(f"Loading Edge LLM: {edge_model_name}")
        self.edge_tokenizer = AutoTokenizer.from_pretrained(edge_model_name)
        if self.edge_tokenizer.pad_token is None:
            self.edge_tokenizer.pad_token = self.edge_tokenizer.eos_token
        self.edge_model = AutoModel.from_pretrained(
            edge_model_name,
            torch_dtype=torch.float16 if use_fp16 else torch.float32,
            device_map=device,
            trust_remote_code=True,
        )
        self.edge_model.eval()
        for param in self.edge_model.parameters():
            param.requires_grad = False
        print(f"  ✓ Edge LLM loaded ({self.edge_model.config.num_hidden_layers} layers)")
        
        self.cloud_num_layers = self.cloud_model.config.num_hidden_layers
        self.edge_num_layers = self.edge_model.config.num_hidden_layers
        
        # 레이어 매칭 전략: Edge 레이어를 Cloud 레이어에 매핑
        # Edge는 레이어가 적으므로 (예: 28층), Cloud (예: 32층)에 비례 매핑
        self.layer_mapping = self._create_layer_mapping()
        
    def _create_layer_mapping(self) -> Dict[int, int]:
        """
        Edge 레이어를 Cloud 레이어에 매핑
        
        예: Edge 28층 → Cloud 32층
        Edge layer 0 → Cloud layer 0
        Edge layer 14 → Cloud layer 16
        Edge layer 27 → Cloud layer 31
        """
        mapping = {}
        for edge_idx in range(self.edge_num_layers):
            cloud_idx = int(edge_idx * self.cloud_num_layers / self.edge_num_layers)
            mapping[edge_idx] = cloud_idx
        return mapping
    
    def _format_input(self, dialogue_history: Optional[List[str]] = None, current_utterance: Optional[str] = None, full_text: Optional[str] = None) -> str:
        """대화 히스토리와 현재 발화를 포맷팅 (대화 형식 또는 reasoning task 형식)"""
        if full_text is not None:
            # Reasoning task format: use full_text directly
            return self.system_prompt + "\n\n" + full_text
        else:
            # Dialogue format: combine history + utterance
            formatted = self.system_prompt + "\n\n"
            for i, turn in enumerate(dialogue_history):
                formatted += f"Turn {i+1}: {turn}\n"
            formatted += f"Current: {current_utterance}"
            return formatted
    
    @torch.no_grad()
    def extract_meta_tokens_batch(
        self,
        texts: List[str],
        pooling: str = "mean",
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        배치로 메타토큰 추출 (속도 최적화)
        
        Args:
            texts: 입력 텍스트 리스트
            pooling: 토큰 풀링 방법
            
        Returns:
            cloud_meta_tokens_list: List of [num_cloud_layers, hidden_dim]
            edge_meta_tokens_list: List of [num_edge_layers, hidden_dim]
        """
        # Cloud LLM 배치 추론
        cloud_inputs = self.cloud_tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)
        
        cloud_outputs = self.cloud_model(
            **cloud_inputs,
            output_hidden_states=True,
        )
        
        # Edge LLM 배치 추론
        edge_inputs = self.edge_tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)
        
        edge_outputs = self.edge_model(
            **edge_inputs,
            output_hidden_states=True,
        )
        
        # 배치 내 각 샘플에 대해 메타토큰 추출
        batch_size = len(texts)
        cloud_meta_tokens_list = []
        edge_meta_tokens_list = []
        
        for b in range(batch_size):
            # Cloud 메타토큰
            cloud_tokens = []
            for layer_output in cloud_outputs.hidden_states:
                if pooling == "mean":
                    token = layer_output[b].mean(dim=0)
                elif pooling == "cls":
                    token = layer_output[b, 0, :]
                elif pooling == "last":
                    token = layer_output[b, -1, :]
                cloud_tokens.append(token.cpu())
            cloud_meta_tokens_list.append(torch.stack(cloud_tokens))
            
            # Edge 메타토큰
            edge_tokens = []
            for layer_output in edge_outputs.hidden_states:
                if pooling == "mean":
                    token = layer_output[b].mean(dim=0)
                elif pooling == "cls":
                    token = layer_output[b, 0, :]
                elif pooling == "last":
                    token = layer_output[b, -1, :]
                edge_tokens.append(token.cpu())
            edge_meta_tokens_list.append(torch.stack(edge_tokens))
        
        return cloud_meta_tokens_list, edge_meta_tokens_list
    
    @torch.no_grad()
    def extract_meta_tokens(
        self,
        dialogue_history: Optional[List[str]] = None,
        current_utterance: Optional[str] = None,
        full_text: Optional[str] = None,
        pooling: str = "mean",  # mean, cls, last
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cloud/Edge LLM에서 메타토큰 추출
        
        Args:
            dialogue_history: 대화 히스토리 (대화 형식)
            current_utterance: 현재 발화 (대화 형식)
            full_text: 전체 텍스트 (reasoning task 형식)
            pooling: 토큰 풀링 방법 (mean/cls/last)
            
        Returns:
            cloud_meta_tokens: [num_cloud_layers, hidden_dim]
            edge_meta_tokens: [num_edge_layers, hidden_dim]
        """
        # 입력 포맷팅
        input_text = self._format_input(dialogue_history, current_utterance, full_text)
        
        # Cloud LLM 추론
        cloud_inputs = self.cloud_tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)
        
        cloud_outputs = self.cloud_model(
            **cloud_inputs,
            output_hidden_states=True,
        )
        
        # Edge LLM 추론
        edge_inputs = self.edge_tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)
        
        edge_outputs = self.edge_model(
            **edge_inputs,
            output_hidden_states=True,
        )
        
        # 메타토큰 추출 (각 레이어에서)
        cloud_meta_tokens = []
        for layer_output in cloud_outputs.hidden_states:
            if pooling == "mean":
                token = layer_output.mean(dim=1).squeeze(0)  # [hidden_dim]
            elif pooling == "cls":
                token = layer_output[:, 0, :].squeeze(0)
            elif pooling == "last":
                token = layer_output[:, -1, :].squeeze(0)
            cloud_meta_tokens.append(token.cpu())
        
        edge_meta_tokens = []
        for layer_output in edge_outputs.hidden_states:
            if pooling == "mean":
                token = layer_output.mean(dim=1).squeeze(0)
            elif pooling == "cls":
                token = layer_output[:, 0, :].squeeze(0)
            elif pooling == "last":
                token = layer_output[:, -1, :].squeeze(0)
            edge_meta_tokens.append(token.cpu())
        
        cloud_meta_tokens = torch.stack(cloud_meta_tokens)  # [num_layers, dim]
        edge_meta_tokens = torch.stack(edge_meta_tokens)
        
        return cloud_meta_tokens, edge_meta_tokens
    
    def compute_layer_differences(
        self,
        cloud_meta_tokens: torch.Tensor,
        edge_meta_tokens: torch.Tensor,
        metric: str = "l2",  # l2, cosine, kl
    ) -> Dict[int, float]:
        """
        레이어별 메타토큰 차이 계산
        
        Args:
            cloud_meta_tokens: [num_cloud_layers, dim]
            edge_meta_tokens: [num_edge_layers, dim]
            metric: 차이 메트릭 (l2/cosine/kl)
            
        Returns:
            layer_diffs: {edge_layer_idx: difference}
        """
        layer_diffs = {}
        
        for edge_idx in range(self.edge_num_layers):
            cloud_idx = self.layer_mapping[edge_idx]
            
            edge_token = edge_meta_tokens[edge_idx]
            cloud_token = cloud_meta_tokens[cloud_idx]
            
            # 차원 맞추기 (필요 시 projection)
            if edge_token.shape != cloud_token.shape:
                # 더 작은 차원에 맞춤
                min_dim = min(edge_token.shape[0], cloud_token.shape[0])
                edge_token = edge_token[:min_dim]
                cloud_token = cloud_token[:min_dim]
            
            if metric == "l2":
                diff = torch.norm(cloud_token - edge_token, p=2).item()
            elif metric == "cosine":
                cos_sim = F.cosine_similarity(
                    cloud_token.unsqueeze(0),
                    edge_token.unsqueeze(0),
                    dim=1
                ).item()
                diff = 1 - cos_sim  # distance
            elif metric == "kl":
                # Softmax + KL divergence
                cloud_prob = F.softmax(cloud_token, dim=0)
                edge_prob = F.softmax(edge_token, dim=0)
                diff = F.kl_div(
                    edge_prob.log(),
                    cloud_prob,
                    reduction='batchmean'
                ).item()
            
            layer_diffs[edge_idx] = diff
        
        return layer_diffs


class DifficultyScorer:
    """
    메타토큰 차이를 종합하여 난이도 점수 계산
    """
    
    def __init__(
        self,
        num_layers: int,
        top_k: int = 3,
        layer_weights: Optional[np.ndarray] = None,
        tau: float = 5.0,  # exponential weight decay
    ):
        self.num_layers = num_layers
        self.top_k = top_k
        
        # 레이어별 가중치 (초기 레이어 중요도 높음)
        if layer_weights is None:
            self.layer_weights = np.exp(-np.arange(num_layers) / tau)
            self.layer_weights /= self.layer_weights.sum()
        else:
            self.layer_weights = layer_weights
    
    def compute_difficulty_scores(
        self,
        layer_diffs: Dict[int, float],
    ) -> Dict[str, float]:
        """
        여러 난이도 메트릭 계산
        
        Returns:
            {
                'mean': 평균 차이,
                'max': 최대 차이,
                'topk_mean': Top-K 평균,
                'weighted_mean': 가중 평균,
                'std': 표준편차,
            }
        """
        diffs = np.array([layer_diffs[i] for i in range(self.num_layers)])
        
        scores = {
            'mean': float(diffs.mean()),
            'max': float(diffs.max()),
            'topk_mean': float(np.sort(diffs)[-self.top_k:].mean()),
            'weighted_mean': float((diffs * self.layer_weights).sum()),
            'std': float(diffs.std()),
            'median': float(np.median(diffs)),
        }
        
        return scores
    
    def assign_percentile(
        self,
        difficulty_score: float,
        all_scores: List[float],
    ) -> float:
        """
        전체 샘플 중 현재 샘플의 난이도 백분위 계산
        
        Returns:
            percentile: 0.0 ~ 1.0
        """
        all_scores_sorted = np.sort(all_scores)
        percentile = np.searchsorted(all_scores_sorted, difficulty_score) / len(all_scores)
        return float(percentile)
    
    def assign_curriculum_stage(
        self,
        percentile: float,
        num_stages: int = 3,
    ) -> int:
        """
        백분위에 따라 커리큘럼 스테이지 할당
        
        Args:
            percentile: 0.0 ~ 1.0
            num_stages: 스테이지 수
            
        Returns:
            stage: 0 ~ (num_stages-1)
        """
        stage = int(percentile * num_stages)
        return min(stage, num_stages - 1)


class MetaTokenDifferenceAnalyzer:
    """
    전체 데이터셋에 대한 메타토큰 차이 분석 및 난이도 레이블링
    """
    
    def __init__(
        self,
        extractor: MetaTokenExtractor,
        scorer: DifficultyScorer,
        metric: str = "l2",
        primary_difficulty_metric: str = "topk_mean",
    ):
        self.extractor = extractor
        self.scorer = scorer
        self.metric = metric
        self.primary_difficulty_metric = primary_difficulty_metric
    
    def analyze_sample(
        self,
        sample_id: str,
        dialogue_history: Optional[List[str]] = None,
        current_utterance: Optional[str] = None,
        full_text: Optional[str] = None,
    ) -> MetaTokenDifference:
        """
        단일 샘플 분석
        """
        # 메타토큰 추출
        cloud_tokens, edge_tokens = self.extractor.extract_meta_tokens(
            dialogue_history=dialogue_history,
            current_utterance=current_utterance,
            full_text=full_text,
        )
        
        # 레이어별 차이 계산
        layer_diffs = self.extractor.compute_layer_differences(
            cloud_tokens,
            edge_tokens,
            metric=self.metric,
        )
        
        # 난이도 점수 계산
        difficulty_scores = self.scorer.compute_difficulty_scores(layer_diffs)
        
        # 임시 백분위 (나중에 전체 데이터셋 보고 재계산)
        difficulty_percentile = 0.5  # placeholder
        curriculum_stage = 0  # placeholder
        
        return MetaTokenDifference(
            sample_id=sample_id,
            layer_diffs=layer_diffs,
            difficulty_scores=difficulty_scores,
            difficulty_percentile=difficulty_percentile,
            curriculum_stage=curriculum_stage,
        )
    
    def analyze_dataset(
        self,
        dataset: List[Dict],
        num_stages: int = 3,
        save_path: Optional[str] = None,
        batch_size: int = 32,
    ) -> List[MetaTokenDifference]:
        """
        전체 데이터셋 분석 (배치 처리로 속도 최적화)
        
        Args:
            dataset: [{'id': ..., 'history': [...], 'utterance': ...}, ...] (대화 형식)
                     또는 [{'id': ..., 'full_text': ...}, ...] (reasoning task 형식)
            num_stages: 커리큘럼 스테이지 수
            save_path: 결과 저장 경로
            batch_size: 배치 크기 (GPU 메모리에 따라 조정)
            
        Returns:
            difficulties: 각 샘플의 난이도 정보
        """
        print(f"Analyzing {len(dataset)} samples with batch_size={batch_size}...")
        
        # 중간 저장 경로 설정
        checkpoint_path = save_path.replace('.json', '_checkpoint.json') if save_path else None
        
        # 1단계: 배치 단위로 모든 샘플 분석
        difficulties = []
        num_batches = (len(dataset) + batch_size - 1) // batch_size
        save_interval = 100  # 100 배치마다 저장 (약 6,400 샘플마다)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(dataset))
            batch = dataset[start_idx:end_idx]
            
            if batch_idx % 10 == 0:
                print(f"  Progress: {start_idx}/{len(dataset)} ({start_idx/len(dataset)*100:.1f}%)")
            
            # 배치 내 텍스트 추출
            batch_texts = []
            batch_ids = []
            for sample in batch:
                if 'full_text' in sample:
                    text = self.extractor._format_input(full_text=sample['full_text'])
                else:
                    text = self.extractor._format_input(
                        dialogue_history=sample['history'],
                        current_utterance=sample['utterance']
                    )
                batch_texts.append(text)
                batch_ids.append(sample['id'])
            
            # 배치 메타토큰 추출
            cloud_tokens_list, edge_tokens_list = self.extractor.extract_meta_tokens_batch(batch_texts)
            
            # 각 샘플의 난이도 계산
            for i, sample_id in enumerate(batch_ids):
                # 레이어별 차이 계산
                layer_diffs = self.extractor.compute_layer_differences(
                    cloud_tokens_list[i],
                    edge_tokens_list[i],
                    metric=self.metric,
                )
                
                # 난이도 점수 계산
                difficulty_scores = self.scorer.compute_difficulty_scores(layer_diffs)
                
                difficulties.append(MetaTokenDifference(
                    sample_id=sample_id,
                    layer_diffs=layer_diffs,
                    difficulty_scores=difficulty_scores,
                    difficulty_percentile=0.5,  # placeholder
                    curriculum_stage=0,  # placeholder
                ))
            
            # 중간 저장 (100 배치마다)
            if checkpoint_path and (batch_idx + 1) % save_interval == 0:
                print(f"  💾 Checkpoint: Saving {len(difficulties)} samples...")
                self._save_difficulties(difficulties, checkpoint_path)
                print(f"  ✓ Checkpoint saved to {checkpoint_path}")
        
        # 2단계: 백분위 계산
        all_scores = [
            d.difficulty_scores[self.primary_difficulty_metric]
            for d in difficulties
        ]
        
        for diff in difficulties:
            score = diff.difficulty_scores[self.primary_difficulty_metric]
            diff.difficulty_percentile = self.scorer.assign_percentile(score, all_scores)
            diff.curriculum_stage = self.scorer.assign_curriculum_stage(
                diff.difficulty_percentile,
                num_stages,
            )
        
        # 3단계: 저장
        if save_path:
            self._save_difficulties(difficulties, save_path)
            print(f"✓ Saved to {save_path}")
        
        return difficulties
    
    def _save_difficulties(
        self,
        difficulties: List[MetaTokenDifference],
        save_path: str,
    ):
        """난이도 정보를 JSON으로 저장"""
        data = []
        for diff in difficulties:
            data.append({
                'sample_id': diff.sample_id,
                'layer_diffs': diff.layer_diffs,
                'difficulty_scores': diff.difficulty_scores,
                'difficulty_percentile': diff.difficulty_percentile,
                'curriculum_stage': diff.curriculum_stage,
            })
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def load_difficulties(load_path: str) -> List[MetaTokenDifference]:
        """저장된 난이도 정보 로드"""
        with open(load_path, 'r') as f:
            data = json.load(f)
        
        difficulties = []
        for item in data:
            difficulties.append(MetaTokenDifference(
                sample_id=item['sample_id'],
                layer_diffs={int(k): v for k, v in item['layer_diffs'].items()},
                difficulty_scores=item['difficulty_scores'],
                difficulty_percentile=item['difficulty_percentile'],
                curriculum_stage=item['curriculum_stage'],
            ))
        
        return difficulties
