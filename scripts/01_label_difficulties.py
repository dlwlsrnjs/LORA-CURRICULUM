"""
데이터셋 난이도 레이블링 파이프라인

전체 학습 데이터셋에 대해 메타토큰 차이를 계산하고
난이도 레이블을 부여합니다.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
import torch
from typing import List, Dict

# Add parent directory to path to import curriculum modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from meta_token_difference import (
    MetaTokenExtractor,
    DifficultyScorer,
    MetaTokenDifferenceAnalyzer,
)


def load_dataset(data_path: str) -> List[Dict]:
    """
    학습 데이터셋 로드
    
    Supports two formats:
    1. Dialogue conversations: list of conversations with turns
    2. Pre-processed samples: list of individual samples
    """
    print(f"Loading dataset from {data_path}")
    
    if data_path.endswith('.jsonl'):
        with open(data_path, 'r') as f:
            data = [json.loads(line) for line in f]
    elif data_path.endswith('.json'):
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
        
        # Check if it's dialogue format (has 'turns' key)
        if isinstance(raw_data, list) and len(raw_data) > 0 and 'turns' in raw_data[0]:
            print("  Detected dialogue conversation format, extracting samples...")
            data = extract_samples_from_conversations(raw_data)
        else:
            data = raw_data
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    print(f"  ✓ Loaded {len(data)} samples")
    return data


def extract_samples_from_conversations(conversations: List[Dict]) -> List[Dict]:
    """
    대화 데이터에서 agent 발화 샘플 추출
    또는 추론 태스크 데이터 처리
    """
    samples = []
    sample_id = 0
    
    for conv in conversations:
        # Check if it's reasoning task format
        if 'full_text' in conv and 'question' in conv:
            # Already in sample format (reasoning tasks)
            samples.append(conv)
            continue
        
        # Otherwise, process as dialogue format
        conv_id = conv.get('conversation_id', f'conv_{sample_id}')
        turns = conv.get('turns', [])
        domain = conv.get('domain', 'unknown')
        
        for turn_idx, turn in enumerate(turns):
            # Agent 발화만 추출
            if turn.get('speaker') != 'agent':
                continue
            
            # History: 이전 턴들
            history = [t['text'] for t in turns[:turn_idx]]
            target = turn['text']
            
            # 이전 customer 턴의 emotion
            emotion = 'neutral'
            for i in range(turn_idx - 1, -1, -1):
                if turns[i].get('speaker') == 'customer':
                    emotion = turns[i].get('emotion', 'neutral')
                    break
            
            sample = {
                'id': f'{conv_id}_turn{turn_idx}',
                'history': history,
                'utterance': target,
                'domain': domain,
                'style': turn.get('style', 'neutral'),
                'emotion': emotion,
                'conversation_id': conv_id,
                'turn_index': turn_idx,
            }
            samples.append(sample)
            sample_id += 1
    
    print(f"  Processed {len(samples)} samples from {len(conversations)} items")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Label dataset with difficulty scores")
    
    # Data paths
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to training dataset (JSON/JSONL)',
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Path to save difficulty labels',
    )
    
    # Model settings
    parser.add_argument(
        '--cloud_model',
        type=str,
        default='meta-llama/Meta-Llama-3-8B',
        help='Cloud LLM model name',
    )
    parser.add_argument(
        '--edge_model',
        type=str,
        default='Qwen/Qwen2.5-1.5B',
        help='Edge LLM model name',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device for inference',
    )
    
    # Meta token settings
    parser.add_argument(
        '--system_prompt',
        type=str,
        default='Analyze the following dialogue and generate embeddings for task understanding.',
        help='System prompt for meta token generation',
    )
    parser.add_argument(
        '--pooling',
        type=str,
        default='mean',
        choices=['mean', 'cls', 'last'],
        help='Token pooling method',
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='l2',
        choices=['l2', 'cosine', 'kl'],
        help='Distance metric for meta token difference',
    )
    
    # Difficulty scoring
    parser.add_argument(
        '--top_k',
        type=int,
        default=3,
        help='Top-K layers for difficulty calculation',
    )
    parser.add_argument(
        '--difficulty_metric',
        type=str,
        default='topk_mean',
        choices=['mean', 'max', 'topk_mean', 'weighted_mean', 'median'],
        help='Primary difficulty metric',
    )
    parser.add_argument(
        '--num_stages',
        type=int,
        default=3,
        help='Number of curriculum stages',
    )
    
    # Processing
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size (currently only 1 supported)',
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to process (for debugging)',
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Load dataset
    dataset = load_dataset(args.data_path)
    if args.max_samples:
        dataset = dataset[:args.max_samples]
        print(f"  → Processing first {args.max_samples} samples only")
    
    # Initialize extractor
    print("\n" + "=" * 60)
    print("Initializing Meta Token Extractor")
    print("=" * 60)
    extractor = MetaTokenExtractor(
        cloud_model_name=args.cloud_model,
        edge_model_name=args.edge_model,
        system_prompt=args.system_prompt,
        device=args.device,
        use_fp16=True,
    )
    
    # Initialize scorer
    num_layers = extractor.edge_num_layers  # Use edge model's layer count
    scorer = DifficultyScorer(
        num_layers=num_layers,
        top_k=args.top_k,
    )
    
    # Initialize analyzer
    analyzer = MetaTokenDifferenceAnalyzer(
        extractor=extractor,
        scorer=scorer,
        metric=args.metric,
        primary_difficulty_metric=args.difficulty_metric,
    )
    
    # Analyze dataset
    print("\n" + "=" * 60)
    print("Analyzing Dataset")
    print("=" * 60)
    difficulties = analyzer.analyze_dataset(
        dataset=dataset,
        num_stages=args.num_stages,
        save_path=args.output_path,
    )
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Difficulty Statistics")
    print("=" * 60)
    
    all_scores = [d.difficulty_scores[args.difficulty_metric] for d in difficulties]
    print(f"Difficulty Metric: {args.difficulty_metric}")
    print(f"  Mean: {sum(all_scores) / len(all_scores):.4f}")
    print(f"  Std:  {(sum([(s - sum(all_scores) / len(all_scores)) ** 2 for s in all_scores]) / len(all_scores)) ** 0.5:.4f}")
    print(f"  Min:  {min(all_scores):.4f}")
    print(f"  Max:  {max(all_scores):.4f}")
    
    # Stage distribution
    print(f"\nStage Distribution:")
    for stage in range(args.num_stages):
        count = sum(1 for d in difficulties if d.curriculum_stage == stage)
        percentage = count / len(difficulties) * 100
        print(f"  Stage {stage}: {count:5d} samples ({percentage:5.1f}%)")
    
    print("\n" + "=" * 60)
    print(f"✓ Difficulty labels saved to: {args.output_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
