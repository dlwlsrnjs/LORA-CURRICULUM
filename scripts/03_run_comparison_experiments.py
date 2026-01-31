"""
커리큘럼 러닝 vs 랜덤 순서 비교 실험

두 가지 방식으로 학습하고 성능을 비교합니다:
1. Baseline: 랜덤 순서로 학습
2. Curriculum: 쉬운 것부터 어려운 것 순서로 학습
"""

import argparse
import json
import os
import sys
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class ReasoningTaskDataset(Dataset):
    """추론 태스크 데이터셋"""
    
    def __init__(self, data_path, tokenizer, max_length=512, shuffle=False, curriculum_order=False):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 순서 설정
        if curriculum_order:
            # 난이도 순으로 정렬 (쉬운 것부터)
            self.data.sort(key=lambda x: x['difficulty_percentile'])
            print(f"  Curriculum order: sorted by difficulty (easy to hard)")
        elif shuffle:
            # 랜덤 셔플
            random.shuffle(self.data)
            print(f"  Random order: shuffled")
        
        print(f"  Loaded {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 원본 데이터에서 텍스트 가져오기
        # reasoning_tasks_combined.json에서 로드한 경우
        if 'full_text' in item:
            text = item['full_text']
        else:
            # difficulty 파일에서 로드한 경우, sample_id로 원본 찾기
            text = item.get('text', '')
        
        # 토크나이징
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze(),
        }


def load_combined_dataset(difficulty_path, original_path):
    """난이도 레이블과 원본 텍스트를 결합"""
    with open(difficulty_path, 'r') as f:
        difficulty_data = json.load(f)
    
    with open(original_path, 'r') as f:
        original_data = json.load(f)
    
    # sample_id로 매핑
    original_map = {item['id']: item for item in original_data}
    
    # 결합
    combined = []
    for diff_item in difficulty_data:
        sample_id = diff_item['sample_id']
        if sample_id in original_map:
            orig_item = original_map[sample_id]
            combined_item = {
                'sample_id': sample_id,
                'full_text': orig_item['full_text'],
                'difficulty_percentile': diff_item['difficulty_percentile'],
                'curriculum_stage': diff_item['curriculum_stage'],
                'difficulty_scores': diff_item['difficulty_scores'],
            }
            combined.append(combined_item)
    
    return combined


def train_epoch(model, dataloader, optimizer, device):
    """1 에폭 학습"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        num_batches += 1
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate(model, dataloader, device):
    """평가"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


def run_experiment(
    model_name,
    data_combined,
    experiment_name,
    curriculum_order,
    num_epochs,
    batch_size,
    learning_rate,
    output_dir,
    max_samples=None,
    device='cuda'
):
    """단일 실험 실행"""
    print(f"\n{'='*60}")
    print(f"Running Experiment: {experiment_name}")
    print(f"{'='*60}")
    
    # 데이터 서브샘플링 (빠른 테스트용)
    if max_samples:
        data_combined = data_combined[:max_samples]
        print(f"Using {max_samples} samples for quick experiment")
    
    # 데이터를 파일로 저장
    temp_data_path = os.path.join(output_dir, f'{experiment_name}_data.json')
    with open(temp_data_path, 'w') as f:
        json.dump(data_combined, f)
    
    # 모델 및 토크나이저 로드
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    
    # LoRA 설정
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 데이터셋 준비
    train_size = int(0.9 * len(data_combined))
    train_data = data_combined[:train_size]
    val_data = data_combined[train_size:]
    
    # 임시 파일로 저장
    train_path = os.path.join(output_dir, f'{experiment_name}_train.json')
    val_path = os.path.join(output_dir, f'{experiment_name}_val.json')
    with open(train_path, 'w') as f:
        json.dump(train_data, f)
    with open(val_path, 'w') as f:
        json.dump(val_data, f)
    
    train_dataset = ReasoningTaskDataset(
        train_path, tokenizer, 
        shuffle=(not curriculum_order),
        curriculum_order=curriculum_order
    )
    val_dataset = ReasoningTaskDataset(val_path, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # 학습
    results = {
        'experiment_name': experiment_name,
        'curriculum_order': curriculum_order,
        'num_samples': len(data_combined),
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'train_losses': [],
        'val_losses': [],
        'epoch_times': [],
    }
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        
        epoch_time = time.time() - start_time
        
        results['train_losses'].append(train_loss)
        results['val_losses'].append(val_loss)
        results['epoch_times'].append(epoch_time)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
    
    # 결과 저장
    results_path = os.path.join(output_dir, f'{experiment_name}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_path}")
    
    # 정리
    os.remove(train_path)
    os.remove(val_path)
    os.remove(temp_data_path)
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--difficulty_path', type=str, required=True)
    parser.add_argument('--original_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-1.5B')
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--max_samples', type=int, default=1000, help='Number of samples for quick test')
    parser.add_argument('--output_dir', type=str, default='curriculum/experiments')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # 시드 설정
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 데이터 로드
    print("Loading datasets...")
    data_combined = load_combined_dataset(args.difficulty_path, args.original_path)
    print(f"Loaded {len(data_combined)} samples with difficulty labels")
    
    # 실험 1: Baseline (Random Order)
    baseline_results = run_experiment(
        model_name=args.model_name,
        data_combined=data_combined.copy(),
        experiment_name='baseline_random',
        curriculum_order=False,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        device=args.device,
    )
    
    # 실험 2: Curriculum Learning (Easy to Hard)
    curriculum_results = run_experiment(
        model_name=args.model_name,
        data_combined=data_combined.copy(),
        experiment_name='curriculum_easy_to_hard',
        curriculum_order=True,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        device=args.device,
    )
    
    # 비교 결과
    print("\n" + "="*60)
    print("Comparison Results")
    print("="*60)
    
    print(f"\nBaseline (Random Order):")
    print(f"  Final Train Loss: {baseline_results['train_losses'][-1]:.4f}")
    print(f"  Final Val Loss: {baseline_results['val_losses'][-1]:.4f}")
    
    print(f"\nCurriculum (Easy to Hard):")
    print(f"  Final Train Loss: {curriculum_results['train_losses'][-1]:.4f}")
    print(f"  Final Val Loss: {curriculum_results['val_losses'][-1]:.4f}")
    
    improvement = (baseline_results['val_losses'][-1] - curriculum_results['val_losses'][-1]) / baseline_results['val_losses'][-1] * 100
    print(f"\nImprovement: {improvement:.2f}%")
    
    # 전체 비교 저장
    comparison = {
        'baseline': baseline_results,
        'curriculum': curriculum_results,
        'improvement_percent': improvement,
    }
    
    comparison_path = os.path.join(args.output_dir, 'comparison_results.json')
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n✓ Comparison saved to: {comparison_path}")


if __name__ == '__main__':
    main()
