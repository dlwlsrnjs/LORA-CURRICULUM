"""
Mixed Curriculum Learning - 점진적으로 난이도 비율 조정
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

sys.path.insert(0, str(Path(__file__).parent.parent))


class ReasoningTaskDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # full_text가 있으면 사용, 없으면 sample_id로 대체
        text = item.get('full_text', item.get('sample_id', ''))
        
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


def split_by_difficulty(data):
    """난이도별로 데이터 분할 (3등분)"""
    sorted_data = sorted(data, key=lambda x: x['difficulty_percentile'])
    n = len(sorted_data)
    
    easy = sorted_data[:n//3]
    medium = sorted_data[n//3:2*n//3]
    hard = sorted_data[2*n//3:]
    
    return easy, medium, hard


def create_mixed_dataset(easy, medium, hard, easy_ratio, medium_ratio, hard_ratio):
    """비율에 따라 혼합 데이터셋 생성"""
    total = len(easy) + len(medium) + len(hard)
    
    n_easy = int(total * easy_ratio)
    n_medium = int(total * medium_ratio)
    n_hard = int(total * hard_ratio)
    
    # 샘플링
    selected_easy = random.sample(easy, min(n_easy, len(easy)))
    selected_medium = random.sample(medium, min(n_medium, len(medium)))
    selected_hard = random.sample(hard, min(n_hard, len(hard)))
    
    # 섞기
    mixed = selected_easy + selected_medium + selected_hard
    random.shuffle(mixed)
    
    return mixed


def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
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
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader)


def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Evaluating")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
    
    return total_loss / len(val_loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-1.5B')
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Mixed Curriculum Learning Experiment")
    print(f"Experiment: {args.experiment_name}")
    print(f"{'='*60}\n")
    
    # 데이터 로드
    with open(args.data_path, 'r') as f:
        data = json.load(f)
    
    print(f"Total samples: {len(data)}")
    
    # Train/Val 분할 (먼저 분할)
    random.seed(42)
    random.shuffle(data)
    train_size = int(0.9 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    # Train 데이터를 난이도별로 분할
    easy, medium, hard = split_by_difficulty(train_data)
    print(f"Train split - Easy: {len(easy)}, Medium: {len(medium)}, Hard: {len(hard)}")
    
    # 모델 로드
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map=args.device,
    )
    
    # LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    model = get_peft_model(model, lora_config)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}\n")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Mixed Curriculum 전략
    curriculum_schedule = [
        # Epoch 1: 쉬운 것 많이
        {'easy': 0.5, 'medium': 0.35, 'hard': 0.15},
        # Epoch 2: 균등하게
        {'easy': 0.33, 'medium': 0.34, 'hard': 0.33},
    ]
    
    if args.num_epochs > 2:
        # Epoch 3+: 어려운 것 많이
        for _ in range(args.num_epochs - 2):
            curriculum_schedule.append({'easy': 0.2, 'medium': 0.3, 'hard': 0.5})
    
    train_losses = []
    val_losses = []
    
    # Val 데이터셋 (고정)
    val_dataset = ReasoningTaskDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        # 이번 에폭의 커리큘럼 비율
        ratios = curriculum_schedule[min(epoch, len(curriculum_schedule) - 1)]
        print(f"Curriculum: Easy {ratios['easy']:.0%}, Medium {ratios['medium']:.0%}, Hard {ratios['hard']:.0%}")
        
        # Mixed dataset 생성
        mixed_train_data = create_mixed_dataset(
            easy, medium, hard,
            ratios['easy'], ratios['medium'], ratios['hard']
        )
        
        train_dataset = ReasoningTaskDataset(mixed_train_data, tokenizer)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        train_loss = train_epoch(model, train_loader, optimizer, args.device)
        val_loss = evaluate(model, val_loader, args.device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # 결과 저장
    results = {
        'experiment_name': args.experiment_name,
        'strategy': 'mixed_curriculum',
        'curriculum_schedule': curriculum_schedule[:args.num_epochs],
        'num_samples': len(data),
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    
    output_path = os.path.join(args.output_dir, f"{args.experiment_name}_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")
    print(f"Final Val Loss: {val_losses[-1]:.4f}")


if __name__ == "__main__":
    main()
