"""
단일 실험 실행 (병렬 실행용)
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
        text = item['full_text']
        
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--curriculum_order', action='store_true')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-1.5B')
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Experiment: {args.experiment_name}")
    print(f"Curriculum Order: {args.curriculum_order}")
    print(f"{'='*60}\n")
    
    # 데이터 로드
    with open(args.data_path, 'r') as f:
        data = json.load(f)
    
    print(f"Total samples: {len(data)}")
    
    # 먼저 Train/Val 분할 (랜덤하게)
    random.seed(42)
    random.shuffle(data)
    train_size = int(0.9 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    # 그 다음 Train만 정렬 (커리큘럼 적용)
    if args.curriculum_order:
        train_data.sort(key=lambda x: x['difficulty_percentile'])
        print(f"Train sorted by difficulty (easy to hard)")
    else:
        random.shuffle(train_data)
        print(f"Train random shuffled")
    
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
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    print(f"Trainable parameters: {model.num_parameters(only_trainable=True):,}")
    
    # Dataset
    train_dataset = ReasoningTaskDataset(train_data, tokenizer)
    val_dataset = ReasoningTaskDataset(val_data, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # 학습
    results = {
        'experiment_name': args.experiment_name,
        'curriculum_order': args.curriculum_order,
        'num_samples': len(data),
        'train_losses': [],
        'val_losses': [],
    }
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Train
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            labels = batch['labels'].to(args.device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
        
        train_loss = total_loss / len(train_loader)
        
        # Eval
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(args.device)
                attention_mask = batch['attention_mask'].to(args.device)
                labels = batch['labels'].to(args.device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += outputs.loss.item()
        
        val_loss = total_loss / len(val_loader)
        
        results['train_losses'].append(train_loss)
        results['val_losses'].append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # 저장
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, f'{args.experiment_name}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_path}")
    print(f"Final Val Loss: {results['val_losses'][-1]:.4f}")


if __name__ == '__main__':
    main()
