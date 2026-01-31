"""
추론 태스크(Reasoning Tasks) 데이터셋 다운로드 및 통합

데이터셋:
- BoolQ: Boolean Questions
- ARC-c: AI2 Reasoning Challenge (Challenge Set)
- ARC-e: AI2 Reasoning Challenge (Easy Set)
- OpenBookQA: Open Book Question Answering
- PIQA: Physical Interaction QA
- SocialQA: Social Interaction QA
- Hellaswag: Commonsense NLI
- Winogrande: Winograd Schema Challenge
"""

import json
import os
from pathlib import Path
from typing import List, Dict
from datasets import load_dataset
from tqdm import tqdm


def format_sample(
    dataset_name: str,
    question: str,
    choices: List[str],
    answer: int,
    context: str = "",
    idx: int = 0,
) -> Dict:
    """
    통일된 형식으로 샘플 변환
    """
    # 질문 + 선택지를 하나의 텍스트로
    if choices:
        choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        full_text = f"Question: {question}\n\nChoices:\n{choices_text}"
    else:
        full_text = f"Question: {question}"
    
    if context:
        full_text = f"Context: {context}\n\n{full_text}"
    
    return {
        'id': f'{dataset_name}_{idx}',
        'dataset': dataset_name,
        'question': question,
        'context': context,
        'choices': choices,
        'answer': answer,
        'full_text': full_text,
    }


def download_boolq(max_samples: int = None) -> List[Dict]:
    """BoolQ: Boolean Questions"""
    print("\nDownloading BoolQ...")
    dataset = load_dataset("boolq", split="train")
    
    samples = []
    for idx, item in enumerate(tqdm(dataset)):
        if max_samples and idx >= max_samples:
            break
        
        sample = format_sample(
            dataset_name="boolq",
            question=item['question'],
            choices=["False", "True"],
            answer=int(item['answer']),
            context=item['passage'],
            idx=idx,
        )
        samples.append(sample)
    
    print(f"  ✓ Loaded {len(samples)} samples from BoolQ")
    return samples


def download_arc(split_name: str, max_samples: int = None) -> List[Dict]:
    """ARC: AI2 Reasoning Challenge"""
    print(f"\nDownloading ARC-{split_name}...")
    dataset = load_dataset("ai2_arc", split_name, split="train")
    
    samples = []
    for idx, item in enumerate(tqdm(dataset)):
        if max_samples and idx >= max_samples:
            break
        
        choices = item['choices']['text']
        # Convert letter labels to indices
        answer_key = item['answerKey']
        if answer_key.isdigit():
            answer = int(answer_key) - 1
        else:
            answer = ord(answer_key.upper()) - ord('A')
        
        sample = format_sample(
            dataset_name=f"arc_{split_name}",
            question=item['question'],
            choices=choices,
            answer=answer,
            idx=idx,
        )
        samples.append(sample)
    
    print(f"  ✓ Loaded {len(samples)} samples from ARC-{split_name}")
    return samples


def download_openbookqa(max_samples: int = None) -> List[Dict]:
    """OpenBookQA"""
    print("\nDownloading OpenBookQA...")
    dataset = load_dataset("openbookqa", "main", split="train")
    
    samples = []
    for idx, item in enumerate(tqdm(dataset)):
        if max_samples and idx >= max_samples:
            break
        
        choices = item['choices']['text']
        answer_key = item['answerKey']
        answer = ord(answer_key.upper()) - ord('A')
        
        sample = format_sample(
            dataset_name="openbookqa",
            question=item['question_stem'],
            choices=choices,
            answer=answer,
            idx=idx,
        )
        samples.append(sample)
    
    print(f"  ✓ Loaded {len(samples)} samples from OpenBookQA")
    return samples


def download_piqa(max_samples: int = None) -> List[Dict]:
    """PIQA: Physical Interaction QA"""
    print("\nDownloading PIQA...")
    try:
        dataset = load_dataset("ybisk/piqa", split="train", trust_remote_code=True)
    except:
        print("  ⚠ PIQA unavailable, skipping...")
        return []
    
    samples = []
    for idx, item in enumerate(tqdm(dataset)):
        if max_samples and idx >= max_samples:
            break
        
        choices = [item['sol1'], item['sol2']]
        
        sample = format_sample(
            dataset_name="piqa",
            question=item['goal'],
            choices=choices,
            answer=item['label'],
            idx=idx,
        )
        samples.append(sample)
    
    print(f"  ✓ Loaded {len(samples)} samples from PIQA")
    return samples


def download_social_iqa(max_samples: int = None) -> List[Dict]:
    """SocialQA: Social Interaction QA"""
    print("\nDownloading SocialQA...")
    try:
        dataset = load_dataset("allenai/social_i_qa", split="train", trust_remote_code=True)
    except:
        print("  ⚠ SocialQA unavailable, skipping...")
        return []
    
    samples = []
    for idx, item in enumerate(tqdm(dataset)):
        if max_samples and idx >= max_samples:
            break
        
        choices = [item['answerA'], item['answerB'], item['answerC']]
        answer = int(item['label']) - 1  # Convert 1,2,3 to 0,1,2
        
        sample = format_sample(
            dataset_name="social_iqa",
            question=item['question'],
            choices=choices,
            answer=answer,
            context=item['context'],
            idx=idx,
        )
        samples.append(sample)
    
    print(f"  ✓ Loaded {len(samples)} samples from SocialQA")
    return samples


def download_hellaswag(max_samples: int = None) -> List[Dict]:
    """Hellaswag: Commonsense NLI"""
    print("\nDownloading Hellaswag...")
    try:
        dataset = load_dataset("Rowan/hellaswag", split="train", trust_remote_code=True)
    except:
        print("  ⚠ Hellaswag unavailable, skipping...")
        return []
    
    samples = []
    for idx, item in enumerate(tqdm(dataset)):
        if max_samples and idx >= max_samples:
            break
        
        choices = item['endings']
        
        sample = format_sample(
            dataset_name="hellaswag",
            question=item['activity_label'] + ": " + item['ctx'],
            choices=choices,
            answer=int(item['label']),
            idx=idx,
        )
        samples.append(sample)
    
    print(f"  ✓ Loaded {len(samples)} samples from Hellaswag")
    return samples


def download_winogrande(max_samples: int = None) -> List[Dict]:
    """Winogrande: Winograd Schema Challenge"""
    print("\nDownloading Winogrande...")
    try:
        dataset = load_dataset("allenai/winogrande", "winogrande_xl", split="train", trust_remote_code=True)
    except:
        print("  ⚠ Winogrande unavailable, skipping...")
        return []
        return []
def download_winogrande(max_samples: int = None) -> List[Dict]:
    """Winogrande: Winograd Schema Challenge"""
    print("\nDownloading Winogrande...")
    dataset = load_dataset("winogrande", "winogrande_xl", split="train")
    
    samples = []
    for idx, item in enumerate(tqdm(dataset)):
        if max_samples and idx >= max_samples:
            break
        
        choices = [item['option1'], item['option2']]
        answer = int(item['answer']) - 1  # Convert 1,2 to 0,1
        
        sample = format_sample(
            dataset_name="winogrande",
            question=item['sentence'],
            choices=choices,
            answer=answer,
            idx=idx,
        )
        samples.append(sample)
    
    print(f"  ✓ Loaded {len(samples)} samples from Winogrande")
    return samples


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download reasoning task datasets")
    parser.add_argument('--output_dir', type=str, default='curriculum/data/reasoning_tasks',
                        help='Output directory')
    parser.add_argument('--max_samples_per_dataset', type=int, default=None,
                        help='Maximum samples per dataset (for testing)')
    parser.add_argument('--datasets', type=str, nargs='+', 
                        default=['boolq', 'arc_easy', 'arc_challenge', 'openbookqa', 
                                'piqa', 'social_iqa', 'hellaswag', 'winogrande'],
                        help='Datasets to download')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Downloading Reasoning Task Datasets")
    print("=" * 60)
    
    all_samples = []
    dataset_stats = {}
    
    # Download each dataset
    if 'boolq' in args.datasets:
        samples = download_boolq(args.max_samples_per_dataset)
        all_samples.extend(samples)
        dataset_stats['boolq'] = len(samples)
    
    if 'arc_easy' in args.datasets:
        samples = download_arc('ARC-Easy', args.max_samples_per_dataset)
        all_samples.extend(samples)
        dataset_stats['arc_easy'] = len(samples)
    
    if 'arc_challenge' in args.datasets:
        samples = download_arc('ARC-Challenge', args.max_samples_per_dataset)
        all_samples.extend(samples)
        dataset_stats['arc_challenge'] = len(samples)
    
    if 'openbookqa' in args.datasets:
        samples = download_openbookqa(args.max_samples_per_dataset)
        all_samples.extend(samples)
        dataset_stats['openbookqa'] = len(samples)
    
    if 'piqa' in args.datasets:
        samples = download_piqa(args.max_samples_per_dataset)
        all_samples.extend(samples)
        dataset_stats['piqa'] = len(samples)
    
    if 'social_iqa' in args.datasets:
        samples = download_social_iqa(args.max_samples_per_dataset)
        all_samples.extend(samples)
        dataset_stats['social_iqa'] = len(samples)
    
    if 'hellaswag' in args.datasets:
        samples = download_hellaswag(args.max_samples_per_dataset)
        all_samples.extend(samples)
        dataset_stats['hellaswag'] = len(samples)
    
    if 'winogrande' in args.datasets:
        samples = download_winogrande(args.max_samples_per_dataset)
        all_samples.extend(samples)
        dataset_stats['winogrande'] = len(samples)
    
    # Save combined dataset
    output_file = output_dir / 'reasoning_tasks_combined.json'
    with open(output_file, 'w') as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)
    
    # Save statistics
    stats_file = output_dir / 'dataset_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(dataset_stats, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Download Complete!")
    print("=" * 60)
    print(f"Total samples: {len(all_samples)}")
    print("\nDataset breakdown:")
    for dataset, count in dataset_stats.items():
        print(f"  {dataset:20s}: {count:6d} samples")
    print(f"\n✓ Saved to: {output_file}")
    print(f"✓ Stats saved to: {stats_file}")


if __name__ == '__main__':
    main()
