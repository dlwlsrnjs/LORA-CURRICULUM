"""
PersonaChat 데이터를 메타토큰 분석용 형식으로 변환
"""
import json
from pathlib import Path
from typing import List, Dict


def convert_personachat_to_samples(data_path: str, output_path: str):
    """
    PersonaChat 대화 데이터를 개별 샘플로 변환
    
    각 대화의 각 turn을 하나의 샘플로 취급
    """
    print(f"Loading PersonaChat data from {data_path}")
    with open(data_path, 'r') as f:
        conversations = json.load(f)
    
    samples = []
    sample_id = 0
    
    for conv in conversations:
        conv_id = conv['conversation_id']
        turns = conv['turns']
        
        # Build dialogue history progressively
        history = []
        for i, turn in enumerate(turns):
            text = turn['text']
            
            # Create sample: predict current turn from history
            if i > 0:  # Skip first turn (no history)
                sample = {
                    'id': f'{conv_id}_turn{i}',
                    'conversation_id': conv_id,
                    'turn_index': i,
                    'history': history.copy(),
                    'utterance': text,
                    'domain': conv.get('domain', 'unknown'),
                    'metadata': {
                        'speaker': turn.get('speaker', 'unknown'),
                        'emotion': turn.get('emotion', None),
                        'style': turn.get('style', None),
                    }
                }
                samples.append(sample)
                sample_id += 1
            
            history.append(text)
    
    # Save
    print(f"Converted {len(conversations)} conversations to {len(samples)} samples")
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved to {output_path}")
    return samples


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input PersonaChat JSON')
    parser.add_argument('--output', required=True, help='Output samples JSON')
    args = parser.parse_args()
    
    convert_personachat_to_samples(args.input, args.output)
