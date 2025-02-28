import sys
import os
from pathlib import Path
from pprint import pprint

# 添加项目根目录到系统路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.dataset import Datasets

def main():
    # 1. 准备示例数据集
    sample_data = [
        {
            "instruction": "Summarize the following text.",
            "input": "Artificial intelligence has transformed many aspects of our daily lives, from virtual assistants to autonomous vehicles.",
            "output": "AI has significantly impacted everyday life through various applications."
        },
        {
            "instruction": "Translate to Spanish.",
            "input": "Hello, how are you?",
            "output": "¡Hola, cómo estás?"
        }
    ]

    # 将示例数据保存到临时文件
    import json
    dataset_path = "example_dataset.json"
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)

    try:
            #dataset="mmlu",
        # 2. 初始化数据集
        dataset = Datasets(
            model_name="Llama-2-7b-hf",  # 使用较小的模型作为示例
            dataset=dataset_path,
            batch_size=2,
            total_sample_count=24576,
            device="cuda"
        )

        # 3. 展示数据集基本信息
        print("\n=== Dataset Information ===")
        print(f"Total samples: {dataset.count}")
        print(f"Batch size: {dataset.batch_size}")
        print(f"Model name: {dataset.model_name}")

        # 4. 展示原始提示语
        print("\n=== Sample Prompts ===")
        for i, source in enumerate(dataset.sources):
            print(f"\nPrompt {i+1}:")
            print(source)
            print("-" * 50)

        # 5. 展示目标输出
        print("\n=== Sample Targets ===")
        for i, target in enumerate(dataset.targets):
            print(f"\nTarget {i+1}:")
            print(target)
            print("-" * 50)

        # 6. 展示tokenization结果
        print("\n=== Tokenization Results ===")
        print("\nInput IDs (first sample):")
        print(dataset.source_encoded_input_ids[0])
        print("\nAttention Mask (first sample):")
        print(dataset.source_encoded_attn_masks[0])

        # 7. 使用tokenizer进行解码示例
        print("\n=== Decoding Example ===")
        input_ids = dataset.source_encoded_input_ids[0].squeeze().tolist() if hasattr(dataset.source_encoded_input_ids[0], 'squeeze') else dataset.source_encoded_input_ids[0]
        decoded_text = dataset.tokenizer.decode(input_ids)
        print("Decoded text from tokens:")
        print(decoded_text)

    finally:
        # 清理临时文件
        if os.path.exists(dataset_path):
            print(os.path.abspath(dataset_path))

if __name__ == "__main__":
    main()
