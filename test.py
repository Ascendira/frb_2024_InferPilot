from transformers import AutoModelForCausalLM
import torch

try:
    print("Loading model...")
    model_path = 'D:\\Files\\Learning\\竞赛\\冯如杯\\InferPilot\\model\\Llama-2-7b-hf'
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",  # 自动分配到 GPU 或 CPU
        torch_dtype=torch.float16  # 使用半精度减少内存占用
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")