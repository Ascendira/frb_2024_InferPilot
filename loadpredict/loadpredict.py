import torch
from typing import Tuple, Any, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from pathlib import Path
import json

@dataclass
class Request:
    """请求对象"""
    text: str
    input_tokens: int = 0
    predicted_output_tokens: int = 0

@dataclass(order=True)
class PrioritizedRequest:
    """带优先级的请求对象"""
    priority: int
    category: str = None
    request: Request = None

class TokenPredictor:
    def __init__(self, model_path: str, thresholds_path: Optional[str] = None):
        """
        初始化Token预测器

        Args:
            model_path: 预训练模型路径
            thresholds_path: 阈值配置文件路径（可选）
        """
        # 加载模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

        # 加载或设置默认阈值
        self.thresholds = self._load_thresholds(thresholds_path)

        # 优先级映射
        self.priority_map = {
            'SS': 1, 'MS': 2, 'LS': 3,
            'SM': 4, 'MM': 5, 'LM': 6,
            'SL': 7, 'ML': 8, 'LL': 9
        }

    def _load_thresholds(self, thresholds_path: Optional[str]) -> dict:
        """加载阈值配置"""
        if thresholds_path and Path(thresholds_path).exists():
            with open(thresholds_path, 'r') as f:
                return json.load(f)
        else:
            # 使用默认阈值
            return {
                'input': {
                    'S': 256,   # Short
                    'M': 1024,  # Medium
                    'L': 8192   # Long
                },
                'output': {
                    'S': 100,   # Short
                    'M': 350,   # Medium
                    'L': 1000   # Long
                }
            }

    def predict_output_tokens(self, text: str) -> int:
        """
        预测输出的token数量

        Args:
            text: 输入文本

        Returns:
            int: 预测的输出token数量
        """
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.thresholds['input']['L']
            )
            outputs = self.model(**inputs)
            # 假设模型输出是token数量的预测值
            predicted_tokens = int(outputs.logits.item())
            return predicted_tokens

    def get_category(self, input_tokens: int, output_tokens: int) -> Tuple[str, str]:
        """
        获取输入和输出的长度类别

        Args:
            input_tokens: 输入token数量
            output_tokens: 输出token数量

        Returns:
            Tuple[str, str]: (输入类别, 输出类别)
        """
        # 确定输入类别
        if input_tokens < self.thresholds['input']['S']:
            input_category = 'S'
        elif input_tokens < self.thresholds['input']['M']:
            input_category = 'M'
        elif input_tokens <= self.thresholds['input']['L']:
            input_category = 'L'
        else:
            raise ValueError(f"输入tokens数量 {input_tokens} 超出范围")

        # 确定输出类别
        if output_tokens < self.thresholds['output']['S']:
            output_category = 'S'
        elif output_tokens < self.thresholds['output']['M']:
            output_category = 'M'
        elif output_tokens <= self.thresholds['output']['L']:
            output_category = 'L'
        else:
            raise ValueError(f"预测输出tokens数量 {output_tokens} 超出范围")

        return input_category, output_category

    def process_request(self, text: str) -> PrioritizedRequest:
        """
        处理请求，返回带优先级的请求对象

        Args:
            text: 输入文本

        Returns:
            PrioritizedRequest: 带优先级的请求对象
        """
        # 计算输入tokens
        input_tokens = len(self.tokenizer.encode(text))

        # 预测输出tokens
        predicted_output_tokens = self.predict_output_tokens(text)

        # 获取类别
        input_category, output_category = self.get_category(
            input_tokens, predicted_output_tokens
        )

        # 创建请求对象
        request = Request(
            text=text,
            input_tokens=input_tokens,
            predicted_output_tokens=predicted_output_tokens
        )

        # 确定类别和优先级
        category = input_category + output_category
        priority = self.priority_map[category]

        return PrioritizedRequest(
            priority=priority,
            category=category,
            request=request
        )

def calculate_thresholds(dataset_path: str, n_samples: int = 100) -> dict:
    """
    计算数据集的阈值

    Args:
        dataset_path: 数据集路径
        n_samples: 样本数量

    Returns:
        dict: 阈值配置
    """
    input_lengths = []
    output_lengths = []

    # 读取数据集
    # TODO: 根据实际数据集格式修改读取逻辑
    with open(dataset_path, 'r') as f:
        data = json.load(f)
        for item in data[:n_samples]:
            input_lengths.append(len(item['input']))
            output_lengths.append(len(item['output']))

    # 计算百分位数
    percentiles = [33, 66, 100]
    input_thresholds = np.percentile(input_lengths, percentiles)
    output_thresholds = np.percentile(output_lengths, percentiles)

    return {
        'input': {
            'S': int(input_thresholds[0]),
            'M': int(input_thresholds[1]),
            'L': int(input_thresholds[2])
        },
        'output': {
            'S': int(output_thresholds[0]),
            'M': int(output_thresholds[1]),
            'L': int(output_thresholds[2])
        }
    }

# 使用示例
def main():
    # 1. 首先计算阈值（如果需要）
    # thresholds = calculate_thresholds('path/to/dataset.json')
    # with open('thresholds.json', 'w') as f:
    #     json.dump(thresholds, f, indent=2)

    # 2. 初始化预测器
    predictor = TokenPredictor(
        model_path="your_model_path",
        thresholds_path="thresholds.json"  # 可选
    )

    # 3. 处理示例请求
    test_texts = [
        "这是一个短文本。",
        "这是一个中等长度的文本。" * 10,
        "这是一个很长的文本。" * 100
    ]

    for text in test_texts:
        try:
            result = predictor.process_request(text)
            print(f"文本长度: {len(text)}")
            print(f"类别: {result.category}")
            print(f"优先级: {result.priority}")
            print(f"输入tokens: {result.request.input_tokens}")
            print(f"预测输出tokens: {result.request.predicted_output_tokens}")
            print("-" * 50)
        except ValueError as e:
            print(f"处理错误: {e}")

if __name__ == "__main__":
    main()
