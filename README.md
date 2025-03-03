# MMBench

## 创建虚拟环境

```
conda create --name mlperf python=3.10
conda activate mlperf

```

## 安装loadgen库

```

cd loadgen
pip install -r requirements.txt
conda install -c conda-forge gcc
python3 -m pip install .

```

## 安装运行依赖

```
cd core
pip install -r requirements.txt

```


## 准备本地模型

```
# 以Llama-2-7b-hf为例,如遇到权限问题，可以找modelscope或者其他作者不需要权限的下载
# 如果用专业工具下载记得设置下载路径，如果用git下载如下
mkdir -p /root/xxx/model/
cd model
GIT_LFS_SKIP_SMUDGE=1 git clone https://hf-mirror.com/meta-llama/Llama-2-7b-hf Llama-2-7b-hf
cd Llama-2-7b-hf
git lfs pull

```

## 测试验证
```
cd examples
export CHECKPOINT_PATH="/root/xxx/model/"
python dataset_example.py #构建虚假数据测试

```

## 准备本地数据集

```
# 以mmlu为例

mkdir -p /root/xxx/dataset/
cd dataset
GIT_LFS_SKIP_SMUDGE=1 git clone https://hf-mirror.com/datasets/cais/mmlu
cd cais/mmlu
git lfs pull

```

注意：模型路径和数据集路径对照相关代码进行设置

## 测试运行

```
export CHECKPOINT_PATH="/root/xxx/model/"
export DATASET_PATH="/root/xxx/dataset/"

python -u main.py --scenario Offline \
		--dataset mmlu or --dataset /root/xxxx/mmbench/examples/example_dataset.json
		--model-name llama2-7b \
		--total-sample-count 24576 \
		--device cuda
	
export CHECKPOINT_PATH="/home/frljc/frb_2024_InferPilot/model/"
export DATASET_PATH="/home/frljc/frb_2024_InferPilot/dataset/"

python -u main.py --scenario Offline --dataset /home/frljc/frb_2024_InferPilot/examples/example_dataset.json --model-name llama2-7b --total-sample-count 24576 --device cuda
	
# windows下运行
$env:CHECKPOINT_PATH="D:/Files/Learning/竞赛/冯如杯/No_2/Project/mmbench/model/"
$env:DATASET_PATH="D:/Files/Learning/竞赛/冯如杯/No_2/Project/mmbench/dataset/"
                  
python -u main.py --scenario Offline `
                  --dataset D:/Files/Learning/竞赛/冯如杯/No_2/Project/mmbench/examples/example_dataset.json `
                  --model-name llama2-7b `
                  --total-sample-count 24576 `
                  --device cuda
                  
--scenario Offline --dataset D:/Files/Learning/竞赛/冯如杯/No_2/Project/mmbench/examples/example_dataset.json --model-name llama2-7b --total-sample-count 24576 --device cuda

```

