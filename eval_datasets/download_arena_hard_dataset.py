import json  # 读 question.jsonl 原始行、写转换结果
from pathlib import Path  # 路径对象：HF 缓存文件与输出文件
from huggingface_hub import hf_hub_download  # 从 HF 仓库下载单个文件到本地缓存

# arena-hard-v2 以原始 JSONL 分发（load_dataset 无法直接读），故手工下载并转换。
src = Path(hf_hub_download(  # 下载题目文件，返回本地缓存路径
    repo_id="lmarena-ai/arena-hard-auto",  # LMArena 困难提问数据集仓库
    filename="data/arena-hard-v2.0/question.jsonl",  # v2.0 版本的 question 文件
    repo_type="dataset",  # 声明下载的是 dataset 仓库而非模型仓库
))
out = Path("eval_datasets/arena-hard-v2.jsonl")  # 输出路径：须在仓库根目录运行

with src.open("r", encoding="utf-8") as f, out.open("w", encoding="utf-8") as g:  # 同时打开源文件与输出文件
    for line in f:
        row = json.loads(line)  # 逐行解析原始题目记录
        g.write(json.dumps({"turns": [row["prompt"]]}, ensure_ascii=False) + "\n")  # 取 prompt 字段包成单轮 turns 写出
