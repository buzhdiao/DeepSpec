> 中文版说明文档，对应原文：[README.md](README.md)（结构与章节保持一致，代码、命令、路径、链接不做翻译）

# DeepSpec

DeepSpec 是一个用于训练和评测投机解码（speculative decoding）草稿模型（draft model）的全栈代码库，包含数据准备工具、草稿模型实现、训练代码和评测脚本。

## 环境

安装 Python 依赖：

```bash
python -m pip install -r requirements.txt
```

数据准备阶段还需要一个推理引擎来为重新生成目标答案提供服务；详情见 [scripts/data/README.md](./scripts/data/README.md)。

## 工作流程

按顺序执行以下阶段——每个阶段的输出是下一阶段的输入：

1. **数据准备** — 下载 prompt、重新生成目标答案、构建目标缓存（target cache）。
2. **训练** — 基于缓存的目标输出训练草稿模型。
3. **评测** — 在基准任务上度量投机解码的接受率（acceptance）。

## 数据准备

数据流水线的分步操作见 [scripts/data/README.md](./scripts/data/README.md)：

1. 下载并切分训练数据，
2. 重新生成答案，
3. 准备目标缓存（存储警告：体积可能非常大——默认 `Qwen/Qwen3-4B` 设置下约 38 TB）。

## 训练

```bash
bash scripts/train/train.sh
```

`train.sh` 启动 `train.py`，后者为每个可见 GPU 拉起一个 worker。把 `config_path` 指向 [config/](./config/) 下的任意一个配置即可选择算法和目标模型（例如 `config/dspark/dspark_qwen3_4b.py`）；完整配置列表、如何覆盖 `config_path` / `target_cache_dir`、以及如何用 `--opts` 覆盖单个配置字段，见脚本头部说明。checkpoint 写入 `~/checkpoints/<project_name>/<exp_name>/step_*`。

硬件：默认配置和脚本假定单节点 8 卡。GPU 数量更少时，调小 `CUDA_VISIBLE_DEVICES` 即可。


## 评测

```bash
bash scripts/eval/eval.sh
```

`eval.sh` 在 [eval_datasets/](./eval_datasets/) 中的投机解码基准（gsm8k、math500、aime25、humaneval、mbpp、livecodebench、mt-bench、alpaca、arena-hard-v2）上，用训练好的草稿 checkpoint 运行 `eval.py`。需要设置：

- `target_name_or_path` — 草稿模型训练时对应的目标模型（例如 `Qwen/Qwen3-4B`），
- `draft_name_or_path` — 草稿 checkpoint，例如 `~/checkpoints/deepspec/dspark_block7_qwen3_4b/step_latest`，或 [已发布 Checkpoint](#released-checkpoints) 中列出的任一 Hugging Face 仓库 ID。

### Released Checkpoints（已发布 Checkpoint）

下表 checkpoint 是[论文](https://arxiv.org/abs/2607.05147)中表 1 使用的模型。每个 checkpoint 均由其对应的目标模型在非思考（non-thinking）模式下生成的 [open-perfectblend](https://huggingface.co/datasets/mlabonne/open-perfectblend) 数据训练而来，是 [config/](./config/) 中对应训练配置的直接产物。


| 算法 | `Qwen/Qwen3-4B` | `Qwen/Qwen3-8B` | `Qwen/Qwen3-14B` | `google/gemma-4-12B-it` |
| --- | --- | --- | --- | --- |
| Eagle3 | [deepseek-ai/eagle3_qwen3_4b_ttt7](https://huggingface.co/deepseek-ai/eagle3_qwen3_4b_ttt7) | [deepseek-ai/eagle3_qwen3_8b_ttt7](https://huggingface.co/deepseek-ai/eagle3_qwen3_8b_ttt7) | [deepseek-ai/eagle3_qwen3_14b_ttt7](https://huggingface.co/deepseek-ai/eagle3_qwen3_14b_ttt7) | [deepseek-ai/eagle3_gemma4_12b_ttt7](https://huggingface.co/deepseek-ai/eagle3_gemma4_12b_ttt7) |
| DFlash | [deepseek-ai/dflash_qwen3_4b_block7](https://huggingface.co/deepseek-ai/dflash_qwen3_4b_block7) | [deepseek-ai/dflash_qwen3_8b_block7](https://huggingface.co/deepseek-ai/dflash_qwen3_8b_block7) | [deepseek-ai/dflash_qwen3_14b_block7](https://huggingface.co/deepseek-ai/dflash_qwen3_14b_block7) | [deepseek-ai/dflash_gemma4_12b_block7](https://huggingface.co/deepseek-ai/dflash_gemma4_12b_block7) |
| DSpark | [deepseek-ai/dspark_qwen3_4b_block7](https://huggingface.co/deepseek-ai/dspark_qwen3_4b_block7) | [deepseek-ai/dspark_qwen3_8b_block7](https://huggingface.co/deepseek-ai/dspark_qwen3_8b_block7) | [deepseek-ai/dspark_qwen3_14b_block7](https://huggingface.co/deepseek-ai/dspark_qwen3_14b_block7) | [deepseek-ai/dspark_gemma4_12b_block7](https://huggingface.co/deepseek-ai/dspark_gemma4_12b_block7) |

> [!IMPORTANT]
> 如果要在新论文中引用这些结果，请与仓库中的训练设置保持一致，否则对比没有意义。面向特定领域使用时，建议对草稿模型重新微调以获得更好效果——尤其是当目标模型预期以思考（thinking）模式运行时。

## 支持的算法

目前 DeepSpec 包含三个草稿模型：[DSpark](https://arxiv.org/abs/2607.05147)、[DFlash](https://arxiv.org/abs/2602.06036) 和 [Eagle3](https://arxiv.org/abs/2503.01840)。

## 许可证

DeepSpec 以 [MIT License](./LICENSE) 发布。其中包含改编自第三方项目、遵循其各自许可证的代码；完整署名信息见 [NOTICE](./NOTICE)。

## 致谢

DeepSpec 建立在多个优秀开源项目的思想与代码之上：

- [SpecForge](https://github.com/sgl-project/SpecForge)（Apache-2.0）— 整体训练框架和 Eagle3 实现；Eagle3 的建模、损失、优化器、注意力与评测代码部分改编自该项目。改编文件带有文件内署名注释，完整声明见 [NOTICE](./NOTICE)。
- [DFlash](https://github.com/z-lab/dflash)（MIT）— DFlash 草稿模型设计与训练配方。
- [Qwen3](https://github.com/QwenLM/Qwen3) 和 [Gemma](https://github.com/google-deepmind/gemma) — 本仓库支持的目标模型系列。

感谢上述项目的作者与维护者。欢迎贡献新算法。

## 引用

```bibtex
@misc{cheng2026dsparkconfidencescheduledspeculativedecoding,
      title={DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation},
      author={Xin Cheng and Xingkai Yu and Chenze Shao and Jiashi Li and Yunfan Xiong and Yi Qian and Jiaqi Zhu and Shirong Ma and Xiaokang Zhang and Jiasheng Ye and Qinyu Chen and Chengqi Deng and Jiping Yu and Damai Dai and Zhengyan Zhang and Yixuan Wei and Yixuan Tan and Wenkai Yang and Runxin Xu and Yu Wu and Zhean Xu and Xuanyu Wang and Muyang Chen and Rui Tian and Xiao Bi and Zhewen Hao and Shaoyuan Chen and Huanqi Cao and Wentao Zhang and Anyi Xu and Huishuai Zhang and Dongyan Zhao and Wenfeng Liang},
      year={2026},
      eprint={2607.05147},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2607.05147},
}
```
