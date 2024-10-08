# SciLitLLM: Adapting LLMs for Scientific Literature Understanding

**SciLitLLM** adapts a general large language model for effective scientific literature understanding. This repository contains all necessary code for the continual pre-training (CPT) and supervised fine-tuning (SFT) methods, which are key components of SciLitLLM.

## ****Update****

- ****🔥 **News**: ``2024/08/26``: We have released the weights of [SciLit-LLM-7B](https://huggingface.co/Uni-SMART/SciLitLLM), and disclosed the [Arxiv paper](https://arxiv.org/pdf/2408.15545).****

## **Model List**

|         Model         | Type | Seq Length |                         Download                         |
| :--------------------: | :--: | :--------: | :-------------------------------------------------------: |
| UniSMART/SciLit-LLM-7B | Base |    128K    | [🤗 Huggingface](https://huggingface.co/Uni-SMART/SciLitLLM) |

## Overview

Scientific literature understanding is essential for extracting valuable insights and advancing scientific discovery. **SciLitLLM** specializes in this by integrating domain-specific knowledge and task-specific instruction-following abilities. The framework achieves this through:

-**Continual Pre-Training (CPT)**: Infusing domain knowledge from scientific corpora.

-**Supervised Fine-Tuning (SFT)**: Enhancing instruction-following using diverse scientific tasks.

### What is Scientific Literature Understanding?

<img src="assets/lit_understanding.png" alt="Scientific Literature Understanding" width="50%">

## Evaluation results

![Scientific Literature Understanding](assets/evaluation.png)

## SciLitLLM Pipeline

SciLitLLM utilizes a two-stage pipeline:

![SciLitLLM Framework Pipeline](assets/pipeline.png)

## Repository Structure

Please refer to each subdirectory for details.

-**cpt/**: data processing codes for CPT corpura.

-**sft/**: data processing codes for SFT instructions.

## Create Your Domian-specific Model

1. Clone the repository and setup environments:

   ```bash
   git clone https://github.com/dptech-corp/Uni-SMART.git

   cd Uni-SMART/SciLitLLM

   conda create --name scilitllm python=3.10

   conda activate scilitllm

   pip install -r requirements.txt
   ```
2. Follow the instructions in the **cpt/** and **sft/** directories to prepare training corpora.

## ****Citation****

****Please consider citing the following papers if you find our work helpful.****

```
@article{li2024scilitllm,
  title={SciLitLLM: How to Adapt LLMs for Scientific Literature Understanding},
  author={Li, Sihang and Huang, Jian and Zhuang, Jiaxi and Shi, Yaorui and Cai, Xiaochen and Xu, Mingjun and Wang, Xiang and Zhang, Linfeng and Ke, Guolin and Cai, Hengxing},
  journal={arXiv preprint arXiv:2408.15545},
  year={2024}
}
```
