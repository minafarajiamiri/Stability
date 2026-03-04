# Agentic retrieval-augmented reasoning improves cross-model robustness but preserves coordinated error in radiology question answering

This is the official repository of the paper *Agentic retrieval-augmented reasoning improves cross-model robustness but preserves coordinated error in radiology question answering*.

## Methodology
The study evaluates the reliability of Large Language Models (LLMs) under model variability using a standardized agentic retrieval-augmented pipeline. The methodology compares zero-shot inference with a multi-step agentic retrieval condition across a heterogeneous panel of 34 LLMs. Each model was evaluated on 169 expert-curated, publicly available radiology questions. 

In the agentic condition, all models received identical structured evidence reports derived from curated radiology knowledge (Radiopaedia.org), synthesized through an orchestration pipeline. The pipeline utilizes multi-step evidence retrieval, automated extraction of key diagnostic concepts, and synthesis into standardized, informative, yet neutral reports to isolate how different models behave when exposed to the same structured evidence.

## Quickstart
1. Prepare the 169 multiple-choice radiology questions drawn from the Benchmark-RadQA and Board-RadQA datasets.
2. **Zero-shot inference:** Run the models by providing only the question stem and answer options.
3. **Agentic inference:** Launch the `RaR` orchestration pipeline using the SearXNG metasearch engine (restricted to Radiopaedia.org) to generate structured evidence reports. Pass these reports alongside the questions to the models.
4. Execute the statistical analysis scripts in the `stability` repository to compute inter-model decision stability, majority decision behavior, and robustness of correctness.

## Environment setup
Evaluations and statistical analyses were conducted between July 1, 2025, and February 2026. The implementation was based on Python, utilizing core scientific computing libraries including NumPy, SciPy, and scikit-learn.The agentic retrieval-augmented pipeline relied on widely used open-source frameworks, comprising LangChain (v0.3.25), LangGraph (v0.4.1), OpenAI Python SDK (v1.77.0), a locally hosted SearXNG metasearch engine, and Docker (v25.0.2). Locally hosted open-weight models were served using vLLM v0.9.0.

## Prerequisites
You can create a compatible environment using Conda and pip to install the required evaluation and agentic pipeline dependencies.

```bash
$conda create -n agentic-radqa python=3.11 -y$ conda activate agentic-radqa
$python -m pip install --upgrade pip$ python -m pip install \
  numpy \
  scipy \
  scikit-learn \
  vllm==0.9.0 \
  langchain==0.3.25 \
  langgraph==0.4.1 \
  openai==1.77.0
```

## Models evaluated
We evaluated a fixed panel of 34 heterogeneous language models. Open-weight initialization weights and models were obtained from official public repositories hosted on Hugging Face:

**Qwen family**: Qwen 2.5-0.5B, 3B, 7B, 14B, 70B, and Qwen 3-8B, 235B.

**Llama family**: Llama 3.3-8B, 70B; Llama 3-Med42-8B, 70B; Llama 4 Scout 16E.

**Mistral family**: Mistral Large, Ministral 8B.

**Gemma family**: Gemma 3-4B-it, 27B-it; Medgemma-4B-it, 27B-text-it.

**DeepSeek family**: DeepSeek-V3, R1, R1-70B.


## Code structure
The codebase is split across two primary repositories:

* `stability/` (https://github.com/minafarajiamiri/stability) — Contains the full evaluation and analysis pipeline used to compute stability (entropy), consensus, robustness, and related metrics.
* `RaR/` (https://github.com/sopajeta/RaR) — Contains the agentic retrieval-augmented orchestration pipeline implemented to generate structured evidence reports.

## Data availability
All data analyzed in this study originate from publicly available, expert-curated radiology question-answering datasets.
The Benchmark-RadQA dataset (comprising RSNA-RadioQA and ExtendedQA items) is available through the original RadioRAG publication and its associated open resources.
The Board-RadQA dataset is publicly available for research use and can be accessed as reported in the RaR study and its supplementary materials.
No new patient data were generated or used in this work.

## Code availability
All code required to reproduce the analyses in this study is publicly available.
The full evaluation and analysis pipeline used to compute stability, consensus, robustness, and related metrics is available at: https://github.com/minafarajiamiri/stability.
This repository contains scripts for data processing, metric computation, and statistical analyses, and is sufficient to reproduce the results reported in this work from model outputs.

Agentic retrieval-augmented inference in this study used a previously described orchestration pipeline.
To support transparency and reproducibility, the implementation used for generating retrieval-augmented reports is publicly available at: https://github.com/sopajeta/RaR.
In the present study, this pipeline was treated as a fixed inference component and was not modified beyond configuration for dataset inputs.
Its availability allows independent reproduction of the agentic condition. The implementation relies on widely used open-source frameworks, including:
* LangChain Open Deep Research: https://github.com/langchain-ai/deep-research
* LangChain (v0.3.25): https://github.com/langchain-ai/langchain
* LangGraph (v0.4.1): https://github.com/langchain-ai/langgraph
* OpenAI Python SDK (v1.77.0): https://platform.openai.com
* SearXNG metasearch engine: https://github.com/searxng/searxng
* Docker (v25.0.2): https://www.docker.com

Locally hosted models were run between July 1 and August 22, 2025. The following open-weight models were evaluated, with sources listed for reproducibility:
* **Qwen 2.5-0.5B**: https://huggingface.co/Qwen/Qwen2.5-0.5B
* **Qwen 2.5-3B**: https://huggingface.co/Qwen/Qwen2.5-3B
* **Qwen 2.5-7B**: https://huggingface.co/Qwen/Qwen2.5-7B
* **Qwen 2.5-14B**: https://huggingface.co/Qwen/Qwen2.5-14B
* **Qwen 2.5-70B**: https://huggingface.co/Qwen/Qwen2.5-72B
* **Qwen 3-8B**: https://huggingface.co/Qwen/Qwen3-8B
* **Qwen 3-235B**: https://huggingface.co/Qwen/Qwen3-235B-A22B
* **Llama 3.3-8B**: https://huggingface.co/meta-llama/Meta-Llama-3-8B
* **Llama 3.3-70B**: https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
* **Llama 3-Med42-70B**: https://huggingface.co/m42-health/Llama3-Med42-70B
* **Llama 3-Med42-8B**: https://huggingface.co/m42-health/Llama3-Med42-8B
* **Llama 4 Scout 16E**: https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E
* **Mistral Large**: https://huggingface.co/mistralai/Mistral-Large-Instruct-2407
* **Ministral 8B**: https://huggingface.co/mistralai/Ministral-8B-Instruct-2410
* **Gemma-3-4B-it**: https://huggingface.co/google/gemma-3-4b-it
* **Gemma-3-27B-it**: https://huggingface.co/google/gemma-3-27b-it
* **Medgemma-4B-it**: https://huggingface.co/google/medgemma-4b-it
* **Medgemma-27B-text-it**: https://huggingface.co/google/medgemma-27b-text-it
* **DeepSeek-V3**: https://huggingface.co/deepseek-ai/DeepSeek-V3
* **DeepSeek-R1**: https://huggingface.co/deepseek-ai/DeepSeek-R1
* **DeepSeek-R1-70B**: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B

These models were served using vLLM v0.9.0 (https://github.com/vllm-project/vllm). Tensor parallelism matched the number of GPUs per node; models under 3B parameters were served without tensor parallelism.

OpenAI proprietary models were accessed via official API. The versions used were:
* GPT-5.2 (2025-08-07)
* GPT-5 (2025-08-07)
* O3 (2025-04-16)
* GPT-4-Turbo (2024-04-09)
* GPT-3.5-Turbo (2024-01-25)

The OpenRouter unified API was used to access more recent models:
* google/gemini-3.1-pro-preview
* anthropic/claude-sonnet-4.6
* z-ai/glm-5
* LiquidAI/LFM2.5-1.2B-Thinking
* minimax/minimax-m2.5
* moonshotai/kimi-k2.5
* writer/palmyra-x5
* xiaomi/mimo-v2-flash

## Citation
In case you use this repository, please cite the original paper:

**BibTex**
```bibtex
@article{stability-2026,
  title={Agentic retrieval-augmented reasoning improves cross-model robustness but preserves coordinated error in radiology question answering},
  author={Farajiamiri, Mina and Sopa, Jeta and Afza, Saba and Adams, Lisa and Barajas Ordonez, Felix and Nguyen, Tri-Thien and Lotfinia, Mahshad and Wind, Sebastian and Bressem, Keno and Nebelung, Sven and Truhn, Daniel and Tayebi Arasteh, Soroosh},
  year={2026}
}
