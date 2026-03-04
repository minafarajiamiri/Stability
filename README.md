# Agentic retrieval-augmented reasoning improves cross-model robustness but preserves coordinated error in radiology question answering

[cite_start]This is the official repository of the paper *Agentic retrieval-augmented reasoning improves cross-model robustness but preserves coordinated error in radiology question answering*[cite: 1].

## Environment setup
[cite_start]Evaluations and statistical analyses were conducted between July 1, 2025, and February 2026[cite: 394, 442, 735]. [cite_start]The implementation was based on Python, utilizing core scientific computing libraries including NumPy, SciPy, and scikit-learn[cite: 394]. [cite_start]The agentic retrieval-augmented pipeline relied on widely used open-source frameworks, comprising LangChain (v0.3.25) [cite: 437][cite_start], LangGraph (v0.4.1) [cite: 438][cite_start], OpenAI Python SDK (v1.77.0) [cite: 439][cite_start], a locally hosted SearXNG metasearch engine [cite: 440][cite_start], and Docker (v25.0.2)[cite: 441]. [cite_start]Locally hosted open-weight models were served using vLLM v0.9.0[cite: 464].

## Prerequisites
[cite_start]You can create a compatible environment using Conda and pip to install the required evaluation and agentic pipeline dependencies[cite: 394, 437, 438, 439, 464].

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

## Models evaluated
[cite_start]We evaluated a fixed panel of 34 heterogeneous language models[cite: 251]. [cite_start]Open-weight initialization weights and models were obtained from official public repositories hosted on Hugging Face[cite: 442]:

* [cite_start]**Qwen family**: Qwen 2.5-0.5B, 3B, 7B, 14B, 70B, and Qwen 3-8B, 235B[cite: 254].
* **Llama family**: Llama 3.3-8B, 70B; Llama 3-Med42-8B, 70B; [cite_start]Llama 4 Scout 16E[cite: 254].
* [cite_start]**Mistral family**: Mistral Large, Ministral 8B[cite: 254].
* **Gemma family**: Gemma 3-4B-it, 27B-it; [cite_start]Medgemma-4B-it, 27B-text-it[cite: 254, 255].
* [cite_start]**DeepSeek family**: DeepSeek-V3, R1, R1-70B[cite: 254].

## Code structure
The codebase is split across two primary repositories:

* [cite_start]`stability/` (https://github.com/minafarajiamiri/stability) — Contains the full evaluation and analysis pipeline used to compute stability (entropy), consensus, robustness, and related metrics[cite: 430, 431].
* [cite_start]`RaR/` (https://github.com/sopajeta/RaR) — Contains the agentic retrieval-augmented orchestration pipeline implemented to generate structured evidence reports[cite: 433, 434].

## Quickstart
1. [cite_start]Prepare the 169 multiple-choice radiology questions drawn from the Benchmark-RadQA and Board-RadQA datasets[cite: 226].
2. [cite_start]**Zero-shot inference:** Run the models by providing only the question stem and answer options[cite: 263].
3. [cite_start]**Agentic inference:** Launch the `RaR` orchestration pipeline using the SearXNG metasearch engine (restricted to Radiopaedia.org) to generate structured evidence reports[cite: 268, 440]. [cite_start]Pass these reports alongside the questions to the models[cite: 269].
4. [cite_start]Execute the statistical analysis scripts in the `stability` repository to compute inter-model decision stability, majority decision behavior, and robustness of correctness[cite: 430, 431].

## Citation
In case you use this repository, please cite the original paper:
Mina Farajiamiri, Jeta Sopa, et al. [cite_start]Agentic retrieval-augmented reasoning improves cross-model robustness but preserves coordinated error in radiology question answering[cite: 1, 2].

**BibTex**
```bibtex
@article{agentic-radqa,
  author = {Farajiamiri, Mina and Sopa, Jeta and others},
  title = {Agentic retrieval-augmented reasoning improves cross-model robustness but preserves coordinated error in radiology question answering},
  year = {2026}
}
