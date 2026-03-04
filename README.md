# Agentic retrieval-augmented reasoning improves cross-model robustness but preserves coordinated error in radiology question answering

This is the official repository of the paper *Agentic retrieval-augmented reasoning improves cross-model robustness but preserves coordinated error in radiology question answering*.

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

## Quickstart
1. Prepare the 169 multiple-choice radiology questions drawn from the Benchmark-RadQA and Board-RadQA datasets.
2. **Zero-shot inference:** Run the models by providing only the question stem and answer options.
3. **Agentic inference:** Launch the `RaR` orchestration pipeline using the SearXNG metasearch engine (restricted to Radiopaedia.org) to generate structured evidence reports. Pass these reports alongside the questions to the models.
4. Execute the statistical analysis scripts in the `stability` repository to compute inter-model decision stability, majority decision behavior, and robustness of correctness.

## Citation
In case you use this repository, please cite the original paper:

**BibTex**
```bibtex
@article{stability-2026,
  title={Agentic retrieval-augmented reasoning improves cross-model robustness but preserves coordinated error in radiology question answering},
  author={Farajiamiri, Mina and Sopa, Jeta and Afza, Saba and Adams, Lisa and Barajas Ordonez, Felix and Nguyen, Tri-Thien and Lotfinia, Mahshad and Wind, Sebastian and Bressem, Keno and Nebelung, Sven and Truhn, Daniel and Tayebi Arasteh, Soroosh},
  year={2026}
}
