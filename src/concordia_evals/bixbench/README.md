# BixBench

This directory contains an [Inspect](https://inspect.ai-safety-institute.org.uk/)-native implementation of [BixBench](https://arxiv.org/abs/2502.09839), a benchmark for evaluating language model agents on bioinformatics data analysis tasks. Agents are given biological datasets and must create Jupyter notebooks to analyze the data and answer open-ended research questions.

- **Paper:** <https://arxiv.org/abs/2502.09839>
- **Official code:** <https://github.com/Future-House/BixBench>
- **Dataset:** <https://huggingface.co/datasets/futurehouse/BixBench>

## Running the evaluation

Make sure you are in the repository root and have installed dependencies with `uv sync` (or your preferred installer).

```bash
uv run inspect eval src/concordia_evals/bixbench/bixbench.py@bixbench --model openai/gpt-4o
```

You can also import the task from Python:

```python
from inspect_ai import eval
from concordia_evals.bixbench import bixbench

eval(bixbench, model="openai/gpt-4o")
```

## Parameters

- `split` (str): Dataset split to use (default: `'train'`)
- `solver` (Solver | None): Custom solver to use (default: `None`)
- `scorer` (Scorer | list[Scorer] | None): Custom scorer to use (default: `None`)
- `epochs` (int): Number of times to repeat each sample (default: `1`)
- `max_messages` (int): Maximum number of messages for the default solver (default: `80`)
- `docker_image_name` (str | None): Docker image name for the sandbox (default: `None`)
- `force_rebuild` (bool): Whether to force rebuild the Docker image (default: `False`)
- `data_dir` (Path | str | None): Directory for downloaded capsule data (default: `None`)
- `limit` (int | None): Maximum number of samples to include (default: `None`)
- `avoid_images` (bool): Whether to instruct the agent to avoid plots/images (default: `True`)
- `judge_model` (str | None): Model to use for grading open-ended answers (default: `None`)

## Docker environment

Each BixBench sample runs in an isolated Docker container based on the `jingedawang/bixbench-inspect-env:latest` image. The image is published to Docker Hub and will be pulled automatically on the first run. The container includes:

- Python 3.11 with Jupyter, pandas, numpy, scipy, scikit-learn, matplotlib, seaborn
- R 4.3.3 with tidyverse, DESeq2, clusterProfiler, and other bioinformatics packages
- Bioinformatics command-line tools: BLAST, MAFFT, IQ-TREE, FastQC, SPAdes, clipkit, phykit, metaeuk

If you prefer to build the image locally:

```bash
docker build -t jingedawang/bixbench-inspect-env:latest --platform linux/amd64 .
```

The `Dockerfile` in the repository root uses four Bioconductor annotation data packages (`GenomeInfoDbData`, `GO.db`, `HDO.db`, `org.Hs.eg.db`). These packages are **not** tracked in this repository. If you place the corresponding `.tar.gz` files in the same directory as the `Dockerfile`, the build will use them directly; otherwise the Dockerfile will download them from Bioconductor automatically (requires Docker BuildKit).

## Scoring

BixBench uses an LLM-as-a-judge scorer for open-ended questions. The grading model compares the agent's proposed answer with the correct answer and returns a binary score (0 or 1) indicating equivalence.

**Note on coverage:** This implementation only supports the **open-answer** judging mode. The original BixBench paper also defines a multiple-choice question (MCQ) judging mode using the provided distractor answers; that mode is **not** implemented here.

## Dataset

The dataset contains bioinformatics research questions with open-ended questions, ground-truth answers, distractor answers (for MCQ evaluation, not used in open mode), and associated data files in ZIP format. The full dataset has ~100 questions. Use `--limit` to evaluate on a subset.
