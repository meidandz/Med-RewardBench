# Med-RewardBench

Med-RewardBench is a vision-language benchmark pipeline for evaluating medical preference and pairwise judgment quality across multiple imaging modalities.

This repository is prepared for GitHub release without bundling the benchmark data. Dataset files should be downloaded separately from Hugging Face and placed under the expected local directory structure.

## Overview

- Evaluate pairwise medical VLM responses across multiple modalities.
- Load benchmark data from local `jsonl` and `parquet` files.
- Export summary tables to Excel, CSV, and JSONL.
- Support both public Hugging Face models and local fine-tuned checkpoints.

## Repository Structure

```text
Med-RewardBench/
|- data/
|  `- README.md
|- vlm/
|  |- 00_benchmark_full_pipeline.py
|  |- model_builder_chosen_respon.py
|  `- prompt.py
|- .gitignore
|- README.md
`- requirements.txt
```

## Setup

1. Create a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Supported Inputs and Outputs

- Input data: modality-specific `jsonl` ground truth files and `parquet` benchmark files under `data/pair_gt_all/`
- Main script: `vlm/00_benchmark_full_pipeline.py`
- Output files: Excel summary plus optional CSV and JSONL tables

## Data Preparation

The benchmark data is **not included** in this repository.

Please download the dataset from Hugging Face and arrange it as:

```text
data/
`- pair_gt_all/
   |- abdomen.jsonl
   |- brain.jsonl
   |- ...
   `- parquet/
      |- abdomen.parquet
      |- brain.parquet
      `- ...
```

Hugging Face dataset link:

- [Nandzy/Med-RewardBench](https://huggingface.co/datasets/Nandzy/Med-RewardBench/tree/main)

Detailed data instructions are also provided in [data/README.md](/F:/mine/4_MedPreferBench/code/Med-RewardBench/data/README.md).

## Run Benchmark

Example:

```bash
python vlm/00_benchmark_full_pipeline.py \
  --models MedMO-8B \
  --modalities abdomen brain breast chest eye foot gastrointestinal_tract heart lower_limb lung oral_cavity pelvic_cavity upper_limb \
  --data_root ./data \
  --excel_output ./outputs/MedMO-8B.xlsx \
  --extra_output_formats csv jsonl \
  --table_output_dir ./outputs/tables
```

If you want prompt formatting to follow Qwen thinking mode:

```bash
QWEN_ENABLE_THINKING=0 python vlm/00_benchmark_full_pipeline.py ...
```

## Notes

- `vlm/model_builder_chosen_respon.py` contains the model builders and prompt formatting adapters.
- Some model entries use local checkpoints. Configure them with environment variables instead of editing private paths into the codebase.
- Local checkpoint environment variables currently used by this repository include `MEDR1_MODEL_PATH`, `QWEN2_VL_7B_SFT_PATH`, `QWEN2_5_VL_7B_SFT_PATH`, and `QWEN2_VL_7B_DPO_PATH`.
- Output artifacts should be written to `outputs/` or another ignored directory, not committed to Git.

## TODO Before Public Release

- Add citation information if this benchmark corresponds to a paper or preprint.
- Remove or update private/local model checkpoint paths if you do not want to expose them.
