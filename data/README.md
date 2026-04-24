# Data Directory

This directory is intentionally kept out of version control except for this file.

Expected layout:

```text
data/
`- pair_gt_all/
   |- abdomen.jsonl
   |- brain.jsonl
   |- breast.jsonl
   |- chest.jsonl
   |- eye.jsonl
   |- foot.jsonl
   |- gastrointestinal_tract.jsonl
   |- heart.jsonl
   |- lower_limb.jsonl
   |- lung.jsonl
   |- oral_cavity.jsonl
   |- pelvic_cavity.jsonl
   |- upper_limb.jsonl
   `- parquet/
      |- abdomen.parquet
      |- brain.parquet
      `- ...
```

Download source:

- [Nandzy/Med-RewardBench](https://huggingface.co/datasets/Nandzy/Med-RewardBench/tree/main)

After downloading, point `--data_root` to this `data/` directory.
