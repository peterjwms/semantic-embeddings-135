# from huggingface_hub import hf_hub_download
# from datasets import load_dataset
from pathlib import Path

# ds = load_dataset("truthfulqa/truthful_qa", "generation")

# print(ds)

import pandas as pd

df = pd.read_parquet("hf://datasets/truthfulqa/truthful_qa/generation/validation-00000-of-00001.parquet")
df.to_csv(Path("datasets/truthfulqa_generation.csv"), index=False)

df = pd.read_parquet("hf://datasets/truthfulqa/truthful_qa/multiple_choice/validation-00000-of-00001.parquet")
df.to_csv(Path("datasets/truthfulqa_multiple_choice.csv"), index=False)


splits = {'train': 'mnli/train-00000-of-00001.parquet', 'validation_matched': 'mnli/validation_matched-00000-of-00001.parquet', 'validation_mismatched': 'mnli/validation_mismatched-00000-of-00001.parquet', 'test_matched': 'mnli/test_matched-00000-of-00001.parquet', 'test_mismatched': 'mnli/test_mismatched-00000-of-00001.parquet'}
for split in splits.keys():
    df = pd.read_parquet("hf://datasets/nyu-mll/glue/" + splits[split]) 
    df.to_csv(Path(f"datasets/mnli_{split}.csv"), index=False)