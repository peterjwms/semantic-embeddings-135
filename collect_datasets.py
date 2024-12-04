from huggingface_hub import hf_hub_download
from datasets import load_dataset
from pathlib import Path

ds = load_dataset("truthfulqa/truthful_qa", "generation")

print(ds)

import pandas as pd

df = pd.read_parquet("hf://datasets/truthfulqa/truthful_qa/generation/validation-00000-of-00001.parquet")

print(df.head())
print(df)

df.to_csv(Path("datasets/truthfulqa_generation.csv"), index=False)