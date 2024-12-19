import json
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path("semantic-embeddings-135/qa_gen_processed.json"))

print(df.head())

print("here")

row1 = df.iloc[0]
print(row1)

# print(json.loads(row1["question"].replace('\'', '"')))
# print(row1["answer"])