import os
os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

import pandas as pd
from pathlib import Path
from datasets import load_dataset, ClassLabel, DatasetDict, concatenate_datasets
import torch
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, AutoModelForSequenceClassification, Trainer
import numpy as np
import evaluate

device = "cuda:0" if torch.cuda.is_available() else "cpu"

'''
TO RUN THE CODE:
install necessary python libraries
specify the file for the embedding schema you want to use under `MNLI_FILE_PATH`
run the code on a GPU - it will take at least 10-20 minutes

'''
MNLI_FILE_PATH = "datasets/processed/featurizedmnli_validation_matched_semantic_syntax.csv"


#specify here
MNLI_CHECKPOINT = "distilbert/distilbert-base-uncased"
NUM_CLASSES = 3


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)


def compute_metrics(eval_preds):
    f1_metrics = evaluate.combine([
        evaluate.load("recall", average="weighted"),
        evaluate.load("precision", average="weighted"),
        evaluate.load("f1", average="weighted"),
    ])
    accuracy = evaluate.load("accuracy")

    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    # Check if score is a float and convert it to a NumPy array if necessary
    f1_results = f1_metrics.compute(predictions=predictions, references=labels, average="weighted")
    
    # Convert float score to an array with size 1
    if isinstance(f1_results["recall"], float):
        f1_results["recall"] = np.array([f1_results["recall"]])

    acc = accuracy.compute(predictions=predictions, references=labels)
    return {
        "f1": f1_results["f1"],
        "precision": f1_results["precision"],
        "recall": f1_results["recall"],
        "accuracy": acc,
    }



def preprocess_mnli(examples):
    # need to put the two sentences together into a text column
    examples["text"] = [f"{examples['premise'][i]} {examples['hypothesis'][i]}" for i in range(len(examples["premise"]))]
    # Tokenize the texts
    result = tokenize_function(examples)
    # Map the labels to their unique IDs
    # result["label"] = [0 if label == "entailment" else 1 if label == "neutral" else 2 for label in examples["label"]]
    return result



mnli_val_ds = load_dataset("csv", data_files=MNLI_FILE_PATH)
print(mnli_val_ds)
combined_ds = mnli_val_ds["train"] 

#split into .8 train .2 test
train_testdev = combined_ds.train_test_split(seed=42, test_size=0.2)
# split .2 test/valid into .1 test, .1 valid
test_dev = train_testdev["test"].train_test_split(seed=42, test_size=0.5)


mnli_ds = DatasetDict({
    "train": train_testdev["train"],
    "validation": test_dev["train"],
    "test": test_dev["test"]
})

#print(mnli_ds) #for debugging

current_vals = {
    "checkpoint": MNLI_CHECKPOINT,
    "ds": mnli_ds,

}

tokenizer = AutoTokenizer.from_pretrained(MNLI_CHECKPOINT)

train = current_vals["ds"]["train"]
val = current_vals["ds"]["validation"]
test = current_vals["ds"]["test"]

tokenized_train = train.map(preprocess_mnli, batched=True)
tokenized_train.set_format(
    "pt", columns=["input_ids", "attention_mask"], output_all_columns=True
)
tokenized_val = val.map(preprocess_mnli, batched=True)
tokenized_val.set_format(
    "pt", columns=["input_ids", "attention_mask"], output_all_columns=True
)
tokenized_test = test.map(preprocess_mnli, batched=True)
tokenized_test.set_format(
    "pt", columns=["input_ids", "attention_mask"], output_all_columns=True
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


mnli_model = AutoModelForSequenceClassification.from_pretrained(
    current_vals["checkpoint"], num_labels=NUM_CLASSES
).to(device)

training_args = TrainingArguments(
    output_dir="mnli_models",
    report_to="none",
    save_strategy="epoch",
    eval_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    weight_decay=0.01,
    # metric_for_best_model="accuracy",
    # load_best_model_at_end=True,
    learning_rate=2e-5,
    # fp16=True,
)

trainer = Trainer(
    mnli_model,
    training_args,
    train_dataset=tokenized_train, #.shuffle(seed=42).select(range(1000)),
    eval_dataset=tokenized_val, #.shuffle(seed=42).select(range(1000)),
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

torch.cuda.empty_cache()
mnli_model.to("cpu")
torch.cuda.empty_cache()
mnli_model.to(device)

trainer.train()

trainer.evaluate() # using the dev set

torch.save(mnli_model.state_dict(), "mnli_model_BIG.pt")

trainer.save_model("trainer_mnli_model_BIG.pt")

predictions = trainer.predict(tokenized_test)
print(predictions)