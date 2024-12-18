import pandas as pd
from pathlib import Path
from datasets import load_dataset, ClassLabel, DatasetDict
import torch
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, AutoModelForSequenceClassification, Trainer
import numpy as np
import evaluate

device = "cuda:0" if torch.cuda.is_available() else "cpu"

QA_PATH = "qa_semtags_mult.csv"
MNLI_TRAIN_PATH = "datasets/mnli_train.csv"
MNLI_VAL_PATH = "datasets/mnli_validation_matched.csv"
MNLI_TEST_PATH = "datasets/mnli_test_matched.csv"

QA_CHECKPOINT = "distilbert/distilbert-base-uncased"
MNLI_CHECKPOINT = "distilbert/distilbert-base-uncased"
NUM_CLASSES = 11

qa_ds = load_dataset("csv", data_files=QA_PATH)
print(qa_ds)
# ds = ds.rename_columns({"body": "text", "subreddit": "label"})
# ds = ds.class_encode_column("label")
for data in qa_ds["train"]:
    if NUM_CLASSES < data['label']:
        NUM_CLASSES = data['label']
        print(NUM_CLASSES)

# split into .8 train, .2 test/dev
train_testdev = qa_ds["train"].train_test_split(seed=42, test_size=0.2)
# split .2 test/valid into .1 test, .1 valid
test_dev = train_testdev["test"].train_test_split(seed=42, test_size=0.5)

qa_ds = DatasetDict({
    "train": train_testdev["train"],
    "validation": test_dev["train"],
    "test": test_dev["test"]
})

current_vals = {"checkpoint": QA_CHECKPOINT, "ds": qa_ds}

tokenizer = AutoTokenizer.from_pretrained(MNLI_CHECKPOINT)

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
    f1_results = f1_metrics.compute(predictions=predictions, references=labels, average="weighted")
    acc = accuracy.compute(predictions=predictions, references=labels)
    return {
        "f1": f1_results["f1"],
        "precision": f1_results["precision"],
        "recall": f1_results["recall"],
        "accuracy": acc,
    }

def preprocess_qa(examples):
    # question_headers = examples["question"]
    # choices = examples["choices"]
    # answer = examples["answer"]
    examples["text"] = [f"{examples['question'][i]} {examples['choices'][i]}" for i in range(len(examples['question']))]
    result = tokenize_function(examples)
    return result

train = current_vals["ds"]["train"]
val = current_vals["ds"]["validation"]
test = current_vals["ds"]["test"]

tokenized_train = train.map(preprocess_qa, batched=True)
tokenized_train.set_format(
    "pt", columns=["input_ids", "attention_mask"], output_all_columns=True
)
tokenized_val = val.map(preprocess_qa, batched=True)
tokenized_val.set_format(
    "pt", columns=["input_ids", "attention_mask"], output_all_columns=True
)
tokenized_test = test.map(preprocess_qa, batched=True)
tokenized_test.set_format(
    "pt", columns=["input_ids", "attention_mask"], output_all_columns=True
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

qa_model = AutoModelForSequenceClassification.from_pretrained(
    current_vals["checkpoint"], num_labels=NUM_CLASSES
).to(device)

# TODO: Hyperparameter finetuning
training_args = TrainingArguments(
    output_dir="qa_models",
    report_to="none",
    # eval_strategy="epoch",
    save_strategy="epoch",
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
    qa_model,
    training_args,
    train_dataset=tokenized_train, #.shuffle(seed=42).select(range(1000)),
    eval_dataset=tokenized_val, #.shuffle(seed=42).select(range(1000)),
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model("qa_models/semantic_qa")

print(trainer.evaluate())