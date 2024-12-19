# semantic-embeddings-135
COSI 135b Final Project exploring contextual embeddings with semantic and syntactic embeddings


[Proposal & Notes](https://docs.google.com/document/d/1TROsfiCp_7LEXnwlXVvwwOFQCagzEnc10vwqtuxpDoU/edit?usp=sharing)
## Overview
<p>This project examines two different NLP tasks using our proposed feature embedding schema: natural language inference labeling and closed-book question answering.</p>

<p>The repo contains code for gathering and pre-processing data, but also has the data generated by these files store in \datasets </p>

You can find more details in the file `Applying Semantic and Syntactic Information to Contextual Embeddings.pdf`

## Run instructions
This project examines two different NLP tasks using our proposed feature embedding schema: natural language inference labeling and closed-book question answering.

 - clone the repo
 - choose which task you want to run, selecting either `train_mnli.py` natural language inference labeling for  or `train_qa.py` for closed-book question answering.
 - within the chosen python file, select the pre-processed data file for the embeddings you want to use - refer to the table Naming scheme and Organization to select the appropriate dataset for the desired task/feature representation.
 - code should be able to run - a GPU is recommended for the MNLI task




## Dataset notes/descriptions
### [MNLI GLUE](https://huggingface.co/datasets/nyu-mll/glue)
 - validation_matched contains sentence pairs in the same domain, mismatched is cross-domain
 - labels are:
   - -1 = contradiction
   - 0 = neutral
   - 1 = entailment


### [TruthfulQA](https://huggingface.co/datasets/truthfulqa/truthful_qa)
 - mc1 is where only one answer to select from is correct
 - mc2 is at least one or more correct answers to select from


### Naming scheme and Organization
Each file is named according to the tags it has, ending in either `_syntax`, `_semantic`, or `_syntax_semantic`. Files are sorted into raw and processed folders. 

| Dataset  | Tags | File |
| --- | --- | --- |
| MNLI Matched Test  | None  | [](/datasets/raw/mnli_test_matched.csv) |
| MNLI Matched Test  | Semantic  | (/datasets/processed/mnli_test_matched_semantic.csv) |
| MNLI Matched Test  | Syntax  | (/datasets/processed/mnli_test_matched_syntax.csv) |
| MNLI Matched Test  | Syntax, Semantic  |  (/datasets/processed/mnli_test_matched_syntax_semantic.csv) |
| MNLI Matched Validation  | None  | (/datasets/raw/mnli_validation_matched.csv) |
| MNLI Matched Validation  | Semantic  | (/datasets/processed/mnli_validation_matched_semantic.csv) |
| MNLI Matched Validation  | Syntax  | (/datasets/processed/mnli_validation_matched_syntax.csv) |
| MNLI Matched Validation  | Syntax, Semantic  | (/datasets/processed/mnli_validation_matched_syntax_semantic.csv) |
| TruthfulQA Generation  | None  | (/datasets/raw/truthfulqa_generation.csv) |
| TruthfulQA Generation  | Semantic  | (/datasets/processed/truthfulqa_generation_semantic.csv) |
| TruthfulQA Generation  | Syntax  | (/datasets/processed/truthfulqa_generation_syntax.csv) |
| TruthfulQA Generation  | Syntax, Semantic  | (/datasets/processed/truthfulqa_generation_syntax_semantic.csv) |
| TruthfulQA Multiple Choice  | None  | (/datasets/raw/truthfulqa_multiple_choice.csv) |
| TruthfulQA Multiple Choice  | Semantic  | (/datasets/processed/truthfulqa_multiple_choice_semantic.csv) |
| TruthfulQA Multiple Choice  | Syntax  | (/datasets/processed/truthfulqa_multiple_choice_syntax.csv) |
| TruthfulQA Multiple Choice  | Syntax, Semantic  | (/datasets/processed/truthfulqa_multiple_choice_syntax_semantic.csv) |
