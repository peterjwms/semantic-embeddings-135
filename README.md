# semantic-embeddings-135
COSI 135b Final Project exploring contextual embeddings with semantic and syntactic embeddings


[Proposal & Notes](https://docs.google.com/document/d/1TROsfiCp_7LEXnwlXVvwwOFQCagzEnc10vwqtuxpDoU/edit?usp=sharing)

## Run instructions
This project examines two different NLP tasks using our proposed feature embedding schema: natural language inference labeling and closed-book question answering.

 -clone the repo
 -choose which task you want to run, selecting either `train_mnli.py` natural language inference labeling for  or `train_qa.py` for closed-book question answering.
 -within the chosen python file, select the pre-processed data file for the embeddings you want to use
 -code should be able to run - a GPU is recommended for the MNLI task


## Dataset notes/descriptions
### [MNLI GLUE](https://huggingface.co/datasets/nyu-mll/glue)
 - matched is on in-domain data, mismatched is cross-domain
 - labels are probably:
   - -1 = contradiction
   - 0 = neutral
   - 1 = entailment


### [TruthfulQA](https://huggingface.co/datasets/truthfulqa/truthful_qa)
 - mc1 is where only one answer to select from is correct
 - mc2 is (probably at least) one or more correct answers to select from
