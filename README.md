# semantic-embeddings-135
COSI 135b Final Project exploring contextual embeddings with semantic and syntactic embeddings

[Proposal & Notes](https://docs.google.com/document/d/1TROsfiCp_7LEXnwlXVvwwOFQCagzEnc10vwqtuxpDoU/edit?usp=sharing)

## TODO
 - Collect the datasets
 - Pipeline for tagging datasets with syntactic information
 - Pipeline for tagging datasets with semantic information
 - Finetuning architecture/pipeline


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
