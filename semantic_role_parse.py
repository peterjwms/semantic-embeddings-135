import requests
import spacy
import ast
import csv
import json
import pandas as pd
import sys

orig_stdout = sys.stdout



def parse_sentence(sentence, nlp,semantic=True,syntax=True,lemmatize=False):
    try:
        doc = nlp(sentence)

    except:
        print("Failed to process ",sentence)
    raw_tokens = [token.text for token in doc if token.pos_ != 'PUNCT']
    if lemmatize:
        processed_tokens = [token.lemma_ for token in doc if token.pos_ != 'PUNCT']
    else:
        processed_tokens = raw_tokens
    if syntax:
        syntax_tags = [token.pos_ for token in doc]
        processed_tokens = [lemma + '_' + pos for lemma, pos in zip(processed_tokens,syntax_tags)]
    if semantic:
        semantic_roles = getSemTags(raw_tokens)
        processed_tokens = [lemma + role for lemma, role in zip(processed_tokens,semantic_roles)]
    return processed_tokens

def getSemTags(sentence):
    num_tokens = len(sentence)
    sentence = '%20'.join(sentence)
    
    sentence = sentence.replace(' ','%20')
    #print('requests req')
    try:
        res = requests.get('http://localhost:8080/predict/semantics?utterance='+ sentence)
    except:
        print("failed to GET ", sentence)
    #print('got resp')
    json = res.json()

    tags = [''] * num_tokens
    for prop in json['props']:
        for span in prop['spans']: #still need to identify if isPredicate
            for i in range(span['start'],min(span['end']+1,num_tokens)):
                tags[i] = tags[i] + "_" + span['label'].replace(" ","").replace("<->","_")
    return tags


print("start")

nlp = spacy.load("en_core_web_sm")
#nlp.tokenizer.rules = {key: value for key, value in nlp.tokenizer.rules.items() if "'" not in key and "’" not in key and "‘" not in key} #modifies pipeline so it doesn't split on apostrophes

# df = pd.read_csv('datasets/truthfulqa_generation.csv')

# for index, row in df.iterrows():
#     question = row['question']
#     best = row['best_answer']
#     correct = row['correct_answers']
#     incorrect = row['incorrect_answers']
# with open('datasets/truthfulqa_multiple_choice.csv', 'r') as file:
#     csv_reader = csv.reader(file)
#     next(csv_reader, None) 
#     for row in csv_reader:
#         output = {}
#         processed_question = parse_sentence(row[0],parser)
#         cleaned_dict = ast.literal_eval(row[1].replace("array(", "").replace("dtype=object", "").replace("\n       ","").replace("])}","]}").replace(",\n      ),",","))
#         answers = [parse_sentence(answer,parser) for answer in cleaned_dict["choices"]]
#         processed_data.append([processed_question,{'choices':answers,'labels':cleaned_dict["labels"]}])



processed_data = []
with open('datasets/truthfulqa_generation.csv', 'r') as file:
    print('opened file')
    csv_reader = csv.reader(file)
    next(csv_reader, None) 
    file_data = [row for row in csv_reader]


for row in file_data:
        output = {}
        output["adversarial"] = row[0]
        output["category"] = row[1]
        output["question"] = parse_sentence(row[2],nlp)
        output["answer"] = parse_sentence(row[3],nlp)
        output["alt_answers"] =  [parse_sentence(alt_answer,nlp) for alt_answer in ast.literal_eval(row[4].replace("\n",","))]
        output["wrong_answers"] =  [parse_sentence(wrong_answer,nlp) for wrong_answer in ast.literal_eval(row[5].replace("\n",","))]
        processed_data.append(json.dumps(output))
        # print(json.dumps(output))


with open('qa_gen_processed.json', 'w') as file:
    json.dump(processed_data, file, indent=4)

print("made it to the end")
        


        





        




