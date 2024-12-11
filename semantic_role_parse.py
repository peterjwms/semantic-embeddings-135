import requests
import spacy
import ast
import csv
import json
import pandas as pd


def parse_sentence(sentence, parser,semantic=True,syntax=True):
    try:
        doc = parser(sentence)

    except:
        print("Failed to process ",sentence)
        return

    
    processed_tokens = [token.lemma_ for token in doc]
    tokens_len = len(processed_tokens)
    if syntax:
        syntax_tags = [token.pos_ for token in doc]
        processed_tokens = [lemma + '_' + pos for lemma, pos in zip(processed_tokens,syntax_tags)]
    if semantic:
        semantic_roles = getSemTags(sentence,tokens_len)
        processed_tokens = [lemma + '_' + role for lemma, role in zip(processed_tokens,semantic_roles)]
    return processed_tokens

def getSemTags(sentence,tokens_len):
    sentence = sentence.replace(' ','%20')
    try:
        res = requests.get('http://localhost:8080/predict/semantics?utterance='+ sentence)
    except:
        print("failed to GET ", sentence)
    json = res.json()
    props = json['props']

    tags = ['unk'] * tokens_len
    for prop in res.json()['props']:
        for span in prop['spans']:
            for i in range(span['start'],min(span['end']+1,tokens_len)):
                tags[i] = span['vn']
    return tags

processed_data = []
print("start")

parser = spacy.load("en_core_web_lg")

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





with open('datasets/truthfulqa_generation.csv', 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader, None) 
    for row in csv_reader:
        output = {}
        output["adversarial"] = row[0]
        output["category"] = row[1]
        output["question"] = parse_sentence(row[2],parser)
        output["answer"] = parse_sentence(row[3],parser)
        output["alt_answers"] =  [parse_sentence(alt_answer,parser) for alt_answer in ast.literal_eval(row[4].replace("\n",","))]
        output["wrong_answers"] =  [parse_sentence(wrong_answer,parser) for wrong_answer in ast.literal_eval(row[5].replace("\n",","))]
        processed_data.append(json.dumps(output))


with open('out.json', 'w') as file:
    json.dump(processed_data, file, indent=4)

print("made it to the end")
        


        





        




