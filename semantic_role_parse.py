import requests
import spacy
import ast
import csv
import json


def parse_sentence(sentence, parser,semantic=True,syntax=True):
    doc = parser(sentence)
    processed_tokens = [token.lemma_ for token in doc]
    if syntax:
        syntax_tags = [token.pos_ for token in doc]
        processed_tokens = [lemma + '_' + pos for lemma, pos in zip(processed_tokens,syntax_tags)]
    if semantic:
        semantic_roles = getSemTags(sentence)
        processed_tokens = [lemma + '_' + role for lemma, role in zip(processed_tokens,semantic_roles)]
    return processed_tokens

def getSemTags(sentence):
    
    sentence = sentence.replace(' ','%20')
    res = requests.get('http://localhost:8080/predict/semantics?utterance='+ sentence)
    json = res.json()
    props = json['props']

    tags = ['unk'] * len(sentence.split('%20'))
    for prop in res.json()['props']:
        for span in prop['spans']:
            for i in range(span['start'],span['end']+1):
                tags[i] = span['vn']
    return tags

processed_data = []

parser = spacy.load("en_core_web_lg")

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
        json_text = json.dumps(output)
        


        





        



doc = nlp("John gave Mary the book")

getSemTags("John gave Mary the book")


for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            token.shape_, token.is_alpha, token.is_stop)

for sentence in ['John%20gave%20Mary%20the%20book']:
    print(sentence)

