from pathlib import Path
import requests
import spacy
import ast
import json
import pandas as pd
import sys

orig_stdout = sys.stdout



def parse_sentence(sentence, nlp,semantic=True,syntax=False,lemmatize=True):
    try:
        doc = nlp(sentence)

    except:
        return None
    raw_tokens = [token.text for token in doc if token.pos_ != 'PUNCT']
    if lemmatize:
        processed_tokens = [token.lemma_ for token in doc if token.pos_ != 'PUNCT']
    else:
        processed_tokens = raw_tokens
    if syntax:
        syntax_tags = [token.pos_ for token in doc]
        processed_tokens = [lemma + '_' + pos for lemma, pos in zip(processed_tokens,syntax_tags)]
    if semantic:
        try:
            semantic_roles = getSemTags(raw_tokens)
        except:
            return None
        processed_tokens = [lemma + role for lemma, role in zip(processed_tokens,semantic_roles)]
    return " ".join(processed_tokens)

def getSemTags(sentence):
    num_tokens = len(sentence)
    sentence = '%20'.join(sentence)
    
    sentence = sentence.replace(' ','%20')
    #print('requests req')
    try:
        res = requests.get('http://localhost:8080/predict/semantics?utterance='+ sentence)
    except:
        raise Exception("Failed to get response")
    
    if res.status_code != 200:
        raise Exception("Failed to get response")
     
    json = res.json()

    tags = [''] * num_tokens
    for prop in json['props']:
        for span in prop['spans']: #still need to identify if isPredicate
            for i in range(span['start'],min(span['end']+1,num_tokens)):
                #tags[i] = tags[i] + "_" + span['label'].replace(" ","").replace("<->","_")
                tags[i] =  "_" + span['vn'] if tags[i] == '' else tags[i]
    return tags

def process_truthful_qa():
    pandas_test = []

    processed_data = []
    with open(Path('datasets/truthfulqa_generation.csv'), 'r') as file:
        file_data = pd.read_csv(file)

    for i, row in file_data.iterrows():
            output = {}
            output["adversarial"] = row[0]
            output["category"] = row[1]
            output["question"] = parse_sentence(row[2],nlp)
            output["answer"] = parse_sentence(row[3],nlp)
            output["alt_answers"] =  [parse_sentence(alt_answer,nlp) for alt_answer in ast.literal_eval(row[4].replace("\n",","))]
            output["wrong_answers"] =  [parse_sentence(wrong_answer,nlp) for wrong_answer in ast.literal_eval(row[5].replace("\n",","))]
            processed_data.append(json.dumps(output))
            pandas_test.append(output)

            # print(json.dumps(output))


    with open('qa_semtags_processed.json', 'w') as file:
        json.dump(processed_data, file, indent=4)

    df = pd.DataFrame(pandas_test)
    df.to_csv('qa_semtags_df.csv', index=False)

def process_mnli(nlp,file_name,semantics,syntax):    
    with open(Path('datasets/' + file_name + ".csv"), 'r',encoding='utf-8',errors='ignore') as file:
        df = pd.read_csv(file)
    if "train" in file_name:
        df = df.sample(100000,random_state=42)
    indices_to_delete = []

    for i, row in df.iterrows():
        processed_premise = parse_sentence(row["premise"],nlp,semantic=semantics,syntax=syntax)
        if processed_premise != None:
            df.at[i,'premise'] = processed_premise
        else:
            indices_to_delete.append(i)
            continue
        processed_hypothesis = parse_sentence(row["hypothesis"],nlp,semantic=semantics,syntax=syntax)
        if processed_hypothesis != None:
            df.at[i,"hypothesis"] = processed_hypothesis
        else:
            indices_to_delete.append(i)
            continue
            


            
    df.drop(indices_to_delete, axis=0, inplace=True)
    outfile = 'datasets/processed/featurized_new_system_' + file_name
    outfile += "_semantic" if semantics else ""
    outfile += "_syntax" if syntax else ""
    df.to_csv(outfile + ".csv", index=False)



print("start")

nlp = spacy.load("en_core_web_sm")

for file_path_name in ["mnli_train","mnli_validation_matched"]:
    for semantics in [True,False]:
        for syntax in [True,False]:
            process_mnli(nlp,file_path_name,semantics,syntax)
#process_truthful_qa(nlp)


        


        





        




