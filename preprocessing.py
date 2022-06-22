import numpy as np
import torch
import json

def squad_processing(f):
    with open(f) as file:
        data = json.load(file)
    for par in data['data']:
        title = " ".join((par['title'].split("_")))
        for elem in par['paragraphs']:
            context = elem['context']
            if title.lower() not in context[:100].lower():
                context = title + " " + context
                elem['context'] = context
                for chunk in elem['qas']:
                    for ans in chunk['answers']:
                        ans['answer_start'] += len(title)+1

def NQ_processing(f):
    file_data = []
    with open(f, 'r') as file:
        json_l = list(file)
    for json_str in json_l:
        result = json.loads(json_str)
        file_data.append(result)
    print(result)

NQ_processing("NQ/v1.0_sample_nq-dev-sample.jsonl")





        

