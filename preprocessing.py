import numpy as np
import json
from bs4 import BeautifulSoup

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
    examples = []
    file_data = []
    with open(f, 'r') as file:
        json_l = list(file)
    for json_str in json_l:
        result = json.loads(json_str)
        file_data.append(result)
    for chunk in file_data:
        for elem in chunk['annotations']:
            long_a = elem['long_answer']
            short_a = elem['short_answers']
            yn_a = elem['yes_no_answer']
            long_a_text = ""
            started = False
            for i in range(long_a['start_token'], long_a['end_token']):
                if not chunk['document_tokens'][i]['html_token']:
                    if chunk['document_tokens'][i]['token'][0].lower() not in "abcdefghijklmnopqrstuvwxyz" or not started:
                        long_a_text += chunk['document_tokens'][i]['token']
                        started = True
                    else:
                        long_a_text += " " + chunk['document_tokens'][i]['token']
            examples.append([chunk['document_tokens'], long_a_text, [long_a['start_token'], long_a['end_token']]])
    return examples

NQ_processing(".././Data/NQ/v1.0_sample_nq-dev-sample.jsonl")





        

