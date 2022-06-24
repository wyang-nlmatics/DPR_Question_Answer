import numpy as np
import json
from bs4 import BeautifulSoup

def squad_processing(f):
    examples = []
    with open(f) as file:
        data = json.load(file)
    for par in data['data']:
        title = " ".join((par['title'].split("_")))
        for elem in par['paragraphs']:
            context = elem['context']
            if title.lower() not in context[:100].lower():
                context = title + " " + context
            for chunk in elem['qas']:
                data_elem = {}
                data_elem['question'] = chunk['question']
                data_elem['document_text'] = context
                examples.append(data_elem)
    return examples

def NQ_processing(f):
    examples = []
    file_data = []
    with open(f, 'r') as file:
        json_l = list(file)
    for json_str in json_l:
        result = json.loads(json_str)
        file_data.append(result)
    for chunk in file_data:
        data_elem = {}
        data_elem['question'] = chunk['question_text']
        doc_string = ""
        started = False
        for elem in chunk['document_tokens']:
            if not elem['html_token']:
                if elem['token'][0].lower() not in "abcdefghijklmnopqrstuvwxyz" or not started:
                    doc_string += elem['token']
                    started = True
                else:
                    doc_string += " " + elem['token']
        data_elem['document_text'] = doc_string
        examples.append(data_elem)
    return examples

print(squad_processing(".././Data/squad2.0/dev-v2.0.json"))





        

