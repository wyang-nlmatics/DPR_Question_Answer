from rank_bm25 import BM25Okapi
from tqdm import tqdm
from nltk.tokenize import word_tokenize

import numpy as np
import json
from bs4 import BeautifulSoup
import torch

from transformers import DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoderTokenizer
from transformers import DPRQuestionEncoder
from transformers import DPRContextEncoder
import csv
from transformers import BertModel, BertTokenizer, BertTokenizerFast

from torch.utils.data import DataLoader, Dataset
from nltk import word_tokenize
import pandas as pd
import random
from torch.nn import CosineSimilarity
from torch.nn.functional import normalize

from IPython import embed
from sklearn.metrics import classification_report
import argparse

torch.cuda.empty_cache() 

"""
python baseline.py --eval_data cuad/CUAD_v1-clean-0.json --ixs_data cuad/CUAD_v1-clean-ix-0.csv

python baseline.py --eval_data pubmed/full_clean_small.json --ixs_data pubmed/pubmed-pubmed-ix.csv

python baseline.py --eval_data pubmed/full_clean_small.json --ixs_data pubmed/pubmed-pubmed-ix.csv

python baseline.py --eval_data scotus/train-clean-new.json --ixs_data scotus/train-ix.csv
python baseline.py --eval_data scotus/dev-clean-new.json --ixs_data scotus/dev-ix.csv

python baseline.py --eval_data cuad/CUAD_v1-clean-lite.json --bm25 True

python baseline.py --eval_data nfcorpus/dev-clean.json --bm25 True

python baseline.py --eval_data scotus/dev-clean-new.json --bm25 True

python baseline.py --eval_data squad/dev-clean.json --bm25 True
"""


def embed_contexts(contexts, tokenizer, encoder):
    print('embedding context.')
    fp = open('contexts.csv', 'w')
    context_embeds = []
    for c in tqdm(contexts):
        with torch.no_grad():
            #t1=BertTokenizerFast.from_pretrained('bert-base-uncased')
            tokenized = tokenizer(c, padding='max_length', max_length = 512,truncation=True )
            encoder = encoder.to(device)
            context_embeds.append( encoder( torch.tensor([tokenized['input_ids']]).to(device) )[0].cpu()  )#  .to(device)   ) )#.to(device) )  #.to(device)))#.detach()  )
    
    return context_embeds #torch.stack(context_embeds).to(device) #.detach()

def embed_queries(query, tokenizer, encoder):
    print('embedding queries.')
    fp = open('queries.csv', 'w')
    query_embeds = []
    for q in tqdm(query):
        with torch.no_grad():
            #t1=BertTokenizerFast.from_pretrained('bert-base-uncased')
            tokenized = tokenizer(q , padding='max_length',max_length = 512, truncation=True)
            #query_embed = encoder( torch.tensor( [tokenized.to(device)] ) )
            encoder = encoder.to(device)
            query_embeds.append( encoder( torch.tensor([tokenized['input_ids']]).to(device)   )[0].cpu() )#.to(device) )[0] )#.to(device)) )
    return query_embeds #torch.stack(query_embeds).to(device)#.detach()

def run_DPR(query_embed, context_embeds, ixs):
    similarities = []
    for i in range(len(context_embeds)):
        similarities.append( torch.dot(query_embed[0], context_embeds[i][0])/torch.norm(context_embeds[i][0]) )
        #embed()
        #a = normalize(e)
        #b = normalize(query_embed)
        
        #similarities.append( CosineSimilarity(dim=-1, eps=1e-6)(e, query_embed)  )
    #embed()
    return ixs[np.argmax(np.array(similarities))].item()   #, dim = 0)



class SimpleDataset(Dataset):
    def __init__(self, context, query, QC_map, ixs):
        with torch.no_grad():
            context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')#.to(device)
            question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')#.to(device)
            question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base').to(device)
            context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base').to(device)

        #embed()
        self.context_embeds = embed_contexts(context, context_tokenizer, context_encoder)
        self.query_embeds = embed_queries(query, question_tokenizer, question_encoder)
        self.map  = QC_map
        self.ixs = ixs

    def __len__(self):
        #assert self.ixs == self.query_embeds
        return len(self.ixs)

    def __getitem__(self, idx):
        # query embed, context embeds, true label
        batch_ixs = self.ixs[idx]
        query_embed = self.query_embeds[   int(batch_ixs[0])    ]
        context_embeds = [ self.context_embeds[int(batch_ixs[i])]  for i in range(len(batch_ixs) -1)  ]
        context_embeds[0] = self.context_embeds[ self.map[int(batch_ixs[0])]  ]
        label = self.map[idx]
        return ( query_embed, context_embeds, batch_ixs[0:-1], label )


# find the indices of the k max elements,
def kmax(k, nums, blocked):
    ix = []
    while len(ix) < k:
        w = max(nums)
        w_ix = nums.tolist().index(w)
        nums[w_ix] = float('-inf')
        if w_ix != blocked:
            ix.append(w_ix)
    return ix

def sample(j, nums, blocked):
    rand = random.sample(list(range(len(nums))), 30)
    true = []
    for r in rand:
        if r not in blocked:
            true.append(r)
    return true[0:20]

def make_batches(name, x, k_hard=10, j_rand=20):
    file = open(name+ '/' + name + '-' + str(name) + '-ix.csv', 'w+', newline ='\n')
    tokenized_corpus = [word_tokenize(a) for a in x['text']]

    # DESIGNING BATCHES
    rows = []
    print('making batches')
    bm25 = BM25Okapi([word_tokenize(con) for con in x['text']])
    for i in tqdm(range(len(x['question']))):

        doc_scores = bm25.get_scores(x['question'][i])

        correct = x['map'][str(i)] 
        
        hard_ix = kmax(k_hard, doc_scores, correct)
        rand_ix = sample(j_rand, doc_scores, hard_ix + [correct])
        
        answers = [correct] + hard_ix + rand_ix
        rows.append(answers)
    
    with file:   
        write = csv.writer(file)
        write.writerows(rows)

    return x, rows


def squad_processing(f, k_hard=10, j_rand=20, make_batches=False):
    fname = f.split('/')[-1].split('.')[0]

    question = [] # list of questions
    contexts = [] # list of contexts
    qc_map = {} # dictionary mapping contexts_ix to relevant questions_ix
    with open(f) as file:
        data = json.load(file)
    
    for i in range(1):#tqdm(range(len(data['data']))):
        par = data['data'][i]
        title = " ".join((par['title'].split("_")))
        for j in range(len(par['paragraphs'])):
            elem = par['paragraphs'][j]
            context = elem['context']
            contexts.append(context)
            if title.lower() not in context[:100].lower():
                context = title + " " + context
            for chunk in elem['qas']:
                data_elem = {}
                qc_map[len(question)] = len(contexts)-1
                question.append(chunk['question'])  

    x = {'question': question, 'text': contexts, 'map': qc_map}

    # writing reorganized data to file
    file = open('squad/' + str(fname) + '-clean-lite.json', 'a',newline ='\n')
    file.truncate(0)
    json_string = json.dumps(x)
    #json.dump(json_string, file)
    file.write(json_string)

    # Creating strategic batches
    
    rows = []
  
    if make_batches:
        file = open('squad/squad-' + str(fname) + '-ix-lite.csv', 'w+', newline ='\n')
        tokenized_corpus = [word_tokenize(a) for a in x['text']]

        # DESIGNING BATCHES
        print('making batches')
        bm25 = BM25Okapi([word_tokenize(con) for con in x['text']])
        for i in tqdm(range(len(x['question']))):

            doc_scores = bm25.get_scores(x['question'][i])

            correct = x['map'][i]
            
            hard_ix = kmax(k_hard, doc_scores, correct)
            rand_ix = sample(j_rand, doc_scores, hard_ix + [correct])
            
            answers = [correct] + hard_ix + rand_ix
            rows.append(answers)
        
        with file:   
            write = csv.writer(file)
            write.writerows(rows)

    return x, rows


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


# BOTH run_BM25 and run_DPR produce the index of the best context
#   query = string, contexts = list of strings

def run_BM(query, bm25):
    scores = bm25.get_scores(word_tokenize(query))
    return np.argmax(scores)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_data', dest='eval_data', help='Name of the eval data file', required=False, default='sample')
    parser.add_argument('--ixs_data', dest='ixs_data', help='name of indices file', required=False, default='')
    parser.add_argument('--bm25', dest='bm25', required=False, default=False)


    args = parser.parse_args()

    f = open(args.eval_data)
    x = json.load(f)
    qc_map = {int(k):int(v) for k,v in x['map'].items()}
    #embed()

    y_true = [qc_map[i] for i in range(len(x['question']))]
    y_pred = []

    if args.bm25:
        print('running bm25!')
        bm25 = BM25Okapi([word_tokenize(con) for con in x['text']])

        CUTOFF = len(x['question']) # USE CUTOFF TO MAKE LIFE EASIER SOMETIMES
        counter = 0
        for q in tqdm(x['question']):

            y_pred.append( run_BM( q, bm25) )

            counter += 1
            if counter > CUTOFF: 
                break

        
        print('BM25 baselines for ', args.eval_data)
        #print(classification_report(y_true, y_pred))
        print(classification_report(y_true[:CUTOFF], y_pred[:CUTOFF]))
    else:
        if args.ixs_data != '':
            batch_ixs = pd.read_csv(args.ixs_data, index_col=None, header=None).values
            f.close()
        else:
            assert 1 == 0 # break the code cuz we need ixs!

        
        dataset = SimpleDataset(x['text'], x['question'], qc_map, batch_ixs)
        dataloader = DataLoader(dataset, batch_size=50)
        
        y_pred = []
        y_label = []

        #context_embeds = dataset.context_embeds
        #context_embeds = [torch.tensor(context_embeds[j][0]) for j in range(len(context_embeds))]
        #context_embeds = [torch.tensor(dataset.context_embeds[j][0][0]) for j in range(len(dataset.context_embeds))]


        for step, (query, context, ixs, label) in tqdm(enumerate(dataloader)): 
            print(step)
            #embed()

            context_embeds = []
            # reorganize! because we didn't remake the collate function :(
            for j in range(len(context)):
                new = []
                for k in range(len(context[j])):
                    new.append(context[j][k])
                context_embeds.append(torch.stack(new))
            context_embeds = torch.stack(context_embeds).permute(1,0,2,3)

            #embed()        
            #embed()
            #ixs = torch.stack(ixs).T
            for i in range(len(query)): # this should be batch size
                print(i)
                #embed()
                question_embed = query[i] #.detach()
                #embed()
                pred = run_DPR(question_embed, context_embeds[i], ixs[i])
                y_pred.append(pred)
                y_label.append(label[i].item())
                print('pred', pred)
                print('label', label[i].item())
                #embed()

                #for j in range(len(context_embeds)):
                #    context_embed = torch.tensor(context_embeds[j][0])#[0]#.cpu().numpy()
                #    pred = run_DPR(question_embed, context_embeds)
                #    y_pred.append(pred)
                #    y_label.append(labels[i])
                    #print(y_pred)
                    #print(y_label)
                    #embed()

        #embed()
        print(classification_report(y_label, y_pred))

if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

main()

