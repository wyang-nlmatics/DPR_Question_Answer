from rank_bm25 import BM25Okapi
from tqdm import tqdm
from nltk.tokenize import word_tokenize

import numpy as np
import json
from bs4 import BeautifulSoup
import torch

from transformers import DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoderTokenizer, RobertaTokenizer
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
from torch.optim import Adam
import copy
import torch.optim

import higher # for the higher order optimizer
from torch.cuda.amp import GradScaler, autocast
import time

device = torch.device("cuda") #torch.device('cpu')#
#torch.cuda.empty_cache()


# function to train the model
def train(model, src_dataloader, targ_dataloader):
    
    total_loss, total_accuracy = 0, 0
    model  = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(),lr = 1e-5, weight_decay=1e-3) # learning rate
    
    #scaler = GradScaler()

        
    for i, (src_batch, targ_batch) in tqdm(enumerate(zip(src_dataloader, targ_dataloader))):
        with higher.innerloop_ctx(model, optimizer,copy_initial_weights=False,device=device) as (meta_model, meta_optimizer), torch.backends.cudnn.flags(enabled=False):
            #model.zero_grad()
            #meta_model.zero_grad()
            #meta_model.train()

            #meta_model  = meta_model.to(device)
            #meta_model.train()
            #embed()
            meta_loss, __ = meta_model.criterion(src_batch)
            #meta_loss.backward(retain_graph=True)
            #with torch.no_grad():
            #"""tensor(4.3433, device='cuda:0', grad_fn=<NllLossBackward>)"""
                #print(meta_loss)
                #embed()
            meta_optimizer.step(meta_loss)

            """
            RuntimeError: Trying to backward through the graph a second time
            (or directly access saved variables after they have already been
            freed). Saved intermediate values of the graph are freed when you
            call .backward() or autograd.grad(). Specify retain_graph=True 
            if you need to backward through the graph a second time or if 
            you need to access saved variables after calling backward.
            """

            targ_loss, targ_info = meta_model.criterion(targ_batch)
            final_loss = targ_loss + meta_loss
            #loss.backward() 

            #meta_model.detach()
            #embed()
            with torch.no_grad():
                final_loss.backward(create_graph=False, retain_graph=False) 
                optimizer.step() 
            
            total_loss = total_loss + targ_info['loss']
            total_accuracy = total_accuracy + targ_info['accuracy']

            del meta_model
            #del meta_optimizer
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            if i % 50 == 0 and not i == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(i, len(src_dataloader)))
                print(targ_info)

    return total_loss/len(targ_dataloader), total_accuracy/len(targ_dataloader)


# function for evaluating the model
def evaluate(model, val_dataloader):

    print("\nEvaluating...")

    # deactivate dropout layers
    model.eval()

    total_loss, total_acc = 0, 0

    # empty list to save the model predictions
    total_preds = []
    total_labels = []

    t0 = time.time()

    # iterate over batches
    for step, batch in tqdm(enumerate(val_dataloader)):

        # Progress update every 50 batches.
        if step % 50 == 0 and not step == 0:
            
            # Calculate elapsed time in minutes.
            #elapsed = format_time(time.time() - t0)
                
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))
        #embed()
        # deactivate autograd
        with torch.no_grad():
            
            model = model.to(device)
            # model predictions
            preds, loss, acc = model.predict(batch)

            total_preds.append(preds.detach().cpu().numpy())
            total_labels.extend(batch[3])
            
            total_loss += loss
            total_acc += acc
            
            #print("F1:", f1(preds, labels))
            #print("Acc:", flat_accuracy(preds, labels))

    # compute the validation loss of the epoch
    avg_loss = total_loss / len(val_dataloader) 
    avg_acc = total_acc / len(val_dataloader) 

    print('Average Loss:', avg_loss)
    print('Average Accuracy:', avg_acc)
    print('\n\n')

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)

    return avg_loss, avg_acc, total_preds, total_labels


