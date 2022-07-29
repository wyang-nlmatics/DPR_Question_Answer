
import os
#os.environ['NVIDIA_VISIBLE_DEVICES']="$gpu_id"
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#print('WHICH GPU ARE WE USING?', os.environ["CUDA_VISIBLE_DEVICES"])

import pandas as pd
import torch
from transformers import DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer, RobertaTokenizer
from transformers import DPRQuestionEncoder, DPRContextEncoder, RobertaModel
from IPython import embed
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from train_model import train, evaluate
from torch.utils.data import DataLoader, Dataset
import json
from torch.nn import LogSoftmax
from transformers import DistilBertModel
from transformers import AutoTokenizer
import random

"""
python run_modeling.py --src_data squad/dev_clean.json \
--targ_data pubmed/full_clean_small.json \
--dev_data pubmed/full_clean_small.json \
--test_data scotus/dev_clean.json \
--num_batches 100 \
--train True \
--eval True


python run_modeling.py --src_data squad/dev_clean.json \
--targ_data squad/dev_clean.json \
--dev_data squad/dev_clean.json \
--test_data squad/dev_clean.json \
--train True --eval True



python run_modeling.py --src_data squad/dev_clean.json \
--targ_data squad/dev_clean.json \
--dev_data squad/dev_clean.json \
--test_data squad/dev_clean.json \
--eval True \
--train True \

python run_modeling.py --src_data ../williamyang/Data/NQ/dev/nq-dev-00.jsonl.gz \
--targ_data ../williamyang/Data/NQ/dev/nq-dev-01.jsonl.gz \
--dev_data ../williamyang/Data/NQ/dev/nq-dev-02.jsonl.gz \
--test_data ../williamyang/Data/NQ/dev/nq-dev-03.jsonl.gz \
--train True \
--eval True
""" 


class SimpleDataset(Dataset):
    # filename denotes the "cleaned" version of the file
    def __init__(self, fname, pre_ixs = False):
        """
        with torch.no_grad():
            context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')#.to(device)
            question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')#.to(device)
            #question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base').to(device)
            #context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base').to(device)
        """
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")#RobertaTokenizer.from_pretrained("roberta-base")
        data = json.load(open(fname))
        context_text = data['question']
        query_text = data['text']

        self.context = tokenizer(context_text, padding='max_length',max_length = 100, truncation=True,return_token_type_ids=False)['input_ids']
        self.query = tokenizer(query_text, padding='max_length',max_length = 50, truncation=True,return_token_type_ids=False)['input_ids']
        self.map  = [int(k) for k in data['map']]
        #self.neg_ixs = neg_ixs # when we use specially designed negative samples from BM25

        #if pre_ixs:
        #    self.neg_ixs = pd.read_csv(fname.split('_')[0] + '_ix.csv')
        #else:
        #    self.neg_ixs = []


    def __len__(self):
        #assert self.ixs == self.query_embeds
        return len(self.query)

    def get_context_batch(self, correct_context_ix, size):
        # TODO: make this more involved 
        randshift = random.randint(0, size-1)
        ixs = [ (correct_context_ix  + i + randshift ) % len(self.context) for i in range(size)  ]
        contexts = [self.context[ix] for ix in ixs]
        return contexts, ixs, randshift
    
    def __getitem__(self, idx):
        # query embed, context embeds, true label
        correct_context_ix = self.map[idx]

        contexts, context_ixs, label = self.get_context_batch(correct_context_ix, 10)

        return [self.query[idx],  contexts,  context_ixs,  label]#int(len(context_ixs)/2)]#context_ixs.index(correct_context_ix) ]
        
def collate_fn(batch):
    query = []
    context = []
    context_ixs = []
    label = []
    for b in batch:
        query.append(b[0])
        context.append(b[1])
        context_ixs.append(b[2])
        label.append(b[3])
    return [query, context, context_ixs, label]#{'query':query, 'context': context, 'context_ixs': context_ixs, 'label':label}


class DPR(torch.nn.Module):

    def __init__(self, context_encoder,  question_encoder):
        super().__init__()
        self.question_encoder = question_encoder
        self.context_encoder = context_encoder
        #self.context_head = torch.nn.Linear(1024, 2048)
        #self.question_head = torch.nn.Linear(1024, 1024)
        #self.tanh = torch.nn.Tanh()
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, context_batch, question_batch):
        #Context_batch (shape) : batchsize x seq_lenth : (questino_batch (shape))
        #embedding context (shape) : batch_size x seq_len x 1024
        #embed()
        #embed()
        #cb = [torch.reshape(torch.tensor(i).to(device), (len(torch.tensor(i).to(device)),1)) for i in context_batch]
        cb = [torch.tensor(i).to(device) for i in context_batch]
        #embeddings_context = self.context_encoder(*cb).last_hidden_state
        
        embeddings_context = torch.stack([self.context_encoder(i).last_hidden_state[:,0,:] for i in cb])
        #e = self.context_encoder(context_batch[0]).last_hidden_state
        #embeddings_context = torch.stack([e for i in range(len(cb))])

        #embeddings_context = torch.ones(torch.tensor(context_batch).size())

        #x = []
        #for i in range(len(context_batch)):
        #    x.append(self.context_encoder(context_batch[i]).last_hidden_state) 
        #embeddings_context = torch.stack(x)       


        #embeddings_context = self.context_encoder(**context_batch).last_hidden_state
        #batch_size, seq_length, _ = embeddings_context.shape
        #embedding_question (shape) : batch_size x 1024
        #embed()
        #q = torch.reshape(torch.tensor(question_batch).to(device), (len(torch.tensor(question_batch).to(device)),1))
        embedding_question = self.question_encoder(torch.tensor(question_batch).to(device)).last_hidden_state[:,0,:]
        #embedding_question = self.question_encoder(q).last_hidden_state
        #embedding_question = self.question_encoder(torch.tensor(question_batch)[0]).last_hidden_state[:, 0]
        #same shape as thier embeddings
        #ctx_token_embeddings = self.tanh(self.context_head( embeddings_context)).reshape(batch_size, seq_length, 1024, 2)
        #question_embedding = self.tanh(self.question_head( embedding_question))
        #logit1 = torch.einsum('ik, ijkm->ijm', question_embedding, ctx_token_embeddings)  
        #embed()
        #logits = torch.einsum('ijkl, ikl ->ij', embeddings_context, embedding_question) 
        logits = torch.einsum('ijk, ik ->ij', embeddings_context, embedding_question) 
        # logits are (batch_size by num_context_choices)
        return logits

    def predict(self,batch):
        
        with torch.no_grad():
            question_batch = batch[0]
            context_batch = batch[1]
            context_ixs = batch[2]
            labels = batch[3]

            logits = self.log_softmax(self.forward(context_batch, question_batch))
            loss = self.loss_fn(logits.to(device), torch.tensor(labels).to(device))
            acc = sum([1 if logits.argmax(axis = -1)[i] == labels[i] else 0 for i in range(len(labels))]) / len(labels) 

        return logits.argmax(axis = -1), loss, acc # predictions, loss


    def criterion(self, batch):#context_batch, question_batch, labels):
        question_batch = batch[0]
        context_batch = batch[1]
        context_ixs = batch[2]
        labels = batch[3]

        #batch_size, _ , = labels.shape
        batch_size = len(context_batch)
        #change this to BCELoss with logits 
        #loss_func = torch.nn.BCEWithLogitsLoss()

        logits = self.forward(context_batch, question_batch)

        #loss = loss_func(logits.reshape(-1, 2), labels.reshape(-1).long())
        loss = self.loss_fn(logits.to(device), torch.tensor(labels).to(device))
        
        logging_output = {
            "loss": loss.item(), 
            "accuracy" : sum([1 if logits.argmax(axis = -1)[i] == labels[i] else 0 for i in range(len(labels))]) / batch_size   #(logits.argmax(axis = -1) == label).prod(axis = -1).sum() / batch_size
        }
        return loss, logging_output


def plot_loss_curves(train_loss, dev_loss):
    plt.plot(list(range(train_loss)), train_loss, label='train')
    plt.plot(list(range(dev_loss)), dev_loss, label='dev')
    plt.savefig('training_curve.png')
    plt.close()

if torch.cuda.is_available():    #False:#
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

if __name__ == '__main__':
    # If there's a GPU available...
    #device = torch.device("cuda")

    parser = argparse.ArgumentParser()
    # two components of the training data
    parser.add_argument('--src_data', dest='src_data', help='Name of the SOURCE train data (clean)', required=True)
    parser.add_argument('--targ_data', dest='targ_data', help='Name of the TARGET train data (clean)', required=True)
    parser.add_argument('--dev_data', dest='dev_data', help='Name of the DEV data (clean)', required=True)
    parser.add_argument('--test_data', dest='test_data', help='Name of the TEST data (clean)', required=True)
    parser.add_argument('--train', dest='train', help='whether we are training', required=False, default=False)
    parser.add_argument('--eval', dest='eval', help='whether we are evaluating', required=False, default=False)

    parser.add_argument('--epochs', dest='epochs', help='number of epochs', required=False, default=5)
    parser.add_argument('--num_batches', dest='num_batches', help='number of batches during training', required=False, default=10)
    parser.add_argument('--save_to', dest='save_to', help='where to save to', required=False, default='/models/test.pt')

    args = parser.parse_args()

    #encoder_question = RobertaModel.from_pretrained("roberta-base")
    #encoder_context = RobertaModel.from_pretrained("roberta-base")
    encoder_question = DistilBertModel.from_pretrained("distilbert-base-uncased")
    encoder_context = DistilBertModel.from_pretrained("distilbert-base-uncased")
    model = DPR(encoder_context, encoder_question).to(device)





    if args.train: 
        print('training!')
        src_dset = SimpleDataset(args.src_data)
        src_dataloader = DataLoader(src_dset, batch_size= 4, collate_fn=collate_fn) # int(len(src_dset)/int(args.num_batches))
        
        targ_dset = SimpleDataset(args.targ_data)
        targ_dataloader = DataLoader(targ_dset, batch_size= 4, collate_fn=collate_fn) #int(len(targ_dset)/int(args.num_batches))
        
        dev_dset = SimpleDataset(args.targ_data)
        dev_dataloader = DataLoader(dev_dset, batch_size= 30, collate_fn=collate_fn) #int(len(src_dset)/100)
        
        #print('batch size:', int(len(dev_dset)/10))

        train_accuracy = []
        dev_accuracy = []
        # training loop
        for i in range(args.epochs):
            print('epoch', i)
            print('  --> training step...')
            train_loss, train_acc = train(model, src_dataloader, targ_dataloader)
            print('  --> dev set evaluation...')
            dev_loss, dev_acc, __, __ = evaluate(model, dev_dataloader)

            train_accuracy.append(train_acc)
            dev_accuracy.append(dev_acc)
        #print('plotting loss curves')
        #plot_loss_curves(train_accuracy, dev_accuracy)
        print('saving the model!')
        torch.save(model.state_dict(), args.save_to)

    if args.eval:
        print('evaluation!')
        test_dataloader = DataLoader(SimpleDataset(args.test_data), batch_size= 2, collate_fn=collate_fn)#int(len(args.targ_data)/10))

        __, __, total_preds, total_labels = evaluate(model, test_dataloader)
        print('\n\n See test set evaluation below.')
        print(classification_report(total_labels, total_preds))




#labels = 

