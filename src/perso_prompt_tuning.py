import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.optimization import Adafactor, AdamW
import pandas as pd
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.translate import meteor
import nltk
from tensorboardX import SummaryWriter
import numpy as np
import pickle
import re
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from torch import nn
import random
import sys
from tqdm import tqdm
import os
import gc
from datetime import datetime


def cleancuda():
    with torch.no_grad():
        torch.cuda.empty_cache()
    gc.collect()

def build_hl_dataset(news,pers,with_tag=None):
    all_stl_src_hl = []
    all_tgt_hl = []
    read_news = news['Headline']
    tag = ''
    if not with_tag is None:
        tag = with_tag
        read_news = news['News body']
    for i,v in pers.iterrows():
        _,_,src_id,tgt_hl = v
        src_id = src_id.split(',')
        tgt_hl = tgt_hl.split(';;')
        for j in range(len(src_id)):
            ids = set(range(len(src_id))) - {j}
            sample_id = random.sample(list(ids),50)
            all_stl_src_hl.append(' ;; '.join([tgt_hl[id] for id in sample_id[:30]]) + ' || ' + tag + read_news[src_id[j]])
            #print(len(all_stl_src_hl[-1]),all_stl_src_hl[-1])
            all_tgt_hl.append(tgt_hl[j])
            #print(all_tgt_hl)
    return all_stl_src_hl,all_tgt_hl


def save(model, optimizer, path):
    # save
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

class HeadlinesDataset(Dataset):
    def __init__(self, stl_src_hl, tgt_hl):
        self.stl_src_hl = stl_src_hl       
        self.tgt_hl = tgt_hl

    def __getitem__(self, idx):
        return [self.stl_src_hl[idx], self.tgt_hl[idx]]
    
    def __len__(self):
        return len(self.tgt_hl)

#def news_collate_fn(news_info):
#    news_info = [torch.LongTensor(i) for i in zip(*news_info)]
#    return news_info

class PromptOnlyModel(nn.Module):
    def __init__(self,prompt_length,dim_out,init_type=1):
        super(PromptOnlyModel,self).__init__()
        self.embedder = nn.Linear(dim_out,prompt_length,bias=False)
        self.prompt_length = prompt_length

    def forward(self):
        return self.embedder.weight

    def len(self):
        return self.prompt_length



# ===== LOADING MODEL =====
MODEL_TYPE = sys.argv[1]
PROMPT_LENGTH = int(sys.argv[2])
BATCH_SIZE = int(sys.argv[3])
MODEL_PATH = sys.argv[4]
PROMPT_PATH = sys.argv[5]
TAG = 'summarize: '

now = datetime.now().strftime("%d%m%y-%Hh%M")
print(now)

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if torch.cuda.is_available():
    dev = torch.device("cuda:0") 
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    print("Running on the CPU")

cleancuda()

news = pd.read_csv('../data/pens/news.tsv',sep='\t',index_col='News ID')
pers = pd.read_csv('../data/pens/personalized_test.tsv',sep='\t')
all_stl_src_hl,all_tgt_hl = build_hl_dataset(news,pers,TAG)

first_split = 18000
last_split = 20000
traindt = HeadlinesDataset(all_stl_src_hl[:first_split], 
                            all_tgt_hl[:first_split])
testdt  = HeadlinesDataset(all_stl_src_hl[first_split:last_split], 
                            all_tgt_hl[first_split:last_split])
evaldt  = HeadlinesDataset(all_stl_src_hl[last_split:], 
                            all_tgt_hl[last_split:])

del news
del pers
del all_stl_src_hl
del all_tgt_hl

tokenizer = T5Tokenizer.from_pretrained(MODEL_TYPE,model_max_length=512)
model = T5ForConditionalGeneration.from_pretrained(MODEL_TYPE, return_dict=True)
model = model.to(dev)

prompt = PromptOnlyModel(PROMPT_LENGTH,model.shared.embedding_dim)

optimizer = Adafactor(
    prompt.parameters(), 
    lr=0.001,
    eps=(1e-30, 1e-3),
    clip_threshold=1.0,
    decay_rate=-0.8,
    beta1=None,
    weight_decay=0.0,
    relative_step=False,
    scale_parameter=False,
    warmup_init=False)

if MODEL_PATH != '.':
    checkpoint = torch.load(PROMPT_PATH, map_location='cpu')
    prompt.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
else:
    MODEL_PATH = f"models/{now}-{MODEL_TYPE.replace('/','-')}_{PROMPT_LENGTH}/"
print('Writing model to', MODEL_PATH)
prompt = prompt.to(dev)

print('Done loading model...')
with torch.no_grad():
    print(prompt())

#optimizer = optimizer.to(dev)

# ===== FREEZE T5 MODEL =====
for p in model.parameters():
    p.requires_grad = False
#print(sum([1 for p in model.parameters() if p.requires_grad]))
#print(sum([1 for p in prompt.parameters() if p.requires_grad]))

# ===== TRAINING PROMPT =====
writer = SummaryWriter()

def trainprompt(model, tokenizer, optimizer, prompt, traindt, testdt, batch_size=8, startep=0, endep=3 ,evaldt=None, model_path=MODEL_PATH):
    if model_path is not None and not os.path.exists(model_path):
        os.makedirs(model_path)
    trainld = DataLoader(traindt, batch_size=batch_size, shuffle=True)
    testld = DataLoader(testdt, batch_size=batch_size, shuffle=True)
    
    for e in range(startep,endep):
        running_loss = 0
        print('EPOCH ',e,':')
        tqdm_util = tqdm(enumerate(trainld))
        for i,(x,y) in tqdm_util:
            cleancuda()
            #tokenizing
            with torch.no_grad():
                srcbatch = tokenizer(list(x),padding='longest',truncation=True,max_length=2000,return_tensors='pt')
                tgtbatch = tokenizer(list(y),padding='longest',truncation=True,max_length=300,return_tensors='pt')
                
                inputbatch, attentionbatch = srcbatch.input_ids, srcbatch.attention_mask
                
                outputbatch = tgtbatch.input_ids
                outputbatch = outputbatch.clone().detach()
                outputbatch[outputbatch == tokenizer.pad_token_id] = -100 #replace padding with embedded padding (-100)
                del srcbatch
                del tgtbatch
                attentionbatch = torch.hstack([torch.tensor([[1]*prompt.len()]*inputbatch.shape[0]),attentionbatch])

                #to GPU
                attentionbatch = attentionbatch.to(dev)
                inputbatch = inputbatch.to(dev)
                outputbatch = outputbatch.to(dev)

                #get T5 embedding and concat to input

                embedbatch = model.encoder.embed_tokens(inputbatch)

            promptbatch = torch.stack([prompt()]*embedbatch.shape[0])
            stackedbatch = torch.hstack([promptbatch,embedbatch])
            stackedbatch = stackedbatch.to(dev)

            res = model(inputs_embeds=stackedbatch,labels=outputbatch,attention_mask=attentionbatch)
            del inputbatch
            del attentionbatch
            del outputbatch
            del promptbatch
            del embedbatch
            del stackedbatch

            loss = res.loss
            loss_num = loss.item()/len(x)
            writer.add_scalar('Loss/train', loss_num, e*len(trainld)+i)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                tqdm_util.set_description('train_loss: {:.5f}'.format(loss_num))
                if model_path is not None:
                    save(prompt, optimizer, model_path+f'prompt_{now}_{PROMPT_LENGTH}.pth')
            #if i > 2:
            #    break
        if model_path is not None:
            save(prompt, optimizer, model_path+f'prompt_{now}_{PROMPT_LENGTH}.pth')

        # ===== TESTING =====
        with torch.no_grad():
            # generate test
            cleancuda()
            sample_id = random.sample(range(len(evaldt)),2)
            testgen = [evaldt[n][0] for n in sample_id]
            target = [evaldt[n][1] for n in sample_id]
            input_ids = tokenizer(testgen, return_tensors="pt",padding='longest',
                                        truncation=True,max_length=2000).input_ids
            input_ids = input_ids.to(dev)
            embedbatch = model.encoder.embed_tokens(input_ids)
            promptbatch = torch.stack([prompt()]*embedbatch.shape[0])
            stackedbatch = torch.hstack([promptbatch,embedbatch])
            output = model.generate(inputs_embeds=stackedbatch,max_length=100)
            output = tokenizer.batch_decode(output)

            for i in range(len(testgen)):
                print('Input::: ',testgen[i])
                print('-'*50)
                print('Generated::: ',output[i])
                print('Perso headline::: ',target[i])
                print('='*100)
            
            # get test score
            tqdm_util = tqdm(enumerate(testld))
            for i,(x,y) in tqdm_util:
                cleancuda()
                #tokenizing
                srcbatch = tokenizer(list(x),padding='longest',truncation=True,max_length=2000,return_tensors='pt')
                tgtbatch = tokenizer(list(y),padding='longest',truncation=True,max_length=300,return_tensors='pt')
                
                inputbatch, attentionbatch = srcbatch.input_ids, srcbatch.attention_mask
                
                outputbatch = tgtbatch.input_ids
                outputbatch = outputbatch.clone().detach()
                outputbatch[outputbatch == tokenizer.pad_token_id] = -100 #replace padding with embedded padding (-100)
                
                attentionbatch = torch.hstack([torch.tensor([[1]*prompt.len()]*inputbatch.shape[0]),attentionbatch])
                del srcbatch
                del tgtbatch
                
                #to GPU
                attentionbatch = attentionbatch.to(dev)
                inputbatch = inputbatch.to(dev)
                outputbatch = outputbatch.to(dev)

                #get T5 embedding and concat to input
                embedbatch = model.encoder.embed_tokens(inputbatch)
                promptbatch = torch.stack([prompt()]*embedbatch.shape[0])
                stackedbatch = torch.hstack([promptbatch,embedbatch])
                stackedbatch = stackedbatch.to(dev)
                
                res = model(inputs_embeds=stackedbatch,labels=outputbatch,attention_mask=attentionbatch)
                del inputbatch
                del attentionbatch
                del outputbatch    
                del promptbatch
                del embedbatch
                del stackedbatch

                loss = res.loss
                loss_num = loss.item()/len(x)
                writer.add_scalar('Loss/test', loss_num, e*len(testld)+i)      

                if i % 10 == 0:
                    tqdm_util.set_description('test_loss: {:.5f}'.format(loss_num))
                #if i > 2:
                #    break         

trainprompt(model, tokenizer, optimizer, prompt, traindt, testdt, batch_size=BATCH_SIZE, startep=0, endep=5, evaldt=evaldt, model_path=MODEL_PATH)                                          
print('Done test...')


#checkpoint = torch.load(MODEL_PATH+'new_prompt.pth', map_location='cpu')

#prompt1 = PromptOnlyModel(PROMPT_LENGTH,model.shared.embedding_dim)
#prompt1.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#with torch.no_grad():
#    print(prompt1())

                            


