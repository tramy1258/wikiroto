import json
import os
import sys
import getopt
import gc
import re
import torch
import torch.utils.data as data_utils
#from torch import optim, nn, utils, Tensor
#from torchvision.datasets import MNIST
import pytorch_lightning as pl
import pandas as pd
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
#from nltk.translate import meteor
#import nltk
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.optimization import Adafactor
from parent.parent import parent
from datetime import datetime
now = datetime.now().strftime("%d%m%y-%Hh%M")
model_path = './models/'
model_size = 't5-small'
if len(sys.argv) > 2:
    model_size = sys.argv[2]
if len(sys.argv) > 1 and 'models' not in sys.argv[1]:
    model_size = sys.argv[1]

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


######### DEFINING FUNCTIONS ###########

#--------- preprocessing -----------
def split_train_test_eval(data,first,last):
    traindt = data['entries'][:first]
    testdt  = data['entries'][first:last]
    evaldt  = data['entries'][last:]
    return traindt,testdt,evaldt

def preprocess_to_df(data):
    inputs = []
    targets = []
    infos = []
    consensus = []
    tables = []
    #neu = {'movie_info','critics_consensus','wiki_summary'}
    for d in data:
        dic = list(d.values())[0]
        cricon = dic.pop('critics_consensus')
        info  = dic.pop('movie_info')
        target = dic.pop('wiki_summary')
        input = [k+' | '+v for k,v in dic.items()]
        inputs.append('WikiRoto: '+' && '.join(input))
        table = [[k,re.split(' |,|\xa0',v)] for k,v in dic.items()]
        tables.append(json.dumps(table))
        targets.append(target)
        infos.append(info)
        consensus.append(cricon)
    return pd.DataFrame({'input':inputs,'target':targets,'table':tables,'movie_info':infos,'critics_consensus':consensus})


##### LIGHTNING #####
def df_to_loader(df,target,features,batch_size):
    # Creating np arrays
    target_values = df[target].values  # torch.tensor()
    feature_values = df[features].values  # torch.tensor()

    # Passing to DataLoader
    dataset = data_utils.TensorDataset(feature_values, target_values)
    loader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader
##### LIGHTNING #####
    
def save(epoch, i, model, optimizer, path):
    torch.save({
        'epoch': epoch,
        'batch': i,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)


#------------- cleaning memory ------------
def cleancuda():
    with torch.no_grad():
        torch.cuda.empty_cache()
    gc.collect()


#------------- tokenizing input batch -----------
def token_input(dt,tokenizer,i,batch_size):
    #print('Tokenizing!')
    with torch.no_grad():
        #encoding inputs and targets
        inputbatch = tokenizer(list(dt['new_input'][i*batch_size:i*batch_size+batch_size]),\
                            padding='longest',truncation=True,max_length=2000,return_tensors='pt')
        targetbatch = tokenizer(list(dt['target'][i*batch_size:i*batch_size+batch_size]),\
                            padding='longest',truncation=True,max_length=1000,return_tensors='pt')

        #extracting inputs,targets and adding index to ignore padding
        input_ids, attention_mask = inputbatch.input_ids, inputbatch.attention_mask

        labels = targetbatch.input_ids
        labels = labels.clone().detach()
        #labels = torch.tensor(labels)
        labels[labels == tokenizer.pad_token_id] = -100 #replace padding with embedded padding (-100)

    #to gpu if available
    inputbatch = input_ids.to(dev)
    attentionbatch = attention_mask.to(dev)
    targetbatch = labels.to(dev)
    #print('Done tokenizing!')

    return inputbatch,attentionbatch,targetbatch


#------------- training model ---------------
def train(model,optimizer,traindt,testdt,evaldt,evaluate,new_path,batch_size=16,num_of_epochs=1,start_epoch=0):
    num_of_train_batches = len(traindt)//batch_size
    num_of_test_batches = len(testdt)//batch_size
    print(f'{num_of_train_batches} training batches and {num_of_test_batches} of testing batches of size {batch_size}')


    for epoch in range(start_epoch,num_of_epochs):
        cleancuda()
        model.train()
        print("EPOCH {}".format(epoch))

        running_loss = 0
        #path = new_path+f'/e{epoch}_b{i}_model.pth'
        old_model_path = new_path+'/old_model.pth'
        new_model_path = new_path+'/new_model.pth'
        path = None
        #out = display(progress(1, num_of_train_batches),display_id=True)
        for i in range(num_of_train_batches):
            if i % 100 == 0:
                print(i)
            if i % 20 == 0:
                if path is None or path == new_model_path:
                    path = old_model_path
                elif path == old_model_path:
                    path = new_model_path
                else:
                    print('Really weird!!!!!!!')
                    return
                #path = new_path+f'/e{epoch}_b{i}_model.pth'
                #with torch.no_grad():
                #    print(f'i={i}:',end=' ')
                #    evaluate(model,traindt.loc[:0],verbose=True)
            cleancuda()
            inputbatch,attentionbatch,targetbatch = token_input(traindt,tokenizer,i,batch_size)

            #clearing old gradients
            optimizer.zero_grad()

            #forwarding
            #print('Forwarding...')
            outputs = model(input_ids=inputbatch, attention_mask=attentionbatch, labels=targetbatch)
            loss = outputs.loss
            loss_num = float(loss.item())
            running_loss+=loss_num
            #out.update(progress(loss_num, i, num_of_train_batches+1))
            writer.add_scalar('Loss/train', loss_num, epoch*num_of_train_batches+i)

            #backwarding
            loss.backward()

            #updating model
            optimizer.step()

            save(epoch, i, model, optimizer, path)

            inputbatch = None
            attentionbatch = None
            targetbatch = None
            #del inputbatch
            #del attentionbatch
            #del targetbatch

        running_loss=running_loss/int(len(traindt))
        print('Running loss during training: {}'.format(running_loss))
        #if epoch % 3 == 0:
        #    save(epoch, i, model, optimizer, new_path+f'/epoch_{epoch}.pth')

        model.eval()
        # test model
        #out = display(progress(1, num_of_test_batches),display_id=True)
        with torch.no_grad():
            running_loss = 0
            for i in range(num_of_test_batches):
                cleancuda()
                inputbatch, attentionbatch, targetbatch = token_input(testdt, tokenizer, i, batch_size)

                #forwarding
                outputs = model(input_ids=inputbatch, attention_mask=attentionbatch, labels=targetbatch)
                loss = outputs.loss
                loss_num = loss.item()
                running_loss+=loss_num
                #out.update(progress(loss_num, i, num_of_test_batches+1))
                writer.add_scalar('Loss/test', loss_num, epoch*num_of_test_batches+i)
            running_loss=running_loss/int(len(testdt))
            print('Running loss during testing: {}'.format(running_loss))

        # eval model
        cleancuda()
        blscore, prscore = evaluate(model, evaldt, 10)
        #writer.add_scalar('bleu', blscore, epoch)
        #print(f'BLEU = {blscore}')
        print(f'BLEU = {blscore} \n PARENT score: accuracy = {prscore[0]}, recall = {prscore[1]}, f-score = {prscore[2]}')
    print('TRAINING ALL DONE!')

#------------ evaluating model ------------
def evaluate(model,dt,partition=1,verbose=False,n_jobs=2,input_path=None,output_path=None,lambda_weights=[0.5]):
    bleu_avg = 0
    prec_avg = [0]*len(lambda_weights)
    reca_avg = [0]*len(lambda_weights)
    f_avg = [0]*len(lambda_weights)
    bsize = len(dt)//partition
    
    for i in range(partition):
        cleancuda()
        with torch.no_grad():
            input_ids = tokenizer(list(dt['new_input'][i*bsize:(i+1)*bsize]),
                                    return_tensors="pt",padding='longest',
                                    truncation=True,max_length=2000).input_ids
            input_ids = input_ids.to(dev)
            output = model.generate(input_ids,max_length=1000)
            output = tokenizer.batch_decode(output)
            output = [o.replace('<pad>','').replace('</s>','').strip() for o in output]
            target = list(dt['target'][i*bsize:(i+1)*bsize])
            table  = list(dt['table'][i*bsize:(i+1)*bsize])
            if verbose:
                for j in range(len(output)):
                    print('---->',output[j])
                    print(target[j])
                    print(table[j])

            target_p = [t.split(' ') for t in target]
            target_b = [[t.split(' ')] for t in target]
            output = [o.split(' ') for o in output]
            table  = [json.loads(t) for t in table]

            bleu_avg += corpus_bleu(target_b,
                                   output,
                                   smoothing_function=SmoothingFunction().method4)
            
            for i in range(len(lambda_weights)):
                precision, recall, f_score = parent(
                                                output,
                                                target_p,
                                                table,
                                                lambda_weight=lambda_weights[i],
                                                avg_results=True,
                                                n_jobs=n_jobs,
                                                use_tqdm='notebook'
                                                )
                prec_avg[i] += precision
                reca_avg[i] += recall
                f_avg[i] += f_score
    bleu_fin = round(bleu_avg/partition,5)
    prec_fin = [round(score/partition,5) for score in prec_avg]
    reca_fin = [round(score/partition,5) for score in reca_avg]
    f_fin =  [round(score/partition,5) for score in f_avg]
    return bleu_fin, (prec_fin, reca_fin,f_fin)


##### SETTING HYPERPARAMETERS ######
batch_size=4
num_of_epochs=10

##### LOADING AND PREPROCESSING DATA #####
'''
with open('../data/wikiroto_onesentence.json', encoding='utf-8') as json_file:
    finaldata = json.load(json_file)
traindt,testdt,evaldt = split_train_test_eval(finaldata,13000,14884)
print('Number of entries for training:',len(traindt))
print('Number of entries for testing:',len(testdt))
print('Number of entries for evaluating:',len(evaldt))
traindf = preprocess_to_df(traindt)
testdf = preprocess_to_df(testdt)
evaldf = preprocess_to_df(evaldt)

os.makedirs('../data/', exist_ok=True)  
traindf.to_csv('../data/wikiroto_train_with_table.csv',index_label=None) 
testdf.to_csv('../data/wikiroto_test_with_table.csv',index_label=None) 
evaldf.to_csv('../data/wikiroto_eval_with_table.csv',index_label=None) 
'''
traindf = pd.read_csv('../data/wikiroto_train_with_table.csv')
testdf = pd.read_csv('../data/wikiroto_test_with_table.csv')
evaldf = pd.read_csv('../data/wikiroto_eval_with_table.csv')
print(len(evaldf))

##### LOADING MODEL #####
if torch.cuda.is_available():
    dev = torch.device("cuda:0") 
    map_location=lambda storage, loc: storage.cuda()
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    map_location='cpu'
    print("Running on the CPU")

tokenizer = T5Tokenizer.from_pretrained(model_size,model_max_length=2000)
model = T5ForConditionalGeneration.from_pretrained(model_size, return_dict=True)
model = model.to(dev)

optimizer = Adafactor(
    model.parameters(),
    lr=1e-3,
    eps=(1e-30, 1e-3),
    clip_threshold=1.0,
    decay_rate=-0.8,
    beta1=None,
    weight_decay=0.0,
    relative_step=False,
    scale_parameter=False,
    warmup_init=False
)
epoch = 0
i = 0
if len(sys.argv) > 2 or (len(sys.argv) > 1 and 'models' in sys.argv[1]):
    checkpoint = torch.load(sys.argv[1], map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    #i = checkpoint['i']
print(f'Done loading {model_size}!')
#print(f'i = {i}, epoch = {epoch}')


##### TRAINING MODEL #####
#------------ saving model -------------
new_path = model_path + now + '-' + model_size
if not os.path.exists(new_path):
    os.makedirs(new_path)
train(model,optimizer,traindf,testdf,evaldf,evaluate,new_path,batch_size=batch_size,num_of_epochs=num_of_epochs,start_epoch=epoch)


##### EVALUATING MODEL #####
#print('Eval on some samples...')
#blscore,prec,reca,fscore = evaluate(model, evaldf.sample(n=5), verbose=True)#, lambda_weights=[0.5,0.0,1.0])
#print(f'BLEU = {blscore}, PARENT =',prec,reca,fscore)
#print('Eval on whole dataset...')
#blscore,prec,reca,fscore = evaluate(model, evaldf.sample(len(evaldf)), n_jobs=32, partition=200)#, lambda_weights=[0.5,0.0,1.0])
#print(f'BLEU = {blscore}, PARENT =',prec,reca,fscore)