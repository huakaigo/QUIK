import os
import numpy as np
import torch
import random
import datasets
from datasets import load_from_disk
import transformers

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)

LOCAL_DATA_PATH="/home/liuhuakai/wksp/dataset"

def get_wikitext2(nsamples, seed, seqlen, model, hf_token):
    # traindata = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    # testdata = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    traindata = load_from_disk(os.path.join(LOCAL_DATA_PATH, 'calib', 'wikitext2'))
    testdata = load_from_disk(os.path.join(LOCAL_DATA_PATH, 'eval', 'wikitext2'))

    if hf_token is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False, use_auth_token=hf_token)
        
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, model, hf_token):
    # traindata = datasets.load_dataset('ptb_text_only', 'penn_treebank', split='train')
    # valdata = datasets.load_dataset('ptb_text_only', 'penn_treebank', split='validation')
    traindata = load_from_disk(os.path.join(LOCAL_DATA_PATH, 'calib', 'ptb'))
    valdata = load_from_disk(os.path.join(LOCAL_DATA_PATH, 'eval', 'ptb'))

    if hf_token is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False, use_auth_token=hf_token)
       
    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4(nsamples, seed, seqlen, model, synthetic_data=False, hf_token=None):    
    
    if not synthetic_data:
        print('Loading C4 Real dataset')
        # traindata = datasets.load_dataset(
        #     'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
        # )
        # valdata = datasets.load_dataset(
        #     'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
        # )
        traindata = load_from_disk(os.path.join(LOCAL_DATA_PATH, 'calib', 'c4', ))
        valdata = load_from_disk(os.path.join(LOCAL_DATA_PATH, 'eval', 'c4'))
    else:
        print('Loading C4 Synthetic dataset')
    
    if hf_token is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False, use_auth_token=hf_token)
       
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        if not synthetic_data:
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
        else:
            inp = torch.rand((1, 2048)).to(torch.long)
            tar = torch.ones_like(inp)
        trainloader.append((inp, tar))

    random.seed(0)
    if not synthetic_data:
        valenc = []
        for _ in range(256):
            while True:
                i = random.randint(0, len(valdata) - 1)
                tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
                if tmp.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            valenc.append(tmp.input_ids[:, i:j])
        valenc = torch.hstack(valenc)
    else:
        valenc = torch.randperm(524288).unsqueeze(0).to(torch.long)
    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc 


def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model='', synthetic_data=False, hf_token=None
):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model, hf_token)
    if 'ptb' in name:
        return get_ptb(nsamples, seed, seqlen, model, hf_token)
    if 'c4' in name:
        return get_c4(nsamples, seed, seqlen, model, synthetic_data, hf_token)

if __name__ == '__main__':
    import os
    os.environ['HTTP_PROXY'] = 'http://10.4.236.41:8888'
    os.environ['HTTPS_PROXY'] = 'http://10.4.236.41:8888'
    get_loaders('c4')