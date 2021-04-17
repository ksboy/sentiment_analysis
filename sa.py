# coding=utf-8

import argparse
import glob
import logging
import os
import random
import json
import numpy as np
import torch
from tqdm import tqdm, trange
import nltk
from nltk.tokenize import sent_tokenize
from scipy.special import softmax

from transformers import (
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

TOKENIZER_ARGS = [ "strip_accents", "keep_accents", "use_fast", "do_lower_case"]

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

class SA(object):
    def __init__(self, args):
        
        # print(args, )
        # Setup CUDA, GPU & distributed training
        if args.local_rank == -1 or args.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
            args.n_gpu = torch.cuda.device_count()
        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            torch.distributed.init_process_group(backend="nccl")
            args.n_gpu = 1
        args.device = device

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
        )
      
        # Set seed
        set_seed(args)

        tokenizer_args = {k: v for k, v in vars(args).items() if v is not None and k in TOKENIZER_ARGS}
        self.tokenizer = AutoTokenizer.from_pretrained(args.output_dir, **tokenizer_args)
        self.model = AutoModelForTokenClassification.from_pretrained(args.output_dir)
        self.model.to(args.device)
    
    def predict(self, args, sentence):
        # sentenceList = self.preprocess([sentence])
        # multi-gpu evaluate
        # fix the bug when using mult-gpu during evaluating
        if args.n_gpu > 1 and not isinstance(self.model, torch.nn.DataParallel):
            self.model = torch.nn.DataParallel(self.model)

        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer.batch_encode_plus([sentence], max_length=args.max_seq_length, pad_to_max_length=True)
            inputs['labels'] = [0]
            for key in inputs:
                inputs[key] = torch.tensor(inputs[key]).to(args.device)
            outputs = self.model(**inputs)
            _, logits = outputs[:2]
            preds = logits.detach().cpu().numpy()
        # preds = np.argmax(preds, axis=-1)
        preds = softmax(preds)
        return preds[0][1]
        
    
class ARGS():
    def __init__(self) -> None:
        self.no_cuda=False
        self.local_rank=-1
        self.max_seq_length=256
        self.model_type="bert"
        # self.model_name_or_path="/home/mhxia/whou/workspace/pretrained_models/roberta-large #roberta-large/",
        self.output_dir="./output/checkpoint-best/"
        self.seed=1

if __name__ == '__main__':
    args= ARGS()
    # print(args)
    sa = SA(args)
    res = sa.predict(args, "来江门出差一直都住这个酒店，觉得性价比还可以")
    print(res)
    
