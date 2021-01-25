import os 
import sys
import json
import logging
import argparse
import time
import pdb
import random
if os.path.exists("external_libraries"):
    sys.path.append('external_libraries')

# third-part libraries
import numpy as np
from apex import amp
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, DistributedSampler
import transformers
from transformers import BertModel,BertTokenizer,AlbertTokenizer,RobertaTokenizer,XLNetTokenizer
from transformers import Trainer, TrainingArguments
from transformers import AdamW,get_linear_schedule_with_warmup
from transformers.modeling_utils import SequenceSummary

from processor import *
from model import *
from modeling_linear import *

# This is ther trainer version of KRD model
def clean_data(cs_str):
    if "The statement " in cs_str:
        cs_str = cs_str.replace("The statement ","")
        cs_str = cs_str.replace(".\" is true because "," because ")
        if " helps answer the question " in cs_str:
            cs_str = cs_str.split(" helps answer the question ")
            cs_str = cs_str[1]+" because "+cs_str[0]
        # cs_str = cs_str.replace(" helps answer the question ",)
        cs_str = cs_str.replace("\"","")
    if "Another way to say" in cs_str:
        cs_str = cs_str.replace("Another way to say ","")
        cs_str = cs_str.replace("\"","")
    if "To understand the event " in cs_str:
        cs_str = cs_str.replace("To understand the event ","")
        cs_str = cs_str.replace(".\", it is important to know that "," because ")
    return cs_str


def clean_omcs(file_name):
    corpus = []
    with open(file_name,'r',encoding="utf8") as f:
        for line in tqdm(f,desc="omcs corpus"):
            if line == "id	text	creator_id	created_on	language_id	activity_id	score\n":
                continue
            if line == "(898159 rows)\n":
                continue
            data = line.split("\t")
            # pdb.set_trace()
            try:
                # cs += len(data[1].split(" "))
                # count += 1
                if data[-3] == "en":
                    # pdb.set_trace()
                    cs_str = clean_data(data[1])
                    corpus.append(cs_str)
            except :
                print(data)
    # print(cs/float(count))
    corpus = list(set(corpus))
    return corpus


def select_tokenizer(args):
    if "albert" in args.origin_model:
        return AlbertTokenizer.from_pretrained(args.origin_model)
    elif "roberta" in args.origin_model:
        return RobertaTokenizer.from_pretrained(args.origin_model)
    elif "bert" in args.origin_model:
        return BertTokenizer.from_pretrained(args.origin_model)
    elif "xlnet" in args.origin_model:
        return XLNetTokenizer.from_pretrained(args.origin_model)


def select_model(args,model_name = None):
    if not model_name:
        model_name = args.origin_model
        cache = os.path.join(args.output_dir,"cache")
    else:
        cache = None
    if args.task_name == "rerank_csqa":
        if "albert" in model_name:
            return AlbertAttRanker.from_pretrained(model_name,cache_dir = cache,cs_len = args.cs_len)
        elif "roberta" in model_name:
            return RobertaAttRanker.from_pretrained(model_name,cache_dir = cache,cs_len = args.cs_len)
        elif "bert" in model_name:
            return BertAttRanker.from_pretrained(model_name,cache_dir = cache,cs_len = args.cs_len)
        elif "xlnet" in model_name:
            return XLNetAttRanker.from_pretrained(model_name,cache_dir = cache,cs_len = args.cs_len)
    elif args.task_name == "rerank_csqa_without_rerank":
        if "bert" in model_name:
            return BertCSmean.from_pretrained(model_name,cache_dir = cache,cs_len = args.cs_len)
    elif args.task_name == "KRD_linear":
            return BertForLinearKRD.from_pretrained(model_name,cache_dir = cache,cs_len = args.cs_len,cs_seq_len = args.cs_seq_len, query_len = args.question_seq_len + args.answer_seq_len)

    else:
        if "albert" in model_name:
            return AlbertForMultipleChoice.from_pretrained(model_name,cache_dir = cache)
        elif "roberta" in model_name:
            return RobertaForMultipleChoice.from_pretrained(model_name,cache_dir = cache)
        elif "bert" in model_name:
            return BertForMultipleChoice.from_pretrained(model_name,cache_dir = cache)
        elif "xlnet" in model_name:
            return XLNetForMultipleChoice.from_pretrained(model_name,cache_dir = cache)
        
def set_seed(args):
    logger.info("Freeze seed : {}".format(str(args.seed)))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def train(args):
    output_dir = os.path.join(args.output_dir,args.save_model_name)
    logging_dir = os.path.join(output_dir,"logging")

    # loading data
    omcs_corpus = load_omcs(args)
    tokenizer = select_tokenizer(args)

    train_dataset = load_csqa_omcs_dataset(tokenizer,args,omcs_corpus,"train",is_training=True)
    dev_dataset = load_csqa_omcs_dataset(tokenizer,args,omcs_corpus,"dev",is_training=True)


    # setup trainig arguments
    save_steps = len(train_dataset)//(args.per_device_train_batch_size*args.gradient_accumulation_steps)
    print("save model in {} steps".format(save_steps))
    training_args = TrainingArguments(
        output_dir = output_dir,          # output directory
        num_train_epochs = args.num_train_epochs,              # total number of training epochs
        per_device_train_batch_size = args.per_device_train_batch_size,  # batch size per device during training
        per_device_eval_batch_size = args.per_device_eval_batch_size,   # batch size for evaluation
        warmup_steps = args.warmup_steps,                # number of warmup steps for learning rate scheduler
        weight_decay= args.weight_decay,               # strength of weight decay
        logging_dir = logging_dir,            # directory for storing logs
        logging_steps = args.logging_steps,
        evaluation_strategy = "epoch",
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        fp16 = args.fp16,
        save_steps = save_steps
    )
    
    

    # setup compute_metrics, note that this kind of outputs must come from transformers library
    def compute_metrics(eval_predictions):
        predictions, label_ids = eval_predictions
        preds = np.argmax(predictions, axis=1)
        return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}
    model = select_model(args)
    # setup trainer
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=dev_dataset,             # evaluation dataset
        compute_metrics = compute_metrics,
    )
    # Train!
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data arguments
    parser.add_argument("--data_dir",type = str,default = "dataset/CSQA")
    parser.add_argument("--cs_dir",type = str, default = "OMCS")
    parser.add_argument("--test_file",type= str,default = "test_rand_split_no_answers.jsonl")
    parser.add_argument("--dev_file",type= str,default = "dev_rand_split.jsonl")
    parser.add_argument("--train_file",type = str,default = "train_rand_split.jsonl")
    parser.add_argument("--output_dir",type = str,default = "model")
    parser.add_argument("--save_model_name",type = str,default = "bert_csqa_2e-5_wholeQA-Match_cslen5")
    parser.add_argument("--tokenizer_name_or_path",type = str,default = "bert-base-cased")
    parser.add_argument("--origin_model",type = str,default = "bert-base-cased", help = "origin model dir for training")
    parser.add_argument("--omcs_file",type=str,default = "omcs-free-origin.json")
    # hyper parameters
    parser.add_argument("--max_length",type=int,default = 80 )
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1,help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_train_epochs",default=5,type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--per_device_train_batch_size", default=15, type=int, help="Batch size for training.")
    parser.add_argument("--per_device_eval_batch_size", default=6, type=int, help="Batch size for eval.")
    parser.add_argument("--logging_steps",default = 400,type = int,help = "output current average loss of training")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--cs_seq_len", default = 20, type = int, help= "Max length of cs")
    parser.add_argument("--question_seq_len", default = 64, type = int , help = "Max length of question")
    parser.add_argument("--answer_seq_len",default = 16, type = int, help = "Max length of answer")
    parser.add_argument("--cs_len",type = int, default = 5)
    parser.add_argument("--dev_cs_len",type = int, default = 0)
    # settings
    parser.add_argument("--n_gpu",type=int , default = 1)
    parser.add_argument("--fp16",action = "store_true")
    parser.add_argument("--save_method",type = str,default = "Best_Current")
    parser.add_argument("--do_finetune",action = "store_true",default = False)
    parser.add_argument("--cs_mode",type = str,default = "wholeQA-Match")
    parser.add_argument("--cs_save_mode",type = str,default = "id")
    parser.add_argument("--seed",type = int,default = None,help = "freeze seed")
    parser.add_argument('--tpu',action = "store_true")
    parser.add_argument('--task_name',type = str, default = "baseline")
    parser.add_argument("--test",action = "store_true")
    parser.add_argument("--dev",action = "store_true")

    args = parser.parse_args()
    if args.test:
        eval(args,"test")
    elif args.dev:
        eval(args,"dev")
    else:
        train(args)
