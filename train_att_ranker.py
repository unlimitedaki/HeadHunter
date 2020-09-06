import os 
import sys
if os.path.exists("external_libraries"):
    sys.path.append('external_libraries')
import torch
import transformers
import json
from transformers import BertModel,BertTokenizer
from tqdm import tqdm
import logging
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import argparse
import time
import torch.nn as nn
from transformers import AdamW,get_linear_schedule_with_warmup
from transformers.modeling_utils import SequenceSummary
import pdb
import numpy as np
from attRankerProcessor import *
from model import *
from apex import amp

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



def train(args):
    # setup output dir for model and log
    output_dir = os.path.join(args.output_dir,args.save_model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # setup logging
    logfilename = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" "+args.save_model_name+".log.txt"
    fh = logging.FileHandler(os.path.join(output_dir,logfilename), mode='a', encoding='utf-8')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # setup tokenizer
    # if args.tokenizer_name_or_path:
    #     tokenizer_name_or_path = args.tokenizer_name_or_path
    # else:
    #     tokenizer_name_or_path = "bert-base-cased"
    # tokenizer = BertTokenizer.from_pretrained(tokenizer_name_or_path)
    tokenizer = BertTokenizer.from_pretrained(args.origin_model)

    # load data
    if args.cs_len > 0:
        # omcs_corpus = None
        omcs_corpus = load_omcs(args)
        _,_,train_dataset= load_csqa_omcs_dataset(tokenizer,args,omcs_corpus,"train")
        dev_examples,_,dev_dataset= load_csqa_omcs_dataset(tokenizer,args,omcs_corpus,"dev")
        # _,_,test_dataset= load_csqa_omcs_dataset(tokenizer,args,omcs_corpus,"test",is_training = False)

    else:
        _,_,train_dataset= load_csqa_dataset(tokenizer,args,"train")
        _,_,dev_dataset= load_csqa_dataset(tokenizer,args,"dev")
        # _,_,test_dataset= load_csqa_dataset(tokenizer,args,"test",is_training = False)
    
    train_sampler = RandomSampler(train_dataset) 
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    dev_sampler = SequentialSampler(dev_dataset) 
    dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=args.eval_batch_size)

    # test_sampler = SequentialSampler(test_dataset) 
    # test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size)

    # load model
    if args.do_finetune:
        status_dir = os.path.join(output_dir,"status.json")
        status = json.load(open(status_dir,'r'))
        current_model = os.path.join(output_dir, "current_model")
        model = BertAttRanker.from_pretrained(current_model)
        
    else:
        cache = os.path.join(args.output_dir,"cache")
        model = BertAttRanker.from_pretrained(args.origin_model,cache_dir = cache,cs_len = args.cs_len)
        status = {}
        status['best_epoch'] = 0
        status['best_Acc'] = 0.0
        status['current_epoch']  = 0
        
    device = torch.device('cuda:0')
    model.to(device)

    # Prepare optimizer and schedule (linear warmup and decay)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    
    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1") 

    ## Training 
    model.zero_grad()
    epochs_trained = 0
    train_iterator = tqdm(range(epochs_trained, int(args.num_train_epochs)), desc="Epoch")
    
    

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        tr_loss = 0
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": batch[3]
            }
            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step() 
                model.zero_grad()
            # check average training loss
            if (step + 1)% args.check_loss_step == 0 or step == len(train_dataloader):
                avg_loss = tr_loss/(step+1)
                logger.info("\t average_step_loss=%s @ step = %s on epoch = %s",str(avg_loss),str(step+1),str(epoch+1))
            # break
        # Eval : one time one epoch
        torch.cuda.empty_cache() # release cuda cache so that we can eval 
        acc,predictions = eval(args,model,dev_dataloader,"dev",device,len(dev_dataset))
        result_json = make_predictions(args,dev_examples,predictions,omcs_corpus)
        logger.info("Accuracy : {} on epoch {}".format(acc,epoch))
        if args.save_method == "Best_Current":
            if acc > status['best_Acc']:
                status['best_Acc'] = acc.cpu().numpy().tolist()
                status['best_epoch'] = epoch
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                best_model_dir = os.path.join(output_dir,"best_model")
                # output_dir = os.path.join(output_dir, 'checkpoint-{}'.format(epoch + 1))
                if not os.path.exists(best_model_dir):
                    os.makedirs(best_model_dir)
                model_to_save.save_pretrained(best_model_dir)
                logger.info("best epoch %d has been saved to %s",epoch,best_model_dir)
                prediction_file = os.path.join(best_model_dir,"{}_{}_{}_prediction_file.json".format("dev",args.cs_mode,args.cs_len))
                with open(prediction_file,'w',encoding= 'utf8') as f:
                    json.dump(result_json,f,indent = 2,ensure_ascii = False)
            # save model 
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model itself
            current_model_dir = os.path.join(output_dir,"current_model")
            if not os.path.exists(current_model_dir):
                os.makedirs(current_model_dir)
            model_to_save.save_pretrained(current_model_dir)
            logger.info("epoch %d has been saved to %s",epoch,current_model_dir)
            # save predictions
            prediction_file = os.path.join(current_model_dir,"{}_{}_{}_prediction_file.json".format("dev",args.cs_mode,args.cs_len))
            with open(prediction_file,'w',encoding= 'utf8') as f:
                json.dump(result_json,f,indent = 2,ensure_ascii = False)
        else:
            # save model of every epoch
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model itself
            output_dir = os.path.join(args.output_dir,args.save_model_name)
            current_model_dir = os.path.join(output_dir,"check_point_{}".format(str(epoch)))
            if not os.path.exists(current_model_dir):
                os.makedirs(current_model_dir)
            model_to_save.save_pretrained(current_model_dir)
            logger.info("epoch %d has been saved to %s",epoch,current_model_dir)
        # save status
        status_dir = os.path.join(output_dir,"status.json")
        json.dump(status,open(status_dir,'w',encoding = 'utf8'))

def make_predictions(args,examples,predictions,omcs_corpus,data_type="dev"):
    cs_file = "{}_{}_omcs_of_dataset.json".format(data_type,args.cs_mode)
    cs_file = os.path.join(args.cs_dir,cs_file)
    with open(cs_file,'r',encoding="utf8") as f:
        cs_data = json.load(f)
    # put result into examples
    bad_prediction = []
    pred_index = 0 #because it's sequential, we can just index it with examples
    result_json = {}
    for example in tqdm(examples,desc="puting result into examples"):
        try:
            result_json[example.id] = {}
            result_json[example.id]['question'] = example.question
            # result_json[example.id]['endings'] = []
            result_json[example.id]['prediction'] = predictions[pred_index]
            result_json[example.id]["prediction_answer"] = example.endings[predictions[pred_index]]
            pred_index += 1
            result_json[example.id]['label'] = example.label
            example_cs = cs_data[example.id]
            result_json[example.id]['endings'] = example_cs['endings']
            for ending in result_json[example.id]['endings']:
                if args.cs_save_mode == 'id':
                    ending["cs"] = [omcs_corpus[int(id)] for id in ending["cs"][:args.cs_len]]
                else:
                    ending['cs'] = ending["cs"][:args.cs_len]
        except Exception as ex:
            bad_pred = {}
            bad_pred['example_id']= example.id
            bad_pred['prediction'] = predictions[pred_index]
            bad_prediction.append(bad_pred)
    
    bad_prediction_file = "{}_{}_bad_prediction.json".format(data_type,args.cs_mode)
    bad_prediction_file = os.path.join(os.path.join(args.output_dir,args.save_model_name))
    bad_prediction_file  = open(bad_prediction_file,'w',encoding = 'utf8')
    json.dump(bad_prediction_file, bad_prediction, indent = 2,ensure_ascii = False)
    return 

def eval(args,model,dataloader,set_name,device,num_examples):
    # pdb.set_trace()
    torch.cuda.empty_cache()
    logger.info("Evaluate on {}".format(set_name))
    iterator = tqdm(dataloader, desc="Iteration")
    correct_count = 0
    predictions = []
    # total = 0
    for step,batch in enumerate(iterator):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "labels": batch[3]
        }
        outputs = model(**inputs)
        logits = outputs[1]
        prediction = torch.argmax(logits,axis = 1)
        correct_count += (prediction == batch[3]).sum().float()
        predictions += prediction.cpu().numpy().tolist()
    return correct_count/num_examples, predictions

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # %cd /content/drive/Shared drives/aki-lab/experiment/IR_CSQA



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
    parser.add_argument("--train_batch_size", default=15, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=6, type=int, help="Batch size for eval.")
    parser.add_argument("--check_loss_step",default = 400,type = int,help = "output current average loss of training")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--cs_len",type = int, default = 5)
    # settings
    parser.add_argument("--n_gpu",type=int , default = 1)
    parser.add_argument("--fp16",type = bool,default = True)
    parser.add_argument("--save_method",type = str,default = "Best_Current")
    parser.add_argument("--do_finetune",action = "store_true",default = False)
    parser.add_argument("--cs_mode",type = str,default = "wholeQA-Match")
    parser.add_argument("--cs_save_mode",type = str,default = "id")

    # args = parser.parse_args() 在notebook 里 args 需要初始化为[],外部调用py文件不需要
    args = parser.parse_args()

    train(args)
