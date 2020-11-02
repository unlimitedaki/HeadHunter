import os 
import sys
if os.path.exists("external_libraries"):
    sys.path.append('external_libraries')
import torch
import transformers
import json
from transformers import BertModel,BertTokenizer,AlbertTokenizer,RobertaTokenizer,XLNetTokenizer
from tqdm.notebook import tqdm
import logging
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, DistributedSampler
import argparse
import time
import torch.nn as nn
from transformers import AdamW,get_linear_schedule_with_warmup
from transformers.modeling_utils import SequenceSummary
import pdb
import numpy as np
from processor import *
from model import *
from apex import amp
import random



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


def select_model(args):
    cache = os.path.join(args.output_dir,"cache")
    if args.task_name == "rerank_csqa":
        if "albert" in args.origin_model:
            return AlbertAttRanker.from_pretrained(args.origin_model,cache_dir = cache,cs_len = args.cs_len)
        elif "roberta" in args.origin_model:
            return RobertaAttRanker.from_pretrained(args.origin_model,cache_dir = cache,cs_len = args.cs_len)
        elif "bert" in args.origin_model:
            return BertAttRanker.from_pretrained(args.origin_model,cache_dir = cache,cs_len = args.cs_len)
        elif "xlnet" in args.origin_model:
            return XLNetAttRanker.from_pretrained(args.origin_model,cache_dir = cache,cs_len = args.cs_len)
    elif args.task_name == "rerank_csqa_without_rerank":
        if "bert" in args.origin_model:
            return BertAttRankerDontRank.from_pretrained(args.origin_model,cache_dir = cache,cs_len = args.cs_len)
    else:
        if "albert" in args.origin_model:
            return AlbertForMultipleChoice.from_pretrained(args.origin_model,cache_dir = cache)
        elif "roberta" in args.origin_model:
            return RobertaForMultipleChoice.from_pretrained(args.origin_model,cache_dir = cache)
        elif "bert" in args.origin_model:
            return BertForMultipleChoice.from_pretrained(args.origin_model,cache_dir = cache)
        elif "xlnet" in args.origin_model:
            return XLNetForMultipleChoice.from_pretrained(args.origin_model,cache_dir = cache)
        
def set_seed(args):
    logger.info("Freeze seed : {}".format(str(args.seed)))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

# for xla mutithread method, seems to be easier to use
def train(args):
    '''
    Train the model, return Nothing
    '''
    # set up output_dir
    output_dir = os.path.join(args.output_dir,args.save_model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # save all the args
    arg_dict = args.__dict__
    with open(os.path.join(output_dir,"args.json"),'w',encoding='utf8') as f:
        json.dump(arg_dict,f,indent=2,ensure_ascii=False)
    # setup logging
    logfilename = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" "+args.save_model_name+".log.txt"
    fh = logging.FileHandler(os.path.join(output_dir,logfilename), mode='a', encoding='utf-8')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    # freeze seed
    if args.seed:
        set_seed(args)
    # loading data
    omcs_corpus = load_omcs(args)
    tokenizer = select_tokenizer(args)
    _,_,train_dataset= load_csqa_omcs_dataset(tokenizer,args,omcs_corpus,"train")
    dev_examples,_,dev_dataset= load_csqa_omcs_dataset(tokenizer,args,omcs_corpus,"dev")
    # _,_,test_dataset= load_csqa_omcs_dataset(tokenizer,args,omcs_corpus,"test",is_training = False)
    # setup device
    if args.tpu:
        # xla packages
        import torch_xla
        import torch_xla.distributed.data_parallel as dp
        import torch_xla.debug.metrics as met
        import torch_xla.utils.utils as xu
        import torch_xla.core.xla_model as xm
        import torch_xla.test.test_utils as test_utils
        # use multi thread method
        devices = (xm.get_xla_supported_devices(max_devices = 8))
        args.learning_rate = args.learning_rate * max(len(devices), 1)
        logger.info("New learning_rate for TPU is {}".format(str(args.learning_rate)))
        device_num = len(devices)
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas = xm.xrt_world_size(),
            rank = xm.get_ordinal(),
            shuffle = True
        )
        dev_sampler = torch.utils.data.distributed.DistributedSampler(
            dev_dataset,
            num_replicas = xm.xrt_world_size(),
            rank = xm.get_ordinal(),
            shuffle = False
        )
        train_dataloader = DataLoader(
            train_dataset, 
            sampler=train_sampler, 
            batch_size = args.train_batch_size)
        
        dev_dataloader = DataLoader(
            dev_dataset, 
            sampler = dev_sampler, 
            batch_size = args.eval_batch_size)
    else:
        # else we use gpu, colab only provide one gpu, so we won't use distributed trainging
        device = torch.device('cuda:0')
        train_sampler = RandomSampler(train_dataset) 
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

        dev_sampler = SequentialSampler(dev_dataset) 
        dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=args.eval_batch_size)
        device_num = 1
        
    train_step = len(train_dataloader)
    t_total = train_step // args.gradient_accumulation_steps * args.num_train_epochs // device_num
    optimizer = None
    scheduler = None
    def train_loop_fn(model,loader,device,context):
        nonlocal t_total,train_step,device_num 
        # t_total = len(loader) * args.num_train_epochs
        if not args.tpu :
            nonlocal optimizer, scheduler
            # don't need to init optimizer every epoch if not using tpu
            if not optimizer:
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

        model.zero_grad()
        tr_loss = 0.0
        iterator = tqdm(enumerate(loader),total = train_step/device_num)
        # iterator = enumerate(loader)
        for step,batch in iterator:
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": batch[3]
            }
            outputs = model(**inputs)
            loss = outputs[0]
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16 and not args.tpu:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if args.tpu:
                    xm.optimizer_step(optimizer)
                else:
                    optimizer.step()
                scheduler.step() 
                model.zero_grad()
            # check average training loss
            if (step + 1)% args.check_loss_step == 0:
                avg_loss = (tr_loss/(step+1)) * args.gradient_accumulation_steps
                logger.info("device:[%s] average_step_loss=%s @ step = %s on epoch = %s",device,str(avg_loss),str(step+1),str(epoch+1))
    
    def test_loop_fn(model,loader,device,context):
        model.eval()
        torch.cuda.empty_cache()
        # logger.info("Evaluate on {}".format(set_name))
        correct_count = 0
        predictions = []
        attention_scores = []
        total_test_items = 0
        with torch.no_grad():
            # iterator = tqdm(enumerate(loader))
            for step,batch in enumerate(loader):
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
                attention_score = outputs[2].cpu().numpy().tolist()
                pdb.set_trace()
                attention_scores += attention_score
                prediction = torch.argmax(logits,axis = 1)
                correct_count += (prediction == inputs["labels"]).sum().float()
                predictions += prediction.cpu().numpy().tolist()
                total_test_items += batch[0].shape[0]
        # logger.info("test_items of device[{}] is {}".format(device,str(total_test_items)))
        return correct_count,predictions,attention_scores

    def init_status():
        ''' 
        set up status for convenient viewing training result
        '''
        return {
            "current_epoch" : 0,
            "best_epoch" : -1,
            "best_Acc" : 0.0
        }
    
    status = init_status()
    model = select_model(args)
    if args.tpu:
        model_parallel = dp.DataParallel(model, device_ids=devices)
    else:
        device = torch.device('cuda:0')
        model = model.to(device)

    for epoch in range(0,args.num_train_epochs):
        logger.info("Epoch: {}".format(str(epoch)))
        if args.tpu:
            model_parallel(train_loop_fn, train_dataloader)
            results = model_parallel(test_loop_fn, dev_dataloader)
            correct_count = sum([float(item[0]) for item in results])
            predictions = [i for item in results for i in item[1]]
            model = model_parallel.models[0]
            acc = correct_count / len(dev_examples)
        else:
            # train_loop_fn(model,train_dataloader,device,None)
            correct_count, predictions, attention_scores = test_loop_fn(model,dev_dataloader,device,None)
            acc = correct_count / len(dev_examples)
            acc = acc.cpu().item() # tpu result don't need to switch device 
        # save model, save status 
        
        logger.info("DEV ACC : {}% on Epoch {}".format(str(acc * 100),str(epoch)))
        if args.save_method == "Best_Current":
            if acc > status["best_Acc"]:
                status['best_Acc'] = acc
                status['best_epoch'] = epoch
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model itself
                best_model_dir = os.path.join(output_dir,"best_model")
                if not os.path.exists(best_model_dir):
                    os.makedirs(best_model_dir)
                # 
                f_atten = open(os.path.join(best_model_dir,"prediction.txt"),'w',encoding="utf8")
                for p,a in zip(predictions,attention_scores):
                    f_atten.write("{}\t{}\n".format(str(p),str(a)))
                f_atten.close()

                model_to_save.save_pretrained(best_model_dir)
                logger.info("best epoch %d has been saved to %s",epoch,best_model_dir)
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model itself
            current_model_dir = os.path.join(output_dir,"current_model")
            if not os.path.exists(current_model_dir):
                os.makedirs(current_model_dir)
            model_to_save.save_pretrained(current_model_dir)
            logger.info("epoch %d has been saved to %s",epoch,current_model_dir)
        status_dir = os.path.join(output_dir,"status.json")
        json.dump(status,open(status_dir,'w',encoding = 'utf8'))


def make_predictions(args,examples,predictions,omcs_corpus,data_type="dev"):
  cs_file = "OMCS/{}_{}_omcs_of_dataset.json".format(data_type,args.cs_mode)
  with open(cs_file,'r',encoding="utf8") as f:
      cs_data = json.load(f)
  pred_index = 0 #because it's sequential, we simply index it with examples
  result_json = {}
  for example in tqdm(examples,desc="puting result into examples"):
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
  return result_json

# def test(args.)

def eval(args,model,dataloader,set_name,device,num_examples):
    torch.cuda.empty_cache()
    logger.info("Evaluating on {}".format(set_name))
    iterator = tqdm(dataloader, desc="Iteration")
    correct_count = 0
    predictions = []

    with torch.no_grad():
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
            
            correct_count += (prediction == inputs["labels"]).sum().float()
            predictions += prediction.cpu().numpy().tolist()
    return correct_count/num_examples, predictions

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',datefmt = '%m/%d/%Y %H:%M:%S',level = logging.INFO)
logger = logging.getLogger(__name__)

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
    parser.add_argument("--train_batch_size", default=15, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=6, type=int, help="Batch size for eval.")
    parser.add_argument("--check_loss_step",default = 400,type = int,help = "output current average loss of training")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--cs_len",type = int, default = 5)
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
    #在notebook 里 args 需要初始化为[],外部调用py文件不需要
    args = parser.parse_args()
    # if args.tpu:
    #     tpu_training(args)
    # else:
    train(args)
