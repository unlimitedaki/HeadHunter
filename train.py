# standard libraries
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
from transformers import AdamW,get_linear_schedule_with_warmup
from transformers.modeling_utils import SequenceSummary

# print(transformers.__version__)
# self build
from processor import *
from model import *


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
    logger.setLevel(logging.INFO)
    logger.info("Logger Level: INFO")
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
    def train_loop_fn(model,loader,device,context,in_optimizer = None, in_scheduler = None):
        nonlocal t_total,train_step,device_num 
        model.cs_len = args.cs_len
        # t_total = len(loader) * args.num_train_epochs
        if not args.tpu :
            optimizer = in_optimizer
            scheduler = in_scheduler
        # if not args.tpu :
        #     nonlocal optimizer, scheduler
        else:
            # logger.info("init optimizer inside train loop while using tpu")
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
        model.cs_len = args.dev_cs_len
        # for dev 
        model.eval()
        torch.cuda.empty_cache()
        # logger.info("Evaluate on {}".format(set_name))
        correct_count = 0
        predictions = []
        attention_scores = []
        total_test_items = 0
        with torch.no_grad():
            for step, batch in tqdm(enumerate(loader)):
            # for step,batch in enumerate(loader):
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
                if len(outputs) == 3:
                    attention_score = outputs[2].cpu().numpy().tolist()
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

    if not args.tpu:
        logger.info("init optimizer outside train loop while using gpu")
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

    

    for epoch in range(0,args.num_train_epochs):
        logger.info("Epoch: {}".format(str(epoch)))
        if args.tpu:
            model_parallel(train_loop_fn, train_dataloader)
            results = model_parallel(test_loop_fn, dev_dataloader)
            correct_count = sum([float(item[0]) for item in results])
            predictions = [i for item in results for i in item[1]]
            attention_scores = [i for item in results for i in item[2]]
            # save the model in cpu way
            import copy
            model = copy.deepcopy(model_parallel.models[0])
            model = model.cpu()
            predictions,attention_scores = truncate_prediction(len(dev_examples),predictions,attention_scores)
            acc = correct_count / len(dev_examples)
        else:
            train_loop_fn(model,train_dataloader,device,None,in_optimizer = optimizer,in_scheduler = scheduler)
            correct_count, predictions, attention_scores = test_loop_fn(model,dev_dataloader,device,None)
            acc = correct_count / len(dev_examples)
            acc = acc.cpu().item() # tpu result don't need to switch device 
        # save model, save status 
        # pdb.set_trace()
        logger.info("DEV ACC : {}% on Epoch {}".format(str(acc * 100),str(epoch)))
        # prediction_json = make_predictions(args,dev_examples,predictions,attention_scores,omcs_corpus,"dev")
        
        if args.save_method == "Best_Current":
            if acc > status["best_Acc"]:
                status['best_Acc'] = acc
                status['best_epoch'] = epoch
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model itself
                best_model_dir = os.path.join(output_dir,"best_model")
                if not os.path.exists(best_model_dir):
                    os.makedirs(best_model_dir)
                model_to_save.save_pretrained(best_model_dir)
                # prediction_file = os.path.join(best_model_dir,"{}_{}_{}_prediction_file.json".format("dev",args.cs_mode,args.cs_len))
                # with open(prediction_file,'w',encoding= 'utf8') as f:
                #     json.dump(prediction_json,f,indent = 2,ensure_ascii = False)
                logger.info("best epoch %d has been saved to %s",epoch,best_model_dir)
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model itself
            current_model_dir = os.path.join(output_dir,"current_model")
            if not os.path.exists(current_model_dir):
                os.makedirs(current_model_dir)
            model_to_save.save_pretrained(current_model_dir)
            # prediction_file = os.path.join(current_model_dir,"{}_{}_{}_prediction_file.json".format("dev",args.cs_mode,args.cs_len))
            # with open(prediction_file,'w',encoding= 'utf8') as f:
            #     json.dump(prediction_json,f,indent = 2,ensure_ascii = False)
            logger.info("epoch %d has been saved to %s",epoch,current_model_dir)
        status_dir = os.path.join(output_dir,"status.json")
        json.dump(status,open(status_dir,'w',encoding = 'utf8'))

def truncate_prediction(num_example,predictions,attention_scores,is_training = True):
    predictions = predictions[:num_example]
    attention_scores = attention_scores[:num_example]
    return predictions, attention_scores

def make_predictions(args,examples,predictions,attention_scores,omcs_corpus,data_type="dev"):
    cs_len = args.dev_cs_len
    cs_file = "OMCS/{}_{}_omcs_of_dataset.json".format(data_type,args.cs_mode)
    with open(cs_file,'r',encoding="utf8") as f:
        cs_data = json.load(f)
    result_json = {}
    for i,example in tqdm(enumerate(examples),desc="puting result into examples"):
        result_json[example.id] = {}
        result_json[example.id]['question'] = example.question
        result_json[example.id]['prediction'] = predictions[i]
        # pdb.set_trace()
        result_json[example.id]["prediction_answer"] = example.endings[predictions[i]]
        
        result_json[example.id]['label'] = example.label
        example_cs = cs_data[example.id]
        result_json[example.id]['endings'] = example_cs['endings']
        for j,ending in enumerate(result_json[example.id]['endings']):
            if args.cs_save_mode == 'id':
                ending["cs"] = [omcs_corpus[int(id)] for id in ending["cs"][:cs_len]]
            else:
                ending['cs'] = ending["cs"][:cs_len]
            if attention_scores != []:
                # some baseline model won't output any attention score. 
                ending["attention_scores"] = attention_scores[i][j]
        # pred_index += 1
    return result_json

# def test(args.)

def eval(args,set_name):
    torch.cuda.empty_cache()
    logger.info("Evaluating on {}".format(set_name))
    output_dir = os.path.join(args.output_dir,args.save_model_name)
    best_model_dir = os.path.join(output_dir,"best_model")
    is_training = args.dev 
    model = select_model(args,best_model_dir)
    omcs_corpus = load_omcs(args)
    tokenizer = select_tokenizer(args)
    examples,_,dataset= load_csqa_omcs_dataset(tokenizer,args,omcs_corpus,set_name,is_training)
    # setup device
    if args.tpu:
        # xla packages
        import torch_xla
        import torch_xla.distributed.data_parallel as dp
        import torch_xla.debug.metrics as met
        import torch_xla.utils.utils as xu
        import torch_xla.core.xla_model as xm
        import torch_xla.test.test_utils as test_utils
        # tpu still has some bugs, to be fixed 
        
    else:
        # else we use gpu, colab only provide one gpu, so we won't use distributed trainging
        device = torch.device('cuda:0')
        sampler = SequentialSampler(dataset) 
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size)
        device_num = 1

    def test_loop_fn(model,loader,device,context):
        # for test 
        model.eval()
        torch.cuda.empty_cache()
        correct_count = 0
        predictions = []
        attention_scores = []
        with torch.no_grad():
            for step,batch in tqdm(enumerate(loader),total = len(loader)):
                model.eval()
                batch = tuple(t.to(device) for t in batch)
                if len(batch) == 4:
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "token_type_ids": batch[2],
                        "labels": batch[3]
                    }
                else:
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "token_type_ids": batch[2],
                    }
                outputs = model(**inputs)
                logits = outputs[1] if len(batch) == 4 else outputs[0]
                if len(batch) == 3 and len(outputs) == 2:
                    attention_score = outputs[1].cpu().numpy().tolist()
                    attention_scores += attention_score
                elif len(outputs) == 3:
                    attention_score = outputs[2].cpu().numpy().tolist()
                    attention_scores += attention_score
    
                prediction = torch.argmax(logits,axis = 1)
                if len(batch) == 4:
                    correct_count += (prediction == inputs["labels"]).sum().float()
                predictions += prediction.cpu().numpy().tolist()
        return correct_count,predictions,attention_scores

    if args.tpu:
        model_parallel = dp.DataParallel(model, device_ids=devices)
    else:
        device = torch.device('cuda:0')
        model = model.to(device)
        if args.fp16:
            model = amp.initialize(model,opt_level = "O1")
    model.cs_len = args.dev_cs_len
    # Test!
    correct_count, predictions, attention_scores = test_loop_fn(model,dataloader,device,None)

    prediction_json = make_predictions(args,examples,predictions,attention_scores,omcs_corpus,set_name)
    prediction_file = os.path.join(best_model_dir,"{}_{}_{}_prediction_file.json".format(set_name,args.cs_mode,args.dev_cs_len))
    if set_name == "dev":
        acc = correct_count/float(len(examples))
        logger.info("DEV ACC is {}".format(acc))
    
        dev_file = os.path.join(output_dir,"dev_res.json")
        if os.path.exists(dev_file):
            with open(dev_file,'r',encoding = "utf8") as f:
                dev_res = json.load(f)
        else:
            dev_res = {}
        dev_res[args.dev_cs_len] = float(acc.cpu().numpy())
        with open(dev_file,'w',encoding = "utf8") as f:
            json.dump(dev_res,f,indent = 2 ,ensure_ascii = False)
    with open(prediction_file,'w',encoding= 'utf8') as f:
        json.dump(prediction_json,f,indent = 2,ensure_ascii = False)
    return 

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
