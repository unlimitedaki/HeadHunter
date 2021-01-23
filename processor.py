import json
import re
import pdb
import xml.dom.minidom
from xml.dom.minidom import parse
import torch
import os
from torch.utils.data import TensorDataset
from tqdm import tqdm
import random

BLANK_STR = "___"
class MultipleChoiceExample(object): # examples for all kind of dataset s
    # including 
    """A single training/test example for the SWAG dataset."""
    def __init__(self,
                 id,
                 context,
                 question,
                 endings,
                 label=None):
        self.id = id,   
        self.context = context
        self.question = question
        self.endings = endings
        self.label = label

class CSQAExample(MultipleChoiceExample):
    def __init__(
            self,
            id,
            context,
            question,
            endings,
            label = None,
            question_concept = None
        ):
        self.id = id
        self.context = context
        self.question = question
        self.endings = endings
        self.label = label
        self.question_concept = question_concept

class InputFeatures(object):
    def __init__(self,
                 id,
                 features,
                 label

    ):
        self.id = id
        self.features = [
            {
                'input_ids': input_ids,
                'attention_mask':attention_mask,
                'token_type_ids':token_type_ids
            }
            for input_ids, attention_mask, token_type_ids in features
        ]
        
        self.label = label

    def select_field(self, field):
        return [
            item[field] for item in self.features
        ]




def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

class CSQAProcessor():
    '''
    CSQA Processor,
    read_examples from dataset file
    concat common sense into the input sequence 
    convert example to features
    '''
    def read_examples(self, input_file, is_training=True):
        with open(input_file, "r", encoding="utf-8") as f:
            examples = []
            for line in f.readlines():
                csqa_json = json.loads(line)
                if is_training:
                    label = ord(csqa_json["answerKey"]) - ord("A")
                else:
                    label = 0  # just as placeholder here for the test data
                id = csqa_json["id"]
                
                examples.append(
                    CSQAExample(
                        id = id, 
                        question=csqa_json["question"]["stem"],
                        context = None,
                        endings  = [csqa_json["question"]["choices"][i]["text"]  for i in range(5) ],
                        label = label,
                        question_concept = csqa_json['question']['question_concept']
                    ))
        return examples

    def join_cs(self,tokenizer,context,question):
        # context = " ".join(context) # just use space to join contexts(cs)
        context = " [SEP] ".join(context)
        context_tokens = tokenizer.tokenize(context)
        question_tokens = tokenizer.tokenize(question)
        sen1_tokens = question_tokens + [tokenizer.sep_token] + context_tokens
        return sen1_tokens


    #  convert csqa examples(MultipleChoiceExamples) to InputFeatures
    def convert_examples_to_features(self, tokenizer,examples, max_seq_length, is_training):
        """Loads a data file into a list of `InputBatch`s."""
        features = []
        for example_index, example in tqdm(enumerate(examples),total = len(examples),desc = "CSQA processing"):
            choices_features = []
            for ending_index, ending in enumerate(example.endings):
                
                if example.context == None:
                    sen1 = example.question
                    sen2 = ending
                    sen1 = tokenizer.tokenize(sen1)
                    sen2 = tokenizer.tokenize(ending)
                else:
                    sen1 = self.join_cs(tokenizer,example.context[ending_index],example.question)
                    # sen1 = tokenizer.tokenize(sen1)
                    sen2 = tokenizer.tokenize(ending)
                inputs = tokenizer.encode_plus(
                    sen1,
                    sen2,
                    add_special_tokens= True,
                    max_length = max_seq_length,
                    pad_to_max_length = True,
                    truncation_strategy = 'longest_first'
                )
                # pdb.set_trace()
                input_ids, attention_mask = inputs['input_ids'],inputs['attention_mask']
                if "token_type_ids" in inputs.keys():
                    token_type_ids= inputs['token_type_ids']
                else:
                    token_type_ids = [0] * max_seq_length
                    
                choices_features.append((input_ids, attention_mask, token_type_ids))

            label = example.label

            features.append(
                InputFeatures(
                    id = example.id,
                    features = choices_features,
                    label = label
                )
            )

        return features



class CSQARankerProcessor(CSQAProcessor):
    '''
    Processor for attentive reranker,
    while one example inputed, it will be num_choices * num_commmonsense(cs_len) inputs for encoding.
    '''
    def __init__(self):
        super().__init__()
    #  convert csqa examples(MultipleChoiceExamples) to InputFeatures
    def convert_examples_to_features(self, tokenizer,examples, max_seq_length, is_training):
        """Loads a data file into a list of `InputBatch`s."""
        features = []
        for example_index, example in tqdm(enumerate(examples),total = len(examples),desc = "CSQA processing"):
            choices_features = []
            question_tokens = tokenizer.tokenize(example.question)
            for ending_index, ending in enumerate(example.endings):
                cs_list = example.context[ending_index]
                ending_tokens = tokenizer.tokenize(ending)
                #
                for cs in cs_list:
                    cs_tokens = tokenizer.tokenize(cs)
                    input_tokens = question_tokens + [tokenizer.sep_token] + ending_tokens + [tokenizer.sep_token] + cs_tokens
                    
                    inputs = tokenizer.encode_plus(
                        input_tokens,
                        add_special_tokens= True,
                        max_length = max_seq_length,
                        pad_to_max_length = True
                    )
                    input_ids, attention_mask = inputs['input_ids'],inputs['attention_mask']
                    if "token_type_ids" in inputs.keys():
                        token_type_ids= inputs['token_type_ids']
                    else:
                        token_type_ids = [0] * max_seq_length
                    choices_features.append((input_ids, attention_mask, token_type_ids))

            label = example.label
            features.append(
                InputFeatures(
                    id = example.id,
                    features = choices_features,
                    label = label
                )
            )
        return features


def put_in_cs(examples,cs_data,omcs_corpus,cs_len):
    '''
    put commonsense into context
    '''
    for example in tqdm(examples,total = len(examples),desc = "puting commonsencs into examples"):
        context = [] # put omcs into context, it will be a [[str],]
        example_cs = cs_data[example.id]['endings']
        for ending_index,ending in enumerate(example.endings):
            
            cs_id_list  = example_cs[ending_index]['cs'][:cs_len] # extract first cs_len cs sentences for example
            omcs_list = [omcs_corpus[int(id)] for id in cs_id_list]
            while len(omcs_list) < cs_len:
                omcs_list.append("")
            context.append(omcs_list)
        example.context = context
    return examples

def load_omcs(args):
    '''
    load omcs data to locate commonsense
    '''
    omcs_file_path = os.path.join(args.cs_dir,args.omcs_file)
    with open(omcs_file_path,'r',encoding='utf8') as f:
        omcs_corpus = json.load(f)
    return omcs_corpus

def feature_padding(args, 
    data_type, 
    all_input_ids, 
    all_attention_masks, 
    all_token_type_ids, 
    all_labels = None):
    '''
    Multi threads tpu sync ensures every tpu trains same amount of batchs, I don't know other ways to fix it
    so,
    we calculate the min Least common multiple between batch size and tpu cores(8) which larger than examples 
    for training set , we randomly pick examples to pad
    for dev set, we pad zero tensors, and set label = -1 
    input: every tensor list of input
    output: nothing
    '''
    cur_number = all_input_ids.shape[0]
    seq_len = all_input_ids.shape[-1]
    batch_x = args.train_batch_size * 8
    target_number = (cur_number//batch_x + 1) * batch_x
    padding_number = target_number - cur_number
    if data_type == "dev":
        padding_input = torch.zeros((padding_number,all_input_ids.shape[1],all_input_ids.shape[2]),dtype = torch.long)
        all_input_ids = torch.cat((all_input_ids,padding_input),dim = 0)
        all_attention_masks = torch.cat((all_attention_masks,padding_input),dim = 0)
        all_token_type_ids = torch.cat((all_token_type_ids,padding_input),dim = 0)
        if all_labels != None:
            all_labels = torch.cat((all_labels,torch.tensor([-1] * padding_number,dtype = torch.long)),dim = 0)
    elif data_type == "train":
        padding_index = random.sample(range(0,cur_number),padding_number)
        all_input_ids = torch.cat((all_input_ids,torch.tensor([all_input_ids[i].numpy() for i in padding_index],dtype=torch.long)),0)
        all_attention_masks = torch.cat((all_attention_masks,torch.tensor([all_attention_masks[i].numpy() for i in padding_index],dtype=torch.long)),0)
        all_token_type_ids = torch.cat((all_token_type_ids,torch.tensor([all_token_type_ids[i].numpy() for i in padding_index],dtype=torch.long)),0)
        if all_labels != None:
            all_labels = torch.cat((all_labels,torch.tensor([all_labels[i] for i in padding_index],dtype=torch.long)),0)
        
    assert len(all_input_ids) == target_number
    print("{} set is padded to {} examples".format(data_type,str(target_number)))
    return all_input_ids, all_attention_masks, all_token_type_ids, all_labels

class CSQADataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class CSQALinearProcessor(CSQAProcessor):
    def __init__(self):
        super().__init__()

    # def encode(example):
    #     question = example.question
    #     endings = example.endings
    #     label = example.label
    #     for ending in endings:


    def convert_examples_to_features(self,
        tokenizer,
        examples,
        question_seq_len,
        answer_seq_len,
        cs_seq_len
        ):
        # def encode()
        def encode(example):
            cls = tokenizer.cls_token_id
            sep = tokenizer.sep_token_id
            pad = tokenizer.pad_token_id
            question = example.question
            endings = example.endings
            label = example.label
            # context = 
            input_ids = []
            attention_mask = []
            token_type_ids = []
            q_tokens = tokenizer(question,add_special_tokens=False,padding = 'max_length', truncation=True,max_length = question_seq_len-1 )
            for ending_index, ending in enumerate(endings):
                cs_list = example.context[ending_index]
                # pdb.set_trace()
                try:
                    cs_tokens = tokenizer(cs_list,add_special_tokens=False,padding = 'max_length', truncation=True,max_length = cs_seq_len-1)
                    cs_joint_input_ids = []
                    cs_joint_attention_mask = []
                    # join cs encodings
                    for i,cs in enumerate(cs_tokens['input_ids']):
                        cs_joint_input_ids += [sep] + cs
                        cs_joint_attention_mask += [1] + cs_tokens['attention_mask'][i]
                except:
                    cs_joint_input_ids = [pad] * len(cs_list)*cs_seq_len
                    cs_joint_attention_mask = [0] * len(cs_list)*cs_seq_len
                a_tokens = tokenizer(ending,add_special_tokens=False,padding = 'max_length', truncation=True,max_length = answer_seq_len-1)
                
                # concat all tokens
                joint_input_ids = [cls] + q_tokens['input_ids'] +  [sep] + a_tokens['input_ids'] + cs_joint_input_ids
                joint_attention_mask = [1] + q_tokens['attention_mask'] + [1] + a_tokens['attention_mask'] + cs_joint_attention_mask
                joint_token_type_ids = [1] * len(joint_attention_mask)
                joint_token_type_ids[:len(q_tokens['input_ids'])+1] = [0] * (len(q_tokens['input_ids'])+1)

                input_ids.append(joint_input_ids)
                attention_mask.append(joint_attention_mask)
                token_type_ids.append(joint_token_type_ids)
            return input_ids, attention_mask, token_type_ids, label


                
        # initialize encoding, encoding 
        encodings = {}
        encodings["input_ids"] = []
        encodings["attention_mask"] = []
        encodings["token_type_ids"] = []
        labels = []
    

        for example in tqdm(examples):
            input_ids, attention_mask, token_type_ids, label = encode(example)
            encodings["input_ids"].append(input_ids)
            encodings["attention_mask"].append(attention_mask) 
            encodings["token_type_ids"].append(token_type_ids)
            labels.append(label)
        dataset = CSQADataset(encodings,labels)
        # dataset.encodings = encodings
        # dataset.labels = labels
        return dataset

def load_csqa_omcs_dataset(tokenizer,args,omcs_corpus,data_type,is_training=True):
    '''
    load csqa dateset and put in commonsense
    Method of joining cs is decided by args.task_name = [rerank_csqa, other]
    '''
    if data_type == "dev":
        file_name = os.path.join(args.data_dir,args.dev_file)
    elif data_type == "test":
        file_name = os.path.join(args.data_dir,args.test_file)
    else :
        file_name = os.path.join(args.data_dir,args.train_file)
    if "rerank_csqa" in args.task_name:
        processor = CSQARankerProcessor()
        max_length = args.max_length
    elif "KRD_linear" in args.task_name:
        processor = CSQALinearProcessor()
    else:
        processor = CSQAProcessor()
        max_length = args.max_length + 12 * args.cs_len

    examples = processor.read_examples(file_name,is_training)

    cs_result_file = "{}_{}_omcs_of_dataset.json".format(data_type,args.cs_mode)
    cs_result_path = os.path.join(args.cs_dir,cs_result_file)
    with open(cs_result_path,'r',encoding='utf8') as f:
        cs_data = json.load(f)
    if args.cs_len > 0:
        if data_type == "dev" or data_type == "test":
            examples = put_in_cs(examples,cs_data,omcs_corpus,args.dev_cs_len)
        else:
            examples = put_in_cs(examples,cs_data,omcs_corpus,args.cs_len)
    processor = CSQALinearProcessor()
    # pdb.set_trace()
    dataset = processor.convert_examples_to_features(
            tokenizer = tokenizer,
            examples = examples,
            question_seq_len = args.question_seq_len,
            answer_seq_len = args.answer_seq_len,
            cs_seq_len = args.cs_seq_len
            )
    

    return dataset