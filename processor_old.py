import json
import re
import pdb
from transformers import XLNetTokenizer
import xml.dom.minidom
from xml.dom.minidom import parse
import torch
import os
from torch.utils.data import TensorDataset
from tqdm import tqdm

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
                
                # sen1 = self.create_hypothesis(self.get_fitb_from_question(example.question), ending)
                # 暂时使用 question 【sep】 answer 【sep】
                if example.context == None:
                    sen1 = example.question
                    sen2 = ending
                    sen1 = tokenizer.tokenize(sen1)
                    sen2 = tokenizer.tokenize(ending)
                else:
                    sen1 = self.join_cs(tokenizer,example.context[ending_index],example.question)
                    sen1 = tokenizer.tokenize(sen1)
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
                # pdb.set_trace()
                if "token_type_ids" in inputs.keys():
                    token_type_ids= inputs['token_type_ids']
                else:
                    token_type_ids = attention_mask.copy()
                    
                    if len(token_type_ids) != max_seq_length:
                        pdb.set_trace()
                # pdb.set_trace()
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
    for example in tqdm(examples,total = len(examples),desc = "puting commonsencs into examples"):
        context = [] # put omcs into context, it will be a [[str],]
        example_cs = cs_data[example.id]['endings']
        for ending_index,ending in enumerate(example.endings):
            # index = example.id+"_{}".format(str(ending_index))
            
            cs_id_list  = example_cs[ending_index]['cs'][:cs_len] # extract first cs_len cs sentences for example
            omcs_list = [omcs_corpus[int(id)] for id in cs_id_list]
            context.append(omcs_list)
        example.context = context
    return examples

def load_omcs(args):
    omcs_file_path = os.path.join(args.cs_dir,args.omcs_file)
    with open(omcs_file_path,'r',encoding='utf8') as f:
        omcs_corpus = json.load(f)
    return omcs_corpus

def load_csqa_omcs_dataset(tokenizer,args,omcs_corpus,data_type,is_training=True):
    if data_type == "dev":
        file_name = os.path.join(args.data_dir,args.dev_file)
    elif data_type == "test":
        file_name = os.path.join(args.data_dir,args.test_file)
    else :
        file_name = os.path.join(args.data_dir,args.train_file)
    # read examples
    processor = CSQAProcessor()
    examples = processor.read_examples(file_name,is_training)
    cs_result_file = "{}_{}_omcs_of_dataset.json".format(data_type,args.cs_mode)
    cs_result_path = os.path.join(args.cs_dir,cs_result_file)
    with open(cs_result_path,'r',encoding='utf8') as f:
        cs_data = json.load(f)
    examples = put_in_cs(examples,cs_data,omcs_corpus,args.cs_len)
    max_length = args.max_length + args.cs_len*12 # dynamicly set max_length

    features = processor.convert_examples_to_features(tokenizer,examples,max_length,is_training)

    all_input_ids = torch.tensor([f.select_field("input_ids") for f in features], dtype=torch.long)
    all_attention_masks = torch.tensor([f.select_field("attention_mask") for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.select_field("token_type_ids") for f in features], dtype=torch.long)
    if is_training :
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids,all_attention_masks, all_token_type_ids, all_labels)
    else:
        dataset = TensorDataset(all_input_ids,all_attention_masks,all_token_type_ids)
    return examples,features,dataset


def load_csqa_dataset(tokenizer,args,data_type,is_training=True):
    if data_type == "dev":
        file_name = os.path.join(args.data_dir,args.dev_file)
    elif data_type == "test":
        file_name = os.path.join(args.data_dir,args.test_file)
    else :
        file_name = os.path.join(args.data_dir,args.train_file)
    # cache_name = os.path.join
    processor = CSQAProcessor()
    examples = processor.read_examples(file_name,is_training)
    features = processor.convert_examples_to_features(tokenizer,examples,args.max_length,is_training)

    all_input_ids = torch.tensor([f.select_field("input_ids") for f in features], dtype=torch.long)
    all_attention_masks = torch.tensor([f.select_field("attention_mask") for f in features], dtype=torch.long)
    # if "roberta" in args.origin_model:
    #     all_token_type_ids = None
    # else:
    all_token_type_ids = torch.tensor([f.select_field("token_type_ids") for f in features], dtype=torch.long)
    if is_training :
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        # if "roberta" in args.origin_model:
        #     dataset = TensorDataset(all_input_ids,all_attention_masks, all_labels)
        # else:
        dataset = TensorDataset(all_input_ids,all_attention_masks, all_token_type_ids, all_labels)
    else:
        # if "roberta" in args.origin_model:
        #     dataset = TensorDataset(all_input_ids,all_attention_masks)
        # else:
        dataset = TensorDataset(all_input_ids,all_attention_masks,all_token_type_ids)
    return examples,features,dataset