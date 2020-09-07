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

    # Identify the wh-word in the question and replace with a blank
    def replace_wh_word_with_blank(self,question_str: str):
        if "What is the name of the government building that houses the U.S. Congress?" in question_str:
            print()
        question_str = question_str.replace("What's", "What is")
        question_str = question_str.replace("whats", "what")
        question_str = question_str.replace("U.S.", "US")
        wh_word_offset_matches = []
        wh_words = ["which", "what", "where", "when", "how", "who", "why"]
        for wh in wh_words:
            # Some Turk-authored SciQ questions end with wh-word
            # E.g. The passing of traits from parents to offspring is done through what?

            if wh == "who" and "people who" in question_str:
                continue

            m = re.search(wh + "\?[^\.]*[\. ]*$", question_str.lower())
            if m:
                wh_word_offset_matches = [(wh, m.start())]
                break
            else:
                # Otherwise, find the wh-word in the last sentence
                m = re.search(wh + "[ ,][^\.]*[\. ]*$", question_str.lower())
                if m:
                    wh_word_offset_matches.append((wh, m.start()))
                # else:
                #     wh_word_offset_matches.append((wh, question_str.index(wh)))

        # If a wh-word is found
        if len(wh_word_offset_matches):
            # Pick the first wh-word as the word to be replaced with BLANK
            # E.g. Which is most likely needed when describing the change in position of an object?
            wh_word_offset_matches.sort(key=lambda x: x[1])
            wh_word_found = wh_word_offset_matches[0][0]
            wh_word_start_offset = wh_word_offset_matches[0][1]
            # Replace the last question mark with period.
            question_str = re.sub("\?$", ".", question_str.strip())
            # Introduce the blank in place of the wh-word
            fitb_question = (question_str[:wh_word_start_offset] + BLANK_STR +
                            question_str[wh_word_start_offset + len(wh_word_found):])
            # Drop "of the following" as it doesn't make sense in the absence of a multiple-choice
            # question. E.g. "Which of the following force ..." -> "___ force ..."
            final = fitb_question.replace(BLANK_STR + " of the following", BLANK_STR)
            final = final.replace(BLANK_STR + " of these", BLANK_STR)
            return final

        elif " them called?" in question_str:
            return question_str.replace(" them called?", " " + BLANK_STR+".")
        elif " meaning he was not?" in question_str:
            return question_str.replace(" meaning he was not?", " he was not " + BLANK_STR+".")
        elif " one of these?" in question_str:
            return question_str.replace(" one of these?", " " + BLANK_STR+".")
        elif re.match(".*[^\.\?] *$", question_str):
            # If no wh-word is found and the question ends without a period/question, introduce a
            # blank at the end. e.g. The gravitational force exerted by an object depends on its
            return question_str + " " + BLANK_STR
        else:
            # If all else fails, assume "this ?" indicates the blank. Used in Turk-authored questions
            # e.g. Virtually every task performed by living organisms requires this?
            return re.sub(" this[ \?]", " ___ ", question_str)
    # the original one



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


    def get_fitb_from_question(self,question_text: str) -> str:
        fitb = self.replace_wh_word_with_blank(question_text)
        if not re.match(".*_+.*", fitb):
            # print("Can't create hypothesis from: '{}'. Appending {} !".format(question_text, BLANK_STR))
            # Strip space, period and question mark at the end of the question and add a blank
            fitb = re.sub("[\.\? ]*$", "", question_text.strip()) + " "+ BLANK_STR
        return fitb


    # Create a hypothesis statement from the the input fill-in-the-blank statement and answer choice.
    def create_hypothesis(self,fitb: str, choice: str) -> str:
        if ". " + BLANK_STR in fitb or fitb.startswith(BLANK_STR):
            choice = choice[0].upper() + choice[1:]
        else:
            choice = choice.lower()
        # Remove period from the answer choice, if the question doesn't end with the blank
        if not fitb.endswith(BLANK_STR):
            choice = choice.rstrip(".")
        # Some questions already have blanks indicated with 2+ underscores
        hypothesis = re.sub("__+", choice, fitb)
        return hypothesis

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
                else:
                    sen1 = self.join_cs(tokenizer,example.context[ending_index],example.question)
                    sen2 = tokenizer.tokenize(ending)
                inputs = tokenizer.encode_plus(
                    sen1,
                    sen2,
                    add_special_tokens= True,
                    max_length = max_seq_length,
                    pad_to_max_length = True,
                    truncation_strategy = 'longest_first'
                )
                input_ids, attention_mask, token_type_ids= inputs['input_ids'],inputs['attention_mask'],inputs['token_type_ids']

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
    # # if input str type cs
    # if os.path.exists(os.path.join(args.cs_dir,"{}_{}_omcs_of_dataset.json".format(data_type,args.cs_mode))):
    #     with open(os.path.join(args.cs_dir,"{}_{}_omcs_of_dataset.json".format(data_type,args.cs_mode)),'r',encoding='utf8') as f:
    #         cs_data = json.load(f)
    #     examples = put_in_cs_json(examples,args.cs_len,cs_data)
    # # else us id type cs
    # else:
    #     cs_filename = os.path.join(args.cs_dir,"{}_{}_omcs_results.json".format(data_type,args.cs_mode))
    #     with open(cs_filename,'r',encoding = "utf8") as f:
    #         cs_json = json.load(f)
        
    #     examples = put_in_cs(examples,cs_json,omcs_corpus,args.cs_len)
    
    # we use same format now, get cs_sentence form omcs_corpus by id in cs_data
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
    all_token_type_ids = torch.tensor([f.select_field("token_type_ids") for f in features], dtype=torch.long)
    if is_training :
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids,all_attention_masks, all_token_type_ids, all_labels)
    else:
        dataset = TensorDataset(all_input_ids,all_attention_masks,all_token_type_ids)
    return examples,features,dataset