from transformers import BertPreTrainedModel,BertModel,AlbertModel,AlbertPreTrainedModel,RobertaModel,XLNetPreTrainedModel,XLNetModel
import pdb
import torch
import torch.nn.functional as F
from transformers import BertTokenizer,BertModel
from transformers.modeling_outputs import MultipleChoiceModelOutput
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from model import *

class AttentionLayer(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.query = nn.Linear(config.hidden_size,config.hidden_size)
        self.key = nn.Linear(config.hidden_size,config.hidden_size)
        self.value = nn.Linear(config.hidden_size,config.hidden_size)
        # self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self,query,source):
        # hidden_states should be (batch_size,document_length,hidden_size)
        query_layer = self.query(query)
        key_layer = self.key(source)
        value_layer = self.value(source)
        attention_probs = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_probs = nn.Softmax(dim=-1)(attention_probs)
        attention_scores = torch.sum(attention_probs,dim = -2)
        context_layer = torch.matmul(attention_probs, value_layer)
        return context_layer,attention_scores

class BertForLinearKRD(BertPreTrainedModel):
    def __init__(self, config, cs_len, cs_seq_len, query_len):
        super().__init__(config)
        self.cs_len = cs_len
        self.transformer = BertModel(config)
        self.self_att = AttentionLayer(config)
        self.cross_att = AttentionLayer(config)
        # length configs
        self.cs_seq_len = cs_seq_len
        self.query_len  = query_len # query_len = question_len + answer_len
        self.classifier = nn.Linear(config.hidden_size*3,1)
        # self.classifier = nn.Linear(config.hidden_size*self.cs_len,1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()
    
    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=False,
                ):
        batch_size = input_ids.shape[0]
        num_choices = input_ids.shape[1]
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs[1]
        sequence_output = outputs[0]
        # separate query and commonsense encoding
        cs_len = int((input_ids.shape[-1] - self.query_len)/ self.cs_seq_len)
        cs_encoding = torch.stack([
            sequence_output[:,self.query_len+i*self.cs_seq_len:self.query_len+(i+1)*self.cs_seq_len,:] for i in range(cs_len)
        ],
        dim = 1)

        query_encoding = sequence_output[:,:self.query_len,:]
        expanded_query_encoding = query_encoding.unsqueeze(1).expand(query_encoding.shape[0],cs_len,query_encoding.shape[1],query_encoding.shape[2])
        # dual attention module 
        qc_attoutput, _ = self.cross_att(expanded_query_encoding,cs_encoding)
        cq_attoutput, _ = self.cross_att(cs_encoding,expanded_query_encoding)
        # re-distribution module, HeadHunter !
        mean_cs = torch.mean(cq_attoutput,dim = -2)
        cs_redistribution,attention_scores = self.self_att(mean_cs,mean_cs) # input same matrix to prefrom self-attention 
        attention_scores = F.softmax(attention_scores,dim = -1).unsqueeze(1)
        cs_rep = torch.tanh(torch.matmul(attention_scores,cs_redistribution)).squeeze(1)
        # mean pooling query encoding
        qu_rep = torch.mean(qc_attoutput,dim = -3)
        qu_rep = torch.mean(qu_rep, dim = -2)
        final_rep = torch.cat((pooled_output,cs_rep,qu_rep),dim = -1)

        logits = self.classifier(final_rep)
        reshaped_logits = logits.view(-1, num_choices)
        
        outputs = (reshaped_logits,attention_scores)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs
        else:
            loss = None
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=None,
            attentions=None,
        )

class AlbertForLinearKRD(AlbertPreTrainedModel):
    def __init__(self, config, cs_len, cs_seq_len, query_len):
        super().__init__(config)
        self.cs_len = cs_len
        self.transformer = AlbertModel(config)
        self.self_att = AttentionLayer(config)
        self.cross_att = AttentionLayer(config)
        # length configs
        self.cs_seq_len = cs_seq_len
        self.query_len  = query_len # query_len = question_len + answer_len
        self.classifier = nn.Linear(config.hidden_size*3,1)
        # self.classifier = nn.Linear(config.hidden_size*self.cs_len,1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()
    
    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=False,
                ):
        batch_size = input_ids.shape[0]
        num_choices = input_ids.shape[1]
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs[1]
        sequence_output = outputs[0]
        # separate query and commonsense encoding
        cs_len = int((input_ids.shape[-1] - self.query_len)/ self.cs_seq_len)
        cs_encoding = torch.stack([
            sequence_output[:,self.query_len+i*self.cs_seq_len:self.query_len+(i+1)*self.cs_seq_len,:] for i in range(cs_len)
        ],
        dim = 1)

        query_encoding = sequence_output[:,:self.query_len,:]
        expanded_query_encoding = query_encoding.unsqueeze(1).expand(query_encoding.shape[0],cs_len,query_encoding.shape[1],query_encoding.shape[2])
        # dual attention module 
        qc_attoutput, _ = self.cross_att(expanded_query_encoding,cs_encoding)
        cq_attoutput, _ = self.cross_att(cs_encoding,expanded_query_encoding)
        # re-distribution module, HeadHunter !
        mean_cs = torch.mean(cq_attoutput,dim = -2)
        cs_redistribution,attention_scores = self.self_att(mean_cs,mean_cs) # input same matrix to prefrom self-attention 
        attention_scores = F.softmax(attention_scores,dim = -1).unsqueeze(1)
        cs_rep = torch.tanh(torch.matmul(attention_scores,cs_redistribution)).squeeze(1)
        # mean pooling query encoding
        qu_rep = torch.mean(qc_attoutput,dim = -3)
        qu_rep = torch.mean(qu_rep, dim = -2)
        final_rep = torch.cat((pooled_output,cs_rep,qu_rep),dim = -1)

        logits = self.classifier(final_rep)
        reshaped_logits = logits.view(-1, num_choices)
        
        outputs = (reshaped_logits,attention_scores)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs
        else:
            loss = None
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=None,
            attentions=None,
        )

# model = BertForLinearKRD.from_pretrained("bert-base-cased",cs_len = 3,cs_seq_len = 20, query_len = 80)

    