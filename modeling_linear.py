import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertPreTrainedModel, BertTokenizer
from transformers.modeling_outputs import MultipleChoiceModelOutput



class AttentionLayer(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.query = nn.Linear(config.hidden_size,config.hidden_size)
        self.key = nn.Linear(config.hidden_size,config.hidden_size)
        self.value = nn.Linear(config.hidden_size,config.hidden_size)
        # self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self,query,source):
        '''
        input:
        - query: [b, cs_len, L1, hidden]
        - source: [b, cs_len, L2, hidden]
        
        output:
        - attention_scores: [b, cs_len, L2]     <= sum of attention_prob [b, cs_len, L1, L2]
        - context_layer: [b, cs_len, L1, hidden]
        '''
        # hidden_states should be (batch_size,document_length,hidden_size)
        query_layer = self.query(query)
        key_layer = self.key(source)
        value_layer = self.value(source)
        
        attention_probs = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # Q @ K
        # attention_probs /= query.shape[-1]**(1/2)   # scale
        attention_probs = nn.Softmax(dim=-1)(attention_probs)   # softmax [b, cs_len, L1, L2]
        attention_scores = torch.sum(attention_probs,dim = -2)  # [b, cs_len, L2]
        # attention_scores = torch.sum(attention_probs,dim = -1)  # [b, cs_len, L1]
        context_layer = torch.matmul(attention_probs, value_layer)  # softmax @ V [b, cs_len, L1, hidden]
        return context_layer,attention_scores

class BertForLinearKRD(BertPreTrainedModel):
    def __init__(self, config, cs_len, cs_seq_len, query_len):
        super().__init__(config)
        self.cs_len = cs_len
        self.bert = BertModel(config)
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
        input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
        inputs_embeds=None, head_mask=None, output_attentions=False, # unuse 
        labels=None,
        ):
        '''
        input_ids [b, 5, seq_len] => [5b, seq_len]
        
        => PTM
        cs_encoding [5b, cs_len, cs_seq_len, hidden]
        query_encoding [5b, query_len, hidden] => [5b, cs_len, query_len, hidden]
        
        => cross_attn
        qc_attoutput  [5b, cs_len, query_seq_len, hidden]
        cq_attoutput  [5b, cs_len, cs_seq_len, hidden]
        
        '''
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

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )       # BaseModelOutputWithPoolingAndCrossAttentions

        pooled_output = outputs.pooler_output   # outputs[1]  CLS token    [5b, hidden]
        sequence_output = outputs.last_hidden_state       # outputs[0]     [5b, seq_len, hidden] 
        
        # separate query and commonsense encoding
        cs_len = int((input_ids.shape[-1] - self.query_len)/ self.cs_seq_len)
        cs_encoding = torch.stack([
            sequence_output[:,self.query_len+i*self.cs_seq_len:self.query_len+(i+1)*self.cs_seq_len,:] for i in range(cs_len)
        ],
        dim = 1)    # cs_encoding [5b, cs_len, cs_seq_len, hidden]

        # cs_encoding = cs_encoding.view(cs_encoding.shape[0]*self.cs_len,cs_encoding.shape[2],cs_encoding.shape[3])

        query_encoding = sequence_output[:,:self.query_len,:]
        expanded_query_encoding = query_encoding.unsqueeze(1).expand(query_encoding.shape[0],cs_len,query_encoding.shape[1],query_encoding.shape[2])

        # dual attention module 
        # [5b, cs_len, query_seq_len, hidden]
        qc_attoutput, _ = self.cross_att(expanded_query_encoding,cs_encoding)
        # [5b, cs_len, cs_seq_len, hidden]
        cq_attoutput, _ = self.cross_att(cs_encoding,expanded_query_encoding)
        
        # re-distribution module, HeadHunter !
        mean_cs = torch.mean(cq_attoutput,dim = -2) # [5b, cs_len, hidden]
        cs_redistribution, attention_scores = self.self_att(mean_cs,mean_cs) # input same matrix to prefrom self-attention
        attention_scores = F.softmax(attention_scores, dim = -1).unsqueeze(1)

        cs_rep = torch.tanh(torch.matmul(attention_scores, cs_redistribution)).squeeze(1)
        
        
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

    