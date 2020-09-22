from transformers import BertPreTrainedModel,BertModel,AlbertModel,AlbertPreTrainedModel,RobertaModel,XLNetPreTrainedModel,XLNetModel
from transformers.modeling_utils import SequenceSummary
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch
import torch.nn as nn
import pdb


# IR-CSQA
class BertForMultipleChoice(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

# @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("(batch_size, num_choices, sequence_length)"))
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=False,
    ):
        
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

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
        )

        pooled_output = outputs[1]
        # pdb.set_trace()
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)

class AlbertForMultipleChoice(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

# @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("(batch_size, num_choices, sequence_length)"))
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=False,
    ):
        
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

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
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)

# class RobertaForMultipleChoice(BertPreTrainedModel):
#     config_class = RobertaConfig
#     base_model_prefix = "roberta"

#     def __init__(self, config):
#         super().__init__(config)

#         self.roberta = RobertaModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, 1)

#         self.init_weights()

#     def forward(
#         self,
#         input_ids=None,
#         token_type_ids=None,
#         attention_mask=None,
#         labels=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#     ):
#         num_choices = input_ids.shape[1]

#         flat_input_ids = input_ids.view(-1, input_ids.size(-1))
#         flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
#         flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
#         flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
#         outputs = self.roberta(
#             flat_input_ids,
#             position_ids=flat_position_ids,
#             token_type_ids=flat_token_type_ids,
#             attention_mask=flat_attention_mask,
#             head_mask=head_mask,
#         )
#         pooled_output = outputs[1]

#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)
#         reshaped_logits = logits.view(-1, num_choices)

#         outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(reshaped_logits, labels)
#             outputs = (loss,) + outputs

#         return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)

class RobertaForMultipleChoice(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

# @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("(batch_size, num_choices, sequence_length)"))
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=False,
    ):
        # pdb.set_trace()
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        # token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        token_type_ids = None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )
        # pdb.set_trace()
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)
        # pdb.set_trace()
        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)



class XLNetForMultipleChoice(XLNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.transformer = XLNetModel(config)
        self.sequence_summary = SequenceSummary(config)
        self.logits_proj = nn.Linear(config.d_model, 1)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        input_mask=None,
        attention_mask=None,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=True,
        labels=None,
    ):

        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_input_mask = input_mask.view(-1, input_mask.size(-1)) if input_mask is not None else None

        transformer_outputs = self.transformer(
            flat_input_ids,
            token_type_ids=flat_token_type_ids,
            input_mask=flat_input_mask,
            attention_mask=flat_attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )

        output = transformer_outputs[0]

        output = self.sequence_summary(output)
        logits = self.logits_proj(output)
        reshaped_logits = logits.view(-1, num_choices)
        outputs = (reshaped_logits,) + transformer_outputs[
            1:
        ]  # Keep mems, hidden states, attentions if there are in it

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # return (loss), logits, (mems), (hidden states), (attentions)

## reranker

class SelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.query = nn.Linear(config.hidden_size,config.hidden_size)
        self.key = nn.Linear(config.hidden_size,config.hidden_size)
        self.value = nn.Linear(config.hidden_size,config.hidden_size)
        # self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self,hidden_states):
        # hidden_states should be (batch_size,document_length,hidden_size)
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)
        return context_layer

class BertAttRanker(BertPreTrainedModel):
    def __init__(self, config, cs_len):
        super().__init__(config)
        self.cs_len = cs_len
        self.bert = BertModel(config)
        self.self_att = SelfAttention(config)
        # self.classifier = nn.Linear(config.hidden_size,1)
        self.classifier = nn.Linear(config.hidden_size*self.cs_len,1)
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
        batch_size,input_size = input_ids.shape[:2]
        num_choices = int(input_size/self.cs_len)
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        bert_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        pooled_output = bert_outputs[1]
        
        reshaped_output = pooled_output.view(int(batch_size*num_choices),self.cs_len,pooled_output.size(-1))

        atten_output = self.self_att(reshaped_output)
        # pdb.set_trace()
        # mean_pooled_output = atten_output.mean(dim=1)

        # mean_pooled_output = self.dropout(mean_pooled_output)
        
        # logits = self.classifier(mean_pooled_output)


        reshaped_output = atten_output.view(int(batch_size*num_choices),self.cs_len*atten_output.size(-1))
        logits = self.classifier(reshaped_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs

class AlbertAttRanker(AlbertPreTrainedModel):
    def __init__(self, config, cs_len):
        super().__init__(config)
        self.cs_len = cs_len
        self.bert = AlbertModel(config)
        self.self_att = SelfAttention(config)
        # self.classifier = nn.Linear(config.hidden_size,1)
        self.classifier = nn.Linear(config.hidden_size*self.cs_len,1)
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
        batch_size,input_size = input_ids.shape[:2]
        num_choices = int(input_size/self.cs_len)
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )
        # pdb.set_trace()

        bert_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        pooled_output = bert_outputs[1]
        
        reshaped_output = pooled_output.view(int(batch_size*num_choices),self.cs_len,pooled_output.size(-1))

        atten_output = self.self_att(reshaped_output)
        # pdb.set_trace()
        # mean_pooled_output = atten_output.mean(dim=1)

        # mean_pooled_output = self.dropout(mean_pooled_output)
        
        # logits = self.classifier(mean_pooled_output)


        reshaped_output = atten_output.view(int(batch_size*num_choices),self.cs_len*atten_output.size(-1))
        logits = self.classifier(reshaped_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs

class RobertaAttRanker(BertPreTrainedModel):
    def __init__(self, config, cs_len):
        super().__init__(config)
        self.cs_len = cs_len
        self.bert = RobertaModel(config)
        self.self_att = SelfAttention(config)
        # self.classifier = nn.Linear(config.hidden_size,1)
        self.classifier = nn.Linear(config.hidden_size*self.cs_len,1)
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
        batch_size,input_size = input_ids.shape[:2]
        num_choices = int(input_size/self.cs_len)
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )
        # pdb.set_trace()

        bert_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        pooled_output = bert_outputs[1]
        
        reshaped_output = pooled_output.view(int(batch_size*num_choices),self.cs_len,pooled_output.size(-1))

        atten_output = self.self_att(reshaped_output)
        # pdb.set_trace()
        # mean_pooled_output = atten_output.mean(dim=1)

        # mean_pooled_output = self.dropout(mean_pooled_output)
        
        # logits = self.classifier(mean_pooled_output)


        reshaped_output = atten_output.view(int(batch_size*num_choices),self.cs_len*atten_output.size(-1))
        logits = self.classifier(reshaped_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs

class XLNetAttRanker(XLNetPreTrainedModel):
    def __init__(self, config, cs_len):
        super().__init__(config)
        self.cs_len = cs_len
        self.transformer = XLNetModel(config)
        self.self_att = SelfAttention(config)
        self.classifier = nn.Linear(config.hidden_size*self.cs_len,1)
        self.dropout = nn.Dropout(config.dropout)
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
        batch_size,input_size = input_ids.shape[:2]
        num_choices = int(input_size/self.cs_len)
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )


        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = self.
        pooled_output = transformer_outputs[1]
        
        reshaped_output = pooled_output.view(int(batch_size*num_choices),self.cs_len,pooled_output.size(-1))

        atten_output = self.self_att(reshaped_output)


        reshaped_output = atten_output.view(int(batch_size*num_choices),self.cs_len*atten_output.size(-1))
        logits = self.classifier(reshaped_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs