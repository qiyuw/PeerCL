import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers
from transformers import RobertaTokenizer, AutoModel, AutoConfig
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel
from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()

    # group wise arguments
    cls.peer_coop = cls.model_args.peer_coop
    cls.posi_cst = cls.model_args.posi_cst

    # temporal useless arguments
    cls.fix_peer_model = cls.model_args.fix_peer_model
    if cls.fix_peer_model:
        cls.peer_model = AutoModel.from_pretrained("princeton-nlp/unsup-simcse-bert-base-uncased")
    else:
        cls.peer_model = AutoModel.from_pretrained(cls.model_args.model_name_or_path)

    cls.sup_or_unsup = cls.model_args.sup_or_unsup
    cls.use_negative = cls.model_args.use_negative
    cls.use_xnli = cls.model_args.use_xnli
    cls.use_wmt = cls.model_args.use_wmt

# calculate denoise loss between two networks (pooler and label)
def get_cross_denoise_loss(cls, pooler_output, label_output):
    # align agreement
    if cls.use_wmt == True:
        z1, zs = pooler_output[:, 0, :], pooler_output[:, 1:, :]
        label_z1, label_zs = label_output[:, 0, :], label_output[:, 1:, :]
    else:
        z1, zs = pooler_output[:, 0, :], pooler_output[:, 1:, :]
        label_z1, label_zs = label_output[:, 0, :], label_output[:, 1:, :]
    
    neg_cos_sim = cls.sim(z1.unsqueeze(1), label_z1.unsqueeze(0)) # bz, bz
    neg_cos_sim = torch.tril(neg_cos_sim, diagonal=-1)[:,:-1] + torch.triu(neg_cos_sim, diagonal=1)[:,1:] # remove diag, (bz, bz-1)
    cos_sim = cls.sim(z1.unsqueeze(1), label_zs) # bz, num_aug
    label_neg_cos_sim = cls.sim(label_z1.unsqueeze(1), z1.unsqueeze(0)) # bz, bz
    label_neg_cos_sim = torch.tril(label_neg_cos_sim, diagonal=-1)[:,:-1] + torch.triu(label_neg_cos_sim, diagonal=1)[:,1:] # remove diag, (bz, bz-1)
    label_cos_sim = cls.sim(label_z1.unsqueeze(1), zs) # bz, num_aug
    m = nn.LogSoftmax(dim=1)
    loss_fct = nn.KLDivLoss(reduction='none', log_target=True)
    label_cos_sim = torch.cat([label_cos_sim, label_neg_cos_sim], dim=1)
    cos_sim = torch.cat([cos_sim, neg_cos_sim], dim=1)
    loss = loss_fct(m(cos_sim), m(label_cos_sim)).mean()
    return loss

# calculate kl loss between two networks (pooler and label)
def get_sym_kl_loss(cls, pooler_output, label_output):
    # align embedding space
    if cls.use_wmt == True:
        z1, zs = pooler_output[:, 0, :], pooler_output[:, 1:, :]
        label_z1, label_zs = label_output[:, 0, :], label_output[:, 1:, :]
    else:
        z1, zs = pooler_output[:, 0, :], pooler_output[:, 1:, :]
        label_z1, label_zs = label_output[:, 0, :], label_output[:, 1:, :]
        
    neg_cos_sim = cls.sim(z1.unsqueeze(1), z1.unsqueeze(0)) # bz, bz
    neg_cos_sim = torch.tril(neg_cos_sim, diagonal=-1)[:,:-1] + torch.triu(neg_cos_sim, diagonal=1)[:,1:] # remove diag, (bz, bz-1)
    cos_sim = cls.sim(z1.unsqueeze(1), zs) # bz, num_aug
    label_neg_cos_sim = cls.sim(label_z1.unsqueeze(1), label_z1.unsqueeze(0)) # bz, bz
    label_neg_cos_sim = torch.tril(label_neg_cos_sim, diagonal=-1)[:,:-1] + torch.triu(label_neg_cos_sim, diagonal=1)[:,1:] # remove diag, (bz, bz-1)
    label_cos_sim = cls.sim(label_z1.unsqueeze(1), label_zs) # bz, num_aug
    m = nn.LogSoftmax(dim=1)
    loss_fct = nn.KLDivLoss(reduction='none', log_target=True)
    label_cos_sim = torch.cat([label_cos_sim, label_neg_cos_sim], dim=1)
    cos_sim = torch.cat([cos_sim, neg_cos_sim], dim=1)
    loss = loss_fct(m(cos_sim), m(label_cos_sim)).mean()
    return loss

'''
def get_cross_kl_loss(cls, pooler_output, label_output):
    # not used
    bsz = pooler_output.size(0)
    pooler_output_1, pooler_output_2 = pooler_output[:bsz//2], pooler_output[bsz//2:]
    label_output_1, label_output_2 = label_output[:bsz//2], label_output[bsz//2:]
    return get_asym_kl_loss(cls, pooler_output_1, label_output_1) + get_asym_kl_loss(cls, label_output_2, pooler_output_2)


def get_asym_kl_loss(cls, pooler_output, label_output):
    z1, zs = pooler_output[:, 0, :], pooler_output[:, 1:, :]
    neg_cos_sim = cls.sim(z1.unsqueeze(1), z1.unsqueeze(0)) # bz, bz
    neg_cos_sim = torch.tril(neg_cos_sim, diagonal=-1)[:,:-1] + torch.triu(neg_cos_sim, diagonal=1)[:,1:] # remove diag, (bz, bz-1)
    cos_sim = cls.sim(z1.unsqueeze(1), zs) # bz, num_aug
    cos_sim = torch.cat([cos_sim, neg_cos_sim], dim=1)
    with torch.no_grad():
        label_z1, label_zs = label_output[:, 0, :], label_output[:, 1:, :]
        label_neg_cos_sim = cls.sim(label_z1.unsqueeze(1), label_z1.unsqueeze(0)) # bz, bz
        label_neg_cos_sim = torch.tril(label_neg_cos_sim, diagonal=-1)[:,:-1] + torch.triu(label_neg_cos_sim, diagonal=1)[:,1:] # remove diag, (bz, bz-1)
        label_cos_sim = cls.sim(label_z1.unsqueeze(1), label_zs) # bz, num_aug
        label_cos_sim = torch.cat([label_cos_sim, label_neg_cos_sim], dim=1)
    m = nn.LogSoftmax(dim=1)
    loss_fct = nn.KLDivLoss(reduction='none', log_target=True)
    return loss_fct(m(cos_sim), m(label_cos_sim)).mean()
'''

def get_bce_loss(cls, pooler_output):
    # peer positive contrast
    if cls.sup_or_unsup == 'sup' and cls.use_negative == True:
        z1, z2, z3, zs = pooler_output[:, 0, :], pooler_output[:, 1, :], pooler_output[:, 2, :], pooler_output[:, 3:, :]
        z2 = z2.unsqueeze(1)
        zs = torch.cat([z2, zs], dim=1)
    elif cls.use_wmt == True:
        z1, zs = pooler_output[:, 0, :], pooler_output[:, 1:, :]
    else:
        z1, zs = pooler_output[:, 0, :], pooler_output[:, 1:, :]
    neg_cos_sim = cls.sim(z1.unsqueeze(1), z1.unsqueeze(0)) # bz, bz
    neg_cos_sim = torch.tril(neg_cos_sim, diagonal=-1)[:,:-1] + torch.triu(neg_cos_sim, diagonal=1)[:,1:] # remove diag, (bz, bz-1)
    if cls.sup_or_unsup == 'sup' and cls.use_negative == True:
        hard_neg_cos_sim = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
        neg_cos_sim = torch.cat([neg_cos_sim, hard_neg_cos_sim], dim=1)
    cos_sim = cls.sim(z1.unsqueeze(1), zs) # bz, num_aug
    labels = torch.cat([torch.ones_like(cos_sim), torch.zeros_like(neg_cos_sim)], dim=1)
    cos_sim = torch.cat([cos_sim, neg_cos_sim], dim=1)

    m = nn.Softmax(dim=1)
    loss_fct = nn.BCEWithLogitsLoss(reduction='none')
    return loss_fct(m(cos_sim), labels).mean()


def get_simcse_loss(cls, pooler_output):
    z1, z2 = pooler_output[:, 0], pooler_output[:, 1]
    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    if cls.use_negative:
        z3 = pooler_output[:, 2]
        z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
        cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)
    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = nn.CrossEntropyLoss()
    if cls.use_negative:
        z3_weight = cls.model_args.hard_negative_weight
        weights = torch.tensor(
            [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
        ).to(cls.device)
        cos_sim = cos_sim + weights
    return loss_fct(cos_sim, labels), cos_sim
    

def cl_forward(cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: multiple augmentation
    num_sent = input_ids.size(1)

    # Test, use same input for two networks, reduce the difference only with the randomness within the model.
    if True:
        peer_input_ids = input_ids
        peer_attention_mask = attention_mask
        peer_token_type_ids = token_type_ids

    mlm_outputs = None
    # Flatten input for encoding
    input_ids = input_ids.reshape((-1, input_ids.size(-1))) # (bs * num_sent, len)
    peer_input_ids = peer_input_ids.reshape((-1, peer_input_ids.size(-1)))
    attention_mask = attention_mask.reshape((-1, attention_mask.size(-1))) # (bs * num_sent len)
    peer_attention_mask = peer_attention_mask.reshape((-1, peer_attention_mask.size(-1)))
    if token_type_ids is not None:
        token_type_ids = token_type_ids.reshape((-1, token_type_ids.size(-1))) # (bs * num_sent, len)
        peer_token_type_ids = peer_token_type_ids.reshape((-1, token_type_ids.size(-1)))

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )
    if cls.peer_coop:
        if cls.fix_peer_model:
            with torch.no_grad():
                label_output = cls.peer_model(peer_input_ids)['pooler_output']
                label_output = label_output.view((batch_size, num_sent, label_output.size(-1))) # (bs, num_sent, hidden)
                if cls.pooler_type == "cls":
                    label_output = cls.mlp(label_output)
        else:
            label_output = cls.peer_model(peer_input_ids)['pooler_output']
            label_output = label_output.view((batch_size, num_sent, label_output.size(-1))) # (bs, num_sent, hidden)
            if cls.pooler_type == "cls":
                label_output = cls.mlp(label_output)
    else:
        label_output = None
        
    # MLM auxiliary objective
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        mlm_outputs = encoder(
            mlm_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )

    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)
        
    # Hard negative NOT IMPLEMENTED
    # if num_sent >= 3:
    #     z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
    #     cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

    # Hard negative NOT IMPLEMENTED
    # if num_sent == 3:
    #     z3 = pooler_output[:, 2]

    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:
        raise NotImplementedError
        # Gather hard negative NOT IMPLEMENTED
        # if num_sent >= 3:
        #     z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
        #     dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
        #     z3_list[dist.get_rank()] = z3
        #     z3 = torch.cat(z3_list, 0)

        # Dummy vectors for allgather
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        zs_list = [torch.zeros_like(zs) for _ in range(dist.get_world_size())]
        # Allgather
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=zs_list, tensor=zs.contiguous())

        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        z1_list[dist.get_rank()] = z1
        zs_list[dist.get_rank()] = zs
        # Get full batch embeddings: (bs x N, hidden)
        z1 = torch.cat(z1_list, 0)
        zs = torch.cat(zs_list, 0)

    # labels = torch.zeros(cos_sim.size(0)).long().to(cls.device) # situate the positive cos sim on the 1st place

    # Calculate loss with hard negatives NOT IMPLEMENTED
    # if num_sent == 3:
    #     # Note that weights are actually logits of weights
    #     z3_weight = cls.model_args.hard_negative_weight
    #     weights = torch.tensor(
    #         [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
    #     ).to(cls.device)
    #     cos_sim = cos_sim + weights

    # Calculate loss for MLM
    if mlm_outputs is not None and mlm_labels is not None:
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        masked_lm_loss = loss_fct(prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1))
        loss = loss + cls.model_args.mlm_weight * masked_lm_loss
    
    if cls.peer_coop:
        kl_loss = get_cross_denoise_loss(cls, pooler_output, label_output) + get_sym_kl_loss(cls, pooler_output, label_output)
    simcse_loss, cos_sim = get_simcse_loss(cls, pooler_output)
    if cls.peer_coop:
        if cls.sup_or_unsup == 'unsup':
            loss = simcse_loss + kl_loss
        else:
            loss = 1*simcse_loss + kl_loss
    else:
        if cls.sup_or_unsup == 'unsup':
            loss = simcse_loss
        else:
            loss = 1*simcse_loss
    if cls.posi_cst:
        bce_loss = get_bce_loss(cls, pooler_output)
        loss += bce_loss 
    if not cls.fix_peer_model and cls.peer_coop:
        denoiser_loss = get_simcse_loss(cls, label_output)[0]
        if cls.posi_cst:
            denoiser_loss += get_bce_loss(cls, label_output)
        loss += denoiser_loss

    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )



class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = RobertaLMHead(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )


class XLMRForCL(XLMRobertaModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = RobertaLMHead(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )


class LaBSEForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )