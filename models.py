import logging
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from transformers import GPT2ForSequenceClassification, modeling_outputs, RobertaForSequenceClassification, RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput
import numpy as np

NUM_LABELS = 24016
DATASET = "toxjson"
logger = logging.getLogger('GPT2 log')

class DemographicsNN(nn.Module):

    def __init__(self,config,):
        super(DemographicsNN, self).__init__()
        self.fc1 = nn.Linear(1024, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.hidden_size * 2)
        self.fc3 = nn.Linear(config.hidden_size * 2, config.hidden_size * 4)
        self.fc4 = nn.Linear(config.hidden_size * 4, config.target_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, head_hidden_size=None):
        super().__init__()
        self.head_hidden_size = config.hidden_size
        if head_hidden_size is not None:
            self.head_hidden_size = head_hidden_size
        self.dense = nn.Linear(self.head_hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, category_info=None, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        if category_info is not None:
            x = torch.cat((x, category_info), dim=-1)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class RobertaMultitaskModel(RobertaForSequenceClassification):
    def __init__(self, config, num_labels=263, is_baseline=False, use_annotators=True, use_demographics=True,
                 use_var=False, use_annot_module=False):
        super().__init__(config)
        config.num_labels = 1

        self.num_labels = config.num_labels
        self.config = config
        self.num_annotators = NUM_LABELS
        self.toxjson_range = 4
        self.is_baseline = is_baseline
        self.use_demographics = use_demographics
        self.use_var = use_var
        self.cur_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Used only if annotator ID is incorporated via separate module (not helpful in current model)
        self.use_annotators = use_annotators
        self.use_annot_module = use_annot_module
        self.annotator_embedding_dim = 180
        self.num_annotator_embeddings = 25000  # this shouldn't have to be more than num_annotators

        head_hidden_size = config.hidden_size
        if self.use_annotators and self.use_annot_module:
            head_hidden_size += self.annotator_embedding_dim
            self.annotator_embed = nn.Embedding(num_embeddings=self.num_annotator_embeddings,
                                                embedding_dim=self.annotator_embedding_dim)

        self.roberta = RobertaModel(config)
        self.annotator_lin = nn.Linear(self.annotator_embedding_dim, self.annotator_embedding_dim)
        self.baseline_classifier = RobertaClassificationHead(config, head_hidden_size)  # don't rename this "classifier"--that's a diff param that shouldn't be used

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = True
        self.device_map = None

        print("Postprocessing config:\n", "num_labels:", config.num_labels, "baseline:", self.is_baseline, "annot:",
              self.use_annotators, "annot module:", self.use_annot_module, "demo:", self.use_demographics,
              "head state:", self.baseline_classifier.state_dict)

    def forward(
            self,
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
            __index_level_0__=None,
            label_mask=None,
            agg_label=None,
            ann_ids=None,
            demo_ids=None,
            demo_attention_mask=None,
            utterance_index=None
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]  # [8, 512, 768]

        category_info = None
        if self.use_annotators and self.use_annot_module:
            category_info = self.annotator_lin(self.annotator_embed(ann_ids.to(torch.int64)))

        if self.is_baseline:
            logits = self.baseline_classifier(sequence_output).flatten()
            if self.use_var:
                labels = torch.masked_select(labels, label_mask.to(dtype=torch.bool))
                labels = labels.reshape(5, int(len(labels)/5))
                labels = torch.var(labels, dim=0).flatten()
            else:
                labels = agg_label
        else:
            logits = self.baseline_classifier(sequence_output, category_info)
            labels = labels.to(dtype=torch.float)

        loss = None
        # train mode only:
        if labels is not None:
            loss_fct = MSELoss()
            if self.is_baseline:
                loss = loss_fct(logits.flatten(), labels.squeeze())
            else:
                if self.use_var:
                    loss = loss_fct(torch.var(logits.flatten(), dim=-1),
                                    torch.var(labels.squeeze()), dim=-1)
                else:
                    loss = loss_fct(logits.flatten().squeeze(), labels.squeeze())

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class MF(nn.Module):

    def __init__(self, num_users, text_dim, emb_size=8,n_hidden_1=256,n_hidden_2 = 128, n_hidden_3 = 64):
        super(MF, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding.from_pretrained(total_emb_tensor)
        self.lin1 = nn.Linear(text_dim, n_hidden_1)
        self.lin2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.lin3 = nn.Linear(n_hidden_2,n_hidden_3)
        self.lin4 = nn.Linear(n_hidden_3, emb_size)
        self.drop1 = nn.Dropout(0.1)
    def forward(self, u,v):
        U = self.user_emb(u) #creates an annotator embedding
        #The section below is where the inputs from RoBERTa are taken and back propogated through
        outputs = model(torch.LongTensor(train[v]["input_ids"]).resize_(1, 512), \
                    torch.LongTensor(train[v]["attention_mask"]).resize_(1, 512))
        V = outputs.last_hidden_state[:,0,:].to(torch.float32)
        x = F.relu(V)
        x = self.drop1(x)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = self.lin4(x)
        return (U*x).sum(1)

class RobertaRecsSysMultitaskModel(RobertaForSequenceClassification):
    def __init__(self, config, append_combo = False,num_labels=263, is_baseline=False, use_annotators=True, use_demographics=True,
                 use_var=False, use_annot_module=False):
        super().__init__(config)
        config.num_labels = 1

        self.num_labels = config.num_labels
        self.config = config
        self.num_annotators = NUM_LABELS
        self.toxjson_range = 4
        self.is_baseline = is_baseline
        self.use_demographics = use_demographics
        self.use_var = use_var
        self.cur_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Used only if annotator ID is incorporated via separate module (not helpful in current model)
        self.use_annotators = use_annotators
        self.use_annot_module = use_annot_module
        self.annotator_embedding_dim = 180
        self.num_annotator_embeddings = 25000  # this shouldn't have to be more than num_annotators

        head_hidden_size = config.hidden_size
        if self.use_annotators and self.use_annot_module:
            head_hidden_size += self.annotator_embedding_dim
            self.annotator_embed = nn.Embedding(num_embeddings=self.num_annotator_embeddings,
                                                embedding_dim=self.annotator_embedding_dim)

        ##self.roberta = RobertaModel(config).from_pretrained("./pretrained_multitask_base/")
        self.roberta = RobertaModel(config)#.from_pretrained("./pretrained_multitask_demographic/")
        freeze = False
        if freeze:
            print("ROBERTA MODEL IS BEING FROZEN")
            for param in self.roberta.parameters():
                param.requires_grad = False
        self.annotator_lin = nn.Linear(self.annotator_embedding_dim, self.annotator_embedding_dim)
        self.baseline_classifier = RobertaRecSysClassificationHead(config, append_combo,head_hidden_size)  # don't rename this "classifier"--that's a diff param that shouldn't be used
        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = True
        self.device_map = None
        
        # recsysconfig = {"num_users":24016,
        #         "emb_size":8,
        #         "text_dim": 768,
        #          "n_hidden_1":256,
        #        "n_hidden_2":128,
        #        "n_hidden_3":64}
        # self.user_emb = nn.Embedding(24016, 8) #.to("cuda")
        # self.lin1 = nn.Linear(768,256) #.to("cuda")
        # self.lin2 = nn.Linear(256,128) #.to("cuda")
        # self.lin3 = nn.Linear(128,64) #.to("cuda")
        # self.lin4 = nn.Linear(64, 8) #.to("cuda")
        # self.drop1 = nn.Dropout(0.1)

        print("Postprocessing config:\n", "num_labels:", config.num_labels, "baseline:", self.is_baseline, "annot:",
              self.use_annotators, "annot module:", self.use_annot_module, "demo:", self.use_demographics,
              "head state:", self.baseline_classifier.state_dict)

    def forward(
            self,
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
            __index_level_0__=None,
            label_mask=None,
            agg_label=None,
            ann_ids=None,
            demo_ids=None,
            demo_attention_mask=None,
            utterance_index=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # U = self.user_emb(ann_ids)
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]  # [8, 512, 768]

        category_info = None
        if self.use_annotators and self.use_annot_module:
            category_info = self.annotator_lin(self.annotator_embed(ann_ids.to(torch.int64)))

        if self.is_baseline:
            logits = self.baseline_classifier(sequence_output).flatten()
            if self.use_var:
                labels = torch.masked_select(labels, label_mask.to(dtype=torch.bool))
                labels = labels.reshape(5, int(len(labels)/5))
                labels = torch.var(labels, dim=0).flatten()
            else:
                labels = agg_label
        else:
            logits = self.baseline_classifier(sequence_output, ann_ids, category_info)
            labels = labels.to(dtype=torch.float)
            
        loss = None
        # train mode only:
        if labels is not None:
            loss_fct = MSELoss()
            if self.is_baseline:
                loss = loss_fct(logits.flatten(), labels.squeeze())
            else:
                if self.use_var:
                    loss = loss_fct(torch.var(logits.flatten(), dim=-1),
                                    torch.var(labels.squeeze()), dim=-1)
                else:
                    loss = loss_fct(logits.flatten().squeeze(), labels.squeeze())
                    
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class RobertaRecSysClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, append_combo=False,head_hidden_size=None):
        super().__init__()
        self.head_hidden_size = config.hidden_size
        if head_hidden_size is not None:
            self.head_hidden_size = head_hidden_size
        self.dense = nn.Linear(self.head_hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        #recsys initialization
        recsysconfig = {"num_users":24016,
                "emb_size":768,
                "text_dim": 768,
                 "n_hidden_1":1024,
               "n_hidden_2":1024,
               "n_hidden_3":256}
        self.user_emb = nn.Embedding(recsysconfig["num_users"], recsysconfig["emb_size"])
        self.lin1 = nn.Linear(recsysconfig["emb_size"], recsysconfig["n_hidden_1"])
        # self.lin2 = nn.Linear(recsysconfig["n_hidden_1"], recsysconfig["n_hidden_2"])
        self.lin3 = nn.Linear(recsysconfig["n_hidden_2"],recsysconfig["text_dim"])
        # self.lin4 = nn.Linear(recsysconfig["emb_size"], recsysconfig["n_hidden_3"])
        self.drop1 = nn.Dropout(0.1)

        # Linear Layers for combo
        self.lin4 = nn.Linear(recsysconfig["text_dim"] * 2, recsysconfig["text_dim"] * 2)
        self.lin5 = nn.Linear(recsysconfig["text_dim"] * 2, recsysconfig["text_dim"] * 2)
        self.lin6 = nn.Linear(recsysconfig["text_dim"] * 2, recsysconfig["text_dim"] * 2)
        self.out_proj_append = nn.Linear(recsysconfig["text_dim"] * 2, config.num_labels)

        self.combo = append_combo # Appending U and x and making a downstream prediction off that
        print(f"CUSTOM CONFIG: Appending Outputs? {self.combo}")

    def forward(self, features, ann_ids, category_info=None,**kwargs):
        U = self.user_emb(ann_ids)
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        if category_info is not None: #category info will be none
            x = torch.cat((x, category_info), dim=-1)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)

        #Recsys
        U = F.relu(U)
        U = self.drop1(U)
        U = F.relu(self.lin1(U))
        U = self.lin3(U)

        if not self.combo:
            return (U * x).sum(1)
        else:
            c = torch.cat([U,x], dim = 0)
            c = F.relu(c)
            c = F.relu(self.lin4(c))
            c = F.relu(self.lin5(c))
            c = F.relu(self.lin6(c))
            c = self.out_proj_append(c)
            return c
            