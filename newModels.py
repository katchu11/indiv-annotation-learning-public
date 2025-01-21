import logging
import torch
from torch import nn
from torch.nn import MSELoss
import torch.nn.functional as F
from transformers import RobertaForSequenceClassification, RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput
from openai import OpenAI

NUM_LABELS = 24016
logger = logging.getLogger('Roberta log')

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
        self.is_baseline = is_baseline
        self.use_demographics = use_demographics
        self.use_var = use_var
        self.cur_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.use_annotators = use_annotators
        self.use_annot_module = use_annot_module
        self.annotator_embedding_dim = 180

        head_hidden_size = config.hidden_size
        if self.use_annotators and self.use_annot_module:
            head_hidden_size += self.annotator_embedding_dim
            self.annotator_embed = nn.Embedding(num_embeddings=25000, embedding_dim=self.annotator_embedding_dim)

        self.roberta = RobertaModel(config)
        self.annotator_lin = nn.Linear(self.annotator_embedding_dim, self.annotator_embedding_dim)
        self.baseline_classifier = RobertaClassificationHead(config, head_hidden_size)

        self.post_init()
        self.model_parallel = True
        self.device_map = None

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None,
                label_mask=None, agg_label=None, ann_ids=None, demo_ids=None, demo_attention_mask=None, 
                utterance_index=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                               position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
                               output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                               return_dict=return_dict)
        sequence_output = outputs[0]

        category_info = None
        if self.use_annotators and self.use_annot_module:
            category_info = self.annotator_lin(self.annotator_embed(ann_ids.to(torch.int64)))

        if self.is_baseline:
            logits = self.baseline_classifier(sequence_output).flatten()
            labels = agg_label if not self.use_var else torch.var(labels.reshape(5, -1), dim=0).flatten()
        else:
            logits = self.baseline_classifier(sequence_output, category_info)
            labels = labels.to(dtype=torch.float)

        loss = None
        if labels is not None:
            loss_fct = MSELoss()
            if self.is_baseline:
                loss = loss_fct(logits.flatten(), labels.squeeze())
            else:
                loss = loss_fct(logits.flatten().squeeze(), labels.squeeze())

        if not return_dict:
            return ((loss,) + outputs[2:]) if loss is not None else (logits,) + outputs[2:]

        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,
                                        attentions=outputs.attentions)

class RobertaRecsSysClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, append_combo=False, head_hidden_size=None):
        super().__init__()
        self.head_hidden_size = config.hidden_size
        if head_hidden_size is not None:
            self.head_hidden_size = head_hidden_size
        self.dense = nn.Linear(self.head_hidden_size, config.hidden_size)
        classifier_dropout = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        recsysconfig = {"num_users": 24016, "emb_size": 768, "text_dim": 768, "n_hidden_1": 1024, "n_hidden_2": 1024, "n_hidden_3": 256}
        self.user_emb = nn.Embedding(recsysconfig["num_users"], recsysconfig["emb_size"])
        self.lin1 = nn.Linear(recsysconfig["emb_size"], recsysconfig["n_hidden_1"])
        self.lin3 = nn.Linear(recsysconfig["n_hidden_2"], recsysconfig["text_dim"])
        self.drop1 = nn.Dropout(0.1)

        self.lin4 = nn.Linear(recsysconfig["text_dim"] * 2, recsysconfig["text_dim"] * 2)
        self.lin5 = nn.Linear(recsysconfig["text_dim"] * 2, recsysconfig["text_dim"] * 2)
        self.lin6 = nn.Linear(recsysconfig["text_dim"] * 2, recsysconfig["text_dim"] * 2)
        self.out_proj_append = nn.Linear(recsysconfig["text_dim"] * 2, config.num_labels)

        self.combo = append_combo
        print(f"CUSTOM CONFIG: Appending Outputs? {self.combo}")

    def forward(self, features, ann_ids, category_info=None, **kwargs):
        U = self.user_emb(ann_ids)
        x = features[:, 0, :]
        if category_info is not None:
            x = torch.cat((x, category_info), dim=-1)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)

        U = F.relu(U)
        U = self.drop1(U)
        U = F.relu(self.lin1(U))
        U = self.lin3(U)

        if not self.combo:
            return (U * x).sum(1)
        else:
            c = torch.cat([U, x], dim=0)
            c = F.relu(c)
            c = F.relu(self.lin4(c))
            c = F.relu(self.lin5(c))
            c = F.relu(self.lin6(c))
            c = self.out_proj_append(c)
            return c

class RobertaRecsSysMultitaskModel(RobertaForSequenceClassification):
    def __init__(self, config, append_combo=False, num_labels=263, is_baseline=False, use_annotators=True, 
                 use_demographics=True, use_var=False, use_annot_module=False):
        super().__init__(config)
        config.num_labels = 1

        self.num_labels = config.num_labels
        self.config = config
        self.is_baseline = is_baseline
        self.use_demographics = use_demographics
        self.use_var = use_var
        self.cur_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.use_annotators = use_annotators
        self.use_annot_module = use_annot_module
        self.annotator_embedding_dim = 180

        head_hidden_size = config.hidden_size
        if self.use_annotators and self.use_annot_module:
            head_hidden_size += self.annotator_embedding_dim
            self.annotator_embed = nn.Embedding(num_embeddings=25000, embedding_dim=self.annotator_embedding_dim)

        self.roberta = RobertaModel(config)
        self.annotator_lin = nn.Linear(self.annotator_embedding_dim, self.annotator_embedding_dim)
        self.baseline_classifier = RobertaRecsSysClassificationHead(config, append_combo, head_hidden_size)

        self.post_init()
        self.model_parallel = True
        self.device_map = None

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None,
                label_mask=None, agg_label=None, ann_ids=None, demo_ids=None, demo_attention_mask=None, 
                utterance_index=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                               position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
                               output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                               return_dict=return_dict)
        sequence_output = outputs[0]

        category_info = None
        if self.use_annotators and self.use_annot_module:
            category_info = self.annotator_lin(self.annotator_embed(ann_ids.to(torch.int64)))

        if self.is_baseline:
            logits = self.baseline_classifier(sequence_output).flatten()
            labels = agg_label if not self.use_var else torch.var(labels.reshape(5, -1), dim=0).flatten()
        else:
            logits = self.baseline_classifier(sequence_output, ann_ids, category_info)
            labels = labels.to(dtype=torch.float)

        loss = None
        if labels is not None:
            loss_fct = MSELoss()
            if self.is_baseline:
                loss = loss_fct(logits.flatten(), labels.squeeze())
            else:
                loss = loss_fct(logits.flatten().squeeze(), labels.squeeze())

        if not return_dict:
            return ((loss,) + outputs[2:]) if loss is not None else (logits,) + outputs[2:]

        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,
                                        attentions=outputs.attentions)
class OpenAITextModel(nn.Module):
    def __init__(self, hidden_size=512, num_classes=10):
        super(OpenAITextModel, self).__init__()
        self.client = OpenAI()
        self.fc1 = nn.Linear(768, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc3 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_classes)

    def forward(self, text):
        response = self.client.embeddings.create(input=text, model="text-embedding-3-small")
        embeddings = torch.tensor(response.data[0].embedding)
        x = F.relu(self.fc1(embeddings))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class OpenAITextAnnotatorModel(nn.Module):
    def __init__(self, hidden_size=512, num_classes=10, annotator_embedding_dim=768):
        super(OpenAITextAnnotatorModel, self).__init__()
        self.client = OpenAI()
        self.annotator_embedding_dim = annotator_embedding_dim
        self.fc1 = nn.Linear(768 + self.annotator_embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc3 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_classes)

    def forward(self, text, annotator):
        text_response = self.client.embeddings.create(input=text, model="text-embedding-3-small")
        text_embeddings = torch.tensor(text_response.data[0].embedding)

        annotator_response = self.client.embeddings.create(input=annotator, model="text-embedding-3-small")
        annotator_embeddings = torch.tensor(annotator_response.data[0].embedding)

        combined_embeddings = torch.cat((text_embeddings, annotator_embeddings), dim=-1)
        x = F.relu(self.fc1(combined_embeddings))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
