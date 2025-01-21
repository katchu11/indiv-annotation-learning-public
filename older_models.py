import logging
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from transformers import GPT2ForSequenceClassification, modeling_outputs, RobertaForSequenceClassification, RobertaModel
import numpy as np

NUM_LABELS = 24016
DATASET = "toxjson"
logger = logging.getLogger('GPT2 log')

# Older model
class MultilabelModel(GPT2ForSequenceClassification):
    def __init__(self, config, num_labels=263):
        super().__init__(config)
        # fself.num_labels = config.num_labels
        # self.transformer1 = GPT2Model(config)
        # self.transformer2 = GPT2Model(config)
        # self.transformer3 = GPT2Model(config)
        self.transformer = self.transformer  # MyDataParallel(
        print("IN INIT")
        self.num_labels = num_labels
        print(self.num_labels)
        self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)  # MyDataParallel(nn.DataParallel()

        # Model parallel
        self.model_parallel = True
        self.device_map = None

        # Initialize weights and apply final processing
        super().post_init()

    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            __index_level_0__=None  # Is this necessary?
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        print("labels at start", labels.shape, labels)

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        logits = self.score(hidden_states)  # We need one of these for each annotator

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        assert (
                self.config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    f"unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = logits[range(batch_size), sequence_lengths]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # p_cpu = pooled_logits.detach().cpu()
                # l_cpu = labels.detach().cpu()
                # print("orig", torch.isnan(pooled_logits).any(), torch.isnan(labels).any())
                # print("multilabel labels", labels.shape, labels)
                labels = torch.nan_to_num(labels).flatten()
                pooled_logits = pooled_logits.flatten()
                good_label_indices = (labels != 42).nonzero(as_tuple=False)
                indices_by_row = np.array(len(labels))
                good_labels = torch.unsqueeze(torch.index_select(labels, 0, good_label_indices.flatten()), 1)
                good_logits = torch.unsqueeze(torch.index_select(pooled_logits, 0, good_label_indices.flatten()), 1)

                good_labels = good_labels.float()
                loss_fct = BCEWithLogitsLoss()
                # print("good", good_logits, good_labels)
                # g_cpu = good_labels.detach().cpu()
                # glog_cpu = good_logits.detach().cpu()
                # print("good", torch.isnan(good_logits).any(), torch.isnan(good_labels).any())
                loss = loss_fct(good_logits, good_labels)
                # print("LOSS:", loss)

        if not return_dict:
            if self.training:  # check!!
                # print("train mode")
                output = (pooled_logits,) + transformer_outputs[
                                            1:]  # ** check whether to use pooled or good logits here
            else:
                # print("test mode")
                output = (good_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        if self.training:
            # print("train mode")

            return modeling_outputs.SequenceClassifierOutputWithPast(
                loss=loss,
                logits=pooled_logits,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
            )
        else:
            return modeling_outputs.SequenceClassifierOutputWithPast(
                loss=loss,
                logits=good_logits,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
            )


class MultitaskModel(GPT2ForSequenceClassification):
    def __init__(self, config, num_labels=263):
        print("before super init")
        super().__init__(config)
        print("after super init")
        # self.num_labels = config.num_labels
        # self.transformer1 = GPT2Model(config)
        # self.transformer2 = GPT2Model(config)
        # self.transformer3 = GPT2Model(config)
        self.transformer = self.transformer  # MyDataParallel()#nn.DataParallel()
        print("IN INIT")
        self.num_labels = NUM_LABELS  # hacky but might work
        print("num labels/num annotators:", self.num_labels)
        self.linear_layers = nn.ModuleList()
        self.num_annotators = self.num_labels  # change for clustering
        self.toxjson_range = 4

        if DATASET == "sbic":
            for i in range(self.num_annotators):
                self.linear_layers.append(nn.Linear(config.n_embd, self.num_labels, bias=False)).to(
                    self.device)  # MyDataParallel(n
        else:
            print("creating toxjson final layers")
            for i in range(self.num_annotators):
                self.linear_layers.append(nn.Linear(config.n_embd, 1, bias=False)).to(self.device)  # MyDataParallel(n

        # Model parallel
        self.model_parallel = True
        self.device_map = None

        # Initialize weights and apply final processing
        super().post_init()
        print("config.num_labels after postprocess", config.num_labels)

    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            __index_level_0__=None,
            label_mask=None,
            agg_label=None
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        assert (
                self.config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    f"unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        if DATASET == "sbic":
            foo = self.linear_layers[0](hidden_states)
            for i, layer in enumerate(
                    self.linear_layers):  # you don't need to pass it thru all the lin layers!! only the relevant ones
                foo = self.linear_layers[i](hidden_states)
                print(i)
            print("lin0", foo.shape, "num lin layers", len(self.linear_layers))
            all_logits = torch.stack([layer(hidden_states) for layer in self.linear_layers], axis=0)
            # print("foo", foo.shape)
            # classifier x batch size x hidden size x logits
        else:
            # print("hidden states", hidden_states.shape) # 8 x 1024 x 768: batch_size x hidden size? x n_embd
            # print("label mask", label_mask.shape)
            # print("n_embd", self.config.n_embd)
            good_indices = label_mask.nonzero(as_tuple=True)
            # print("good indices", good_indices)
            # for batch_i, ann_i in zip(good_indices[0], good_indices[1]):
            #     print("indexing:", batch_i, ann_i, hidden_states[batch_i].shape)

            all_logits = torch.stack([self.linear_layers[ann_i](hidden_states[batch_i]) for batch_i, ann_i in
                                      zip(good_indices[0], good_indices[1])], axis=0)
            all_seqlens = torch.stack([sequence_lengths[batch_i] for batch_i in good_indices[0]], axis=0)
            labels = torch.masked_select(labels, label_mask.to(dtype=torch.bool))
            # print("labels", labels.shape, labels, "logits", all_logits.shape) # should be same size

        if DATASET == "sbic":
            # all_logits: classifier x batch size x hidden size x logits; seq_lens: num classifiers x batch size x hidden size? x n_labels
            all_logits_pooled = all_logits[:, range(batch_size),
                                sequence_lengths]  # 263 (classifiers) x 8 x 263 (labels)

        else:
            # print("all logits", all_logits.shape, "seq lengths", sequence_lengths, "all seq lens", all_seqlens) # 40 x 1024 x 1: n_classifiers x batch size x hidden size x n_labels
            # print("labels", labels.shape)
            good_logits = all_logits[range(all_logits.shape[0]), all_seqlens].squeeze()  # for each batch item, get output for last item of sequence
            # print("toxjson all logits pooled", good_logits.shape)                           # 40

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if DATASET == "toxjson" or self.config.problem_type == "regression":
                print("gl before", good_logits)
                good_logits = self.toxjson_range * F.softmax(good_logits)
                loss_fct = MSELoss()
                # print("labels", labels)#, "logits", good_logits)
                if self.num_labels == 1:
                    loss = loss_fct(good_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(good_logits, labels)
                # print("loss", loss)

            elif self.config.problem_type == "multi_label_classification":
                # print("multi label class")
                print("labels b4", labels)
                labels = torch.nan_to_num(labels)
                if self.training:
                    print("labels after", labels)
                    good_label_inds_logits = (labels != 42).nonzero()  # 8 x 38???

                    # print("good labels not flat", good_label_inds_logits, "label shape", labels.shape)
                    good_labels_inds_reproj = np.array(
                        [[gl[1].item(), gl[0].item(), gl[1].item()] for gl in good_label_inds_logits]).T
                    labels = labels.flatten()
                    all_logits_pooled = all_logits_pooled.flatten()
                    # Then send only the nonzero values to loss
                    good_label_indices = (labels != 42).nonzero(as_tuple=False)
                    # print(good_labels_inds_reproj, "n labels", self.num_labels)
                    good_label_inds_raveled = torch.IntTensor(np.ravel_multi_index(good_labels_inds_reproj, (
                        self.num_labels, batch_size, self.num_labels))).type(torch.int64).to(self.device)  # CHANGE

                    indices_by_row = np.array(len(labels))
                    good_labels = torch.unsqueeze(torch.index_select(labels, 0, good_label_indices.flatten()), 1)
                    good_logits = torch.unsqueeze(
                        torch.index_select(all_logits_pooled, 0, good_label_inds_raveled.flatten()), 1)

                else:
                    good_labels = labels
                    # print(all_logits_pooled.shape, all_logits_pooled)
                    good_logits_raw = torch.zeros((all_logits_pooled.shape[0], all_logits_pooled.shape[1])).to(
                        self.device)  # 263 x 8
                    # print("raw", good_logits_raw.shape)
                    for i in range(all_logits_pooled.shape[0]):
                        good_logits_raw[i] = all_logits_pooled[i, :, i]
                    # print(good_logits_raw)
                    good_logits = torch.mean(good_logits_raw, axis=0)  # 1 x 8
                    # print("glog", good_logits.shape, good_logits)
                good_labels = good_labels.float()
                # print("glabels", good_labels)
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(good_logits, good_labels)
                # print("loss out", loss, good_logits, good_labels)

        if DATASET == "toxjson":
            all_logits_pooled = good_logits
        if not return_dict:
            if model.training:
                # print("train mode")
                output = (all_logits_pooled,) + transformer_outputs[
                                                1:]  # ** check whether to use pooled or good logits here
            else:
                # print("test mode")
                output = (good_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        if model.training:
            # print("train mode")

            return modeling_outputs.SequenceClassifierOutputWithPast(
                loss=loss,
                logits=all_logits_pooled,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
            )
        else:
            return modeling_outputs.SequenceClassifierOutputWithPast(
                loss=loss,
                logits=good_logits,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
            )
