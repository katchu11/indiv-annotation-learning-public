import argparse
import csv
import json
import logging
import os
import pickle
from statistics import mean, variance
from typing import List, Optional

import datasets
import numpy as np
from nltk.corpus import wordnet
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from transformers import (AutoConfig, AutoTokenizer, GPT2Config, GPT2LMHeadModel,
                          GPT2TokenizerFast, RobertaForSequenceClassification,
                          RobertaModel, Trainer, TrainingArguments)
from datasets import load_metric, load_from_disk
import wandb
from models import RobertaMultitaskModel, RobertaRecsSysMultitaskModel

logger = logging.getLogger('GPT2 log')
NUM_LABELS = 24016
FAST_HEAD_LR = False
FREEZE_LM = False
DATASET = "toxjson"


def tokenize(dataset, tokenizer, lm_version,):
    if args.use_synonyms:
        dataset = [replace_with_synonyms(text) for text in dataset]
    if lm_version == "roberta":
        tokenized_data = tokenizer(dataset, padding="max_length", truncation=True, max_length=512)
    else:
        tokenized_data = tokenizer(dataset, padding="max_length", truncation=True)
    
    if ADD_NOISE:
        tokenized_data = add_noise_to_embeddings(tokenized_data, tokenizer)
    
    return tokenized_data
def replace_with_synonyms(text):
    """
    Replace words in the text with their more common synonyms.

    Args:
        text (str): The input text.

    Returns:
        str: The text with words replaced by their common synonyms.
    """
    words = text.split()
    new_words = []

    for word in words:
        synonyms = wordnet.synsets(word)
        if synonyms:
            # Get the most common synonym (the lemma name of the first synonym)
            common_synonym = synonyms[0].lemmas()[0].name()
            new_words.append(common_synonym)
        else:
            new_words.append(word)
    
    return ' '.join(new_words)
def add_noise_to_embeddings(tokenized_data, tokenizer, noise_factor=0.1):
    """
    Adds Gaussian noise to token embeddings.
    
    Args:
        tokenized_data: The tokenized data.
        tokenizer: The tokenizer used for tokenizing the data.
        noise_factor: The standard deviation of the Gaussian noise to be added.
        
    Returns:
        The tokenized data with added noise.
    """
    # Convert input IDs to embeddings
    embeddings = tokenizer.convert_ids_to_tokens(tokenized_data["input_ids"])
    embeddings = np.array([tokenizer.encode(token, add_special_tokens=False) for token in embeddings])
    
    # Add Gaussian noise to embeddings
    noise = np.random.normal(0, noise_factor, embeddings.shape)
    noisy_embeddings = embeddings + noise
    
    # Convert embeddings back to input IDs
    tokenized_data["input_ids"] = [tokenizer.convert_tokens_to_ids(tokenizer.decode(embedding)) for embedding in noisy_embeddings]
    
    return tokenized_data

def make_disagg_labels_toxjson(dataset, model_type, survey_info_setting=None, use_lgbt=True,
                               tokenizer=AutoTokenizer.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier'),
                               prepend_other_ratings=False, ablations=""):
    dataset_out = []
    survey_cols = ["gender", "race", "technology_impact", "uses_media_social", "uses_media_news", "uses_media_video",
                   "uses_media_forums", "personally_seen_toxic_content", "personally_been_target",
                   "identify_as_transgender", "toxic_comments_problem", "education", "age_range", "lgbtq_status",
                   "political_affilation", "is_parent", "religion_important"]
    survey_iders = {}

    for row in dataset:
        for rating in row["ratings"]:
            survey_ider = "".join([str(rating[s]) for s in survey_cols])

            if survey_ider in survey_iders:
                survey_iders[survey_ider]["count"] += 1
            else:
                survey_iders[survey_ider] = {}
                survey_iders[survey_ider]["count"] = 1
                survey_iders[survey_ider]["features"] = {col: rating[col] for col in survey_cols}
                survey_iders[survey_ider]["replies"] = survey_ider

    n_ann = sorted(list(survey_iders.values()), reverse=True, key=lambda x: x["count"])
    print("# response ids:", len(n_ann))
    print("Sample demo info:", n_ann[0]["demographic_info"])
    print("Sample survey info:", n_ann[0]["survey_info"])

    id_dict = {}
    for i, ann in enumerate(n_ann):
        id_dict[ann["replies"]] = i
        n_ann[i][ann["replies"]] = i

    def truncated_text(input_text, tokenizer, max_length = 8192):
        tokens = tokenizer.tokenize(input_text)
        num_tokens = len(tokens)
        if num_tokens > max_length:
            truncated_tokens = tokens[-max_length:]
            truncated_text = tokenizer.convert_tokens_to_string(truncated_tokens)
            return truncated_text
        return input_text

    id_toxic_dict = {}
    for elem in dataset:
        comment = elem['comment']
        for rating in elem['ratings']:
            if rating['id'] not in id_toxic_dict:
                id_toxic_dict[rating["id"]] = dict()
                id_toxic_dict[rating["id"]][comment] = rating['toxic_score']
            else:
                id_toxic_dict[rating["id"]][comment] = rating['toxic_score']

    def return_rating(data_dict, text, id):
        data_dict[id].pop(text, None)
        result = ", ".join(f"\"{key}\" is rated {value}" for key, value in data_dict[id].items())
        return result

    for row in dataset_out:
        history_str = return_rating(id_toxic_dict, row["post"], row["ann_ids"])
        row["post"] = truncated_text(history_str + " [SEP] " + row["post"], tokenizer)

    print(dataset_out[0])
    return dataset_out, len(n_ann), id_dict


def compute_metrics_sbic(eval_pred, labels=None, predictions=None):
    if labels is None:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
    acc = load_metric("accuracy").compute(predictions=predictions, references=labels)
    f1 = load_metric("f1").compute(predictions=predictions, references=labels)
    prec = load_metric("precision").compute(predictions=predictions, references=labels)
    rec = load_metric("recall").compute(predictions=predictions, references=labels)
    print(acc, f1, prec, rec)
    logger.info("Metrics: {} {} {} {}".format(acc, f1, prec, rec))
    return {**acc, **f1, **prec, **rec}


class CustomTrainer(Trainer):
    def evaluate(self, eval_dataset: Optional[Dataset] = None, ignore_keys: Optional[List[str]] = None,
                 metric_key_prefix: str = "eval", return_preds=True):
        print("Evaluating results for epoch", self.state.epoch, "; num labels", NUM_LABELS)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if FREEZE_LM:
            if self.state.epoch == 2:
                print("Freezing LM")
                for name, param in self.model.roberta.named_parameters():
                    if "baseline_classifier" not in name and "annotator" not in name:
                        param.requires_grad = False

        acc = load_metric("accuracy")
        f1 = load_metric("f1")
        prec = load_metric("precision")
        rec = load_metric("recall")

        agg_mae = 0
        var_mae = 0
        disagg_mae = 0
        agg_denom = 0
        disagg_denom = 0
        self.model.eval()

        dataloader = self.get_eval_dataloader(eval_dataset)
        num_batches = len(dataloader)

        agg_preds = {}
        all_disagg_preds = []
        all_disagg_labels = []
        for batch_index, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
            logits = outputs.logits

            disagg_gt = batch["labels"]
            agg_gt = batch["agg_label"]
            disagg_predictions = logits.squeeze().flatten()
            all_disagg_preds += disagg_predictions
            all_disagg_labels += disagg_gt
            cur_disagg_mae = torch.abs(disagg_predictions - disagg_gt).sum()
            disagg_mae += cur_disagg_mae
            disagg_denom += len(disagg_predictions)

            for index_within_batch, utt_index in enumerate(batch["utterance_index"]):
                utt_index = utt_index.item()
                if utt_index in agg_preds:
                    agg_preds[utt_index]["disagg_pred"].append(disagg_predictions[index_within_batch])
                    agg_preds[utt_index]["var_gt"].append(disagg_gt[index_within_batch])
                else:
                    agg_preds[utt_index] = {}
                    agg_preds[utt_index]["disagg_pred"] = [disagg_predictions[index_within_batch]]
                    agg_preds[utt_index]["agg_gt"] = agg_gt[index_within_batch]
                    agg_preds[utt_index]["var_gt"] = [disagg_gt[index_within_batch]]

                var_gt = -1
                agg_gt = -1
                agg_predictions = -1

                if num_batches < 10 or batch_index % int(num_batches / 10) == 1:
                    print("\nBatch:", batch_index)
                    print("Predictions: agg", agg_predictions, "disagg", disagg_predictions, "var",
                          torch.var(disagg_predictions))
                    print("references: agg", agg_gt, "disagg", disagg_gt, "var", var_gt)

        if DATASET == "toxjson":
            agg_mae = 0
            var_mae = 0
            for pred in agg_preds.values():
                pred["disagg_pred"] = torch.stack(pred["disagg_pred"])
                pred["var_gt"] = torch.stack(pred["var_gt"]).to(dtype=torch.float)
                cur_agg_mae = torch.abs(torch.mean(pred["disagg_pred"]) - pred["agg_gt"])
                agg_mae += cur_agg_mae
                cur_var_mae = torch.abs(torch.var(pred["disagg_pred"]) - torch.var(pred["var_gt"]))
                var_mae += cur_var_mae

            agg_denom = len(agg_preds)
            print("raw disagg MAE:", disagg_mae, "raw agg MAE:", agg_mae, "raw var", var_mae, "disagg denom",
                  disagg_denom, "agg denom", agg_denom)
            disagg_mae = disagg_mae / disagg_denom
            print("[***] Evaluation: Epoch", self.state.epoch)
            print("[*] Disagg MAE:", disagg_mae)
            agg_mae = agg_mae / agg_denom
            print("[*] Agg MAE:", agg_mae)
            var_mae = var_mae / agg_denom
            print("[*] Var MAE:", var_mae)
        else:
            print("Print Metrics: {} {} {} {}".format(acc.compute(), f1.compute(), prec.compute(), rec.compute()))

        super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        if return_preds:
            print("all_disagg_preds", len(all_disagg_preds), "labels", len(all_disagg_labels))

            disagg_preds_dict = {"predictions": torch.stack(all_disagg_preds), "labels": torch.stack(all_disagg_labels)}
            pickle.dump(disagg_preds_dict, open("disagg_preds_dict.p", "wb"))
            return disagg_preds_dict


def map_demo(dataset):
    demo_info = []
    attn_mask = []
    demo_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    for k in dataset["demo_ids"]:
        tokens = tokenize(k, demo_tokenizer, args.lm_version)
        demo_info.append(tokens["input_ids"])
        attn_mask.append(tokens["attention_mask"])
    return {"demo_ids": demo_info, "demo_attention_mask": attn_mask}


def load_data(args):
    global ADD_NOISE
    ADD_NOISE = args.add_noise

    global USE_SYNONYMS 
    USE_SYNONYMS = args.use_synonyms 
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.lm_version == 'roberta':
        tokenizer = AutoTokenizer.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')
    else:
        tokenizer = GPT2TokenizerFast.from_pretrained(args.lm_version)

    use_lgbt = not args.no_lgbt
    FAST_HEAD_LR = args.fast_head_lr
    FREEZE_LM = args.freeze_lm
    DATASET = args.dataset
    NUM_LABELS = 24016
    splits = ["train", "dev", "test"]
    model_id_string = args.model_type + str(args.survey_info) + str(use_lgbt) + args.feature_ablation.replace(",", "_")
    eval_cap = 1000

    tokenizer.pad_token = tokenizer.eos_token
    test_size = 5000

    tokenized_inputs = {}
    if args.model_type == "multilabel" or args.model_type.startswith("multitask"):
        if args.cap:
            if args.dataset == 'toxjson':
                with open('data/toxicity_ratings.json') as f:
                    raw_data = json.load(f)["all"]
                train_0, test_0 = train_test_split(raw_data, test_size=10000, random_state=42)
                dev_0, test_0 = train_test_split(test_0, test_size=test_size, random_state=42)
                train_0, num_workers, ann_list = make_disagg_labels_toxjson(
                    train_0[:args.cap], model_type=args.model_type, survey_info_setting=args.survey_info,
                    use_lgbt=use_lgbt, prepend_other_ratings=args.prepend_other_ratings, ablations=args.feature_ablation)
                train_0 = pd.DataFrame(train_0[:args.cap])
            else:
                train_0, num_workers = make_disagg_labels_sbic(pd.read_csv("data/SBIC.v2.trn.csv")[:args.cap])
        else:
            if args.dataset == 'toxjson':
                if args.reload_dataset:
                    with open('data/toxicity_ratings.json') as f:
                        raw_data = json.load(f)["all"]
                    train_0, test_0 = train_test_split(raw_data, test_size=10000, random_state=42)
                    dev_0, test_0 = train_test_split(test_0, test_size=test_size, random_state=42)
                    train_0, num_workers, ann_dict = make_disagg_labels_toxjson(
                        train_0, model_type=args.model_type, survey_info_setting=args.survey_info,
                        use_lgbt=use_lgbt, prepend_other_ratings=args.prepend_other_ratings, ablations=args.feature_ablation)
                    train_0 = pd.DataFrame(train_0)
                else:
                    for split in splits:
                        tokenized_inputs[split] = load_from_disk(
                            "tokenized_toxjson/" + "multitask-demographic/" + split)
                    num_workers = 24016
            else:
                train_0, num_workers = make_disagg_labels_sbic(pd.read_csv("data/SBIC.v2.trn.csv"))

        if args.reload_dataset:
            raw_data_train = datasets.Dataset.from_pandas(train_0)
            print("Tokenizing...")
            tokenized_inputs["train"] = raw_data_train.map(
                lambda x: tokenize(x["post"], tokenizer, args.lm_version), batched=True)
            tokenized_inputs["train"] = tokenized_inputs["train"].map(map_demo, batched=True)

        print("Tokenized. Number of workers:", NUM_LABELS)
        if args.cap_eval:
            dev_0 = dev_0[:eval_cap]
            test_0 = test_0[:eval_cap]
        if args.reload_dataset:
            dev_0 = pd.DataFrame(make_disagg_labels_toxjson(
                dev_0, model_type=args.model_type, survey_info_setting=args.survey_info, use_lgbt=use_lgbt, prepend_other_ratings=args.prepend_other_ratings,
                ablations=args.feature_ablation)[0])
            test_0 = pd.DataFrame(make_disagg_labels_toxjson(
                test_0, model_type=args.model_type, survey_info_setting=args.survey_info, use_lgbt=use_lgbt, prepend_other_ratings=args.prepend_other_ratings,
                ablations=args.feature_ablation)[0])
            raw_data_dev = datasets.Dataset.from_pandas(dev_0)
            raw_data_test = datasets.Dataset.from_pandas(test_0)
            tokenized_inputs["dev"] = raw_data_dev.map(lambda x: tokenize(x["post"], tokenizer, args.lm_version),
                                                       batched=True)
            tokenized_inputs["test"] = raw_data_test.map(lambda x: tokenize(x["post"], tokenizer, args.lm_version),
                                                         batched=True)
            tokenized_inputs["dev"] = tokenized_inputs["dev"].map(map_demo, batched=True)
            tokenized_inputs["test"] = tokenized_inputs["test"].map(map_demo, batched=True)

    if args.dataset == "sbic":
        tokenized_inputs["dev"] = tokenized_inputs["dev"].add_column(
            "labels", [int(x > 0.5) for x in tokenized_inputs["dev"]["offensiveYN"]])
        tokenized_inputs["test"] = tokenized_inputs["test"].add_column(
            "labels", [int(x > 0.5) for x in tokenized_inputs["test"]["offensiveYN"]])
    if args.reload_dataset:
        if args.cap:
            for split in splits:
                tokenized_inputs[split].save_to_disk(
                    "tokenized_toxjson/" + model_id_string + "_cap_" + str(args.cap) + "/" + split)
        else:
            for split in splits:
                tokenized_inputs[split].save_to_disk("tokenized_toxjson/" + model_id_string + "/" + split)

    return tokenized_inputs, tokenizer, num_workers


def run_training(args, tokenized_inputs, tokenizer, num_workers, use_wandb=False):
    FAST_HEAD_LR = args.fast_head_lr
    FREEZE_LM = args.freeze_lm
    DATASET = args.dataset
    NUM_LABELS = 24016

    if args.seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if args.lm_version == 'roberta':
        model_config = AutoConfig.from_pretrained("roberta-base")
    else:
        model_config = GPT2Config()

    train_batch_size = args.train_batch_size
    lr = 5e-05
    shuffle_seed = 42
    if use_wandb:
        run = wandb.init()
        wandb.run.name = args.save_model_to[args.save_model_to.index("/") + 1]
        wandb.run.save()

        train_batch_size = run.config.sweep_batch_size
        lr = run.config.sweep_lr
        shuffle_seed = run.config.sweep_data_seed
        print("Sweep params:", train_batch_size, lr, run.config['sweep_data_seed'])
        args.save_model_to += str(train_batch_size) + "_" + str(lr) + "_" + str(shuffle_seed)

    training_args = TrainingArguments(
        args.save_model_to,
        evaluation_strategy="epoch",
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.n_epochs,
        learning_rate=lr,
        save_total_limit=5,
        save_strategy="epoch",
        logging_steps=500,
        report_to="none"
    )
    print("Training args:", training_args)

    if args.model_type == "multilabel" or args.model_type.startswith("multitask"):
        model_config.num_labels = 1
        if args.lm_version == "roberta":
            use_annot_module = (args.survey_info == "module")
            is_baseline = ("multitask" not in args.model_type)
            use_annotators = (args.model_type == "multitask-annotator" or args.model_type == "multitask")
            use_demographics = (args.model_type == "multitask-demographic" or args.model_type == "multitask")

        if args.from_saved_model and len(args.from_saved_model) > 0:
            print("Loading from saved model:", args.from_saved_model)
            if args.model_type == "multilabel":
                model = MultilabelModel(model_config, num_labels=num_workers).from_pretrained(
                    args.from_saved_model, num_labels=1)
            else:
                if args.lm_version == "roberta":
                    if args.recsys:
                        print("In RECSYS")
                        model = RobertaRecsSysMultitaskModel(
                            model_config, append_combo=args.append_combo, num_labels=1, is_baseline=is_baseline,
                            use_annotators=use_annotators, use_demographics=use_demographics,
                            use_var=args.use_var_objective, use_annot_module=use_annot_module).from_pretrained(
                            args.from_saved_model, is_baseline=is_baseline, use_annotators=use_annotators,
                            use_demographics=use_demographics, use_var=args.use_var_objective, num_labels=1,
                            ignore_mismatched_sizes=True, use_annot_module=use_annot_module)
                    else:
                        model = RobertaMultitaskModel(
                            model_config, num_labels=1, is_baseline=is_baseline, use_annotators=use_annotators,
                            use_demographics=use_demographics, use_var=args.use_var_objective,
                            use_annot_module=use_annot_module).from_pretrained(
                            args.from_saved_model, is_baseline=is_baseline, use_annotators=use_annotators,
                            use_demographics=use_demographics, use_var=args.use_var_objective, num_labels=1,
                            ignore_mismatched_sizes=True, use_annot_module=use_annot_module)
                else:
                    model = MultitaskModel(model_config, num_labels=num_workers).from_pretrained(
                        args.from_saved_model, num_labels=1)
        else:
            if args.model_type == "multilabel":
                model = MultilabelModel(model_config, num_labels=num_workers).from_pretrained(args.lm_version)
            else:
                if args.lm_version == "roberta":
                    if args.recsys:
                        print("In RECSYS")
                        model = RobertaRecsSysMultitaskModel(
                            model_config, append_combo=args.append_combo, num_labels=1, is_baseline=is_baseline,
                            use_annotators=use_annotators, use_demographics=use_demographics,
                            use_var=args.use_var_objective, use_annot_module=use_annot_module).from_pretrained(
                            'SkolkovoInstitute/roberta_toxicity_classifier', is_baseline=is_baseline,
                            use_annotators=use_annotators, use_demographics=use_demographics,
                            use_var=args.use_var_objective, use_annot_module=use_annot_module)
                    else:
                        model = RobertaMultitaskModel(
                            model_config, num_labels=1, is_baseline=is_baseline, use_annotators=use_annotators,
                            use_demographics=use_demographics, use_var=args.use_var_objective,
                            use_annot_module=use_annot_module).from_pretrained(
                            'SkolkovoInstitute/roberta_toxicity_classifier', is_baseline=is_baseline,
                            use_annotators=use_annotators, use_demographics=use_demographics,
                            use_var=args.use_var_objective, use_annot_module=use_annot_module)
                else:
                    model = MultitaskModel(model_config, num_labels=num_workers).from_pretrained(args.lm_version)

        model.config.pad_token_id = model.config.eos_token_id
        model.resize_token_embeddings(len(tokenizer))

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print("Using device", device, "; number of devices:", torch.cuda.device_count())
        model.to(device)

        tokenized_inputs["train"] = tokenized_inputs["train"].shuffle(seed=shuffle_seed)
        tokenized_inputs["dev"] = tokenized_inputs["dev"].shuffle(seed=shuffle_seed)

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_inputs["train"],
            eval_dataset=tokenized_inputs["dev"]
        )
        if not args.no_train:
            print("Beginning training...")
            trainer.train()
            model.save_pretrained(args.save_model_to + "/final")

        print("-- FINAL EVAL --")
        print(tokenized_inputs["dev"][0])
        if args.eval_on_test:
            return trainer.evaluate(tokenized_inputs["test"])
        return trainer.evaluate(tokenized_inputs["dev"])

    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--train_batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Fixed seed to train with')
    parser.add_argument('--save_model_to', type=str, default="saved_models", help='Output path for saved model')
    parser.add_argument('--reload_dataset', action='store_true', help='Whether to reload the toxjson dataset or use the saved tokenized version')
    parser.add_argument('--lr_scheduler', choices=['exp', 'lambda'], default='exp', help='Learning rate scheduler to use')
    parser.add_argument('--from_saved_model', type=str, help='Whether to evaluate from already-saved model')
    parser.add_argument('--no_train', action='store_true', help='Whether to train')
    parser.add_argument('--no_test', action='store_true', help='Whether to evaluate')
    parser.add_argument('--cap', type=int, default=None, help='Option to cap amount of training data')
    parser.add_argument('--model_type',
                        choices=["baseline", "disagg", "ensemble", "multitask", "multilabel", "ensemble-base",
                                 "multitask-base", "multitask-annotator", "multitask-demographic"],
                        default="baseline", help='Type of model to use')
    parser.add_argument('--survey_info', choices=["id-sep", "text-sep", "both-sep", "module", None],
                        default=None, help='How to incorporate survey info')
    parser.add_argument('--no_lgbt', action='store_true', help='Whether to avoid using LGBT status in demographic info')
    parser.add_argument('--dataset', choices=['sbic', 'toxjson'], default='toxjson')
    parser.add_argument('--lm_version', choices=['gpt2', 'gpt2-large', 'roberta'], default='roberta')
    parser.add_argument('--cap_eval', action='store_true', help='Whether to cap amount of eval data (for debugging purposes)')
    parser.add_argument('--fast_head_lr', action='store_true', help='Whether to use a faster learning rate for the head and categorical information')
    parser.add_argument('--freeze_lm', action='store_true', help='Whether to freeze the language model after the first 3 epochs')
    parser.add_argument("--use_var_objective", action='store_true', help='Whether to optimize for minimizing variance MAE')
    parser.add_argument('--project_name', type=str, default="sbic", help='Project name for W&B')
    parser.add_argument('--use_wandb', action='store_true', help="Whether to send results to W&B")
    parser.add_argument('--eval_on_test', action='store_true', help="Whether to evaluate on the test set instead of dev set.")
    parser.add_argument('--saved_predictions', type=str, default="disagg_preds_dict.p", help="Saved predictions for faster evaluation. Set to `none` to generate predictions again.")
    parser.add_argument('--feature_ablation', type=str, default="", help="Comma-separated list of demographic & survey feature names for ablation")
    parser.add_argument('--recsys', action='store_true', help="Whether to use the recsys model head")
    parser.add_argument('--append_combo', action='store_true', help="Whether to append the output of recsys [U,x] or take the dot product")
    parser.add_argument('--prepend_other_ratings', action='store_true', help="Whether to prepend other texts and their associated ratings")
    parser.add_argument('--use_synonyms', action='store_true', help='Whether to replace words with their more common synonyms')
    parser.add_argument('--add_noise', action='store_true', help='Whether to add noise to the embeddings')

    print("Begin trainer", flush=True)
    args = parser.parse_args()
    tokenized_inputs, tokenizer, num_workers = load_data(args)

    run_training(args, tokenized_inputs, tokenizer, num_workers)
