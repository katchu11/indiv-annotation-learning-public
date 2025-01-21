import argparse
import csv
from collections import OrderedDict
from functools import partial
import logging
import json
import os
from statistics import mean, variance
import sys
from typing import List, Optional

from matplotlib import pyplot as plt
import pandas as pd
import pickle
import datasets
from datasets import load_metric, load_from_disk
#import gensim.downloader as gensim_downloader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, f1_score
import torch
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer

from transformers import GPT2TokenizerFast, GPT2LMHeadModel, TrainingArguments, Trainer, GPT2Config, \
    GPT2Model, get_scheduler, modeling_outputs, AutoConfig, RobertaForSequenceClassification, \
    RobertaModel, TrainingArguments, Trainer, trainer_pt_utils, trainer_utils, training_args, trainer_callback
import wandb

from models import RobertaMultitaskModel, RobertaClassificationHead, RobertaRecsSysMultitaskModel, RobertaRecSysClassificationHead
from older_models import MultitaskModel, MultilabelModel
import target_model

logger = logging.getLogger('GPT2 log')
NUM_LABELS = 24016  # warning -- this may override values set in run_training when inside eval
FAST_HEAD_LR = False
FREEZE_LM = False
DATASET = "toxjson"


def tokenize(dataset, tokenizer, lm_version):
    """
    Tokenize the dataset.

    :param dataset: Data to tokenize
    :param tokenizer: Tokenizer for the dataset.
    :param lm_version: Whether to tokenize for Roberta or GPT-2 (different max lengths)
    :return: Tokenized Huggingface dataset
    """
    if lm_version == "roberta":
        return tokenizer(dataset, padding="max_length", truncation=True, max_length=512)
    return tokenizer(dataset, padding="max_length", truncation=True)


def make_disagg_labels_sbic(df):
    """
    Given annotations for each example from the SBIC dataset, add a column containing an array of labels for each
    annotator.

    :param df: Dataset containing annotations
    :return: Dataset with labels for each annotator added
    """
    textFields = ['targetMinority', 'targetCategory', 'targetStereotype']
    classFields = ['whoTarget', 'intentYN', 'sexYN', 'offensiveYN']

    worker_ids = df.WorkerId.unique()
    annot_dicts = {}

    # print("Num workers:", len(worker_ids), worker_ids)
    for row_index in range(len(df)):
        if df["post"][row_index] not in annot_dicts:
            annot_dicts[df["post"][row_index]] = np.empty(len(worker_ids))
            annot_dicts[df["post"][row_index]].fill(42)  # placeholder
        annot_dicts[df["post"][row_index]][np.where(worker_ids == df['WorkerId'][row_index])[0]] = df['offensiveYN'][
            row_index]

    aggDict = {c: lambda x: sorted(filter(lambda x: x, set(x)))
               for c in textFields}
    aggDict.update({c: lambda x: list(x) for c in classFields})
    df[textFields] = df[textFields].fillna("")
    gDf = df.groupby("post", as_index=False, group_keys=True).agg(aggDict)
    gDf["hasBiasedImplication"] = (gDf["targetStereotype"].apply(len) == 0).astype(int)
    gDf["labels"] = [annot_dicts[post] for post in gDf["post"]]

    gDf[textFields] = gDf[textFields].apply(lambda c: c.apply(json.dumps))
    return gDf, len(worker_ids)


def make_disagg_labels_toxjson(dataset, model_type, survey_info_setting=None, use_lgbt=True, tokenizer = AutoTokenizer.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier'), prepend_other_ratings = False, ablations=""):
    """
    Given annotations for each example from the SBIC dataset, add a column containing labels for each annotator.

    :param dataset: Dataset containing annotations
    :param model_type: Features to include in model. Choose from:
    :param survey_info_setting: Survey info to use in model.
    :param use_lgbt: Whether to include LGBT status in survey data.
    :param ablations: Comma-separated list of features to include in ablations.
    :return: Dataset with annotator labels added.
    """
    dataset_out = []
    # We don't have worker IDs so we need to infer them from survey responses
    survey_cols = ["gender", "race", "technology_impact", "uses_media_social", "uses_media_news", "uses_media_video",
                   "uses_media_forums", "personally_seen_toxic_content", "personally_been_target",
                   "identify_as_transgender", "toxic_comments_problem", "education", "age_range", "lgbtq_status",
                   "political_affilation", "is_parent", "religion_important"]
    survey_iders = {}
    race_dem = {}
    gender_dem = {}
    edu_dem = {}
    pol_dem = {}
    age_dem = {}
    rel_dem = {}

    # REMOVE
    axis_ratings = {col: {} for col in survey_cols}
    axis_counts = {col: {} for col in survey_cols}
    # END

    for row in dataset:
        for rating in row["ratings"]:
            survey_ider = "".join([str(rating[s]) for s in survey_cols])

            # REMOVE
            for attribute in survey_cols:
                if rating[attribute] not in axis_ratings[attribute]:
                    axis_ratings[attribute][rating[attribute]] = 0
                    axis_counts[attribute][rating[attribute]] = 0
                axis_ratings[attribute][rating[attribute]] += rating["toxic_score"]
                axis_counts[attribute][rating[attribute]] += 1

            # END

            if survey_ider in survey_iders:
                survey_iders[survey_ider]["count"] += 1
            else:
                survey_iders[survey_ider] = {}
                survey_iders[survey_ider]["count"] = 1
                survey_iders[survey_ider]["features"] = {col: rating[col] for col in survey_cols}
                survey_iders[survey_ider]["replies"] = survey_ider
                parent_status = "parent" if rating["is_parent"] is True else "not a parent"
                religion_status = "religion is " + str(rating["religion_important"]).lower()
                education = str(rating["education"])
                if education in edu_dem:
                    edu_dem[education] += 1
                else:
                    edu_dem[education] = 1
                race = str(rating["race"])
                if race in race_dem:
                    race_dem[race] += 1
                else:
                    race_dem[race] = 1
                gender = str(rating["gender"]).lower()
                if gender in gender_dem:
                    gender_dem[gender] += 1
                else:
                    gender_dem[gender] = 1
                age = str(rating["age_range"]).lower()
                if age in age_dem:
                    age_dem[age] += 1
                else:
                    age_dem[age] = 1
                pol = str(rating["political_affilation"]).lower()
                if pol in pol_dem:
                    pol_dem[pol] += 1
                else:
                    pol_dem[pol] = 1
                rel = str(rating["religion_important"]).lower()
                if rel in rel_dem:
                    rel_dem[rel] += 1
                else:
                    rel_dem[rel] = 1
                if education.startswith("High school graduate"):
                    education = "High school graduate"
                elif education.startswith("Bachelor"):
                    education = "Bachelor\'s degree"
                elif education.startswith("Associate"):
                    education = "Associate degree"

                demo_dict = OrderedDict()
                demo_dict["starter_1"] = "is"
                demo_dict["starter_2"] = "a"
                demo_dict["age_range"] = str(rating["age_range"]) + " year old"
                demo_dict["race"] = race
                demo_dict["gender"] = gender
                demo_dict["joiner"] = "who"
                demo_dict["education"] = "has a " + education + ","
                demo_dict["political_affilation"] = "is politically " + pol + ","  # (JSON label has the misspelling)
                demo_dict["is_parent"] = "is " + parent_status + ","
                demo_dict["religion_important"] = "and thinks " + religion_status

                if ablations == "":
                    demo_list = [v for v in demo_dict.values()
                                 if "prefer not to say" not in v.lower() and "unknown" not in v.lower()]
                else:
                    use_starter_1 = "age_range" in ablations or "race" in ablations or "gender" in ablations
                    use_starter_2 = "age_range" in ablations or "gender" in ablations
                    use_joiner = use_starter_1 and any(x in ablations for x in ["education", "political_affilation",
                                                                              "is_parent", "religion_important"])
                    demo_list = [v for k, v in demo_dict.items()
                                 if "prefer not to say" not in v.lower() and "unknown" not in v.lower()
                                 and ((k == "starter_1" and use_starter_1) or (k == "starter_1" and use_starter_2)
                                      or (k == "joiner" and use_joiner) or k in ablations)]

                survey_iders[survey_ider]["demographic_info"] = ""
                if len(demo_list) > 1:
                    survey_iders[survey_ider]["demographic_info"] = "The reader " + " ".join(demo_list) + "."

                soc_media_str = ""
                soc_media_str_no = ""
                sites = {"uses_media_social": "social media", "uses_media_news": "news sites",
                         "uses_media_video": "video sites",
                         "uses_media_forums": "web forums"}  # leave out unused sites?
                for site, site_str in sites.items():
                    if rating[site]:
                        soc_media_str += site_str + ", "
                    else:
                        soc_media_str_no += site_str + ", "

                soc_media_str = soc_media_str[:-2]
                soc_media_str_no = soc_media_str_no[:-2]
                last_sm_comma = soc_media_str.rfind(",")
                if last_sm_comma >= 0:
                    soc_media_str = soc_media_str[:last_sm_comma + 1] + " and" + soc_media_str[last_sm_comma + 1:]
                last_sm_no_comma = soc_media_str_no.rfind(",")
                if last_sm_no_comma >= 0:
                    soc_media_str_no = soc_media_str_no[:last_sm_no_comma + 1] + " or" + soc_media_str_no[
                                                                                         last_sm_no_comma + 1:]
                if len(soc_media_str) > 4:
                    soc_media_str = "uses " + soc_media_str
                if len(soc_media_str_no) > 4:
                    soc_media_str_no = " but does not use " + soc_media_str_no

                soc_media_str += soc_media_str_no

                seen_tox_str = "has never seen toxic comments,"
                if rating["personally_seen_toxic_content"]:
                    seen_tox_str = "has seen toxic comments,"
                was_target_str = "has never been personally targeted by toxic comments,"
                if rating["personally_seen_toxic_content"]:
                    was_target_str = "has been personally targeted by toxic comments,"
                impact_str = "thinks technology has a " + rating[
                    "technology_impact"].lower() + " impact on people's lives,"
                tox_prob_str = "and thinks toxic comments are " + rating["toxic_comments_problem"].lower() + "."
                trans_dict = {"Yes": "transgender", "No": "cisgender", "Prefer not to say": "", "nan": "", "Other": ""}
                lgbt_dict = {"Homosexual": "gay and ", "Heterosexual": "straight and ", "Bisexual": "bisexual and ",
                             "Prefer not to say": "", "nan": "", "Other": ""}
                lgbt_str = "The reader is " + lgbt_dict[str(rating["lgbtq_status"])] \
                           + trans_dict[str(rating["identify_as_transgender"])] + "."
                survey_dict = OrderedDict()
                survey_dict["all_social_media"] = soc_media_str
                survey_dict["joiner"] = ". The reader"
                survey_dict["personally_seen_toxic_content"] = seen_tox_str
                survey_dict["personally_been_target"] = was_target_str
                survey_dict["technology_impact"] = impact_str
                survey_dict["toxic_comments_problem"] = tox_prob_str

                if use_lgbt:
                    survey_dict["lgbtq_status"] = lgbt_str

                if ablations == "":
                    survey_list = [v for v in survey_dict.values() if
                                   "prefer not to say" not in v.lower() and "unknown" not in v.lower()]
                else:
                    use_joiner = ("all_social_media" in ablations and any(
                        x in ablations for x in ["personally_seen_toxic_content", "personally_been_target",
                                                 "technology_impact", "toxic_comments_problem"]))
                    survey_list = [v for k, v in survey_dict.items() if
                                   "prefer not to say" not in v.lower() and "unknown" not in v.lower()
                                   and ((k == "joiner") and use_joiner) or k in ablations]

                survey_iders[survey_ider]["survey_info"] = ""
                if len(survey_list) > 0:
                    survey_iders[survey_ider]["survey_info"] = "The reader " + " ".join(survey_list)

    n_ann = sorted(list(survey_iders.values()), reverse=True, key=lambda x: x["count"])
    print("# response ids:", len(n_ann))
    print("Sample demo info:", n_ann[0]["demographic_info"])
    print("Sample survey info:", n_ann[0]["survey_info"])

    # Assign each of the responses an id
    id_dict = {}
    for i, ann in enumerate(n_ann):
        id_dict[ann["replies"]] = i
        n_ann[i][ann["replies"]] = i
    
   


    for row_index, row in enumerate(dataset):
        agg_rating = sum([rating["toxic_score"] for rating in row["ratings"]]) / len(row["ratings"])
        for rating_index, rating in enumerate(row["ratings"]):
            cur_row = {}

            rating["id"] = id_dict["".join([str(rating[s]) for s in survey_cols])]
            demo_info = n_ann[rating["id"]]["demographic_info"]
            survey_info = n_ann[rating["id"]]["survey_info"]

            cur_row['source'] = row["source"]
            cur_row['perspective_score'] = row['perspective_score']
            cur_row['post_id'] = row['comment_id']
            cur_row["labels"] = rating["toxic_score"]
            cur_row["agg_label"] = agg_rating

            if model_type == "multitask-base":
                cur_row["post"] = row["comment"]
            elif model_type == "multitask-demographic":
                if use_lgbt:
                    cur_row["post"] = demo_info + lgbt_str + " [SEP] " + row["comment"]
                else:
                    cur_row["post"] = demo_info + " [SEP] " + row["comment"]
            elif model_type == "multitask" or model_type == "multitask-annotator":
                post_info = ""
                if survey_info_setting == "text-sep":
                    post_info += survey_info
                elif survey_info_setting == "id-sep":
                    if use_lgbt:
                        post_info += lgbt_str
                    post_info += str(rating["id"])
                elif survey_info_setting == "both-sep":
                    post_info += str(rating["id"]) + " [SEP] " + survey_info
                if len(post_info) > 0:
                    post_info += " [SEP] "
                if model_type == "multitask":
                    post_info += demo_info + " [SEP] "
                post_info += row["comment"]
                cur_row['post'] = post_info

            cur_row["demo_ids"] = demo_info
            cur_row["ann_ids"] = rating["id"]
            cur_row["utterance_index"] = row_index
            dataset_out.append(cur_row)

    def truncated_text(input_text, tokenizer, max_length):
        # Tokenize input text once
        tokens = tokenizer.tokenize(input_text)
        num_tokens = len(tokens)
        if num_tokens > max_length:
            # Calculate how many tokens to keep
            truncated_tokens = tokens[-max_length:]
            # Convert truncated tokens back to string
            truncated_text = tokenizer.convert_tokens_to_string(truncated_tokens)
            return truncated_text
        return input_text
    
    id_toxic_dict = {}
    for elem in dataset:
        comment = elem['comment']
        for rating in elem['ratings']:
            print(rating)
            if(rating['id'] not in id_toxic_dict):
                id_toxic_dict[rating["id"]] = dict()
                id_toxic_dict[rating["id"]][comment]= rating['toxic_score']
            else:
                id_toxic_dict[rating["id"]][comment]= rating['toxic_score']
    def return_rating(data_dict, text, id):
        # remove the indicated text from data
        data_dict[id].pop(text, None)
        # generate the resulting string
        result = ", ".join(f"\"{key}\" is rated {value}" for key, value in data_dict[id].items())
        return result
    
    for row in dataset_out:
        history_str = return_rating(id_toxic_dict, row["post"], row["ann_ids"])
        row["post"] = truncated_text(history_str + " [SEP] " + row["post"], tokenizer, 512)

    print(dataset_out[0])
    return dataset_out, len(n_ann), id_dict


def compute_metrics_sbic(eval_pred, labels=None, predictions=None):
    """
    Compute metrics on the SBIC data (not in use).

    :param eval_pred: (logits, labels) if labels aren't provided separately
    :param labels: Dataset labels
    :param predictions: Dataset predictions
    :return: Dictionary of accuracy, F1, precision, and recall
    """
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


def get_text_target(text):
    """
    Map a piece of text to the (standardized) demographics group(s) it targets.

    :param text: Text with (non-standardized) targets
    :return: Set of targets, from targets_map if any matches are found or the individual words of input text if not.
    """
    targets_map = {"lgbtq+ people": "gender & sexuality: queer",
                   "lgbtq": "gender & sexuality: queer",
                   "lgbt": "gender & sexuality: queer",
                   "lgbt people": "gender & sexuality: queer",
                   "lgbtq people": "gender & sexuality: queer",
                   "lgbqt": "gender & sexuality: queer",
                   "lgbtq+": "gender & sexuality: queer",
                   "gay and trans people": "gender & sexuality: queer",
                   "queer": "gender & sexuality: queer",
                   "queers": "gender & sexuality: queer",
                   "lgbtqia people": "gender & sexuality: queer",
                   "lbgqt": "gender & sexuality: queer",
                   "queer people": "gender & sexuality: queer",
                   "gay people": "gender & sexuality: gay",
                   "bisexuals": "sexuality: bisexual",
                   "gays": "sexuality: gay",
                   "gay people": "sexuality: gay",
                   "gay": "sexuality: gay",
                   "homosexuals": "sexuality: gay",
                   "homosexual men": "gender & sexuality: gay men",
                   "homosexual people": "sexuality: gay",
                   "gay peolpe": "sexuality: gay",
                   "homosexual": "sexuality: gay",
                   "lesbian": "sexuality: lesbian",
                   "lesbian women": "sexuality: lesbian",
                   "female": "gender: women",
                   "a woman": "gender: women",
                   "woman": "gender: women",
                   "women": "gender: women",
                   "trans people": "gender: trans",
                   "wives": "gender & other: married women",
                   "young women": "age & gender: young women",
                   "bitch": "gender: women",
                   "femaloids": "gender: women",
                   "femoids": "gender: women",
                   "females": "gender: women",
                   "women bitch": "gender: women",
                   "girl": "age & gender: girls",
                   "a girl": "age & gender: girls",
                   "bitches": "gender: women",
                   "feminist": "other: feminists",
                   "minorities": "race: people of color",
                   "transgender people": "gender: trans",
                   "transwomen": "gender: trans women",
                   "transgenders": "gender: trans",
                   "transgender": "gender: trans",
                   "trans": "gender: trans",
                   "trans sexual men": "gender: trans men",
                   "trans sexual women": "gender: trans women",
                   "transsexuals": "gender: trans",
                   "trans men": "gender: trans men",
                   "trans women": "gender: trans women",
                   "transwoman": "gender: trans",
                   "people who transition": "gender: trans people who transition",
                   "religious people": "religion: religious",
                   "gay men": "gender & sexuality: gay men",
                   "homesexual people": "sexuality: gay",
                   "transgender women": "gender: trans",
                   "lesbians": "sexuality: lesbian",
                   "non-christians": "religion: non-christian",
                   "lgbt peopole": "gender & sexuality: queer",
                   "lgbtq community": "gender & sexuality: queer",
                   "blacks": "race: black",
                   "Male (Wrestler)": "gender: men",
                   "activists": "other: activists",
                   "feminists": "other: feminists",
                   "black woman": "gender & race: black women",
                   "black man": "gender & race: black men",
                   "black women": "gender & race: black women",
                   "black men": "gender & race: black men",
                   "involuntarily celibate men": "other: incels",
                   "man": "gender: men",
                   "men": "gender: men",
                   "asian people": "race: asian",
                   "black people": "race: black",
                   "ugly women": "appearance & gender: unattractive women",
                   "ugly men": "appearance & gender: unattractive men",
                   "unattractive women": "appearance & gender: unattractive women",
                   "unattractive men": "appearance & gender: unattractive men",
                   "attractive women": "appearance & gender: attractive women",
                   "attractive men": "appearance & gender: attractive men",
                   "unattractive people": "appearance: unattractive",
                   "attractive people": "appearance: attractive",
                   "fat women": "appearance & gender: overweight women",
                   "fat men": "appearance & gender: overweight men",
                   "overweight women": "appearance & gender: overweight women",
                   "overweight men": "appearance & gender: overweight men",
                   "fat people": "appearance: overweight",
                   "overweight people": "appearance: overweight",
                   "the human speaker": "other: human speaker",
                   "incels": "other: incels",
                   "mentally disabled people": "medical: mental disability",
                   "disabled people": "medical: disability",
                   "handicap people": "medical: disability",
                   "handicapped people": "medical: disability",
                   "disabled women": "gender & medical: women with disabilities",
                   "mentally ill people": "medical: mental illness",
                   "cis people": "gender: cis",
                   "people with autism": "medical: autism",
                   "poor people": "other: low-income",
                   "single mothers": "gender & other: single mothers",
                   "mothers": "gender & other: mothers",
                   "prisoners": "other: incarcerated",
                   "married people": "other: married",
                   "professors": "profession: professors",
                   "gay women": "sexuality: lesbian",
                   "girls": "age & gender: girls",
                   "boys": "age & gender: girls",
                   "children": "age: children",
                   "teenagers": "age: teens",
                   "young girls": "age & gender: girls",
                   "female children": "age & gender: girls",
                   "boys": "age & gender: boys",
                   "children": "age: children",
                   "males": "gender: men",
                   "prostitutes": "profession: sex workers",
                   "sex workers": "profession: sex workers",
                   "transgender individuals": "gender: trans",
                   "incels girls": "age & gender & other: incel girls",
                   "gey men": "gender & sexuality: gay men",
                   "liberal people": "politics: liberals",
                   "non-binary individuals": "gender: nonbinary",
                   "trans person": "gender: trans",
                   "white": "race: white",
                   "whoever you is": "other: unknown addressee",
                   "she": "gender: women",
                   "transsexuals": "gender: trans",
                   "gay or trans people": "gender & sexuality: queer",
                   "transpeople": "gender: trans",
                   "incel": "other: incels",
                   "male": "gender: men",
                   "wife": "gender & other: married women",
                   "lesbianism": "gender & sexuality: lesbian",
                   "kids": "age: children",
                   "some woman": "gender: women",
                   "women with children": "gender & other: mothers",
                   "mothers": "gender & other: mothers",
                   "women`": "gender: women",
                   "trans sexual people": "gender: trans",
                   "straight women": "gender & sexuality: straight women",
                   "bisexual people": "sexuality: bisexual",
                   "bisexuals": "sexuality: bisexual",
                   "queer people": "gender & sexuality: queer",
                   "queer women": "gender & sexuality: queer women",
                   "gay and trans": "gender & sexuality: queer",
                   "men": "gender: men",
                   "women with children": "gender & other: mothers",
                   "transgender men": "gender: trans men",
                   "transgender women": "gender: trans women",
                   "transgender people who transition": "gender: trans people who transition",
                   "trasngender people": "gender: trans",
                   "gay couples": "sexuality & other: queer couples",
                   "same sex couples": "sexuality & other: queer couples",
                   "unknown - maybe the person they're talking to? Maybe member of a different group?": "other: unknown",
                   "married women": "gender & other: married women",
                   "homosexual couples": "sexuality: gay",
                   "children of lesbian couples": "other: children of lesbian couples",
                   "bisexual": "sexuality: bisexual",
                   "LGBT+ community": "gender & sexuality: queer",
                   "she": "gender: women",
                   "lesbian": "sexuality: lesbian",
                   "bulemic": "other: eating disorder",
                   "french": "nationality: french",
                   "igb": "gender & sexuality: queer",
                   "self": "other: self",
                   "jewish people": "religion: jewish",
                   "jews": "religion: jewish",
                   "muslims": "religion: muslim",
                   "christians": "religion: christian",
                   "the original human speaker": "other: human speaker",
                   "homophobic people": "attitude: homophobia",
                   "csa survivors": "other: child sexual abuse victims",
                   "rape victims": "other: sexual abuse or assault victims",
                   "assault victims": "other: sexual abuse or assault victims",
                   "abuse victims": "other: abuse victims",
                   "jewish": "religion: jewish",
                   "pedophiles": "other: pedophiles",
                   "conservatives": "politics: conservative",
                   "liberals": "politics: liberal",
                   "democrats": "politics: democratic",
                   "republicans": "politics: republican",
                   "one woman (the bad mother)": "gender & other: mothers",
                   "same sex marriage": "sexuality: gay",
                   "female students": "gender: women",
                   "women in my family": "gender: women",
                   "white people": "race: white",
                   "white women": "gender & race: white women",
                   "white men": "gender & race: white men",
                   "all non-heteronormative people": "gender & sexuality: queer",
                   "women (trans and cis)": "gender: women",
                   "alcoholics": "medical: alcoholism",
                   "anxious people": "medical: anxiety",
                   "lawyers": "profession: lawyers",
                   "female lawyers": "gender & other: women lawyers",
                   "people who practice incest": "other: incest",
                   '"bitches" (women?)': "gender: women",
                   "mormons": "religion: mormon",
                   "sex worker": "other: sex workers",
                   "pro choicers": "other: pro-choice supporters",
                   "LGBT community": "gender & sexuality: queer",
                   "catholics": "religion: catholic",
                   "sexually active people": "other: sexually active",
                   "non-binary people": "gender: nonbinary",
                   "hipsters": "other: hipsters",
                   "narrator's wife": "gender: women",
                   "narrator's daughter": "gender: women",
                   "pedophiles": "other: pedophiles",
                   "prisoners": "other: incarcerated",
                   "japanese": "nationality: japanese",
                   "japanese people": "nationality: japanese",
                   "chinese": "nationality: chinese",
                   "chinese people": "nationality: chinese",
                   "mexican": "nationality: mexican",
                   "mexican people": "nationality: mexican",
                   "syrian": "nationality: syrian",
                   "europeans": "region: european",
                   "africans": "region: african",
                   "Vanessa": "gender: women",
                   "lesbian people": "sexuality: lesbian",
                   "transgender ftm people": "gender: trans men",
                   "transgender mtf people": "gender: trans women",
                   "cisgender people": "gender: cis people",
                   "religion": "religion: religious",
                   "women in science field": "gender & other: women in stem",
                   "queers": "gender & sexuality: queer",
                   "american women": "gender & nationality: american women",
                   "gay peoplw": "sexuality: gay",
                   "asexual people": "sexuality: asexual",
                   "women in science": "gender & profession: women in stem",
                   "straight people": "sexuality: straight",
                   "transgender woman": "gender: trans women",
                   "gay community": "sexuality: gay",
                   "gay couples": "sexuality: gay",
                   "same-sex couples": "sexuality: gay",
                   "lesbian couples": "sexuality: lesbian",
                   "lesiban": "sexuality: lesbian",
                   "scientists": "profession: scientists",
                   "autistic people": "medical: autism",
                   "heterosexual people": "sexuality: straight",
                   "children in same sex households": "other: children of queer couples",
                   "christian people": "religion: christian",
                   "politicians": "profession: politicians",
                   "misogynistic people": "attitude: sexist",
                   "misogynists": "attitude: sexist",
                   "homophobic people": "attitude: homophobia",
                   "racists": "attitude: racism",
                   "racist": "attitude: racism",
                   "white woman": "gender & race: white women",
                   "celibate men": "gender & other: celibate men",
                   "fehemorrhoids": "gender: women",
                   "trans community": "gender: trans",
                   "parents": "other: parents",
                   "genderqueer people": "gender: genderqueer",
                   "LGBTQ+ peoplee": "gender & sexuality: queer",
                   "the left": "politics: liberal",
                   "minority groups": "race: people of color",
                   "people of color": "race: people of color",
                   "LGBTQ+ community": "gender & sexuality: queer",
                   "self": "other: self",
                   "tansgender people": "gender: trans",
                   "gay poeple": "sexuality: gay",
                   "gender nonconforming people": "gender: gender nonconforming",
                   "gay couple": "sexuality & other: gay couples",
                   "gay peoples": "sexuality: gay",
                   "the human input": "other: human speaker",
                   "human input": "other: human speaker",
                   "deaf people": "medical: deafness",
                   "jewish women": "gender & religion: jewish women",
                   "necrophiliacs": "nonprotected: necrophiliacs",
                   "hoes": "gender: women",
                   "femoid": "gender: women",
                   "female athletes": "women athletes",
                   "gay black people": "race & sexuality: gay black people",
                   "russians": "nationality: russian",
                   "asian women": "gender & race: asian women",
                   "furries": "other: furries",
                   "man gay": "gender & sexuality: gay men",
                   "promiscuous men": "gender & other: sexually active men",
                   "promiscuous women": "gender & other: sexually active women",
                   "non virgin women": "gender & other: sexually active women",
                   "virgins": "other: virgin",
                   "virgin men": "gender & other: virgin men",
                   "male virgins": "gender & other: virgin men",
                   "sexy women": "appearance: attractive women",
                   "gay males": "gender & sexuality: gay men",
                   "trans peopel": "gender: trans",
                   "lesbias": "sexuality: lesbian",
                   "wmoen": "gender: women",
                   "incels people": "other: incels",
                   "sexist people": "attitude: sexism",
                   "unknown": "other: unclear",
                   "black": "race: black",
                   "cis men": "gender: cis men",
                   "straight men": "gender & sexuality: straight men",
                   "transexuals": "gender: trans",
                   "feminism": "other: feminists",
                   "black guy": "gender & race: black men",
                   "black guys": "gender & race: black men",
                   "nazi people": "nonprotected: nazis",
                   "bad bitches": "gender: women",
                   "stacy": "gender: women",
                   "mother": "gender & other: mothers",
                   "women drivers": "gender & other: women drivers",
                   "incel men": "gender & other: incel men",
                   "ladies": "gender: women",
                   "girlfriend": "gender: women",
                   "non binary": "gender: nonbinary",
                   "jamaicans": "nationality: jamaican",
                   "redheads": "appearance: redheaded",
                   "all women": "gender: women",
                   "women comedians": "gender & profession: women comedians",
                   "female comedians": "gender & profession: women comedians",
                   "black girls": "age & gender & race: black girls",
                   "hookers": "profession: sex workers",
                   "bitch's": "gender: women",
                   "muslim women": "gender & religion: muslim women",
                   "socialists": "politics: socialist",
                   "the poor": "other: low-income",
                   "older women": "age & gender: older women",
                   "asexuals": "sexuality: asexual",
                   "asexual men": "gender & sexuality: asexual men",
                   "women athletes": "gender & profession: women athletes",
                   "non-black men": "gender & race: non-black men",
                   "mexicans": "nationality: mexican",
                   "jewish girls": "age & gender & religion: jewish girls",
                   "whites": "race: white",
                   "roastie": "gender & other: sexually active women",
                   "hooker": "profession: sex workers",
                   "babies": "age: infants",
                   "african nationals": "region: african",
                   "mass-shooting victims": "other: mass shooting victims",
                   "southerners": "region: southern american",
                   "rural residents": "region: rural",
                   "womoen": "gender: women",
                   "femminists": "gender: women",
                   "his girlfriend": "gender: women",
                   "my girlfriend": "gender: women",
                   "the disabled": "other: disabled",
                   "gay persons": "sexuality: gay",
                   "anyone not white": "race: people of color",
                   "women bankers": "gender & profession: women bankers",
                   "non-white men": "gender & race: men of color",
                   "white females": "gender & race: white women",
                   "western women": "gender & race: white women",
                   "mixed race humans": "race: multiracial",
                   "terrorists": "other: terrorists",
                   "terrorist": "other: terrorists",
                   "transgender females": "gender: trans women",
                   "reddit users": "other: reddit users",
                   "french gay people": "nationality & sexuality: french gay people",
                   "whoever the human speaker is": "other: human speaker",
                   "women in the military": "gender & other: women in the military",
                   "person typing": "other: human speaker",
                   "kidnapping victims": "other: kidnapping victims",
                   "hispanics": "race: hispanic",
                   "mexican women": "gender & nationality: mexican women",
                   "gynecologists": "profession: gynecologists",
                   "russian citizens": "nationality: russian",
                   "virgin girls": "age & gender & other: virgin girls",
                   "frenchmen": "gender & nationality: french men",
                   "daughter": "age & gender: girls",
                   "girl": "age & gender: girls",
                   "black baby": "age & race: black children",
                   "sexism": "attitude: sexism",
                   "involuntary celibate men": "gender & other: incel men",
                   "young boys": "age & gender: boys",
                   "little girls": "age & gender: girls",
                   "childrens": "age: children",
                   "women politicians": "gender & profession: women politicians",
                   "latino/latina folks": "race: hispanic",
                   "immigrants": "other: immigrants",
                   "mentally disabled folks": "medical: mental disability",
                   "disabled": "medical: disability",
                   "arabic": "region: middle eastern",
                   "mental": "medical: mental disability",
                   "color": "race: people of color",
                   "latino/latina": "race: hispanic",
                   "asian": "race: asian"
                   }

    targets = set()
    for word in text.replace(",", " ").split():
        if word in targets_map:
            targets.add(targets_map[word])
        if word[:-1] in targets_map:
            targets.add(targets_map[word[:-1]])
        if "folks" in word and word[:word.index("folks") - 1] in targets_map:
            targets.add(targets_map[word[:word.index("folks") - 1]])
    if len(targets) > 0:
        return targets

    text = [t.strip().rstrip("s") for t in text.split()]
    targets.update(text)
    return targets


class CustomTrainer(Trainer):
    def evaluate(self,
                 eval_dataset: Optional[Dataset] = None,
                 ignore_keys: Optional[List[str]] = None,
                 metric_key_prefix: str = "eval",
                 return_preds=True):
        """
        Evaluate the annotator rating model.

        :param eval_dataset:
        :param ignore_keys:
        :param metric_key_prefix:
        :param return_preds:
        :return:
        """

        print("Evaluating results for epoch", self.state.epoch, "; num labels", NUM_LABELS)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Freeze LM head after 2nd epoch (not particularly helpful, so not in use)
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
            batch = {k: v.to(device) for k, v in batch.items()}  # why 50?
            with torch.no_grad():
                outputs = self.model(**batch)
            logits = outputs.logits

            if DATASET == "sbic":
                predictions = torch.sigmoid(logits).detach().cpu().numpy()  # why?
                predictions = [1.0 if x > 0.5 else 0.0 for x in predictions]
                acc.add_batch(predictions=predictions, references=batch["labels"])
                f1.add_batch(predictions=predictions, references=batch["labels"])
                prec.add_batch(predictions=predictions, references=batch["labels"])
                rec.add_batch(predictions=predictions, references=batch["labels"])
                # print("Print Metrics: {} {} {} {}".format(acc.compute(), f1.compute(), prec.compute(), rec.compute()))
            else:
                disagg_gt = batch["labels"]
                agg_gt = batch["agg_label"]

                # This is for the old baseline--try without
                # if args.model_type == "multitask-base":
                #     batch_size = batch["label_mask"].shape[0]
                #     disagg_gt = torch.masked_select(batch["labels"], batch["label_mask"].to(dtype=torch.bool)) # double-check this
                #     var_gt = torch.var(disagg_gt)
                #     base_agg_predictions = logits
                #     base_disagg_predictions = torch.Tensor(
                #         [[base_agg_predictions[i].item()] * batch["label_mask"][i].count_nonzero() for i in
                #          range(batch_size)]).to(device).flatten()
                #
                #     # Calculate individual MAE
                #     cur_disagg_mae = torch.abs(base_disagg_predictions - disagg_gt).sum()
                #     disagg_mae += cur_disagg_mae
                #     disagg_denom += len(base_disagg_predictions)
                #
                #     cur_agg_mae = torch.abs(base_agg_predictions.flatten() - agg_gt).sum()
                #     agg_mae += cur_agg_mae
                #     agg_denom += len(base_agg_predictions)
                #
                #     cur_var_mae = torch.abs(torch.var(base_disagg_predictions) - var_gt).sum()
                #     var_mae += cur_var_mae
                #     print("Predictions: agg", base_agg_predictions, "disagg", base_disagg_predictions, "var",
                #           torch.var(base_disagg_predictions))
                #     print("references: agg", agg_gt, "disagg", disagg_gt, "var", var_gt)
                # else:
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

                var_gt = -1  # just to know these aren't valid
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
                cur_agg_mae = torch.abs(torch.mean(pred["disagg_pred"]) - pred["agg_gt"])  # ojo
                agg_mae += cur_agg_mae
                cur_var_mae = torch.abs(torch.var(pred["disagg_pred"]) - torch.var(pred["var_gt"]))
                var_mae += cur_var_mae
                # print("disagg pred", pred["disagg_pred"], "var gt", pred["var_gt"])
                # print("cur agg mae", cur_agg_mae, "cur var mae", cur_var_mae, "pred", pred)

            # if args.model_type == "multitask-base":
            #     agg_denom = len(base_agg_predictions)
            # else:
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
            # logger.info("Metrics: {} {} {} {}".format(acc.compute(), f1.compute(), prec.compute(), rec.compute()))
            print("Print Metrics: {} {} {} {}".format(acc.compute(), f1.compute(), prec.compute(), rec.compute()))

        super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        # wandb.log({
        #     'disagg_mae': disagg_mae,
        #     'agg_mae': agg_mae,
        #     'var_mae': var_mae
        # })

        if return_preds:
            print("all_disagg_preds", len(all_disagg_preds), "labels", len(all_disagg_labels))

            disagg_preds_dict = {"predictions": torch.stack(all_disagg_preds), "labels": torch.stack(all_disagg_labels)}
            pickle.dump(disagg_preds_dict, open("disagg_preds_dict.p", "wb"))
            return disagg_preds_dict


def map_demo(dataset):
    """
    Tokenizes the demographic information.

    :param dataset: Dataset containing demographic information to tokenize.
    :return: dict with format {"demo_ids": demo_input_ids, "demo_attention_mask": demo_attn_mask}
    """

    demo_info = []
    attn_mask = []
    demo_tokenizer = AutoTokenizer.from_pretrained('roberta-base')  # smallbenchnlp/roberta-small
    for k in dataset["demo_ids"]:
        tokens = tokenize(k, demo_tokenizer, args.lm_version)
        demo_info.append(tokens["input_ids"])
        attn_mask.append(tokens["attention_mask"])
    return {"demo_ids": demo_info, "demo_attention_mask": attn_mask}


def load_data(args):
    """
    Load data from toxJSON or (no longer in use) Social Bias Frames datasets.

    :param args: Model training args.
    :return: Tokenized Huggingface dataset with train/val/test splits.
    """

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.lm_version == 'roberta':
        tokenizer = AutoTokenizer.from_pretrained(
            'SkolkovoInstitute/roberta_toxicity_classifier')  # roberta-base
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

    # Load Social Bias Frames data (no longer in use)
    if args.model_type == "baseline" or args.model_type == "ensemble-base":
        raw_data = datasets.load_dataset("csv", data_files={"train": "data/SBIC.v2.agg.trn.csv",
                                                            "test": "data/SBIC.v2.agg.tst.csv",
                                                            "dev": "data/SBIC.v2.agg.dev.csv"})
        tokenized_inputs = raw_data.map(lambda x: tokenize(x["post"], tokenizer, args.lm_version), batched=True)
    # Load toxJSON data
    else:
        tokenized_inputs = {}
        if args.model_type == "multilabel" or args.model_type.startswith("multitask"):
            if args.cap:
                if args.dataset == 'toxjson':
                    with open('data/toxicity_ratings.json') as f:
                        raw_data = json.load(f)["all"]
                    # Same split size as Gordon et al.
                    train_0, test_0 = train_test_split(raw_data, test_size=10000, random_state=42)
                    dev_0, test_0 = train_test_split(test_0, test_size=test_size, random_state=42)
                    train_0, num_workers, ann_list = make_disagg_labels_toxjson(
                        train_0[:args.cap], model_type=args.model_type, survey_info_setting=args.survey_info,
                        use_lgbt=use_lgbt, prepend_other_ratings = args.prepend_other_ratings, ablations=args.feature_ablation)
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
                            use_lgbt=use_lgbt,  prepend_other_ratings = args.prepend_other_ratings, ablations=args.feature_ablation)
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
                dev_0, model_type=args.model_type, survey_info_setting=args.survey_info, use_lgbt=use_lgbt, prepend_other_ratings = args.prepend_other_ratings,
                ablations=args.feature_ablation)[0])
            test_0 = pd.DataFrame(make_disagg_labels_toxjson(
                test_0, model_type=args.model_type, survey_info_setting=args.survey_info, use_lgbt=use_lgbt, prepend_other_ratings = args.prepend_other_ratings,
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


def save_toxjson_for_prelabeling():
    """
    Helper method: Save the toxJSON dev dataset to a CSV file for easier hand-labeling.

    :return: None
    """
    val_set = load_from_disk("tokenized_toxjson/multitasktext-sepTrue/dev")

    writer = csv.writer(open("toxjson_hand_labeled.csv", "w"))
    prev_post = ""
    for i, row in enumerate(val_set):
        if i > 6000:
            if row["post"][row["post"].rindex("SEP"):] != prev_post:
                writer.writerow([row["post"][row["post"].rindex("SEP"):]])
        prev_post = row["post"][row["post"].rindex("SEP"):]
        if i > 12000:
            break


def run_training(args, tokenized_inputs, tokenizer, num_workers, use_wandb=False):
    """
    Train the annotator rating model.

    :param args: Model training args
    :param tokenized_inputs: Tokenized inputs of the model
    :param tokenizer: Tokenizer used for tokenized_inputs
    :param num_workers: Number of annotators in the training set
    :param use_wandb: Whether to send results to Weights & Biases
    :return: Final evaluation metrics from trainer.evaluate()
    """

    FAST_HEAD_LR = args.fast_head_lr
    FREEZE_LM = args.freeze_lm
    DATASET = args.dataset
    NUM_LABELS = 24016

    # Initial setup: seeding, model configuration, W&B metric reporting
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
        report_to="none" #wandb
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
                        model_config, append_combo = args.append_combo,num_labels=1, is_baseline=is_baseline, use_annotators=use_annotators,
                        use_demographics=use_demographics, use_var=args.use_var_objective,
                        use_annot_module=use_annot_module).from_pretrained(
                        args.from_saved_model, is_baseline=is_baseline, use_annotators=use_annotators,
                        use_demographics=use_demographics, use_var=args.use_var_objective, num_labels=1,
                        ignore_mismatched_sizes=True, use_annot_module=use_annot_module)
                    else:
                        model = RobertaMultitaskModel(
                            model_config,num_labels=1, is_baseline=is_baseline, use_annotators=use_annotators,
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
                        model_config, append_combo = args.append_combo,num_labels=1, is_baseline=is_baseline, use_annotators=use_annotators,
                        use_demographics=use_demographics, use_var=args.use_var_objective,
                        use_annot_module=use_annot_module).from_pretrained(
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

        # Initialize device and set seed if given
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
            #print("Evaluation before training...")
            #trainer.evaluate(tokenized_inputs["dev"])
            print("Beginning training...")
            trainer.train()
            model.save_pretrained(args.save_model_to + "/final")

        print("-- FINAL EVAL --")
        print(tokenized_inputs["dev"][0])
        if args.eval_on_test:
            return trainer.evaluate(tokenized_inputs["test"])
        return trainer.evaluate(tokenized_inputs["dev"])

    print("Done")


def quick_replace(t, replacements):
    for k, v in replacements.items():
        t.replace(k, v)
    return t


def get_target_group_model_metrics(target_group_model, dataset, dev_target_gt, gpt2_tokenizer):
    wmd_model = None#gensim_downloader.load('word2vec-google-news-300')

    # evaluate target model accuracy
    target_acc = 0
    num_targets = 0
    stats = {"exact_precision": 0, "exact_recall": 0, "inexact_acc": 0, "acc": 0,
             "prec_with_inoff": 0, "rec_with_inoff": 0, "acc_with_inoff": 0, "inexact_acc_with_inoff": 0,
             "prec_inoff_sum": 0, "rec_inoff_sum": 0, "acc_inoff_sum": 0, "inexact_sum_with_inoff": 0,
             "acc_sum": 0, "prec_sum": 0, "rec_sum": 0, "inexact_sum": 0, "wmd_distances": [],
             "wmd_distances_inoff": []}

    for index, (entry, rating_pred, label) in enumerate(zip(dataset, eval_preds["predictions"], eval_preds["labels"])):

        # get the standardized target of the post
        if "[SEP]" in entry["post"]:
            post = entry["post"][entry["post"].rindex("[SEP]") + 5:]
            demo = entry["post"][entry["post"].index("[SEP]") - 30:entry["post"].rindex("[SEP]")]
        else:
            post = entry["post"]

        agg_index = index // 5
        if agg_index >= len(dev_target_gt):
            break

        base_replacements = {"gay": "queer", "lesbian": "queer", "bisexual": "queer", "mental disability": "disability"}
        # replace with union of [agg_index][1] & [agg_index][2] for target group mentions, not just harm targets
        target_gts = quick_replace(dev_target_gt[agg_index][1], base_replacements)

        # if you only want ones where there actually is a target, check for that here
        # if len(target_gts) > 0 and index % 5 == 0:
        # print("-\n", flush=True)

        # get the predicted target group
        input_tok = gpt2_tokenizer(post + gpt2_tokenizer.sep_token, return_tensors="pt")
        output_sequence = target_group_model.generate(input_tok['input_ids'].to(device),
                                                      max_length=input_tok['input_ids'].shape[1] + 20)
        target_group = target_model.get_substring(output_sequence,
                                                  gpt2_tokenizer)  # input here should be 2d. calling the *class* not the model
        t_pred = gpt2_tokenizer.decode(target_group)

        # extra standardization
        mapped_t_pred = list(get_text_target(t_pred.replace(gpt2_tokenizer.pad_token, " ")))
        drop_list = ["folk", "all", "the", "of", "people", "folks"]
        target_preds = list(set([quick_replace(t, base_replacements) for t in mapped_t_pred if t not in drop_list]))

        # print("gt (raw):", dev_target_gt[agg_index][1], "\tgt:", target_gts, "post:", dev_target_gt[agg_index][0])
        target_gts_no_cat = [w[w.index(":") + 1:] if ":" in w else w for w in target_gts]

        # Stats including inoffensive cases
        cur_acc_inoff = 0
        cur_acc_sum_inoff = 0

        # check whether target == target_gt
        for word in target_preds:
            if word in target_gts or (":" in word and word[word.index(":") + 2:] in target_gts_no_cat):
                stats["inexact_acc_with_inoff"] += 1
                break
        stats["inexact_sum_with_inoff"] += 1

        for word in target_preds:
            if word in target_gts:
                stats["prec_with_inoff"] += 1
            elif ":" in word and word[word.index(":") + 2:] in target_gts_no_cat:
                stats["prec_with_inoff"] += 1
            else:
                cur_acc_sum_inoff += 1
        stats["prec_inoff_sum"] += len(target_preds)

        stats["wmd_distances_inoff"].append(wmd_model.wmdistance(", ".join(target_preds), target_gts))

        for word in target_gts.split(","):
            word = word.strip()

            if word in target_preds:
                cur_acc_inoff += 1
                cur_acc_sum_inoff += 1

                stats["rec_with_inoff"] += 1
            elif ":" in word and word[word.index(":") + 2:] in target_preds:
                stats["rec_with_inoff"] += 1
                cur_acc_inoff += 1
                cur_acc_sum_inoff += 1
            else:
                cur_acc_sum_inoff += 1
        stats["rec_inoff_sum"] += len(target_gts.split(","))

        stats["acc_with_inoff"] += cur_acc_inoff / cur_acc_sum_inoff
        stats["acc_inoff_sum"] += 1

        # Stats for cases where there's a target group
        if len(target_gts) > 0 and index % 5 == 0:
            cur_acc = 0
            cur_acc_sum = 0

            # check whether target == target_gt
            for word in target_preds:
                if word in target_gts or (":" in word and word[word.index(":") + 2:] in target_gts_no_cat):
                    stats["inexact_acc"] += 1
                    break

            stats["inexact_sum"] += 1

            for word in target_preds:
                if word in target_gts:
                    stats["exact_precision"] += 1
                elif ":" in word and word[word.index(":") + 2:] in target_gts_no_cat:
                    stats["exact_precision"] += 1
                else:
                    cur_acc_sum += 1
                    print("[precision] word not found", word)
                    print("target gt words:", target_gts.split(","), "target_preds:", target_preds)
                    print("post:", dev_target_gt[agg_index][0])
            stats["prec_sum"] += len(target_preds)

            stats["wmd_distances"].append(wmd_model.wmdistance(", ".join(target_preds), target_gts))

            for word in target_gts.split(","):
                word = word.strip()

                if word in target_preds:
                    cur_acc += 1
                    cur_acc_sum += 1

                    stats["exact_recall"] += 1
                elif ":" in word and word[word.index(":") + 2:] in target_preds:
                    stats["exact_recall"] += 1
                    cur_acc += 1
                    cur_acc_sum += 1
                else:
                    cur_acc_sum += 1
            stats["rec_sum"] += len(target_gts.split(","))
            stats["acc"] += cur_acc / cur_acc_sum
            stats["acc_sum"] += 1

    stats["exact_precision"] = stats["exact_precision"] / stats["prec_sum"]
    stats["exact_recall"] = stats["exact_recall"] / stats["rec_sum"]
    stats["acc"] = stats["acc"] / stats["acc_sum"]
    stats["exact_f1_score"] = (2 * stats["exact_precision"] * stats["exact_recall"]) / (
                stats["exact_precision"] + stats["exact_recall"])
    stats["wmd_distance"] = sum(stats["wmd_distances"]) / len(stats["wmd_distances"])
    stats["wmd_distances_inoff"] = sum(stats["wmd_distances_inoff"]) / len(stats["wmd_distances_inoff"])
    stats["inexact_acc"] = stats["inexact_acc"] / stats["inexact_sum"]
    stats["prec_with_inoff"] = stats["prec_with_inoff"] / stats["prec_inoff_sum"]
    stats["rec_with_inoff"] = stats["rec_with_inoff"] / stats["rec_inoff_sum"]
    stats["f1_with_inoff"] = (2 * stats["prec_with_inoff"] * stats["rec_with_inoff"]) / (
                stats["prec_with_inoff"] + stats["rec_with_inoff"])
    stats["acc_with_inoff"] = stats["acc_with_inoff"] / stats["acc_inoff_sum"]

    print("-- TARGET GROUP MODEL METRICS --")
    for k, v in stats.items():
        print(k, ":\t", str(v), flush=True)


def run_eval(ann_rating_model, target_group_model, dataset, eval_tokenizer, eval_preds=None, is_baseline=False):
    """
    Given trained target group and annotator rating models, evaluate on a dataset (split of toxJSON).

    :param ann_rating_model: Model that predicts individual annotator ratings.
    :param target_group_model: Model that predicts the target group harmed by a statement.
    :param dataset: Dataset to evaluate on.
    :param eval_tokenizer: GPT2Tokenizer used to tokenize data for ann_rating_model.
    :param eval_preds: Optional predictions of ann_rating_model on dataset.
    :return: None
    """

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    prev_post = ""
    avg_target_rating = 0
    avg_pred_target_rating = 0
    num_target_ann = 0
    examples_disagrees = []
    examples_disagrees_agg = {}
    num_target_found = 0

    csv_reader = csv.reader(open("toxjson_hand_labeled_dev.csv", "r"))
    dev_target_gt = list(csv_reader)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    prev_post = ""
    avg_target_rating = 0
    avg_pred_target_rating = 0
    num_target_ann = 0
    examples_disagrees = []
    examples_disagrees_agg = {}
    num_target_found = 0
    if not eval_preds:
        eval_preds = pickle.load(open("disagg_preds_dict.p", "rb"))
    csv_reader = csv.reader(open("toxjson_hand_labeled_dev.csv", "r"))
    dev_target_gt = list(csv_reader)

    # stats we want:
    # overall MAE
    # number of *examples* where {all target members} disagrees with {majority}
    # for those examples, accuracy on {all target members}
    # accuracy on each target member for examples where {all target members} disagrees with {majority}

    # the MAE for eval_preds overall should be the same as we got in the eval method
    overall_mae = torch.abs(eval_preds["predictions"] - eval_preds["labels"]).sum() / len(eval_preds["predictions"])
    print("Overall MAE (same as in earlier eval):", overall_mae, flush=True)

    gpt2_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-large')
    gpt2_tokenizer.add_special_tokens({'pad_token': '[PAD]', 'sep_token': '[SEP]'})

    # get_target_group_model_metrics(target_group_model, dataset, dev_target_gt, gpt2_tokenizer)

    # Evaluate rating model performance
    # print("-- RATING MODEL METRICS --", flush=True)

    # # Group all examples by post
    # examples_agg = {}
    # eval_preds["is_target"] = []
    # print("pred len", len(eval_preds["predictions"]), "label len", len(eval_preds["labels"]), "dataset len", len(dataset), flush=True)
    # for index, (entry, pred, label) in enumerate(zip(dataset, eval_preds["predictions"], eval_preds["labels"])):
    #     if index % 1000 == 1:
    #         print(index, flush=True)
    #
    #     if "[SEP]" in entry["post"]:
    #         post = entry["post"][entry["post"].rindex("[SEP]") + 5:]
    #         demo = entry["post"][entry["post"].index("[SEP]") - 30:entry["post"].rindex("[SEP]")]
    #     else:
    #         post = entry["post"]
    #
    #     # for testing only: gets target group based off post text, not target_group_model prediction
    #     # target = list(get_text_target(post))[0]
    #
    #     # predict the target group
    #     input_tok = gpt2_tokenizer(post + gpt2_tokenizer.sep_token, return_tensors="pt")
    #     output_sequence = target_group_model.generate(input_tok['input_ids'].to(device),
    #                                                   max_length=input_tok['input_ids'].shape[1] + 20)
    #     target_group = target_model.get_substring(output_sequence, gpt2_tokenizer)
    #     t_pred = gpt2_tokenizer.decode(target_group)
    #
    #     # extra standardization
    #     mapped_t_pred = list(get_text_target(t_pred.replace(gpt2_tokenizer.pad_token, " ")))
    #     drop_list = ["folk", "all", "the", "of", "people", "folks"]
    #     std_replacements = {"gay": "queer", "lesbian": "queer", "bisexual": "queer", "mental disability": "disability"}
    #
    #     target_preds = list(set([quick_replace(t, std_replacements) for t in mapped_t_pred if t not in drop_list]))
    #     dataset[index]["text_target"] = target_preds
    #
    #     # find people who are in the target group
    #     is_target = False
    #     for target in target_preds:
    #         # print("target:", target)
    #         if ":" in target:
    #             target = target[target.index(":") + 2:]
    #         word_replacements = {"women": "female", "men": "male", "people": "", "girls": "female"}
    #         target_words = quick_replace(target, word_replacements)
    #         if len(target_words) > 0 and target_words[-1] == "s":
    #             target_words = target_words[:-1]
    #
    #         if target_words in demo.lower():
    #             is_target = True
    #             break
    #     eval_preds["is_target"].append(is_target)
    #
    #     if post in examples_agg:
    #         examples_agg[post].append([pred.item(), label.item(), is_target])
    #     else:
    #         examples_agg[post] = [[pred.item(), label.item(), is_target]]
    #
    # print("Getting post-wide stats", flush=True)
    # pickle.dump(examples_agg, open("examples_agg" + str(is_baseline) + ".p", "wb"))

    examples_agg = pickle.load(open("examples_agg" + str(is_baseline) + ".p", "rb"))

    threshold = 1.5

    # Post-wide stats
    total_disagree = 0
    total_disagree_disjoint = 0
    total_disagree_pop = 0
    target_mae = 0
    target_agg_mae = 0  # same for baseline
    target_disagree_agg_mae = 0
    target_disagree_mae = 0
    avg_number_target_members = 0
    total_any_target = 0
    total_target = 0
    target_var_mae = 0
    total_var = 0
    target_harm_acc = 0  # "recall" on cases where target group finds it offensive
    target_harm_denom = 0
    target_harm_mae = 0
    avg_target_ratings = []
    avg_pred_target_ratings = []
    disagreement_classifier = {"pred": [], "gt": [], "for_acc": []}
    for post, post_stats in examples_agg.items():
        num_target_in_post = len([row[0] for row in post_stats if row[2]])
        if num_target_in_post > 0:
            total_any_target += 1
            total_target += num_target_in_post
            avg_pred_rating = mean([row[0] for row in post_stats])
            avg_rating = mean([row[1] for row in post_stats])
            avg_pred_target_rating = mean([row[0] for row in post_stats if row[2]])
            avg_target_rating = mean([row[1] for row in post_stats if row[2]])
            #avg_non_target_rating = mean([row[1] for row in post_stats if not row[2]])
            avg_target_ratings.append(avg_target_rating)
            avg_pred_target_ratings.append(avg_pred_target_rating)
            avg_number_target_members += num_target_in_post

            if avg_target_rating > threshold:
                if avg_pred_target_rating > threshold:
                    target_harm_acc += 1
                target_harm_denom += 1
                target_harm_mae += abs(avg_pred_target_rating - avg_target_rating)

            target_agg_mae += abs(avg_pred_target_rating - avg_target_rating)
            if num_target_in_post > 1:
                target_var_mae += abs(variance([row[0] for row in post_stats if row[2]]) - variance(
                    [row[1] for row in post_stats if row[2]]))
                total_var += 1

            for row in post_stats:
                if row[2]:
                    target_mae += abs(row[1] - row[0])

            # if abs(avg_target_rating - avg_non_target_rating) >= 1:
            #     total_disagree_disjoint += 1

            # if you correctly predict no disagreement, or correctly predict no disagreement *and the sign*, correct, else incorrect
            def get_class(avg, target_avg):
                if abs(avg - target_avg) < 1:
                    return 0
                elif avg - target_avg >= 1:
                    return 1
                else:
                    return 2

            pred_class = get_class(avg_pred_rating, avg_pred_target_rating)
            gt_class = get_class(avg_rating, avg_target_rating)
            disagreement_classifier["gt"].append(gt_class)
            disagreement_classifier["pred"].append(pred_class)
            disagreement_classifier["for_acc"].append(pred_class==gt_class)




            if abs(avg_target_rating - avg_rating) >= 1: # disagreement w/ avg

                total_disagree += 1
                total_disagree_pop += len([row[2] for row in post_stats if row[2]])
                target_disagree_agg_mae += abs(avg_pred_target_rating - avg_target_rating)

                for row in post_stats:
                    if row[2]:
                        target_disagree_mae += abs(row[1] - row[0])

    avg_target_ratings = [int(r) for r in avg_target_ratings]
    # print(avg_target_ratings, flush=True)
    # print(avg_pred_target_ratings, flush=True)
    # fpr, tpr, thresholds = roc_curve(avg_target_ratings, avg_pred_target_ratings)
    # gmeans = [np.sqrt(tpr0 * (1 - fpr0)) for tpr0, fpr0 in zip(tpr, fpr)]
    # best_threshold = max(range(len(gmeans)), key=lambda i: gmeans[i])
    # print("best threshold:", best_threshold, "fpr:", fpr, "tpr:", tpr, flush=True)
    # plt.plot(fpr, tpr)
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.savefig('roc.png')

    for threshold in np.linspace(1, 4, 7):
        tp = 0
        tn = 0
        fn = 0
        fp = 0
        for avg_pred, avg_gt in zip(avg_pred_target_ratings, avg_target_ratings):
            if avg_pred >= threshold and avg_gt >= threshold:
                tp += 1
            elif avg_pred < threshold and avg_gt < threshold:
                tn += 1
            elif avg_pred < threshold <= avg_gt:
                fn += 1
            else:
                fp += 1

        acc = (tp + tn) / len(avg_target_ratings)
        prec = tp / (tp + fp + 0.000000001)
        rec = fp / (tp + fn + 0.000000001)
        f1 = (2 * prec * rec) / (prec + rec + 0.000000001)

        print("Threshold:", threshold, "acc:", acc, "prec:", prec, "rec:", rec, "f1:", f1)

    target_agg_mae = target_agg_mae / total_any_target
    target_disagree_agg_mae = target_disagree_agg_mae / total_disagree
    target_mae = target_mae / total_target
    target_disagree_mae = target_disagree_mae / total_disagree_pop
    avg_number_target_members = avg_number_target_members / len(examples_agg)
    target_var_mae = target_var_mae / total_var
    print("raw t_harm acc", target_harm_acc, "t_harm denom", target_harm_denom, "raw mae", target_harm_mae, flush=True)
    target_harm_acc = target_harm_acc / target_harm_denom
    target_harm_mae = target_harm_mae / target_harm_denom

    disagreement_f1 = f1_score(disagreement_classifier["gt"], disagreement_classifier["pred"], average=None)
    disagreement_f1_macro = f1_score(disagreement_classifier["gt"], disagreement_classifier["pred"], average="micro")
    disagreement_acc = sum(disagreement_classifier["for_acc"])/len(disagreement_classifier["for_acc"])

    print("Total examples where avg of targets disagrees with overall avg:", total_disagree)
    print("Total examples where avg of targets disagrees with non-targets:", total_disagree_disjoint)
    print("Total examples with at least one target member annotating:", total_any_target)
    print("Disagreement acc:", disagreement_acc, "F1:", disagreement_f1, "macro F1:", disagreement_f1_macro)
    print("Agg MAE for target group, all examples:", target_agg_mae)
    print("Var MAE for target group, all examples:", target_var_mae)
    print("Disagg MAE for target group, all examples:", target_mae)
    print("Agg MAE for target group, when disagrees:", target_disagree_agg_mae)
    print("Disagg MAE for target group, when disagrees:", target_disagree_mae)
    print("Avg number of target members annotating a post:", avg_number_target_members)
    print("Acc on finding cases where target group thinks it's offensive:", target_harm_acc)
    print("MAE on cases where target group thinks it's offensive:", target_harm_mae, flush=True)


if __name__ == "__main__":
    # Define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=3,
                        help='Number of epochs')
    parser.add_argument('--train_batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--eval_batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Fixed seed to train with')
    parser.add_argument('--save_model_to', type=str, default="saved_models",
                        help='Output path for saved model')
    parser.add_argument('--reload_dataset', action='store_true',
                        help='Whether to reload the toxjson dataset or use the saved tokenized version')
    parser.add_argument('--lr_scheduler', choices=['exp', 'lambda'], default='exp',
                        help='Learning rate scheduler to use')
    parser.add_argument('--from_saved_model', type=str,
                        help='Whether to evaluate from already-saved model')
    parser.add_argument('--from_saved_target_model', type=str,
                        help='Whether to evaluate from already-saved model')
    parser.add_argument('--no_train', action='store_true',
                        help='Whether to train')
    parser.add_argument('--no_test', action='store_true',
                        help='Whether to evaluate')
    parser.add_argument('--cap', type=int, default=None,
                        help='Option to cap amount of training data')
    parser.add_argument('--n_gpu', type=int, default=1,
                        help='Number of gpus to use for training')
    parser.add_argument('--model_type',
                        choices=["baseline", "disagg", "ensemble", "multitask", "multilabel", "ensemble-base",
                                 "multitask-base", "multitask-annotator", "multitask-demographic"],
                        default="baseline", help='Type of model to use')
    parser.add_argument('--survey_info', choices=["id-sep", "text-sep", "both-sep", "module", None],
                        default=None, help='How to incorporate survey info')
    parser.add_argument('--no_lgbt', action='store_true',
                        help='Whether to avoid using LGBT status in demographic info')
    parser.add_argument('--dataset', choices=['sbic', 'toxjson'], default='toxjson')
    parser.add_argument('--lm_version', choices=['gpt2', 'gpt2-large', 'roberta'], default='roberta')
    parser.add_argument('--cap_eval', action='store_true',
                        help='Whether to cap amount of eval data (for debugging purposes)')
    parser.add_argument('--fast_head_lr', action='store_true',
                        help='Whether to use a faster learning rate for the head and categorical information')
    parser.add_argument('--freeze_lm', action='store_true',
                        help='Whether to freeze the language model after the first 3 epochs')
    parser.add_argument("--use_var_objective", action='store_true',
                        help='Whether to optimize for minimizing variance MAE')
    parser.add_argument("--run_sweep", action='store_true',
                        help='Whether to run hyperparameter search')
    parser.add_argument('--project_name', type=str, default="sbic",
                        help='Project name for W&B')
    parser.add_argument('--use_wandb', action='store_true', help="Whether to send results to W&B")
    parser.add_argument('--joint_eval', action='store_true',
                        help="Whether to run joint evaluation of target and group models")
    parser.add_argument('--eval_on_test', action='store_true',
                    help="Whether to evaluate on the test set instead of dev set.")
    parser.add_argument('--saved_predictions', type=str, default="disagg_preds_dict.p",
                        help="Saved predictions for faster evaluation. Set to `none` to generate predictions again.")
    parser.add_argument('--feature_ablation', type=str, default="",
                        help="Comma-separated list of demographic & survey feature names for ablation")
    parser.add_argument('--recsys', action='store_true',help="Whether to use the recsys model head")
    parser.add_argument('--append_combo', action='store_true',help="Whether to append the output of recsys [U,x] or take the dot product")
    parser.add_argument('--prepend_other_ratings', action='store_true',help="Whether to prepend other texts and their associated ratings")


    print("Begin trainer", flush=True)
    args = parser.parse_args()
    tokenized_inputs, tokenizer, num_workers = load_data(args)

    # Evaluation only: test annotator rating model & target group identification model jointly
    if args.joint_eval:
        model_config = AutoConfig.from_pretrained("roberta-base")
        use_annot_module = (args.survey_info == "module")
        is_baseline = ("multitask" not in args.model_type)
        use_annotators = (args.model_type == "multitask-annotator" or args.model_type == "multitask")
        use_demographics = (args.model_type == "multitask-demographic" or args.model_type == "multitask")
        ann_rating_model = RobertaMultitaskModel(
            model_config, num_labels=1, is_baseline=is_baseline, use_annotators=use_annotators,
            use_demographics=use_demographics, use_var=args.use_var_objective,
            use_annot_module=use_annot_module).from_pretrained(
            args.from_saved_model, is_baseline=is_baseline, use_annotators=use_annotators,
            use_demographics=use_demographics, use_var=args.use_var_objective, num_labels=1,
            ignore_mismatched_sizes=True, use_annot_module=use_annot_module)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        target_group_model = GPT2LMHeadModel.from_pretrained(args.from_saved_target_model).to(device)

        if args.saved_predictions == "none":
            eval_preds = run_training(args, tokenized_inputs, tokenizer, num_workers, use_wandb=args.use_wandb)
            pickle.dump(eval_preds, open("preds_disagg" + args.model_type + ".p", "wb"))
        else:
            eval_preds = pickle.load(open(args.saved_predictions, "rb"))

        run_eval(ann_rating_model, target_group_model, tokenized_inputs["dev"], tokenizer, eval_preds, is_baseline=False)#True)

    # Run hyperparameter tuning sweep
    elif args.run_sweep:
        project_name = args.save_model_to[args.save_model_to.index("/") + 1]
        sweep_configuration = {
            'method': 'random',
            'name': 'sweep',
            'early_terminate': {
                'type': 'hyperband',
                'min_iter': 3
            },
            'metric': {
                'goal': 'minimize',
                'name': 'disagg_mae'
            },
            'parameters': {
                'sweep_batch_size': {'values': [16, 32, 40]},
                'sweep_lr': {'values': [1e-04, 5e-05, 3e-05, 1e-05, 8e-06, 5e-06, 1e-06]},
                'sweep_data_seed': {'values': [42, 57, 63, 74, 86, 95]},
            },
            'demo-feature-ablate': {
                'demo-features'
            }
        }
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)
        print("Sweep ID:", sweep_id, sweep_configuration)
        wandb.agent(sweep_id=sweep_id, function=partial(
            run_training, args, tokenized_inputs, tokenizer, num_workers, use_wandb=True), count=10)
    else:
        run_training(args, tokenized_inputs, tokenizer, num_workers)
