import argparse
import atexit
import datasets
import evaluate
import numpy as np
import pickle
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config, TrainingArguments, Trainer, \
    GPTNeoXForCausalLM, AutoTokenizer
import torch
#import gensim.downloader as gensim_downloader
import wandb
import os

@atexit.register
def notify():
    print("\a")


def get_target_labels(targets_map, ex):
    """
    Get target labels for an example in the SBIC dataset.

    :param targets_map: Maps unusual target group names to standardized versions
    :param ex: Example to extract target labels for.
    :return:
    """

    if ex['targetMinority'] == "[]":
        target = ex['targetCategory'][1:-1].lower().strip().split(",")
    else:
        target = ex['targetMinority'][1:-1].lower().strip().split(",")

        for index, t in enumerate(target):
            if t in targets_map:
                t = targets_map[t]

            t = t.strip("'").strip().strip('"')
            target[index] = t

    return ",".join(set(target))


def load_data(cap, data_tokenizer):
    """
    Load data from Social Bias Frames (SBIC) dataset for target model prediction.

    :param cap: Max amount of training data
    :param data_tokenizer: Tokenizer for input text
    :return: Huggingface Dataset of Tokenized data (dict keys "train", "val", "test" for each split)
    """

    # Standardize unusual target group labels
    targets_map = {"folks with physical illness/disorder": "physically disabled folks",
                   "folks with mental illness/disorder": "mentally disabled folks",
                   "ugly folks": "appearance",
                   "gypsies": "romani",
                   "rape victims": "assault victims",
                   "refugees": "immigrants",
                   "illegal immigrants": "immigrants",
                   "syrian": "syrians",
                   "africans": "black folks",
                   "Islam": "muslim folks",
                   "arabic folks": "middle eastern folks",
                   "overweight/fat folks": "appearance",
                   "everyone": None,
                   "well off folks": "rich folks",
                   "Child sexual assault victims": "assault victims",
                   "cancer patients": "cancer victims",
                   "bald folks": "appearance",
                   "all non white races": "people of color",
                   "minorities": "people of color",
                   "Mulatto folks": "multiracial folks",
                   "pedophilia victims": "assault victims",
                   "aborigines": "aboriginal folks",
                   "aboriginal": "aboriginal folks",
                   "cis": "cisgender folks",
                   "native american/first nation folks": "indigenous folks",
                   "911 victims": "terrorism victims",
                   "all religious folks": "religious folks",
                   "homeless victim": "unhoused folks",
                   "cops": "police",
                   "everybody not in the US (foreigners)": "foreigners",
                   "russins": "russians",
                   "red heads": "appearance",
                   "mexican": "mexican folks",
                   "thai": "thai folks",
                   "thai people": "thai folks",
                   "arabs": "middle eastern folks",
                   "israel": "israeli folks",
                   "israel folks": "israeli folks",
                   "harassment victims": "assault victims",
                   "pakistan": "pakistani folks",
                   "pakistani": "pakistani folks",
                   "islamic": "muslim folks",
                   "child molestation victims": "assault victims",
                   "girls and boys": "children",
                   "non-whites": "people of color",
                   "Catholic priests": "priests",
                   "women who have had abortions": "people who have had abortions",
                   "child rape victims": "assault victims",
                   "shia": "muslim folks",
                   "ukrainians": "ukrainian folks",
                   "sexual assault victims": "assault victims",
                   "islam": "muslim folks",
                   "any racial/ethnic minority in america": "people of color",
                   "indians": "indian folks",
                   "arabian": "middle eastern folks",
                   "people with anorexia": "anorexic folks",
                   "anorexics": "anorexic folks",
                   "non-whites": "people of color",
                   "russia": "russians",
                   "homeless folks": "unhoused folks",
                   "shorts folk": "appearance",
                   "japanese": "japanese folks",
                   "indian": "indian folks",
                   "catholics": "catholic folks",
                   "mutilation victims": "genital mutilation victims",
                   "young children": "children",
                   "chinese": "chinese folks",
                   "anorexic": "anorexic folks",
                   "africa": "african folks",
                   "africans": "african folks",
                   "white": "white folks",
                   "sudanese": "sudanese folks",
                   "ethiopian people": "ethiopian folks",
                   "ethiopia": "ethiopian folks",
                   "ethiopens": "ethiopian folks",
                   "ethiopians": "ethiopian folks",
                   "seniors": "old folks",
                   "japan": "japanese folks",
                   "mexicans": "mexican folks",
                   "mu slims": "muslim folks",
                   "male": "men",
                   "brazillian folks": "brazilian folks",
                   "rednecks": "rural folks",
                   "hispanics": "latino/latina folks",
                   "all non whites": "people of color",
                   "mexican people": "mexican folks",
                   "republicans": "conservatives",
                   "german": "german folks",
                   "germans": "german folks",
                   "disabled people": "disabled folks",
                   "breast size": "appearance",
                   "gays": "queer folks",
                   "vegans": "vegetarians",
                   "people addicted to drugs": "folks with substance abuse disorders",
                   "OD victims": "folks with substance abuse disorders",
                   "young boys": "boys",
                   "redneck": "rural folks",
                   "minors": "children",
                   "catholic": "catholic folks",
                   "all races": None,
                   "antifascists": "antifa",
                   "antifas": "antifa",
                   "asian": "asian folks",
                   "polish  people": "polish folks",
                   "pollack": "polish folks",
                   "pakis": "pakistani folks",
                   "sexual assualt": "assault victims",
                   "middle-eastern folks": "middle eastern folks",
                   "pakistanis": "pakistani folks",
                   "skinny men": "appearance",
                   "anyone not white": "people of color",
                   "rural/country folks": "rural folks",
                   "rich people": "rich folks"
    }

    raw_data = datasets.load_dataset("csv", data_files={"train": "data/SBIC.v2.agg.trn.csv",
                                                        "test": "data/SBIC.v2.agg.tst.csv",
                                                        "dev": "data/SBIC.v2.agg.dev.csv"})

    raw_data.filter(lambda ex: ex['targetMinority'] != "[]" or ex['targetCategory'] != "[]")
    raw_data = raw_data.shuffle(seed=42)

    for split in raw_data.keys():
        raw_data[split] = raw_data[split].add_column("targets",
                                                     [get_target_labels(targets_map, x) for x in raw_data[split]])
        raw_data[split] = raw_data[split].filter(lambda ex: ex["targets"] != "")
        raw_data[split] = raw_data[split].add_column(
            "input_text", [ex["post"] + data_tokenizer.sep_token + ex["targets"] for ex in raw_data[split]])
        raw_data[split] = raw_data[split].map(lambda x: tokenize(x["input_text"], data_tokenizer), batched=True)
        raw_data[split] = raw_data[split].remove_columns(
            ["dataSource", "Unnamed: 0", "targetStereotype", "targetMinority", "targetCategory", "intentYN",
             "whoTarget", "sexYN", "hasBiasedImplication", "post", "targets"])
        raw_data[split] = raw_data[split].add_column(
            "labels", [ex["input_ids"] for ex in raw_data[split]])  # GPT2 shifts labels inside the model
        if cap:
            raw_data[split] = raw_data[split].select(range(cap))  # .take(cap) only exists for IterableDataset

    return raw_data


def tokenize(s1, my_tokenizer):
    return my_tokenizer(text=s1, padding="max_length", truncation=True, max_length=256)


def get_substring(row, tokenizer):
    """
    Get substring of generated output corresponding to predicted target group (i.e., everything after [SEP] token).

    :param row: Generated output to get the substring of
    :param tokenizer: GPT2Tokenizer used to tokenize data
    :return: Substring of row starting after [SEP] token, or a list consisting of just the [EOS] token if not found.
    """

    try:
        sep_ind = torch.where(row == tokenizer.sep_token_id)
    except TypeError:
        sep_ind = np.where(row == tokenizer.sep_token_id)

    if len(sep_ind[1]) > 0:
        return row[0][sep_ind[1][-1].item() + 1:]
    return [tokenizer.eos_token_id]


def compute_metrics(eval_pred):
    """
    Compute metrics for evaluation of target group prediciton.

    :param eval_pred: Model predictions to evaluate
    :return: dictionary of metrics (precision, recall, bleu, f1)
    """

    print("[***] EVAl")
    preds, labels_raw = eval_pred
    predictions_raw = preds[0]

    preds_text = real_tokenizer.batch_decode(predictions_raw)
    labels_text = real_tokenizer.batch_decode(labels_raw)

    # get only the group prediction
    pred_groups = [get_substring(row, real_tokenizer) for row in predictions_raw]
    label_groups = [get_substring(row, real_tokenizer) for row in labels_raw]

    preds_text = real_tokenizer.batch_decode(pred_groups)
    labels_text = real_tokenizer.batch_decode(label_groups)
    # print("pred text:", preds_text[:100])
    # print("label text:", labels_text[:100])

    if len("".join(labels_text)) == 0:
        pickle.dump({"predictions": [pred_groups, predictions_raw], "labels": [label_groups, labels_raw]},
                    open("group_predictions.p", "wb"))

    if len("".join(labels_text)) == 0:
        return {}

    # wmd_distances = [wmd_model.wmdistance(preds_text[i], labels_text[i]) for i in range(len(preds_text))]
    # wmd_distance = sum(wmd_distances)/len(wmd_distances)

    precision = 0
    recall = 0
    pred_len = 0
    label_len = 0
    for pred_i, pred_text in enumerate(preds_text):
        preds_text_split = pred_text.split(" ")
        labels_text_split = labels_text[pred_i].split(" ")
        for word in preds_text_split:
            if word in labels_text[pred_i]:
                precision += 1
        pred_len += len(preds_text_split)
        for word in labels_text_split:
            if word in pred_text:
                recall += 1
        label_len += len(labels_text_split)

    precision = precision / pred_len
    recall = recall / label_len
    f1_score = {"f1": (2 * precision * recall)/(precision + recall)}

    bleu_score = bleu.compute(predictions=preds_text, references=labels_text)

    metrics_dict = bleu_score | f1_score
    # metrics_dict["wmd_distance"] = wmd_distance
    wandb.log(metrics_dict)
    wandb.alert(
        title="New model evaluation",
        text="New model results:" + str(metrics_dict),
        wait_duration=600
    )
    print(metrics_dict)
    return metrics_dict

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels


if __name__ == "__main__":
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
    parser.add_argument('--from_saved_model', type=str,
                        help='Whether to evaluate from already-saved model')
    parser.add_argument('--no_train', action='store_true',
                        help='Whether to train')
    parser.add_argument('--no_test', action='store_true',
                        help='Whether to evaluate')
    parser.add_argument('--cap', type=int, default=None,
                        help='Option to cap amount of training data')
    parser.add_argument('--n_gpu', type=int, default=1,
                        help='Number of gpus to use for training')
    parser.add_argument('--cap_eval', action='store_true',
                        help='Whether to cap amount of eval data (for debugging purposes)')
    parser.add_argument('--project_name', type=str, default="target_model",
                        help='Project name for W&B')
    args = parser.parse_args()

    print("Initializing model...")

    # Set up args and initialize model
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if "/" in args.save_model_to:
        run_name = args.save_model_to[args.save_model_to.index("/") + 1:]
    else:
        run_name = args.save_model_to

    wandb.init(project=args.project_name)
    wandb.run.name = run_name
    wandb.run.save()

    real_tokenizer = AutoTokenizer.from_pretrained(
      "EleutherAI/pythia-1.4b-deduped",
      revision="step3000",
      cache_dir="./pythia-1.4b-deduped/step3000",
    )

    #real_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-large')
    real_tokenizer.add_special_tokens({'pad_token': '[PAD]', 'sep_token': '[SEP]'})
    tokenized_inputs = load_data(cap=args.cap, data_tokenizer=real_tokenizer)
    # config = GPT2Config.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')

    if args.from_saved_model:
        model = GPT2LMHeadModel.from_pretrained(args.from_saved_model)
    else:
        #model = GPT2LMHeadModel.from_pretrained('gpt2-large')  # config=config, ignore_mismatched_sizes=True)
        model = GPTNeoXForCausalLM.from_pretrained(
            "EleutherAI/pythia-1.4b-deduped",
            revision="step3000",
            cache_dir="./pythia-1.4b-deduped/step3000",
        )

    print("Initialized model")

    bleu = evaluate.load("bleu")
    f1_metric = evaluate.load("f1")
    # wmd_model = gensim_downloader.load('word2vec-google-news-300')

    training_args = TrainingArguments(
        args.save_model_to,
        evaluation_strategy="epoch",
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.n_epochs,
        save_total_limit=5,
        save_strategy="epoch",
        logging_steps=300,
        report_to="wandb"
    )

    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(len(real_tokenizer))

    # Initialize device and set seed if given
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device", device, "; number of devices:", torch.cuda.device_count())
    model.to(device)

    # Train and evaluate model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_inputs["train"],
        eval_dataset=tokenized_inputs["dev"],
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )
    if not args.no_train:
        print("Evaluation before training...")
        trainer.evaluate(tokenized_inputs["dev"])
        print("Beginning training...")
        trainer.train()
        model.save_pretrained(args.save_model_to + "/final")

    print("-- FINAL EVAL --")
    print("DEV SET EVAL")
    trainer.evaluate(tokenized_inputs["dev"])

    print("Done")
    wandb.alert(
        title="Run completed",
        text="Run " + args.project_name + " ran to completion."
    )
